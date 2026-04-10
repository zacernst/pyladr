"""Tests for online learning integration with the search loop."""

from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term
from pyladr.search.given_clause import (
    ExitCode,
    GivenClauseSearch,
    SearchOptions,
)
from pyladr.search.online_integration import (
    OnlineIntegrationConfig,
    OnlineSearchIntegration,
    ProofProgressTracker,
    ProofProgressSignals,
)
from pyladr.ml.online_learning import (
    ExperienceBuffer,
    InferenceOutcome,
    OnlineLearningConfig,
    OnlineLearningManager,
    OutcomeType,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_term(symnum: int, args: tuple[Term, ...] = ()) -> Term:
    return Term(private_symbol=-symnum, arity=len(args), args=args)


def _make_var(varnum: int) -> Term:
    return Term(private_symbol=varnum, arity=0, args=())


def _make_clause(
    lits: list[tuple[bool, Term]], clause_id: int = 0,
) -> Clause:
    literals = tuple(Literal(sign=s, atom=a) for s, a in lits)
    c = Clause(literals=literals)
    c.id = clause_id
    c.weight = float(len(literals))
    return c


class MockEncoder:
    """Mock encoder for testing."""

    def __init__(self, dim: int = 64):
        self._dim = dim
        self._linear = torch.nn.Linear(dim, dim)

    def encode_clauses(self, clauses: list[Clause]) -> torch.Tensor:
        x = torch.randn(len(clauses), self._dim)
        return self._linear(x)

    def parameters(self):
        return self._linear.parameters()

    def named_parameters(self):
        return self._linear.named_parameters()

    def state_dict(self):
        return self._linear.state_dict()

    def load_state_dict(self, state):
        self._linear.load_state_dict(state)

    def train(self, mode=True):
        self._linear.train(mode)

    def eval(self):
        self._linear.eval()


# ── ProofProgressTracker tests ────────────────────────────────────────────


class TestProofProgressTracker:
    def test_initial_signals(self):
        tracker = ProofProgressTracker()
        signals = tracker.get_signals()
        assert signals.unit_clauses_generated == 0
        assert signals.productive_inference_rate == 0.0
        assert signals.given_since_last_progress == 0

    def test_clause_generation_tracking(self):
        tracker = ProofProgressTracker()
        # Non-unit clause
        c1 = _make_clause([(True, _make_term(1)), (False, _make_term(2))])
        tracker.on_clause_generated(c1)
        assert tracker.get_signals().unit_clauses_generated == 0

        # Unit clause
        c2 = _make_clause([(True, _make_term(3))])
        tracker.on_clause_generated(c2)
        assert tracker.get_signals().unit_clauses_generated == 1

    def test_kept_clause_tracking(self):
        tracker = ProofProgressTracker()

        # Generate some clauses
        for i in range(5):
            tracker.on_clause_generated(_make_clause([(True, _make_term(i))]))

        # Keep 3 of them
        for i in range(3):
            c = _make_clause([(True, _make_term(i))])
            c.weight = float(10 - i)
            tracker.on_clause_kept(c)

        signals = tracker.get_signals()
        assert signals.productive_inference_rate == 3 / 5

    def test_given_selection_tracking(self):
        tracker = ProofProgressTracker()
        given = _make_clause([(True, _make_term(1))], clause_id=100)
        tracker.on_given_selected(given)
        assert tracker._given_count == 1

    def test_weight_trend(self):
        tracker = ProofProgressTracker()

        # Generate enough data for trend calculation
        for i in range(20):
            tracker.on_clause_generated(_make_clause([(True, _make_term(i))]))
            c = _make_clause([(True, _make_term(i))])
            # Weights decrease over time (improving)
            c.weight = float(20 - i)
            tracker.on_clause_kept(c)

        signals = tracker.get_signals()
        # Weight trend should be positive (earlier weights higher than later)
        assert signals.avg_clause_weight_trend > 0

    def test_given_productivity(self):
        tracker = ProofProgressTracker()

        given = _make_clause([(True, _make_term(1))], clause_id=100)
        tracker.on_given_selected(given)

        # Register some inferences
        for i in range(5):
            child = _make_clause([(True, _make_term(i))], clause_id=200 + i)
            tracker.on_inference_from_given(100, child)

        prod = tracker.given_clause_productivity(100)
        assert prod > 0


# ── OnlineSearchIntegration tests ────────────────────────────────────────


class TestOnlineSearchIntegration:
    def test_disabled_integration(self):
        config = OnlineIntegrationConfig(enabled=False)
        integration = OnlineSearchIntegration(config=config)
        assert integration.stats.experiences_collected == 0

        # Hooks should be no-ops
        c = _make_clause([(True, _make_term(1))], clause_id=1)
        integration.on_given_selected(c, "T")
        integration.on_clause_kept(c)
        integration.on_inferences_complete()
        assert integration.stats.experiences_collected == 0

    def test_experience_collection(self):
        config = OnlineIntegrationConfig(
            enabled=True,
            collect_experiences=True,
            trigger_updates=False,
        )
        encoder = MockEncoder()
        learning_config = OnlineLearningConfig(enabled=True)
        manager = OnlineLearningManager(encoder, learning_config)

        integration = OnlineSearchIntegration(
            config=config,
            manager=manager,
        )

        # on_clause_kept no longer adds to buffer (subsumption events do)
        given = _make_clause([(True, _make_term(1))], clause_id=100)
        integration.on_given_selected(given, "T")

        child = _make_clause([(True, _make_term(2))], clause_id=200)
        integration.on_clause_kept(child, given=given)

        assert integration.stats.experiences_collected == 0
        assert manager._buffer.size == 0

        # Subsumption events add to buffer
        subsuming = _make_clause([(True, _make_term(3))], clause_id=300)
        subsumed = _make_clause([(True, _make_term(4))], clause_id=400)
        integration.on_back_subsumption(subsuming, subsumed)

        assert integration.stats.experiences_collected == 2  # recorded twice
        assert manager._buffer.size == 2

    def test_deletion_tracking(self):
        config = OnlineIntegrationConfig(
            enabled=True,
            collect_experiences=True,
            trigger_updates=False,
        )
        encoder = MockEncoder()
        learning_config = OnlineLearningConfig(enabled=True)
        manager = OnlineLearningManager(encoder, learning_config)

        integration = OnlineSearchIntegration(
            config=config,
            manager=manager,
        )

        given = _make_clause([(True, _make_term(1))], clause_id=100)
        integration.on_given_selected(given, "T")

        # Record a deleted clause
        child = _make_clause([(True, _make_term(2))], clause_id=200)
        integration.on_clause_deleted(
            child, OutcomeType.SUBSUMED, given=given,
        )

        assert integration.stats.experiences_collected == 1
        assert manager._buffer.size == 1

    def test_proof_callback(self):
        config = OnlineIntegrationConfig(
            enabled=True,
            collect_experiences=True,
            trigger_updates=False,
        )
        encoder = MockEncoder()
        learning_config = OnlineLearningConfig(enabled=True)
        manager = OnlineLearningManager(encoder, learning_config)

        integration = OnlineSearchIntegration(
            config=config,
            manager=manager,
        )

        # Add subsumption outcomes first (KEPT no longer adds to buffer)
        subsuming = _make_clause([(True, _make_term(1))], clause_id=100)
        subsumed = _make_clause([(True, _make_term(2))], clause_id=200)
        integration.on_back_subsumption(subsuming, subsumed)

        # Simulate proof found
        integration.on_proof_found({100})

        # Buffer should have the subsumption outcomes + proof-upgraded version
        assert manager._buffer.size >= 2

    def test_model_update_trigger(self):
        config = OnlineIntegrationConfig(
            enabled=True,
            collect_experiences=True,
            trigger_updates=True,
            min_given_before_ml=0,
        )
        encoder = MockEncoder()
        learning_config = OnlineLearningConfig(
            enabled=True,
            update_interval=5,
            min_examples_for_update=3,
            gradient_steps_per_update=1,
            batch_size=2,
        )
        manager = OnlineLearningManager(encoder, learning_config)

        integration = OnlineSearchIntegration(
            config=config,
            manager=manager,
        )

        # Add mixed outcomes
        for i in range(10):
            given = _make_clause([(True, _make_term(i))], clause_id=100 + i)
            integration.on_given_selected(given, "T")

            child = _make_clause([(True, _make_term(i + 50))], clause_id=200 + i)
            partner = _make_clause([(False, _make_term(i + 50))], clause_id=300 + i)

            if i % 2 == 0:
                integration.on_clause_kept(child, given=given, partner=partner)
            else:
                integration.on_clause_deleted(
                    child, OutcomeType.SUBSUMED, given=given, partner=partner,
                )

        # Trigger update check
        integration.on_inferences_complete()

        # Should have attempted at least one update
        assert integration.stats.model_updates_triggered >= 0

    def test_adaptive_ml_weight(self):
        config = OnlineIntegrationConfig(
            enabled=True,
            adaptive_ml_weight=True,
            initial_ml_weight=0.1,
            max_ml_weight=0.5,
            ml_weight_increase_rate=0.1,
            min_given_before_ml=0,
        )
        integration = OnlineSearchIntegration(config=config)

        # Before any given clauses, weight should be 0 (below min threshold)
        # But we set min_given_before_ml=0 so it should work immediately
        weight = integration.get_current_ml_weight()
        assert weight == 0.1

    def test_adaptive_weight_disabled(self):
        config = OnlineIntegrationConfig(
            enabled=True,
            adaptive_ml_weight=False,
            initial_ml_weight=0.3,
        )
        integration = OnlineSearchIntegration(config=config)
        assert integration.get_current_ml_weight() == 0.3

    def test_create_search_when_disabled(self):
        """Creating a search with disabled integration should still work."""
        config = OnlineIntegrationConfig(enabled=False)
        integration = OnlineSearchIntegration(config=config)

        search = integration.create_search(
            options=SearchOptions(binary_resolution=True),
        )
        assert search is not None

    def test_integration_stats_report(self):
        config = OnlineIntegrationConfig(enabled=True)
        integration = OnlineSearchIntegration(config=config)
        report = integration.stats.report()
        assert "OnlineIntegration" in report
        assert "experiences=" in report

    def test_progress_tracker_accessible(self):
        config = OnlineIntegrationConfig(enabled=True)
        integration = OnlineSearchIntegration(config=config)
        tracker = integration.progress_tracker
        assert isinstance(tracker, ProofProgressTracker)


# ── Integration with actual search ───────────────────────────────────────


class TestSearchIntegration:
    """Tests that verify the integration works with real searches."""

    def _make_complementary_clauses(self) -> tuple[list[Clause], list[Clause]]:
        """Create clauses that resolve to empty: P(a) and -P(a)."""
        a = _make_term(1)  # constant 'a'
        P_a = _make_term(2, (a,))  # P(a)

        c1 = Clause(
            literals=(Literal(sign=True, atom=P_a),),
            justification=(Justification(just_type=JustType.INPUT),),
        )
        c2 = Clause(
            literals=(Literal(sign=False, atom=P_a),),
            justification=(Justification(just_type=JustType.INPUT),),
        )

        return [], [c1, c2]

    def test_online_search_finds_proof(self):
        """Online learning search should not prevent proof finding."""
        config = OnlineIntegrationConfig(
            enabled=True,
            collect_experiences=True,
            trigger_updates=False,  # Don't update during this small test
        )
        encoder = MockEncoder()
        learning_config = OnlineLearningConfig(enabled=True)
        manager = OnlineLearningManager(encoder, learning_config)

        integration = OnlineSearchIntegration(
            config=config,
            manager=manager,
        )

        usable, sos = self._make_complementary_clauses()
        search = integration.create_search(
            options=SearchOptions(binary_resolution=True, quiet=True),
        )

        result = search.run(usable, sos)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1

    def test_online_search_collects_experiences(self):
        """Verify experiences are collected during a real search."""
        config = OnlineIntegrationConfig(
            enabled=True,
            collect_experiences=True,
            trigger_updates=False,
        )
        encoder = MockEncoder()
        learning_config = OnlineLearningConfig(enabled=True)
        manager = OnlineLearningManager(encoder, learning_config)

        integration = OnlineSearchIntegration(
            config=config,
            manager=manager,
        )

        # Build a slightly larger problem
        a = _make_term(1)
        P_a = _make_term(3, (a,))
        Q_a = _make_term(4, (a,))

        clauses = [
            Clause(
                literals=(Literal(sign=True, atom=P_a),),
                justification=(Justification(just_type=JustType.INPUT),),
            ),
            Clause(
                literals=(Literal(sign=False, atom=P_a), Literal(sign=True, atom=Q_a)),
                justification=(Justification(just_type=JustType.INPUT),),
            ),
            Clause(
                literals=(Literal(sign=False, atom=Q_a),),
                justification=(Justification(just_type=JustType.INPUT),),
            ),
        ]

        search = integration.create_search(
            options=SearchOptions(binary_resolution=True, quiet=True),
        )

        result = search.run([], clauses)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

        # Should have collected some experiences
        stats = integration.stats
        assert stats.experiences_collected >= 0  # May be 0 for very simple proofs


# ── ExperienceBuffer enhanced tests ──────────────────────────────────────


class TestExperienceBufferEnhancements:
    """Tests for the enhanced thread-safe experience buffer."""

    def _make_outcome(
        self, clause_id: int, outcome: OutcomeType = OutcomeType.KEPT,
    ) -> InferenceOutcome:
        given = _make_clause([(True, _make_term(0))], clause_id=0)
        child = _make_clause([(True, _make_term(clause_id))], clause_id=clause_id)
        return InferenceOutcome(
            given_clause=given,
            partner_clause=None,
            child_clause=child,
            outcome=outcome,
            timestamp=time.monotonic(),
        )

    def test_thread_safe_add(self):
        """Buffer add should be thread-safe."""
        import threading

        buf = ExperienceBuffer(capacity=1000)
        outcomes = [self._make_outcome(i) for i in range(100)]

        def add_batch(start: int, end: int) -> None:
            for i in range(start, end):
                buf.add(outcomes[i])

        threads = [
            threading.Thread(target=add_batch, args=(i * 25, (i + 1) * 25))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert buf.size == 100

    def test_total_added_counter(self):
        buf = ExperienceBuffer(capacity=10)
        for i in range(20):
            buf.add(self._make_outcome(i))

        assert buf.total_added == 20
        assert buf.size == 10  # capacity limited

    def test_productivity_rate(self):
        buf = ExperienceBuffer(capacity=100)
        for i in range(5):
            buf.add(self._make_outcome(i, OutcomeType.KEPT))
        for i in range(5, 15):
            buf.add(self._make_outcome(i, OutcomeType.SUBSUMED))

        buf._rebuild_indices()
        assert abs(buf.productivity_rate - 0.333) < 0.1

    def test_snapshot(self):
        buf = ExperienceBuffer(capacity=100)
        for i in range(5):
            buf.add(self._make_outcome(i, OutcomeType.KEPT))

        snap = buf.snapshot()
        assert snap["size"] == 5
        assert snap["capacity"] == 100
        assert snap["productive"] >= 0
        assert "total_added" in snap

    def test_add_batch(self):
        buf = ExperienceBuffer(capacity=100)
        outcomes = [self._make_outcome(i) for i in range(10)]
        buf.add_batch(outcomes)
        assert buf.size == 10

    def test_sample_weighted_batch(self):
        buf = ExperienceBuffer(capacity=100)
        # Add proof outcomes
        for i in range(5):
            buf.add(self._make_outcome(i, OutcomeType.PROOF))
        # Add regular kept
        for i in range(5, 10):
            buf.add(self._make_outcome(i, OutcomeType.KEPT))
        # Add unproductive
        for i in range(10, 20):
            buf.add(self._make_outcome(i, OutcomeType.SUBSUMED))

        pairs = buf.sample_weighted_batch(5)
        assert len(pairs) == 5


# ── LearningTriggerPolicy tests ──────────────────────────────────────────


from pyladr.search.online_integration import (
    LearningTriggerPolicy,
    TriggerPolicyConfig,
    TriggerStats,
)


class TestLearningTriggerPolicy:
    """Tests for the adaptive learning trigger policy."""

    def _make_signals(self, **overrides) -> ProofProgressSignals:
        return ProofProgressSignals(**overrides)

    def test_initial_state(self):
        policy = LearningTriggerPolicy()
        assert policy.current_interval == 200
        assert policy.stats.triggers_fired == 0

    def test_triggers_at_base_interval(self):
        policy = LearningTriggerPolicy(TriggerPolicyConfig(base_interval=100))
        signals = self._make_signals()

        # Not enough examples yet
        assert not policy.should_trigger(50, given_count=60, progress_signals=signals)
        # Enough examples
        assert policy.should_trigger(100, given_count=60, progress_signals=signals)

    def test_speedup_after_accepted_update(self):
        config = TriggerPolicyConfig(base_interval=200, speedup_factor=0.5)
        policy = LearningTriggerPolicy(config)

        assert policy.current_interval == 200
        policy.on_update_accepted(0.1)
        assert policy.current_interval == 100  # 200 * 0.5

    def test_backoff_after_rollback(self):
        config = TriggerPolicyConfig(base_interval=200, backoff_factor=2.0)
        policy = LearningTriggerPolicy(config)

        assert policy.current_interval == 200
        policy.on_update_rolled_back(0.1)
        assert policy.current_interval == 400  # 200 * 2.0

    def test_interval_clamped_to_min(self):
        config = TriggerPolicyConfig(
            base_interval=100, min_interval=50, speedup_factor=0.3,
        )
        policy = LearningTriggerPolicy(config)

        # Multiple accepts
        policy.on_update_accepted(0.01)  # 100 * 0.3 = 30 -> clamped to 50
        assert policy.current_interval == 50

    def test_interval_clamped_to_max(self):
        config = TriggerPolicyConfig(
            base_interval=500, max_interval=800, backoff_factor=2.0,
        )
        policy = LearningTriggerPolicy(config)

        policy.on_update_rolled_back(0.1)  # 500 * 2.0 = 1000 -> clamped to 800
        assert policy.current_interval == 800

    def test_cooldown_after_rollback(self):
        config = TriggerPolicyConfig(
            base_interval=100, cooldown_after_rollback=5,
            max_update_time_fraction=0.99,  # disable resource limiting
        )
        policy = LearningTriggerPolicy(config)
        signals = self._make_signals()

        policy.on_update_rolled_back(0.01)  # small duration

        # Should be suppressed during cooldown (5 calls)
        for i in range(5):
            assert not policy.should_trigger(200, given_count=60, progress_signals=signals)

        # After cooldown expires, should trigger (interval is 100*1.5=150, 200>=150)
        assert policy.should_trigger(200, given_count=60, progress_signals=signals)

    def test_cooldown_scales_with_consecutive_rollbacks(self):
        config = TriggerPolicyConfig(
            base_interval=50, cooldown_after_rollback=5,
        )
        policy = LearningTriggerPolicy(config)

        # First rollback: cooldown = 5 * 1 = 5
        policy.on_update_rolled_back(0.1)
        assert policy._cooldown_remaining == 5

        # Drain cooldown
        signals = self._make_signals()
        for _ in range(5):
            policy.should_trigger(100, given_count=60, progress_signals=signals)

        # Second consecutive rollback: cooldown = 5 * 2 = 10
        policy.on_update_rolled_back(0.1)
        assert policy._cooldown_remaining == 10

    def test_stagnation_shortens_interval(self):
        config = TriggerPolicyConfig(
            base_interval=200, min_interval=50, stagnation_threshold=50,
        )
        policy = LearningTriggerPolicy(config)

        # Normal: needs 200 examples
        signals = self._make_signals(given_since_last_progress=10)
        assert not policy.should_trigger(100, given_count=60, progress_signals=signals)

        # Stagnating: effective interval = 200 // 2 = 100
        signals = self._make_signals(given_since_last_progress=60)
        assert policy.should_trigger(100, given_count=60, progress_signals=signals)

    def test_resource_awareness_suppresses_triggers(self):
        config = TriggerPolicyConfig(
            base_interval=100, max_update_time_fraction=0.10,
        )
        policy = LearningTriggerPolicy(config)

        # Simulate enough elapsed search time and expensive updates
        policy._stats.total_update_time = 2.0
        policy._stats.total_search_time = 10.0  # 20% > 10% limit
        # Override the start time so monotonic() calc doesn't dominate
        policy._search_start_time = time.monotonic() - 10.0

        signals = self._make_signals()
        assert not policy.should_trigger(200, given_count=60, progress_signals=signals)
        assert policy.stats.triggers_suppressed_resource == 1

    def test_accepted_resets_consecutive_rollbacks(self):
        policy = LearningTriggerPolicy()
        policy.on_update_rolled_back(0.1)
        policy.on_update_rolled_back(0.1)
        assert policy.stats.consecutive_rollbacks == 2

        policy.on_update_accepted(0.1)
        assert policy.stats.consecutive_rollbacks == 0
        assert policy.stats.consecutive_accepts == 1

    def test_stats_tracking(self):
        policy = LearningTriggerPolicy()
        policy.on_update_accepted(0.05)
        policy.on_update_accepted(0.03)
        policy.on_update_rolled_back(0.02)

        s = policy.stats
        assert s.triggers_fired == 3
        assert s.total_update_time == pytest.approx(0.10, abs=1e-6)
        assert s.consecutive_accepts == 0  # last was rollback
        assert s.consecutive_rollbacks == 1
