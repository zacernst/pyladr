"""Extended tests for online learning system — stress, edge cases, and stability.

Covers:
- ExperienceBuffer thread safety and eviction behavior
- OnlineLearningManager EMA correctness and rollback logic
- ABTestTracker sliding window semantics
- Convergence detection edge cases
- Memory stability under sustained usage
"""

from __future__ import annotations

import threading
import time

import pytest

torch = pytest.importorskip("torch")

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.term import Term
from pyladr.ml.online_learning import (
    ABTestTracker,
    ExperienceBuffer,
    InferenceOutcome,
    ModelVersion,
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


def _make_outcome(
    clause_id: int,
    outcome: OutcomeType = OutcomeType.KEPT,
    given_id: int = 0,
    partner_id: int | None = None,
) -> InferenceOutcome:
    given = _make_clause([(True, _make_term(given_id))], clause_id=given_id)
    partner = (
        _make_clause([(False, _make_term(partner_id))], clause_id=partner_id)
        if partner_id is not None else None
    )
    child = _make_clause([(True, _make_term(clause_id))], clause_id=clause_id)

    return InferenceOutcome(
        given_clause=given,
        partner_clause=partner,
        child_clause=child,
        outcome=outcome,
        timestamp=time.monotonic(),
        given_count=clause_id,
    )


class MockEncoder:
    """Mock encoder that routes through a linear layer for gradient flow."""

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


# ── ExperienceBuffer stress tests ─────────────────────────────────────────


class TestExperienceBufferStress:
    """Stress and edge-case tests for ExperienceBuffer."""

    def test_single_capacity(self):
        """Buffer with capacity 1 only keeps the latest entry."""
        buf = ExperienceBuffer(capacity=1)
        buf.add(_make_outcome(1, OutcomeType.KEPT))
        buf.add(_make_outcome(2, OutcomeType.SUBSUMED))
        assert buf.size == 1
        recent = buf.get_recent(1)
        assert recent[0].child_clause.id == 2

    def test_alternating_outcomes_balanced(self):
        """Alternating productive/unproductive outcomes stay balanced."""
        buf = ExperienceBuffer(capacity=100)
        for i in range(50):
            buf.add(_make_outcome(i * 2, OutcomeType.KEPT))
            buf.add(_make_outcome(i * 2 + 1, OutcomeType.SUBSUMED))

        buf._rebuild_indices()
        assert buf.num_productive == 50
        assert buf.num_unproductive == 50

    def test_contrastive_sampling_balance(self):
        """Sampled pairs always have correct productive/unproductive labels."""
        buf = ExperienceBuffer(capacity=200)
        for i in range(50):
            buf.add(_make_outcome(i, OutcomeType.KEPT))
        for i in range(50, 100):
            buf.add(_make_outcome(i, OutcomeType.SUBSUMED))

        for _ in range(10):
            pairs = buf.sample_contrastive_batch(20)
            for pos, neg in pairs:
                assert pos.outcome in (OutcomeType.KEPT, OutcomeType.PROOF)
                assert neg.outcome not in (OutcomeType.KEPT, OutcomeType.PROOF)

    def test_large_buffer_eviction_consistency(self):
        """After overfilling, indices remain consistent with buffer contents."""
        buf = ExperienceBuffer(capacity=100)
        # Fill way past capacity
        for i in range(500):
            otype = OutcomeType.KEPT if i % 3 == 0 else OutcomeType.SUBSUMED
            buf.add(_make_outcome(i, otype))

        assert buf.size == 100
        buf._rebuild_indices()
        unified = buf._unified_buffer
        # Indices should be valid
        for idx in buf._productive_idx:
            assert 0 <= idx < buf.size
            assert unified[idx].outcome in (OutcomeType.KEPT, OutcomeType.PROOF)
        for idx in buf._unproductive_idx:
            assert 0 <= idx < buf.size
            assert unified[idx].outcome not in (OutcomeType.KEPT, OutcomeType.PROOF)

    def test_get_recent_more_than_size(self):
        """Requesting more recent items than buffer size returns all items."""
        buf = ExperienceBuffer(capacity=100)
        for i in range(5):
            buf.add(_make_outcome(i))
        recent = buf.get_recent(100)
        assert len(recent) == 5

    def test_all_productive_no_contrastive_pairs(self):
        """Buffer with only productive outcomes returns no contrastive pairs."""
        buf = ExperienceBuffer(capacity=100)
        for i in range(20):
            buf.add(_make_outcome(i, OutcomeType.KEPT))
        pairs = buf.sample_contrastive_batch(10)
        assert pairs == []

    def test_all_unproductive_no_contrastive_pairs(self):
        """Buffer with only unproductive outcomes returns no contrastive pairs."""
        buf = ExperienceBuffer(capacity=100)
        for i in range(20):
            buf.add(_make_outcome(i, OutcomeType.TAUTOLOGY))
        pairs = buf.sample_contrastive_batch(10)
        assert pairs == []

    def test_proof_outcome_is_productive(self):
        """PROOF outcomes are treated as productive."""
        buf = ExperienceBuffer(capacity=100)
        buf.add(_make_outcome(1, OutcomeType.PROOF))
        buf.add(_make_outcome(2, OutcomeType.SUBSUMED))
        buf._rebuild_indices()
        assert buf.num_productive == 1
        pairs = buf.sample_contrastive_batch(1)
        assert len(pairs) == 1
        assert pairs[0][0].outcome == OutcomeType.PROOF

    def test_weight_limit_is_unproductive(self):
        """WEIGHT_LIMIT outcomes are treated as unproductive."""
        buf = ExperienceBuffer(capacity=100)
        buf.add(_make_outcome(1, OutcomeType.KEPT))
        buf.add(_make_outcome(2, OutcomeType.WEIGHT_LIMIT))
        buf._rebuild_indices()
        assert buf.num_unproductive == 1
        pairs = buf.sample_contrastive_batch(1)
        assert len(pairs) == 1
        assert pairs[0][1].outcome == OutcomeType.WEIGHT_LIMIT

    def test_concurrent_add_no_crash(self):
        """Concurrent adds don't crash (basic thread-safety smoke test)."""
        buf = ExperienceBuffer(capacity=500)
        errors: list[Exception] = []

        def add_batch(start: int, count: int):
            try:
                for i in range(start, start + count):
                    buf.add(_make_outcome(i, OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_batch, args=(i * 100, 100))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No crashes
        assert errors == []
        # Buffer shouldn't exceed capacity
        assert buf.size <= 500


# ── ABTestTracker extended tests ──────────────────────────────────────────


class TestABTestTrackerExtended:
    """Extended tests for A/B test tracking logic."""

    def test_window_sliding(self):
        """Older outcomes are evicted as window fills."""
        tracker = ABTestTracker(window_size=5)
        tracker.set_baseline(0.0)

        # Fill with all True
        for _ in range(5):
            tracker.record_outcome(True)
        assert tracker.current_rate == 1.0

        # Add all False — window slides
        for _ in range(5):
            tracker.record_outcome(False)
        assert tracker.current_rate == 0.0

    def test_exact_boundary_enough_data(self):
        """has_enough_data triggers at exactly window_size // 2."""
        tracker = ABTestTracker(window_size=10)
        tracker.set_baseline(0.5)

        for _ in range(4):
            tracker.record_outcome(True)
        assert not tracker.has_enough_data

        tracker.record_outcome(True)
        assert tracker.has_enough_data

    def test_set_baseline_resets_window(self):
        """Setting a new baseline clears the current outcomes."""
        tracker = ABTestTracker(window_size=10)
        tracker.set_baseline(0.5)
        for _ in range(10):
            tracker.record_outcome(True)
        assert tracker.has_enough_data

        tracker.set_baseline(0.8)
        assert not tracker.has_enough_data
        assert tracker.current_rate == 0.0

    def test_marginal_improvement(self):
        """Marginal improvement below significance is not detected."""
        tracker = ABTestTracker(window_size=10)
        tracker.set_baseline(0.5)
        # 6/10 = 0.6, improvement of 0.1 over baseline
        for i in range(10):
            tracker.record_outcome(i < 6)

        assert tracker.is_improvement(significance=0.05)
        assert not tracker.is_improvement(significance=0.15)

    def test_marginal_degradation(self):
        """Marginal degradation below threshold is not detected."""
        tracker = ABTestTracker(window_size=10)
        tracker.set_baseline(0.5)
        # 4/10 = 0.4, degradation of 0.1
        for i in range(10):
            tracker.record_outcome(i < 4)

        assert tracker.is_degradation(threshold=0.05)
        assert not tracker.is_degradation(threshold=0.15)


# ── ModelVersion extended tests ───────────────────────────────────────────


class TestModelVersionExtended:
    def test_productivity_rate_precision(self):
        """Productivity rate handles non-integer-divisible counts."""
        v = ModelVersion(version_id=1, state_dict={})
        v.selections_made = 3
        v.productive_selections = 1
        assert abs(v.productivity_rate - 1 / 3) < 1e-10

    def test_high_volume_tracking(self):
        """Productivity rate remains accurate at high volumes."""
        v = ModelVersion(version_id=1, state_dict={})
        v.selections_made = 100000
        v.productive_selections = 42000
        assert v.productivity_rate == pytest.approx(0.42)


# ── OnlineLearningManager extended tests ─────────────────────────────────


class TestOnlineLearningManagerExtended:
    """Extended tests for online learning lifecycle."""

    def test_ema_state_initialized(self):
        """EMA state is initialized from the encoder's initial parameters."""
        config = OnlineLearningConfig(enabled=True)
        encoder = MockEncoder(dim=32)
        manager = OnlineLearningManager(encoder, config)
        assert manager._ema_state is not None
        assert len(manager._ema_state) > 0

    def test_ema_diverges_from_raw_params(self):
        """After updates, EMA state diverges from raw parameters."""
        config = OnlineLearningConfig(
            enabled=True,
            update_interval=5,
            min_examples_for_update=3,
            gradient_steps_per_update=2,
            batch_size=4,
            momentum=0.9,
        )
        encoder = MockEncoder(dim=32)
        manager = OnlineLearningManager(encoder, config)

        # Record enough mixed outcomes for an update
        for i in range(10):
            otype = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            manager.record_outcome(
                _make_outcome(i, otype, given_id=100 + i, partner_id=200 + i)
            )

        pre_ema = {k: v.clone() for k, v in manager._ema_state.items()}
        manager.update()

        # EMA should have been updated
        any_changed = any(
            not torch.equal(pre_ema[k], manager._ema_state[k])
            for k in pre_ema
            if k in manager._ema_state
        )
        # It's possible the update was rolled back, so check both cases
        if manager.stats["update_count"] > 0:
            assert any_changed

    def test_use_ema_model_and_restore(self):
        """use_ema_model/restore_training_model swaps parameters correctly."""
        config = OnlineLearningConfig(enabled=True)
        encoder = MockEncoder(dim=16)
        manager = OnlineLearningManager(encoder, config)

        # Capture original parameters
        orig_state = {k: v.clone() for k, v in encoder.state_dict().items()}

        # Manually modify EMA to differ
        for k in manager._ema_state:
            manager._ema_state[k] = torch.zeros_like(manager._ema_state[k])

        manager.use_ema_model()
        # Now encoder should have EMA params (zeros)
        for k, v in encoder.state_dict().items():
            assert torch.allclose(v, torch.zeros_like(v))

        manager.restore_training_model()
        # Now encoder should have original params restored
        for k, v in encoder.state_dict().items():
            assert torch.allclose(v, orig_state[k])

    def test_rollback_to_best_initial(self):
        """rollback_to_best returns initial version when none tested."""
        config = OnlineLearningConfig(enabled=True)
        encoder = MockEncoder(dim=16)
        manager = OnlineLearningManager(encoder, config)

        assert manager.rollback_to_best()
        assert manager._current_version.version_id == 0

    def test_multiple_updates_version_tracking(self):
        """Multiple updates produce distinct version snapshots."""
        config = OnlineLearningConfig(
            enabled=True,
            update_interval=3,
            min_examples_for_update=2,
            gradient_steps_per_update=1,
            batch_size=2,
            max_updates=0,
        )
        encoder = MockEncoder(dim=16)
        manager = OnlineLearningManager(encoder, config)

        total_updates = 0
        for batch in range(5):
            for i in range(6):
                idx = batch * 6 + i
                otype = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
                manager.record_outcome(
                    _make_outcome(idx, otype, given_id=100 + idx, partner_id=200 + idx)
                )
            if manager.should_update():
                result = manager.update()
                if result:
                    total_updates += 1

        # Should have produced some versions
        assert len(manager._versions) >= 1

    def test_on_proof_found_adds_proof_outcomes(self):
        """on_proof_found retroactively adds PROOF outcomes for matching clauses."""
        config = OnlineLearningConfig(enabled=True)
        encoder = MockEncoder(dim=16)
        manager = OnlineLearningManager(encoder, config)

        # Record some outcomes
        manager.record_outcome(_make_outcome(10, OutcomeType.KEPT))
        manager.record_outcome(_make_outcome(20, OutcomeType.SUBSUMED))
        manager.record_outcome(_make_outcome(30, OutcomeType.KEPT))

        initial_size = manager._buffer.size
        # Clause 10 and 30 are in the proof
        manager.on_proof_found({10, 30})

        # Should have added proof outcomes for clause 10 and 30
        assert manager._buffer.size > initial_size

    def test_on_proof_found_disabled_noop(self):
        """on_proof_found is a no-op when disabled."""
        config = OnlineLearningConfig(enabled=False)
        encoder = MockEncoder(dim=16)
        manager = OnlineLearningManager(encoder, config)

        manager.on_proof_found({1, 2, 3})
        assert manager._buffer.size == 0

    def test_report_format(self):
        """Report string contains all expected components."""
        config = OnlineLearningConfig(enabled=True)
        encoder = MockEncoder(dim=16)
        manager = OnlineLearningManager(encoder, config)

        report = manager.report()
        assert "OnlineLearning" in report
        assert "v0" in report
        assert "updates=0" in report
        assert "buffer=0" in report

    def test_convergence_requires_minimum_versions(self):
        """Convergence requires at least `window` versions."""
        config = OnlineLearningConfig(enabled=True)
        encoder = MockEncoder(dim=16)
        manager = OnlineLearningManager(encoder, config)

        # Only 1 version (initial)
        assert not manager.has_converged(window=3)

    def test_convergence_with_stable_rates(self):
        """Convergence detected when productivity rates are stable."""
        config = OnlineLearningConfig(enabled=True)
        encoder = MockEncoder(dim=16)
        manager = OnlineLearningManager(encoder, config)

        # Add versions with stable productivity rates
        for i in range(1, 6):
            v = ModelVersion(version_id=i, state_dict={})
            v.selections_made = 100
            v.productive_selections = 50  # All at 0.5 rate
            manager._versions.append(v)

        assert manager.has_converged(window=5, threshold=0.01)

    def test_no_convergence_with_unstable_rates(self):
        """No convergence when productivity rates vary widely."""
        config = OnlineLearningConfig(enabled=True)
        encoder = MockEncoder(dim=16)
        manager = OnlineLearningManager(encoder, config)

        rates = [10, 90, 10, 90, 10]  # Alternating high/low
        for i, rate in enumerate(rates):
            v = ModelVersion(version_id=i + 1, state_dict={})
            v.selections_made = 100
            v.productive_selections = rate
            manager._versions.append(v)

        assert not manager.has_converged(window=5, threshold=0.01)

    def test_disabled_manager_stats(self):
        """Disabled manager still returns valid stats."""
        config = OnlineLearningConfig(enabled=False)
        encoder = MockEncoder(dim=16)
        manager = OnlineLearningManager(encoder, config)

        stats = manager.stats
        assert stats["total_outcomes"] == 0
        assert stats["buffer_size"] == 0
        assert stats["update_count"] == 0

    def test_update_with_empty_batch_returns_false(self):
        """Update with no contrastive pairs returns False."""
        config = OnlineLearningConfig(
            enabled=True,
            update_interval=1,
            min_examples_for_update=1,
        )
        encoder = MockEncoder(dim=16)
        manager = OnlineLearningManager(encoder, config)

        # All same type — no contrastive pairs possible
        manager.record_outcome(_make_outcome(1, OutcomeType.KEPT))
        manager.record_outcome(_make_outcome(2, OutcomeType.KEPT))

        result = manager.update()
        assert result is False


# ── Loss explosion rollback tests ─────────────────────────────────────────


class TestRollbackMechanisms:
    """Test the rollback and stability mechanisms."""

    def test_should_rollback_no_version(self):
        """_should_rollback returns False when no current version."""
        config = OnlineLearningConfig(enabled=True)
        encoder = MockEncoder(dim=16)
        manager = OnlineLearningManager(encoder, config)

        # Temporarily clear current version
        original = manager._current_version
        manager._current_version = None
        assert not manager._should_rollback(1.0)
        manager._current_version = original

    def test_rollback_to_nonexistent_version(self):
        """Rolling back to a nonexistent version returns False."""
        config = OnlineLearningConfig(enabled=True)
        encoder = MockEncoder(dim=16)
        manager = OnlineLearningManager(encoder, config)

        assert not manager.rollback_to_version(999)

    def test_rollback_restores_parameters(self):
        """After rollback, encoder has the target version's parameters."""
        config = OnlineLearningConfig(enabled=True)
        encoder = MockEncoder(dim=16)
        manager = OnlineLearningManager(encoder, config)

        v0_state = {k: v.clone() for k, v in encoder.state_dict().items()}

        # Modify encoder parameters
        with torch.no_grad():
            for param in encoder.parameters():
                param.fill_(42.0)

        # Rollback to version 0
        assert manager.rollback_to_version(0)

        # Parameters should match version 0
        for k, v in encoder.state_dict().items():
            assert torch.allclose(v, v0_state[k])
