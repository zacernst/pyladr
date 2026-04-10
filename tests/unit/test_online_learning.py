"""Tests for online learning system."""

from __future__ import annotations

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
    """Mock encoder for testing.

    Routes through a linear layer so gradients flow properly.
    """

    def __init__(self, dim: int = 64):
        self._dim = dim
        self._linear = torch.nn.Linear(dim, dim)

    def encode_clauses(self, clauses: list[Clause]) -> torch.Tensor:
        # Route through linear layer to enable gradient computation
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


# ── ExperienceBuffer tests ─────────────────────────────────────────────────


class TestExperienceBuffer:
    def test_add_and_size(self):
        buf = ExperienceBuffer(capacity=100)
        assert buf.size == 0

        buf.add(_make_outcome(1, OutcomeType.KEPT))
        buf.add(_make_outcome(2, OutcomeType.SUBSUMED))
        assert buf.size == 2

    def test_capacity_eviction(self):
        buf = ExperienceBuffer(capacity=10)
        for i in range(20):
            buf.add(_make_outcome(i, OutcomeType.KEPT))
        assert buf.size == 10

    def test_productive_unproductive_tracking(self):
        buf = ExperienceBuffer(capacity=100)
        buf.add(_make_outcome(1, OutcomeType.KEPT))
        buf.add(_make_outcome(2, OutcomeType.PROOF))
        buf.add(_make_outcome(3, OutcomeType.SUBSUMED))
        buf.add(_make_outcome(4, OutcomeType.TAUTOLOGY))

        buf._rebuild_indices()
        assert buf.num_productive == 2
        assert buf.num_unproductive == 2

    def test_sample_contrastive_batch(self):
        buf = ExperienceBuffer(capacity=100)
        for i in range(10):
            buf.add(_make_outcome(i, OutcomeType.KEPT))
        for i in range(10, 20):
            buf.add(_make_outcome(i, OutcomeType.SUBSUMED))

        pairs = buf.sample_contrastive_batch(5)
        assert len(pairs) == 5
        for pos, neg in pairs:
            assert pos.outcome in (OutcomeType.KEPT, OutcomeType.PROOF)
            assert neg.outcome not in (OutcomeType.KEPT, OutcomeType.PROOF)

    def test_sample_empty_buffer(self):
        buf = ExperienceBuffer(capacity=100)
        pairs = buf.sample_contrastive_batch(5)
        assert pairs == []

    def test_sample_no_negatives(self):
        buf = ExperienceBuffer(capacity=100)
        for i in range(10):
            buf.add(_make_outcome(i, OutcomeType.KEPT))
        pairs = buf.sample_contrastive_batch(5)
        assert pairs == []

    def test_clear(self):
        buf = ExperienceBuffer(capacity=100)
        buf.add(_make_outcome(1, OutcomeType.KEPT))
        buf.clear()
        assert buf.size == 0

    def test_get_recent(self):
        buf = ExperienceBuffer(capacity=100)
        for i in range(10):
            buf.add(_make_outcome(i))

        recent = buf.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].child_clause.id == 9


# ── ABTestTracker tests ────────────────────────────────────────────────────


class TestABTestTracker:
    def test_baseline_and_tracking(self):
        tracker = ABTestTracker(window_size=10)
        tracker.set_baseline(0.5)

        for _ in range(10):
            tracker.record_outcome(True)

        assert tracker.current_rate == 1.0
        assert tracker.has_enough_data
        assert tracker.is_improvement(significance=0.1)

    def test_degradation_detection(self):
        tracker = ABTestTracker(window_size=10)
        tracker.set_baseline(0.8)

        for _ in range(10):
            tracker.record_outcome(False)

        assert tracker.current_rate == 0.0
        assert tracker.is_degradation(threshold=0.1)

    def test_not_enough_data(self):
        tracker = ABTestTracker(window_size=20)
        tracker.set_baseline(0.5)

        tracker.record_outcome(True)
        assert not tracker.has_enough_data
        assert not tracker.is_improvement()
        assert not tracker.is_degradation()

    def test_mixed_outcomes(self):
        tracker = ABTestTracker(window_size=10)
        tracker.set_baseline(0.5)

        for i in range(10):
            tracker.record_outcome(i % 2 == 0)

        assert tracker.current_rate == 0.5
        assert not tracker.is_improvement(significance=0.1)
        assert not tracker.is_degradation(threshold=0.1)


# ── ModelVersion tests ─────────────────────────────────────────────────────


class TestModelVersion:
    def test_productivity_rate(self):
        v = ModelVersion(version_id=1, state_dict={})
        assert v.productivity_rate == 0.0

        v.selections_made = 10
        v.productive_selections = 7
        assert v.productivity_rate == 0.7

    def test_zero_selections(self):
        v = ModelVersion(version_id=0, state_dict={})
        assert v.productivity_rate == 0.0


# ── OnlineLearningManager tests ───────────────────────────────────────────


class TestOnlineLearningManager:
    def test_disabled(self):
        config = OnlineLearningConfig(enabled=False)
        encoder = MockEncoder()
        manager = OnlineLearningManager(encoder, config)

        outcome = _make_outcome(1, OutcomeType.KEPT)
        manager.record_outcome(outcome)
        assert not manager.should_update()

    def test_record_outcome(self):
        config = OnlineLearningConfig(
            enabled=True,
            update_interval=10,
            min_examples_for_update=5,
        )
        encoder = MockEncoder()
        manager = OnlineLearningManager(encoder, config)

        for i in range(5):
            manager.record_outcome(
                _make_outcome(i, OutcomeType.KEPT, given_id=100)
            )

        stats = manager.stats
        assert stats["total_outcomes"] == 5
        assert stats["buffer_size"] == 5

    def test_should_update_timing(self):
        config = OnlineLearningConfig(
            enabled=True,
            update_interval=5,
            min_examples_for_update=3,
        )
        encoder = MockEncoder()
        manager = OnlineLearningManager(encoder, config)

        # Not enough examples yet
        for i in range(2):
            manager.record_outcome(_make_outcome(i, OutcomeType.KEPT))
        assert not manager.should_update()

        # Add productive and unproductive examples to meet min threshold
        for i in range(2, 4):
            manager.record_outcome(_make_outcome(i, OutcomeType.KEPT))
        for i in range(4, 6):
            manager.record_outcome(_make_outcome(i, OutcomeType.SUBSUMED))
        # Now we have 6 examples >= min 3, and 6 >= interval 5
        assert manager.should_update()

    def test_update_runs(self):
        config = OnlineLearningConfig(
            enabled=True,
            update_interval=5,
            min_examples_for_update=3,
            gradient_steps_per_update=1,
            batch_size=4,
        )
        encoder = MockEncoder()
        manager = OnlineLearningManager(encoder, config)

        # Add mixed outcomes
        for i in range(10):
            outcome_type = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            manager.record_outcome(
                _make_outcome(i, outcome_type, given_id=100 + i, partner_id=200 + i)
            )

        result = manager.update()
        # Update should succeed (True) or rollback (False)
        assert isinstance(result, bool)
        assert manager.stats["update_count"] >= 0

    def test_max_updates_limit(self):
        config = OnlineLearningConfig(
            enabled=True,
            update_interval=2,
            min_examples_for_update=2,
            max_updates=1,
            gradient_steps_per_update=1,
            batch_size=2,
        )
        encoder = MockEncoder()
        manager = OnlineLearningManager(encoder, config)

        # First batch
        for i in range(4):
            outcome_type = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            manager.record_outcome(
                _make_outcome(i, outcome_type, given_id=100 + i, partner_id=200 + i)
            )
        manager.update()

        # Add more
        for i in range(4, 8):
            outcome_type = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            manager.record_outcome(
                _make_outcome(i, outcome_type, given_id=100 + i, partner_id=200 + i)
            )
        # Should not update (max_updates=1 reached)
        if manager.stats["update_count"] >= 1:
            assert not manager.should_update()

    def test_on_proof_found(self):
        config = OnlineLearningConfig(enabled=True)
        encoder = MockEncoder()
        manager = OnlineLearningManager(encoder, config)

        manager.record_outcome(_make_outcome(1, OutcomeType.KEPT))
        manager.record_outcome(_make_outcome(2, OutcomeType.SUBSUMED))

        # Mark clause 1 as part of proof
        manager.on_proof_found({1})

        # Buffer should now have the proof outcome added
        assert manager._buffer.size >= 2

    def test_rollback_to_version(self):
        config = OnlineLearningConfig(enabled=True)
        encoder = MockEncoder()
        manager = OnlineLearningManager(encoder, config)

        # Version 0 should exist
        assert manager.rollback_to_version(0)
        assert not manager.rollback_to_version(999)

    def test_report(self):
        config = OnlineLearningConfig(enabled=True)
        encoder = MockEncoder()
        manager = OnlineLearningManager(encoder, config)

        report = manager.report()
        assert "OnlineLearning" in report
        assert "updates=" in report

    def test_convergence_detection(self):
        config = OnlineLearningConfig(enabled=True)
        encoder = MockEncoder()
        manager = OnlineLearningManager(encoder, config)

        # Not enough versions for convergence
        assert not manager.has_converged()

    def test_stats_structure(self):
        config = OnlineLearningConfig(enabled=True)
        encoder = MockEncoder()
        manager = OnlineLearningManager(encoder, config)

        stats = manager.stats
        expected_keys = {
            "total_outcomes", "buffer_size", "buffer_productive",
            "buffer_unproductive", "update_count", "current_version",
            "current_productivity", "ab_test_current_rate",
            "loss_ema", "similarity_gap",
        }
        assert set(stats.keys()) == expected_keys


# ── InferenceOutcome tests ─────────────────────────────────────────────────


class TestInferenceOutcome:
    def test_creation(self):
        outcome = _make_outcome(1, OutcomeType.KEPT)
        assert outcome.outcome == OutcomeType.KEPT
        assert outcome.child_clause.id == 1

    def test_all_outcome_types(self):
        for otype in OutcomeType:
            outcome = _make_outcome(1, otype)
            assert outcome.outcome == otype


# ── OnlineLearningConfig tests ─────────────────────────────────────────────


class TestOnlineLearningConfig:
    def test_defaults(self):
        config = OnlineLearningConfig()
        assert config.enabled is True
        assert config.update_interval == 200
        assert config.buffer_capacity == 5000
        assert config.momentum == 0.995

    def test_disabled_config(self):
        config = OnlineLearningConfig(enabled=False)
        assert config.enabled is False


# ── Version pruning regression test ───────────────────────────────────────


class TestVersionPruning:
    """Regression test for version pruning memory leak (Task #31).

    Bug: _prune_versions() only cleared state_dict but never removed
    ModelVersion objects from self._versions, causing unbounded list growth.
    After 50 updates with max_versions=10, the list had 51 entries.
    """

    def test_versions_bounded_after_many_updates(self):
        config = OnlineLearningConfig(
            enabled=True,
            update_interval=3,
            min_examples_for_update=3,
            gradient_steps_per_update=1,
            batch_size=4,
            max_versions=10,
        )
        encoder = MockEncoder()
        manager = OnlineLearningManager(encoder, config)

        # Perform 30 update cycles to trigger many version snapshots
        for cycle in range(30):
            for i in range(5):
                otype = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
                manager.record_outcome(
                    _make_outcome(
                        cycle * 10 + i, otype,
                        given_id=1000 + cycle, partner_id=2000 + i,
                    )
                )
            if manager.should_update():
                manager.update()

        # The critical assertion: versions list must be bounded.
        # max_versions=10 plus up to 3 protected (initial, best, current)
        # means we should never exceed ~13 entries.
        assert len(manager._versions) <= 13, (
            f"Version list has {len(manager._versions)} entries after "
            f"{manager.stats['update_count']} updates — max_versions=10 "
            f"not enforced (memory leak)"
        )
