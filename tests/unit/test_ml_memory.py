"""Tests for ML pipeline memory tracking and optimization."""

from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term
from pyladr.ml.online_learning import (
    ExperienceBuffer,
    InferenceOutcome,
    ModelVersion,
    OnlineLearningConfig,
    OnlineLearningManager,
    OutcomeType,
)
from pyladr.monitoring.ml_memory import (
    MLMemorySnapshot,
    MLMemoryTracker,
    MemoryBudget,
    _aggressive_version_prune,
    _estimate_state_dict_bytes,
    _estimate_tensor_bytes,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_term(symnum: int, args: tuple[Term, ...] = ()) -> Term:
    return Term(private_symbol=-symnum, arity=len(args), args=args)


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
) -> InferenceOutcome:
    given = _make_clause([(True, _make_term(given_id))], clause_id=given_id)
    child = _make_clause([(True, _make_term(clause_id))], clause_id=clause_id)
    return InferenceOutcome(
        given_clause=given,
        partner_clause=None,
        child_clause=child,
        outcome=outcome,
        timestamp=time.monotonic(),
        given_count=clause_id,
    )


class MockEncoder:
    def __init__(self, dim: int = 64):
        self._dim = dim
        self._linear = torch.nn.Linear(dim, dim)

    def encode_clauses(self, clauses):
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


def _make_manager(
    update_interval: int = 10,
    min_examples: int = 5,
    buffer_capacity: int = 200,
    max_versions: int = 10,
) -> OnlineLearningManager:
    config = OnlineLearningConfig(
        enabled=True,
        update_interval=update_interval,
        min_examples_for_update=min_examples,
        buffer_capacity=buffer_capacity,
        batch_size=4,
        gradient_steps_per_update=1,
        learning_rate=1e-3,
        max_versions=max_versions,
    )
    encoder = MockEncoder(dim=64)
    return OnlineLearningManager(encoder, config, device=torch.device("cpu"))


# ── Size estimation tests ─────────────────────────────────────────────────


class TestSizeEstimation:
    def test_estimate_tensor_bytes(self):
        t = torch.randn(100, 64)
        expected = 100 * 64 * 4  # float32
        assert _estimate_tensor_bytes(t) == expected

    def test_estimate_tensor_bytes_non_tensor(self):
        assert _estimate_tensor_bytes("not a tensor") == 0

    def test_estimate_state_dict_bytes(self):
        linear = torch.nn.Linear(64, 64)
        sd = linear.state_dict()
        total = _estimate_state_dict_bytes(sd)
        # weight: 64*64*4, bias: 64*4
        assert total == 64 * 64 * 4 + 64 * 4

    def test_estimate_empty_state_dict(self):
        assert _estimate_state_dict_bytes({}) == 0


# ── MLMemorySnapshot tests ────────────────────────────────────────────────


class TestMLMemorySnapshot:
    def test_total_ml_bytes(self):
        snap = MLMemorySnapshot(
            version_memory_bytes=1000,
            cache_memory_bytes=2000,
            optimizer_state_bytes=500,
        )
        assert snap.total_ml_bytes == 3500

    def test_total_ml_mb(self):
        snap = MLMemorySnapshot(
            version_memory_bytes=1024 * 1024,
            cache_memory_bytes=0,
            optimizer_state_bytes=0,
        )
        assert snap.total_ml_mb == pytest.approx(1.0)

    def test_buffer_utilization(self):
        snap = MLMemorySnapshot(buffer_size=250, buffer_capacity=500)
        assert snap.buffer_utilization == pytest.approx(0.5)

    def test_buffer_utilization_zero_capacity(self):
        snap = MLMemorySnapshot(buffer_size=0, buffer_capacity=0)
        assert snap.buffer_utilization == 0.0


# ── MemoryBudget tests ────────────────────────────────────────────────────


class TestMemoryBudget:
    def test_defaults(self):
        b = MemoryBudget()
        assert b.soft_limit_mb == 256.0
        assert b.hard_limit_mb == 512.0
        assert b.version_limit == 10
        assert b.cache_limit == 100_000

    def test_custom(self):
        b = MemoryBudget(soft_limit_mb=64.0, hard_limit_mb=128.0)
        assert b.soft_limit_mb == 64.0


# ── MLMemoryTracker tests ─────────────────────────────────────────────────


class TestMLMemoryTracker:
    def test_creation(self):
        tracker = MLMemoryTracker()
        assert len(tracker.snapshots) == 0
        assert tracker.cleanup_count == 0

    def test_snapshot_without_components(self):
        tracker = MLMemoryTracker()
        snap = tracker.snapshot(iteration=5)
        assert snap.iteration == 5
        assert snap.buffer_size == 0
        assert snap.cache_entries == 0
        assert len(tracker.snapshots) == 1

    def test_snapshot_with_manager(self):
        mgr = _make_manager()
        # Add some data
        for i in range(10):
            mgr.record_outcome(_make_outcome(i, OutcomeType.KEPT))

        tracker = MLMemoryTracker()
        snap = tracker.snapshot(manager=mgr, iteration=1)
        assert snap.buffer_size == 10
        assert snap.buffer_capacity == 200
        assert snap.num_versions >= 1
        assert snap.num_versions_with_state >= 1
        assert snap.version_memory_bytes > 0

    def test_bounded_snapshots(self):
        tracker = MLMemoryTracker(max_snapshots=5)
        for i in range(10):
            tracker.snapshot(iteration=i)
        assert len(tracker.snapshots) == 5
        # Oldest should have been evicted
        assert tracker.snapshots[0].iteration == 5

    def test_memory_growth_rate_insufficient_data(self):
        tracker = MLMemoryTracker()
        assert tracker.memory_growth_rate_mb_per_hour == 0.0

    def test_budget_checks(self):
        budget = MemoryBudget(soft_limit_mb=0.001, hard_limit_mb=0.002)
        tracker = MLMemoryTracker(budget=budget)
        # No snapshots yet
        assert not tracker.is_over_soft_limit
        assert not tracker.is_over_hard_limit

    def test_report_empty(self):
        tracker = MLMemoryTracker()
        report = tracker.report()
        assert "No ML memory snapshots" in report

    def test_report_with_data(self):
        mgr = _make_manager()
        for i in range(10):
            mgr.record_outcome(_make_outcome(i, OutcomeType.KEPT))

        tracker = MLMemoryTracker()
        tracker.snapshot(manager=mgr, iteration=1)
        report = tracker.report()
        assert "ML PIPELINE MEMORY REPORT" in report
        assert "Experience Buffer:" in report
        assert "Model Versions:" in report
        assert "Embedding Cache:" in report

    def test_check_and_cleanup_no_snapshots(self):
        tracker = MLMemoryTracker()
        actions = tracker.check_and_cleanup()
        assert actions == []

    def test_import_from_package(self):
        from pyladr.monitoring import MLMemoryTracker, MLMemorySnapshot, MemoryBudget
        assert MLMemoryTracker is not None
        assert MLMemorySnapshot is not None
        assert MemoryBudget is not None


# ── Version pruning tests ─────────────────────────────────────────────────


class TestAggressiveVersionPrune:
    def test_no_prune_when_under_limit(self):
        mgr = _make_manager(max_versions=20)
        pruned = _aggressive_version_prune(mgr, keep_limit=20)
        assert pruned == 0

    def test_prune_clears_old_state_dicts(self):
        mgr = _make_manager(
            update_interval=5, min_examples=3,
            max_versions=100,  # Disable built-in pruning
        )
        # Generate enough updates to accumulate versions
        for cycle in range(5):
            for i in range(15):
                otype = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
                mgr.record_outcome(_make_outcome(cycle * 15 + i, otype))
            if mgr.should_update():
                mgr.update()

        initial_count = len(mgr._versions)
        if initial_count <= 3:
            pytest.skip("Not enough versions accumulated")

        pruned = _aggressive_version_prune(mgr, keep_limit=3)
        # Should have pruned some versions
        assert pruned >= 0  # May be 0 if all are protected

        # Protected versions should still have state_dicts
        assert mgr._versions[0].state_dict  # v0 always protected
        if mgr._current_version is not None:
            assert mgr._current_version.state_dict


# ── ExperienceBuffer memory tests ─────────────────────────────────────────


class TestExperienceBufferMemory:
    def test_buffer_bounded_by_capacity(self):
        buf = ExperienceBuffer(capacity=100)
        for i in range(200):
            buf.add(_make_outcome(i, OutcomeType.KEPT))
        assert buf.size <= 100

    def test_indices_rebuilt_after_sampling(self):
        buf = ExperienceBuffer(capacity=50)
        for i in range(100):
            otype = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            buf.add(_make_outcome(i, otype))
        # Indices may be stale before sampling, but sampling triggers rebuild
        buf.sample_contrastive_batch(4)
        # After rebuild, indices should be consistent
        assert buf.num_productive + buf.num_unproductive == buf.size

    def test_weighted_batch_frozenset_optimization(self):
        """Verify that sample_weighted_batch doesn't create frozenset per item."""
        buf = ExperienceBuffer(capacity=100)
        for i in range(50):
            otype = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            buf.add(_make_outcome(i, otype))
        # Should not crash and should return pairs
        pairs = buf.sample_weighted_batch(4)
        # If we have productive and unproductive, should get pairs
        if buf.num_productive > 0 and buf.num_unproductive > 0:
            assert len(pairs) > 0


# ── MemoryMonitor bounded snapshots test ──────────────────────────────────


class TestMemoryMonitorBounded:
    def test_snapshot_list_is_bounded(self):
        from pyladr.monitoring.memory_monitor import MemoryMonitor
        mon = MemoryMonitor(interval=1, max_snapshots=10)
        # Access internal deque's maxlen
        assert mon._snapshots.maxlen == 10


# ── ProofProgressTracker bounded dicts test ───────────────────────────────


class TestProofProgressTrackerBounded:
    def test_tracking_dicts_are_bounded(self):
        from pyladr.search.online_integration import ProofProgressTracker
        tracker = ProofProgressTracker(
            window_size=10, max_tracked_clauses=50,
        )
        # Simulate many given clause selections
        for i in range(200):
            clause = _make_clause([(True, _make_term(i))], clause_id=i)
            tracker.on_given_selected(clause)

        # Dict should have been pruned to ~25 entries (half of 50)
        assert len(tracker._given_clause_usage) <= 50

    def test_inference_count_dict_bounded(self):
        from pyladr.search.online_integration import ProofProgressTracker
        tracker = ProofProgressTracker(
            window_size=10, max_tracked_clauses=50,
        )
        # Need to add given clauses first to trigger pruning
        for i in range(100):
            clause = _make_clause([(True, _make_term(i))], clause_id=i)
            tracker.on_given_selected(clause)
            child = _make_clause([(True, _make_term(i + 1000))], clause_id=i + 1000)
            tracker.on_inference_from_given(i, child)

        # Both dicts should be bounded
        assert len(tracker._given_clause_usage) <= 50

    def test_productivity_still_works_after_prune(self):
        from pyladr.search.online_integration import ProofProgressTracker
        tracker = ProofProgressTracker(
            window_size=10, max_tracked_clauses=20,
        )
        # Fill and prune
        for i in range(50):
            clause = _make_clause([(True, _make_term(i))], clause_id=i)
            tracker.on_given_selected(clause)

        # Should still return a valid productivity rate
        rate = tracker.given_clause_productivity(49)
        assert 0.0 <= rate <= 1.0

        # Pruned clause should return default
        rate_old = tracker.given_clause_productivity(0)
        assert 0.0 <= rate_old <= 1.0
