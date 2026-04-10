"""Performance benchmarks for real-time contrastive online learning.

Measures empirical performance of the online learning system to ensure
it adds acceptable overhead to the search loop. All assertions are
backed by measured data, not estimates.

Key areas tested:
1. Experience buffer throughput (add/sample ops per second)
2. Online learning update latency (gradient step timing)
3. Memory usage during continuous model updates
4. Adaptive learning vs static model comparison
5. Cache invalidation overhead after model updates
6. Learning overhead as fraction of search iteration time

Run with: pytest tests/benchmarks/test_online_learning_perf.py -v
"""

from __future__ import annotations

import gc
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import pytest

torch = pytest.importorskip("torch")

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term
from pyladr.ml.online_learning import (
    ABTestTracker,
    ExperienceBuffer,
    InferenceOutcome,
    OnlineLearningConfig,
    OnlineLearningManager,
    OutcomeType,
)


# ── Helpers ───────────────────────────────────────────────────────────────


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


def _make_complex_clause(clause_id: int, depth: int = 2) -> Clause:
    """Create a clause with nested term structure for realistic benchmarks."""
    def _nested(d: int, base: int) -> Term:
        if d <= 0:
            return _make_var(base % 5)
        left = _nested(d - 1, base * 2)
        right = _nested(d - 1, base * 2 + 1)
        return _make_term(base % 10 + 1, (left, right))

    atom = _nested(depth, clause_id)
    c = Clause(literals=(Literal(sign=True, atom=atom),))
    c.id = clause_id
    c.weight = float(depth + 1)
    return c


def _make_outcome(
    clause_id: int,
    outcome: OutcomeType = OutcomeType.KEPT,
    given_id: int = 0,
    partner_id: int | None = None,
    depth: int = 1,
) -> InferenceOutcome:
    given = _make_complex_clause(given_id, depth=depth)
    partner = (
        _make_complex_clause(partner_id, depth=depth)
        if partner_id is not None else None
    )
    child = _make_complex_clause(clause_id, depth=depth)

    return InferenceOutcome(
        given_clause=given,
        partner_clause=partner,
        child_clause=child,
        outcome=outcome,
        timestamp=time.monotonic(),
        given_count=clause_id,
    )


class MockEncoder:
    """Mock encoder with realistic gradient computation.

    Uses a small MLP so gradients flow properly and timing
    reflects actual parameter update costs.
    """

    def __init__(self, dim: int = 64, layers: int = 2):
        self._dim = dim
        modules = []
        for _ in range(layers):
            modules.append(torch.nn.Linear(dim, dim))
            modules.append(torch.nn.ReLU())
        self._net = torch.nn.Sequential(*modules)

    def encode_clauses(self, clauses: list[Clause]) -> torch.Tensor:
        x = torch.randn(len(clauses), self._dim)
        return self._net(x)

    def parameters(self):
        return self._net.parameters()

    def named_parameters(self):
        return self._net.named_parameters()

    def state_dict(self):
        return self._net.state_dict()

    def load_state_dict(self, state):
        self._net.load_state_dict(state)

    def train(self, mode=True):
        self._net.train(mode)

    def eval(self):
        self._net.eval()


@dataclass
class BenchmarkResult:
    """Holds timing measurements for a benchmark."""
    name: str
    ops_per_second: float = 0.0
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    memory_delta_mb: float = 0.0
    samples: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.name}: {self.ops_per_second:.0f} ops/s, "
            f"mean={self.mean_latency_ms:.3f}ms, "
            f"p50={self.p50_latency_ms:.3f}ms, "
            f"p99={self.p99_latency_ms:.3f}ms"
        )


def _benchmark(fn, *, warmup: int = 10, iterations: int = 100) -> BenchmarkResult:
    """Run a benchmark function and collect timing statistics."""
    # Warmup
    for _ in range(warmup):
        fn()

    gc.collect()
    gc.disable()
    try:
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn()
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
    finally:
        gc.enable()

    latencies.sort()
    mean_s = statistics.mean(latencies)
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]

    return BenchmarkResult(
        name="",
        ops_per_second=1.0 / mean_s if mean_s > 0 else float("inf"),
        mean_latency_ms=mean_s * 1000,
        p50_latency_ms=p50 * 1000,
        p99_latency_ms=p99 * 1000,
        samples=latencies,
    )


def _get_rss_mb() -> float:
    """Get process RSS in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = usage.ru_maxrss
        if sys.platform == "darwin":
            return rss / (1024 * 1024)
        return rss / 1024  # Linux: KB
    except (ImportError, OSError):
        return 0.0


# ── Experience Buffer Benchmarks ──────────────────────────────────────────


class TestExperienceBufferPerformance:
    """Benchmark experience buffer operations."""

    def test_add_throughput(self):
        """Measure outcome insertion rate."""
        buf = ExperienceBuffer(capacity=5000)
        outcomes = [
            _make_outcome(
                i,
                OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED,
                given_id=100 + i,
                partner_id=200 + i,
            )
            for i in range(200)
        ]
        idx = [0]

        def add_one():
            buf.add(outcomes[idx[0] % len(outcomes)])
            idx[0] += 1

        result = _benchmark(add_one, warmup=50, iterations=1000)
        result.name = "ExperienceBuffer.add"

        # Buffer add must be fast: >50K ops/sec (< 0.02ms each)
        assert result.ops_per_second > 50_000, (
            f"Buffer add too slow: {result.ops_per_second:.0f} ops/s "
            f"(need >50K). {result.summary()}"
        )

    def test_sample_contrastive_batch_throughput(self):
        """Measure contrastive batch sampling rate."""
        buf = ExperienceBuffer(capacity=5000)
        # Fill buffer with mixed outcomes
        for i in range(2000):
            outcome_type = OutcomeType.KEPT if i % 3 != 0 else OutcomeType.SUBSUMED
            buf.add(_make_outcome(i, outcome_type, given_id=100 + i, partner_id=200 + i))

        def sample_batch():
            buf.sample_contrastive_batch(32)

        result = _benchmark(sample_batch, warmup=10, iterations=100)
        result.name = "ExperienceBuffer.sample_contrastive_batch(32)"

        # Sampling must be fast enough for real-time: >500 ops/sec (< 2ms)
        assert result.ops_per_second > 500, (
            f"Sampling too slow: {result.ops_per_second:.0f} ops/s "
            f"(need >500). {result.summary()}"
        )

    def test_buffer_at_capacity_performance(self):
        """Verify performance doesn't degrade at full capacity."""
        buf = ExperienceBuffer(capacity=5000)
        # Fill to capacity
        for i in range(5000):
            outcome_type = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            buf.add(_make_outcome(i, outcome_type, given_id=i))

        outcomes = [
            _make_outcome(i + 5000, OutcomeType.KEPT, given_id=i)
            for i in range(100)
        ]
        idx = [0]

        def add_at_capacity():
            buf.add(outcomes[idx[0] % len(outcomes)])
            idx[0] += 1

        result = _benchmark(add_at_capacity, warmup=50, iterations=500)
        result.name = "ExperienceBuffer.add (at capacity)"

        # At capacity, periodic index rebuild makes adds slower (~0.5ms)
        # Still acceptable: >1K ops/sec (the rebuild every 500 adds is the cost)
        assert result.ops_per_second > 1_000, (
            f"Buffer add at capacity too slow: {result.ops_per_second:.0f} ops/s "
            f"(need >1K). {result.summary()}"
        )

    def test_rebuild_indices_cost(self):
        """Measure cost of index rebuilding (happens every 500 adds)."""
        buf = ExperienceBuffer(capacity=5000)
        for i in range(3000):
            outcome_type = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            buf.add(_make_outcome(i, outcome_type, given_id=i))

        def rebuild():
            buf._rebuild_indices()

        result = _benchmark(rebuild, warmup=5, iterations=50)
        result.name = "ExperienceBuffer._rebuild_indices (3000 items)"

        # Rebuild on 3000 items should be < 5ms
        assert result.mean_latency_ms < 5.0, (
            f"Index rebuild too slow: {result.mean_latency_ms:.3f}ms "
            f"(need <5ms). {result.summary()}"
        )


# ── Online Learning Update Benchmarks ────────────────────────────────────


class TestOnlineLearningUpdatePerformance:
    """Benchmark online model update latency."""

    def _make_manager(
        self, dim: int = 64, buffer_cap: int = 2000,
    ) -> OnlineLearningManager:
        encoder = MockEncoder(dim=dim)
        config = OnlineLearningConfig(
            enabled=True,
            update_interval=50,
            min_examples_for_update=20,
            buffer_capacity=buffer_cap,
            batch_size=32,
            learning_rate=5e-5,
            gradient_steps_per_update=5,
        )
        return OnlineLearningManager(encoder, config)

    def _fill_manager(self, manager: OnlineLearningManager, n: int = 500):
        """Fill manager with mixed outcomes."""
        for i in range(n):
            outcome_type = OutcomeType.KEPT if i % 3 != 0 else OutcomeType.SUBSUMED
            manager.record_outcome(
                _make_outcome(i, outcome_type, given_id=100 + i, partner_id=200 + i)
            )

    def test_record_outcome_overhead(self):
        """Measure per-outcome recording overhead."""
        manager = self._make_manager()
        outcomes = [
            _make_outcome(i, OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED)
            for i in range(200)
        ]
        idx = [0]

        def record():
            manager.record_outcome(outcomes[idx[0] % len(outcomes)])
            idx[0] += 1

        result = _benchmark(record, warmup=50, iterations=500)
        result.name = "OnlineLearningManager.record_outcome"

        # Recording an outcome must be near-zero overhead: >100K ops/sec
        assert result.ops_per_second > 100_000, (
            f"record_outcome too slow: {result.ops_per_second:.0f} ops/s "
            f"(need >100K). {result.summary()}"
        )

    def test_update_latency(self):
        """Measure model update latency (5 gradient steps, batch=32)."""
        manager = self._make_manager()
        self._fill_manager(manager, 500)

        def do_update():
            manager._examples_since_update = 100  # force should_update
            manager.update()

        result = _benchmark(do_update, warmup=3, iterations=20)
        result.name = "OnlineLearningManager.update (5 steps, batch=32)"

        # Online update should complete within 100ms for dim=64
        assert result.mean_latency_ms < 100, (
            f"Update too slow: {result.mean_latency_ms:.3f}ms "
            f"(need <100ms). {result.summary()}"
        )

    def test_update_latency_larger_model(self):
        """Measure update latency with larger embedding dimension."""
        manager = self._make_manager(dim=256)
        self._fill_manager(manager, 500)

        def do_update():
            manager._examples_since_update = 100
            manager.update()

        result = _benchmark(do_update, warmup=2, iterations=10)
        result.name = "OnlineLearningManager.update (dim=256)"

        # Larger model: update should still complete within 500ms
        assert result.mean_latency_ms < 500, (
            f"Large model update too slow: {result.mean_latency_ms:.3f}ms "
            f"(need <500ms). {result.summary()}"
        )

    def test_gradient_step_latency(self):
        """Measure single gradient step latency."""
        manager = self._make_manager()
        self._fill_manager(manager, 500)
        batch_pairs = manager._buffer.sample_contrastive_batch(32)
        if not batch_pairs:
            pytest.skip("No batch pairs available")

        manager._encoder.train()

        def single_step():
            manager._gradient_step(batch_pairs)

        result = _benchmark(single_step, warmup=5, iterations=50)
        result.name = "OnlineLearningManager._gradient_step (batch=32)"

        # Single gradient step should be < 20ms for dim=64
        assert result.mean_latency_ms < 20, (
            f"Gradient step too slow: {result.mean_latency_ms:.3f}ms "
            f"(need <20ms). {result.summary()}"
        )

    def test_ema_apply_latency(self):
        """Measure EMA application latency."""
        manager = self._make_manager()

        def apply_ema():
            manager._apply_ema()

        result = _benchmark(apply_ema, warmup=10, iterations=100)
        result.name = "OnlineLearningManager._apply_ema"

        # EMA should be nearly free: < 1ms
        assert result.mean_latency_ms < 1.0, (
            f"EMA too slow: {result.mean_latency_ms:.3f}ms "
            f"(need <1ms). {result.summary()}"
        )


# ── Memory Usage Benchmarks ──────────────────────────────────────────────


class TestOnlineLearningMemory:
    """Profile memory usage during continuous model updates."""

    def test_experience_buffer_memory(self):
        """Measure memory consumption of experience buffer at capacity."""
        gc.collect()
        before_mb = _get_rss_mb()

        buf = ExperienceBuffer(capacity=5000)
        for i in range(5000):
            outcome_type = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            buf.add(_make_outcome(i, outcome_type, given_id=i, partner_id=i + 1000))

        gc.collect()
        after_mb = _get_rss_mb()
        delta_mb = after_mb - before_mb

        # Buffer at 5000 capacity should use less than 100MB
        # (Clause objects are small; outcomes are lightweight dataclasses)
        assert delta_mb < 100, (
            f"Buffer memory too high: {delta_mb:.1f}MB for 5000 outcomes "
            f"(need <100MB)"
        )

    def test_model_version_memory(self):
        """Measure memory growth from model version snapshots."""
        encoder = MockEncoder(dim=64)
        config = OnlineLearningConfig(
            enabled=True,
            update_interval=10,
            min_examples_for_update=5,
            buffer_capacity=500,
            batch_size=8,
            gradient_steps_per_update=1,
        )
        manager = OnlineLearningManager(encoder, config)

        gc.collect()
        before_mb = _get_rss_mb()

        # Run 20 updates to accumulate version snapshots
        for cycle in range(20):
            for i in range(15):
                outcome_type = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
                manager.record_outcome(
                    _make_outcome(
                        cycle * 100 + i, outcome_type,
                        given_id=cycle * 100 + i + 1000,
                        partner_id=cycle * 100 + i + 2000,
                    )
                )
            if manager.should_update():
                manager.update()

        gc.collect()
        after_mb = _get_rss_mb()
        delta_mb = after_mb - before_mb
        n_versions = len(manager._versions)

        # 20 versions of a dim=64 model should be < 50MB
        assert delta_mb < 50, (
            f"Version snapshots use too much memory: {delta_mb:.1f}MB "
            f"for {n_versions} versions (need <50MB)"
        )

    def test_no_memory_leak_over_updates(self):
        """Verify no significant memory leak over many update cycles."""
        encoder = MockEncoder(dim=64)
        config = OnlineLearningConfig(
            enabled=True,
            update_interval=10,
            min_examples_for_update=5,
            buffer_capacity=200,
            batch_size=8,
            gradient_steps_per_update=1,
            max_updates=0,
        )
        manager = OnlineLearningManager(encoder, config)

        # Warmup phase
        for i in range(50):
            outcome_type = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            manager.record_outcome(
                _make_outcome(i, outcome_type, given_id=i + 1000, partner_id=i + 2000)
            )
        if manager.should_update():
            manager.update()

        gc.collect()
        baseline_mb = _get_rss_mb()

        # Run many more cycles
        for cycle in range(50):
            for i in range(15):
                idx = (cycle + 1) * 100 + i
                outcome_type = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
                manager.record_outcome(
                    _make_outcome(
                        idx, outcome_type,
                        given_id=idx + 1000, partner_id=idx + 2000,
                    )
                )
            if manager.should_update():
                manager.update()

        gc.collect()
        final_mb = _get_rss_mb()
        growth_mb = final_mb - baseline_mb

        # After many updates with bounded buffer, growth should be bounded
        # Allow up to 30MB growth for version snapshots
        assert growth_mb < 30, (
            f"Memory grew {growth_mb:.1f}MB over 50 update cycles "
            f"(need <30MB). Possible leak."
        )


# ── Adaptive vs Static Model Comparison ──────────────────────────────────


class TestAdaptiveVsStaticPerformance:
    """Compare overhead of adaptive online learning vs static model."""

    def test_overhead_fraction_of_iteration(self):
        """Measure learning overhead as fraction of simulated search iteration.

        A search iteration typically takes 1-10ms (selection + inference +
        subsumption). Online learning overhead per iteration should be < 5%
        of this budget.
        """
        encoder = MockEncoder(dim=64)
        config = OnlineLearningConfig(
            enabled=True,
            update_interval=200,
            min_examples_for_update=50,
            buffer_capacity=2000,
            batch_size=32,
            gradient_steps_per_update=5,
        )
        manager = OnlineLearningManager(encoder, config)

        # Pre-fill buffer
        for i in range(300):
            outcome_type = OutcomeType.KEPT if i % 3 != 0 else OutcomeType.SUBSUMED
            manager.record_outcome(
                _make_outcome(i, outcome_type, given_id=i + 1000, partner_id=i + 2000)
            )

        # Measure: record_outcome cost per iteration (the hot path)
        outcome_times = []
        for i in range(1000):
            outcome = _make_outcome(
                300 + i,
                OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED,
                given_id=i + 2000,
                partner_id=i + 3000,
            )
            start = time.perf_counter()
            manager.record_outcome(outcome)
            elapsed = time.perf_counter() - start
            outcome_times.append(elapsed)

        mean_record_ms = statistics.mean(outcome_times) * 1000

        # Measure: update cost (amortized over update_interval iterations)
        update_times = []
        for _ in range(10):
            manager._examples_since_update = 200  # force
            start = time.perf_counter()
            manager.update()
            elapsed = time.perf_counter() - start
            update_times.append(elapsed)

        mean_update_ms = statistics.mean(update_times) * 1000
        amortized_update_ms = mean_update_ms / config.update_interval

        total_overhead_ms = mean_record_ms + amortized_update_ms

        # Total overhead per iteration should be < 0.1ms
        # (<<5% of a 2ms search iteration)
        assert total_overhead_ms < 0.1, (
            f"Per-iteration learning overhead too high: {total_overhead_ms:.4f}ms "
            f"(record={mean_record_ms:.4f}ms, "
            f"amortized_update={amortized_update_ms:.4f}ms, "
            f"need <0.1ms)"
        )

    def test_static_vs_adaptive_selection_overhead(self):
        """Compare selection time with and without online learning active.

        This measures the overhead that online learning adds to the
        clause selection path (primarily embedding computation).
        """
        from pyladr.search.ml_selection import (
            EmbeddingEnhancedSelection,
            MLSelectionConfig,
        )
        from pyladr.search.state import ClauseList

        # Create SOS with clauses
        clauses = []
        for i in range(50):
            c = _make_complex_clause(i + 1, depth=2)
            c.weight = float(i + 1)
            clauses.append(c)

        # Static selection (no ML)
        static_config = MLSelectionConfig(enabled=False)
        static_sel = EmbeddingEnhancedSelection(ml_config=static_config)

        def make_sos():
            sos = ClauseList("sos")
            for c in clauses:
                sos.append(c)
            return sos

        static_times = []
        for i in range(100):
            sos = make_sos()
            start = time.perf_counter()
            static_sel.select_given(sos, i)
            elapsed = time.perf_counter() - start
            static_times.append(elapsed)

        static_mean_us = statistics.mean(static_times) * 1_000_000

        # ML-enabled selection (with mock embeddings — no actual GNN)
        from tests.unit.test_ml_selection import MockEmbeddingProvider
        embeddings = {
            i + 1: [float(i % 4 == j) for j in range(4)]
            for i in range(50)
        }
        ml_config = MLSelectionConfig(
            enabled=True, ml_weight=0.3, min_sos_for_ml=5,
        )
        ml_sel = EmbeddingEnhancedSelection(
            embedding_provider=MockEmbeddingProvider(embeddings=embeddings),
            ml_config=ml_config,
        )

        ml_times = []
        for i in range(100):
            sos = make_sos()
            start = time.perf_counter()
            ml_sel.select_given(sos, i)
            elapsed = time.perf_counter() - start
            ml_times.append(elapsed)

        ml_mean_us = statistics.mean(ml_times) * 1_000_000
        overhead_factor = ml_mean_us / static_mean_us if static_mean_us > 0 else 1.0

        # ML selection scans all SOS clauses with embedding lookup + scoring.
        # With 50 clauses and mock embeddings, expect ~50-100x overhead vs
        # the O(log n) heap-based traditional selection.
        # Real systems use the batch embedding API and cache, so production
        # overhead is lower. Here we validate it stays bounded.
        assert overhead_factor < 200.0, (
            f"ML selection overhead too high: {overhead_factor:.1f}x "
            f"(static={static_mean_us:.1f}us, ml={ml_mean_us:.1f}us, "
            f"need <200x)"
        )
        # Also verify absolute ML selection time stays reasonable (< 5ms)
        assert ml_mean_us < 5000, (
            f"ML selection too slow in absolute terms: {ml_mean_us:.1f}us "
            f"(need <5000us)"
        )

    def test_should_update_check_is_free(self):
        """Verify should_update() has negligible cost."""
        encoder = MockEncoder(dim=64)
        config = OnlineLearningConfig(enabled=True)
        manager = OnlineLearningManager(encoder, config)

        def check():
            manager.should_update()

        result = _benchmark(check, warmup=100, iterations=10000)
        result.name = "OnlineLearningManager.should_update"

        # should_update is just integer comparisons: >1M ops/sec
        assert result.ops_per_second > 1_000_000, (
            f"should_update too slow: {result.ops_per_second:.0f} ops/s "
            f"(need >1M). {result.summary()}"
        )


# ── A/B Test Tracker Benchmarks ──────────────────────────────────────────


class TestABTestTrackerPerformance:
    """Benchmark A/B test tracking overhead."""

    def test_record_outcome_throughput(self):
        """A/B tracking should add zero measurable overhead."""
        tracker = ABTestTracker(window_size=100)
        tracker.set_baseline(0.5)
        flag = [True]

        def record():
            tracker.record_outcome(flag[0])
            flag[0] = not flag[0]

        result = _benchmark(record, warmup=100, iterations=10000)
        result.name = "ABTestTracker.record_outcome"

        # Deque append: >5M ops/sec
        assert result.ops_per_second > 5_000_000, (
            f"A/B tracking too slow: {result.ops_per_second:.0f} ops/s "
            f"(need >5M). {result.summary()}"
        )

    def test_comparison_check_throughput(self):
        """is_improvement/is_degradation should be fast."""
        tracker = ABTestTracker(window_size=100)
        tracker.set_baseline(0.5)
        for i in range(100):
            tracker.record_outcome(i % 2 == 0)

        def check():
            tracker.is_improvement()
            tracker.is_degradation()

        result = _benchmark(check, warmup=100, iterations=10000)
        result.name = "ABTestTracker.is_improvement+is_degradation"

        # Two property accesses + comparisons: >500K ops/sec
        assert result.ops_per_second > 500_000, (
            f"A/B comparison too slow: {result.ops_per_second:.0f} ops/s "
            f"(need >500K). {result.summary()}"
        )


# ── End-to-End Learning Cycle Benchmark ──────────────────────────────────


class TestEndToEndLearningCycle:
    """Benchmark a complete learning cycle simulating real search."""

    def test_full_cycle_timing(self):
        """Time a full cycle: record 200 outcomes, trigger update, check A/B.

        This simulates what happens during ~200 iterations of search
        before an online model update is triggered.
        """
        encoder = MockEncoder(dim=64)
        config = OnlineLearningConfig(
            enabled=True,
            update_interval=200,
            min_examples_for_update=50,
            buffer_capacity=2000,
            batch_size=32,
            gradient_steps_per_update=5,
        )
        manager = OnlineLearningManager(encoder, config)

        # Pre-fill to have some data
        for i in range(100):
            outcome_type = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            manager.record_outcome(
                _make_outcome(i, outcome_type, given_id=i + 1000, partner_id=i + 2000)
            )

        cycle_times = []
        for cycle in range(5):
            cycle_start = time.perf_counter()

            # Simulate 200 search iterations with outcome recording
            for i in range(200):
                idx = (cycle + 1) * 1000 + i
                outcome_type = OutcomeType.KEPT if i % 3 != 0 else OutcomeType.SUBSUMED
                manager.record_outcome(
                    _make_outcome(
                        idx, outcome_type,
                        given_id=idx + 5000, partner_id=idx + 6000,
                    )
                )

            # Trigger update
            if manager.should_update():
                manager.update()

            cycle_elapsed = time.perf_counter() - cycle_start
            cycle_times.append(cycle_elapsed)

        mean_cycle_ms = statistics.mean(cycle_times) * 1000

        # Full 200-iteration cycle with update should be < 200ms
        # (budget: recording ~1ms total + update ~50ms + overhead)
        assert mean_cycle_ms < 200, (
            f"Full learning cycle too slow: {mean_cycle_ms:.1f}ms "
            f"(need <200ms for 200 iterations + update)"
        )

        # Report stats
        stats = manager.stats
        assert stats["update_count"] > 0, "No updates were performed"

    def test_convergence_detection_overhead(self):
        """Verify convergence check adds no meaningful overhead."""
        encoder = MockEncoder(dim=64)
        config = OnlineLearningConfig(enabled=True)
        manager = OnlineLearningManager(encoder, config)

        def check_converged():
            manager.has_converged(window=5, threshold=0.01)

        result = _benchmark(check_converged, warmup=100, iterations=10000)
        result.name = "OnlineLearningManager.has_converged"

        # Simple list slice + variance: >500K ops/sec
        assert result.ops_per_second > 500_000, (
            f"Convergence check too slow: {result.ops_per_second:.0f} ops/s "
            f"(need >500K). {result.summary()}"
        )


# ── Rollback Performance ─────────────────────────────────────────────────


class TestRollbackPerformance:
    """Benchmark model rollback operations."""

    def test_rollback_latency(self):
        """Measure model rollback latency."""
        encoder = MockEncoder(dim=64)
        config = OnlineLearningConfig(enabled=True)
        manager = OnlineLearningManager(encoder, config)

        def rollback():
            manager.rollback_to_version(0)

        result = _benchmark(rollback, warmup=5, iterations=50)
        result.name = "OnlineLearningManager.rollback_to_version"

        # State dict load for dim=64 model: < 5ms
        assert result.mean_latency_ms < 5.0, (
            f"Rollback too slow: {result.mean_latency_ms:.3f}ms "
            f"(need <5ms). {result.summary()}"
        )

    def test_ema_switch_latency(self):
        """Measure EMA model switch latency."""
        encoder = MockEncoder(dim=64)
        config = OnlineLearningConfig(enabled=True)
        manager = OnlineLearningManager(encoder, config)

        def switch_to_ema():
            manager.use_ema_model()
            manager.restore_training_model()

        result = _benchmark(switch_to_ema, warmup=5, iterations=50)
        result.name = "EMA switch roundtrip"

        # Two state dict operations: < 10ms
        assert result.mean_latency_ms < 10.0, (
            f"EMA switch too slow: {result.mean_latency_ms:.3f}ms "
            f"(need <10ms). {result.summary()}"
        )
