"""Tests for adaptive learning rate scheduling."""

from __future__ import annotations

import pytest

from pyladr.ml.lr_scheduling import (
    ConvergenceAwareScheduler,
    ExponentialDecayScheduler,
    LRSchedulerConfig,
    LRSchedulerState,
    PerformanceBasedScheduler,
    SchedulerType,
    create_lr_scheduler,
)


# ── ExponentialDecayScheduler ─────────────────────────────────────────────


class TestExponentialDecayScheduler:
    def test_warmup_ramp(self):
        config = LRSchedulerConfig(
            strategy=SchedulerType.EXPONENTIAL_DECAY,
            initial_lr=1e-4,
            warmup_updates=3,
        )
        sched = ExponentialDecayScheduler(config)

        lr1 = sched.step(0.5, 1.0)
        lr2 = sched.step(0.5, 0.9)
        lr3 = sched.step(0.5, 0.8)

        # Should ramp up during warmup
        assert lr1 < lr2 < lr3
        assert lr3 == pytest.approx(1e-4, rel=0.01)

    def test_decay_after_warmup(self):
        config = LRSchedulerConfig(
            strategy=SchedulerType.EXPONENTIAL_DECAY,
            initial_lr=1e-4,
            warmup_updates=2,
            decay_factor=0.9,
        )
        sched = ExponentialDecayScheduler(config)

        # Warmup
        sched.step(0.5, 1.0)
        sched.step(0.5, 1.0)

        # Decay steps
        lr3 = sched.step(0.5, 0.8)
        lr4 = sched.step(0.5, 0.7)

        assert lr4 < lr3
        assert lr3 == pytest.approx(1e-4 * 0.9, rel=0.01)
        assert lr4 == pytest.approx(1e-4 * 0.9**2, rel=0.01)

    def test_respects_min_lr(self):
        config = LRSchedulerConfig(
            strategy=SchedulerType.EXPONENTIAL_DECAY,
            initial_lr=1e-4,
            min_lr=5e-5,
            warmup_updates=0,
            decay_factor=0.1,
        )
        sched = ExponentialDecayScheduler(config)

        for _ in range(20):
            sched.step(0.5, 0.5)

        assert sched.current_lr >= config.min_lr

    def test_state_tracking(self):
        sched = ExponentialDecayScheduler()
        sched.step(0.4, 1.0)
        sched.step(0.6, 0.8)

        state = sched.state
        assert state.update_count == 2
        assert state.adjustments == 2
        assert state.last_productivity == 0.6
        assert "ExponentialDecay" in state.strategy_name


# ── PerformanceBasedScheduler ────────────────────────────────────────────


class TestPerformanceBasedScheduler:
    def test_boosts_on_struggle(self):
        config = LRSchedulerConfig(
            strategy=SchedulerType.PERFORMANCE_BASED,
            initial_lr=1e-4,
            warmup_updates=1,
            struggle_threshold=0.2,
            boost_factor=1.5,
        )
        sched = PerformanceBasedScheduler(config)
        sched.step(0.1, 1.0)  # warmup

        lr_before = sched.current_lr
        sched.step(0.1, 1.0)  # struggling
        lr_after = sched.current_lr

        assert lr_after > lr_before

    def test_reduces_on_success(self):
        config = LRSchedulerConfig(
            strategy=SchedulerType.PERFORMANCE_BASED,
            initial_lr=1e-4,
            warmup_updates=1,
            success_threshold=0.6,
            reduction_factor=0.7,
        )
        sched = PerformanceBasedScheduler(config)
        sched.step(0.8, 0.5)  # warmup

        lr_before = sched.current_lr
        sched.step(0.8, 0.5)  # successful
        lr_after = sched.current_lr

        assert lr_after < lr_before

    def test_neutral_zone_gentle_decay(self):
        config = LRSchedulerConfig(
            strategy=SchedulerType.PERFORMANCE_BASED,
            initial_lr=1e-4,
            warmup_updates=1,
            struggle_threshold=0.2,
            success_threshold=0.6,
        )
        sched = PerformanceBasedScheduler(config)
        sched.step(0.4, 0.7)  # warmup

        lr_before = sched.current_lr
        sched.step(0.4, 0.7)  # neutral
        lr_after = sched.current_lr

        # Should decay gently (0.98x)
        assert lr_after < lr_before
        assert lr_after > lr_before * 0.95

    def test_respects_bounds(self):
        config = LRSchedulerConfig(
            strategy=SchedulerType.PERFORMANCE_BASED,
            initial_lr=1e-4,
            min_lr=1e-6,
            max_lr=5e-4,
            warmup_updates=1,
            boost_factor=10.0,  # aggressive
        )
        sched = PerformanceBasedScheduler(config)
        sched.step(0.1, 1.0)  # warmup

        # Boost repeatedly
        for _ in range(20):
            sched.step(0.01, 2.0)

        assert sched.current_lr <= config.max_lr
        assert sched.current_lr >= config.min_lr


# ── ConvergenceAwareScheduler ────────────────────────────────────────────


class TestConvergenceAwareScheduler:
    def test_reduces_on_plateau(self):
        config = LRSchedulerConfig(
            strategy=SchedulerType.CONVERGENCE_AWARE,
            initial_lr=1e-4,
            warmup_updates=1,
            convergence_patience=3,
            convergence_reduction=0.5,
            decay_factor=1.0,  # disable decay to isolate plateau effect
        )
        sched = ConvergenceAwareScheduler(config)
        sched.step(0.5, 0.5)  # warmup

        # Feed identical productivity to trigger plateau
        for _ in range(10):
            sched.step(0.5, 0.5)

        # Should have reduced LR due to detected convergence
        assert sched.current_lr < config.initial_lr
        assert sched.state.converged

    def test_loss_spike_recovery(self):
        config = LRSchedulerConfig(
            strategy=SchedulerType.CONVERGENCE_AWARE,
            initial_lr=1e-4,
            warmup_updates=1,
            boost_factor=1.5,
            decay_factor=1.0,
        )
        sched = ConvergenceAwareScheduler(config)
        sched.step(0.5, 0.5)  # warmup

        # Normal losses
        sched.step(0.5, 0.5)
        sched.step(0.5, 0.5)
        sched.step(0.5, 0.5)

        lr_before = sched.current_lr

        # Spike losses (3x higher)
        sched.step(0.3, 1.5)
        sched.step(0.3, 1.5)
        sched.step(0.3, 1.5)

        # LR should have been boosted to recover
        assert sched.current_lr >= lr_before

    def test_default_gentle_decay(self):
        config = LRSchedulerConfig(
            strategy=SchedulerType.CONVERGENCE_AWARE,
            initial_lr=1e-4,
            warmup_updates=1,
            decay_factor=0.95,
            convergence_patience=100,  # high patience = no plateau trigger
        )
        sched = ConvergenceAwareScheduler(config)
        sched.step(0.5, 0.5)  # warmup

        # Varying productivity (no plateau)
        rates = [0.3, 0.5, 0.7, 0.4, 0.6]
        for r in rates:
            sched.step(r, 0.5)

        # Should have decayed gently
        assert sched.current_lr < config.initial_lr


# ── Factory ──────────────────────────────────────────────────────────────


class TestFactory:
    def test_creates_exponential(self):
        config = LRSchedulerConfig(strategy=SchedulerType.EXPONENTIAL_DECAY)
        sched = create_lr_scheduler(config)
        assert isinstance(sched, ExponentialDecayScheduler)

    def test_creates_performance(self):
        config = LRSchedulerConfig(strategy=SchedulerType.PERFORMANCE_BASED)
        sched = create_lr_scheduler(config)
        assert isinstance(sched, PerformanceBasedScheduler)

    def test_creates_convergence(self):
        config = LRSchedulerConfig(strategy=SchedulerType.CONVERGENCE_AWARE)
        sched = create_lr_scheduler(config)
        assert isinstance(sched, ConvergenceAwareScheduler)

    def test_default_is_convergence_aware(self):
        sched = create_lr_scheduler()
        assert isinstance(sched, ConvergenceAwareScheduler)


# ── Optimizer integration ────────────────────────────────────────────────


class TestOptimizerIntegration:
    def test_attach_optimizer(self):
        torch = pytest.importorskip("torch")

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        sched = ExponentialDecayScheduler(
            LRSchedulerConfig(initial_lr=5e-5, warmup_updates=0, decay_factor=0.9)
        )
        sched.attach_optimizer(optimizer)

        # Optimizer LR should now match scheduler
        assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)

        sched.step(0.5, 0.5)
        new_lr = optimizer.param_groups[0]["lr"]
        assert new_lr == pytest.approx(5e-5 * 0.9, rel=0.01)

    def test_lr_stays_in_bounds_with_optimizer(self):
        torch = pytest.importorskip("torch")

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        config = LRSchedulerConfig(
            initial_lr=1e-4, min_lr=1e-6, max_lr=5e-4,
            warmup_updates=0, decay_factor=0.5,
        )
        sched = ExponentialDecayScheduler(config)
        sched.attach_optimizer(optimizer)

        for _ in range(50):
            sched.step(0.5, 0.5)

        lr = optimizer.param_groups[0]["lr"]
        assert lr >= config.min_lr
        assert lr <= config.max_lr


# ── State summary ────────────────────────────────────────────────────────


class TestStateSummary:
    def test_summary_format(self):
        sched = ExponentialDecayScheduler()
        sched.step(0.42, 0.8)

        summary = sched.state.summary()
        assert "LR=" in summary
        assert "updates=1" in summary
        assert "productivity=0.420" in summary


# ── Performance benchmarks ───────────────────────────────────────────────


class TestSchedulerPerformance:
    """Verify scheduling overhead is negligible."""

    def test_step_latency(self):
        import time

        sched = ConvergenceAwareScheduler()

        # Warmup
        for i in range(10):
            sched.step(0.5, 0.5)

        # Benchmark
        start = time.perf_counter()
        for _ in range(10000):
            sched.step(0.45, 0.6)
        elapsed = time.perf_counter() - start

        # 10K steps should complete in < 100ms (< 10us each)
        assert elapsed < 0.1, (
            f"Scheduler step too slow: {elapsed / 10000 * 1e6:.1f}us per step "
            f"(need <10us)"
        )
