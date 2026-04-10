"""Tests for production health checks."""

from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term
from pyladr.ml.online_learning import (
    InferenceOutcome,
    OnlineLearningConfig,
    OnlineLearningManager,
    OutcomeType,
)
from pyladr.monitoring.health import (
    CircuitBreaker,
    CircuitBreakerState,
    ComponentHealth,
    HealthChecker,
    HealthStatus,
    ProductionConfig,
    PRODUCTION_DEFAULTS,
    SystemHealth,
)
from pyladr.monitoring.learning_curves import LearningCurveAnalyzer
from pyladr.monitoring.learning_monitor import LearningMonitor
from pyladr.monitoring.ml_memory import MLMemoryTracker, MemoryBudget


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
        return self._linear(torch.randn(len(clauses), self._dim))

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


def _make_manager(**kwargs) -> OnlineLearningManager:
    defaults = dict(
        enabled=True, update_interval=10, min_examples_for_update=5,
        buffer_capacity=200, batch_size=4, gradient_steps_per_update=1,
        learning_rate=1e-3,
    )
    defaults.update(kwargs)
    config = OnlineLearningConfig(**defaults)
    return OnlineLearningManager(
        MockEncoder(dim=64), config, device=torch.device("cpu"),
    )


# ── HealthStatus tests ────────────────────────────────────────────────────


class TestHealthStatus:
    def test_str_representation(self):
        assert str(HealthStatus.HEALTHY) == "healthy"
        assert str(HealthStatus.DEGRADED) == "degraded"
        assert str(HealthStatus.CRITICAL) == "critical"

    def test_ordering(self):
        assert HealthStatus.HEALTHY < HealthStatus.DEGRADED
        assert HealthStatus.DEGRADED < HealthStatus.CRITICAL


# ── ComponentHealth tests ──────────────────────────────────────────────────


class TestComponentHealth:
    def test_defaults(self):
        c = ComponentHealth(name="test")
        assert c.status == HealthStatus.HEALTHY
        assert c.message == "ok"
        assert c.metrics == {}


# ── SystemHealth tests ─────────────────────────────────────────────────────


class TestSystemHealth:
    def test_defaults(self):
        h = SystemHealth()
        assert h.is_healthy
        assert not h.is_degraded
        assert not h.is_critical

    def test_to_dict(self):
        h = SystemHealth(
            status=HealthStatus.DEGRADED,
            timestamp=1.0,
            components=[
                ComponentHealth(name="test", status=HealthStatus.DEGRADED),
            ],
            alerts=["test alert"],
        )
        d = h.to_dict()
        assert d["status"] == "degraded"
        assert len(d["components"]) == 1
        assert d["components"][0]["name"] == "test"
        assert d["alerts"] == ["test alert"]

    def test_report(self):
        h = SystemHealth(
            status=HealthStatus.HEALTHY,
            components=[
                ComponentHealth(
                    name="online_learning",
                    message="v2, 5 updates",
                    metrics={"update_count": 5},
                ),
            ],
        )
        report = h.report()
        assert "SYSTEM HEALTH: HEALTHY" in report
        assert "[OK] online_learning" in report

    def test_report_with_alerts(self):
        h = SystemHealth(
            status=HealthStatus.DEGRADED,
            alerts=["Loss spike detected"],
        )
        report = h.report()
        assert "Active alerts:" in report
        assert "Loss spike detected" in report


# ── HealthChecker tests ───────────────────────────────────────────────────


class TestHealthChecker:
    def test_empty_checker(self):
        checker = HealthChecker()
        health = checker.check()
        assert health.is_healthy
        assert len(health.components) == 0

    def test_with_learning_manager_healthy(self):
        mgr = _make_manager()
        for i in range(20):
            otype = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            mgr.record_outcome(_make_outcome(i, otype))

        checker = HealthChecker(learning_manager=mgr)
        health = checker.check()
        assert len(health.components) == 1
        assert health.components[0].name == "online_learning"
        assert health.components[0].status == HealthStatus.HEALTHY

    def test_with_memory_tracker(self):
        tracker = MLMemoryTracker()
        tracker.snapshot(iteration=1)

        checker = HealthChecker(memory_tracker=tracker)
        health = checker.check()
        comp = health.components[0]
        assert comp.name == "memory"
        assert comp.status == HealthStatus.HEALTHY

    def test_memory_tracker_no_data(self):
        tracker = MLMemoryTracker()
        checker = HealthChecker(memory_tracker=tracker)
        health = checker.check()
        comp = health.components[0]
        assert comp.message == "no data"

    def test_with_learning_monitor(self):
        mon = LearningMonitor()
        checker = HealthChecker(learning_monitor=mon)
        health = checker.check()
        comp = health.components[0]
        assert comp.name == "learning_stability"
        assert comp.status == HealthStatus.HEALTHY

    def test_with_curve_analyzer(self):
        curves = LearningCurveAnalyzer()
        checker = HealthChecker(curve_analyzer=curves)
        health = checker.check()
        comp = health.components[0]
        assert comp.name == "convergence"
        assert comp.message == "insufficient data"

    def test_with_curve_analyzer_converged(self):
        curves = LearningCurveAnalyzer(convergence_window=3, convergence_threshold=0.1)
        # Add losses that converge
        for _ in range(10):
            curves.add_loss(1.0)
        checker = HealthChecker(curve_analyzer=curves)
        health = checker.check()
        comp = health.components[0]
        assert "converged" in comp.message

    def test_aggregate_status_worst_wins(self):
        mgr = _make_manager()
        mon = LearningMonitor()
        # Record some updates with rollbacks to degrade stability
        for i in range(10):
            mon.record_update(
                update_id=i, accepted=False, avg_loss=5.0,
                grad_norm=0.1, max_grad=0.1,
                buffer_size=100, buffer_productive=50,
                buffer_unproductive=50, model_version=i,
                was_rollback=True,
            )

        checker = HealthChecker(learning_manager=mgr, learning_monitor=mon)
        health = checker.check()
        # Monitor should be degraded (high rollback rate), so overall degraded
        assert health.status >= HealthStatus.DEGRADED

    def test_full_system_check(self):
        mgr = _make_manager()
        for i in range(20):
            otype = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            mgr.record_outcome(_make_outcome(i, otype))

        tracker = MLMemoryTracker()
        tracker.snapshot(manager=mgr, iteration=1)

        mon = LearningMonitor()
        curves = LearningCurveAnalyzer()

        checker = HealthChecker(
            learning_manager=mgr,
            memory_tracker=tracker,
            learning_monitor=mon,
            curve_analyzer=curves,
        )
        health = checker.check()
        assert len(health.components) == 4
        assert health.timestamp > 0

    def test_health_report_readable(self):
        mgr = _make_manager()
        for i in range(10):
            mgr.record_outcome(_make_outcome(i, OutcomeType.KEPT))

        checker = HealthChecker(learning_manager=mgr)
        health = checker.check()
        report = health.report()
        assert "SYSTEM HEALTH" in report
        assert "online_learning" in report


# ── ProductionConfig tests ─────────────────────────────────────────────────


class TestProductionConfig:
    def test_defaults(self):
        cfg = ProductionConfig()
        assert cfg.learning_rate == 5e-5
        assert cfg.buffer_capacity == 5000
        assert cfg.memory_soft_limit_mb == 256.0
        assert cfg.cache_max_entries == 100_000
        assert cfg.initial_ml_weight == 0.1

    def test_singleton(self):
        assert PRODUCTION_DEFAULTS.learning_rate == 5e-5
        assert PRODUCTION_DEFAULTS.ema_momentum == 0.995

    def test_frozen(self):
        with pytest.raises(AttributeError):
            PRODUCTION_DEFAULTS.learning_rate = 1e-3  # type: ignore

    def test_import_from_package(self):
        from pyladr.monitoring import (
            HealthChecker, SystemHealth, ComponentHealth,
            HealthStatus, ProductionConfig, PRODUCTION_DEFAULTS,
            CircuitBreaker, CircuitBreakerState,
        )
        assert HealthChecker is not None
        assert PRODUCTION_DEFAULTS is not None
        assert CircuitBreaker is not None


# ── CircuitBreaker tests ──────────────────────────────────────────────────


class TestCircuitBreakerState:
    def test_str_representation(self):
        assert str(CircuitBreakerState.CLOSED) == "closed"
        assert str(CircuitBreakerState.HALF_OPEN) == "half_open"
        assert str(CircuitBreakerState.OPEN) == "open"

    def test_ordering(self):
        assert CircuitBreakerState.CLOSED < CircuitBreakerState.HALF_OPEN
        assert CircuitBreakerState.HALF_OPEN < CircuitBreakerState.OPEN


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        checker = HealthChecker()
        breaker = CircuitBreaker(checker)
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.ml_weight == 0.3

    def test_healthy_stays_closed(self):
        checker = HealthChecker()  # No components = healthy
        breaker = CircuitBreaker(checker)
        weight = breaker.evaluate()
        assert weight == 0.3
        assert breaker.state == CircuitBreakerState.CLOSED

    def test_degraded_transitions_to_half_open(self):
        # Create a monitor with 1 warning alert -> degraded (not critical)
        mon = LearningMonitor()
        # Mix of accepted and rollback to get degraded (high rollback rate)
        # but avoid excessive_rollbacks alert (which fires at ≥3 in 5 updates)
        for i in range(6):
            mon.record_update(
                update_id=i, accepted=(i < 2), avg_loss=1.0,
                grad_norm=0.1, max_grad=0.1,
                buffer_size=100, buffer_productive=50,
                buffer_unproductive=50, model_version=i,
                was_rollback=(i >= 2),
            )
        checker = HealthChecker(learning_monitor=mon)
        health = checker.check()
        # Verify it's actually degraded
        assert health.status == HealthStatus.DEGRADED

        breaker = CircuitBreaker(checker, normal_ml_weight=0.3, degraded_ml_weight=0.1)
        weight = breaker.evaluate()
        assert breaker.state == CircuitBreakerState.HALF_OPEN
        assert weight == 0.1
        assert len(breaker.actions) == 1

    def test_critical_transitions_to_open(self):
        # Memory over hard limit -> critical
        tracker = MLMemoryTracker(budget=MemoryBudget(hard_limit_mb=0.0001))
        mgr = _make_manager()
        for i in range(10):
            mgr.record_outcome(_make_outcome(i, OutcomeType.KEPT))
        tracker.snapshot(manager=mgr, iteration=1)

        checker = HealthChecker(memory_tracker=tracker)
        breaker = CircuitBreaker(checker)

        weight = breaker.evaluate()
        assert breaker.state == CircuitBreakerState.OPEN
        assert weight == 0.0

    def test_recovery_from_open(self):
        checker = HealthChecker()  # Healthy
        breaker = CircuitBreaker(checker, recovery_checks=2)

        # Force into OPEN state
        breaker._state = CircuitBreakerState.OPEN
        breaker._consecutive_healthy = 0

        # First healthy check — not enough to recover
        breaker.evaluate()
        assert breaker.state == CircuitBreakerState.OPEN

        # Second healthy check — transitions to HALF_OPEN
        breaker.evaluate()
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        # Two more healthy checks — transitions back to CLOSED
        breaker.evaluate()
        breaker.evaluate()
        assert breaker.state == CircuitBreakerState.CLOSED

    def test_non_healthy_while_open_stays_open(self):
        # Use same degraded monitor as above
        mon = LearningMonitor()
        for i in range(6):
            mon.record_update(
                update_id=i, accepted=(i < 2), avg_loss=1.0,
                grad_norm=0.1, max_grad=0.1,
                buffer_size=100, buffer_productive=50,
                buffer_unproductive=50, model_version=i,
                was_rollback=(i >= 2),
            )
        checker = HealthChecker(learning_monitor=mon)
        breaker = CircuitBreaker(checker)

        # Force to OPEN
        breaker._state = CircuitBreakerState.OPEN

        # Non-healthy should stay OPEN
        breaker.evaluate()
        assert breaker.state == CircuitBreakerState.OPEN

    def test_reset(self):
        checker = HealthChecker()
        breaker = CircuitBreaker(checker)
        breaker._state = CircuitBreakerState.OPEN

        breaker.reset()
        assert breaker.state == CircuitBreakerState.CLOSED
        assert len(breaker.actions) == 1
        assert breaker.actions[0].reason == "Manual reset"

    def test_trip_count(self):
        checker = HealthChecker()
        breaker = CircuitBreaker(checker)
        assert breaker.trip_count == 0

        # Simulate transitions
        breaker._transition(CircuitBreakerState.HALF_OPEN, "degraded")
        breaker._transition(CircuitBreakerState.OPEN, "critical")
        breaker._transition(CircuitBreakerState.HALF_OPEN, "recovering")
        breaker._transition(CircuitBreakerState.OPEN, "critical again")
        assert breaker.trip_count == 2

    def test_action_history_bounded(self):
        checker = HealthChecker()
        breaker = CircuitBreaker(checker, max_actions=5)
        for i in range(10):
            breaker._transition(CircuitBreakerState.HALF_OPEN, f"test {i}")
            breaker._transition(CircuitBreakerState.CLOSED, f"recover {i}")
        assert len(breaker.actions) <= 5

    def test_custom_weights(self):
        checker = HealthChecker()
        breaker = CircuitBreaker(
            checker,
            normal_ml_weight=0.5,
            degraded_ml_weight=0.15,
        )
        assert breaker.ml_weight == 0.5

        breaker._state = CircuitBreakerState.HALF_OPEN
        assert breaker.ml_weight == 0.15

        breaker._state = CircuitBreakerState.OPEN
        assert breaker.ml_weight == 0.0
