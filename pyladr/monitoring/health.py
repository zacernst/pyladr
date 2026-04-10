"""Production health checks for the ML-enhanced theorem prover.

Provides a single entry point to assess the health of all ML subsystems
during a running search.  Designed for ops integration: returns structured
status that can be logged, serialized to JSON, or consumed by monitoring
dashboards.

Usage:
    from pyladr.monitoring.health import HealthChecker, SystemHealth

    checker = HealthChecker(
        learning_manager=mgr,
        embedding_cache=cache,
        memory_tracker=mem_tracker,
        learning_monitor=learning_mon,
        curve_analyzer=curves,
    )

    health = checker.check()
    print(health.status)         # "healthy" | "degraded" | "critical"
    print(health.to_dict())      # JSON-serializable summary
    print(health.report())       # Human-readable report
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyladr.ml.embeddings.cache import EmbeddingCache
    from pyladr.ml.online_learning import OnlineLearningManager
    from pyladr.monitoring.learning_curves import LearningCurveAnalyzer
    from pyladr.monitoring.learning_monitor import LearningMonitor
    from pyladr.monitoring.ml_memory import MLMemoryTracker

logger = logging.getLogger(__name__)


# ── Status types ───────────────────────────────────────────────────────────


class HealthStatus(IntEnum):
    """Overall system health."""

    HEALTHY = 0
    DEGRADED = 1
    CRITICAL = 2

    def __str__(self) -> str:
        return self.name.lower()


@dataclass(slots=True)
class ComponentHealth:
    """Health status for a single ML subsystem component."""

    name: str
    status: HealthStatus = HealthStatus.HEALTHY
    message: str = "ok"
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SystemHealth:
    """Aggregate health of the entire ML pipeline."""

    status: HealthStatus = HealthStatus.HEALTHY
    timestamp: float = 0.0
    components: list[ComponentHealth] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    @property
    def is_degraded(self) -> bool:
        return self.status == HealthStatus.DEGRADED

    @property
    def is_critical(self) -> bool:
        return self.status == HealthStatus.CRITICAL

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "status": str(self.status),
            "timestamp": self.timestamp,
            "components": [
                {
                    "name": c.name,
                    "status": str(c.status),
                    "message": c.message,
                    "metrics": c.metrics,
                }
                for c in self.components
            ],
            "alerts": self.alerts,
        }

    def report(self) -> str:
        """Human-readable health report."""
        lines = [
            "=" * 60,
            f"SYSTEM HEALTH: {str(self.status).upper()}",
            "=" * 60,
        ]

        for c in self.components:
            status_marker = {
                HealthStatus.HEALTHY: "[OK]",
                HealthStatus.DEGRADED: "[!!]",
                HealthStatus.CRITICAL: "[XX]",
            }[c.status]
            lines.append(f"  {status_marker} {c.name}: {c.message}")
            for k, v in c.metrics.items():
                lines.append(f"       {k}: {v}")

        if self.alerts:
            lines.append("")
            lines.append("Active alerts:")
            for alert in self.alerts:
                lines.append(f"  - {alert}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ── Health checker ─────────────────────────────────────────────────────────


class HealthChecker:
    """Unified health checker for all ML subsystem components.

    Accepts optional references to each ML component. Missing components
    are skipped — the checker works with whatever is available.
    """

    __slots__ = (
        "_manager", "_cache", "_memory_tracker",
        "_learning_monitor", "_curve_analyzer",
    )

    def __init__(
        self,
        learning_manager: OnlineLearningManager | None = None,
        embedding_cache: EmbeddingCache | None = None,
        memory_tracker: MLMemoryTracker | None = None,
        learning_monitor: LearningMonitor | None = None,
        curve_analyzer: LearningCurveAnalyzer | None = None,
    ) -> None:
        self._manager = learning_manager
        self._cache = embedding_cache
        self._memory_tracker = memory_tracker
        self._learning_monitor = learning_monitor
        self._curve_analyzer = curve_analyzer

    def check(self) -> SystemHealth:
        """Run all health checks and return aggregate status."""
        health = SystemHealth(timestamp=time.monotonic())

        if self._manager is not None:
            health.components.append(self._check_learning_manager())

        if self._cache is not None:
            health.components.append(self._check_embedding_cache())

        if self._memory_tracker is not None:
            health.components.append(self._check_memory())

        if self._learning_monitor is not None:
            health.components.append(self._check_learning_stability())

        if self._curve_analyzer is not None:
            health.components.append(self._check_convergence())

        # Aggregate: worst component status determines overall
        if health.components:
            health.status = max(c.status for c in health.components)

        # Collect alerts from learning monitor
        if self._learning_monitor is not None:
            for alert in self._learning_monitor.recent_alerts(limit=5):
                health.alerts.append(
                    f"[{alert.severity.name}] {alert.message}"
                )

        return health

    def _check_learning_manager(self) -> ComponentHealth:
        """Check online learning manager health."""
        mgr = self._manager
        assert mgr is not None

        stats = mgr.stats
        comp = ComponentHealth(
            name="online_learning",
            metrics={
                "update_count": stats["update_count"],
                "buffer_size": stats["buffer_size"],
                "buffer_productive": stats["buffer_productive"],
                "current_version": stats["current_version"],
                "loss_ema": f"{stats.get('loss_ema', 0.0):.4f}",
            },
        )

        # Check buffer balance
        total = stats["buffer_productive"] + stats["buffer_unproductive"]
        if total > 0:
            prod_rate = stats["buffer_productive"] / total
            comp.metrics["productivity_rate"] = f"{prod_rate:.3f}"

            if prod_rate < 0.05:
                comp.status = HealthStatus.DEGRADED
                comp.message = (
                    f"Very low productivity rate ({prod_rate:.1%}) — "
                    "ML may not be learning useful patterns"
                )
                return comp

        # Check if updates are happening
        if stats["buffer_size"] > 500 and stats["update_count"] == 0:
            comp.status = HealthStatus.DEGRADED
            comp.message = "Buffer has data but no updates have occurred"
            return comp

        comp.message = f"v{stats['current_version']}, {stats['update_count']} updates"
        return comp

    def _check_embedding_cache(self) -> ComponentHealth:
        """Check embedding cache health."""
        cache = self._cache
        assert cache is not None

        cache_stats = cache.stats.snapshot()
        comp = ComponentHealth(
            name="embedding_cache",
            metrics={
                "entries": len(cache),
                "hit_rate": f"{cache_stats['hit_rate']:.1%}",
                "evictions": cache_stats["evictions"],
                "memory_pressure_events": cache_stats["memory_pressure_events"],
            },
        )

        # Check hit rate (only meaningful after warmup)
        if cache_stats["total_lookups"] > 100:
            if cache_stats["hit_rate"] < 0.3:
                comp.status = HealthStatus.DEGRADED
                comp.message = (
                    f"Low cache hit rate ({cache_stats['hit_rate']:.1%}) — "
                    "may indicate cache thrashing"
                )
                return comp

        # Check memory pressure
        if cache_stats["memory_pressure_events"] > 10:
            comp.status = HealthStatus.DEGRADED
            comp.message = (
                f"{cache_stats['memory_pressure_events']} memory pressure events"
            )
            return comp

        comp.message = (
            f"{len(cache)} entries, "
            f"{cache_stats['hit_rate']:.0%} hit rate"
        )
        return comp

    def _check_memory(self) -> ComponentHealth:
        """Check ML memory usage."""
        tracker = self._memory_tracker
        assert tracker is not None

        comp = ComponentHealth(name="memory")

        if not tracker.snapshots:
            comp.message = "no data"
            return comp

        last = tracker.snapshots[-1]
        comp.metrics = {
            "ml_memory_mb": f"{last.total_ml_mb:.1f}",
            "version_memory_mb": f"{last.version_memory_bytes / (1024*1024):.1f}",
            "cache_memory_mb": f"{last.cache_memory_bytes / (1024*1024):.1f}",
            "growth_rate_mb_hr": f"{tracker.memory_growth_rate_mb_per_hour:.2f}",
            "cleanups": tracker.cleanup_count,
        }

        if tracker.is_over_hard_limit:
            comp.status = HealthStatus.CRITICAL
            comp.message = (
                f"ML memory ({last.total_ml_mb:.0f} MB) exceeds "
                f"hard limit ({tracker.budget.hard_limit_mb:.0f} MB)"
            )
            return comp

        if tracker.is_over_soft_limit:
            comp.status = HealthStatus.DEGRADED
            comp.message = (
                f"ML memory ({last.total_ml_mb:.0f} MB) exceeds "
                f"soft limit ({tracker.budget.soft_limit_mb:.0f} MB)"
            )
            return comp

        comp.message = f"{last.total_ml_mb:.1f} MB total ML memory"
        return comp

    def _check_learning_stability(self) -> ComponentHealth:
        """Check learning stability via the learning monitor."""
        mon = self._learning_monitor
        assert mon is not None

        comp = ComponentHealth(
            name="learning_stability",
            metrics={
                "updates_recorded": mon.update_count,
                "alerts_total": len(mon.alerts),
            },
        )

        # Check for critical alerts
        critical_count = sum(
            1 for a in mon.recent_alerts(limit=10)
            if a.severity.value >= 2  # WARNING or higher
        )

        if critical_count >= 3:
            comp.status = HealthStatus.CRITICAL
            comp.message = f"{critical_count} critical/warning alerts in recent history"
            return comp
        elif critical_count >= 1:
            comp.status = HealthStatus.DEGRADED
            comp.message = f"{critical_count} warning alert(s) detected"
            return comp

        # Check rollback frequency
        if mon.update_count > 5:
            rollback_rate = mon.rollback_count / mon.update_count
            comp.metrics["rollback_rate"] = f"{rollback_rate:.1%}"
            if rollback_rate > 0.5:
                comp.status = HealthStatus.DEGRADED
                comp.message = (
                    f"High rollback rate ({rollback_rate:.0%}) — "
                    "model updates frequently degrading"
                )
                return comp

        comp.message = f"{mon.update_count} updates, {len(mon.alerts)} alerts"
        return comp

    def _check_convergence(self) -> ComponentHealth:
        """Check learning convergence status."""
        curves = self._curve_analyzer
        assert curves is not None

        comp = ComponentHealth(
            name="convergence",
            metrics={
                "loss_count": curves.loss_count,
            },
        )

        if curves.loss_count < 3:
            comp.message = "insufficient data"
            return comp

        if curves.is_overfitting():
            comp.status = HealthStatus.DEGRADED
            comp.message = "overfitting detected — consider stopping updates"
            return comp

        if curves.has_converged():
            comp.message = "converged — learning complete"
        else:
            comp.message = "learning in progress"

        return comp


# ── Circuit breaker ────────────────────────────────────────────────────────


class CircuitBreakerState(IntEnum):
    """Circuit breaker states."""

    CLOSED = 0       # Normal operation — ML fully active
    HALF_OPEN = 1    # Degraded — ML weight reduced, monitoring increased
    OPEN = 2         # ML disabled — traditional selection only

    def __str__(self) -> str:
        return self.name.lower()


@dataclass(slots=True)
class CircuitBreakerAction:
    """An action taken by the circuit breaker."""

    timestamp: float
    old_state: CircuitBreakerState
    new_state: CircuitBreakerState
    reason: str
    ml_weight: float


class CircuitBreaker:
    """Automatic ML fallback based on health status.

    Monitors the health of the ML pipeline and automatically adjusts
    or disables ML guidance when problems are detected.  Implements
    a three-state circuit breaker:

    - **CLOSED** (normal): ML active at configured weight.
    - **HALF_OPEN** (degraded): ML weight reduced to ``degraded_ml_weight``.
      If health recovers, transitions back to CLOSED.  If health worsens,
      transitions to OPEN.
    - **OPEN** (disabled): ML completely disabled.  After ``recovery_checks``
      consecutive healthy checks, transitions to HALF_OPEN to probe recovery.

    Usage::

        breaker = CircuitBreaker(checker)

        # In the search loop (every N given clauses):
        ml_weight = breaker.evaluate()
        # Use ml_weight for clause selection (0.0 = pure traditional)
    """

    __slots__ = (
        "_checker", "_state", "_normal_ml_weight", "_degraded_ml_weight",
        "_consecutive_healthy", "_recovery_checks", "_actions",
        "_max_actions",
    )

    def __init__(
        self,
        checker: HealthChecker,
        normal_ml_weight: float = 0.3,
        degraded_ml_weight: float = 0.1,
        recovery_checks: int = 3,
        max_actions: int = 100,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            checker: HealthChecker to evaluate system health.
            normal_ml_weight: ML weight when CLOSED (healthy).
            degraded_ml_weight: ML weight when HALF_OPEN (degraded).
            recovery_checks: Consecutive healthy checks needed to recover.
            max_actions: Maximum action history entries (bounded).
        """
        self._checker = checker
        self._state = CircuitBreakerState.CLOSED
        self._normal_ml_weight = normal_ml_weight
        self._degraded_ml_weight = degraded_ml_weight
        self._consecutive_healthy = 0
        self._recovery_checks = recovery_checks
        self._actions: list[CircuitBreakerAction] = []
        self._max_actions = max_actions

    @property
    def state(self) -> CircuitBreakerState:
        return self._state

    @property
    def ml_weight(self) -> float:
        """Current ML weight based on circuit breaker state."""
        if self._state == CircuitBreakerState.CLOSED:
            return self._normal_ml_weight
        elif self._state == CircuitBreakerState.HALF_OPEN:
            return self._degraded_ml_weight
        else:
            return 0.0

    @property
    def actions(self) -> list[CircuitBreakerAction]:
        return list(self._actions)

    @property
    def trip_count(self) -> int:
        """Number of times the breaker has opened."""
        return sum(
            1 for a in self._actions
            if a.new_state == CircuitBreakerState.OPEN
        )

    def evaluate(self) -> float:
        """Run a health check and return the appropriate ML weight.

        This is the main entry point — call it periodically during search.
        Returns the ML selection weight to use (0.0 means ML disabled).
        """
        health = self._checker.check()
        old_state = self._state

        if health.is_critical:
            self._consecutive_healthy = 0
            if self._state != CircuitBreakerState.OPEN:
                self._transition(
                    CircuitBreakerState.OPEN,
                    f"Critical health: {self._summarize_health(health)}",
                )
        elif health.is_degraded:
            self._consecutive_healthy = 0
            if self._state == CircuitBreakerState.CLOSED:
                self._transition(
                    CircuitBreakerState.HALF_OPEN,
                    f"Degraded health: {self._summarize_health(health)}",
                )
            # If already OPEN, stay OPEN (degraded isn't good enough to recover)
        else:
            # Healthy
            self._consecutive_healthy += 1
            if self._state == CircuitBreakerState.OPEN:
                if self._consecutive_healthy >= self._recovery_checks:
                    self._transition(
                        CircuitBreakerState.HALF_OPEN,
                        f"Recovery probe after {self._consecutive_healthy} healthy checks",
                    )
                    self._consecutive_healthy = 0
            elif self._state == CircuitBreakerState.HALF_OPEN:
                if self._consecutive_healthy >= self._recovery_checks:
                    self._transition(
                        CircuitBreakerState.CLOSED,
                        "Fully recovered",
                    )

        return self.ml_weight

    def reset(self) -> None:
        """Force reset to CLOSED state."""
        if self._state != CircuitBreakerState.CLOSED:
            self._transition(CircuitBreakerState.CLOSED, "Manual reset")
        self._consecutive_healthy = 0

    def _transition(self, new_state: CircuitBreakerState, reason: str) -> None:
        """Record a state transition."""
        action = CircuitBreakerAction(
            timestamp=time.monotonic(),
            old_state=self._state,
            new_state=new_state,
            reason=reason,
            ml_weight=self._weight_for_state(new_state),
        )
        self._actions.append(action)
        # Bound the action history
        if len(self._actions) > self._max_actions:
            self._actions = self._actions[-self._max_actions:]

        logger.info(
            "Circuit breaker: %s -> %s (ml_weight=%.2f) — %s",
            self._state, new_state, action.ml_weight, reason,
        )
        self._state = new_state

    def _weight_for_state(self, state: CircuitBreakerState) -> float:
        if state == CircuitBreakerState.CLOSED:
            return self._normal_ml_weight
        elif state == CircuitBreakerState.HALF_OPEN:
            return self._degraded_ml_weight
        return 0.0

    def _summarize_health(self, health: SystemHealth) -> str:
        """One-line summary of degraded/critical components."""
        bad = [
            c.name for c in health.components
            if c.status != HealthStatus.HEALTHY
        ]
        return ", ".join(bad) if bad else "unknown"


# ── Production configuration presets ───────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ProductionConfig:
    """Recommended production-safe configuration values.

    These are conservative defaults tuned for stability over speed.
    Import and use as reference when constructing component configs.
    """

    # Online learning
    learning_rate: float = 5e-5
    buffer_capacity: int = 5000
    update_interval: int = 200
    min_examples_for_update: int = 50
    max_versions: int = 10
    rollback_threshold: float = 0.1
    ema_momentum: float = 0.995

    # Memory budgets
    memory_soft_limit_mb: float = 256.0
    memory_hard_limit_mb: float = 512.0

    # Embedding cache
    cache_max_entries: int = 100_000
    cache_gpu_memory_fraction: float = 0.85

    # Selection
    initial_ml_weight: float = 0.1
    max_ml_weight: float = 0.5
    min_given_before_ml: int = 50

    # Trigger policy
    trigger_base_interval: int = 200
    trigger_max_update_time_fraction: float = 0.15
    trigger_cooldown_after_rollback: int = 30

    # Monitoring
    memory_snapshot_interval: int = 50
    max_memory_snapshots: int = 1000
    health_check_interval: int = 100


# Singleton for easy import
PRODUCTION_DEFAULTS = ProductionConfig()
