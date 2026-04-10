"""Bridge between OnlineLearningManager and the monitoring stack.

Provides a thin wrapper that automatically feeds learning events into
the LearningMonitor and LearningCurveAnalyzer without modifying the
OnlineLearningManager or the search loop.

Usage:
    from pyladr.monitoring.learning_bridge import MonitoredLearning

    # Wrap the existing learning manager:
    monitored = MonitoredLearning(learning_manager)

    # Use exactly like OnlineLearningManager:
    monitored.record_outcome(outcome)
    if monitored.should_update():
        accepted = monitored.update()

    # Access monitoring data:
    print(monitored.monitor.report())
    print(monitored.curves.report())
    health = monitored.buffer_health()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pyladr.monitoring.learning_monitor import (
    LearningMonitor,
    BufferHealth,
)
from pyladr.monitoring.learning_curves import LearningCurveAnalyzer

if TYPE_CHECKING:
    from pyladr.ml.online_learning import (
        InferenceOutcome,
        OnlineLearningManager,
    )
    from pyladr.search.statistics import SearchStatistics

logger = logging.getLogger(__name__)


class MonitoredLearning:
    """Transparent monitoring wrapper for OnlineLearningManager.

    Delegates all learning operations to the underlying manager while
    automatically recording metrics into the monitoring stack. Drop-in
    replacement: call the same methods, get monitoring for free.

    Does not modify any learning or search behavior.
    """

    __slots__ = ("_manager", "_monitor", "_curves", "_search_stats")

    def __init__(
        self,
        manager: OnlineLearningManager,
        monitor: LearningMonitor | None = None,
        curves: LearningCurveAnalyzer | None = None,
    ) -> None:
        """Initialize monitored learning wrapper.

        Args:
            manager: The OnlineLearningManager to monitor.
            monitor: Optional pre-configured LearningMonitor.
            curves: Optional pre-configured LearningCurveAnalyzer.
        """
        self._manager = manager
        self._monitor = monitor or LearningMonitor()
        self._curves = curves or LearningCurveAnalyzer()
        self._search_stats: SearchStatistics | None = None

    @property
    def manager(self) -> OnlineLearningManager:
        """Access the underlying learning manager."""
        return self._manager

    @property
    def monitor(self) -> LearningMonitor:
        """Access the learning monitor."""
        return self._monitor

    @property
    def curves(self) -> LearningCurveAnalyzer:
        """Access the learning curve analyzer."""
        return self._curves

    def set_search_stats(self, stats: SearchStatistics) -> None:
        """Set current search statistics for correlation tracking.

        Call this before update() so the monitor can correlate
        learning events with search progress.
        """
        self._search_stats = stats

    # ── Delegated operations with monitoring ──────────────────────────

    def record_outcome(self, outcome: InferenceOutcome) -> None:
        """Record inference outcome (delegates to manager)."""
        self._manager.record_outcome(outcome)

    def record_selection(self, ml_guided: bool, productive: bool) -> None:
        """Record a clause selection outcome into the monitor.

        Call this after each given clause selection to track
        ML vs traditional selection effectiveness.
        """
        self._monitor.record_selection(ml_guided, productive)

    def should_update(self) -> bool:
        """Check if a model update is due (delegates to manager)."""
        return self._manager.should_update()

    def update(self) -> bool:
        """Perform model update and record monitoring data.

        Wraps OnlineLearningManager.update() with automatic
        metric collection into the monitor and curve analyzer.

        Returns:
            True if the update was accepted, False if rolled back.
        """
        accepted = self._manager.update()

        # Extract metrics from manager's stats
        stats = self._manager.stats
        update_count = stats["update_count"]

        # Compute gradient norm from model parameters if available
        grad_norm = 0.0
        max_grad = 0.0
        try:
            import torch
            for param in self._manager._encoder.parameters():
                if param.grad is not None:
                    pn = param.grad.data.norm(2).item()
                    grad_norm += pn ** 2
                    max_grad = max(max_grad, param.grad.data.abs().max().item())
            grad_norm = grad_norm ** 0.5
        except (ImportError, AttributeError, RuntimeError):
            pass

        # Get loss from the current version
        avg_loss = 0.0
        if self._manager._current_version is not None:
            avg_loss = self._manager._current_version.avg_loss
            if avg_loss == float("inf"):
                avg_loss = 0.0

        # Record in monitor
        self._monitor.record_update(
            update_id=update_count,
            accepted=accepted,
            avg_loss=avg_loss,
            grad_norm=grad_norm,
            max_grad=max_grad,
            buffer_size=stats["buffer_size"],
            buffer_productive=stats["buffer_productive"],
            buffer_unproductive=stats["buffer_unproductive"],
            model_version=stats["current_version"],
            was_rollback=not accepted,
            search_stats=self._search_stats,
        )

        # Record in curve analyzer
        if avg_loss > 0:
            self._curves.add_loss(avg_loss)
        productivity = stats.get("current_productivity", 0.0)
        if isinstance(productivity, (int, float)):
            self._curves.add_productivity(float(productivity))

        return accepted

    def on_proof_found(self, proof_clause_ids: set[int]) -> None:
        """Notify of proof found (delegates to manager)."""
        self._manager.on_proof_found(proof_clause_ids)

    # ── Monitoring queries ────────────────────────────────────────────

    def buffer_health(self) -> BufferHealth:
        """Assess current experience buffer health."""
        stats = self._manager.stats
        return self._monitor.assess_buffer_health(
            buffer_size=stats["buffer_size"],
            capacity=self._manager._buffer._capacity,
            productive=stats["buffer_productive"],
            unproductive=stats["buffer_unproductive"],
        )

    def report(self) -> str:
        """Combined monitoring and curve analysis report."""
        parts = [self._monitor.report()]

        if self._curves.loss_count > 0:
            parts.append("")
            parts.append(self._curves.report())

        return "\n".join(parts)

    def has_converged(self) -> bool:
        """Check convergence via both manager and curve analyzer."""
        return (
            self._manager.has_converged()
            or self._curves.has_converged()
        )

    def is_overfitting(self) -> bool:
        """Check if the learning curve suggests overfitting."""
        return self._curves.is_overfitting()
