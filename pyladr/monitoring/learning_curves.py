"""Learning curve analysis and convergence detection for online learning.

Provides tools to analyze learning behavior patterns, detect convergence,
identify overfitting, and compute learning effectiveness metrics.

Usage:
    analyzer = LearningCurveAnalyzer()

    # Feed loss values as they come in:
    for loss in loss_history:
        analyzer.add_loss(loss)

    # Check convergence:
    if analyzer.has_converged():
        print("Learning has converged")

    # Analyze curves:
    metrics = analyzer.compute_metrics()
    print(analyzer.report())
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# ── Learning curve metrics ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LearningCurveMetrics:
    """Computed metrics from a learning curve."""

    total_updates: int = 0

    # Loss statistics
    initial_loss: float = 0.0
    final_loss: float = 0.0
    min_loss: float = 0.0
    max_loss: float = 0.0
    mean_loss: float = 0.0
    loss_std: float = 0.0
    loss_reduction_pct: float = 0.0  # (initial - final) / initial * 100

    # Convergence
    is_converged: bool = False
    convergence_update: int = -1  # Update at which convergence was detected
    convergence_loss: float = 0.0

    # Overfitting indicators
    is_overfitting: bool = False
    best_loss_update: int = -1
    updates_since_best: int = 0

    # Learning speed
    loss_per_update: float = 0.0  # Average loss reduction per update
    half_life_updates: int = -1  # Updates to halve the loss

    # Stability
    loss_variance_recent: float = 0.0  # Variance over recent window
    oscillation_count: int = 0  # Number of loss direction changes

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_updates": self.total_updates,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "min_loss": self.min_loss,
            "max_loss": self.max_loss,
            "mean_loss": self.mean_loss,
            "loss_std": self.loss_std,
            "loss_reduction_pct": self.loss_reduction_pct,
            "is_converged": self.is_converged,
            "convergence_update": self.convergence_update,
            "convergence_loss": self.convergence_loss,
            "is_overfitting": self.is_overfitting,
            "best_loss_update": self.best_loss_update,
            "updates_since_best": self.updates_since_best,
            "loss_per_update": self.loss_per_update,
            "half_life_updates": self.half_life_updates,
            "loss_variance_recent": self.loss_variance_recent,
            "oscillation_count": self.oscillation_count,
        }


# ── Productivity curve metrics ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ProductivityMetrics:
    """Metrics from the productivity (search effectiveness) curve."""

    total_samples: int = 0
    initial_rate: float = 0.0
    final_rate: float = 0.0
    peak_rate: float = 0.0
    peak_sample: int = -1
    mean_rate: float = 0.0
    improvement_pct: float = 0.0  # (final - initial) / initial * 100
    trend: str = "unknown"  # "improving", "stable", "degrading"

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "initial_rate": self.initial_rate,
            "final_rate": self.final_rate,
            "peak_rate": self.peak_rate,
            "peak_sample": self.peak_sample,
            "mean_rate": self.mean_rate,
            "improvement_pct": self.improvement_pct,
            "trend": self.trend,
        }


# ── Learning curve analyzer ───────────────────────────────────────────────


class LearningCurveAnalyzer:
    """Analyzes learning curves to detect convergence, overfitting, and trends.

    Processes sequential loss values and productivity rates to provide
    actionable insights about the learning process.

    Usage:
        analyzer = LearningCurveAnalyzer()
        for loss in loss_values:
            analyzer.add_loss(loss)
        for rate in productivity_rates:
            analyzer.add_productivity(rate)

        metrics = analyzer.compute_loss_metrics()
        if analyzer.has_converged():
            print("Converged!")
    """

    __slots__ = (
        "_losses", "_productivity", "_convergence_window",
        "_convergence_threshold", "_overfitting_patience",
    )

    def __init__(
        self,
        convergence_window: int = 10,
        convergence_threshold: float = 0.005,
        overfitting_patience: int = 10,
    ) -> None:
        """Initialize learning curve analyzer.

        Args:
            convergence_window: Window of updates to check for convergence.
            convergence_threshold: Max relative change to consider converged.
            overfitting_patience: Updates after best loss before declaring overfit.
        """
        self._losses: list[float] = []
        self._productivity: list[float] = []
        self._convergence_window = convergence_window
        self._convergence_threshold = convergence_threshold
        self._overfitting_patience = overfitting_patience

    def add_loss(self, loss: float) -> None:
        """Add a loss value from a model update."""
        self._losses.append(loss)

    def add_productivity(self, rate: float) -> None:
        """Add a productivity rate measurement."""
        self._productivity.append(rate)

    @property
    def loss_count(self) -> int:
        return len(self._losses)

    @property
    def productivity_count(self) -> int:
        return len(self._productivity)

    # ── Convergence detection ─────────────────────────────────────────

    def has_converged(self) -> bool:
        """Check whether learning has converged.

        Uses a sliding window: converged when the relative change in
        mean loss over the window is below the threshold.
        """
        w = self._convergence_window
        if len(self._losses) < 2 * w:
            return False

        prev_window = self._losses[-2 * w:-w]
        curr_window = self._losses[-w:]

        prev_mean = sum(prev_window) / len(prev_window)
        curr_mean = sum(curr_window) / len(curr_window)

        if prev_mean == 0:
            return True  # Loss is zero, trivially converged

        relative_change = abs(curr_mean - prev_mean) / abs(prev_mean)
        return relative_change < self._convergence_threshold

    def convergence_point(self) -> int:
        """Find the update index at which convergence first occurred.

        Returns -1 if not yet converged.
        """
        w = self._convergence_window
        if len(self._losses) < 2 * w:
            return -1

        for i in range(2 * w, len(self._losses) + 1):
            prev_window = self._losses[i - 2 * w:i - w]
            curr_window = self._losses[i - w:i]

            prev_mean = sum(prev_window) / len(prev_window)
            curr_mean = sum(curr_window) / len(curr_window)

            if prev_mean == 0:
                return i - w

            relative_change = abs(curr_mean - prev_mean) / abs(prev_mean)
            if relative_change < self._convergence_threshold:
                return i - w

        return -1

    # ── Overfitting detection ─────────────────────────────────────────

    def is_overfitting(self) -> bool:
        """Check if the model appears to be overfitting.

        Overfitting is detected when loss has been increasing or not
        improving for more than `overfitting_patience` updates after
        the best loss was achieved.
        """
        if len(self._losses) < self._overfitting_patience + 1:
            return False

        best_idx = self._losses.index(min(self._losses))
        updates_since_best = len(self._losses) - 1 - best_idx

        return updates_since_best >= self._overfitting_patience

    # ── Metrics computation ───────────────────────────────────────────

    def compute_loss_metrics(self) -> LearningCurveMetrics:
        """Compute comprehensive metrics from the loss curve."""
        if not self._losses:
            return LearningCurveMetrics()

        losses = self._losses
        n = len(losses)

        mean_loss = sum(losses) / n
        variance = sum((l - mean_loss) ** 2 for l in losses) / n
        std_loss = math.sqrt(variance)

        min_loss = min(losses)
        max_loss = max(losses)
        best_idx = losses.index(min_loss)

        # Loss reduction
        initial = losses[0]
        final = losses[-1]
        reduction_pct = 0.0
        if initial > 0:
            reduction_pct = (initial - final) / initial * 100

        # Loss per update (linear approximation)
        loss_per_update = 0.0
        if n > 1:
            loss_per_update = (initial - final) / (n - 1)

        # Half-life
        half_life = -1
        if initial > 0:
            half = initial / 2
            for i, l in enumerate(losses):
                if l <= half:
                    half_life = i
                    break

        # Recent variance
        recent_n = min(10, n)
        recent = losses[-recent_n:]
        recent_mean = sum(recent) / len(recent)
        recent_var = sum((l - recent_mean) ** 2 for l in recent) / len(recent)

        # Oscillation count (direction changes)
        oscillations = 0
        for i in range(2, n):
            d1 = losses[i - 1] - losses[i - 2]
            d2 = losses[i] - losses[i - 1]
            if d1 * d2 < 0:
                oscillations += 1

        # Convergence
        converged = self.has_converged()
        conv_point = self.convergence_point()

        return LearningCurveMetrics(
            total_updates=n,
            initial_loss=initial,
            final_loss=final,
            min_loss=min_loss,
            max_loss=max_loss,
            mean_loss=mean_loss,
            loss_std=std_loss,
            loss_reduction_pct=reduction_pct,
            is_converged=converged,
            convergence_update=conv_point,
            convergence_loss=losses[conv_point] if conv_point >= 0 else 0.0,
            is_overfitting=self.is_overfitting(),
            best_loss_update=best_idx,
            updates_since_best=n - 1 - best_idx,
            loss_per_update=loss_per_update,
            half_life_updates=half_life,
            loss_variance_recent=recent_var,
            oscillation_count=oscillations,
        )

    def compute_productivity_metrics(self) -> ProductivityMetrics:
        """Compute metrics from the productivity curve."""
        if not self._productivity:
            return ProductivityMetrics()

        rates = self._productivity
        n = len(rates)
        initial = rates[0]
        final = rates[-1]
        peak = max(rates)
        peak_idx = rates.index(peak)
        mean_rate = sum(rates) / n

        improvement = 0.0
        if initial > 0:
            improvement = (final - initial) / initial * 100

        # Trend detection: compare first third vs last third
        if n >= 6:
            third = n // 3
            first_third = sum(rates[:third]) / third
            last_third = sum(rates[-third:]) / third
            if last_third > first_third * 1.05:
                trend = "improving"
            elif last_third < first_third * 0.95:
                trend = "degrading"
            else:
                trend = "stable"
        elif n >= 2:
            trend = "improving" if final > initial * 1.05 else (
                "degrading" if final < initial * 0.95 else "stable"
            )
        else:
            trend = "unknown"

        return ProductivityMetrics(
            total_samples=n,
            initial_rate=initial,
            final_rate=final,
            peak_rate=peak,
            peak_sample=peak_idx,
            mean_rate=mean_rate,
            improvement_pct=improvement,
            trend=trend,
        )

    # ── Smoothing ─────────────────────────────────────────────────────

    def smoothed_losses(self, window: int = 5) -> list[float]:
        """Return EMA-smoothed loss values.

        Args:
            window: Smoothing window (higher = smoother).

        Returns:
            List of smoothed loss values, same length as input.
        """
        if not self._losses:
            return []

        alpha = 2.0 / (window + 1)
        smoothed = [self._losses[0]]
        for loss in self._losses[1:]:
            smoothed.append(alpha * loss + (1 - alpha) * smoothed[-1])
        return smoothed

    def smoothed_productivity(self, window: int = 5) -> list[float]:
        """Return EMA-smoothed productivity values."""
        if not self._productivity:
            return []

        alpha = 2.0 / (window + 1)
        smoothed = [self._productivity[0]]
        for rate in self._productivity[1:]:
            smoothed.append(alpha * rate + (1 - alpha) * smoothed[-1])
        return smoothed

    # ── Reporting ─────────────────────────────────────────────────────

    def report(self) -> str:
        """Generate a learning curve analysis report."""
        lines = [
            "=" * 60,
            "LEARNING CURVE ANALYSIS",
            "=" * 60,
        ]

        if not self._losses:
            lines.append("No loss data recorded.")
            lines.append("=" * 60)
            return "\n".join(lines)

        lm = self.compute_loss_metrics()

        lines.append(f"Updates: {lm.total_updates}")
        lines.append("")

        # Loss summary
        lines.append("Loss curve:")
        lines.append(f"  Initial: {lm.initial_loss:.4f}")
        lines.append(f"  Final:   {lm.final_loss:.4f}")
        lines.append(f"  Best:    {lm.min_loss:.4f} (at update {lm.best_loss_update})")
        lines.append(f"  Reduction: {lm.loss_reduction_pct:.1f}%")

        if lm.half_life_updates >= 0:
            lines.append(f"  Half-life: {lm.half_life_updates} updates")

        # Convergence
        lines.append("")
        if lm.is_converged:
            lines.append(
                f"Converged at update {lm.convergence_update} "
                f"(loss: {lm.convergence_loss:.4f})"
            )
        else:
            lines.append("Not yet converged")

        # Overfitting
        if lm.is_overfitting:
            lines.append(
                f"WARNING: Possible overfitting detected "
                f"({lm.updates_since_best} updates since best loss)"
            )

        # Stability
        lines.append("")
        lines.append("Stability:")
        lines.append(f"  Recent variance: {lm.loss_variance_recent:.6f}")
        lines.append(f"  Oscillations: {lm.oscillation_count}")
        lines.append(f"  Loss std: {lm.loss_std:.4f}")

        # Productivity
        if self._productivity:
            pm = self.compute_productivity_metrics()
            lines.append("")
            lines.append("Productivity:")
            lines.append(f"  Initial: {pm.initial_rate:.3f}")
            lines.append(f"  Final:   {pm.final_rate:.3f}")
            lines.append(f"  Peak:    {pm.peak_rate:.3f} (at sample {pm.peak_sample})")
            lines.append(f"  Trend:   {pm.trend}")
            if pm.improvement_pct != 0:
                lines.append(f"  Change:  {pm.improvement_pct:+.1f}%")

        lines.append("=" * 60)
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all recorded data."""
        self._losses.clear()
        self._productivity.clear()
