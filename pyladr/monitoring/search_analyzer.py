"""Search statistics collection and analysis.

Extends the basic SearchStatistics with per-iteration tracking,
rate computation, and trend analysis for identifying bottlenecks
and optimization opportunities.

Non-intrusive: reads from SearchStatistics and SearchState without
modifying any search behavior.

Usage:
    analyzer = SearchAnalyzer()

    # After each iteration in the search loop:
    analyzer.record_iteration(state.stats, state)

    # After search completes:
    print(analyzer.report())
    hotspots = analyzer.identify_hotspots()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyladr.search.state import SearchState
    from pyladr.search.statistics import SearchStatistics


@dataclass(slots=True)
class IterationSnapshot:
    """Snapshot of search statistics at a single iteration."""

    iteration: int = 0
    timestamp: float = 0.0

    # Cumulative stats at this point
    given: int = 0
    generated: int = 0
    kept: int = 0
    subsumed: int = 0
    back_subsumed: int = 0
    demodulated: int = 0

    # Per-iteration deltas (computed from previous snapshot)
    delta_generated: int = 0
    delta_kept: int = 0
    delta_subsumed: int = 0

    # State sizes
    usable_size: int = 0
    sos_size: int = 0

    # Derived metrics
    keep_rate: float = 0.0  # kept / generated for this iteration

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "given": self.given,
            "generated": self.generated,
            "kept": self.kept,
            "subsumed": self.subsumed,
            "back_subsumed": self.back_subsumed,
            "demodulated": self.demodulated,
            "delta_generated": self.delta_generated,
            "delta_kept": self.delta_kept,
            "delta_subsumed": self.delta_subsumed,
            "usable_size": self.usable_size,
            "sos_size": self.sos_size,
            "keep_rate": self.keep_rate,
        }


class SearchAnalyzer:
    """Non-intrusive search statistics analyzer.

    Records per-iteration snapshots and computes derived metrics
    for identifying bottlenecks and optimization opportunities.

    Usage:
        analyzer = SearchAnalyzer()
        # Record after each given clause iteration:
        analyzer.record_iteration(stats, state)
        # Generate report:
        print(analyzer.report())
    """

    __slots__ = ("_snapshots", "_start_time")

    def __init__(self) -> None:
        self._snapshots: list[IterationSnapshot] = []
        self._start_time = time.monotonic()

    @property
    def snapshots(self) -> list[IterationSnapshot]:
        return list(self._snapshots)

    @property
    def iteration_count(self) -> int:
        return len(self._snapshots)

    def record_iteration(
        self,
        stats: SearchStatistics,
        state: SearchState | None = None,
    ) -> IterationSnapshot:
        """Record statistics after a search iteration.

        Reads from SearchStatistics and optionally SearchState to
        compute per-iteration deltas and derived metrics.
        """
        iteration = len(self._snapshots)
        now = time.monotonic() - self._start_time

        snap = IterationSnapshot(
            iteration=iteration,
            timestamp=now,
            given=stats.given,
            generated=stats.generated,
            kept=stats.kept,
            subsumed=stats.subsumed,
            back_subsumed=stats.back_subsumed,
            demodulated=stats.demodulated,
        )

        # Compute deltas from previous snapshot
        if self._snapshots:
            prev = self._snapshots[-1]
            snap.delta_generated = stats.generated - prev.generated
            snap.delta_kept = stats.kept - prev.kept
            snap.delta_subsumed = stats.subsumed - prev.subsumed

            # Keep rate for this iteration
            if snap.delta_generated > 0:
                snap.keep_rate = snap.delta_kept / snap.delta_generated

        # State sizes
        if state is not None:
            snap.usable_size = state.usable.length
            snap.sos_size = state.sos.length

        self._snapshots.append(snap)
        return snap

    def identify_hotspots(self, top_n: int = 5) -> list[dict[str, Any]]:
        """Identify iterations with highest inference generation.

        Returns the top N iterations by delta_generated, which typically
        correspond to the most expensive given clauses.
        """
        if not self._snapshots:
            return []

        sorted_snaps = sorted(
            self._snapshots,
            key=lambda s: s.delta_generated,
            reverse=True,
        )

        return [
            {
                "iteration": s.iteration,
                "given_clause": s.given,
                "delta_generated": s.delta_generated,
                "delta_kept": s.delta_kept,
                "keep_rate": s.keep_rate,
                "usable_size": s.usable_size,
            }
            for s in sorted_snaps[:top_n]
        ]

    def compute_rates(self) -> dict[str, float]:
        """Compute overall and windowed rates."""
        if len(self._snapshots) < 2:
            return {}

        first = self._snapshots[0]
        last = self._snapshots[-1]
        duration = last.timestamp - first.timestamp

        if duration <= 0:
            return {}

        rates: dict[str, float] = {
            "overall_generated_per_sec": last.generated / duration,
            "overall_kept_per_sec": last.kept / duration,
            "overall_given_per_sec": last.given / duration,
            "overall_keep_rate": last.kept / last.generated if last.generated > 0 else 0.0,
            "overall_subsumption_rate": last.subsumed / last.generated if last.generated > 0 else 0.0,
        }

        # Recent window (last 10 iterations)
        window_size = min(10, len(self._snapshots))
        window_start = self._snapshots[-window_size]
        window_duration = last.timestamp - window_start.timestamp

        if window_duration > 0:
            window_gen = last.generated - window_start.generated
            window_kept = last.kept - window_start.kept
            rates["recent_generated_per_sec"] = window_gen / window_duration
            rates["recent_kept_per_sec"] = window_kept / window_duration
            if window_gen > 0:
                rates["recent_keep_rate"] = window_kept / window_gen

        return rates

    def detect_slowdown(self, threshold: float = 0.5) -> list[dict[str, Any]]:
        """Detect iterations where throughput dropped significantly.

        Returns iterations where generated_per_sec dropped below
        `threshold` fraction of the running average.
        """
        if len(self._snapshots) < 3:
            return []

        slowdowns: list[dict[str, Any]] = []
        running_rate = 0.0

        for i in range(1, len(self._snapshots)):
            prev = self._snapshots[i - 1]
            curr = self._snapshots[i]
            dt = curr.timestamp - prev.timestamp

            if dt <= 0:
                continue

            current_rate = curr.delta_generated / dt

            if running_rate > 0 and current_rate < running_rate * threshold:
                slowdowns.append({
                    "iteration": curr.iteration,
                    "rate": current_rate,
                    "avg_rate": running_rate,
                    "ratio": current_rate / running_rate if running_rate > 0 else 0.0,
                    "usable_size": curr.usable_size,
                    "sos_size": curr.sos_size,
                })

            # Update running average (exponential moving average)
            if running_rate == 0.0:
                running_rate = current_rate
            else:
                running_rate = 0.9 * running_rate + 0.1 * current_rate

        return slowdowns

    def report(self) -> str:
        """Generate a search analysis report."""
        if not self._snapshots:
            return "No search iterations recorded."

        lines = [
            "=" * 60,
            "SEARCH ANALYSIS REPORT",
            "=" * 60,
            f"Total iterations: {len(self._snapshots)}",
        ]

        last = self._snapshots[-1]
        lines.append(f"Final stats: given={last.given}, generated={last.generated}, "
                      f"kept={last.kept}, subsumed={last.subsumed}")

        rates = self.compute_rates()
        if rates:
            lines.append("")
            lines.append("Throughput:")
            for key in ["overall_generated_per_sec", "overall_kept_per_sec",
                         "overall_given_per_sec"]:
                if key in rates:
                    label = key.replace("overall_", "").replace("_", " ")
                    lines.append(f"  {label}: {rates[key]:.1f}")

            lines.append("")
            lines.append("Effectiveness:")
            if "overall_keep_rate" in rates:
                lines.append(f"  Keep rate: {rates['overall_keep_rate']:.3f}")
            if "overall_subsumption_rate" in rates:
                lines.append(f"  Subsumption rate: {rates['overall_subsumption_rate']:.3f}")

            # Recent window comparison
            if "recent_generated_per_sec" in rates and "overall_generated_per_sec" in rates:
                ratio = rates["recent_generated_per_sec"] / rates["overall_generated_per_sec"]
                trend = "stable" if 0.8 <= ratio <= 1.2 else "slowing" if ratio < 0.8 else "accelerating"
                lines.append(f"  Recent vs overall throughput: {ratio:.2f}x ({trend})")

        hotspots = self.identify_hotspots(3)
        if hotspots:
            lines.append("")
            lines.append("Highest inference iterations:")
            for h in hotspots:
                lines.append(
                    f"  iter {h['iteration']}: {h['delta_generated']} generated, "
                    f"{h['delta_kept']} kept (usable={h['usable_size']})"
                )

        slowdowns = self.detect_slowdown()
        if slowdowns:
            lines.append("")
            lines.append(f"Slowdown events detected: {len(slowdowns)}")
            for s in slowdowns[:3]:
                lines.append(
                    f"  iter {s['iteration']}: {s['ratio']:.2f}x of avg rate "
                    f"(usable={s['usable_size']}, sos={s['sos_size']})"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_csv_rows(self) -> list[dict[str, Any]]:
        """Export all snapshots as list of dicts for CSV/DataFrame consumption."""
        return [s.to_dict() for s in self._snapshots]
