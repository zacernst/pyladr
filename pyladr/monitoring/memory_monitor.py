"""Memory usage monitoring for PyLADR search engine.

Tracks memory consumption of clause lists, indexes, and overall
process memory at configurable intervals during search.

Non-intrusive: reads sizes from existing data structures without
modifying any search state.

Usage:
    monitor = MemoryMonitor()
    monitor.snapshot(state)  # Take a memory snapshot from SearchState
    print(monitor.report())
"""

from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyladr.search.state import SearchState


@dataclass(slots=True)
class MemorySnapshot:
    """Memory usage at a point in time during search."""

    timestamp: float = 0.0
    iteration: int = 0

    # Clause list sizes (count of clauses)
    usable_count: int = 0
    sos_count: int = 0
    limbo_count: int = 0
    disabled_count: int = 0
    demod_count: int = 0

    # Process-level memory (bytes)
    process_rss_bytes: int = 0

    # Estimated object counts
    total_clauses_tracked: int = 0
    clause_ids_assigned: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return {
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            "usable_count": self.usable_count,
            "sos_count": self.sos_count,
            "limbo_count": self.limbo_count,
            "disabled_count": self.disabled_count,
            "demod_count": self.demod_count,
            "process_rss_bytes": self.process_rss_bytes,
            "total_clauses_tracked": self.total_clauses_tracked,
            "clause_ids_assigned": self.clause_ids_assigned,
        }

    @property
    def process_rss_mb(self) -> float:
        """Process RSS in megabytes."""
        return self.process_rss_bytes / (1024 * 1024)

    @property
    def total_active_clauses(self) -> int:
        """Total clauses in active lists (usable + sos + limbo)."""
        return self.usable_count + self.sos_count + self.limbo_count


def _get_process_rss() -> int:
    """Get current process RSS in bytes. Returns 0 if unavailable."""
    try:
        import resource
        # ru_maxrss is in bytes on Linux, kilobytes on macOS
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = usage.ru_maxrss
        if sys.platform == "darwin":
            return rss  # Already in bytes on macOS
        return rss * 1024  # Convert KB to bytes on Linux
    except (ImportError, OSError):
        return 0


class MemoryMonitor:
    """Non-intrusive memory usage monitor for the search engine.

    Takes periodic snapshots of clause list sizes and process memory
    without modifying any search state.

    Usage:
        monitor = MemoryMonitor(interval=10)
        # In the search loop:
        if monitor.should_snapshot(iteration):
            monitor.snapshot(state, iteration)
        # After search:
        print(monitor.report())
    """

    __slots__ = ("_snapshots", "_interval", "_start_time", "_peak_rss")

    def __init__(self, interval: int = 10, max_snapshots: int = 1000) -> None:
        """Initialize memory monitor.

        Args:
            interval: Take a snapshot every N iterations (0 = manual only).
            max_snapshots: Maximum snapshots retained (bounded deque).
        """
        self._snapshots: deque[MemorySnapshot] = deque(maxlen=max_snapshots)
        self._interval = interval
        self._start_time = time.monotonic()
        self._peak_rss: int = 0

    @property
    def snapshots(self) -> list[MemorySnapshot]:
        """All recorded snapshots."""
        return list(self._snapshots)

    @property
    def peak_rss_bytes(self) -> int:
        """Peak RSS observed across all snapshots."""
        return self._peak_rss

    @property
    def peak_rss_mb(self) -> float:
        """Peak RSS in megabytes."""
        return self._peak_rss / (1024 * 1024)

    def should_snapshot(self, iteration: int) -> bool:
        """Check if a snapshot should be taken at this iteration."""
        if self._interval <= 0:
            return False
        return iteration % self._interval == 0

    def snapshot(self, state: SearchState, iteration: int = 0) -> MemorySnapshot:
        """Take a memory snapshot from the current search state.

        Non-intrusive: reads only from public properties of SearchState.
        """
        rss = _get_process_rss()
        if rss > self._peak_rss:
            self._peak_rss = rss

        snap = MemorySnapshot(
            timestamp=time.monotonic() - self._start_time,
            iteration=iteration,
            usable_count=state.usable.length,
            sos_count=state.sos.length,
            limbo_count=state.limbo.length,
            disabled_count=state.disabled.length,
            demod_count=state.demods.length,
            process_rss_bytes=rss,
            total_clauses_tracked=(
                state.usable.length + state.sos.length
                + state.limbo.length + state.disabled.length
            ),
            clause_ids_assigned=state.clause_ids_assigned(),
        )
        self._snapshots.append(snap)
        return snap

    def report(self) -> str:
        """Generate a memory usage report."""
        if not self._snapshots:
            return "No memory snapshots recorded."

        lines = [
            "=" * 60,
            "MEMORY USAGE REPORT",
            "=" * 60,
            f"Snapshots: {len(self._snapshots)}",
            f"Peak RSS: {self.peak_rss_mb:.1f} MB",
        ]

        # Show first, peak clause count, and last snapshots
        first = self._snapshots[0]
        last = self._snapshots[-1]
        peak_clauses = max(self._snapshots, key=lambda s: s.total_active_clauses)

        lines.append("")
        lines.append("Clause list evolution:")
        lines.append(
            f"  Start (iter {first.iteration}): "
            f"usable={first.usable_count}, sos={first.sos_count}, "
            f"disabled={first.disabled_count}"
        )
        lines.append(
            f"  Peak active (iter {peak_clauses.iteration}): "
            f"usable={peak_clauses.usable_count}, sos={peak_clauses.sos_count}, "
            f"total_active={peak_clauses.total_active_clauses}"
        )
        lines.append(
            f"  End (iter {last.iteration}): "
            f"usable={last.usable_count}, sos={last.sos_count}, "
            f"disabled={last.disabled_count}, "
            f"ids_assigned={last.clause_ids_assigned}"
        )

        # Growth rates
        if len(self._snapshots) >= 2:
            duration = last.timestamp - first.timestamp
            if duration > 0:
                clause_growth_rate = (
                    (last.clause_ids_assigned - first.clause_ids_assigned)
                    / duration
                )
                lines.append("")
                lines.append(f"Clause ID growth rate: {clause_growth_rate:.1f} IDs/sec")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_csv_rows(self) -> list[dict[str, Any]]:
        """Export snapshots as list of dicts for CSV/DataFrame consumption."""
        return [s.to_dict() for s in self._snapshots]
