"""Performance profiling for the given-clause search engine.

Provides phase-level timing and per-iteration profiling without
modifying the search algorithm. Uses a wrapper approach that
observes SearchState and SearchStatistics at key points.

Usage:
    profiler = SearchProfiler()
    profiler.start()

    # ... run search iterations ...
    profiler.begin_phase("selection")
    # ... select given clause ...
    profiler.end_phase("selection")

    profiler.begin_phase("inference")
    # ... generate inferences ...
    profiler.end_phase("inference")

    report = profiler.report()
    print(report.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PhaseTimer:
    """Accumulating timer for a named search phase.

    Tracks total time, call count, and min/max/avg per call.
    Thread-safe for reading; single-writer assumed (main search thread).
    """

    name: str
    total_seconds: float = 0.0
    call_count: int = 0
    min_seconds: float = float("inf")
    max_seconds: float = 0.0
    _start: float = 0.0

    def begin(self) -> None:
        """Start timing this phase."""
        self._start = time.monotonic()

    def end(self) -> float:
        """End timing this phase. Returns elapsed seconds."""
        elapsed = time.monotonic() - self._start
        self.total_seconds += elapsed
        self.call_count += 1
        if elapsed < self.min_seconds:
            self.min_seconds = elapsed
        if elapsed > self.max_seconds:
            self.max_seconds = elapsed
        self._start = 0.0
        return elapsed

    @property
    def avg_seconds(self) -> float:
        """Average seconds per call."""
        if self.call_count == 0:
            return 0.0
        return self.total_seconds / self.call_count

    def summary(self) -> str:
        """One-line summary of this phase timer."""
        if self.call_count == 0:
            return f"{self.name}: no calls"
        return (
            f"{self.name}: {self.total_seconds:.4f}s total, "
            f"{self.call_count} calls, "
            f"avg={self.avg_seconds:.6f}s, "
            f"min={self.min_seconds:.6f}s, "
            f"max={self.max_seconds:.6f}s"
        )


@dataclass(slots=True)
class ProfileReport:
    """Aggregated profiling report from a search run."""

    total_seconds: float = 0.0
    phases: dict[str, PhaseTimer] = field(default_factory=dict)
    iteration_count: int = 0
    custom_counters: dict[str, int] = field(default_factory=dict)
    custom_timings: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Multi-line summary of the profiling report."""
        lines = [
            "=" * 60,
            "SEARCH PROFILING REPORT",
            "=" * 60,
            f"Total time: {self.total_seconds:.4f}s",
            f"Iterations: {self.iteration_count}",
            "",
            "Phase breakdown:",
        ]

        # Sort phases by total time descending
        sorted_phases = sorted(
            self.phases.values(),
            key=lambda p: p.total_seconds,
            reverse=True,
        )

        for phase in sorted_phases:
            pct = (
                (phase.total_seconds / self.total_seconds * 100)
                if self.total_seconds > 0
                else 0.0
            )
            lines.append(f"  {phase.summary()} ({pct:.1f}%)")

        # Account for untracked time
        tracked = sum(p.total_seconds for p in self.phases.values())
        untracked = self.total_seconds - tracked
        if self.total_seconds > 0 and untracked > 0.001:
            pct = untracked / self.total_seconds * 100
            lines.append(f"  (untracked): {untracked:.4f}s ({pct:.1f}%)")

        if self.custom_counters:
            lines.append("")
            lines.append("Counters:")
            for name, value in sorted(self.custom_counters.items()):
                lines.append(f"  {name}: {value}")

        if self.custom_timings:
            lines.append("")
            lines.append("Custom timings:")
            for name, value in sorted(self.custom_timings.items()):
                lines.append(f"  {name}: {value:.4f}s")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export report as a dictionary for JSON serialization."""
        return {
            "total_seconds": self.total_seconds,
            "iteration_count": self.iteration_count,
            "phases": {
                name: {
                    "total_seconds": p.total_seconds,
                    "call_count": p.call_count,
                    "avg_seconds": p.avg_seconds,
                    "min_seconds": p.min_seconds if p.call_count > 0 else 0.0,
                    "max_seconds": p.max_seconds,
                }
                for name, p in self.phases.items()
            },
            "custom_counters": dict(self.custom_counters),
            "custom_timings": dict(self.custom_timings),
        }


class SearchProfiler:
    """Non-intrusive profiler for the given-clause search engine.

    Tracks timing of named phases within the search loop. Does not
    modify any search state or behavior.

    Usage:
        profiler = SearchProfiler()
        profiler.start()

        with profiler.phase("selection"):
            given = select_given(sos)

        with profiler.phase("inference"):
            inferences = generate_inferences(given, usable)

        report = profiler.report()
    """

    __slots__ = ("_phases", "_start_time", "_iteration_count",
                 "_custom_counters", "_custom_timings", "_enabled")

    def __init__(self, enabled: bool = True) -> None:
        self._phases: dict[str, PhaseTimer] = {}
        self._start_time: float = 0.0
        self._iteration_count: int = 0
        self._custom_counters: dict[str, int] = {}
        self._custom_timings: dict[str, float] = {}
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        """Record profiling start time."""
        self._start_time = time.monotonic()

    def stop(self) -> float:
        """Record profiling stop time. Returns total elapsed seconds."""
        return time.monotonic() - self._start_time

    def _get_phase(self, name: str) -> PhaseTimer:
        """Get or create a phase timer."""
        if name not in self._phases:
            self._phases[name] = PhaseTimer(name=name)
        return self._phases[name]

    def begin_phase(self, name: str) -> None:
        """Start timing a named phase."""
        if not self._enabled:
            return
        self._get_phase(name).begin()

    def end_phase(self, name: str) -> float:
        """End timing a named phase. Returns elapsed seconds."""
        if not self._enabled:
            return 0.0
        return self._get_phase(name).end()

    def phase(self, name: str) -> _PhaseContext:
        """Context manager for timing a named phase.

        Usage:
            with profiler.phase("inference"):
                inferences = generate(given, usable)
        """
        return _PhaseContext(self, name)

    def increment_iteration(self) -> None:
        """Record completion of one search iteration."""
        self._iteration_count += 1

    def increment_counter(self, name: str, amount: int = 1) -> None:
        """Increment a custom counter."""
        if not self._enabled:
            return
        self._custom_counters[name] = self._custom_counters.get(name, 0) + amount

    def record_timing(self, name: str, seconds: float) -> None:
        """Record a custom timing measurement."""
        if not self._enabled:
            return
        self._custom_timings[name] = self._custom_timings.get(name, 0.0) + seconds

    def report(self) -> ProfileReport:
        """Generate a profiling report."""
        total = time.monotonic() - self._start_time if self._start_time > 0 else 0.0
        return ProfileReport(
            total_seconds=total,
            phases=dict(self._phases),
            iteration_count=self._iteration_count,
            custom_counters=dict(self._custom_counters),
            custom_timings=dict(self._custom_timings),
        )

    def reset(self) -> None:
        """Reset all profiling data."""
        self._phases.clear()
        self._start_time = 0.0
        self._iteration_count = 0
        self._custom_counters.clear()
        self._custom_timings.clear()


class _PhaseContext:
    """Context manager for phase timing."""

    __slots__ = ("_profiler", "_name")

    def __init__(self, profiler: SearchProfiler, name: str) -> None:
        self._profiler = profiler
        self._name = name

    def __enter__(self) -> _PhaseContext:
        self._profiler.begin_phase(self._name)
        return self

    def __exit__(self, *exc: object) -> None:
        self._profiler.end_phase(self._name)
