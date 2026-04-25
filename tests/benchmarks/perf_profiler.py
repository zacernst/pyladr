"""Fine-grained performance profiler for the PyLADR search engine.

Instruments the given-clause search loop to collect detailed timing and
count data for each phase: selection, inference generation, clause
processing (simplification, subsumption, demodulation), and limbo processing.

Usage:
    from tests.benchmarks.perf_profiler import ProfiledSearch, profile_problem

    # Profile a single problem
    report = profile_problem(input_path, timeout=60.0)
    report.print_summary()

    # Or wrap an existing search
    search = ProfiledSearch(options)
    result = search.run(usable, sos)
    search.profile.print_summary()
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TimingBucket:
    """Accumulates wall-clock time and call count for a named operation."""

    name: str
    total_seconds: float = 0.0
    call_count: int = 0
    max_seconds: float = 0.0
    min_seconds: float = float("inf")

    def record(self, elapsed: float) -> None:
        self.total_seconds += elapsed
        self.call_count += 1
        if elapsed > self.max_seconds:
            self.max_seconds = elapsed
        if elapsed < self.min_seconds:
            self.min_seconds = elapsed

    @property
    def avg_seconds(self) -> float:
        return self.total_seconds / self.call_count if self.call_count else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_s": self.total_seconds,
            "calls": self.call_count,
            "avg_us": self.avg_seconds * 1e6,
            "max_us": self.max_seconds * 1e6,
            "min_us": self.min_seconds * 1e6 if self.min_seconds != float("inf") else 0.0,
        }


@dataclass(slots=True)
class ProfileData:
    """Collected profiling data from one search run."""

    # Phase timings
    buckets: dict[str, TimingBucket] = field(default_factory=dict)

    # Per-given-clause breakdown
    given_clause_times: list[float] = field(default_factory=list)
    inference_counts_per_given: list[int] = field(default_factory=list)

    # Clause size distribution
    generated_weights: list[float] = field(default_factory=list)
    kept_weights: list[float] = field(default_factory=list)

    # Search progress snapshots (sampled every N given clauses)
    progress_snapshots: list[dict[str, Any]] = field(default_factory=list)

    # Overall wall clock
    total_wall_seconds: float = 0.0

    def bucket(self, name: str) -> TimingBucket:
        if name not in self.buckets:
            self.buckets[name] = TimingBucket(name=name)
        return self.buckets[name]

    def print_summary(self) -> None:
        """Print a human-readable profile summary."""
        print("\n" + "=" * 78)
        print("PERFORMANCE PROFILE SUMMARY")
        print("=" * 78)
        print(f"Total wall time: {self.total_wall_seconds:.3f}s")
        print(f"Given clauses processed: {len(self.given_clause_times)}")
        print()

        # Sort buckets by total time descending
        sorted_buckets = sorted(
            self.buckets.values(), key=lambda b: b.total_seconds, reverse=True,
        )

        print(
            f"{'Phase':<30} {'Total(s)':>10} {'Calls':>10} "
            f"{'Avg(us)':>10} {'Max(us)':>10} {'%Total':>8}"
        )
        print("-" * 88)
        for b in sorted_buckets:
            pct = (b.total_seconds / self.total_wall_seconds * 100) if self.total_wall_seconds > 0 else 0
            print(
                f"{b.name:<30} {b.total_seconds:>10.4f} {b.call_count:>10} "
                f"{b.avg_seconds * 1e6:>10.1f} {b.max_seconds * 1e6:>10.1f} "
                f"{pct:>7.1f}%"
            )

        # Given clause time distribution
        if self.given_clause_times:
            import statistics
            times = self.given_clause_times
            print(f"\nPer-given-clause time (ms):")
            print(f"  median={statistics.median(times)*1000:.2f}  "
                  f"mean={statistics.mean(times)*1000:.2f}  "
                  f"p95={sorted(times)[int(len(times)*0.95)]*1000:.2f}  "
                  f"max={max(times)*1000:.2f}")

        # Clause weight distribution
        if self.kept_weights:
            import statistics
            print(f"\nKept clause weights:")
            print(f"  median={statistics.median(self.kept_weights):.1f}  "
                  f"mean={statistics.mean(self.kept_weights):.1f}  "
                  f"max={max(self.kept_weights):.1f}")

        print("=" * 78)

    def as_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON export."""
        return {
            "total_wall_seconds": self.total_wall_seconds,
            "given_clause_count": len(self.given_clause_times),
            "phases": {name: b.as_dict() for name, b in self.buckets.items()},
            "progress_snapshots": self.progress_snapshots,
        }


class _Timer:
    """Context manager for timing a named operation into a ProfileData."""

    __slots__ = ("_profile", "_name", "_start")

    def __init__(self, profile: ProfileData, name: str) -> None:
        self._profile = profile
        self._name = name
        self._start = 0.0

    def __enter__(self) -> _Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        elapsed = time.perf_counter() - self._start
        self._profile.bucket(self._name).record(elapsed)


class ProfiledSearch:
    """Wraps GivenClauseSearch with profiling via subclassing.

    Creates a dynamic subclass that overrides key methods to collect
    timing data, since GivenClauseSearch uses __slots__.
    """

    def __init__(
        self,
        options: Any = None,
        selection: Any = None,
        symbol_table: Any = None,
        snapshot_interval: int = 50,
    ) -> None:
        from pyladr.search.given_clause import ExitCode, GivenClauseSearch

        profile = ProfileData()
        interval = snapshot_interval

        class _Instrumented(GivenClauseSearch):
            """Subclass with profiling instrumentation."""

            # No extra __slots__ needed — we store profile data externally

            def _make_inferences(self) -> ExitCode | None:
                t0 = time.perf_counter()
                result = super()._make_inferences()
                elapsed = time.perf_counter() - t0
                profile.bucket("make_inferences").record(elapsed)
                profile.given_clause_times.append(elapsed)

                # Progress snapshot
                stats = self._state.stats
                if stats.given > 0 and stats.given % interval == 0:
                    profile.progress_snapshots.append({
                        "given": stats.given,
                        "generated": stats.generated,
                        "kept": stats.kept,
                        "subsumed": stats.subsumed,
                        "back_subsumed": stats.back_subsumed,
                        "sos_size": self._state.sos.length,
                        "usable_size": self._state.usable.length,
                        "wall_seconds": stats.elapsed_seconds(),
                        "given_time_ms": elapsed * 1000,
                    })
                return result

            def _cl_process(self, c):
                t0 = time.perf_counter()
                result = super()._cl_process(c)
                profile.bucket("cl_process").record(time.perf_counter() - t0)
                return result

            def _simplify(self, c):
                t0 = time.perf_counter()
                result = super()._simplify(c)
                profile.bucket("simplify").record(time.perf_counter() - t0)
                return result

            def _should_delete(self, c) -> bool:
                t0 = time.perf_counter()
                result = super()._should_delete(c)
                profile.bucket("delete_checks").record(time.perf_counter() - t0)
                return result

            def _forward_subsumed(self, c) -> bool:
                t0 = time.perf_counter()
                result = super()._forward_subsumed(c)
                profile.bucket("forward_subsumption").record(time.perf_counter() - t0)
                return result

            def _keep_clause(self, c):
                if hasattr(c, 'weight'):
                    profile.kept_weights.append(c.weight)
                t0 = time.perf_counter()
                result = super()._keep_clause(c)
                profile.bucket("keep_clause").record(time.perf_counter() - t0)
                return result

            def _limbo_process(self):
                t0 = time.perf_counter()
                result = super()._limbo_process()
                profile.bucket("limbo_process").record(time.perf_counter() - t0)
                return result

            def _given_infer(self, given):
                t0 = time.perf_counter()
                result = super()._given_infer(given)
                profile.bucket("inference_gen").record(time.perf_counter() - t0)
                return result

        self._search = _Instrumented(
            options=options,
            selection=selection,
            symbol_table=symbol_table,
        )
        self.profile = profile

    def run(
        self,
        usable: list[Any] | None = None,
        sos: list[Any] | None = None,
    ) -> Any:
        """Run the profiled search."""
        t0 = time.perf_counter()
        result = self._search.run(usable=usable, sos=sos)
        self.profile.total_wall_seconds = time.perf_counter() - t0
        return result

    @property
    def state(self):
        return self._search.state

    @property
    def stats(self):
        return self._search.stats


def parse_and_setup(
    input_path: Path | str,
    *,
    timeout: float = 60.0,
    quiet: bool = True,
) -> tuple[Any, Any, list[Any], list[Any]]:
    """Parse an input file and return (options, symbol_table, usable, sos).

    Replicates the setup logic from pyladr.apps.prover9.run_prover() without
    running the search, so we can wrap it with profiling.
    """
    from pyladr.apps.prover9 import _auto_inference, _deny_goals
    from pyladr.core.symbol import SymbolTable
    from pyladr.parsing.ladr_parser import LADRParser
    from pyladr.search.given_clause import SearchOptions

    input_path = Path(input_path)
    input_text = input_path.read_text()

    symbol_table = SymbolTable()
    parser = LADRParser(symbol_table)
    parsed = parser.parse_input(input_text)

    usable, sos, _denied = _deny_goals(parsed, symbol_table)

    opts = SearchOptions(
        quiet=quiet,
        print_given=not quiet,
    )

    # Apply auto-inference settings (matches run_prover logic)
    _auto_inference(parsed, opts)

    # Override quiet/timeout after auto-inference
    opts.quiet = quiet
    opts.print_given = not quiet
    if timeout > 0:
        opts.max_seconds = timeout

    return opts, symbol_table, usable, sos


def profile_problem(
    input_path: Path | str,
    *,
    timeout: float = 60.0,
    quiet: bool = True,
    snapshot_interval: int = 50,
) -> tuple[Any, ProfileData]:
    """Profile a complete problem run from an input file.

    Returns (SearchResult, ProfileData).
    """
    opts, symbol_table, usable, sos = parse_and_setup(
        input_path, timeout=timeout, quiet=quiet,
    )

    search = ProfiledSearch(
        options=opts,
        symbol_table=symbol_table,
        snapshot_interval=snapshot_interval,
    )

    result = search.run(usable=usable, sos=sos)
    return result, search.profile


def run_problem(
    input_path: Path | str,
    *,
    timeout: float = 60.0,
    quiet: bool = True,
) -> Any:
    """Run a problem without profiling. Returns SearchResult."""
    from pyladr.search.given_clause import GivenClauseSearch

    opts, symbol_table, usable, sos = parse_and_setup(
        input_path, timeout=timeout, quiet=quiet,
    )
    engine = GivenClauseSearch(options=opts, symbol_table=symbol_table)
    return engine.run(usable=usable, sos=sos)
