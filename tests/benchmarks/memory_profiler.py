"""Memory allocation profiler for the PyLADR search engine.

Identifies memory allocation hotspots by running search with tracemalloc
and producing per-file and per-function allocation reports.

Usage:
    from tests.benchmarks.memory_profiler import profile_memory, MemoryReport

    report = profile_memory("tests/fixtures/inputs/bench_ring_comm.in")
    report.print_summary()
    report.print_top_allocations(n=20)

    # Or from command line:
    python -m tests.benchmarks.memory_profiler tests/fixtures/inputs/bench_ring_comm.in
"""

from __future__ import annotations

import tracemalloc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AllocationEntry:
    """A single allocation hotspot."""

    filename: str
    lineno: int
    size_bytes: int
    count: int

    @property
    def size_kb(self) -> float:
        return self.size_bytes / 1024

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


@dataclass(slots=True)
class FileAllocation:
    """Aggregated allocations for a single file."""

    filename: str
    total_bytes: int = 0
    total_count: int = 0

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)


@dataclass
class MemoryReport:
    """Memory profiling report from a search run."""

    problem: str
    peak_memory_bytes: int = 0
    current_memory_bytes: int = 0
    wall_seconds: float = 0.0
    top_allocations: list[AllocationEntry] = field(default_factory=list)
    file_allocations: list[FileAllocation] = field(default_factory=list)
    # Pyladr-specific breakdown
    pyladr_bytes: int = 0
    stdlib_bytes: int = 0
    other_bytes: int = 0

    @property
    def peak_mb(self) -> float:
        return self.peak_memory_bytes / (1024 * 1024)

    @property
    def pyladr_mb(self) -> float:
        return self.pyladr_bytes / (1024 * 1024)

    def print_summary(self) -> None:
        """Print a concise memory summary."""
        print(f"\n{'=' * 70}")
        print(f"MEMORY PROFILE: {self.problem}")
        print(f"{'=' * 70}")
        print(f"Peak memory:    {self.peak_mb:.1f} MB")
        print(f"PyLADR code:    {self.pyladr_mb:.1f} MB "
              f"({self.pyladr_bytes / max(self.peak_memory_bytes, 1) * 100:.0f}%)")
        print(f"Wall time:      {self.wall_seconds:.3f}s")

        if self.file_allocations:
            print(f"\nTop files by allocation:")
            for fa in self.file_allocations[:10]:
                print(f"  {fa.total_mb:>8.2f} MB  {fa.total_count:>8} allocs  {fa.filename}")

        print(f"{'=' * 70}")

    def print_top_allocations(self, n: int = 20) -> None:
        """Print top allocation sites."""
        print(f"\nTop {n} allocation sites:")
        print(f"{'Size':>10}  {'Count':>8}  {'Location'}")
        print("-" * 70)
        for entry in self.top_allocations[:n]:
            if entry.size_bytes >= 1024 * 1024:
                size_str = f"{entry.size_mb:.1f} MB"
            else:
                size_str = f"{entry.size_kb:.1f} KB"
            print(f"{size_str:>10}  {entry.count:>8}  {entry.filename}:{entry.lineno}")

    def as_dict(self) -> dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "problem": self.problem,
            "peak_memory_mb": self.peak_mb,
            "pyladr_memory_mb": self.pyladr_mb,
            "wall_seconds": self.wall_seconds,
            "top_files": [
                {"file": fa.filename, "mb": fa.total_mb, "count": fa.total_count}
                for fa in self.file_allocations[:20]
            ],
            "top_sites": [
                {
                    "file": e.filename,
                    "line": e.lineno,
                    "bytes": e.size_bytes,
                    "count": e.count,
                }
                for e in self.top_allocations[:50]
            ],
        }


def profile_memory(
    input_path: Path | str,
    *,
    timeout: float = 60.0,
    nframes: int = 5,
) -> MemoryReport:
    """Profile memory allocations during a search run.

    Args:
        input_path: Path to .in problem file.
        timeout: Maximum search time in seconds.
        nframes: Stack depth for tracemalloc (higher = more detail, slower).

    Returns:
        MemoryReport with allocation hotspots.
    """
    from tests.benchmarks.perf_profiler import run_problem

    input_path = Path(input_path)
    report = MemoryReport(problem=input_path.stem)

    tracemalloc.start(nframes)

    t0 = time.perf_counter()
    try:
        run_problem(input_path, timeout=timeout, quiet=True)
    except Exception:
        pass  # Still collect memory data
    report.wall_seconds = time.perf_counter() - t0

    # Capture memory state
    current, peak = tracemalloc.get_traced_memory()
    report.current_memory_bytes = current
    report.peak_memory_bytes = peak

    # Take snapshot and analyze
    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Filter out importlib/frozen modules
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))

    # Top allocations by line
    line_stats = snapshot.statistics("lineno")
    for stat in line_stats[:100]:
        entry = AllocationEntry(
            filename=stat.traceback[0].filename,
            lineno=stat.traceback[0].lineno,
            size_bytes=stat.size,
            count=stat.count,
        )
        report.top_allocations.append(entry)

    # Aggregate by file
    file_stats = snapshot.statistics("filename")
    pyladr_total = 0
    stdlib_total = 0
    other_total = 0

    for stat in file_stats:
        fname = stat.traceback[0].filename
        fa = FileAllocation(
            filename=fname,
            total_bytes=stat.size,
            total_count=stat.count,
        )
        report.file_allocations.append(fa)

        if "pyladr" in fname:
            pyladr_total += stat.size
        elif "lib/python" in fname or "site-packages" in fname:
            stdlib_total += stat.size
        else:
            other_total += stat.size

    report.pyladr_bytes = pyladr_total
    report.stdlib_bytes = stdlib_total
    report.other_bytes = other_total

    # Sort files by allocation size
    report.file_allocations.sort(key=lambda f: f.total_bytes, reverse=True)

    return report


def compare_memory_profiles(
    before: MemoryReport,
    after: MemoryReport,
) -> dict[str, Any]:
    """Compare two memory profiles to identify allocation changes.

    Returns a dict summarizing growth/reduction in allocations.
    """
    return {
        "problem": before.problem,
        "peak_before_mb": before.peak_mb,
        "peak_after_mb": after.peak_mb,
        "peak_change_pct": (
            (after.peak_memory_bytes - before.peak_memory_bytes)
            / max(before.peak_memory_bytes, 1) * 100
        ),
        "pyladr_before_mb": before.pyladr_mb,
        "pyladr_after_mb": after.pyladr_mb,
        "pyladr_change_pct": (
            (after.pyladr_bytes - before.pyladr_bytes)
            / max(before.pyladr_bytes, 1) * 100
        ),
    }


def main() -> None:
    """CLI entry point for memory profiling."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="PyLADR memory profiler")
    parser.add_argument("input_file", type=str, help="Path to .in problem file")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--json", type=str, help="Save report to JSON")
    parser.add_argument("--top", type=int, default=20, help="Top N allocation sites")
    args = parser.parse_args()

    report = profile_memory(args.input_file, timeout=args.timeout)
    report.print_summary()
    report.print_top_allocations(n=args.top)

    if args.json:
        Path(args.json).write_text(json.dumps(report.as_dict(), indent=2) + "\n")
        print(f"\nJSON report saved to: {args.json}")


if __name__ == "__main__":
    main()
