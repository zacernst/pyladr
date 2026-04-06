"""Performance benchmarking harness for pyladr vs C Prover9.

Runs benchmark problems against both the C reference implementation and
the Python implementation, collecting timing and search statistics for
comparison.

Usage:
    python -m tests.benchmarks.bench_harness [--c-only] [--py-only] [--problem NAME]
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
C_PROVER9_BIN = PROJECT_ROOT / "bin" / "prover9"
INPUTS_DIR = PROJECT_ROOT / "tests" / "fixtures" / "inputs"
RESULTS_DIR = PROJECT_ROOT / "tests" / "benchmarks" / "results"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    problem: str
    implementation: str  # "c" or "python"
    theorem_proved: bool = False
    search_failed: bool = False
    user_cpu_seconds: float = 0.0
    wall_clock_seconds: float = 0.0
    clauses_given: int = 0
    clauses_generated: int = 0
    clauses_kept: int = 0
    proof_length: int = 0
    megabytes: float = 0.0
    demod_attempts: int = 0
    demod_rewrites: int = 0
    error: str | None = None


@dataclass
class BenchmarkComparison:
    """Comparison of C vs Python results on a single problem."""

    problem: str
    c_result: BenchmarkResult
    py_result: BenchmarkResult | None = None
    speedup_ratio: float | None = None  # C_time / Py_time (>1 means Py is faster)
    slowdown_ratio: float | None = None  # Py_time / C_time (target: 2-10x)
    search_equivalent: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results across multiple problems."""

    timestamp: str = ""
    python_version: str = ""
    platform: str = ""
    results: list[BenchmarkComparison] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def compute_summary(self) -> None:
        """Compute aggregate statistics."""
        c_times = []
        py_times = []
        slowdowns = []
        search_match = 0
        total = len(self.results)

        for comp in self.results:
            if comp.c_result.theorem_proved:
                c_times.append(comp.c_result.user_cpu_seconds)
            if comp.py_result and comp.py_result.theorem_proved:
                py_times.append(comp.py_result.user_cpu_seconds)
            if comp.slowdown_ratio is not None:
                slowdowns.append(comp.slowdown_ratio)
            if comp.search_equivalent:
                search_match += 1

        self.summary = {
            "total_problems": total,
            "c_solved": len(c_times),
            "py_solved": len(py_times),
            "search_equivalent": search_match,
            "c_total_cpu": sum(c_times),
            "py_total_cpu": sum(py_times) if py_times else None,
            "median_slowdown": statistics.median(slowdowns) if slowdowns else None,
            "max_slowdown": max(slowdowns) if slowdowns else None,
            "min_slowdown": min(slowdowns) if slowdowns else None,
        }


def run_c_benchmark(input_file: Path, *, runs: int = 3) -> BenchmarkResult:
    """Run a benchmark against the C Prover9 binary.

    Runs the problem multiple times and takes the median CPU time.
    """
    if not C_PROVER9_BIN.exists():
        return BenchmarkResult(
            problem=input_file.stem,
            implementation="c",
            error=f"C binary not found at {C_PROVER9_BIN}",
        )

    cpu_times: list[float] = []
    wall_times: list[float] = []
    last_result: BenchmarkResult | None = None

    for _ in range(runs):
        wall_start = time.perf_counter()
        try:
            proc = subprocess.run(
                [str(C_PROVER9_BIN), "-f", str(input_file)],
                capture_output=True,
                text=True,
                timeout=120.0,
            )
        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                problem=input_file.stem,
                implementation="c",
                error="Timeout after 120s",
            )
        wall_elapsed = time.perf_counter() - wall_start

        result = _parse_c_output(input_file.stem, proc.stdout + proc.stderr)
        cpu_times.append(result.user_cpu_seconds)
        wall_times.append(wall_elapsed)
        last_result = result

    if last_result is None:
        return BenchmarkResult(
            problem=input_file.stem, implementation="c", error="No results"
        )

    # Use median timing from multiple runs
    last_result.user_cpu_seconds = statistics.median(cpu_times)
    last_result.wall_clock_seconds = statistics.median(wall_times)
    return last_result


def _parse_c_output(problem_name: str, raw: str) -> BenchmarkResult:
    """Parse C Prover9 output into a BenchmarkResult."""
    import re

    result = BenchmarkResult(problem=problem_name, implementation="c")
    result.theorem_proved = "THEOREM PROVED" in raw
    result.search_failed = "SEARCH FAILED" in raw

    # Parse statistics line: Given=12. Generated=118. Kept=23. proofs=1.
    m = re.search(r"Given=(\d+)\.\s+Generated=(\d+)\.\s+Kept=(\d+)", raw)
    if m:
        result.clauses_given = int(m.group(1))
        result.clauses_generated = int(m.group(2))
        result.clauses_kept = int(m.group(3))

    m = re.search(r"Length of proof is (\d+)", raw)
    if m:
        result.proof_length = int(m.group(1))

    m = re.search(r"Megabytes=(\d+\.\d+)", raw)
    if m:
        result.megabytes = float(m.group(1))

    m = re.search(r"User_CPU=([0-9.]+)", raw)
    if m:
        result.user_cpu_seconds = float(m.group(1))

    m = re.search(r"Demod_attempts=(\d+)\.\s+Demod_rewrites=(\d+)", raw)
    if m:
        result.demod_attempts = int(m.group(1))
        result.demod_rewrites = int(m.group(2))

    return result


def run_python_benchmark(input_file: Path, *, runs: int = 3) -> BenchmarkResult:
    """Run a benchmark against the Python pyladr implementation.

    Placeholder: will be implemented once the Python prover is functional.
    Returns a result with error indicating not yet implemented.
    """
    # TODO: Implement once pyladr search engine is functional
    # This will invoke pyladr.search or the pyprover9 CLI
    return BenchmarkResult(
        problem=input_file.stem,
        implementation="python",
        error="Python implementation not yet available",
    )


def compare_results(
    c_result: BenchmarkResult,
    py_result: BenchmarkResult | None,
) -> BenchmarkComparison:
    """Compare C and Python benchmark results."""
    comp = BenchmarkComparison(
        problem=c_result.problem,
        c_result=c_result,
        py_result=py_result,
    )

    if py_result is None or py_result.error:
        comp.notes.append("Python result unavailable")
        return comp

    # Compute timing ratios
    if c_result.user_cpu_seconds > 0 and py_result.user_cpu_seconds > 0:
        comp.slowdown_ratio = py_result.user_cpu_seconds / c_result.user_cpu_seconds
        comp.speedup_ratio = c_result.user_cpu_seconds / py_result.user_cpu_seconds
    elif c_result.user_cpu_seconds == 0 and py_result.user_cpu_seconds == 0:
        comp.slowdown_ratio = 1.0
        comp.speedup_ratio = 1.0
        comp.notes.append("Both completed in <0.01s, ratio meaningless")

    # Check search equivalence
    comp.search_equivalent = (
        c_result.theorem_proved == py_result.theorem_proved
        and c_result.clauses_given == py_result.clauses_given
        and c_result.clauses_generated == py_result.clauses_generated
        and c_result.clauses_kept == py_result.clauses_kept
    )

    if not comp.search_equivalent:
        if c_result.theorem_proved != py_result.theorem_proved:
            comp.notes.append("CRITICAL: Different theorem status!")
        if c_result.clauses_given != py_result.clauses_given:
            comp.notes.append(
                f"Given clauses differ: C={c_result.clauses_given} "
                f"Py={py_result.clauses_given}"
            )

    # Assess performance target (2-10x acceptable, 2-4x with parallelism)
    if comp.slowdown_ratio is not None:
        if comp.slowdown_ratio <= 2.0:
            comp.notes.append("EXCELLENT: Within 2x of C")
        elif comp.slowdown_ratio <= 4.0:
            comp.notes.append("GOOD: Within 4x of C (parallelizable to parity)")
        elif comp.slowdown_ratio <= 10.0:
            comp.notes.append("ACCEPTABLE: Within 10x of C")
        else:
            comp.notes.append(f"SLOW: {comp.slowdown_ratio:.1f}x slower than C")

    return comp


def discover_benchmark_problems() -> list[Path]:
    """Find all benchmark input files."""
    problems = sorted(INPUTS_DIR.glob("bench_*.in"))
    # Also include the standard x2 problem
    x2 = PROJECT_ROOT / "prover9.examples" / "x2.in"
    if x2.exists():
        problems.insert(0, x2)
    return problems


def run_benchmark_suite(
    *,
    c_only: bool = False,
    py_only: bool = False,
    problem_filter: str | None = None,
    runs: int = 3,
) -> BenchmarkSuite:
    """Run the full benchmark suite."""
    import datetime
    import platform

    suite = BenchmarkSuite(
        timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        python_version=sys.version,
        platform=platform.platform(),
    )

    problems = discover_benchmark_problems()
    if problem_filter:
        problems = [p for p in problems if problem_filter in p.stem]

    for input_file in problems:
        print(f"  Benchmarking: {input_file.stem}...", flush=True)

        c_result = None
        py_result = None

        if not py_only:
            c_result = run_c_benchmark(input_file, runs=runs)
            print(
                f"    C: {'PROVED' if c_result.theorem_proved else 'FAILED'} "
                f"in {c_result.user_cpu_seconds:.3f}s CPU "
                f"(given={c_result.clauses_given}, "
                f"gen={c_result.clauses_generated}, "
                f"kept={c_result.clauses_kept})"
            )

        if not c_only:
            py_result = run_python_benchmark(input_file, runs=runs)
            if py_result.error:
                print(f"    Python: {py_result.error}")
            else:
                print(
                    f"    Python: {'PROVED' if py_result.theorem_proved else 'FAILED'} "
                    f"in {py_result.user_cpu_seconds:.3f}s CPU"
                )

        if c_result:
            comp = compare_results(c_result, py_result)
            suite.results.append(comp)

    suite.compute_summary()
    return suite


def print_suite_report(suite: BenchmarkSuite) -> None:
    """Print a formatted benchmark report."""
    print("\n" + "=" * 78)
    print("PERFORMANCE BENCHMARK REPORT")
    print("=" * 78)
    print(f"Timestamp: {suite.timestamp}")
    print(f"Python:    {suite.python_version}")
    print(f"Platform:  {suite.platform}")
    print()

    # Table header
    print(
        f"{'Problem':<25} {'Status':>8} {'C CPU':>8} {'Py CPU':>8} "
        f"{'Ratio':>8} {'Given':>7} {'Gen':>10} {'Kept':>7}"
    )
    print("-" * 92)

    for comp in suite.results:
        c = comp.c_result
        status_str = "PROVED" if c.theorem_proved else "FAILED"
        c_cpu = f"{c.user_cpu_seconds:.3f}s"
        py_cpu = "N/A"
        ratio = "N/A"

        if comp.py_result and not comp.py_result.error:
            py_cpu = f"{comp.py_result.user_cpu_seconds:.3f}s"
        if comp.slowdown_ratio is not None:
            ratio = f"{comp.slowdown_ratio:.1f}x"

        print(
            f"{c.problem:<25} {status_str:>8} {c_cpu:>8} {py_cpu:>8} "
            f"{ratio:>8} {c.clauses_given:>7} {c.clauses_generated:>10} "
            f"{c.clauses_kept:>7}"
        )

        for note in comp.notes:
            print(f"  -> {note}")

    # Summary
    s = suite.summary
    print()
    print(f"Problems: {s['total_problems']} total, {s['c_solved']} C-solved", end="")
    if s.get("py_solved") is not None:
        print(f", {s['py_solved']} Py-solved", end="")
    print()

    if s.get("median_slowdown"):
        print(
            f"Slowdown: median={s['median_slowdown']:.1f}x, "
            f"min={s['min_slowdown']:.1f}x, max={s['max_slowdown']:.1f}x"
        )
    print(f"C total CPU: {s['c_total_cpu']:.3f}s")
    if s.get("py_total_cpu") is not None:
        print(f"Python total CPU: {s['py_total_cpu']:.3f}s")

    # Performance targets
    print()
    print("Performance targets:")
    print("  - Single-thread: 2-10x slowdown vs C is acceptable")
    print("  - With free-threading (4 cores): target 2-4x speedup")
    print("  - Net: should approach C parity on parallelizable problems")
    print("=" * 78)


def save_results(suite: BenchmarkSuite, output_path: Path | None = None) -> Path:
    """Save benchmark results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = RESULTS_DIR / f"bench_{suite.timestamp.replace(':', '-')}.json"

    data = {
        "timestamp": suite.timestamp,
        "python_version": suite.python_version,
        "platform": suite.platform,
        "summary": suite.summary,
        "results": [
            {
                "problem": comp.problem,
                "c_result": asdict(comp.c_result),
                "py_result": asdict(comp.py_result) if comp.py_result else None,
                "slowdown_ratio": comp.slowdown_ratio,
                "speedup_ratio": comp.speedup_ratio,
                "search_equivalent": comp.search_equivalent,
                "notes": comp.notes,
            }
            for comp in suite.results
        ],
    }

    output_path.write_text(json.dumps(data, indent=2) + "\n")
    return output_path


def main() -> None:
    """CLI entry point for benchmark harness."""
    import argparse

    parser = argparse.ArgumentParser(description="pyladr performance benchmarks")
    parser.add_argument("--c-only", action="store_true", help="Only run C benchmarks")
    parser.add_argument(
        "--py-only", action="store_true", help="Only run Python benchmarks"
    )
    parser.add_argument("--problem", type=str, help="Filter to specific problem name")
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of runs per problem (default: 3)"
    )
    parser.add_argument(
        "--save", action="store_true", help="Save results to JSON file"
    )
    args = parser.parse_args()

    print("Running pyladr benchmark suite...")
    suite = run_benchmark_suite(
        c_only=args.c_only,
        py_only=args.py_only,
        problem_filter=args.problem,
        runs=args.runs,
    )

    print_suite_report(suite)

    if args.save:
        path = save_results(suite)
        print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    main()
