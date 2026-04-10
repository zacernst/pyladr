"""Before/after performance comparison framework.

Runs benchmark problems multiple times, computes statistics, and reports
whether an optimization delivers statistically significant improvement.

Usage:
    from tests.benchmarks.perf_compare import compare_runs, ComparisonResult

    # Compare two sets of timing results
    result = compare_runs(baseline_times, optimized_times)
    print(result.summary())

    # Or use the full benchmark comparison
    from tests.benchmarks.perf_compare import BenchmarkComparator
    comp = BenchmarkComparator()
    comp.run_baseline()
    # ... apply optimization ...
    comp.run_optimized()
    comp.print_report()
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUTS_DIR = PROJECT_ROOT / "tests" / "fixtures" / "inputs"
RESULTS_DIR = PROJECT_ROOT / "tests" / "benchmarks" / "results"


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Statistical comparison of two sets of timing measurements."""

    problem: str
    baseline_times: tuple[float, ...]
    optimized_times: tuple[float, ...]
    baseline_median: float
    optimized_median: float
    speedup_ratio: float  # baseline / optimized (>1 = faster)
    change_pct: float  # negative = improvement
    t_statistic: float
    p_value: float
    significant: bool  # at alpha=0.05
    # Search correctness
    baseline_given: int = 0
    optimized_given: int = 0
    search_equivalent: bool = True

    def summary(self) -> str:
        direction = "FASTER" if self.speedup_ratio > 1.0 else "SLOWER"
        sig = "***" if self.significant else ""
        return (
            f"{self.problem:<25} "
            f"{self.baseline_median*1000:>8.1f}ms -> {self.optimized_median*1000:>8.1f}ms  "
            f"{self.speedup_ratio:>5.2f}x {direction} "
            f"(p={self.p_value:.4f}) {sig}"
            f"{'  SEARCH DIFF!' if not self.search_equivalent else ''}"
        )


def _welch_t_test(
    a: list[float], b: list[float],
) -> tuple[float, float]:
    """Welch's t-test for unequal variances. Returns (t_statistic, p_value)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    m1 = statistics.mean(a)
    m2 = statistics.mean(b)
    v1 = statistics.variance(a)
    v2 = statistics.variance(b)

    # Avoid division by zero
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se < 1e-15:
        return 0.0, 1.0

    t_stat = (m1 - m2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    if denom < 1e-15:
        return t_stat, 1.0
    df = num / denom

    # Approximate p-value using normal distribution for large df
    # For small df, use a conservative approximation
    if df >= 30:
        # Normal approximation
        p_value = 2.0 * _normal_cdf(-abs(t_stat))
    else:
        # Student-t approximation (conservative)
        p_value = 2.0 * _student_t_cdf(-abs(t_stat), df)

    return t_stat, p_value


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _student_t_cdf(x: float, df: float) -> float:
    """Conservative approximation of Student-t CDF.

    Uses the relationship between t-distribution and regularized
    incomplete beta function, with a simple approximation.
    """
    # For negative x, use symmetry
    if x >= 0:
        # Approximation: for large df, approaches normal
        return _normal_cdf(x * (1 - 1 / (4 * max(df, 1))))
    return 1.0 - _student_t_cdf(-x, df)


def compare_runs(
    baseline_times: list[float],
    optimized_times: list[float],
    problem: str = "unknown",
    *,
    baseline_given: int = 0,
    optimized_given: int = 0,
    alpha: float = 0.05,
) -> ComparisonResult:
    """Compare two sets of wall-clock times statistically."""
    b_med = statistics.median(baseline_times)
    o_med = statistics.median(optimized_times)

    speedup = b_med / o_med if o_med > 1e-9 else float("inf")
    change_pct = ((o_med - b_med) / b_med * 100) if b_med > 1e-9 else 0.0

    t_stat, p_value = _welch_t_test(baseline_times, optimized_times)

    return ComparisonResult(
        problem=problem,
        baseline_times=tuple(baseline_times),
        optimized_times=tuple(optimized_times),
        baseline_median=b_med,
        optimized_median=o_med,
        speedup_ratio=speedup,
        change_pct=change_pct,
        t_statistic=t_stat,
        p_value=p_value,
        significant=p_value < alpha,
        baseline_given=baseline_given,
        optimized_given=optimized_given,
        search_equivalent=(baseline_given == optimized_given or baseline_given == 0),
    )


@dataclass
class ProblemRun:
    """Timing + correctness data for one problem across N runs."""

    problem: str
    wall_times: list[float] = field(default_factory=list)
    given_counts: list[int] = field(default_factory=list)
    kept_counts: list[int] = field(default_factory=list)
    generated_counts: list[int] = field(default_factory=list)
    proved: list[bool] = field(default_factory=list)


def _discover_bench_problems() -> list[Path]:
    """Find all benchmark .in files."""
    return sorted(INPUTS_DIR.glob("bench_*.in"))


def run_benchmark_set(
    problems: list[Path] | None = None,
    *,
    runs: int = 5,
    timeout: float = 60.0,
    warmup: int = 1,
) -> dict[str, ProblemRun]:
    """Run a set of benchmark problems multiple times.

    Returns mapping from problem name to collected measurements.
    """
    import time

    from tests.benchmarks.perf_profiler import run_problem

    if problems is None:
        problems = _discover_bench_problems()

    results: dict[str, ProblemRun] = {}

    for input_path in problems:
        name = input_path.stem
        pr = ProblemRun(problem=name)

        # Warmup runs (not counted)
        for _ in range(warmup):
            try:
                run_problem(input_path, timeout=timeout, quiet=True)
            except Exception:
                pass

        # Timed runs
        for i in range(runs):
            try:
                t0 = time.perf_counter()
                result = run_problem(input_path, timeout=timeout, quiet=True)
                elapsed = time.perf_counter() - t0

                pr.wall_times.append(elapsed)
                pr.given_counts.append(result.stats.given)
                pr.kept_counts.append(result.stats.kept)
                pr.generated_counts.append(result.stats.generated)
                from pyladr.search.given_clause import ExitCode
                pr.proved.append(result.exit_code == ExitCode.MAX_PROOFS_EXIT)
            except Exception as e:
                pr.wall_times.append(float("inf"))
                pr.proved.append(False)

        results[name] = pr

    return results


class BenchmarkComparator:
    """Runs before/after benchmark comparisons with full statistical analysis."""

    def __init__(
        self,
        problems: list[Path] | None = None,
        runs: int = 5,
        timeout: float = 60.0,
    ) -> None:
        self.problems = problems or _discover_bench_problems()
        self.runs = runs
        self.timeout = timeout
        self.baseline: dict[str, ProblemRun] | None = None
        self.optimized: dict[str, ProblemRun] | None = None
        self.comparisons: list[ComparisonResult] = []

    def run_baseline(self, label: str = "baseline") -> dict[str, ProblemRun]:
        """Collect baseline measurements."""
        print(f"Running {label} benchmarks ({self.runs} runs each)...")
        self.baseline = run_benchmark_set(
            self.problems, runs=self.runs, timeout=self.timeout,
        )
        return self.baseline

    def run_optimized(self, label: str = "optimized") -> dict[str, ProblemRun]:
        """Collect optimized measurements."""
        print(f"Running {label} benchmarks ({self.runs} runs each)...")
        self.optimized = run_benchmark_set(
            self.problems, runs=self.runs, timeout=self.timeout,
        )
        return self.optimized

    def compare(self) -> list[ComparisonResult]:
        """Compare baseline vs optimized results."""
        if self.baseline is None or self.optimized is None:
            raise RuntimeError("Must run both baseline and optimized first")

        self.comparisons = []
        for name in self.baseline:
            if name not in self.optimized:
                continue
            bl = self.baseline[name]
            opt = self.optimized[name]

            b_given = bl.given_counts[0] if bl.given_counts else 0
            o_given = opt.given_counts[0] if opt.given_counts else 0

            cr = compare_runs(
                bl.wall_times,
                opt.wall_times,
                problem=name,
                baseline_given=b_given,
                optimized_given=o_given,
            )
            self.comparisons.append(cr)

        return self.comparisons

    def print_report(self) -> None:
        """Print formatted comparison report."""
        if not self.comparisons:
            self.compare()

        print("\n" + "=" * 90)
        print("PERFORMANCE COMPARISON REPORT")
        print("=" * 90)
        print(
            f"{'Problem':<25} {'Baseline':>10} {'Optimized':>10} "
            f"{'Speedup':>9} {'p-value':>9} {'Sig':>4} {'Search':>7}"
        )
        print("-" * 90)

        faster_count = 0
        slower_count = 0
        sig_faster = 0
        search_diffs = 0

        for cr in self.comparisons:
            sig = "***" if cr.significant else ""
            equiv = "OK" if cr.search_equivalent else "DIFF!"
            if cr.search_equivalent is False:
                search_diffs += 1

            if cr.speedup_ratio > 1.0:
                faster_count += 1
                if cr.significant:
                    sig_faster += 1
            elif cr.speedup_ratio < 1.0:
                slower_count += 1

            print(
                f"{cr.problem:<25} "
                f"{cr.baseline_median*1000:>8.1f}ms "
                f"{cr.optimized_median*1000:>8.1f}ms "
                f"{cr.speedup_ratio:>8.2f}x "
                f"{cr.p_value:>8.4f} "
                f"{sig:>4} "
                f"{equiv:>7}"
            )

        # Geometric mean of speedups
        speedups = [cr.speedup_ratio for cr in self.comparisons if cr.speedup_ratio > 0]
        if speedups:
            geo_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        else:
            geo_mean = 1.0

        print("-" * 90)
        print(f"Geometric mean speedup: {geo_mean:.3f}x")
        print(f"Faster: {faster_count}/{len(self.comparisons)}  "
              f"(significant: {sig_faster})")
        print(f"Slower: {slower_count}/{len(self.comparisons)}")
        if search_diffs:
            print(f"WARNING: {search_diffs} problem(s) have different search behavior!")
        print("=" * 90)

    def save_results(self, path: Path | None = None) -> Path:
        """Save comparison results to JSON."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            import datetime
            ts = datetime.datetime.now(datetime.UTC).isoformat().replace(":", "-")
            path = RESULTS_DIR / f"comparison_{ts}.json"

        data = {
            "comparisons": [
                {
                    "problem": cr.problem,
                    "baseline_median_ms": cr.baseline_median * 1000,
                    "optimized_median_ms": cr.optimized_median * 1000,
                    "speedup_ratio": cr.speedup_ratio,
                    "change_pct": cr.change_pct,
                    "p_value": cr.p_value,
                    "significant": cr.significant,
                    "search_equivalent": cr.search_equivalent,
                }
                for cr in self.comparisons
            ]
        }
        path.write_text(json.dumps(data, indent=2) + "\n")
        return path
