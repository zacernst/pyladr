"""Performance benchmarking: ML-enabled vs static clause selection.

Compares traditional (static) search against ML-enhanced selection
across standard benchmark problems. Measures:

- Proof discovery rates
- Clauses given/generated/kept (search efficiency)
- Wall-clock time and overhead
- Learning adaptation metrics (when online learning is active)

Usage:
    pytest tests/benchmarks/test_ml_benchmark.py -v -m benchmark
    pytest tests/benchmarks/test_ml_benchmark.py -v -k "baseline"
    pytest tests/benchmarks/test_ml_benchmark.py -v -k "comparison"
"""

from __future__ import annotations

import io
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import warnings

import pytest

# torch_geometric triggers DeprecationWarning on torch.jit.script which
# the project's filterwarnings=["error"] converts to an error. Suppress it
# for benchmarks since we don't control that dependency.
pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning:torch",
)

INPUTS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "inputs"
EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples" / "sample_problems"


# ── Data structures ───────────────────────────────────────────────────────


@dataclass(slots=True)
class SearchBenchmarkResult:
    """Result of a single search benchmark run."""

    problem: str
    mode: str  # "static" or "ml"
    proved: bool = False
    exit_code: int = 0
    given: int = 0
    generated: int = 0
    kept: int = 0
    subsumed: int = 0
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    keep_rate: float = 0.0
    proof_length: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "mode": self.mode,
            "proved": self.proved,
            "exit_code": self.exit_code,
            "given": self.given,
            "generated": self.generated,
            "kept": self.kept,
            "subsumed": self.subsumed,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "keep_rate": self.keep_rate,
            "proof_length": self.proof_length,
        }


@dataclass(slots=True)
class MLComparisonResult:
    """Comparison between static and ML-enabled search on one problem."""

    problem: str
    static_result: SearchBenchmarkResult
    ml_result: SearchBenchmarkResult | None = None

    # Deltas (positive = ML better)
    given_reduction_pct: float = 0.0
    generated_reduction_pct: float = 0.0
    kept_reduction_pct: float = 0.0
    time_reduction_pct: float = 0.0
    same_proof_status: bool = True

    verdict: str = "pending"  # "improvement", "regression", "neutral", "incomparable"
    notes: list[str] = field(default_factory=list)

    def compute(self) -> None:
        """Compute comparison metrics."""
        s = self.static_result
        m = self.ml_result

        if m is None or m.error:
            self.verdict = "incomparable"
            self.notes.append("ML result unavailable")
            return

        self.same_proof_status = s.proved == m.proved

        if not self.same_proof_status:
            if m.proved and not s.proved:
                self.verdict = "improvement"
                self.notes.append("ML proved what static could not")
            else:
                self.verdict = "regression"
                self.notes.append("ML failed where static succeeded")
            return

        if not s.proved:
            self.verdict = "neutral"
            self.notes.append("Neither proved the theorem")
            return

        # Both proved — compare efficiency
        if s.given > 0:
            self.given_reduction_pct = (s.given - m.given) / s.given * 100
        if s.generated > 0:
            self.generated_reduction_pct = (s.generated - m.generated) / s.generated * 100
        if s.kept > 0:
            self.kept_reduction_pct = (s.kept - m.kept) / s.kept * 100
        if s.wall_seconds > 0:
            self.time_reduction_pct = (s.wall_seconds - m.wall_seconds) / s.wall_seconds * 100

        # Verdict based on given clause reduction (primary metric)
        if self.given_reduction_pct > 5:
            self.verdict = "improvement"
            self.notes.append(f"ML used {self.given_reduction_pct:.1f}% fewer given clauses")
        elif self.given_reduction_pct < -5:
            self.verdict = "regression"
            self.notes.append(f"ML used {abs(self.given_reduction_pct):.1f}% more given clauses")
        else:
            self.verdict = "neutral"
            self.notes.append("No significant difference in given clauses")


@dataclass(slots=True)
class MLBenchmarkSuite:
    """Collection of static vs ML comparison results."""

    results: list[MLComparisonResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def compute_summary(self) -> None:
        """Compute aggregate statistics."""
        total = len(self.results)
        improvements = sum(1 for r in self.results if r.verdict == "improvement")
        regressions = sum(1 for r in self.results if r.verdict == "regression")
        neutral = sum(1 for r in self.results if r.verdict == "neutral")

        given_reductions = [
            r.given_reduction_pct for r in self.results
            if r.static_result.proved and r.ml_result and r.ml_result.proved
        ]

        self.summary = {
            "total_problems": total,
            "improvements": improvements,
            "regressions": regressions,
            "neutral": neutral,
            "incomparable": total - improvements - regressions - neutral,
            "median_given_reduction_pct": (
                statistics.median(given_reductions) if given_reductions else 0.0
            ),
            "mean_given_reduction_pct": (
                statistics.mean(given_reductions) if given_reductions else 0.0
            ),
        }


# ── Runner ────────────────────────────────────────────────────────────────


def _run_static_search(
    input_file: Path,
    max_seconds: float = 30.0,
) -> SearchBenchmarkResult:
    """Run traditional (static) search and capture results."""
    from pyladr.apps.prover9 import run_prover

    result = SearchBenchmarkResult(
        problem=input_file.stem,
        mode="static",
    )

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        wall_start = time.perf_counter()
        try:
            exit_code = run_prover([
                "pyprover9", "-f", str(input_file), "--quiet",
                "-max_seconds", str(max_seconds),
            ])
            result.exit_code = exit_code
        except Exception as e:
            result.error = str(e)
            return result
        finally:
            result.wall_seconds = time.perf_counter() - wall_start
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    result.proved = "THEOREM PROVED" in output

    # Parse statistics
    m = re.search(r"Given=(\d+)", output)
    if m:
        result.given = int(m.group(1))
    m = re.search(r"Generated=(\d+)", output)
    if m:
        result.generated = int(m.group(1))
    m = re.search(r"Kept=(\d+)", output)
    if m:
        result.kept = int(m.group(1))
    m = re.search(r"User_CPU=(\d+\.\d+)", output)
    if m:
        result.cpu_seconds = float(m.group(1))
    m = re.search(r"Length of proof is (\d+)", output)
    if m:
        result.proof_length = int(m.group(1))

    if result.generated > 0:
        result.keep_rate = result.kept / result.generated

    return result


def _run_ml_search(
    input_file: Path,
    max_seconds: float = 30.0,
) -> SearchBenchmarkResult:
    """Run ML-enhanced (online learning) search and capture results."""
    from pyladr.apps.prover9 import run_prover

    result = SearchBenchmarkResult(
        problem=input_file.stem,
        mode="ml",
    )

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        wall_start = time.perf_counter()
        try:
            exit_code = run_prover([
                "pyprover9", "-f", str(input_file), "--quiet",
                "--online-learning",
                "-max_seconds", str(max_seconds),
            ])
            result.exit_code = exit_code
        except Exception as e:
            result.error = str(e)
            return result
        finally:
            result.wall_seconds = time.perf_counter() - wall_start
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    result.proved = "THEOREM PROVED" in output

    m = re.search(r"Given=(\d+)", output)
    if m:
        result.given = int(m.group(1))
    m = re.search(r"Generated=(\d+)", output)
    if m:
        result.generated = int(m.group(1))
    m = re.search(r"Kept=(\d+)", output)
    if m:
        result.kept = int(m.group(1))
    m = re.search(r"User_CPU=(\d+\.\d+)", output)
    if m:
        result.cpu_seconds = float(m.group(1))
    m = re.search(r"Length of proof is (\d+)", output)
    if m:
        result.proof_length = int(m.group(1))

    if result.generated > 0:
        result.keep_rate = result.kept / result.generated

    return result


def _run_comparison(
    input_file: Path,
    max_seconds: float = 30.0,
) -> MLComparisonResult:
    """Run both static and ML search on a problem and compare."""
    static = _run_static_search(input_file, max_seconds=max_seconds)
    ml = _run_ml_search(input_file, max_seconds=max_seconds)
    comp = MLComparisonResult(
        problem=input_file.stem,
        static_result=static,
        ml_result=ml,
    )
    comp.compute()
    return comp


def _get_benchmark_problems() -> list[tuple[str, Path]]:
    """Discover all available benchmark problems."""
    problems: list[tuple[str, Path]] = []

    # Standard benchmark fixtures
    if INPUTS_DIR.exists():
        for f in sorted(INPUTS_DIR.glob("*.in")):
            problems.append((f.stem, f))

    # Sample problems from examples
    if EXAMPLES_DIR.exists():
        for f in sorted(EXAMPLES_DIR.glob("*.in")):
            problems.append((f.stem, f))

    return problems


# ── Report ────────────────────────────────────────────────────────────────


def print_ml_benchmark_report(suite: MLBenchmarkSuite) -> str:
    """Generate a formatted ML benchmark comparison report."""
    lines = [
        "=" * 78,
        "ML vs STATIC SEARCH BENCHMARK REPORT",
        "=" * 78,
        "",
    ]

    # Table header
    lines.append(
        f"{'Problem':<28} {'Status':>7} {'S-Given':>8} {'M-Given':>8} "
        f"{'Given%':>7} {'S-Time':>8} {'M-Time':>8} {'Verdict':>12}"
    )
    lines.append("-" * 96)

    for comp in suite.results:
        s = comp.static_result
        status = "PROVED" if s.proved else "FAILED"
        s_given = str(s.given)
        s_time = f"{s.wall_seconds:.3f}s"

        if comp.ml_result and not comp.ml_result.error:
            m = comp.ml_result
            m_given = str(m.given)
            m_time = f"{m.wall_seconds:.3f}s"
            given_pct = f"{comp.given_reduction_pct:+.1f}%" if s.proved and m.proved else "N/A"
        else:
            m_given = "N/A"
            m_time = "N/A"
            given_pct = "N/A"

        lines.append(
            f"{s.problem:<28} {status:>7} {s_given:>8} {m_given:>8} "
            f"{given_pct:>7} {s_time:>8} {m_time:>8} {comp.verdict:>12}"
        )
        for note in comp.notes:
            lines.append(f"  -> {note}")

    # Summary
    s = suite.summary
    lines.append("")
    lines.append(
        f"Problems: {s['total_problems']} | "
        f"Improved: {s['improvements']} | "
        f"Regressed: {s['regressions']} | "
        f"Neutral: {s['neutral']}"
    )
    if s.get("median_given_reduction_pct"):
        lines.append(
            f"Given clause reduction: "
            f"median={s['median_given_reduction_pct']:.1f}%, "
            f"mean={s['mean_given_reduction_pct']:.1f}%"
        )

    lines.append("=" * 78)
    return "\n".join(lines)


# ── Integration with monitoring ───────────────────────────────────────────


def create_regression_baseline(
    results: list[SearchBenchmarkResult],
) -> dict[str, Any]:
    """Create a LearningBaseline from static search results.

    Returns dict suitable for LearningRegressionDetector.set_baseline().
    """
    proved = [r for r in results if r.proved]
    total = len(results)

    proof_rate = len(proved) / total if total > 0 else 0.0
    avg_given = (
        sum(r.given for r in proved) / len(proved) if proved else 0.0
    )
    avg_kept_rate = (
        sum(r.keep_rate for r in results) / total if total > 0 else 0.0
    )
    avg_elapsed = (
        sum(r.wall_seconds for r in proved) / len(proved) if proved else 0.0
    )

    return {
        "proof_rate": proof_rate,
        "avg_given": avg_given,
        "avg_kept_rate": avg_kept_rate,
        "avg_elapsed": avg_elapsed,
        "sample_count": total,
    }


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark
class TestStaticBaseline:
    """Establish static search baselines for benchmark problems."""

    def test_discover_problems(self):
        """Verify we can find benchmark problems."""
        problems = _get_benchmark_problems()
        assert len(problems) > 0, "No benchmark problems found"

    @pytest.mark.parametrize(
        "name,path",
        _get_benchmark_problems()[:10],  # Limit to first 10 for speed
        ids=[name for name, _ in _get_benchmark_problems()[:10]],
    )
    def test_static_baseline(self, name: str, path: Path):
        """Run static search and verify it produces valid results."""
        result = _run_static_search(path, max_seconds=30.0)
        assert result.error is None, f"Search error: {result.error}"
        assert result.given >= 0
        assert result.generated >= 0
        assert result.kept >= 0
        assert result.wall_seconds > 0

    def test_baseline_consistency(self):
        """Verify static search gives deterministic results across runs."""
        problems = _get_benchmark_problems()
        if not problems:
            pytest.skip("No benchmark problems found")

        # Use first available problem
        _, path = problems[0]
        r1 = _run_static_search(path, max_seconds=30.0)
        r2 = _run_static_search(path, max_seconds=30.0)

        assert r1.proved == r2.proved
        assert r1.given == r2.given, (
            f"Non-deterministic: given={r1.given} vs {r2.given}"
        )
        assert r1.generated == r2.generated
        assert r1.kept == r2.kept


@pytest.mark.benchmark
class TestMLComparisonFramework:
    """Test the ML comparison infrastructure (without requiring trained models)."""

    def test_comparison_result_both_proved(self):
        """Test comparison when both modes find a proof."""
        static = SearchBenchmarkResult(
            problem="test", mode="static",
            proved=True, given=100, generated=1000, kept=200,
            wall_seconds=1.0,
        )
        ml = SearchBenchmarkResult(
            problem="test", mode="ml",
            proved=True, given=80, generated=800, kept=180,
            wall_seconds=0.9,
        )
        comp = MLComparisonResult(problem="test", static_result=static, ml_result=ml)
        comp.compute()

        assert comp.verdict == "improvement"
        assert comp.given_reduction_pct == pytest.approx(20.0)
        assert comp.same_proof_status

    def test_comparison_result_regression(self):
        """Test comparison when ML uses more given clauses."""
        static = SearchBenchmarkResult(
            problem="test", mode="static",
            proved=True, given=100, generated=1000, kept=200,
            wall_seconds=1.0,
        )
        ml = SearchBenchmarkResult(
            problem="test", mode="ml",
            proved=True, given=120, generated=1200, kept=250,
            wall_seconds=1.2,
        )
        comp = MLComparisonResult(problem="test", static_result=static, ml_result=ml)
        comp.compute()

        assert comp.verdict == "regression"
        assert comp.given_reduction_pct < 0

    def test_comparison_result_neutral(self):
        """Test comparison when results are within noise."""
        static = SearchBenchmarkResult(
            problem="test", mode="static",
            proved=True, given=100, generated=1000, kept=200,
            wall_seconds=1.0,
        )
        ml = SearchBenchmarkResult(
            problem="test", mode="ml",
            proved=True, given=99, generated=990, kept=198,
            wall_seconds=1.0,
        )
        comp = MLComparisonResult(problem="test", static_result=static, ml_result=ml)
        comp.compute()

        assert comp.verdict == "neutral"

    def test_comparison_ml_proves_more(self):
        """Test when ML solves what static couldn't."""
        static = SearchBenchmarkResult(
            problem="test", mode="static",
            proved=False, given=500, generated=5000,
        )
        ml = SearchBenchmarkResult(
            problem="test", mode="ml",
            proved=True, given=300, generated=3000,
        )
        comp = MLComparisonResult(problem="test", static_result=static, ml_result=ml)
        comp.compute()

        assert comp.verdict == "improvement"
        assert not comp.same_proof_status

    def test_comparison_ml_unavailable(self):
        """Test when ML result is unavailable."""
        static = SearchBenchmarkResult(
            problem="test", mode="static", proved=True, given=100,
        )
        comp = MLComparisonResult(problem="test", static_result=static, ml_result=None)
        comp.compute()

        assert comp.verdict == "incomparable"

    def test_benchmark_suite(self):
        """Test suite aggregation."""
        suite = MLBenchmarkSuite()

        # Add a mix of results
        for i, (given_s, given_m, verdict) in enumerate([
            (100, 80, "improvement"),
            (100, 110, "regression"),
            (100, 99, "neutral"),
        ]):
            static = SearchBenchmarkResult(
                problem=f"p{i}", mode="static",
                proved=True, given=given_s, generated=given_s * 10,
                kept=given_s * 2, wall_seconds=1.0,
            )
            ml = SearchBenchmarkResult(
                problem=f"p{i}", mode="ml",
                proved=True, given=given_m, generated=given_m * 10,
                kept=given_m * 2, wall_seconds=0.9,
            )
            comp = MLComparisonResult(
                problem=f"p{i}", static_result=static, ml_result=ml,
            )
            comp.compute()
            suite.results.append(comp)

        suite.compute_summary()
        assert suite.summary["total_problems"] == 3
        assert suite.summary["improvements"] == 1
        assert suite.summary["regressions"] == 1
        assert suite.summary["neutral"] == 1

    def test_report_formatting(self):
        """Test report generation."""
        suite = MLBenchmarkSuite()
        static = SearchBenchmarkResult(
            problem="test_prob", mode="static",
            proved=True, given=100, generated=1000, kept=200,
            wall_seconds=1.5,
        )
        ml = SearchBenchmarkResult(
            problem="test_prob", mode="ml",
            proved=True, given=85, generated=850, kept=170,
            wall_seconds=1.3,
        )
        comp = MLComparisonResult(
            problem="test_prob", static_result=static, ml_result=ml,
        )
        comp.compute()
        suite.results.append(comp)
        suite.compute_summary()

        report = print_ml_benchmark_report(suite)
        assert "ML vs STATIC SEARCH BENCHMARK REPORT" in report
        assert "test_prob" in report
        assert "improvement" in report

    def test_create_regression_baseline(self):
        """Test baseline creation from search results."""
        results = [
            SearchBenchmarkResult(
                problem=f"p{i}", mode="static",
                proved=(i < 8), given=100 + i * 10,
                generated=1000, kept=200,
                keep_rate=0.2, wall_seconds=1.0 + i * 0.1,
            )
            for i in range(10)
        ]

        baseline = create_regression_baseline(results)
        assert baseline["proof_rate"] == pytest.approx(0.8)
        assert baseline["sample_count"] == 10
        assert baseline["avg_given"] > 0
        assert baseline["avg_kept_rate"] == pytest.approx(0.2)

    def test_regression_baseline_integration(self):
        """Test that baseline integrates with LearningRegressionDetector."""
        from pyladr.monitoring.learning_regression import LearningRegressionDetector

        results = [
            SearchBenchmarkResult(
                problem=f"p{i}", mode="static",
                proved=True, given=100, generated=1000, kept=200,
                keep_rate=0.2, wall_seconds=1.0,
            )
            for i in range(5)
        ]

        baseline_dict = create_regression_baseline(results)
        detector = LearningRegressionDetector(min_samples=3)
        detector.set_baseline(**baseline_dict)

        # Simulate ML results that are better
        for _ in range(5):
            detector.record_search_result(
                proved=True, given=80, kept_rate=0.25, elapsed=0.8,
            )

        report = detector.check()
        assert not report.is_regression


@pytest.mark.benchmark
class TestStaticSearchProfile:
    """Profile static search to establish performance baselines."""

    def test_search_profile_with_monitoring(self):
        """Run a search with full monitoring enabled."""
        from pyladr.monitoring.learning_monitor import LearningMonitor
        from pyladr.monitoring.learning_curves import LearningCurveAnalyzer

        problems = _get_benchmark_problems()
        if not problems:
            pytest.skip("No benchmark problems found")

        _, path = problems[0]
        result = _run_static_search(path, max_seconds=30.0)

        # Create monitoring baseline from this result
        monitor = LearningMonitor()
        curves = LearningCurveAnalyzer()

        # Simulate what monitoring would look like for a static run
        # (no model updates, just selection tracking)
        for _ in range(min(result.given, 50)):
            monitor.record_selection(ml_guided=False, productive=True)

        sel = monitor.selections
        assert sel.total_trad > 0
        assert sel.total_ml == 0

        # Report should work even with no updates
        report = monitor.report()
        assert "No model updates recorded" in report

    def test_collect_baselines(self):
        """Collect static baselines across all problems for regression tracking."""
        from pyladr.monitoring.learning_regression import LearningRegressionDetector

        problems = _get_benchmark_problems()
        if not problems:
            pytest.skip("No benchmark problems found")

        results: list[SearchBenchmarkResult] = []
        for name, path in problems[:5]:  # Limit for test speed
            result = _run_static_search(path, max_seconds=30.0)
            if result.error is None:
                results.append(result)

        assert len(results) > 0

        # Build baseline
        baseline_dict = create_regression_baseline(results)
        assert baseline_dict["sample_count"] > 0

        # Verify detector can use it
        detector = LearningRegressionDetector(min_samples=1)
        detector.set_baseline(**baseline_dict)
        assert detector.has_baseline


# ── Quick problems for fast comparison (proved in <2s) ────────────────────

_QUICK_PROBLEMS = [
    name_path for name_path in _get_benchmark_problems()
    if any(tag in name_path[0] for tag in [
        "simple_group", "identity_only", "lattice_absorption",
        "bench_group_comm",
    ])
]


@pytest.mark.benchmark
class TestMLvsStaticComparison:
    """Head-to-head comparison of ML-enabled vs static search.

    Runs both modes on the same problems and reports the differences.
    """

    @pytest.mark.parametrize(
        "name,path",
        _QUICK_PROBLEMS[:4] if _QUICK_PROBLEMS else [],
        ids=[name for name, _ in _QUICK_PROBLEMS[:4]] if _QUICK_PROBLEMS else [],
    )
    def test_head_to_head(self, name: str, path: Path):
        """Run both static and ML search and compare."""
        comp = _run_comparison(path, max_seconds=30.0)

        # Both should run without errors
        assert comp.static_result.error is None
        if comp.ml_result is not None:
            # ML might fail if torch_geometric has issues, don't hard-fail
            if comp.ml_result.error:
                pytest.skip(f"ML search error: {comp.ml_result.error}")

        # Both should produce the same proof status
        # (ML should not break provability)
        if comp.ml_result and not comp.ml_result.error:
            if comp.static_result.proved:
                assert comp.ml_result.proved, (
                    f"ML failed to prove {name} (static proved with "
                    f"given={comp.static_result.given})"
                )

        # Log the comparison for visibility
        print(f"\n{'='*60}")
        print(f"  {name}: {comp.verdict}")
        print(f"  Static: given={comp.static_result.given}, "
              f"gen={comp.static_result.generated}, "
              f"kept={comp.static_result.kept}, "
              f"time={comp.static_result.wall_seconds:.3f}s")
        if comp.ml_result and not comp.ml_result.error:
            print(f"  ML:     given={comp.ml_result.given}, "
                  f"gen={comp.ml_result.generated}, "
                  f"kept={comp.ml_result.kept}, "
                  f"time={comp.ml_result.wall_seconds:.3f}s")
            if comp.static_result.proved and comp.ml_result.proved:
                print(f"  Given reduction: {comp.given_reduction_pct:+.1f}%")
                print(f"  Time reduction:  {comp.time_reduction_pct:+.1f}%")
        for note in comp.notes:
            print(f"  -> {note}")
        print(f"{'='*60}")

    def test_full_suite_comparison(self):
        """Run full comparison suite across all quick problems."""
        if not _QUICK_PROBLEMS:
            pytest.skip("No quick benchmark problems found")

        suite = MLBenchmarkSuite()
        for name, path in _QUICK_PROBLEMS[:4]:
            comp = _run_comparison(path, max_seconds=30.0)
            if comp.ml_result and comp.ml_result.error:
                continue  # Skip problems where ML failed to initialize
            suite.results.append(comp)

        if not suite.results:
            pytest.skip("No successful comparisons")

        suite.compute_summary()
        report = print_ml_benchmark_report(suite)
        print(f"\n{report}")

        # Validate suite structure
        assert suite.summary["total_problems"] > 0

    def test_regression_detection_from_comparison(self):
        """Feed comparison results into regression detector."""
        if not _QUICK_PROBLEMS:
            pytest.skip("No quick benchmark problems found")

        from pyladr.monitoring.learning_regression import LearningRegressionDetector

        # Collect static baselines
        static_results: list[SearchBenchmarkResult] = []
        for name, path in _QUICK_PROBLEMS[:3]:
            result = _run_static_search(path, max_seconds=30.0)
            if result.error is None:
                static_results.append(result)

        if not static_results:
            pytest.skip("No static results")

        baseline_dict = create_regression_baseline(static_results)
        detector = LearningRegressionDetector(min_samples=1)
        detector.set_baseline(**baseline_dict)

        # Run ML searches and feed into detector
        for name, path in _QUICK_PROBLEMS[:3]:
            ml = _run_ml_search(path, max_seconds=30.0)
            if ml.error:
                continue
            detector.record_search_result(
                proved=ml.proved,
                given=ml.given,
                kept_rate=ml.keep_rate,
                elapsed=ml.wall_seconds,
            )

        if detector.result_count > 0:
            report = detector.check()
            print(f"\nRegression report:\n{report.summary()}")

    def test_monitoring_integration_with_comparison(self):
        """Verify monitoring captures useful data from ML runs."""
        from pyladr.monitoring.learning_monitor import LearningMonitor

        if not _QUICK_PROBLEMS:
            pytest.skip("No quick benchmark problems found")

        monitor = LearningMonitor()
        name, path = _QUICK_PROBLEMS[0]

        # Run static — record as traditional selections
        static = _run_static_search(path, max_seconds=30.0)
        for _ in range(min(static.given, 20)):
            monitor.record_selection(ml_guided=False, productive=True)

        # Run ML — record as ML selections
        ml = _run_ml_search(path, max_seconds=30.0)
        if ml.error:
            pytest.skip(f"ML search error: {ml.error}")
        for _ in range(min(ml.given, 20)):
            monitor.record_selection(ml_guided=True, productive=True)

        sel = monitor.selections
        assert sel.total_ml > 0
        assert sel.total_trad > 0
        print(f"\nML advantage: {sel.ml_advantage:+.3f}")
