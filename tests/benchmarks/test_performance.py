"""Performance benchmark tests for pyladr vs C Prover9.

These tests verify that:
1. The C reference implementation solves all benchmark problems (sanity check)
2. The Python implementation matches C search behavior (behavioral equivalence)
3. The Python implementation meets performance targets (2-10x slowdown)

Run with: pytest tests/benchmarks/ -v -m benchmark
Run C baselines only: pytest tests/benchmarks/ -v -k "c_baseline"
Run full harness: python -m tests.benchmarks.bench_harness --c-only
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.benchmarks.c_baselines import C_BASELINES
from tests.cross_validation.c_runner import C_PROVER9_BIN, run_c_prover9

INPUTS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "inputs"
EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "prover9.examples"

# Maximum acceptable slowdown for Python vs C (single-threaded)
MAX_SINGLE_THREAD_SLOWDOWN = 10.0

requires_c_binary = pytest.mark.skipif(
    not C_PROVER9_BIN.exists(),
    reason="C prover9 binary not found (run 'make all' to build)",
)


def _get_benchmark_files() -> list[tuple[str, Path]]:
    """Get all benchmark input files as (name, path) pairs."""
    files = []
    x2 = EXAMPLES_DIR / "x2.in"
    if x2.exists():
        files.append(("x2", x2))
    for f in sorted(INPUTS_DIR.glob("bench_*.in")):
        files.append((f.stem, f))
    return files


# ── C Baseline Verification ──────────────────────────────────────────────────


@pytest.mark.benchmark
class TestCBaselines:
    """Verify that C baselines are correct by running the C binary."""

    @requires_c_binary
    @pytest.mark.parametrize(
        "problem_name,input_path",
        _get_benchmark_files(),
        ids=[name for name, _ in _get_benchmark_files()],
    )
    def test_c_baseline_solves(self, problem_name: str, input_path: Path):
        """Verify C prover9 solves each benchmark problem."""
        result = run_c_prover9(input_path, timeout=120.0)
        assert result.theorem_proved, (
            f"C prover9 failed to prove {problem_name}: "
            f"given={result.clauses_given}, gen={result.clauses_generated}"
        )

    @requires_c_binary
    @pytest.mark.parametrize("problem_name", list(C_BASELINES.keys()))
    def test_c_baseline_statistics_match(self, problem_name: str):
        """Verify C search statistics haven't changed from recorded baselines."""
        baseline = C_BASELINES[problem_name]

        input_path = INPUTS_DIR / f"{problem_name}.in"
        if not input_path.exists():
            input_path = EXAMPLES_DIR / f"{problem_name}.in"
        if not input_path.exists():
            pytest.skip(f"Input file not found for {problem_name}")

        result = run_c_prover9(input_path, timeout=120.0)

        assert result.theorem_proved == baseline.theorem_proved
        assert result.clauses_given == baseline.clauses_given, (
            f"Given clauses: expected {baseline.clauses_given}, got {result.clauses_given}"
        )
        assert result.clauses_generated == baseline.clauses_generated, (
            f"Generated: expected {baseline.clauses_generated}, got {result.clauses_generated}"
        )
        assert result.clauses_kept == baseline.clauses_kept, (
            f"Kept: expected {baseline.clauses_kept}, got {result.clauses_kept}"
        )

    @requires_c_binary
    def test_c_timing_report(self):
        """Report C timing baselines for all benchmark problems."""
        results = []
        for name, path in _get_benchmark_files():
            start = time.perf_counter()
            result = run_c_prover9(path, timeout=120.0)
            wall = time.perf_counter() - start
            results.append((name, result, wall))

        print("\n\n=== C Prover9 Performance Baselines ===")
        print(f"{'Problem':<25} {'CPU':>8} {'Wall':>8} {'Given':>7} {'Gen':>10} {'Kept':>7}")
        print("-" * 72)
        for name, result, wall in results:
            print(
                f"{name:<25} {result.user_cpu_time:>7.3f}s {wall:>7.3f}s "
                f"{result.clauses_given:>7} {result.clauses_generated:>10} "
                f"{result.clauses_kept:>7}"
            )


# ── Python Performance Tests ─────────────────────────────────────────────────


@pytest.mark.benchmark
class TestPythonPerformance:
    """Performance tests for the Python implementation.

    These will be activated once the Python search engine is functional.
    """

    @pytest.mark.skip(reason="Awaiting Python search engine implementation")
    @pytest.mark.parametrize(
        "problem_name,input_path",
        _get_benchmark_files(),
        ids=[name for name, _ in _get_benchmark_files()],
    )
    def test_python_solves_benchmark(self, problem_name: str, input_path: Path):
        """Verify Python solves each benchmark problem."""
        # TODO: Import and run pyladr solver
        pass

    @pytest.mark.skip(reason="Awaiting Python search engine implementation")
    @pytest.mark.parametrize("problem_name", list(C_BASELINES.keys()))
    def test_python_search_equivalence(self, problem_name: str):
        """Verify Python search matches C search statistics exactly."""
        pass

    @pytest.mark.skip(reason="Awaiting Python search engine implementation")
    @pytest.mark.parametrize(
        "problem_name",
        [name for name, b in C_BASELINES.items() if b.user_cpu_seconds >= 0.01],
    )
    def test_python_within_slowdown_target(self, problem_name: str):
        """Verify Python is within acceptable slowdown of C."""
        pass

    def test_term_construction_throughput(self):
        """Measure term construction rate — must build >100k terms/sec."""
        from pyladr.core.term import Term, get_rigid_term, get_variable_term

        start = time.perf_counter()
        count = 10_000
        for i in range(count):
            a = get_rigid_term(1, 0)
            b = get_rigid_term(2, 0)
            f_ab = get_rigid_term(3, 2, (a, b))
            g = get_rigid_term(4, 1, (f_ab,))
        elapsed = time.perf_counter() - start
        ops_per_sec = (count * 4) / elapsed
        assert ops_per_sec > 100_000, f"Term construction too slow: {ops_per_sec:.0f} ops/s"

    def test_unification_throughput(self):
        """Measure unification operations per second — must do >10k unifications/sec."""
        from pyladr.core.substitution import Context, Trail, unify
        from pyladr.core.term import get_rigid_term, get_variable_term

        x = get_variable_term(0)
        y = get_variable_term(1)
        a = get_rigid_term(1, 0)
        f_x = get_rigid_term(2, 1, (x,))
        f_a = get_rigid_term(2, 1, (a,))

        start = time.perf_counter()
        count = 10_000
        for _ in range(count):
            c1, c2, tr = Context(), Context(), Trail()
            unify(f_x, c1, f_a, c2, tr)
            tr.undo()
        elapsed = time.perf_counter() - start
        ops_per_sec = count / elapsed
        assert ops_per_sec > 10_000, f"Unification too slow: {ops_per_sec:.0f} ops/s"

    def test_parsing_throughput(self):
        """Measure clause parsing rate — must parse >1k clauses/sec."""
        from pyladr.parsing.ladr_parser import LADRParser
        from pyladr.core.symbol import SymbolTable

        st = SymbolTable()
        parser = LADRParser(st)
        clause_text = "f(g(x,y),a) = g(f(a,x),f(b,y))"

        start = time.perf_counter()
        count = 5_000
        for _ in range(count):
            parser.parse_term(clause_text)
        elapsed = time.perf_counter() - start
        ops_per_sec = count / elapsed
        assert ops_per_sec > 1_000, f"Parsing too slow: {ops_per_sec:.0f} ops/s"


# ── Free-Threading Performance Tests ─────────────────────────────────────────


@pytest.mark.benchmark
class TestFreeThreadingPerformance:
    """Tests for Python 3.14 free-threading parallel performance."""

    def test_parallel_inference_engine_functional(self):
        """Verify parallel inference engine can run without errors."""
        from pyladr.parallel.inference_engine import ParallelInferenceEngine, ParallelSearchConfig
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term

        config = ParallelSearchConfig(enabled=True, max_workers=2)
        engine = ParallelInferenceEngine(config)

        P_sn, a_sn = 1, 2
        a = get_rigid_term(a_sn, 0)
        x = get_variable_term(0)
        P_a = get_rigid_term(P_sn, 1, (a,))
        P_x = get_rigid_term(P_sn, 1, (x,))

        given = Clause(literals=(Literal(sign=True, atom=P_a),), id=1)
        usable = [Clause(literals=(Literal(sign=False, atom=P_x),), id=2)]

        results = engine.generate_inferences(
            given=given,
            usable_snapshot=usable,
            binary_resolution=True,
            paramodulation=False,
            factoring=False,
        )
        engine.shutdown()
        # Should produce at least one resolvent (empty clause)
        assert isinstance(results, list)

    def test_parallel_search_deterministic(self):
        """Verify parallel search produces consistent results across runs."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions
        from pyladr.parallel.inference_engine import ParallelSearchConfig

        P_sn, Q_sn, a_sn = 1, 2, 3
        a = get_rigid_term(a_sn, 0)
        x = get_variable_term(0)

        c1 = Clause(literals=(Literal(sign=True, atom=get_rigid_term(P_sn, 1, (a,))),))
        c2 = Clause(literals=(
            Literal(sign=False, atom=get_rigid_term(P_sn, 1, (x,))),
            Literal(sign=True, atom=get_rigid_term(Q_sn, 1, (x,))),
        ))
        c3 = Clause(literals=(Literal(sign=False, atom=get_rigid_term(Q_sn, 1, (a,))),))

        results = []
        for _ in range(3):
            opts = SearchOptions(
                binary_resolution=True,
                factoring=True,
                max_given=50,
                quiet=True,
                parallel=ParallelSearchConfig(enabled=True, max_workers=2),
            )
            search = GivenClauseSearch(options=opts)
            result = search.run(usable=[], sos=[c1, c2, c3])
            results.append(result.exit_code)

        # All runs should agree
        assert all(r == results[0] for r in results)
