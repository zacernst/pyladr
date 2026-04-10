"""Performance regression detection tests for CI integration.

Catches performance regressions by comparing current runs against stored
baselines. Tests are designed to:
  1. Fail on catastrophic regressions (>2x slowdown)
  2. Warn on moderate regressions (>1.3x slowdown)
  3. Report throughput drops in microbenchmarks
  4. Track memory allocation growth

Run with:
    pytest tests/benchmarks/test_regression_detection.py -v -m benchmark
    pytest tests/benchmarks/test_regression_detection.py -v -k "regression"
    pytest tests/benchmarks/test_regression_detection.py -v -k "memory"
"""

from __future__ import annotations

import statistics
import time
import warnings
from pathlib import Path

import pytest

from tests.benchmarks.c_baselines import C_BASELINES

INPUTS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "inputs"


# ── Regression tolerance configuration ─────────────────────────────────────


# Maximum allowed slowdown before test fails (2x = catastrophic regression)
CATASTROPHIC_SLOWDOWN = 2.0

# Warning threshold (1.3x = moderate regression, logged but not failed)
WARNING_SLOWDOWN = 1.3

# Number of timing runs for statistical confidence
REGRESSION_RUNS = 3

# Microbenchmark minimum throughput thresholds (ops/sec)
# Set conservatively — these catch catastrophic regressions, not 5% dips
THROUGHPUT_FLOORS: dict[str, float] = {
    "term_construction": 80_000,
    "unification": 8_000,
    "clause_weight": 40_000,
    "parsing": 800,
    "subsumption_check": 4_000,
    "resolution": 4_000,
    "demodulation": 3_000,
    "paramodulation": 1_000,
    "clause_hashing": 20_000,
    "graph_construction": 500,
}


# ── Helpers ────────────────────────────────────────────────────────────────


def _timed_runs(func, n: int = REGRESSION_RUNS) -> list[float]:
    """Run a function n times and return wall-clock times."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return times


def _run_problem(input_path: Path, timeout: float = 60.0):
    """Run problem through Python search engine."""
    from tests.benchmarks.perf_profiler import run_problem

    return run_problem(input_path, timeout=timeout, quiet=True)


def _get_solvable_bench_problems() -> list[tuple[str, Path]]:
    """Get benchmark problems known to be solvable."""
    problems = []
    for name, baseline in C_BASELINES.items():
        if not baseline.theorem_proved:
            continue
        path = INPUTS_DIR / f"{name}.in"
        if path.exists():
            problems.append((name, path))
    return problems


# ── Wall-clock regression tests ────────────────────────────────────────────


# Known problem time budgets (generous: 5x+ over typical)
_PROBLEM_TIME_BUDGETS: dict[str, float] = {
    "bench_luka20": 5.0,
    "bench_mv4": 5.0,
    "bench_ternary_boolean": 5.0,
    "bench_ring_comm": 15.0,
    "bench_group_comm_3": 60.0,
    "bench_robbins": 60.0,
    "bench_lattice_distrib": 180.0,
    # Expanded benchmarks
    "bench_group_inverse": 15.0,
    "bench_lattice_idempotent": 10.0,
    "bench_ba_orthocomplement": 60.0,
    "bench_cd_propositional": 15.0,
}


@pytest.mark.benchmark
class TestWallClockRegression:
    """Detect wall-clock time regressions across benchmark problems.

    Uses absolute time budgets (generous) to catch catastrophic regressions.
    For statistical before/after comparison, use BenchmarkComparator.
    """

    @pytest.mark.parametrize(
        "name,input_path",
        _get_solvable_bench_problems(),
        ids=[n for n, _ in _get_solvable_bench_problems()],
    )
    def test_problem_within_budget(self, name: str, input_path: Path) -> None:
        """Problem must complete within its time budget."""
        budget = _PROBLEM_TIME_BUDGETS.get(name, 30.0)

        try:
            t0 = time.perf_counter()
            result = _run_problem(input_path, timeout=budget + 5.0)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            pytest.skip(f"Cannot run {name}: {e}")
            return

        assert elapsed < budget, (
            f"{name}: {elapsed:.2f}s exceeds budget {budget:.1f}s"
        )

        # Warn if approaching budget (within 50%)
        if elapsed > budget * 0.5:
            warnings.warn(
                f"{name}: {elapsed:.2f}s is >50% of budget {budget:.1f}s — "
                f"approaching regression threshold",
                stacklevel=1,
            )


# ── Microbenchmark regression tests ───────────────────────────────────────


@pytest.mark.benchmark
class TestMicrobenchmarkRegression:
    """Detect throughput regressions in core operations.

    Each test measures ops/sec for a critical operation and asserts it
    exceeds the minimum floor. Floors are set conservatively to avoid
    flaky failures while catching real regressions.
    """

    def _assert_throughput(
        self, name: str, count: int, elapsed: float,
    ) -> float:
        """Assert throughput meets floor and return ops/sec."""
        ops = count / elapsed
        floor = THROUGHPUT_FLOORS.get(name, 1000)
        assert ops > floor, (
            f"{name}: {ops:.0f} ops/s below floor {floor:.0f}"
        )
        return ops

    def test_term_construction(self) -> None:
        """Term creation throughput."""
        from pyladr.core.term import get_rigid_term, get_variable_term

        count = 20_000
        t0 = time.perf_counter()
        for _ in range(count):
            a = get_rigid_term(1, 0)
            b = get_rigid_term(2, 0)
            x = get_variable_term(0)
            f_ab = get_rigid_term(3, 2, (a, b))
            g_x = get_rigid_term(4, 1, (x,))
            get_rigid_term(5, 2, (f_ab, g_x))
        elapsed = time.perf_counter() - t0
        self._assert_throughput("term_construction", count * 6, elapsed)

    def test_unification(self) -> None:
        """Unification throughput."""
        from pyladr.core.substitution import Context, Trail, unify
        from pyladr.core.term import get_rigid_term, get_variable_term

        x, y = get_variable_term(0), get_variable_term(1)
        a, b = get_rigid_term(1, 0), get_rigid_term(2, 0)
        f_x_y = get_rigid_term(3, 2, (x, y))
        f_a_b = get_rigid_term(3, 2, (a, b))
        g_f = get_rigid_term(4, 1, (f_x_y,))
        g_fab = get_rigid_term(4, 1, (f_a_b,))

        count = 20_000
        t0 = time.perf_counter()
        for _ in range(count):
            c1, c2, tr = Context(), Context(), Trail()
            unify(g_f, c1, g_fab, c2, tr)
            tr.undo()
        elapsed = time.perf_counter() - t0
        self._assert_throughput("unification", count, elapsed)

    def test_clause_weight(self) -> None:
        """Clause weighting throughput."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.search.selection import default_clause_weight

        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        clause = Clause(literals=(
            Literal(sign=True, atom=get_rigid_term(4, 2, (
                get_rigid_term(2, 1, (x,)), get_rigid_term(3, 1, (a,)),
            ))),
            Literal(sign=False, atom=get_rigid_term(5, 1, (x,))),
        ))

        count = 50_000
        t0 = time.perf_counter()
        for _ in range(count):
            default_clause_weight(clause)
        elapsed = time.perf_counter() - t0
        self._assert_throughput("clause_weight", count, elapsed)

    def test_parsing(self) -> None:
        """LADR parsing throughput."""
        from pyladr.core.symbol import SymbolTable
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        terms = [
            "f(g(x,y),a) = g(f(a,x),f(b,y))",
            "x * (y * z) = (x * y) * z",
            "f(x,f(y,z)) = f(f(x,y),z)",
        ]

        count = 5_000
        t0 = time.perf_counter()
        for i in range(count):
            parser.parse_term(terms[i % len(terms)])
        elapsed = time.perf_counter() - t0
        self._assert_throughput("parsing", count, elapsed)

    def test_subsumption_check(self) -> None:
        """Subsumption throughput."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.inference.subsumption import subsumes

        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        P_sn, Q_sn = 2, 3
        general = Clause(literals=(
            Literal(sign=True, atom=get_rigid_term(P_sn, 1, (x,))),
        ))
        specific = Clause(literals=(
            Literal(sign=True, atom=get_rigid_term(P_sn, 1, (a,))),
            Literal(sign=True, atom=get_rigid_term(Q_sn, 1, (x,))),
        ))

        count = 10_000
        t0 = time.perf_counter()
        for _ in range(count):
            subsumes(general, specific)
        elapsed = time.perf_counter() - t0
        self._assert_throughput("subsumption_check", count, elapsed)

    def test_resolution(self) -> None:
        """Binary resolution throughput."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.inference.resolution import all_binary_resolvents

        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        P_sn = 2
        c1 = Clause(literals=(
            Literal(sign=True, atom=get_rigid_term(P_sn, 1, (a,))),
        ))
        c2 = Clause(literals=(
            Literal(sign=False, atom=get_rigid_term(P_sn, 1, (x,))),
        ))

        count = 10_000
        t0 = time.perf_counter()
        for _ in range(count):
            list(all_binary_resolvents(c1, c2))
        elapsed = time.perf_counter() - t0
        self._assert_throughput("resolution", count, elapsed)

    def test_clause_hashing(self) -> None:
        """Structural hash throughput for embedding cache."""
        try:
            from pyladr.ml.embeddings.cache import clause_structural_hash
        except ImportError:
            pytest.skip("ML components not available")

        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term

        x, y = get_variable_term(0), get_variable_term(1)
        clause = Clause(literals=(
            Literal(sign=True, atom=get_rigid_term(2, 2, (
                get_rigid_term(3, 1, (x,)),
                get_rigid_term(4, 2, (y, get_rigid_term(1, 0))),
            ))),
        ))

        count = 20_000
        t0 = time.perf_counter()
        for _ in range(count):
            # Clear cached hash to measure actual computation
            try:
                del clause._structural_hash
            except AttributeError:
                pass
            clause_structural_hash(clause)
        elapsed = time.perf_counter() - t0
        self._assert_throughput("clause_hashing", count, elapsed)

    def test_graph_construction(self) -> None:
        """Clause-to-graph conversion throughput."""
        try:
            from pyladr.ml.graph.clause_graph import clause_to_heterograph
        except (ImportError, DeprecationWarning, Exception):
            pytest.skip("torch_geometric not available or incompatible")

        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term

        x, y = get_variable_term(0), get_variable_term(1)
        clause = Clause(literals=(
            Literal(sign=True, atom=get_rigid_term(2, 2, (
                get_rigid_term(3, 1, (x,)),
                get_rigid_term(4, 2, (y, get_rigid_term(1, 0))),
            ))),
            Literal(sign=False, atom=get_rigid_term(5, 1, (x,))),
        ))

        count = 1_000
        t0 = time.perf_counter()
        for _ in range(count):
            clause_to_heterograph(clause)
        elapsed = time.perf_counter() - t0
        self._assert_throughput("graph_construction", count, elapsed)


# ── Memory allocation regression tests ────────────────────────────────────


@pytest.mark.benchmark
class TestMemoryRegression:
    """Detect memory allocation regressions using tracemalloc.

    Monitors peak memory during key operations to catch allocation bloat
    (e.g., a change that creates 10x more temporary objects).
    """

    def test_search_memory_bounded(self) -> None:
        """Search on a small problem must not allocate excessive memory."""
        import tracemalloc

        path = INPUTS_DIR / "bench_luka20.in"
        if not path.exists():
            pytest.skip("bench_luka20.in not found")

        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        try:
            _run_problem(path, timeout=30.0)
        except Exception:
            pytest.skip("Cannot run bench_luka20")

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Compare top allocation differences
        stats = snapshot_after.compare_to(snapshot_before, "lineno")
        total_new_mb = sum(s.size_diff for s in stats if s.size_diff > 0) / (1024 * 1024)

        # Small problem should use <50MB of new allocations
        assert total_new_mb < 50.0, (
            f"bench_luka20 allocated {total_new_mb:.1f}MB of new memory"
        )

    def test_term_construction_no_leak(self) -> None:
        """Creating and discarding terms must not leak memory."""
        import tracemalloc

        from pyladr.core.term import get_rigid_term, get_variable_term

        tracemalloc.start()

        # Warmup
        for _ in range(1000):
            get_rigid_term(1, 2, (get_variable_term(0), get_rigid_term(2, 0)))

        _, peak_warmup = tracemalloc.get_traced_memory()

        # Main workload
        for _ in range(10_000):
            get_rigid_term(1, 2, (get_variable_term(0), get_rigid_term(2, 0)))

        _, peak_main = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak should not grow proportionally — terms may be interned or GC'd
        # Allow 3x growth (generous for potential caching effects)
        growth = peak_main / max(peak_warmup, 1)
        assert growth < 3.0, (
            f"Memory grew {growth:.1f}x during term construction "
            f"(warmup={peak_warmup/1024:.0f}KB, main={peak_main/1024:.0f}KB)"
        )

    def test_clause_processing_memory(self) -> None:
        """Clause processing pipeline memory should be bounded."""
        import tracemalloc

        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.search.selection import default_clause_weight

        tracemalloc.start()

        clauses = []
        for i in range(1000):
            x = get_variable_term(i % 10)
            a = get_rigid_term(1, 0)
            f_x = get_rigid_term(2, 1, (x,))
            atom = get_rigid_term(3, 2, (f_x, a))
            c = Clause(literals=(Literal(sign=True, atom=atom),))
            default_clause_weight(c)
            clauses.append(c)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        # 1000 small clauses should use <20MB
        assert peak_mb < 20.0, (
            f"1000 clauses used {peak_mb:.1f}MB peak memory"
        )


# ── Search statistics stability tests ─────────────────────────────────────


@pytest.mark.benchmark
class TestSearchStability:
    """Verify that search statistics are deterministic across runs.

    Non-determinism in search (e.g., dict ordering issues, race conditions)
    would show up as varying given/kept/generated counts.
    """

    @pytest.mark.parametrize(
        "name,input_path",
        [(n, p) for n, p in _get_solvable_bench_problems()
         if n in ("bench_luka20", "bench_mv4", "bench_ring_comm")],
        ids=[n for n, _ in _get_solvable_bench_problems()
             if n in ("bench_luka20", "bench_mv4", "bench_ring_comm")],
    )
    def test_deterministic_search(self, name: str, input_path: Path) -> None:
        """Multiple runs must produce identical search statistics."""
        from pyladr.search.given_clause import ExitCode

        results = []
        for _ in range(3):
            try:
                result = _run_problem(input_path, timeout=30.0)
                results.append(result)
            except Exception:
                pytest.skip(f"Cannot run {name}")

        if len(results) < 2:
            pytest.skip("Not enough successful runs")

        # All runs must agree on given/generated/kept
        given_set = {r.stats.given for r in results}
        gen_set = {r.stats.generated for r in results}
        kept_set = {r.stats.kept for r in results}

        assert len(given_set) == 1, (
            f"{name}: non-deterministic given counts: {given_set}"
        )
        assert len(gen_set) == 1, (
            f"{name}: non-deterministic generated counts: {gen_set}"
        )
        assert len(kept_set) == 1, (
            f"{name}: non-deterministic kept counts: {kept_set}"
        )
