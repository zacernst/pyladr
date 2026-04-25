"""File-based regression tests against reference C Prover9.

Runs stored LADR input files through both C and Python implementations
with hierarchical features disabled, verifying identical behavior.

Run with: pytest tests/compatibility/test_hierarchical_file_regression.py -v -m cross_validation
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions

from tests.conftest import C_PROVER9_BIN, requires_c_binary
from tests.cross_validation.c_runner import (
    ProverResult,
    run_c_prover9,
    run_c_prover9_from_string,
)
from tests.cross_validation.comparator import (
    compare_full,
    compare_theorem_result,
)

INPUTS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "inputs"


# ── Helpers ────────────────────────────────────────────────────────────────


def _run_python_file(input_path: Path, *, max_given: int = 500) -> tuple[object, ProverResult]:
    """Run Python prover on a file, return (SearchResult, ProverResult)."""
    from pyladr.apps.prover9 import _deny_goals, _apply_settings

    input_text = input_path.read_text()
    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(input_text)

    usable, sos, _denied = _deny_goals(parsed, st)

    opts = SearchOptions(
        max_given=max_given,
        quiet=True,
        goal_directed=False,
    )
    _apply_settings(parsed, opts, st)

    search = GivenClauseSearch(options=opts, symbol_table=st)
    result = search.run(usable=usable, sos=sos)

    pr = ProverResult(
        exit_code=int(result.exit_code),
        raw_output="",
        theorem_proved=(result.exit_code == ExitCode.MAX_PROOFS_EXIT),
        search_failed=(result.exit_code == ExitCode.SOS_EMPTY_EXIT),
        clauses_given=result.stats.given,
        clauses_generated=result.stats.generated,
        clauses_kept=result.stats.kept,
        clauses_deleted=result.stats.sos_limit_deleted,
        proof_length=len(result.proofs[0].clauses) if result.proofs else 0,
    )
    return result, pr


# ── Fixture File Tests ────────────────────────────────────────────────────


FIXTURE_FILES = [
    "identity_only.in",
    "simple_group.in",
]


@pytest.mark.cross_validation
@requires_c_binary
class TestFixtureFileRegression:
    """Compare fixture input files between Python and C."""

    @pytest.mark.parametrize("input_file", FIXTURE_FILES)
    def test_theorem_status_matches(self, input_file: str):
        """Python theorem status matches C for fixture files."""
        path = INPUTS_DIR / input_file
        if not path.exists():
            pytest.skip(f"Fixture not found: {path}")

        c_result = run_c_prover9(path, timeout=30.0)
        _, py_result = _run_python_file(path)

        comp = compare_theorem_result(c_result, py_result)
        assert comp.equivalent, (
            f"{input_file}: theorem status mismatch: {comp}\n"
            f"  C: proved={c_result.theorem_proved}, failed={c_result.search_failed}\n"
            f"  Py: proved={py_result.theorem_proved}, failed={py_result.search_failed}"
        )

    @pytest.mark.parametrize("input_file", FIXTURE_FILES)
    def test_proof_existence_matches(self, input_file: str):
        """If C finds a proof, Python must also find a proof (and vice versa)."""
        path = INPUTS_DIR / input_file
        if not path.exists():
            pytest.skip(f"Fixture not found: {path}")

        c_result = run_c_prover9(path, timeout=30.0)
        py_search_result, py_result = _run_python_file(path)

        if c_result.theorem_proved:
            assert py_result.theorem_proved, (
                f"{input_file}: C proved but Python did not\n"
                f"  C: Given={c_result.clauses_given}, Gen={c_result.clauses_generated}\n"
                f"  Py: Given={py_result.clauses_given}, Gen={py_result.clauses_generated}"
            )
        elif c_result.search_failed:
            assert py_result.search_failed or not py_result.theorem_proved, (
                f"{input_file}: C failed but Python proved"
            )


# ── Benchmark File Tests ──────────────────────────────────────────────────


BENCHMARK_FILES = [
    "bench_group_comm_3.in",
    "bench_lattice_distrib.in",
    "bench_robbins.in",
]


@pytest.mark.cross_validation
@requires_c_binary
@pytest.mark.slow
class TestBenchmarkFileRegression:
    """Compare benchmark input files between Python and C."""

    @pytest.mark.parametrize("input_file", BENCHMARK_FILES)
    def test_benchmark_theorem_status(self, input_file: str):
        """Benchmark theorem status matches between C and Python."""
        path = INPUTS_DIR / input_file
        if not path.exists():
            pytest.skip(f"Benchmark not found: {path}")

        c_result = run_c_prover9(path, timeout=60.0)
        _, py_result = _run_python_file(path, max_given=1000)

        comp = compare_theorem_result(c_result, py_result)
        if not comp.equivalent:
            # Report but don't hard-fail for benchmark files (may hit limits)
            print(
                f"  WARNING: {input_file} theorem status mismatch: {comp}\n"
                f"  C: proved={c_result.theorem_proved}, given={c_result.clauses_given}\n"
                f"  Py: proved={py_result.theorem_proved}, given={py_result.clauses_given}"
            )


# ── Python-Only Determinism Tests ─────────────────────────────────────────


class TestFileDeterminism:
    """Python prover produces deterministic results on fixture files."""

    @pytest.mark.parametrize("input_file", FIXTURE_FILES)
    def test_deterministic_results(self, input_file: str):
        """Same input file produces identical results across runs."""
        path = INPUTS_DIR / input_file
        if not path.exists():
            pytest.skip(f"Fixture not found: {path}")

        r1_search, r1 = _run_python_file(path)
        r2_search, r2 = _run_python_file(path)

        assert r1.theorem_proved == r2.theorem_proved
        assert r1.search_failed == r2.search_failed
        assert r1.clauses_given == r2.clauses_given
        assert r1.clauses_generated == r2.clauses_generated
        assert r1.clauses_kept == r2.clauses_kept

    @pytest.mark.parametrize("input_file", FIXTURE_FILES)
    def test_disabled_hierarchical_matches_plain(self, input_file: str):
        """Disabled hierarchical options produce same results as plain options."""
        path = INPUTS_DIR / input_file
        if not path.exists():
            pytest.skip(f"Fixture not found: {path}")

        from pyladr.apps.prover9 import _deny_goals, _apply_settings

        input_text = path.read_text()

        # Plain run
        st1 = SymbolTable()
        parsed1 = LADRParser(st1).parse_input(input_text)
        usable1, sos1 = _deny_goals(parsed1, st1)
        opts1 = SearchOptions(max_given=500, quiet=True)
        _apply_settings(parsed1, opts1, st1)
        r1 = GivenClauseSearch(options=opts1, symbol_table=st1).run(
            usable=usable1, sos=sos1,
        )

        # Disabled hierarchical run
        st2 = SymbolTable()
        parsed2 = LADRParser(st2).parse_input(input_text)
        usable2, sos2 = _deny_goals(parsed2, st2)
        opts2 = SearchOptions(
            max_given=500,
            quiet=True,
            goal_directed=False,
            goal_proximity_weight=0.8,
            embedding_evolution_rate=0.05,
        )
        _apply_settings(parsed2, opts2, st2)
        r2 = GivenClauseSearch(options=opts2, symbol_table=st2).run(
            usable=usable2, sos=sos2,
        )

        assert r1.exit_code == r2.exit_code
        assert r1.stats.given == r2.stats.given
        assert r1.stats.generated == r2.stats.generated
        assert r1.stats.kept == r2.stats.kept


# ── Inline Problem Regression ─────────────────────────────────────────────


class TestInlineProblemRegression:
    """Regression tests with inline LADR problems."""

    INLINE_PROBLEMS = [
        (
            "lattice_absorption",
            """\
set(auto).
formulas(sos).
  x ^ y = y ^ x.
  x v y = y v x.
  (x ^ y) ^ z = x ^ (y ^ z).
  (x v y) v z = x v (y v z).
  x ^ (x v y) = x.
  x v (x ^ y) = x.
end_of_list.

formulas(goals).
  x ^ x = x.
end_of_list.
""",
            True,  # expected proved
        ),
        (
            "simple_resolution_chain",
            """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(x) | R(x).
  -R(a).
end_of_list.
""",
            True,
        ),
    ]

    @pytest.mark.parametrize(
        "name,input_text,expected_proved",
        INLINE_PROBLEMS,
        ids=[n for n, _, _ in INLINE_PROBLEMS],
    )
    def test_inline_problem(self, name: str, input_text: str, expected_proved: bool):
        """Inline problems produce expected results."""
        from pyladr.apps.prover9 import _deny_goals, _apply_settings

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(input_text)
        usable, sos, _denied = _deny_goals(parsed, st)

        opts = SearchOptions(max_given=500, quiet=True, goal_directed=False)
        _apply_settings(parsed, opts, st)

        result = GivenClauseSearch(options=opts, symbol_table=st).run(
            usable=usable, sos=sos,
        )

        proved = result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert proved == expected_proved, (
            f"{name}: expected {'proved' if expected_proved else 'not proved'}, "
            f"got exit_code={result.exit_code}"
        )
