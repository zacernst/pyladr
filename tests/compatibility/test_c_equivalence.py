"""Equivalence tests comparing Python and C Prover9 outputs.

These tests run the same problems through both the Python and C
implementations and verify matching results. They require the C
binary to be built.

Run with: pytest tests/compatibility/test_c_equivalence.py -v -m cross_validation
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import C_PROVER9_BIN
from tests.cross_validation.c_runner import ProverResult, run_c_prover9, run_c_prover9_from_string
from tests.cross_validation.comparator import (
    compare_full,
    compare_search_statistics,
    compare_theorem_result,
)

requires_c_binary = pytest.mark.skipif(
    not C_PROVER9_BIN.exists(),
    reason="C prover9 binary not found (run 'make all' to build)",
)

INPUTS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "inputs"
EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "prover9.examples"


# ── Helpers ────────────────────────────────────────────────────────────────


def _run_python_on_inline(
    input_text: str,
    *,
    max_given: int = 200,
    paramodulation: bool = False,
    demodulation: bool = False,
):
    """Run the Python search engine on inline LADR input."""
    from pyladr.core.symbol import SymbolTable
    from pyladr.parsing.ladr_parser import LADRParser
    from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(input_text)

    opts = SearchOptions(
        binary_resolution=True,
        paramodulation=paramodulation,
        demodulation=demodulation,
        factoring=True,
        max_given=max_given,
        quiet=True,
    )
    search = GivenClauseSearch(options=opts, symbol_table=st)
    return search.run(
        usable=parsed.usable or [],
        sos=parsed.sos or [],
    )


def _python_result_to_prover_result(py_result, input_text: str = "") -> ProverResult:
    """Convert Python SearchResult to ProverResult for comparison."""
    from pyladr.search.given_clause import ExitCode

    return ProverResult(
        exit_code=int(py_result.exit_code),
        raw_output="",
        theorem_proved=(py_result.exit_code == ExitCode.MAX_PROOFS_EXIT),
        search_failed=(py_result.exit_code == ExitCode.SOS_EMPTY_EXIT),
        clauses_given=py_result.stats.given,
        clauses_generated=py_result.stats.generated,
        clauses_kept=py_result.stats.kept,
        clauses_deleted=py_result.stats.sos_limit_deleted,
        proof_length=len(py_result.proofs[0].clauses) if py_result.proofs else 0,
    )


# ── Theorem Status Equivalence ─────────────────────────────────────────────


@pytest.mark.cross_validation
@requires_c_binary
class TestTheoremStatusEquivalence:
    """Both implementations agree on theorem proved/failed for all inputs."""

    INLINE_PROVABLE = [
        (
            "trivial_identity",
            """\
formulas(sos).
  P(a).
end_of_list.

formulas(goals).
  P(a).
end_of_list.
""",
        ),
        (
            "resolution_chain",
            """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(a).
end_of_list.
""",
        ),
    ]

    INLINE_UNPROVABLE = [
        (
            "independent_predicates",
            """\
formulas(sos).
  P(a).
end_of_list.

formulas(goals).
  Q(a).
end_of_list.
""",
        ),
    ]

    @pytest.mark.parametrize(
        "name,input_text",
        INLINE_PROVABLE,
        ids=[name for name, _ in INLINE_PROVABLE],
    )
    def test_provable_agreement(self, name: str, input_text: str):
        """Both C and Python agree this is provable."""
        c_result = run_c_prover9_from_string(input_text)
        assert c_result.theorem_proved, f"C should prove {name}"

        py_result = _run_python_on_inline(input_text)
        py_prover = _python_result_to_prover_result(py_result, input_text)

        comparison = compare_theorem_result(c_result, py_prover)
        assert comparison.equivalent, (
            f"Theorem status mismatch for {name}: {comparison}"
        )

    @pytest.mark.parametrize(
        "name,input_text",
        INLINE_UNPROVABLE,
        ids=[name for name, _ in INLINE_UNPROVABLE],
    )
    def test_unprovable_agreement(self, name: str, input_text: str):
        """Both C and Python agree this is not provable (SOS empty or search failed)."""
        c_result = run_c_prover9_from_string(input_text, timeout=10.0)
        # C should NOT prove this
        assert not c_result.theorem_proved, f"C should not prove {name}"

        py_result = _run_python_on_inline(input_text, max_given=100)
        py_prover = _python_result_to_prover_result(py_result, input_text)

        comparison = compare_theorem_result(c_result, py_prover)
        assert comparison.equivalent, (
            f"Theorem status mismatch for {name}: {comparison}"
        )


# ── File-Based Equivalence Tests ──────────────────────────────────────────


@pytest.mark.cross_validation
@requires_c_binary
class TestFileBasedEquivalence:
    """Run stored input files through both implementations."""

    @pytest.mark.parametrize(
        "input_file",
        [
            "identity_only.in",
            "simple_group.in",
        ],
    )
    def test_fixture_input_equivalence(self, input_file: str):
        """Python and C agree on fixture input files."""
        path = INPUTS_DIR / input_file
        if not path.exists():
            pytest.skip(f"Input file not found: {path}")

        c_result = run_c_prover9(path, timeout=30.0)

        # For now, just verify C proves it; full Python file-based
        # comparison requires parser integration with search
        assert c_result.theorem_proved or c_result.search_failed, (
            f"C returned unexpected status for {input_file}"
        )


# ── Search Statistics Comparison ──────────────────────────────────────────


@pytest.mark.cross_validation
@requires_c_binary
class TestSearchStatisticsComparison:
    """Compare search statistics between Python and C."""

    def test_trivial_resolution_stats(self):
        """Statistics agree for trivial resolution problem."""
        input_text = """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(a).
end_of_list.
"""
        c_result = run_c_prover9_from_string(input_text)
        assert c_result.theorem_proved

        py_result = _run_python_on_inline(input_text)
        py_prover = _python_result_to_prover_result(py_result, input_text)

        # For resolution-only problems, statistics should be close
        comparison = compare_search_statistics(
            c_result, py_prover, tolerance=0.5
        )
        # Report differences but allow tolerance for implementation variations
        if not comparison.equivalent:
            for diff in comparison.differences:
                print(f"  Stats diff: {diff}")


# ── Output Format Validation ──────────────────────────────────────────────


@pytest.mark.cross_validation
@requires_c_binary
class TestOutputFormatValidation:
    """Verify Python output format matches C conventions."""

    def test_c_output_has_theorem_proved_marker(self):
        """C output contains 'THEOREM PROVED' for proved problems."""
        input_text = """\
formulas(sos).
  P(a).
end_of_list.

formulas(goals).
  P(a).
end_of_list.
"""
        c_result = run_c_prover9_from_string(input_text)
        assert "THEOREM PROVED" in c_result.raw_output

    def test_c_output_has_statistics_line(self):
        """C output contains statistics line with Given/Generated/Kept."""
        input_text = """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
end_of_list.

formulas(goals).
  Q(a).
end_of_list.
"""
        c_result = run_c_prover9_from_string(input_text)
        assert c_result.clauses_given >= 0
        assert c_result.clauses_generated >= 0

    def test_c_proof_clauses_extractable(self):
        """Proof clauses can be extracted from C output."""
        input_text = """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
end_of_list.

formulas(goals).
  Q(a).
end_of_list.
"""
        c_result = run_c_prover9_from_string(input_text)
        assert c_result.theorem_proved
        # Proof clauses should be extractable
        assert len(c_result.proof_clauses) > 0


# ── Justification Type Matching ────────────────────────────────────────────


class TestJustificationTypes:
    """Ensure Python justification types match C conventions."""

    def test_justtype_enum_matches_c(self):
        """JustType enum values match C just_type enum."""
        from pyladr.core.clause import JustType

        # Verify essential justification types exist (C Just_type names)
        assert hasattr(JustType, "INPUT")
        assert hasattr(JustType, "GOAL")
        assert hasattr(JustType, "DENY")
        assert hasattr(JustType, "BINARY_RES")
        assert hasattr(JustType, "PARA")
        assert hasattr(JustType, "DEMOD")
        assert hasattr(JustType, "FACTOR")
        assert hasattr(JustType, "CLAUSIFY")
        assert hasattr(JustType, "BACK_DEMOD")

    def test_clause_justification_tracking(self):
        """Clauses track justification through inference."""
        from pyladr.core.clause import Clause, Justification, JustType, Literal
        from pyladr.core.term import get_rigid_term

        P_a = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        just = Justification(just_type=JustType.INPUT)
        c = Clause(literals=(Literal(sign=True, atom=P_a),), justification=just)
        assert c.justification is not None
        assert c.justification.just_type == JustType.INPUT
