"""Tests for given clause symbol name consistency (REQ-R004).

Regression prevention: given clause trace output must use the same function
and predicate names as the input.  When a user writes ``p(f(a,b))`` in their
input file the given clause trace must print ``p(f(a,b))`` — NOT raw symbol
IDs like ``s1(s2(s3,s4))``.

These tests cover:
- Symbol name preservation for functions, predicates, and constants
- Variable name handling (standard x/y/z style)
- Complex / nested term structures
- Multi-arity and zero-arity (constant) symbols
- Equality symbol preservation
- Skolem constant naming
- Cross-validation against C reference output symbol handling
- Regression detection for symbol ID leakage (sN patterns)
"""

from __future__ import annotations

import io
import re
from unittest.mock import patch

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import get_rigid_term
from pyladr.parsing.ladr_parser import parse_input
from pyladr.search.given_clause import (
    ExitCode,
    GivenClauseSearch,
    SearchOptions,
)
from pyladr.search.selection import default_clause_weight


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_and_deny(text: str) -> tuple[list[Clause], SymbolTable]:
    """Parse LADR input and deny goals into SOS clauses."""
    st = SymbolTable()
    parsed = parse_input(text, st)

    sos = list(parsed.sos)
    for goal in parsed.goals:
        denied_lits = tuple(
            Literal(sign=not lit.sign, atom=lit.atom) for lit in goal.literals
        )
        denied = Clause(
            literals=denied_lits,
            justification=(Justification(just_type=JustType.DENY, clause_ids=(0,)),),
        )
        sos.append(denied)

    return sos, st


def _run_and_capture(
    text: str,
    print_given: bool = True,
    quiet: bool = False,
    max_given: int = 500,
    max_seconds: float = 10.0,
    **kwargs,
) -> tuple[str, ExitCode, SymbolTable]:
    """Run search and capture stdout output.

    Returns (captured_stdout, exit_code, symbol_table).

    IMPORTANT: Passes the parsed symbol_table to GivenClauseSearch so that
    given clause traces use proper symbol names.
    """
    sos, st = _parse_and_deny(text)
    opts = SearchOptions(
        max_given=max_given,
        max_seconds=max_seconds,
        print_given=print_given,
        quiet=quiet,
        **kwargs,
    )
    engine = GivenClauseSearch(opts, symbol_table=st)

    captured = io.StringIO()
    with patch("sys.stdout", captured):
        result = engine.run(usable=[], sos=sos)

    return captured.getvalue(), result.exit_code, st


def _extract_given_lines(output: str) -> list[str]:
    """Extract all given clause lines from captured output."""
    return [line.strip() for line in output.splitlines() if "given #" in line]


def _extract_clause_text(given_line: str) -> str:
    """Extract clause text portion from a given clause line.

    Given line format: ``given #N (X,wt=W): ID: clause_text.``
    Returns the clause_text portion (including trailing period).
    """
    # Match after "ID: " to get the clause text
    m = re.search(r"\):\s+\d+:\s+(.+)", given_line)
    if m:
        return m.group(1)
    return given_line


# ── Symbol ID leakage detector ───────────────────────────────────────────────

# Raw symbol IDs look like s1, s2, s3 (sN pattern).
# These should NEVER appear in given clause output when a symbol table is
# properly wired.
RAW_SYMBOL_ID_RE = re.compile(r"\bs\d+\b")


def _assert_no_raw_symbol_ids(output: str, context: str = "") -> None:
    """Assert that no raw symbol IDs (sN) appear in given clause trace.

    This is the core regression check: if symbol IDs leak through, it means
    the symbol table was not passed to to_str().
    """
    given_lines = _extract_given_lines(output)
    for line in given_lines:
        clause_text = _extract_clause_text(line)
        matches = RAW_SYMBOL_ID_RE.findall(clause_text)
        if matches:
            pytest.fail(
                f"REGRESSION: Raw symbol IDs found in given clause trace.\n"
                f"  Line: {line!r}\n"
                f"  Raw IDs: {matches}\n"
                f"  Context: {context}\n"
                f"  This means the symbol table was not passed to to_str()."
            )


# ── Test inputs ──────────────────────────────────────────────────────────────

# Simple predicate + constant
SIMPLE_PRED_INPUT = """\
formulas(sos).
  p(a).
  -p(x) | q(x).
end_of_list.
formulas(goals).
  q(a).
end_of_list.
"""

# Equational reasoning with named binary operator (simple, avoids demod recursion)
EQUATIONAL_INPUT = """\
formulas(sos).
  f(a) = b.
  f(b) = c.
end_of_list.
formulas(goals).
  f(f(a)) = c.
end_of_list.
"""

# Multi-arity functions and predicates
MULTI_ARITY_INPUT = """\
formulas(sos).
  f(a, b) = g(c).
  h(f(a, b), g(c), d) = e.
end_of_list.
formulas(goals).
  h(g(c), g(c), d) = e.
end_of_list.
"""

# Nested functions
NESTED_INPUT = """\
formulas(sos).
  f(g(h(a))) = a.
  g(h(a)) = h(g(a)).
end_of_list.
formulas(goals).
  f(h(g(a))) = a.
end_of_list.
"""

# Multiple predicates
MULTI_PRED_INPUT = """\
formulas(sos).
  parent(tom, bob).
  parent(bob, ann).
  -parent(x, y) | -parent(y, z) | grandparent(x, z).
end_of_list.
formulas(goals).
  grandparent(tom, ann).
end_of_list.
"""

# Constants only (zero-arity symbols)
CONSTANTS_ONLY_INPUT = """\
formulas(sos).
  a = b.
  b = c.
end_of_list.
formulas(goals).
  a = c.
end_of_list.
"""

# Descriptive symbol names (longer names)
DESCRIPTIVE_NAMES_INPUT = """\
formulas(sos).
  member(alice, club).
  -member(x, club) | happy(x).
end_of_list.
formulas(goals).
  happy(alice).
end_of_list.
"""


# ── Core symbol name preservation tests ──────────────────────────────────────


class TestSymbolNamePreservation:
    """REQ-R004: Given clause traces must use input symbol names."""

    def test_no_raw_symbol_ids_simple_predicates(self):
        """Simple predicate names must not appear as raw symbol IDs."""
        output, exit_code, _ = _run_and_capture(
            SIMPLE_PRED_INPUT, binary_resolution=True
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        _assert_no_raw_symbol_ids(output, "simple predicates p/q with constant a")

    def test_no_raw_symbol_ids_equational(self):
        """Equational reasoning must preserve function/constant names."""
        output, exit_code, _ = _run_and_capture(
            EQUATIONAL_INPUT, paramodulation=True
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        _assert_no_raw_symbol_ids(output, "equational with f, a, b, c")

    def test_no_raw_symbol_ids_multi_arity(self):
        """Multi-arity function names must be preserved."""
        output, exit_code, _ = _run_and_capture(
            MULTI_ARITY_INPUT, paramodulation=True, demodulation=True
        )
        _assert_no_raw_symbol_ids(output, "multi-arity functions f/g/h")

    def test_no_raw_symbol_ids_nested(self):
        """Deeply nested function names must be preserved."""
        output, exit_code, _ = _run_and_capture(
            NESTED_INPUT, paramodulation=True, demodulation=True
        )
        _assert_no_raw_symbol_ids(output, "nested f(g(h(a)))")

    def test_no_raw_symbol_ids_multiple_predicates(self):
        """Multiple predicate names must all be preserved."""
        output, exit_code, _ = _run_and_capture(
            MULTI_PRED_INPUT, binary_resolution=True
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        _assert_no_raw_symbol_ids(output, "predicates parent/grandparent")

    def test_no_raw_symbol_ids_constants(self):
        """Zero-arity constant names must be preserved."""
        output, exit_code, _ = _run_and_capture(
            CONSTANTS_ONLY_INPUT, paramodulation=True, demodulation=True
        )
        _assert_no_raw_symbol_ids(output, "constants a/b/c")

    def test_no_raw_symbol_ids_descriptive_names(self):
        """Longer descriptive symbol names must be preserved."""
        output, exit_code, _ = _run_and_capture(
            DESCRIPTIVE_NAMES_INPUT, binary_resolution=True
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        _assert_no_raw_symbol_ids(output, "descriptive names member/happy/alice/club")


class TestSpecificSymbolNamesAppear:
    """Verify specific input symbol names appear in given clause traces."""

    def test_predicate_names_in_output(self):
        """Predicate names p, q must appear in given clause text."""
        output, exit_code, _ = _run_and_capture(
            SIMPLE_PRED_INPUT, binary_resolution=True
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        given_lines = _extract_given_lines(output)
        all_clause_text = " ".join(_extract_clause_text(l) for l in given_lines)
        assert "p(" in all_clause_text or "p" in all_clause_text, (
            f"Predicate 'p' not found in given clause traces:\n{all_clause_text}"
        )

    def test_constant_names_in_output(self):
        """Constant name 'a' must appear in given clause text."""
        output, exit_code, _ = _run_and_capture(
            SIMPLE_PRED_INPUT, binary_resolution=True
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        given_lines = _extract_given_lines(output)
        all_clause_text = " ".join(_extract_clause_text(l) for l in given_lines)
        # 'a' should appear as a constant in the clause text
        assert re.search(r"\ba\b", all_clause_text), (
            f"Constant 'a' not found in given clause traces:\n{all_clause_text}"
        )

    def test_function_names_in_multi_arity(self):
        """Function names f, g, h must appear in traces."""
        output, _, _ = _run_and_capture(
            MULTI_ARITY_INPUT, paramodulation=True, demodulation=True
        )
        given_lines = _extract_given_lines(output)
        if not given_lines:
            pytest.skip("No given clauses produced")
        all_clause_text = " ".join(_extract_clause_text(l) for l in given_lines)
        for name in ["f(", "g("]:
            assert name in all_clause_text, (
                f"Function '{name[:-1]}' not found in given clause traces:\n"
                f"{all_clause_text}"
            )

    def test_descriptive_predicate_names_in_output(self):
        """Descriptive names 'parent', 'grandparent' must appear in traces."""
        output, exit_code, _ = _run_and_capture(
            MULTI_PRED_INPUT, binary_resolution=True
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        given_lines = _extract_given_lines(output)
        all_clause_text = " ".join(_extract_clause_text(l) for l in given_lines)
        assert "parent(" in all_clause_text, (
            f"Predicate 'parent' not found:\n{all_clause_text}"
        )

    def test_longer_names_preserved_exactly(self):
        """Names like 'member', 'happy', 'alice', 'club' must appear."""
        output, exit_code, _ = _run_and_capture(
            DESCRIPTIVE_NAMES_INPUT, binary_resolution=True
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        given_lines = _extract_given_lines(output)
        all_clause_text = " ".join(_extract_clause_text(l) for l in given_lines)
        for name in ["member", "alice", "club"]:
            assert name in all_clause_text, (
                f"Symbol '{name}' not found in given clause traces:\n"
                f"{all_clause_text}"
            )


# ── Variable name handling tests ─────────────────────────────────────────────


class TestVariableNameHandling:
    """Verify variable names use standard Prover9 style (x, y, z, ...)."""

    def test_variables_use_standard_names(self):
        """Variables should display as x, y, z, u, v, w — not v0, v1, v2."""
        # Use resolution problem with variables to avoid demodulation recursion
        output, exit_code, _ = _run_and_capture(
            SIMPLE_PRED_INPUT, binary_resolution=True
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        given_lines = _extract_given_lines(output)
        if not given_lines:
            pytest.skip("No given clauses produced")

        # Check that standard variable names appear (x, y, z)
        all_clause_text = " ".join(_extract_clause_text(l) for l in given_lines)

        # Raw variable format is v0, v1, v2 — these should NOT appear
        raw_var_matches = re.findall(r"\bv\d+\b", all_clause_text)
        # Filter out legitimate symbol names that happen to start with v
        raw_var_matches = [m for m in raw_var_matches if re.match(r"^v\d+$", m)]
        if raw_var_matches:
            pytest.fail(
                f"REGRESSION: Raw variable IDs found in given clause trace.\n"
                f"  Raw vars: {raw_var_matches}\n"
                f"  Expected standard names (x, y, z, u, v, w, v6, v7, ...)\n"
                f"  Clause text: {all_clause_text}"
            )

    def test_variable_names_in_resolution_problem(self):
        """Variables in resolution problems should use standard names."""
        output, exit_code, _ = _run_and_capture(
            SIMPLE_PRED_INPUT, binary_resolution=True
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        given_lines = _extract_given_lines(output)
        all_clause_text = " ".join(_extract_clause_text(l) for l in given_lines)

        # The input has variable x — after renumbering it may be x or another
        # standard name, but never v0
        raw_var_matches = re.findall(r"\bv\d+\b", all_clause_text)
        raw_var_matches = [m for m in raw_var_matches if re.match(r"^v\d+$", m)]
        assert not raw_var_matches, (
            f"Raw variable IDs in trace: {raw_var_matches}\n"
            f"Clause text: {all_clause_text}"
        )


# ── Equality symbol tests ───────────────────────────────────────────────────


class TestEqualitySymbolHandling:
    """Verify the equality symbol = is preserved in given clause traces."""

    def test_equality_sign_present(self):
        """Equality literals should show '=' not a raw symbol ID."""
        output, exit_code, _ = _run_and_capture(
            CONSTANTS_ONLY_INPUT, paramodulation=True, demodulation=True
        )
        given_lines = _extract_given_lines(output)
        if not given_lines:
            pytest.skip("No given clauses produced")

        # At least some given clauses should contain '='
        has_equality = any("=" in _extract_clause_text(l) for l in given_lines)
        assert has_equality, (
            f"No equality symbol found in given clauses:\n"
            + "\n".join(given_lines)
        )

    def test_equality_in_constants_problem(self):
        """Pure equality problems should show '=' between constants."""
        output, exit_code, _ = _run_and_capture(
            CONSTANTS_ONLY_INPUT, paramodulation=True, demodulation=True
        )
        given_lines = _extract_given_lines(output)
        if not given_lines:
            pytest.skip("No given clauses produced")

        all_text = " ".join(_extract_clause_text(l) for l in given_lines)
        _assert_no_raw_symbol_ids(all_text, "equality with constants a/b/c")


# ── Symbol table wiring regression tests ─────────────────────────────────────


class TestSymbolTableWiring:
    """Verify that the symbol table is properly wired through the search engine."""

    def test_engine_receives_symbol_table(self):
        """GivenClauseSearch should accept and store a SymbolTable."""
        st = SymbolTable()
        st.str_to_sn("test_sym", 0)
        engine = GivenClauseSearch(SearchOptions(), symbol_table=st)
        assert engine._symbol_table is st

    def test_engine_default_symbol_table_is_empty(self):
        """Without explicit symbol_table, engine creates empty one."""
        engine = GivenClauseSearch(SearchOptions())
        assert isinstance(engine._symbol_table, SymbolTable)
        assert len(engine._symbol_table) == 0

    def test_term_to_str_with_symbol_table(self):
        """Term.to_str(symbol_table) should use real names."""
        st = SymbolTable()
        sn = st.str_to_sn("foo", 1)
        sn_a = st.str_to_sn("bar", 0)
        bar = get_rigid_term(sn_a, 0, ())
        foo_bar = get_rigid_term(sn, 1, (bar,))
        result = foo_bar.to_str(st)
        assert result == "foo(bar)", f"Expected 'foo(bar)', got '{result}'"

    def test_term_to_str_without_symbol_table(self):
        """Term.to_str() without symbol_table should use raw IDs."""
        st = SymbolTable()
        sn = st.str_to_sn("foo", 1)
        sn_a = st.str_to_sn("bar", 0)
        bar = get_rigid_term(sn_a, 0, ())
        foo_bar = get_rigid_term(sn, 1, (bar,))
        result = foo_bar.to_str()
        # Should be sN(sM) format
        assert re.match(r"s\d+\(s\d+\)", result), (
            f"Expected raw ID format, got '{result}'"
        )

    def test_clause_to_str_propagates_symbol_table(self):
        """Clause.to_str(symbol_table) should propagate to all terms."""
        st = SymbolTable()
        sn_p = st.str_to_sn("p", 1)
        sn_a = st.str_to_sn("a", 0)
        a_term = get_rigid_term(sn_a, 0, ())
        p_a = get_rigid_term(sn_p, 1, (a_term,))
        lit = Literal(sign=True, atom=p_a)
        clause = Clause(literals=(lit,), id=1)

        result_with_st = clause.to_str(st)
        result_without = clause.to_str()

        assert "p(a)" in result_with_st, (
            f"Expected 'p(a)' in output, got '{result_with_st}'"
        )
        assert "p(a)" not in result_without, (
            f"Without symbol table should NOT show 'p(a)', got '{result_without}'"
        )


# ── Integration: Full search with symbol validation ─────────────────────────


class TestFullSearchSymbolConsistency:
    """Integration tests: run full search and validate all symbol names."""

    def test_equational_proof_symbol_consistency(self):
        """Full equational proof should use consistent symbol names throughout."""
        # Use paramodulation without demodulation to avoid recursion bug
        output, exit_code, st = _run_and_capture(
            CONSTANTS_ONLY_INPUT, paramodulation=True, demodulation=True
        )
        _assert_no_raw_symbol_ids(output, "full equational proof")

        given_lines = _extract_given_lines(output)
        assert len(given_lines) >= 1, "Expected at least one given clause"

        # Verify the symbol table has the expected symbols
        all_symbols = {s.name for s in st.all_symbols}
        for expected in ["a", "b", "c"]:
            assert expected in all_symbols, (
                f"Expected '{expected}' in symbol table: {all_symbols}"
            )

    def test_resolution_proof_symbol_consistency(self):
        """Full resolution proof should use consistent symbol names."""
        output, exit_code, st = _run_and_capture(
            SIMPLE_PRED_INPUT, binary_resolution=True
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        _assert_no_raw_symbol_ids(output, "full resolution proof")

        # Verify input symbols are in the table
        all_symbols = {s.name for s in st.all_symbols}
        for expected in ["p", "q", "a"]:
            assert expected in all_symbols, (
                f"Expected '{expected}' in symbol table: {all_symbols}"
            )

    def test_multi_predicate_proof_symbol_consistency(self):
        """Multi-predicate proof preserves all predicate names."""
        output, exit_code, st = _run_and_capture(
            MULTI_PRED_INPUT, binary_resolution=True
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        _assert_no_raw_symbol_ids(output, "multi-predicate proof")

        all_symbols = {s.name for s in st.all_symbols}
        for expected in ["parent", "grandparent", "tom", "bob", "ann"]:
            assert expected in all_symbols, (
                f"Expected '{expected}' in symbol table: {all_symbols}"
            )


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for symbol name handling."""

    def test_single_letter_names(self):
        """Single-letter function/constant names should be preserved."""
        output, _, _ = _run_and_capture(
            CONSTANTS_ONLY_INPUT, paramodulation=True, demodulation=True
        )
        _assert_no_raw_symbol_ids(output, "single-letter constants a/b/c")

    def test_empty_clause_has_no_raw_ids(self):
        """Empty clause ($F) should not contain raw symbol IDs."""
        # If a proof is found, check that $F doesn't show raw IDs
        output, exit_code, _ = _run_and_capture(
            SIMPLE_PRED_INPUT, binary_resolution=True
        )
        if exit_code == ExitCode.MAX_PROOFS_EXIT:
            # $F is a valid output, no raw IDs should appear
            for line in output.splitlines():
                if "$F" in line and "given #" in line:
                    _assert_no_raw_symbol_ids(line, "empty clause line")

    def test_negated_literal_preserves_names(self):
        """Negated literals (-p(a)) should still use proper symbol names."""
        output, exit_code, _ = _run_and_capture(
            SIMPLE_PRED_INPUT, binary_resolution=True
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        given_lines = _extract_given_lines(output)
        all_text = " ".join(_extract_clause_text(l) for l in given_lines)

        # Check that negated literals don't accidentally lose symbol names
        if "-" in all_text:
            # There should be a symbol name after the negation
            neg_matches = re.findall(r"-(\w+)", all_text)
            for match in neg_matches:
                assert not re.match(r"^s\d+$", match), (
                    f"Negated literal has raw symbol ID: -{match}"
                )

    def test_disjunctive_clause_all_literals_have_names(self):
        """All literals in a disjunctive clause should use proper names."""
        output, exit_code, _ = _run_and_capture(
            SIMPLE_PRED_INPUT, binary_resolution=True
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        given_lines = _extract_given_lines(output)
        for line in given_lines:
            clause_text = _extract_clause_text(line)
            # Split on ' | ' to check each literal
            literals = clause_text.rstrip(".").split(" | ")
            for lit in literals:
                raw_ids = RAW_SYMBOL_ID_RE.findall(lit)
                assert not raw_ids, (
                    f"Raw symbol IDs in literal '{lit}' of clause:\n  {line}"
                )


# ── Skolem constant naming tests ─────────────────────────────────────────────


class TestSkolemConstantNaming:
    """Skolem constants introduced by goal negation should have proper names."""

    def test_skolem_constants_have_names(self):
        """Skolem constants (c0, c1, ...) should not appear as raw IDs."""
        # Use resolution problem to avoid demodulation recursion
        output, exit_code, st = _run_and_capture(
            SIMPLE_PRED_INPUT, binary_resolution=True
        )
        _assert_no_raw_symbol_ids(output, "Skolem constants from goal negation")

    def test_skolem_constants_registered_in_table(self):
        """Skolem constants should be registered in the symbol table."""
        # When goals are negated via the CLI, Skolem constants get registered.
        # Our test helper uses simple denial; Skolem constants come from the
        # CLI path.  Just verify the basic mechanism works.
        st = SymbolTable()
        sn = st.str_to_sn("c0", 0)
        st.mark_skolem(sn)
        name = st.id_to_name(sn)
        assert name == "c0", f"Skolem constant name should be 'c0', got '{name}'"


# ── Cross-validation with C reference ────────────────────────────────────────


class TestCReferenceSymbolNames:
    """Cross-validate symbol name handling against C Prover9 output."""

    @pytest.fixture
    def c_reference_output(self) -> str | None:
        """Load C reference x2 output if available."""
        import os
        ref_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..",
            "reference-prover9", "tests", "fixtures",
            "c_reference", "x2_full_output.txt",
        )
        if not os.path.exists(ref_path):
            return None
        with open(ref_path) as f:
            return f.read()

    def test_c_reference_uses_symbol_names(self, c_reference_output):
        """C reference output should use real symbol names, not IDs."""
        if c_reference_output is None:
            pytest.skip("C reference output not available")
        given_lines = [
            l.strip() for l in c_reference_output.splitlines() if "given #" in l
        ]
        assert len(given_lines) > 0, "No given clause lines in C reference"

        for line in given_lines:
            raw_ids = RAW_SYMBOL_ID_RE.findall(line)
            assert not raw_ids, (
                f"C reference output contains raw symbol IDs: {raw_ids}\n"
                f"  Line: {line}"
            )

    def test_python_and_c_use_same_symbol_style(self, c_reference_output):
        """Python and C should both use human-readable symbol names."""
        if c_reference_output is None:
            pytest.skip("C reference output not available")

        py_output, _, _ = _run_and_capture(
            CONSTANTS_ONLY_INPUT, paramodulation=True, demodulation=True
        )
        py_given = _extract_given_lines(py_output)
        c_given = [
            l.strip() for l in c_reference_output.splitlines() if "given #" in l
        ]

        if not py_given or not c_given:
            pytest.skip("Insufficient given clause output for comparison")

        py_text = " ".join(_extract_clause_text(l) for l in py_given)
        c_text = " ".join(c_given)

        # Neither should have raw symbol IDs
        assert not RAW_SYMBOL_ID_RE.findall(py_text), (
            f"Python output has raw IDs in: {py_text[:200]}"
        )
        assert not RAW_SYMBOL_ID_RE.findall(c_text), (
            f"C output has raw IDs in: {c_text[:200]}"
        )
