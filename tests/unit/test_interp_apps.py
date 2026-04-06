"""Tests for interpretation-dependent apps (Phase B).

Tests clausefilter, clausetester, interpformat, and isofilter family.
"""

from __future__ import annotations

import io
import tempfile
from unittest.mock import patch

import pytest

from pyladr.apps.clausefilter import FilterMode, filter_clause, main as clausefilter_main
from pyladr.apps.clausetester import main as clausetester_main
from pyladr.apps.interpformat import main as interpformat_main
from pyladr.apps.isofilter import (
    _filter_operations,
    _iso_member,
    _remove_constants,
    main as isofilter_main,
)
from pyladr.apps.isofilter0 import main as isofilter0_main
from pyladr.apps.isofilter2 import main as isofilter2_main
from pyladr.core.clause import Clause, Literal
from pyladr.core.interpretation import (
    Interpretation,
    compile_interp_from_text,
    isomorphic_interps,
    permute_interp,
)
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term


# ── Shared test data ─────────────────────────────────────────────────────────

Z2_TEXT = """interpretation(2, [number=1, seconds=0], [
  function(f(_,_), [0,1,1,0]),
  function(e, [0])
])."""

Z3_TEXT = """interpretation(3, [number=1, seconds=0], [
  function(f(_,_), [0,1,2,1,2,0,2,0,1])
])."""


@pytest.fixture
def interp_file(tmp_path):
    """Write Z2 interpretation to a temp file."""
    p = tmp_path / "interps.txt"
    p.write_text(Z2_TEXT)
    return str(p)


@pytest.fixture
def symbol_table():
    return SymbolTable()


def _make_term(sym_id: int, args: tuple[Term, ...] = ()) -> Term:
    return Term(private_symbol=-sym_id, arity=len(args), args=args)


def _make_var(n: int) -> Term:
    from pyladr.core.term import get_variable_term
    return get_variable_term(n)


def _make_unit_eq(eq_id: int, lhs: Term, rhs: Term) -> Clause:
    atom = Term(private_symbol=-eq_id, arity=2, args=(lhs, rhs))
    lit = Literal(sign=True, atom=atom)
    return Clause(literals=(lit,))


# ── ClauseFilter Tests ───────────────────────────────────────────────────────


class TestClauseFilter:
    def test_filter_true_in_all(self, symbol_table):
        """Commutativity holds in Z2 (XOR is commutative)."""
        interp = compile_interp_from_text(Z2_TEXT)
        eq = symbol_table.str_to_sn("=", 2)
        f = symbol_table.str_to_sn("f", 2)
        x, y = _make_var(0), _make_var(1)
        clause = _make_unit_eq(eq, _make_term(f, (x, y)), _make_term(f, (y, x)))

        assert filter_clause(clause, [interp], FilterMode.TRUE_IN_ALL, symbol_table, eq)

    def test_filter_false_in_some(self, symbol_table):
        """Idempotency f(x,x)=x fails in Z2 (f(1,1)=0 != 1)."""
        interp = compile_interp_from_text(Z2_TEXT)
        eq = symbol_table.str_to_sn("=", 2)
        f = symbol_table.str_to_sn("f", 2)
        x = _make_var(0)
        clause = _make_unit_eq(eq, _make_term(f, (x, x)), x)

        assert filter_clause(clause, [interp], FilterMode.FALSE_IN_SOME, symbol_table, eq)

    def test_main_usage(self):
        assert clausefilter_main([]) == 1

    def test_main_bad_mode(self, interp_file):
        assert clausefilter_main([interp_file, "bad_mode"]) == 1

    def test_main_missing_file(self):
        assert clausefilter_main(["/nonexistent", "true_in_all"]) == 1


# ── ClauseTester Tests ───────────────────────────────────────────────────────


class TestClauseTester:
    def test_main_usage(self):
        assert clausetester_main([]) == 1

    def test_main_missing_file(self):
        assert clausetester_main(["/nonexistent"]) == 1


# ── InterpFormat Tests ───────────────────────────────────────────────────────


class TestInterpFormat:
    def test_format_standard(self):
        with patch("sys.stdin", io.StringIO(Z2_TEXT)):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                result = interpformat_main(["standard"])
        assert result == 0
        output = mock_out.getvalue()
        assert "interpretation" in output
        assert "function" in output

    def test_format_cooked(self):
        with patch("sys.stdin", io.StringIO(Z2_TEXT)):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                result = interpformat_main(["cooked"])
        assert result == 0
        output = mock_out.getvalue()
        assert "f(0,0) = 0." in output

    def test_format_xml(self):
        with patch("sys.stdin", io.StringIO(Z2_TEXT)):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                result = interpformat_main(["xml"])
        assert result == 0
        assert "<interp" in mock_out.getvalue()

    def test_format_tex(self):
        with patch("sys.stdin", io.StringIO(Z2_TEXT)):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                result = interpformat_main(["tex"])
        assert result == 0
        assert "\\begin{tabular}" in mock_out.getvalue()

    def test_format_from_file(self, tmp_path):
        p = tmp_path / "interps.txt"
        p.write_text(Z2_TEXT)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            result = interpformat_main(["cooked", "-f", str(p)])
        assert result == 0
        assert "f(0,0) = 0." in mock_out.getvalue()

    def test_no_interps(self):
        with patch("sys.stdin", io.StringIO("nothing here")):
            result = interpformat_main([])
        assert result == 1

    def test_wrap(self):
        with patch("sys.stdin", io.StringIO(Z2_TEXT)):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                result = interpformat_main(["standard", "wrap"])
        assert result == 0
        output = mock_out.getvalue()
        assert "list(interpretations)." in output
        assert "end_of_list." in output


# ── Isofilter Tests ──────────────────────────────────────────────────────────


class TestIsofilter:
    def test_remove_constants(self):
        interp = compile_interp_from_text(Z2_TEXT)
        assert "e" in interp.operations
        filtered = _remove_constants(interp)
        assert "e" not in filtered.operations
        assert "f" in filtered.operations

    def test_filter_operations_keep(self):
        interp = compile_interp_from_text(Z2_TEXT)
        filtered = _filter_operations(interp, {"f"}, keep=True)
        assert "f" in filtered.operations
        assert "e" not in filtered.operations

    def test_filter_operations_remove(self):
        interp = compile_interp_from_text(Z2_TEXT)
        filtered = _filter_operations(interp, {"e"}, keep=False)
        assert "f" in filtered.operations
        assert "e" not in filtered.operations

    def test_iso_member_finds_isomorphic(self):
        interp = compile_interp_from_text(Z2_TEXT)
        perm = [1, 0]
        permuted = permute_interp(interp, perm)
        assert _iso_member(permuted, [interp])

    def test_iso_member_rejects_different(self):
        z2 = compile_interp_from_text(Z2_TEXT)
        z3 = compile_interp_from_text(Z3_TEXT)
        assert not _iso_member(z3, [z2])

    def test_isofilter_removes_duplicates(self):
        """Two copies of Z2 should yield one output."""
        two_z2 = Z2_TEXT + "\n" + Z2_TEXT
        with patch("sys.stdin", io.StringIO(two_z2)):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                isofilter_main([])
        output = mock_out.getvalue()
        assert output.count("interpretation(") == 1

    def test_isofilter_keeps_non_isomorphic(self):
        """Z2 and Z3 are not isomorphic, both should be kept."""
        both = Z2_TEXT + "\n" + Z3_TEXT
        with patch("sys.stdin", io.StringIO(both)):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                isofilter_main([])
        output = mock_out.getvalue()
        assert output.count("interpretation(") == 2

    def test_isofilter_wrap(self):
        with patch("sys.stdin", io.StringIO(Z2_TEXT)):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                isofilter_main(["wrap"])
        output = mock_out.getvalue()
        assert "list(interpretations)." in output
        assert "end_of_list." in output


class TestIsofilter0:
    def test_removes_duplicates(self):
        two_z2 = Z2_TEXT + "\n" + Z2_TEXT
        with patch("sys.stdin", io.StringIO(two_z2)):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                isofilter0_main([])
        assert mock_out.getvalue().count("interpretation(") == 1


class TestIsofilter2:
    def test_removes_identical(self):
        two_z2 = Z2_TEXT + "\n" + Z2_TEXT
        with patch("sys.stdin", io.StringIO(two_z2)):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                isofilter2_main([])
        assert mock_out.getvalue().count("interpretation(") == 1

    def test_keeps_non_identical(self):
        both = Z2_TEXT + "\n" + Z3_TEXT
        with patch("sys.stdin", io.StringIO(both)):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                isofilter2_main([])
        assert mock_out.getvalue().count("interpretation(") == 2
