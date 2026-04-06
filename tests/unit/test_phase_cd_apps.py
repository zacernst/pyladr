"""Tests for Phase C/D apps: complex, ladr_to_tptp, rewriter stubs."""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from pyladr.apps.complex import _term_size, complex4, main as complex_main
from pyladr.apps.ladr_to_tptp import main as ladr_to_tptp_main
from pyladr.apps.rewriter import main as rewriter_main
from pyladr.apps.rewriter2 import main as rewriter2_main
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, get_variable_term


def _make_term(sym_id: int, args: tuple[Term, ...] = ()) -> Term:
    return Term(private_symbol=-sym_id, arity=len(args), args=args)


# ── Complex Tests ────────────────────────────────────────────────────────────


class TestComplex:
    def test_term_size_constant(self):
        st = SymbolTable()
        a = st.str_to_sn("a", 0)
        assert _term_size(_make_term(a)) == 1

    def test_term_size_binary(self):
        st = SymbolTable()
        f = st.str_to_sn("f", 2)
        a = st.str_to_sn("a", 0)
        b = st.str_to_sn("b", 0)
        t = _make_term(f, (_make_term(a), _make_term(b)))
        assert _term_size(t) == 3

    def test_complex4_constant(self):
        st = SymbolTable()
        a = st.str_to_sn("a", 0)
        assert complex4(_make_term(a)) == 0.0

    def test_complex4_positive(self):
        """A term with structure should have positive complexity."""
        st = SymbolTable()
        f = st.str_to_sn("f", 2)
        x = get_variable_term(0)
        # f(x, f(x, x)) has overlapping subterms
        inner = _make_term(f, (x, x))
        t = _make_term(f, (x, inner))
        score = complex4(t)
        assert score > 0.0


# ── LADR to TPTP Tests ──────────────────────────────────────────────────────


class TestLadrToTptp:
    def test_basic_conversion(self):
        ladr_input = (
            "formulas(sos).\n"
            "x = x.\n"
            "f(x,y) = f(y,x).\n"
            "end_of_list.\n"
        )
        with patch("sys.stdin", io.StringIO(ladr_input)):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                result = ladr_to_tptp_main([])
        assert result == 0
        output = mock_out.getvalue()
        assert "fof(f1,axiom" in output
        assert "fof(f2,axiom" in output

    def test_goal_becomes_conjecture(self):
        ladr_input = (
            "formulas(goals).\n"
            "f(x) = x.\n"
            "end_of_list.\n"
        )
        with patch("sys.stdin", io.StringIO(ladr_input)):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                result = ladr_to_tptp_main([])
        assert result == 0
        assert "conjecture" in mock_out.getvalue()

    def test_from_file(self, tmp_path):
        p = tmp_path / "input.in"
        p.write_text("formulas(sos).\nx = x.\nend_of_list.\n")
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            result = ladr_to_tptp_main(["-f", str(p)])
        assert result == 0
        assert "fof(" in mock_out.getvalue()

    def test_no_formulas(self):
        with patch("sys.stdin", io.StringIO("nothing here")):
            result = ladr_to_tptp_main([])
        assert result == 1

    def test_missing_file(self):
        assert ladr_to_tptp_main(["-f", "/nonexistent"]) == 1


# ── Rewriter / IDFilter Tests ────────────────────────────────────────────────


class TestRewriterApps:
    def test_rewriter_usage(self):
        assert rewriter_main([]) == 1

    def test_rewriter_missing_file(self):
        assert rewriter_main(["/nonexistent"]) == 1

    def test_rewriter2_stub(self):
        assert rewriter2_main([]) == 1


class TestIDFilter:
    def test_idfilter_usage(self):
        from pyladr.apps.idfilter import main as idfilter_main
        assert idfilter_main([]) == 1

    def test_idfilter_missing_file(self):
        from pyladr.apps.idfilter import main as idfilter_main
        assert idfilter_main(["/nonexistent"]) == 1
