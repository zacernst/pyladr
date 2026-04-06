"""Tests for remaining Phase B apps: interpfilter, dprofiles, sigtest,
latfilter, upper-covers.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from pyladr.apps.dprofiles import main as dprofiles_main
from pyladr.apps.interpfilter import main as interpfilter_main
from pyladr.apps.latfilter import lattice_identity, lattice_leq, main as latfilter_main
from pyladr.apps.sigtest import main as sigtest_main
from pyladr.apps.upper_covers import compute_upper_covers, main as upper_covers_main
from pyladr.core.interpretation import compile_interp_from_text
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term


Z2_TEXT = """interpretation(2, [number=1, seconds=0], [
  function(f(_,_), [0,1,1,0]),
  function(e, [0])
])."""

# A 4-element lattice: Boolean algebra B2 (diamond)
# meet table for {0,1,2,3} where 0=bot, 3=top, 1,2 incomparable
LATTICE_TEXT = """interpretation(4, [number=1, seconds=0], [
  function(meet(_,_), [0,0,0,0, 0,1,0,1, 0,0,2,2, 0,1,2,3])
])."""


def _make_term(sym_id: int, args: tuple[Term, ...] = ()) -> Term:
    return Term(private_symbol=-sym_id, arity=len(args), args=args)


def _make_var(n: int) -> Term:
    from pyladr.core.term import get_variable_term
    return get_variable_term(n)


# ── InterpFilter Tests ───────────────────────────────────────────────────────


class TestInterpFilter:
    def test_main_usage(self):
        assert interpfilter_main([]) == 1

    def test_main_bad_mode(self, tmp_path):
        p = tmp_path / "clauses.txt"
        p.write_text("x = x.\n")
        assert interpfilter_main([str(p), "bad_mode"]) == 1

    def test_main_missing_file(self):
        assert interpfilter_main(["/nonexistent", "all_true"]) == 1


# ── Dprofiles Tests ──────────────────────────────────────────────────────────


class TestDprofiles:
    def test_basic(self):
        with patch("sys.stdin", io.StringIO(Z2_TEXT)):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                result = dprofiles_main()
        assert result == 0
        output = mock_out.getvalue()
        assert "Occurrences" in output
        assert "interpretation" in output

    def test_no_interps(self):
        with patch("sys.stdin", io.StringIO("nothing")):
            result = dprofiles_main()
        assert result == 1


# ── Sigtest Tests ────────────────────────────────────────────────────────────


class TestSigtest:
    def test_main_usage(self):
        assert sigtest_main([]) == 1

    def test_main_missing_file(self):
        assert sigtest_main(["/nonexistent"]) == 1


# ── Latfilter Tests ──────────────────────────────────────────────────────────


class TestLatfilter:
    def test_lattice_leq_var_equal(self):
        """x <= x should hold."""
        st = SymbolTable()
        x = _make_var(0)
        assert lattice_leq(x, x, set(), set())

    def test_lattice_leq_var_different(self):
        """x <= y should NOT hold."""
        x, y = _make_var(0), _make_var(1)
        assert not lattice_leq(x, y, set(), set())

    def test_lattice_leq_join_left(self):
        """x v y <= t iff x <= t and y <= t."""
        st = SymbolTable()
        join_sym = st.str_to_sn("v", 2)
        join_syms = {join_sym}
        x, y = _make_var(0), _make_var(1)
        # x v y <= x should fail (y <= x fails)
        join_xy = _make_term(join_sym, (x, y))
        assert not lattice_leq(join_xy, x, set(), join_syms)
        # x v x <= x should hold
        join_xx = _make_term(join_sym, (x, x))
        assert lattice_leq(join_xx, x, set(), join_syms)

    def test_lattice_leq_meet_right(self):
        """s <= t1 ^ t2 iff s <= t1 and s <= t2."""
        st = SymbolTable()
        meet_sym = st.str_to_sn("^", 2)
        meet_syms = {meet_sym}
        x, y = _make_var(0), _make_var(1)
        # x <= x ^ y should fail (x <= y fails)
        meet_xy = _make_term(meet_sym, (x, y))
        assert not lattice_leq(x, meet_xy, meet_syms, set())
        # x <= x ^ x should hold
        meet_xx = _make_term(meet_sym, (x, x))
        assert lattice_leq(x, meet_xx, meet_syms, set())

    def test_absorption_identity(self):
        """x ^ (x v y) = x is a lattice identity."""
        st = SymbolTable()
        eq_sym = st.str_to_sn("=", 2)
        meet_sym = st.str_to_sn("^", 2)
        join_sym = st.str_to_sn("v", 2)
        meet_syms = {meet_sym}
        join_syms = {join_sym}

        x, y = _make_var(0), _make_var(1)
        join_xy = _make_term(join_sym, (x, y))
        meet_x_jxy = _make_term(meet_sym, (x, join_xy))
        atom = _make_term(eq_sym, (meet_x_jxy, x))

        assert lattice_identity(atom, eq_sym, meet_syms, join_syms)

    def test_non_identity(self):
        """x ^ y = x is NOT a lattice identity."""
        st = SymbolTable()
        eq_sym = st.str_to_sn("=", 2)
        meet_sym = st.str_to_sn("^", 2)
        meet_syms = {meet_sym}

        x, y = _make_var(0), _make_var(1)
        meet_xy = _make_term(meet_sym, (x, y))
        atom = _make_term(eq_sym, (meet_xy, x))

        assert not lattice_identity(atom, eq_sym, meet_syms, set())


# ── Upper-Covers Tests ───────────────────────────────────────────────────────


class TestUpperCovers:
    def test_compute_covers_lattice(self):
        """Test upper-covers on diamond lattice."""
        interp = compile_interp_from_text(LATTICE_TEXT)
        covers = compute_upper_covers(interp)
        # 0 (bottom) covers 1 and 2
        assert sorted(covers[0]) == [1, 2]
        # 1 covers 3 (top)
        assert covers[1] == [3]
        # 2 covers 3 (top)
        assert covers[2] == [3]
        # 3 (top) has no upper covers
        assert covers[3] == []

    def test_main_basic(self):
        with patch("sys.stdin", io.StringIO(LATTICE_TEXT)):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                result = upper_covers_main()
        assert result == 0
        output = mock_out.getvalue()
        assert "[1, 2]" in output or "1" in output

    def test_main_no_interps(self):
        with patch("sys.stdin", io.StringIO("nothing")):
            result = upper_covers_main()
        assert result == 1
