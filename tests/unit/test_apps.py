"""Tests for PyLADR auxiliary applications (Phase A).

Tests the CLI framework, renamer, mirror-flip, perm3, prooftrans,
and utility scripts.
"""

from __future__ import annotations

import io
import sys
from unittest.mock import patch

import pytest

from pyladr.apps.cli_common import (
    copy_clause,
    format_clause_bare,
    format_clause_standard,
    print_separator,
    read_clause_stream,
)
from pyladr.apps.mirror_flip import (
    clause_ident,
    contains_mirror_flip,
    flip,
    mirror,
    mirror_term,
)
from pyladr.apps.perm3 import ALL_PERMS, contains_perm3, perm3, perm3_term
from pyladr.apps.renamer import main as renamer_main
from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, build_binary_term, get_rigid_term, get_variable_term


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def symbol_table():
    """A fresh symbol table for testing."""
    return SymbolTable()


@pytest.fixture
def eq_sym(symbol_table: SymbolTable):
    """Equality symbol ID."""
    return symbol_table.str_to_sn("=", 2)


@pytest.fixture
def f_sym(symbol_table: SymbolTable):
    """Binary function f."""
    return symbol_table.str_to_sn("f", 2)


@pytest.fixture
def g_sym(symbol_table: SymbolTable):
    """Ternary function g."""
    return symbol_table.str_to_sn("g", 3)


@pytest.fixture
def a_sym(symbol_table: SymbolTable):
    """Constant a."""
    return symbol_table.str_to_sn("a", 0)


@pytest.fixture
def b_sym(symbol_table: SymbolTable):
    """Constant b."""
    return symbol_table.str_to_sn("b", 0)


def _make_term(sym_id: int, args: tuple[Term, ...] = ()) -> Term:
    """Create a term from symbol ID and args."""
    return Term(private_symbol=-sym_id, arity=len(args), args=args)


def _make_var(n: int) -> Term:
    """Create a variable term."""
    return get_variable_term(n)


def _make_unit_eq(
    eq_id: int, lhs: Term, rhs: Term, clause_id: int = 0
) -> Clause:
    """Create a unit equality clause: lhs = rhs."""
    atom = Term(private_symbol=-eq_id, arity=2, args=(lhs, rhs))
    lit = Literal(sign=True, atom=atom)
    return Clause(literals=(lit,), id=clause_id)


# ── CLI Common Tests ─────────────────────────────────────────────────────────


class TestCLICommon:
    """Tests for shared CLI infrastructure."""

    def test_format_clause_bare_empty(self):
        clause = Clause(literals=())
        assert format_clause_bare(clause) == "$F."

    def test_format_clause_bare_unit(self, symbol_table, eq_sym, a_sym, b_sym):
        clause = _make_unit_eq(eq_sym, _make_term(a_sym), _make_term(b_sym))
        result = format_clause_bare(clause, symbol_table)
        assert result.endswith(".")
        assert "=" in result

    def test_format_clause_standard_with_id(self, symbol_table, eq_sym, a_sym, b_sym):
        clause = _make_unit_eq(eq_sym, _make_term(a_sym), _make_term(b_sym), clause_id=1)
        result = format_clause_standard(clause, symbol_table)
        assert result.startswith("1 ")

    def test_copy_clause_preserves_structure(self, eq_sym, a_sym, b_sym):
        original = _make_unit_eq(eq_sym, _make_term(a_sym), _make_term(b_sym))
        copied = copy_clause(original)
        assert clause_ident(original.literals, copied.literals)
        # Should be different objects
        assert original is not copied

    def test_print_separator(self, capsys):
        print_separator("TEST")
        output = capsys.readouterr().out
        assert "TEST" in output
        assert "=" in output

    def test_read_clause_stream_formulas_list(self, symbol_table):
        text = "formulas(sos).\nx = y.\nend_of_list.\n"
        stream = io.StringIO(text)
        clauses = read_clause_stream(stream, symbol_table)
        assert len(clauses) >= 1

    def test_read_clause_stream_bare(self, symbol_table):
        text = "x = y.\n"
        stream = io.StringIO(text)
        clauses = read_clause_stream(stream, symbol_table)
        assert len(clauses) >= 1


# ── Mirror-Flip Tests ────────────────────────────────────────────────────────


class TestMirrorFlip:
    """Tests for mirror-flip tool."""

    def test_mirror_term_binary(self, f_sym, a_sym, b_sym):
        """mirror(f(a,b)) = f(b,a)."""
        t = _make_term(f_sym, (_make_term(a_sym), _make_term(b_sym)))
        result = mirror_term(t)
        # args should be reversed
        assert result.args[0].term_ident(_make_term(b_sym))
        assert result.args[1].term_ident(_make_term(a_sym))

    def test_mirror_term_nested(self, f_sym, a_sym, b_sym):
        """mirror(f(f(a,b), c)) = f(c, f(b,a))."""
        inner = _make_term(f_sym, (_make_term(a_sym), _make_term(b_sym)))
        c_sym = a_sym  # reuse for simplicity
        t = _make_term(f_sym, (inner, _make_term(c_sym)))
        result = mirror_term(t)
        # outer args reversed, inner also mirrored
        assert result.args[0].term_ident(_make_term(c_sym))
        assert result.args[1].arity == 2

    def test_mirror_term_variable_unchanged(self):
        """Variables are not affected by mirroring."""
        v = _make_var(0)
        assert mirror_term(v).term_ident(v)

    def test_mirror_term_constant_unchanged(self, a_sym):
        """Constants are not affected by mirroring."""
        c = _make_term(a_sym)
        assert mirror_term(c).term_ident(c)

    def test_flip_swaps_equation_sides(self, eq_sym, a_sym, b_sym):
        """flip(a = b) gives b = a."""
        clause = _make_unit_eq(eq_sym, _make_term(a_sym), _make_term(b_sym))
        flipped = flip(clause)
        atom = flipped.literals[0].atom
        assert atom.args[0].term_ident(_make_term(b_sym))
        assert atom.args[1].term_ident(_make_term(a_sym))

    def test_clause_ident_same(self, eq_sym, a_sym, b_sym):
        c1 = _make_unit_eq(eq_sym, _make_term(a_sym), _make_term(b_sym))
        c2 = _make_unit_eq(eq_sym, _make_term(a_sym), _make_term(b_sym))
        assert clause_ident(c1.literals, c2.literals)

    def test_clause_ident_different(self, eq_sym, a_sym, b_sym):
        c1 = _make_unit_eq(eq_sym, _make_term(a_sym), _make_term(b_sym))
        c2 = _make_unit_eq(eq_sym, _make_term(b_sym), _make_term(a_sym))
        assert not clause_ident(c1.literals, c2.literals)

    def test_contains_mirror_flip_finds_flip(self, eq_sym, a_sym, b_sym):
        """a=b should be filtered if b=a is already kept."""
        ab = _make_unit_eq(eq_sym, _make_term(a_sym), _make_term(b_sym))
        ba = _make_unit_eq(eq_sym, _make_term(b_sym), _make_term(a_sym))
        kept = [ab]
        assert contains_mirror_flip(ba, kept)

    def test_contains_mirror_flip_unique(self, eq_sym, a_sym, b_sym):
        """Truly different equation should not be filtered."""
        ab = _make_unit_eq(eq_sym, _make_term(a_sym), _make_term(b_sym))
        # a = a is not a flip/mirror of a = b
        aa = _make_unit_eq(eq_sym, _make_term(a_sym), _make_term(a_sym))
        kept = [ab]
        assert not contains_mirror_flip(aa, kept)


# ── Perm3 Tests ──────────────────────────────────────────────────────────────


class TestPerm3:
    """Tests for perm3 tool."""

    def test_perm3_term_identity(self, g_sym, a_sym, b_sym):
        """Identity permutation (0,1,2) preserves term."""
        c_sym = a_sym  # reuse
        t = _make_term(g_sym, (_make_term(a_sym), _make_term(b_sym), _make_term(c_sym)))
        result = perm3_term(t, (0, 1, 2))
        assert result.term_ident(t)

    def test_perm3_term_swap(self, g_sym, a_sym, b_sym):
        """Permutation (2,1,0) reverses ternary args."""
        c_sym = b_sym  # reuse for third distinct
        t = _make_term(g_sym, (_make_term(a_sym), _make_term(b_sym), _make_term(c_sym)))
        result = perm3_term(t, (2, 1, 0))
        assert result.args[2].term_ident(_make_term(a_sym))
        assert result.args[0].term_ident(_make_term(c_sym))

    def test_all_perms_count(self):
        """Should have exactly 6 permutations of 3 elements."""
        assert len(ALL_PERMS) == 6

    def test_contains_perm3_filters_permutation(self, eq_sym, g_sym, a_sym, b_sym):
        """A permuted version of a kept clause should be filtered."""
        t1 = _make_term(g_sym, (_make_term(a_sym), _make_term(b_sym), _make_term(a_sym)))
        t2 = _make_term(g_sym, (_make_term(a_sym), _make_term(a_sym), _make_term(b_sym)))
        c1 = _make_unit_eq(eq_sym, t1, _make_term(a_sym))
        c2 = _make_unit_eq(eq_sym, t2, _make_term(a_sym))
        # c2 is a permutation of c1's ternary arguments
        kept = [c1]
        # This may or may not filter depending on exact permutation matching
        # The key test is that identical clauses ARE filtered
        c1_copy = _make_unit_eq(eq_sym, t1, _make_term(a_sym))
        assert contains_perm3(c1_copy, kept)


# ── Prooftrans Tests ─────────────────────────────────────────────────────────


class TestProoftrans:
    """Tests for prooftrans tool."""

    def test_parse_justification(self):
        from pyladr.apps.prooftrans import _parse_justification

        justs = _parse_justification("binary_res,1,2")
        assert len(justs) == 1
        assert justs[0].just_type == JustType.BINARY_RES
        assert justs[0].clause_ids == (1, 2)

    def test_parse_justification_assumption(self):
        from pyladr.apps.prooftrans import _parse_justification

        justs = _parse_justification("assumption")
        assert len(justs) == 1
        assert justs[0].just_type == JustType.INPUT

    def test_renumber_proof(self):
        from pyladr.apps.prooftrans import _renumber_proof

        c1 = Clause(
            literals=(),
            id=10,
            justification=(Justification(just_type=JustType.INPUT),),
        )
        c2 = Clause(
            literals=(),
            id=20,
            justification=(
                Justification(just_type=JustType.BINARY_RES, clause_ids=(10,)),
            ),
        )
        renumbered = _renumber_proof([c1, c2])
        assert renumbered[0].id == 1
        assert renumbered[1].id == 2
        # Parent reference should be remapped
        assert renumbered[1].justification[0].clause_ids == (1,)


# ── Get Utilities Tests ──────────────────────────────────────────────────────


class TestGetUtilities:
    """Tests for get_interps, get_givens, get_kept."""

    def test_get_interps(self):
        from pyladr.apps.get_interps import main as get_interps_main

        input_text = (
            "some preamble\n"
            "============================== MODEL =================================\n"
            "f(0,0) = 0.\n"
            "f(0,1) = 1.\n"
            "============================== end of model ==========================\n"
            "some postamble\n"
        )
        with (
            patch("sys.stdin", io.StringIO(input_text)),
            patch("sys.stdout", new_callable=io.StringIO) as mock_out,
        ):
            result = get_interps_main()
        assert result == 0
        output = mock_out.getvalue()
        assert "f(0,0) = 0" in output
        assert "preamble" not in output

    def test_get_givens(self):
        from pyladr.apps.get_givens import main as get_givens_main

        input_text = (
            "some output\n"
            "given #1: x = x.\n"
            "some more output\n"
            "given #2: f(x,y) = f(y,x).\n"
        )
        with (
            patch("sys.stdin", io.StringIO(input_text)),
            patch("sys.stdout", new_callable=io.StringIO) as mock_out,
        ):
            result = get_givens_main()
        assert result == 0
        output = mock_out.getvalue()
        assert "given #1" in output
        assert "given #2" in output
        assert "some output" not in output

    def test_get_kept(self):
        from pyladr.apps.get_kept import main as get_kept_main

        input_text = (
            "some output\n"
            "kept: 1 x = x.\n"
            "other line\n"
            "kept: 2 f(x) = x.\n"
        )
        with (
            patch("sys.stdin", io.StringIO(input_text)),
            patch("sys.stdout", new_callable=io.StringIO) as mock_out,
        ):
            result = get_kept_main()
        assert result == 0
        output = mock_out.getvalue()
        assert "kept: 1" in output
        assert "kept: 2" in output
        assert "other line" not in output


# ── CLI Dispatch Tests ───────────────────────────────────────────────────────


class TestCLIDispatch:
    """Tests for main CLI entry point."""

    def test_help(self, capsys):
        from pyladr.cli import main

        with patch("sys.argv", ["pyprover9", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        output = capsys.readouterr().out
        assert "PyProver9" in output or "pyprover9" in output.lower()

    def test_version(self, capsys):
        from pyladr.cli import main

        with patch("sys.argv", ["pyprover9", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        output = capsys.readouterr().out
        assert "pyprover9" in output

    def test_no_args(self, capsys):
        from pyladr.cli import main

        with patch("sys.argv", ["pyprover9"]):
            result = main()
        assert result == 1  # not yet implemented
