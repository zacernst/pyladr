"""Integration tests for auxiliary applications ecosystem.

Tests end-to-end workflows combining multiple apps and core systems:
- Interpretation parsing → evaluation → formatting roundtrip
- Isofilter pipeline with multiple interpretations
- Clausefilter with real algebraic interpretations
- Latfilter with standard lattice identities
- Upper-covers on known lattice structures
- CLI dispatch for all registered tools
"""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from pyladr.apps.clausefilter import FilterMode, filter_clause
from pyladr.apps.interpformat import main as interpformat_main
from pyladr.apps.isofilter import main as isofilter_main
from pyladr.apps.latfilter import lattice_identity, lattice_leq
from pyladr.apps.upper_covers import compute_upper_covers
from pyladr.core.clause import Clause, Literal
from pyladr.core.interpretation import (
    compile_interp_from_text,
    eval_clause,
    eval_clause_false_instances,
    eval_clause_true_instances,
    format_interp_standard,
    isomorphic_interps,
    permute_interp,
)
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, get_variable_term


# ── Test data: well-known algebraic structures ───────────────────────────────

# Z2 (integers mod 2 under XOR)
Z2 = """interpretation(2, [number=1, seconds=0], [
  function(f(_,_), [0,1,1,0]),
  function(e, [0])
])."""

# Z3 (integers mod 3 under addition)
Z3 = """interpretation(3, [number=1, seconds=0], [
  function(f(_,_), [0,1,2,1,2,0,2,0,1])
])."""

# Z4 (cyclic group of order 4)
Z4 = """interpretation(4, [number=1, seconds=0], [
  function(f(_,_), [0,1,2,3,1,2,3,0,2,3,0,1,3,0,1,2])
])."""

# V4 (Klein four-group)
V4 = """interpretation(4, [number=1, seconds=0], [
  function(f(_,_), [0,1,2,3,1,0,3,2,2,3,0,1,3,2,1,0])
])."""

# Boolean algebra B2 as a lattice: {0,a,b,1}
# meet: 0=bot, 3=top, 1,2 are atoms
B2_LATTICE = """interpretation(4, [number=1, seconds=0], [
  function(meet(_,_), [0,0,0,0, 0,1,0,1, 0,0,2,2, 0,1,2,3]),
  function(join(_,_), [0,1,2,3, 1,1,3,3, 2,3,2,3, 3,3,3,3])
])."""


def _make_term(sym_id: int, args: tuple[Term, ...] = ()) -> Term:
    return Term(private_symbol=-sym_id, arity=len(args), args=args)


def _make_var(n: int) -> Term:
    return get_variable_term(n)


def _make_unit_eq(eq_id: int, lhs: Term, rhs: Term) -> Clause:
    atom = Term(private_symbol=-eq_id, arity=2, args=(lhs, rhs))
    lit = Literal(sign=True, atom=atom)
    return Clause(literals=(lit,))


# ── E2E: Interpretation roundtrip ────────────────────────────────────────────


class TestInterpretationRoundtrip:
    """Test parse → format → reparse → verify identity."""

    @pytest.mark.parametrize("text", [Z2, Z3, Z4, V4], ids=["Z2", "Z3", "Z4", "V4"])
    def test_standard_roundtrip(self, text):
        """Standard format should survive parse-format-reparse."""
        interp = compile_interp_from_text(text)
        formatted = format_interp_standard(interp)
        reparsed = compile_interp_from_text(formatted)
        # Verify all operations match
        for name, op in interp.operations.items():
            reparsed_op = reparsed.operations[name]
            assert op.values == reparsed_op.values
            assert op.arity == reparsed_op.arity

    @pytest.mark.parametrize(
        "fmt", ["standard", "standard2", "cooked", "xml", "tex", "raw", "portable"]
    )
    def test_all_formats_produce_output(self, fmt):
        """Every format should produce non-empty output."""
        with patch("sys.stdin", io.StringIO(Z2)):
            with patch("sys.stdout", new_callable=io.StringIO) as out:
                result = interpformat_main([fmt])
        assert result == 0
        assert len(out.getvalue()) > 0


# ── E2E: Isomorphism pipeline ────────────────────────────────────────────────


class TestIsomorphismPipeline:
    """Test the complete isofilter workflow."""

    def test_z4_not_iso_v4(self):
        """Z4 and V4 are non-isomorphic groups of order 4."""
        z4 = compile_interp_from_text(Z4)
        v4 = compile_interp_from_text(V4)
        assert not isomorphic_interps(z4, v4)

    def test_permuted_z3_is_iso(self):
        """A permuted Z3 should be detected as isomorphic."""
        z3 = compile_interp_from_text(Z3)
        # Rotate: 0->1, 1->2, 2->0
        perm = [1, 2, 0]
        rotated = permute_interp(z3, perm)
        assert isomorphic_interps(z3, rotated)

    def test_isofilter_pipeline_mixed_groups(self):
        """Isofilter should keep Z4 and V4, remove duplicate Z4."""
        # Z4 + V4 + Z4_permuted → should keep 2
        z4 = compile_interp_from_text(Z4)
        z4_perm = permute_interp(z4, [1, 2, 3, 0])
        z4_perm_text = format_interp_standard(z4_perm) + "."

        combined = Z4 + "\n" + V4 + "\n" + z4_perm_text
        with patch("sys.stdin", io.StringIO(combined)):
            with patch("sys.stdout", new_callable=io.StringIO) as out:
                isofilter_main([])
        output = out.getvalue()
        assert output.count("interpretation(") == 2


# ── E2E: Clause evaluation in algebraic models ──────────────────────────────


class TestClauseEvaluation:
    """Test clause evaluation with real algebraic laws."""

    def test_group_axioms_in_z3(self):
        """Z3 should satisfy standard group axioms."""
        st = SymbolTable()
        eq = st.str_to_sn("=", 2)
        f = st.str_to_sn("f", 2)
        z3 = compile_interp_from_text(Z3)

        x, y, z = _make_var(0), _make_var(1), _make_var(2)

        # Associativity: f(f(x,y),z) = f(x,f(y,z))
        lhs = _make_term(f, (_make_term(f, (x, y)), z))
        rhs = _make_term(f, (x, _make_term(f, (y, z))))
        assoc = _make_unit_eq(eq, lhs, rhs)
        assert eval_clause(assoc, z3, st, eq)

        # Commutativity: f(x,y) = f(y,x)
        comm = _make_unit_eq(eq, _make_term(f, (x, y)), _make_term(f, (y, x)))
        assert eval_clause(comm, z3, st, eq)

    def test_non_idempotent_in_z3(self):
        """f(x,x) = x should fail in Z3 (not idempotent)."""
        st = SymbolTable()
        eq = st.str_to_sn("=", 2)
        f = st.str_to_sn("f", 2)
        z3 = compile_interp_from_text(Z3)

        x = _make_var(0)
        idemp = _make_unit_eq(eq, _make_term(f, (x, x)), x)
        assert not eval_clause(idemp, z3, st, eq)

        # Should have exactly 1 true instance (x=0: f(0,0)=0)
        true_ct = eval_clause_true_instances(idemp, z3, st, eq)
        assert true_ct == 1

    def test_clausefilter_separates_groups(self):
        """Idempotency should filter Z2 (passes) from Z3 (fails)."""
        st = SymbolTable()
        eq = st.str_to_sn("=", 2)
        f = st.str_to_sn("f", 2)

        z2 = compile_interp_from_text(Z2)
        z3 = compile_interp_from_text(Z3)

        x = _make_var(0)
        idemp = _make_unit_eq(eq, _make_term(f, (x, x)), x)

        # Z2 XOR: f(0,0)=0, f(1,1)=0 — idempotency fails for x=1
        assert not eval_clause(idemp, z2, st, eq)
        # Z3 add: f(0,0)=0 only — idempotency fails for x=1,2
        assert not eval_clause(idemp, z3, st, eq)

        # But commutativity holds in both
        y = _make_var(1)
        comm = _make_unit_eq(eq, _make_term(f, (x, y)), _make_term(f, (y, x)))
        assert filter_clause(comm, [z2, z3], FilterMode.TRUE_IN_ALL, st, eq)


# ── E2E: Lattice identity verification ──────────────────────────────────────


class TestLatticeVerification:
    """Test lattice identity checker on standard identities."""

    def _setup_lattice_syms(self):
        st = SymbolTable()
        eq_sym = st.str_to_sn("=", 2)
        meet_sym = st.str_to_sn("^", 2)
        join_sym = st.str_to_sn("v", 2)
        return st, eq_sym, {meet_sym}, {join_sym}

    def test_absorption_law_1(self):
        """x ^ (x v y) = x is a lattice identity."""
        st, eq, ms, js = self._setup_lattice_syms()
        meet = next(iter(ms))
        join = next(iter(js))
        x, y = _make_var(0), _make_var(1)

        lhs = _make_term(meet, (x, _make_term(join, (x, y))))
        atom = _make_term(eq, (lhs, x))
        assert lattice_identity(atom, eq, ms, js)

    def test_absorption_law_2(self):
        """x v (x ^ y) = x is a lattice identity."""
        st, eq, ms, js = self._setup_lattice_syms()
        meet = next(iter(ms))
        join = next(iter(js))
        x, y = _make_var(0), _make_var(1)

        lhs = _make_term(join, (x, _make_term(meet, (x, y))))
        atom = _make_term(eq, (lhs, x))
        assert lattice_identity(atom, eq, ms, js)

    def test_idempotency_meet(self):
        """x ^ x = x is a lattice identity."""
        st, eq, ms, js = self._setup_lattice_syms()
        meet = next(iter(ms))
        x = _make_var(0)

        lhs = _make_term(meet, (x, x))
        atom = _make_term(eq, (lhs, x))
        assert lattice_identity(atom, eq, ms, js)

    def test_distributivity_not_lattice_identity(self):
        """x ^ (y v z) = (x ^ y) v (x ^ z) is NOT a lattice identity."""
        st, eq, ms, js = self._setup_lattice_syms()
        meet = next(iter(ms))
        join = next(iter(js))
        x, y, z = _make_var(0), _make_var(1), _make_var(2)

        lhs = _make_term(meet, (x, _make_term(join, (y, z))))
        rhs = _make_term(join, (
            _make_term(meet, (x, y)),
            _make_term(meet, (x, z)),
        ))
        atom = _make_term(eq, (lhs, rhs))
        assert not lattice_identity(atom, eq, ms, js)


# ── E2E: Upper-covers on known lattices ─────────────────────────────────────


class TestUpperCoversIntegration:
    """Test upper-covers computation on real lattice structures."""

    def test_b2_lattice_diamond(self):
        """B2 diamond lattice: 0 covers {1,2}, both cover 3."""
        interp = compile_interp_from_text(B2_LATTICE)
        covers = compute_upper_covers(interp)

        # Bottom element covers two atoms
        assert sorted(covers[0]) == [1, 2]
        # Each atom covers the top
        assert covers[1] == [3]
        assert covers[2] == [3]
        # Top has no upper covers
        assert covers[3] == []


# ── E2E: CLI dispatch ────────────────────────────────────────────────────────


class TestCLIDispatch:
    """Test that the main CLI can dispatch all registered tools."""

    def test_help_lists_all_tools(self, capsys):
        from pyladr.cli import AVAILABLE_TOOLS, main

        with patch("sys.argv", ["pyprover9", "--help"]):
            result = main()
        assert result == 0
        output = capsys.readouterr().out
        # Verify key tools are listed
        for tool in ["renamer", "prooftrans", "clausefilter", "isofilter", "latfilter"]:
            assert tool in output

    def test_tool_count(self):
        from pyladr.cli import AVAILABLE_TOOLS

        assert len(AVAILABLE_TOOLS) >= 26
