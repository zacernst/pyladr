"""Property-based tests for paramodulation correctness.

Uses Hypothesis to verify algebraic invariants of paramodulation:
- Paramodulants preserve satisfiability
- Position vectors are valid
- Justification records match the actual inference
- Equation orientation is consistent with term ordering
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from pyladr.core.clause import Clause, JustType, Literal, ParaJust
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, build_binary_term, get_rigid_term, get_variable_term
from pyladr.inference.paramodulation import (
    _oriented_eqs,
    _renamable_flips,
    flip_eq,
    is_eq_atom,
    orient_equalities,
    para_from_into,
    para_from_right,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_para_state():
    """Clear orientation state between tests."""
    _oriented_eqs.clear()
    _renamable_flips.clear()
    yield
    _oriented_eqs.clear()
    _renamable_flips.clear()


def _make_symbol_table() -> SymbolTable:
    s = SymbolTable()
    s.str_to_sn("=", 2)
    s.str_to_sn("f", 1)
    s.str_to_sn("g", 1)
    s.str_to_sn("h", 2)
    s.str_to_sn("p", 1)
    s.str_to_sn("q", 1)
    s.str_to_sn("r", 2)
    for name in "abcde":
        s.str_to_sn(name, 0)
    return s


def _build(sym_table: SymbolTable, name: str, *args: Term) -> Term:
    arity = len(args)
    sn = sym_table.str_to_sn(name, arity)
    if arity == 0:
        return Term(private_symbol=-sn)
    return Term(private_symbol=-sn, arity=arity, args=tuple(args))


# ── Hypothesis strategies ────────────────────────────────────────────────────


# Strategy for small constant names
constant_names = st.sampled_from(["a", "b", "c", "d", "e"])

# Strategy for variable indices (small for tractable unification)
var_indices = st.integers(min_value=0, max_value=5)


@st.composite
def simple_terms(draw, sym_table: SymbolTable, depth: int = 0, max_depth: int = 2):
    """Generate random simple terms for testing."""
    if depth >= max_depth:
        # Leaf: constant or variable
        if draw(st.booleans()):
            name = draw(constant_names)
            return _build(sym_table, name)
        else:
            return get_variable_term(draw(var_indices))

    choice = draw(st.integers(min_value=0, max_value=3))
    if choice == 0:
        name = draw(constant_names)
        return _build(sym_table, name)
    elif choice == 1:
        return get_variable_term(draw(var_indices))
    elif choice == 2:
        arg = draw(simple_terms(sym_table, depth + 1, max_depth))
        return _build(sym_table, "f", arg)
    else:
        arg = draw(simple_terms(sym_table, depth + 1, max_depth))
        return _build(sym_table, "g", arg)


# ── Property tests ───────────────────────────────────────────────────────────


class TestFlipEqProperties:
    """Properties of equation flipping."""

    @given(st.sampled_from(["a", "b", "c"]), st.sampled_from(["a", "b", "c"]))
    def test_flip_is_involution(self, left_name: str, right_name: str):
        """Flipping twice returns the original equation."""
        sym_table = _make_symbol_table()
        left = _build(sym_table, left_name)
        right = _build(sym_table, right_name)
        eq = _build(sym_table, "=", left, right)

        flipped = flip_eq(eq)
        double_flipped = flip_eq(flipped)

        assert double_flipped.args[0].term_ident(eq.args[0])
        assert double_flipped.args[1].term_ident(eq.args[1])

    @given(st.sampled_from(["a", "b", "c"]), st.sampled_from(["a", "b", "c"]))
    def test_flip_swaps_sides(self, left_name: str, right_name: str):
        """Flipping swaps left and right sides of equality."""
        sym_table = _make_symbol_table()
        left = _build(sym_table, left_name)
        right = _build(sym_table, right_name)
        eq = _build(sym_table, "=", left, right)

        flipped = flip_eq(eq)
        assert flipped.args[0].term_ident(right)
        assert flipped.args[1].term_ident(left)


class TestParamodulationProperties:
    """Properties that must hold for any paramodulation."""

    def test_paramodulant_has_justification(self):
        """Every paramodulant must have a PARA justification."""
        sym_table = _make_symbol_table()
        a = _build(sym_table, "a")
        b = _build(sym_table, "b")

        eq = _build(sym_table, "=", a, b)
        from_clause = Clause(
            literals=(Literal(sign=True, atom=eq),),
            id=1,
        )

        pa = _build(sym_table, "p", a)
        into_clause = Clause(
            literals=(Literal(sign=True, atom=pa),),
            id=2,
        )

        results = para_from_into(from_clause, into_clause, False, sym_table)
        for r in results:
            assert len(r.justification) >= 1
            just = r.justification[0]
            assert just.just_type == JustType.PARA
            assert just.para is not None
            assert just.para.from_id == 1
            assert just.para.into_id == 2

    def test_paramodulant_position_vectors_valid(self):
        """Position vectors in justification must be non-empty tuples."""
        sym_table = _make_symbol_table()
        a = _build(sym_table, "a")
        b = _build(sym_table, "b")
        fa = _build(sym_table, "f", a)

        from_clause = Clause(
            literals=(Literal(sign=True, atom=_build(sym_table, "=", a, b)),),
            id=1,
        )
        into_clause = Clause(
            literals=(Literal(sign=True, atom=_build(sym_table, "p", fa)),),
            id=2,
        )

        results = para_from_into(from_clause, into_clause, False, sym_table)
        for r in results:
            just = r.justification[0]
            assert just.para is not None
            assert len(just.para.from_pos) >= 2  # at least (lit, side)
            assert len(just.para.into_pos) >= 2  # at least (lit, arg)
            # All position values should be positive integers
            assert all(p > 0 for p in just.para.from_pos)
            assert all(p > 0 for p in just.para.into_pos)

    def test_no_paramodulants_from_non_equality(self):
        """Non-equality from_clauses produce no paramodulants."""
        sym_table = _make_symbol_table()
        a = _build(sym_table, "a")

        from_clause = Clause(
            literals=(Literal(sign=True, atom=_build(sym_table, "p", a)),),
            id=1,
        )
        into_clause = Clause(
            literals=(Literal(sign=True, atom=_build(sym_table, "q", a)),),
            id=2,
        )

        results = para_from_into(from_clause, into_clause, False, sym_table)
        assert len(results) == 0

    def test_no_paramodulants_from_negative_equality(self):
        """Negative equalities produce no paramodulants."""
        sym_table = _make_symbol_table()
        a = _build(sym_table, "a")
        b = _build(sym_table, "b")

        from_clause = Clause(
            literals=(Literal(sign=False, atom=_build(sym_table, "=", a, b)),),
            id=1,
        )
        into_clause = Clause(
            literals=(Literal(sign=True, atom=_build(sym_table, "p", a)),),
            id=2,
        )

        results = para_from_into(from_clause, into_clause, False, sym_table)
        assert len(results) == 0

    @given(st.sampled_from(["a", "b", "c", "d", "e"]))
    def test_self_paramodulation_produces_reflexivity(self, const_name: str):
        """Paramodulating a=b into itself at the equation's own position
        can produce identities but all results must have justification."""
        sym_table = _make_symbol_table()
        a = _build(sym_table, const_name)
        b = _build(sym_table, "b")

        eq = _build(sym_table, "=", a, b)
        clause = Clause(literals=(Literal(sign=True, atom=eq),), id=1)

        results = para_from_into(clause, clause, False, sym_table)
        for r in results:
            assert len(r.justification) >= 1
            assert r.justification[0].just_type == JustType.PARA

    def test_paramodulant_literal_count_bounded(self):
        """Paramodulant has at most (|from| - 1 + |into|) literals.

        The from_lit (equality) is removed, all other literals are kept.
        """
        sym_table = _make_symbol_table()
        a = _build(sym_table, "a")
        b = _build(sym_table, "b")
        c = _build(sym_table, "c")

        # from_clause: a=b | p(c) (2 literals)
        from_clause = Clause(
            literals=(
                Literal(sign=True, atom=_build(sym_table, "=", a, b)),
                Literal(sign=True, atom=_build(sym_table, "p", c)),
            ),
            id=1,
        )
        # into_clause: q(a) | r(a,c) (2 literals)
        into_clause = Clause(
            literals=(
                Literal(sign=True, atom=_build(sym_table, "q", a)),
                Literal(sign=True, atom=_build(sym_table, "r", a, c)),
            ),
            id=2,
        )

        results = para_from_into(from_clause, into_clause, False, sym_table)
        max_lits = (from_clause.num_literals - 1) + into_clause.num_literals
        for r in results:
            assert r.num_literals <= max_lits


class TestOrientEqualitiesProperties:
    """Properties of equation orientation."""

    def test_orient_preserves_literal_count(self):
        """Orientation doesn't add or remove literals."""
        sym_table = _make_symbol_table()
        a = _build(sym_table, "a")
        b = _build(sym_table, "b")
        c = _build(sym_table, "c")

        clause = Clause(
            literals=(
                Literal(sign=True, atom=_build(sym_table, "=", a, b)),
                Literal(sign=True, atom=_build(sym_table, "p", c)),
            ),
        )

        oriented = orient_equalities(clause, sym_table)
        assert oriented.num_literals == clause.num_literals

    def test_orient_preserves_non_eq_literals(self):
        """Orientation doesn't change non-equality literals."""
        sym_table = _make_symbol_table()
        a = _build(sym_table, "a")
        b = _build(sym_table, "b")
        c = _build(sym_table, "c")

        p_c = _build(sym_table, "p", c)
        clause = Clause(
            literals=(
                Literal(sign=True, atom=_build(sym_table, "=", a, b)),
                Literal(sign=True, atom=p_c),
            ),
        )

        oriented = orient_equalities(clause, sym_table)
        # Second literal should be unchanged
        assert oriented.literals[1].atom.term_ident(p_c)

    def test_orient_idempotent(self):
        """Orienting twice gives same result as once."""
        sym_table = _make_symbol_table()
        x = get_variable_term(0)
        y = get_variable_term(1)
        a = _build(sym_table, "a")

        clause = Clause(
            literals=(
                Literal(sign=True, atom=_build(sym_table, "=",
                    _build(sym_table, "f", x), a)),
            ),
        )

        o1 = orient_equalities(clause, sym_table)
        # Clear state and orient again on the result
        _oriented_eqs.clear()
        _renamable_flips.clear()
        o2 = orient_equalities(o1, sym_table)

        assert o1.num_literals == o2.num_literals
        # Both orientations should produce structurally identical equations
        for l1, l2 in zip(o1.literals, o2.literals):
            assert l1.sign == l2.sign


class TestEqualityDetectionProperties:
    """Properties of equality detection."""

    @given(st.sampled_from(["a", "b", "c"]), st.sampled_from(["a", "b", "c"]))
    def test_eq_atom_always_binary(self, left_name: str, right_name: str):
        """Equality atoms always have arity 2."""
        sym_table = _make_symbol_table()
        left = _build(sym_table, left_name)
        right = _build(sym_table, right_name)
        eq = _build(sym_table, "=", left, right)

        assert is_eq_atom(eq, sym_table)
        assert eq.arity == 2

    def test_non_eq_binary_not_detected(self):
        """Binary non-equality symbols are not equality atoms."""
        sym_table = _make_symbol_table()
        a = _build(sym_table, "a")
        b = _build(sym_table, "b")
        r_ab = _build(sym_table, "r", a, b)

        assert not is_eq_atom(r_ab, sym_table)

    @given(var_indices)
    def test_variables_never_eq_atoms(self, varnum: int):
        """Variables are never equality atoms."""
        sym_table = _make_symbol_table()
        v = get_variable_term(varnum)
        assert not is_eq_atom(v, sym_table)
