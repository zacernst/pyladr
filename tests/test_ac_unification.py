"""Tests for AC unification, matching, and normal form."""

from __future__ import annotations

import pytest

from pyladr.core.ac_normal_form import (
    ac_canonical,
    flatten_ac,
    flatten_with_multiplicities,
    right_associate,
    term_compare_ncv,
)
from pyladr.core.clause import Clause, Literal
from pyladr.core.substitution import Context, Trail, apply_substitution, match
from pyladr.core.symbol import SymbolTable, UnifTheory
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.inference.ac_unification import (
    ac_match_all,
    ac_unify_all,
    commutative_match_all,
    commutative_unify_all,
    is_ac_tautology,
)
from pyladr.inference.diophantine import DioResult, dio, next_combo_a


# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def symtab():
    return SymbolTable()


def _var(n: int) -> Term:
    return get_variable_term(n)


def _const(st: SymbolTable, name: str) -> Term:
    return get_rigid_term(st.str_to_sn(name, 0), 0)


def _func(st: SymbolTable, name: str, *args: Term) -> Term:
    return get_rigid_term(st.str_to_sn(name, len(args)), len(args), args)


def _make_ac(st: SymbolTable, name: str) -> int:
    """Create an AC binary operator and return its symnum."""
    sn = st.str_to_sn(name, 2)
    sym = st.get_symbol(sn)
    sym.unif_theory = UnifTheory.ASSOC_COMMUTE
    return sn


def _make_comm(st: SymbolTable, name: str) -> int:
    """Create a commutative binary operator and return its symnum."""
    sn = st.str_to_sn(name, 2)
    sym = st.get_symbol(sn)
    sym.unif_theory = UnifTheory.COMMUTE
    return sn


def _is_ac(st: SymbolTable):
    """Create is_ac predicate for a symbol table."""
    def check(symnum: int) -> bool:
        sym = st.get_symbol(symnum)
        return sym is not None and sym.unif_theory == UnifTheory.ASSOC_COMMUTE
    return check


def _ac_term(st: SymbolTable, name: str, *args: Term) -> Term:
    """Build a binary AC term (right-associated for >2 args)."""
    sn = st.str_to_sn(name, 2)
    if len(args) == 2:
        return get_rigid_term(sn, 2, args)
    if len(args) > 2:
        result = args[-1]
        for i in range(len(args) - 2, -1, -1):
            result = get_rigid_term(sn, 2, (args[i], result))
        return result
    raise ValueError("AC term needs at least 2 args")


# ── Diophantine solver tests ─────────────────────────────────────────────────


class TestDiophantine:
    """Test the Diophantine equation solver."""

    def test_simple_1_1(self):
        """1*x1 = 1*y1 → basis = {(1,1)}"""
        ab = [1, 1]
        constraints = [0, 0]
        result = dio(ab, 1, 1, constraints)
        assert result.status == 1
        assert result.num_basis >= 1
        # Should have at least the trivial solution [1, 1]
        assert any(b[0] == 1 and b[1] == 1 for b in result.basis)

    def test_simple_2_2(self):
        """1*x1 + 1*x2 = 1*y1 + 1*y2"""
        ab = [1, 1, 1, 1]
        constraints = [0, 0, 0, 0]
        result = dio(ab, 2, 2, constraints)
        assert result.status == 1
        assert result.num_basis >= 1

    def test_weighted(self):
        """2*x1 = 1*y1 → basis includes (1, 2)"""
        ab = [2, 1]
        constraints = [0, 0]
        result = dio(ab, 1, 1, constraints)
        assert result.status == 1
        assert any(b[0] == 1 and b[1] == 2 for b in result.basis)

    def test_no_solution_with_constraints(self):
        """Constraints prevent solution: x must be function f, y must be function g."""
        ab = [1, 1]
        constraints = [1, 2]  # Different symbols
        result = dio(ab, 1, 1, constraints)
        # With these constraints, a solution [1,1] violates var_check_2
        # since both are constrained but different

    def test_empty_sides(self):
        """Empty equation: m=0 or n=0."""
        result = dio([], 0, 0, [])
        assert result.status == 1
        assert result.num_basis == 0

    def test_next_combo_basic(self):
        """Basic combo enumeration."""
        ab = [1, 1]
        constraints = [0, 0]
        result = dio(ab, 1, 1, constraints)
        assert result.num_basis > 0

        combo = [0] * result.num_basis
        sumvec = [0] * 2
        found = next_combo_a(2, result.basis, result.num_basis, constraints,
                             combo, sumvec, True)
        assert found


# ── AC normal form tests ─────────────────────────────────────────────────────


class TestACNormalForm:
    """Test AC term canonicalization."""

    def test_flatten_simple(self, symtab):
        """Flatten f(a, f(b, c)) → [a, b, c]."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        c = _const(symtab, "c")
        f_sn = _make_ac(symtab, "+")
        f_bc = _ac_term(symtab, "+", b, c)
        f_a_bc = _ac_term(symtab, "+", a, f_bc)

        flat = flatten_ac(f_a_bc, f_a_bc.private_symbol)
        assert len(flat) == 3

    def test_flatten_nested(self, symtab):
        """Flatten f(f(a, b), f(c, d)) → [a, b, c, d]."""
        a, b, c, d = [_const(symtab, n) for n in "abcd"]
        f_ab = _ac_term(symtab, "+", a, b)
        f_cd = _ac_term(symtab, "+", c, d)
        f_top = _ac_term(symtab, "+", f_ab, f_cd)

        flat = flatten_ac(f_top, f_top.private_symbol)
        assert len(flat) == 4

    def test_right_associate(self, symtab):
        """Right-associate [a, b, c] → f(a, f(b, c))."""
        a, b, c = [_const(symtab, n) for n in "abc"]
        f_sn = _make_ac(symtab, "+")
        f_ps = get_rigid_term(f_sn, 2, (a, b)).private_symbol  # get private_symbol

        result = right_associate(f_ps, 2, [a, b, c])
        assert result.arity == 2
        assert result.args[0].term_ident(a)
        assert result.args[1].arity == 2

    def test_canonical_sorts_arguments(self, symtab):
        """AC canonical sorts: f(b, a) → f(a, b) if a < b."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        _make_ac(symtab, "+")
        is_ac = _is_ac(symtab)
        f_ba = _ac_term(symtab, "+", b, a)
        f_ab = _ac_term(symtab, "+", a, b)

        canon = ac_canonical(f_ba, is_ac)
        expected = ac_canonical(f_ab, is_ac)
        assert canon.term_ident(expected)

    def test_canonical_flattens_and_sorts(self, symtab):
        """f(f(c, a), b) → f(a, f(b, c)) after flatten-sort-reassociate."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        c = _const(symtab, "c")
        _make_ac(symtab, "+")
        is_ac = _is_ac(symtab)

        # f(f(c, a), b) should become f(a, f(b, c))
        f_ca = _ac_term(symtab, "+", c, a)
        f_top = _ac_term(symtab, "+", f_ca, b)
        canon = ac_canonical(f_top, is_ac)

        flat = flatten_ac(canon, canon.private_symbol)
        assert len(flat) == 3

    def test_canonical_non_ac_unchanged(self, symtab):
        """Non-AC terms are not affected by canonicalization."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        g_ba = _func(symtab, "g", b, a)
        is_ac = _is_ac(symtab)
        canon = ac_canonical(g_ba, is_ac)
        assert canon.term_ident(g_ba)

    def test_term_compare_ncv_ordering(self, symtab):
        """Constants < Complex < Variables."""
        a = _const(symtab, "a")
        f_a = _func(symtab, "f", a)
        x = _var(0)

        assert term_compare_ncv(a, f_a) < 0  # const < complex
        assert term_compare_ncv(f_a, x) < 0  # complex < variable
        assert term_compare_ncv(a, x) < 0    # const < variable

    def test_multiplicities(self, symtab):
        """f(a, f(a, b)) → [(a, 2), (b, 1)]."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        _make_ac(symtab, "+")
        f_ab = _ac_term(symtab, "+", a, b)
        f_a_ab = _ac_term(symtab, "+", a, f_ab)

        mults = flatten_with_multiplicities(f_a_ab, f_a_ab.private_symbol)
        # a appears twice, b once
        assert any(t.term_ident(a) and m == 2 for t, m in mults)
        assert any(t.term_ident(b) and m == 1 for t, m in mults)


# ── Commutative unification tests ────────────────────────────────────────────


class TestCommutativeUnification:
    """Test commutative (non-AC) unification."""

    def test_comm_unify_straight(self, symtab):
        """f(a, b) unifies with f(a, b) in straight order."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        _make_comm(symtab, "f")
        f_ab = _func(symtab, "f", a, b)

        c1 = Context()
        c2 = Context()
        results = commutative_unify_all(f_ab, c1, f_ab, c2)
        assert len(results) >= 1

    def test_comm_unify_flipped(self, symtab):
        """f(a, b) unifies with f(b, a) via commutativity."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        _make_comm(symtab, "f")
        f_ab = _func(symtab, "f", a, b)
        f_ba = _func(symtab, "f", b, a)

        c1 = Context()
        c2 = Context()
        results = commutative_unify_all(f_ab, c1, f_ba, c2)
        assert len(results) >= 1

    def test_comm_unify_with_vars(self, symtab):
        """f(x, a) unifies with f(b, y) two ways."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        x = _var(0)
        y = _var(1)
        _make_comm(symtab, "f")
        f_xa = _func(symtab, "f", x, a)
        f_by = _func(symtab, "f", b, y)

        c1 = Context()
        c2 = Context()
        results = commutative_unify_all(f_xa, c1, f_by, c2)
        # Should get at least one unifier: {x->b, y->a}
        assert len(results) >= 1

    def test_comm_match_straight(self, symtab):
        """f(x, a) matches f(b, a)."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        x = _var(0)
        _make_comm(symtab, "f")
        pattern = _func(symtab, "f", x, a)
        target = _func(symtab, "f", b, a)

        ctx = Context()
        results = commutative_match_all(pattern, ctx, target)
        assert len(results) >= 1

    def test_comm_match_flipped(self, symtab):
        """f(x, a) matches f(a, b) via commutativity → x=b."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        x = _var(0)
        _make_comm(symtab, "f")
        pattern = _func(symtab, "f", x, a)
        target = _func(symtab, "f", a, b)

        ctx = Context()
        results = commutative_match_all(pattern, ctx, target)
        assert len(results) >= 1


# ── AC unification tests ────────────────────────────────────────────────────


class TestACUnification:
    """Test full AC unification."""

    def test_ac_unify_identical_ground(self, symtab):
        """f(a, b) AC-unifies with f(a, b)."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        _make_ac(symtab, "+")
        is_ac = _is_ac(symtab)
        t = _ac_term(symtab, "+", a, b)

        c1 = Context()
        c2 = Context()
        results = ac_unify_all(t, c1, t, c2, is_ac)
        assert len(results) >= 1

    def test_ac_unify_commuted_ground(self, symtab):
        """f(a, b) AC-unifies with f(b, a)."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        _make_ac(symtab, "+")
        is_ac = _is_ac(symtab)
        t1 = _ac_term(symtab, "+", a, b)
        t2 = _ac_term(symtab, "+", b, a)

        c1 = Context()
        c2 = Context()
        results = ac_unify_all(t1, c1, t2, c2, is_ac)
        assert len(results) >= 1

    def test_ac_unify_associated_ground(self, symtab):
        """f(a, f(b, c)) AC-unifies with f(f(a, b), c)."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        c = _const(symtab, "c")
        _make_ac(symtab, "+")
        is_ac = _is_ac(symtab)
        f_bc = _ac_term(symtab, "+", b, c)
        t1 = _ac_term(symtab, "+", a, f_bc)
        f_ab = _ac_term(symtab, "+", a, b)
        t2 = _ac_term(symtab, "+", f_ab, c)

        c1 = Context()
        c2 = Context()
        results = ac_unify_all(t1, c1, t2, c2, is_ac)
        assert len(results) >= 1

    def test_ac_unify_variable_ground(self, symtab):
        """f(x, a) AC-unifies with f(b, a) → x=b."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        x = _var(0)
        _make_ac(symtab, "+")
        is_ac = _is_ac(symtab)
        t1 = _ac_term(symtab, "+", x, a)
        t2 = _ac_term(symtab, "+", b, a)

        c1 = Context()
        c2 = Context()
        results = ac_unify_all(t1, c1, t2, c2, is_ac)
        assert len(results) >= 1

    def test_ac_unify_no_match(self, symtab):
        """f(a, b) does not AC-unify with g(a, b) (different symbol)."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        _make_ac(symtab, "+")
        is_ac = _is_ac(symtab)
        t1 = _ac_term(symtab, "+", a, b)
        t2 = _func(symtab, "g", a, b)

        c1 = Context()
        c2 = Context()
        results = ac_unify_all(t1, c1, t2, c2, is_ac)
        assert len(results) == 0


# ── AC matching tests ────────────────────────────────────────────────────────


class TestACMatching:
    """Test AC matching."""

    def test_ac_match_ground_identical(self, symtab):
        """f(a, b) AC-matches f(a, b)."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        _make_ac(symtab, "+")
        is_ac = _is_ac(symtab)
        t = _ac_term(symtab, "+", a, b)

        ctx = Context()
        results = ac_match_all(t, ctx, t, is_ac)
        assert len(results) >= 1

    def test_ac_match_commuted(self, symtab):
        """f(a, b) AC-matches f(b, a)."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        _make_ac(symtab, "+")
        is_ac = _is_ac(symtab)
        t1 = _ac_term(symtab, "+", a, b)
        t2 = _ac_term(symtab, "+", b, a)

        ctx = Context()
        results = ac_match_all(t1, ctx, t2, is_ac)
        assert len(results) >= 1

    def test_ac_match_variable(self, symtab):
        """f(x, a) AC-matches f(b, a) → x=b."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        x = _var(0)
        _make_ac(symtab, "+")
        is_ac = _is_ac(symtab)
        pattern = _ac_term(symtab, "+", x, a)
        target = _ac_term(symtab, "+", b, a)

        ctx = Context()
        results = ac_match_all(pattern, ctx, target, is_ac)
        assert len(results) >= 1


# ── AC tautology tests ───────────────────────────────────────────────────────


class TestACTautology:
    """Test AC tautology detection."""

    def test_commuted_is_tautology(self, symtab):
        """f(a, b) = f(b, a) is a tautology modulo AC."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        _make_ac(symtab, "+")
        is_ac = _is_ac(symtab)
        t1 = _ac_term(symtab, "+", a, b)
        t2 = _ac_term(symtab, "+", b, a)
        assert is_ac_tautology(t1, t2, is_ac)

    def test_reassociated_is_tautology(self, symtab):
        """f(a, f(b, c)) = f(f(a, b), c) is a tautology modulo AC."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        c = _const(symtab, "c")
        _make_ac(symtab, "+")
        is_ac = _is_ac(symtab)
        f_bc = _ac_term(symtab, "+", b, c)
        t1 = _ac_term(symtab, "+", a, f_bc)
        f_ab = _ac_term(symtab, "+", a, b)
        t2 = _ac_term(symtab, "+", f_ab, c)
        assert is_ac_tautology(t1, t2, is_ac)

    def test_different_is_not_tautology(self, symtab):
        """f(a, b) = f(a, c) is NOT a tautology."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        c = _const(symtab, "c")
        _make_ac(symtab, "+")
        is_ac = _is_ac(symtab)
        t1 = _ac_term(symtab, "+", a, b)
        t2 = _ac_term(symtab, "+", a, c)
        assert not is_ac_tautology(t1, t2, is_ac)
