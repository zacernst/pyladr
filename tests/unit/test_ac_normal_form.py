"""Tests for pyladr.core.ac_normal_form — AC canonical form computation.

Tests behavioral equivalence with C accanon.c:
- Flattening nested AC terms
- Right-association of flattened argument lists
- Canonical ordering (CONSTANT < COMPLEX < VARIABLE)
- Full AC canonicalization
- Flatten with multiplicities
"""

from __future__ import annotations

import pytest

from pyladr.core.ac_normal_form import (
    ac_canonical,
    flatten_ac,
    flatten_with_multiplicities,
    right_associate,
    term_compare_ncv,
)
from pyladr.core.term import Term, get_rigid_term, get_variable_term


# Symbol numbers used throughout tests
SN_PLUS = -1   # private_symbol for '+'
SN_STAR = -2   # private_symbol for '*'
SN_F = -3
SN_A = -4
SN_B = -5
SN_C = -6
SN_D = -7


def _const(sn: int) -> Term:
    return get_rigid_term(sn, 0)


def _bin(sn_priv: int, left: Term, right: Term) -> Term:
    """Build binary term with given private_symbol."""
    return Term(private_symbol=sn_priv, arity=2, args=(left, right))


def _unary(sn_priv: int, arg: Term) -> Term:
    return Term(private_symbol=sn_priv, arity=1, args=(arg,))


# Convenience: use positive symnums for constants/functions
def _a() -> Term:
    return get_rigid_term(1, 0)  # constant a


def _b() -> Term:
    return get_rigid_term(2, 0)  # constant b


def _c() -> Term:
    return get_rigid_term(3, 0)  # constant c


def _d() -> Term:
    return get_rigid_term(4, 0)  # constant d


def _f(arg: Term) -> Term:
    return get_rigid_term(5, 1, (arg,))  # f(arg)


def _plus(left: Term, right: Term) -> Term:
    return get_rigid_term(10, 2, (left, right))  # +(left, right)


def _star(left: Term, right: Term) -> Term:
    return get_rigid_term(11, 2, (left, right))  # *(left, right)


def _is_plus(symnum: int) -> bool:
    return symnum == 10


def _is_star(symnum: int) -> bool:
    return symnum == 11


def _is_ac(symnum: int) -> bool:
    return symnum in (10, 11)


# ── Flatten tests ───────────────────────────────────────────────────────────


class TestFlattenAC:
    """Test flattening nested AC terms."""

    def test_flatten_leaf(self):
        """Non-AC term returns itself."""
        a = _a()
        result = flatten_ac(a, _plus(a, a).private_symbol)
        assert len(result) == 1
        assert result[0] is a

    def test_flatten_binary(self):
        """+(a, b) flattens to [a, b]."""
        a, b = _a(), _b()
        t = _plus(a, b)
        result = flatten_ac(t, t.private_symbol)
        assert len(result) == 2

    def test_flatten_nested_left(self):
        """+(+(a, b), c) flattens to [a, b, c]."""
        a, b, c = _a(), _b(), _c()
        t = _plus(_plus(a, b), c)
        result = flatten_ac(t, t.private_symbol)
        assert len(result) == 3

    def test_flatten_nested_right(self):
        """+(a, +(b, c)) flattens to [a, b, c]."""
        a, b, c = _a(), _b(), _c()
        t = _plus(a, _plus(b, c))
        result = flatten_ac(t, t.private_symbol)
        assert len(result) == 3

    def test_flatten_deeply_nested(self):
        """+(+(a, b), +(c, d)) flattens to [a, b, c, d]."""
        a, b, c, d = _a(), _b(), _c(), _d()
        t = _plus(_plus(a, b), _plus(c, d))
        result = flatten_ac(t, t.private_symbol)
        assert len(result) == 4

    def test_flatten_mixed_ac_symbols(self):
        """+(*(a, b), c) — only flattens + not *."""
        a, b, c = _a(), _b(), _c()
        star_ab = _star(a, b)
        t = _plus(star_ab, c)
        result = flatten_ac(t, t.private_symbol)
        assert len(result) == 2  # *(a,b) and c, not 3

    def test_flatten_variable(self):
        """Variable is a leaf."""
        x = get_variable_term(0)
        result = flatten_ac(x, _plus(_a(), _b()).private_symbol)
        assert len(result) == 1
        assert result[0] is x

    def test_flatten_with_variable_args(self):
        """+(x, +(a, y)) flattens to [x, a, y]."""
        x, y = get_variable_term(0), get_variable_term(1)
        a = _a()
        t = _plus(x, _plus(a, y))
        result = flatten_ac(t, t.private_symbol)
        assert len(result) == 3


# ── Right associate tests ────────────────────────────────────────────────────


class TestRightAssociate:
    """Test building right-associated terms from argument lists."""

    def test_single_arg(self):
        """Single arg returns that arg."""
        a = _a()
        result = right_associate(_plus(a, a).private_symbol, 2, [a])
        assert result is a

    def test_two_args(self):
        """[a, b] → +(a, b)."""
        a, b = _a(), _b()
        ps = _plus(a, b).private_symbol
        result = right_associate(ps, 2, [a, b])
        assert result.arity == 2
        assert result.args[0].term_ident(a)
        assert result.args[1].term_ident(b)

    def test_three_args(self):
        """[a, b, c] → +(a, +(b, c))."""
        a, b, c = _a(), _b(), _c()
        ps = _plus(a, b).private_symbol
        result = right_associate(ps, 2, [a, b, c])
        assert result.arity == 2
        assert result.args[0].term_ident(a)
        # Second arg is +(b, c)
        inner = result.args[1]
        assert inner.arity == 2
        assert inner.args[0].term_ident(b)
        assert inner.args[1].term_ident(c)

    def test_four_args(self):
        """[a, b, c, d] → +(a, +(b, +(c, d)))."""
        a, b, c, d = _a(), _b(), _c(), _d()
        ps = _plus(a, b).private_symbol
        result = right_associate(ps, 2, [a, b, c, d])
        assert result.arity == 2
        assert result.args[0].term_ident(a)
        # Check nesting depth
        inner = result.args[1]
        assert inner.args[0].term_ident(b)
        inner2 = inner.args[1]
        assert inner2.args[0].term_ident(c)
        assert inner2.args[1].term_ident(d)

    def test_empty_raises(self):
        ps = _plus(_a(), _b()).private_symbol
        with pytest.raises(ValueError, match="empty"):
            right_associate(ps, 2, [])


# ── Canonical ordering tests ────────────────────────────────────────────────


class TestTermCompareNCV:
    """Test NAME < COMPLEX < VARIABLE ordering (C compare_ncv)."""

    def test_constant_less_than_complex(self):
        a = _a()
        fa = _f(a)
        assert term_compare_ncv(a, fa) == -1
        assert term_compare_ncv(fa, a) == 1

    def test_constant_less_than_variable(self):
        a = _a()
        x = get_variable_term(0)
        assert term_compare_ncv(a, x) == -1
        assert term_compare_ncv(x, a) == 1

    def test_complex_less_than_variable(self):
        fa = _f(_a())
        x = get_variable_term(0)
        assert term_compare_ncv(fa, x) == -1
        assert term_compare_ncv(x, fa) == 1

    def test_same_constant(self):
        a1 = get_rigid_term(1, 0)
        a2 = get_rigid_term(1, 0)
        assert term_compare_ncv(a1, a2) == 0

    def test_different_constants_by_symnum(self):
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        assert term_compare_ncv(a, b) == -1
        assert term_compare_ncv(b, a) == 1

    def test_same_variable(self):
        x = get_variable_term(3)
        assert term_compare_ncv(x, x) == 0

    def test_different_variables_by_varnum(self):
        x = get_variable_term(0)
        y = get_variable_term(1)
        assert term_compare_ncv(x, y) == -1
        assert term_compare_ncv(y, x) == 1

    def test_complex_by_symnum(self):
        """f(a) < g(a) when symnum(f) < symnum(g)."""
        a = _a()
        f_a = get_rigid_term(5, 1, (a,))
        g_a = get_rigid_term(6, 1, (a,))
        assert term_compare_ncv(f_a, g_a) == -1

    def test_complex_lexicographic(self):
        """f(a, b) < f(a, c) by lexicographic comparison."""
        a, b, c = _a(), _b(), _c()
        fab = get_rigid_term(5, 2, (a, b))
        fac = get_rigid_term(5, 2, (a, c))
        assert term_compare_ncv(fab, fac) == -1


# ── AC canonical tests ──────────────────────────────────────────────────────


class TestACCanonical:
    """Test full AC canonicalization."""

    def test_non_ac_unchanged(self):
        """Non-AC term is returned unchanged (up to recursive canonicalization)."""
        a = _a()
        result = ac_canonical(a, _is_ac)
        assert result.term_ident(a)

    def test_variable_unchanged(self):
        x = get_variable_term(0)
        result = ac_canonical(x, _is_ac)
        assert result is x

    def test_already_canonical(self):
        """+(a, b) where a < b is already canonical."""
        a, b = _a(), _b()
        t = _plus(a, b)
        result = ac_canonical(t, _is_ac)
        # Should still be +(a, b) since a(sn=1) < b(sn=2)
        assert result.arity == 2
        assert result.args[0].term_ident(a)
        assert result.args[1].term_ident(b)

    def test_swap_needed(self):
        """+(b, a) → +(a, b) when a < b in canonical order."""
        a, b = _a(), _b()
        t = _plus(b, a)
        result = ac_canonical(t, _is_ac)
        assert result.args[0].term_ident(a)
        assert result.args[1].term_ident(b)

    def test_flatten_and_sort(self):
        """+(+(c, a), b) → +(a, +(b, c)) (flatten, sort, right-associate)."""
        a, b, c = _a(), _b(), _c()
        t = _plus(_plus(c, a), b)
        result = ac_canonical(t, _is_ac)
        # Should be +(a, +(b, c))
        flat = flatten_ac(result, result.private_symbol)
        assert len(flat) == 3
        assert flat[0].term_ident(a)
        assert flat[1].term_ident(b)
        assert flat[2].term_ident(c)

    def test_variables_sorted_last(self):
        """+(x, a) → +(a, x) since constants < variables."""
        a = _a()
        x = get_variable_term(0)
        t = _plus(x, a)
        result = ac_canonical(t, _is_ac)
        assert result.args[0].term_ident(a)
        assert result.args[1].term_ident(x)

    def test_nested_non_ac(self):
        """f(+(b, a)) → f(+(a, b)): canonicalize inside non-AC wrapper."""
        a, b = _a(), _b()
        inner = _plus(b, a)
        t = _f(inner)
        result = ac_canonical(t, _is_ac)
        assert result.arity == 1
        inner_result = result.args[0]
        assert inner_result.args[0].term_ident(a)
        assert inner_result.args[1].term_ident(b)

    def test_idempotent(self):
        """Canonicalizing a canonical term returns an identical term."""
        a, b, c = _a(), _b(), _c()
        t = _plus(_plus(c, a), b)
        canonical = ac_canonical(t, _is_ac)
        canonical2 = ac_canonical(canonical, _is_ac)
        assert canonical.term_ident(canonical2)


# ── Flatten with multiplicities tests ────────────────────────────────────────


class TestFlattenWithMultiplicities:
    """Test flatten_with_multiplicities for AC matching."""

    def test_no_duplicates(self):
        """+(a, b) → [(a, 1), (b, 1)]."""
        a, b = _a(), _b()
        t = _plus(a, b)
        result = flatten_with_multiplicities(t, t.private_symbol)
        assert len(result) == 2
        assert all(count == 1 for _, count in result)

    def test_with_duplicates(self):
        """+(a, +(a, b)) → [(a, 2), (b, 1)]."""
        a, b = _a(), _b()
        t = _plus(a, _plus(a, b))
        result = flatten_with_multiplicities(t, t.private_symbol)
        # After sorting: a appears twice, b once
        found_a = False
        found_b = False
        for term, count in result:
            if term.term_ident(a):
                assert count == 2
                found_a = True
            if term.term_ident(b):
                assert count == 1
                found_b = True
        assert found_a and found_b

    def test_all_same(self):
        """+(a, +(a, a)) → [(a, 3)]."""
        a = _a()
        t = _plus(a, _plus(a, a))
        result = flatten_with_multiplicities(t, t.private_symbol)
        assert len(result) == 1
        assert result[0][1] == 3
