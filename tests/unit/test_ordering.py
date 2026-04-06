"""Tests for pyladr.ordering — KBO and LRPO term ordering.

Tests behavioral equivalence with C termorder.c:
- KBO weight calculation
- KBO comparison
- LRPO comparison (lexicographic and multiset)
- term_order dispatch
- Variable handling with lex_order_vars
"""

from __future__ import annotations

import pytest

from pyladr.core.symbol import LrpoStatus, SymbolTable
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.ordering.base import OrderMethod, Ordertype
from pyladr.ordering.kbo import kbo, kbo_weight
from pyladr.ordering.lrpo import lrpo
from pyladr.ordering.termorder import assign_order_method, term_greater, term_order


@pytest.fixture
def st() -> SymbolTable:
    """Symbol table with standard test symbols."""
    s = SymbolTable()
    # Register symbols: a=1, b=2, c=3, f=4(unary), g=5(binary), h=6(unary)
    s.str_to_sn("a", 0)  # sn=1
    s.str_to_sn("b", 0)  # sn=2
    s.str_to_sn("c", 0)  # sn=3
    s.str_to_sn("f", 1)  # sn=4
    s.str_to_sn("g", 2)  # sn=5
    s.str_to_sn("h", 1)  # sn=6
    # Set precedence: a < b < c < f < g < h
    for i, sn in enumerate([1, 2, 3, 4, 5, 6], start=1):
        sym = s.get_symbol(sn)
        sym.lex_val = i
    return s


def _a(st: SymbolTable) -> tuple:
    """Return commonly used test terms."""
    a = get_rigid_term(1, 0)
    b = get_rigid_term(2, 0)
    c = get_rigid_term(3, 0)
    x = get_variable_term(0)
    y = get_variable_term(1)
    return a, b, c, x, y


class TestKBOWeight:
    """Test KBO weight calculation (C kbo_weight)."""

    def test_variable_weight(self, st):
        """Variables have weight 1."""
        x = get_variable_term(0)
        assert kbo_weight(x, st) == 1

    def test_constant_weight(self, st):
        """Constants have their kb_weight (default 1)."""
        a = get_rigid_term(1, 0)
        assert kbo_weight(a, st) == 1

    def test_complex_weight(self, st):
        """f(a) has weight kb_weight(f) + weight(a) = 1 + 1 = 2."""
        a = get_rigid_term(1, 0)
        fa = get_rigid_term(4, 1, (a,))
        assert kbo_weight(fa, st) == 2

    def test_nested_weight(self, st):
        """g(a, b) has weight kb_weight(g) + weight(a) + weight(b) = 3."""
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        gab = get_rigid_term(5, 2, (a, b))
        assert kbo_weight(gab, st) == 3

    def test_custom_weight(self, st):
        """Symbol with custom kb_weight."""
        st.set_kb_weight(4, 3)  # f has weight 3
        a = get_rigid_term(1, 0)
        fa = get_rigid_term(4, 1, (a,))
        assert kbo_weight(fa, st) == 4  # 3 + 1
        st.set_kb_weight(4, 1)  # reset


class TestKBO:
    """Test Knuth-Bendix Ordering (C kbo)."""

    def test_heavier_term_greater(self, st):
        """g(a, b) > a because weight 3 > 1."""
        a, b, c, x, y = _a(st)
        gab = get_rigid_term(5, 2, (a, b))
        assert kbo(gab, a, False, st)
        assert not kbo(a, gab, False, st)

    def test_same_weight_higher_precedence(self, st):
        """b > a when same weight, higher precedence."""
        a, b, c, x, y = _a(st)
        assert kbo(b, a, False, st)
        assert not kbo(a, b, False, st)

    def test_same_symbol_lex_comparison(self, st):
        """f(b) > f(a) by lexicographic comparison on arguments."""
        a, b, c, x, y = _a(st)
        fa = get_rigid_term(4, 1, (a,))
        fb = get_rigid_term(4, 1, (b,))
        assert kbo(fb, fa, False, st)
        assert not kbo(fa, fb, False, st)

    def test_variable_not_greater(self, st):
        """Variable is never greater than anything (without lex_order_vars)."""
        a, b, c, x, y = _a(st)
        assert not kbo(x, a, False, st)
        assert not kbo(x, y, False, st)

    def test_term_greater_than_contained_variable(self, st):
        """f(x) > x because x occurs in f(x)."""
        a, b, c, x, y = _a(st)
        fx = get_rigid_term(4, 1, (x,))
        assert kbo(fx, x, False, st)

    def test_term_not_greater_than_unrelated_variable(self, st):
        """f(x) is NOT > y because y doesn't occur in f(x)."""
        a, b, c, x, y = _a(st)
        fx = get_rigid_term(4, 1, (x,))
        assert not kbo(fx, y, False, st)

    def test_variable_multisubset_required(self, st):
        """g(x, y) is NOT > f(x) because V({x}) is not multisubset of V({x,y})
        actually wait — V(beta=f(x))={x} IS multisubset of V(alpha=g(x,y))={x,y}.
        So g(x,y) > f(x) by weight (3 > 2).
        """
        a, b, c, x, y = _a(st)
        gxy = get_rigid_term(5, 2, (x, y))
        fx = get_rigid_term(4, 1, (x,))
        assert kbo(gxy, fx, False, st)

    def test_variable_multisubset_fails(self, st):
        """f(x) is NOT > g(y, y) because V(g(y,y))={y,y} is not
        multisubset of V(f(x))={x}."""
        a, b, c, x, y = _a(st)
        fx = get_rigid_term(4, 1, (x,))
        gyy = get_rigid_term(5, 2, (y, y))
        assert not kbo(fx, gyy, False, st)

    def test_identical_terms(self, st):
        """a is NOT > a."""
        a, b, c, x, y = _a(st)
        assert not kbo(a, a, False, st)

    def test_lex_order_vars(self, st):
        """With lex_order_vars: v1 > v0."""
        a, b, c, x, y = _a(st)
        assert kbo(y, x, True, st)
        assert not kbo(x, y, True, st)

    def test_unary_same_symbol_optimization(self, st):
        """f(f(a)) > f(a) — unary same symbol recurses directly."""
        a, b, c, x, y = _a(st)
        fa = get_rigid_term(4, 1, (a,))
        ffa = get_rigid_term(4, 1, (fa,))
        assert kbo(ffa, fa, False, st)


class TestLRPO:
    """Test LRPO (Lexicographic Recursive Path Ordering)."""

    def test_higher_precedence_greater(self, st):
        """b > a when precedence(b) > precedence(a)."""
        a, b, c, x, y = _a(st)
        assert lrpo(b, a, False, st)
        assert not lrpo(a, b, False, st)

    def test_variable_not_greater(self, st):
        """Variable is never greater (without lex_order_vars)."""
        a, b, c, x, y = _a(st)
        assert not lrpo(x, a, False, st)

    def test_contains_variable(self, st):
        """f(x) > x because x occurs in f(x)."""
        a, b, c, x, y = _a(st)
        fx = get_rigid_term(4, 1, (x,))
        assert lrpo(fx, x, False, st)

    def test_not_contains_unrelated_var(self, st):
        """f(x) is NOT > y because y doesn't occur in f(x)."""
        a, b, c, x, y = _a(st)
        fx = get_rigid_term(4, 1, (x,))
        assert not lrpo(fx, y, False, st)

    def test_same_symbol_lr_status_lex(self, st):
        """g(b, a) > g(a, b) by lexicographic comparison (first arg differs)."""
        a, b, c, x, y = _a(st)
        gba = get_rigid_term(5, 2, (b, a))
        gab = get_rigid_term(5, 2, (a, b))
        # g has LR_STATUS by default
        assert lrpo(gba, gab, False, st)
        assert not lrpo(gab, gba, False, st)

    def test_same_symbol_multiset_status(self, st):
        """With multiset status, g(b, a) and g(a, b) are compared as multisets."""
        a, b, c, x, y = _a(st)
        st.set_lrpo_status(5, LrpoStatus.MULTISET_STATUS)
        gba = get_rigid_term(5, 2, (b, a))
        gab = get_rigid_term(5, 2, (a, b))
        # As multisets {a, b} = {a, b}, so neither is greater
        assert not lrpo(gba, gab, False, st)
        assert not lrpo(gab, gba, False, st)
        st.set_lrpo_status(5, LrpoStatus.LR_STATUS)  # reset

    def test_higher_prec_must_beat_all_args(self, st):
        """h > g requires h > each arg of g.
        h(a) > g(a, a)? h has higher prec. Need h(a) > a twice. Yes."""
        a, b, c, x, y = _a(st)
        ha = get_rigid_term(6, 1, (a,))
        gaa = get_rigid_term(5, 2, (a, a))
        assert lrpo(ha, gaa, False, st)

    def test_lex_order_vars(self, st):
        """With lex_order_vars: v1 > v0."""
        a, b, c, x, y = _a(st)
        assert lrpo(y, x, True, st)
        assert not lrpo(x, y, True, st)

    def test_identical_terms(self, st):
        """a is NOT > a."""
        a, b, c, x, y = _a(st)
        assert not lrpo(a, a, False, st)


class TestTermOrder:
    """Test term_order dispatch (C term_order, term_greater)."""

    def test_kbo_dispatch(self, st):
        a, b, c, x, y = _a(st)
        assign_order_method(OrderMethod.KBO)
        assert term_greater(b, a, False, st)
        assert not term_greater(a, b, False, st)

    def test_lrpo_dispatch(self, st):
        a, b, c, x, y = _a(st)
        assign_order_method(OrderMethod.LRPO)
        assert term_greater(b, a, False, st)
        assert not term_greater(a, b, False, st)

    def test_term_order_greater(self, st):
        a, b, c, x, y = _a(st)
        assign_order_method(OrderMethod.LRPO)
        assert term_order(b, a, st) == Ordertype.GREATER_THAN

    def test_term_order_less(self, st):
        a, b, c, x, y = _a(st)
        assign_order_method(OrderMethod.LRPO)
        assert term_order(a, b, st) == Ordertype.LESS_THAN

    def test_term_order_same(self, st):
        a, b, c, x, y = _a(st)
        assign_order_method(OrderMethod.LRPO)
        assert term_order(a, a, st) == Ordertype.SAME_AS

    def test_term_order_not_comparable(self, st):
        """f(x) and f(y) are not comparable (different variables)."""
        a, b, c, x, y = _a(st)
        assign_order_method(OrderMethod.LRPO)
        fx = get_rigid_term(4, 1, (x,))
        fy = get_rigid_term(4, 1, (y,))
        assert term_order(fx, fy, st) == Ordertype.NOT_COMPARABLE
