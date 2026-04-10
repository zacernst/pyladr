"""Property-based tests for term orderings (KBO and LRPO).

Verifies algebraic properties required for well-founded simplification orderings:
- Irreflexivity: NOT (t > t)
- Asymmetry: if t1 > t2 then NOT (t2 > t1)
- Transitivity: if t1 > t2 and t2 > t3 then t1 > t3
- Subterm property: t > s for proper subterms s of t (with caveats for KBO)
- Stability under substitution (for ground terms)
"""

from __future__ import annotations

import pytest

from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.ordering.kbo import kbo
from pyladr.ordering.lrpo import lrpo


@pytest.fixture
def st() -> SymbolTable:
    """Symbol table for ordering tests."""
    s = SymbolTable()
    s.str_to_sn("a", 0)  # sn=1
    s.str_to_sn("b", 0)  # sn=2
    s.str_to_sn("c", 0)  # sn=3
    s.str_to_sn("f", 1)  # sn=4
    s.str_to_sn("g", 2)  # sn=5
    s.str_to_sn("h", 1)  # sn=6
    for i, sn in enumerate([1, 2, 3, 4, 5, 6], start=1):
        sym = s.get_symbol(sn)
        sym.lex_val = i
    return s


def _ground_terms() -> list[Term]:
    """Generate a set of ground terms for property testing."""
    a = get_rigid_term(1, 0)
    b = get_rigid_term(2, 0)
    c = get_rigid_term(3, 0)
    f = lambda arg: get_rigid_term(4, 1, (arg,))
    g = lambda l, r: get_rigid_term(5, 2, (l, r))
    h = lambda arg: get_rigid_term(6, 1, (arg,))

    return [
        a, b, c,
        f(a), f(b), f(c),
        h(a), h(b),
        g(a, b), g(b, a), g(a, a),
        f(f(a)), f(g(a, b)),
        g(f(a), b), g(a, f(b)),
    ]


class TestKBOProperties:
    """Property-based tests for Knuth-Bendix Ordering."""

    @pytest.mark.parametrize("t", _ground_terms())
    def test_irreflexivity(self, t, st):
        """NOT (t > t) for any term."""
        assert not kbo(t, t, False, st)

    def test_asymmetry(self, st):
        """If t1 > t2 then NOT (t2 > t1)."""
        terms = _ground_terms()
        for t1 in terms:
            for t2 in terms:
                if kbo(t1, t2, False, st):
                    assert not kbo(t2, t1, False, st), (
                        f"Asymmetry violation: {t1} > {t2} and {t2} > {t1}"
                    )

    def test_transitivity(self, st):
        """If t1 > t2 and t2 > t3 then t1 > t3."""
        terms = _ground_terms()
        greater_pairs = []
        for t1 in terms:
            for t2 in terms:
                if kbo(t1, t2, False, st):
                    greater_pairs.append((t1, t2))

        for t1, t2 in greater_pairs:
            for t3 in terms:
                if kbo(t2, t3, False, st):
                    assert kbo(t1, t3, False, st), (
                        f"Transitivity violation: {t1} > {t2} > {t3} "
                        f"but NOT {t1} > {t3}"
                    )


class TestLRPOProperties:
    """Property-based tests for LRPO."""

    @pytest.mark.parametrize("t", _ground_terms())
    def test_irreflexivity(self, t, st):
        """NOT (t > t) for any term."""
        assert not lrpo(t, t, False, st)

    def test_asymmetry(self, st):
        """If t1 > t2 then NOT (t2 > t1)."""
        terms = _ground_terms()
        for t1 in terms:
            for t2 in terms:
                if lrpo(t1, t2, False, st):
                    assert not lrpo(t2, t1, False, st), (
                        f"Asymmetry violation: {t1} > {t2} and {t2} > {t1}"
                    )

    def test_transitivity(self, st):
        """If t1 > t2 and t2 > t3 then t1 > t3."""
        terms = _ground_terms()
        greater_pairs = []
        for t1 in terms:
            for t2 in terms:
                if lrpo(t1, t2, False, st):
                    greater_pairs.append((t1, t2))

        for t1, t2 in greater_pairs:
            for t3 in terms:
                if lrpo(t2, t3, False, st):
                    assert lrpo(t1, t3, False, st), (
                        f"Transitivity violation: {t1} > {t2} > {t3} "
                        f"but NOT {t1} > {t3}"
                    )

    def test_subterm_property(self, st):
        """f(t) > t for any ground term t and function symbol f with higher precedence."""
        terms = _ground_terms()
        for t in terms:
            ft = get_rigid_term(4, 1, (t,))  # f(t)
            assert lrpo(ft, t, False, st), (
                f"Subterm property: f({t}) should be > {t}"
            )

    def test_subterm_property_binary(self, st):
        """g(t, s) > t and g(t, s) > s for ground terms."""
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        gab = get_rigid_term(5, 2, (a, b))
        assert lrpo(gab, a, False, st)
        assert lrpo(gab, b, False, st)
