"""Tests for subsumption system matching C subsume.c behavior."""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.inference.subsumption import (
    back_subsume_from_lists,
    forward_subsume_from_lists,
    reset_subsumption_stats,
    subsumes,
)
from pyladr.indexing.feature_index import FeatureIndex, FeatureVector


# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_stats():
    """Reset subsumption stats before each test."""
    reset_subsumption_stats()


@pytest.fixture()
def symtab():
    """Fresh symbol table for each test."""
    return SymbolTable()


def _var(n: int) -> Term:
    return get_variable_term(n)


def _const(symtab: SymbolTable, name: str) -> Term:
    sn = symtab.str_to_sn(name, 0)
    return get_rigid_term(sn, 0)


def _func(symtab: SymbolTable, name: str, *args: Term) -> Term:
    sn = symtab.str_to_sn(name, len(args))
    return get_rigid_term(sn, len(args), args)


def _pos_lit(atom: Term) -> Literal:
    return Literal(sign=True, atom=atom)


def _neg_lit(atom: Term) -> Literal:
    return Literal(sign=False, atom=atom)


def _clause(*lits: Literal, cid: int = 0) -> Clause:
    c = Clause(literals=tuple(lits))
    c.id = cid
    return c


# ── subsumes() unit tests ───────────────────────────────────────────────────


class TestSubsumes:
    """Test the core subsumes() function."""

    def test_empty_subsumes_anything(self, symtab):
        """Empty clause subsumes everything."""
        empty = _clause()
        a = _const(symtab, "a")
        unit = _clause(_pos_lit(a))
        assert subsumes(empty, unit)

    def test_identical_unit_clauses(self, symtab):
        """P(a) subsumes P(a)."""
        a = _const(symtab, "a")
        p_a = _func(symtab, "P", a)
        c1 = _clause(_pos_lit(p_a))
        c2 = _clause(_pos_lit(p_a))
        assert subsumes(c1, c2)

    def test_unit_variable_subsumes_ground(self, symtab):
        """P(x) subsumes P(a) via {x -> a}."""
        a = _const(symtab, "a")
        x = _var(0)
        p_x = _func(symtab, "P", x)
        p_a = _func(symtab, "P", a)
        c1 = _clause(_pos_lit(p_x))
        c2 = _clause(_pos_lit(p_a))
        assert subsumes(c1, c2)

    def test_ground_does_not_subsume_variable(self, symtab):
        """P(a) does NOT subsume P(x) — cannot match a to variable."""
        a = _const(symtab, "a")
        x = _var(0)
        p_a = _func(symtab, "P", a)
        p_x = _func(symtab, "P", x)
        c1 = _clause(_pos_lit(p_a))
        c2 = _clause(_pos_lit(p_x))
        assert not subsumes(c1, c2)

    def test_sign_mismatch(self, symtab):
        """P(x) does not subsume ~P(a) — sign differs."""
        a = _const(symtab, "a")
        x = _var(0)
        p_x = _func(symtab, "P", x)
        p_a = _func(symtab, "P", a)
        c1 = _clause(_pos_lit(p_x))
        c2 = _clause(_neg_lit(p_a))
        assert not subsumes(c1, c2)

    def test_longer_cannot_subsume_shorter(self, symtab):
        """{P(x), Q(x)} cannot subsume {P(a)}."""
        a = _const(symtab, "a")
        x = _var(0)
        p_x = _func(symtab, "P", x)
        q_x = _func(symtab, "Q", x)
        p_a = _func(symtab, "P", a)
        c1 = _clause(_pos_lit(p_x), _pos_lit(q_x))
        c2 = _clause(_pos_lit(p_a))
        assert not subsumes(c1, c2)

    def test_unit_subsumes_multi_literal(self, symtab):
        """P(x) subsumes {P(a), Q(b)} via {x -> a}."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        x = _var(0)
        p_x = _func(symtab, "P", x)
        p_a = _func(symtab, "P", a)
        q_b = _func(symtab, "Q", b)
        c1 = _clause(_pos_lit(p_x))
        c2 = _clause(_pos_lit(p_a), _pos_lit(q_b))
        assert subsumes(c1, c2)

    def test_nonunit_subsumption_with_shared_variable(self, symtab):
        """{P(x), Q(x)} subsumes {P(a), Q(a)} via {x -> a}."""
        a = _const(symtab, "a")
        x = _var(0)
        p_x = _func(symtab, "P", x)
        q_x = _func(symtab, "Q", x)
        p_a = _func(symtab, "P", a)
        q_a = _func(symtab, "Q", a)
        c1 = _clause(_pos_lit(p_x), _pos_lit(q_x))
        c2 = _clause(_pos_lit(p_a), _pos_lit(q_a))
        assert subsumes(c1, c2)

    def test_nonunit_subsumption_no_consistent_binding(self, symtab):
        """{P(x), Q(x)} does NOT subsume {P(a), Q(b)} — x can't map to both a and b."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        x = _var(0)
        p_x = _func(symtab, "P", x)
        q_x = _func(symtab, "Q", x)
        p_a = _func(symtab, "P", a)
        q_b = _func(symtab, "Q", b)
        c1 = _clause(_pos_lit(p_x), _pos_lit(q_x))
        c2 = _clause(_pos_lit(p_a), _pos_lit(q_b))
        assert not subsumes(c1, c2)

    def test_nonunit_independent_variables(self, symtab):
        """{P(x), Q(y)} subsumes {P(a), Q(b)} via {x->a, y->b}."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        x = _var(0)
        y = _var(1)
        p_x = _func(symtab, "P", x)
        q_y = _func(symtab, "Q", y)
        p_a = _func(symtab, "P", a)
        q_b = _func(symtab, "Q", b)
        c1 = _clause(_pos_lit(p_x), _pos_lit(q_y))
        c2 = _clause(_pos_lit(p_a), _pos_lit(q_b))
        assert subsumes(c1, c2)

    def test_subsumption_with_backtracking(self, symtab):
        """Tests backtracking: first literal match may fail, need to retry.

        {P(x), P(f(x))} subsumes {P(a), P(f(a)), P(b)}
        First try: x->a matches P(a), then P(f(a)) needs to match — succeeds.
        """
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        x = _var(0)
        f_x = _func(symtab, "f", x)
        f_a = _func(symtab, "f", a)
        p_x = _func(symtab, "P", x)
        p_fx = _func(symtab, "P", f_x)
        p_a = _func(symtab, "P", a)
        p_fa = _func(symtab, "P", f_a)
        p_b = _func(symtab, "P", b)
        c1 = _clause(_pos_lit(p_x), _pos_lit(p_fx))
        c2 = _clause(_pos_lit(p_a), _pos_lit(p_fa), _pos_lit(p_b))
        assert subsumes(c1, c2)

    def test_subsumption_negative_literals(self, symtab):
        """~P(x) subsumes ~P(a)."""
        a = _const(symtab, "a")
        x = _var(0)
        p_x = _func(symtab, "P", x)
        p_a = _func(symtab, "P", a)
        c1 = _clause(_neg_lit(p_x))
        c2 = _clause(_neg_lit(p_a))
        assert subsumes(c1, c2)

    def test_mixed_sign_subsumption(self, symtab):
        """{P(x), ~Q(x)} subsumes {P(a), ~Q(a), R(b)}."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        x = _var(0)
        p_x = _func(symtab, "P", x)
        q_x = _func(symtab, "Q", x)
        r_b = _func(symtab, "R", b)
        p_a = _func(symtab, "P", a)
        q_a = _func(symtab, "Q", a)
        c1 = _clause(_pos_lit(p_x), _neg_lit(q_x))
        c2 = _clause(_pos_lit(p_a), _neg_lit(q_a), _pos_lit(r_b))
        assert subsumes(c1, c2)

    def test_self_subsumption(self, symtab):
        """A clause subsumes itself (variant check)."""
        x = _var(0)
        p_x = _func(symtab, "P", x)
        c = _clause(_pos_lit(p_x))
        assert subsumes(c, c)

    def test_variant_subsumption(self, symtab):
        """P(x) subsumes P(y) and vice versa (both are variants)."""
        x = _var(0)
        y = _var(1)
        p_x = _func(symtab, "P", x)
        p_y = _func(symtab, "P", y)
        c1 = _clause(_pos_lit(p_x))
        c2 = _clause(_pos_lit(p_y))
        assert subsumes(c1, c2)
        assert subsumes(c2, c1)

    def test_nested_function_subsumption(self, symtab):
        """P(f(x, g(y))) subsumes P(f(a, g(b)))."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        x = _var(0)
        y = _var(1)
        g_y = _func(symtab, "g", y)
        g_b = _func(symtab, "g", b)
        f_x_gy = _func(symtab, "f", x, g_y)
        f_a_gb = _func(symtab, "f", a, g_b)
        p1 = _func(symtab, "P", f_x_gy)
        p2 = _func(symtab, "P", f_a_gb)
        c1 = _clause(_pos_lit(p1))
        c2 = _clause(_pos_lit(p2))
        assert subsumes(c1, c2)


# ── Forward/backward subsumption from lists ──────────────────────────────────


class TestForwardSubsume:
    """Test forward subsumption (is new clause subsumed by existing?)."""

    def test_not_subsumed_empty_list(self, symtab):
        a = _const(symtab, "a")
        p_a = _func(symtab, "P", a)
        c = _clause(_pos_lit(p_a))
        assert forward_subsume_from_lists(c, []) is None

    def test_subsumed_by_more_general(self, symtab):
        """P(x) in list subsumes P(a)."""
        a = _const(symtab, "a")
        x = _var(0)
        p_x = _func(symtab, "P", x)
        p_a = _func(symtab, "P", a)
        general = _clause(_pos_lit(p_x), cid=1)
        specific = _clause(_pos_lit(p_a), cid=2)
        result = forward_subsume_from_lists(specific, [[general]])
        assert result is general

    def test_not_subsumed_different_predicate(self, symtab):
        a = _const(symtab, "a")
        p_a = _func(symtab, "P", a)
        q_a = _func(symtab, "Q", a)
        c1 = _clause(_pos_lit(p_a), cid=1)
        c2 = _clause(_pos_lit(q_a), cid=2)
        assert forward_subsume_from_lists(c2, [[c1]]) is None

    def test_subsumed_across_multiple_lists(self, symtab):
        """Check multiple lists for subsumer."""
        a = _const(symtab, "a")
        x = _var(0)
        p_x = _func(symtab, "P", x)
        p_a = _func(symtab, "P", a)
        subsumer = _clause(_pos_lit(p_x), cid=1)
        target = _clause(_pos_lit(p_a), cid=2)
        assert forward_subsume_from_lists(target, [[], [subsumer]]) is subsumer


class TestBackSubsume:
    """Test backward subsumption (does new clause subsume existing?)."""

    def test_back_subsume_empty(self, symtab):
        a = _const(symtab, "a")
        p_a = _func(symtab, "P", a)
        c = _clause(_pos_lit(p_a))
        assert back_subsume_from_lists(c, []) == []

    def test_back_subsume_finds_victims(self, symtab):
        """P(x) subsumes P(a) — should find P(a) in backward search."""
        a = _const(symtab, "a")
        x = _var(0)
        p_x = _func(symtab, "P", x)
        p_a = _func(symtab, "P", a)
        general = _clause(_pos_lit(p_x), cid=1)
        specific = _clause(_pos_lit(p_a), cid=2)
        victims = back_subsume_from_lists(general, [[specific]])
        assert specific in victims

    def test_back_subsume_skips_self(self, symtab):
        """Should not report self as subsumed."""
        x = _var(0)
        p_x = _func(symtab, "P", x)
        c = _clause(_pos_lit(p_x), cid=1)
        victims = back_subsume_from_lists(c, [[c]])
        assert c not in victims

    def test_back_subsume_multiple_victims(self, symtab):
        """P(x) subsumes both P(a) and P(b)."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        x = _var(0)
        p_x = _func(symtab, "P", x)
        p_a = _func(symtab, "P", a)
        p_b = _func(symtab, "P", b)
        general = _clause(_pos_lit(p_x), cid=1)
        s1 = _clause(_pos_lit(p_a), cid=2)
        s2 = _clause(_pos_lit(p_b), cid=3)
        victims = back_subsume_from_lists(general, [[s1, s2]])
        assert s1 in victims
        assert s2 in victims


# ── Feature vector tests ─────────────────────────────────────────────────────


class TestFeatureVector:
    """Test feature vector prefiltering."""

    def test_basic_feature_computation(self, symtab):
        a = _const(symtab, "a")
        p_a = _func(symtab, "P", a)
        c = _clause(_pos_lit(p_a))
        fv = FeatureVector.from_clause(c)
        assert fv.pos_lit_count == 1
        assert fv.neg_lit_count == 0

    def test_can_subsume_positive(self, symtab):
        """Fewer literals can subsume more literals."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        p_a = _func(symtab, "P", a)
        q_b = _func(symtab, "Q", b)
        c1 = _clause(_pos_lit(p_a))
        c2 = _clause(_pos_lit(p_a), _pos_lit(q_b))
        fv1 = FeatureVector.from_clause(c1)
        fv2 = FeatureVector.from_clause(c2)
        assert fv1.can_subsume(fv2)

    def test_can_subsume_negative(self, symtab):
        """More literals cannot subsume fewer."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        p_a = _func(symtab, "P", a)
        q_b = _func(symtab, "Q", b)
        c1 = _clause(_pos_lit(p_a), _pos_lit(q_b))
        c2 = _clause(_pos_lit(p_a))
        fv1 = FeatureVector.from_clause(c1)
        fv2 = FeatureVector.from_clause(c2)
        assert not fv1.can_subsume(fv2)

    def test_feature_index_forward(self, symtab):
        """Feature index returns valid forward candidates."""
        a = _const(symtab, "a")
        x = _var(0)
        p_x = _func(symtab, "P", x)
        p_a = _func(symtab, "P", a)
        general = _clause(_pos_lit(p_x), cid=1)
        specific = _clause(_pos_lit(p_a), cid=2)
        idx = FeatureIndex()
        idx.insert(general)
        candidates = idx.forward_candidates(specific)
        assert general in candidates


# ── Literal index tests ──────────────────────────────────────────────────────


class TestLiteralIndex:
    """Test literal index for subsumption."""

    def test_insert_and_check(self, symtab):
        from pyladr.indexing.literal_index import LiteralIndex

        a = _const(symtab, "a")
        p_a = _func(symtab, "P", a)
        c = _clause(_pos_lit(p_a), cid=1)
        lidx = LiteralIndex()
        lidx.update(c, insert=True)
        assert not lidx.is_empty()

    def test_insert_delete(self, symtab):
        from pyladr.indexing.literal_index import LiteralIndex

        a = _const(symtab, "a")
        p_a = _func(symtab, "P", a)
        c = _clause(_pos_lit(p_a), cid=1)
        lidx = LiteralIndex()
        lidx.update(c, insert=True)
        lidx.update(c, insert=False)
        assert lidx.is_empty()

    def test_pos_neg_separation(self, symtab):
        """Positive and negative literals go to different indexes."""
        from pyladr.indexing.literal_index import LiteralIndex

        a = _const(symtab, "a")
        p_a = _func(symtab, "P", a)
        c1 = _clause(_pos_lit(p_a), cid=1)
        c2 = _clause(_neg_lit(p_a), cid=2)
        lidx = LiteralIndex()
        lidx.update(c1, insert=True)
        lidx.update(c2, insert=True)
        # Both should be indexed but in different components
        assert not lidx.is_empty()
