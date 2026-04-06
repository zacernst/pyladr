"""Tests for pyladr.core.substitution — Unification and matching.

Tests behavioral equivalence with C unify.h/unify.c:
- Context creation and binding
- Dereference chain following
- Trail-based undo and partial rollback
- Two-way unification with automatic rollback
- One-way matching
- Variant checking
- Occur check
- Apply substitution with variable renaming
- Paramodulation apply_substitute
- Context utility functions
"""

from __future__ import annotations

import pytest

from pyladr.core.substitution import (
    Context,
    Trail,
    apply_demod,
    apply_substitute,
    apply_substitute_at_pos,
    apply_substitution,
    context_to_pairs,
    dereference,
    empty_substitution,
    match,
    occur_check,
    reset_multiplier,
    subst_changes_term,
    unify,
    variable_substitution,
    variant,
)
from pyladr.core.term import Term, get_rigid_term, get_variable_term


@pytest.fixture(autouse=True)
def _reset_multipliers():
    """Reset multiplier counter before each test."""
    reset_multiplier()


class TestContext:
    def test_initial_state(self):
        c = Context()
        assert not c.is_bound(0)
        assert c.multiplier == 0

    def test_bind_and_check(self):
        c = Context()
        t = get_rigid_term(1, 0)  # constant 'a'
        c.bind(0, t, None)
        assert c.is_bound(0)
        assert c.terms[0] is t

    def test_unbind(self):
        c = Context()
        c.bind(0, get_rigid_term(1, 0), None)
        c.unbind(0)
        assert not c.is_bound(0)

    def test_clear(self):
        c = Context()
        c.bind(0, get_rigid_term(1, 0), None)
        c.bind(1, get_rigid_term(2, 0), None)
        c.clear()
        assert not c.is_bound(0)
        assert not c.is_bound(1)

    def test_multiplier_increments(self):
        c1 = Context()
        c2 = Context()
        assert c1.multiplier == 0
        assert c2.multiplier == 1


class TestTrail:
    def test_empty_trail(self):
        trail = Trail()
        assert trail.is_empty
        assert len(trail) == 0

    def test_bind_records_on_trail(self):
        c = Context()
        trail = Trail()
        t = get_rigid_term(1, 0)
        trail.bind(0, c, t, None)
        assert c.is_bound(0)
        assert len(trail) == 1

    def test_undo_restores_state(self):
        c = Context()
        trail = Trail()
        trail.bind(0, c, get_rigid_term(1, 0), None)
        trail.bind(1, c, get_rigid_term(2, 0), None)
        assert c.is_bound(0)
        assert c.is_bound(1)
        trail.undo()
        assert not c.is_bound(0)
        assert not c.is_bound(1)
        assert trail.is_empty

    def test_undo_to_position(self):
        """Test partial trail rollback (C behavior for complex term failure)."""
        c = Context()
        trail = Trail()
        trail.bind(0, c, get_rigid_term(1, 0), None)
        saved = trail.position
        trail.bind(1, c, get_rigid_term(2, 0), None)
        trail.bind(2, c, get_rigid_term(3, 0), None)
        # Undo back to saved position
        trail.undo_to(saved)
        assert c.is_bound(0)  # first binding preserved
        assert not c.is_bound(1)  # rolled back
        assert not c.is_bound(2)  # rolled back
        assert trail.position == saved

    def test_vars_in_trail(self):
        c = Context()
        trail = Trail()
        trail.bind(3, c, get_rigid_term(1, 0), None)
        trail.bind(7, c, get_rigid_term(2, 0), None)
        assert trail.vars_in_trail() == [3, 7]


class TestDereference:
    def test_unbound_variable(self):
        x = get_variable_term(0)
        c = Context()
        t, ctx = dereference(x, c)
        assert t is x
        assert ctx is c

    def test_bound_variable_to_constant(self):
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        c = Context()
        c.bind(0, a, None)
        t, ctx = dereference(x, c)
        assert t is a
        assert ctx is None

    def test_chain_of_bindings(self):
        """v0 in c1 → v1 in c2 → constant a."""
        x = get_variable_term(0)
        y = get_variable_term(1)
        a = get_rigid_term(1, 0)
        c1 = Context()
        c2 = Context()
        c1.bind(0, y, c2)
        c2.bind(1, a, None)
        t, ctx = dereference(x, c1)
        assert t is a
        assert ctx is None

    def test_constant_unchanged(self):
        a = get_rigid_term(1, 0)
        c = Context()
        t, ctx = dereference(a, c)
        assert t is a


class TestOccurCheck:
    def test_variable_occurs_directly(self):
        """x occurs in x."""
        c = Context()
        assert occur_check(0, c, get_variable_term(0), c)

    def test_variable_occurs_in_subterm(self):
        """x occurs in f(x)."""
        x = get_variable_term(0)
        f = get_rigid_term(1, 1, (x,))
        c = Context()
        assert occur_check(0, c, f, c)

    def test_variable_not_occurs(self):
        """x does not occur in f(y)."""
        y = get_variable_term(1)
        f = get_rigid_term(1, 1, (y,))
        c = Context()
        assert not occur_check(0, c, f, c)

    def test_variable_occurs_through_binding(self):
        """x occurs in y when y is bound to f(x)."""
        x = get_variable_term(0)
        y = get_variable_term(1)
        f = get_rigid_term(1, 1, (x,))
        c = Context()
        c.bind(1, f, c)
        assert occur_check(0, c, y, c)


class TestUnify:
    """Test two-way unification (C unify)."""

    def test_two_constants_same(self):
        """a unifies with a."""
        a1 = get_rigid_term(1, 0)
        a2 = get_rigid_term(1, 0)
        c1, c2 = Context(), Context()
        trail = Trail()
        assert unify(a1, c1, a2, c2, trail)

    def test_two_constants_different(self):
        """a does not unify with b."""
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        c1, c2 = Context(), Context()
        trail = Trail()
        assert not unify(a, c1, b, c2, trail)

    def test_variable_with_constant(self):
        """x unifies with a, binding x → a."""
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        c1, c2 = Context(), Context()
        trail = Trail()
        assert unify(x, c1, a, c2, trail)
        assert c1.terms[0] is a

    def test_constant_with_variable(self):
        """a unifies with x, binding x → a."""
        a = get_rigid_term(1, 0)
        x = get_variable_term(0)
        c1, c2 = Context(), Context()
        trail = Trail()
        assert unify(a, c1, x, c2, trail)
        assert c2.terms[0] is a

    def test_two_variables_different(self):
        """x unifies with y (different contexts), no occur check needed."""
        x = get_variable_term(0)
        y = get_variable_term(1)
        c1, c2 = Context(), Context()
        trail = Trail()
        assert unify(x, c1, y, c2, trail)
        # x in c1 should be bound to y in c2
        assert c1.terms[0] is y

    def test_two_same_variables_different_contexts(self):
        """v0 in c1 unifies with v0 in c2 (different variables)."""
        x = get_variable_term(0)
        c1, c2 = Context(), Context()
        trail = Trail()
        assert unify(x, c1, x, c2, trail)
        # Should bind one to the other
        assert len(trail) == 1

    def test_complex_terms(self):
        """f(x, a) unifies with f(b, y), binding x→b, y→a."""
        x = get_variable_term(0)
        y = get_variable_term(0)  # var 0 in different context
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        fx = get_rigid_term(3, 2, (x, a))
        fy = get_rigid_term(3, 2, (b, y))
        c1, c2 = Context(), Context()
        trail = Trail()
        assert unify(fx, c1, fy, c2, trail)
        assert c1.terms[0] is b
        assert c2.terms[0] is a

    def test_occur_check_prevents_infinite(self):
        """x does not unify with f(x) (occur check)."""
        x = get_variable_term(0)
        f = get_rigid_term(1, 1, (x,))
        c1, c2 = Context(), Context()
        trail = Trail()
        assert not unify(x, c1, f, c1, trail)

    def test_automatic_rollback_on_complex_failure(self):
        """C behavior: unify() rolls back partial bindings on complex term failure.

        f(x, a) vs f(b, b): x→b succeeds at arg 0, but a≠b fails at arg 1.
        The binding x→b should be automatically rolled back by unify().
        """
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        fxa = get_rigid_term(3, 2, (x, a))
        fbb = get_rigid_term(3, 2, (b, b))
        c1, c2 = Context(), Context()
        trail = Trail()
        result = unify(fxa, c1, fbb, c2, trail)
        assert not result
        # Key: x→b binding should have been rolled back automatically
        assert not c1.is_bound(0)
        assert trail.is_empty

    def test_same_variable_same_context(self):
        """x unifies with x in the same context (trivially)."""
        x = get_variable_term(0)
        c = Context()
        trail = Trail()
        assert unify(x, c, x, c, trail)

    def test_nested_complex_unification(self):
        """f(g(x), h(y)) unifies with f(g(a), h(b))."""
        x, y = get_variable_term(0), get_variable_term(1)
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        # sn: g=3, h=4, f=5
        gx = get_rigid_term(3, 1, (x,))
        hy = get_rigid_term(4, 1, (y,))
        fgh1 = get_rigid_term(5, 2, (gx, hy))
        ga = get_rigid_term(3, 1, (a,))
        hb = get_rigid_term(4, 1, (b,))
        fgh2 = get_rigid_term(5, 2, (ga, hb))
        c1, c2 = Context(), Context()
        trail = Trail()
        assert unify(fgh1, c1, fgh2, c2, trail)
        assert c1.terms[0] is a  # x → a
        assert c1.terms[1] is b  # y → b


class TestMatch:
    """Test one-way matching (C match)."""

    def test_pattern_variable_matches_constant(self):
        """Pattern x matches target a."""
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        c = Context()
        trail = Trail()
        assert match(x, c, a, trail)
        assert c.terms[0] is a

    def test_pattern_constant_matches_same(self):
        """Pattern a matches target a."""
        a1 = get_rigid_term(1, 0)
        a2 = get_rigid_term(1, 0)
        c = Context()
        trail = Trail()
        assert match(a1, c, a2, trail)

    def test_pattern_constant_fails_different(self):
        """Pattern a does not match target b."""
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        c = Context()
        trail = Trail()
        assert not match(a, c, b, trail)

    def test_rigid_pattern_vs_variable_target_fails(self):
        """Pattern f(a) does not match variable target x."""
        a = get_rigid_term(1, 0)
        fa = get_rigid_term(2, 1, (a,))
        x = get_variable_term(0)
        c = Context()
        trail = Trail()
        assert not match(fa, c, x, trail)

    def test_complex_pattern_match(self):
        """Pattern f(x, a) matches f(b, a), binding x→b."""
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        pattern = get_rigid_term(3, 2, (x, a))
        target = get_rigid_term(3, 2, (b, a))
        c = Context()
        trail = Trail()
        assert match(pattern, c, target, trail)
        assert c.terms[0] is b


class TestVariant:
    """Test variant checking (C variant)."""

    def test_identical_constants(self):
        a = get_rigid_term(1, 0)
        c = Context()
        trail = Trail()
        assert variant(a, c, a, trail)

    def test_variable_renaming(self):
        """f(x, y) is a variant of f(y, x) (up to variable renaming)."""
        x, y = get_variable_term(0), get_variable_term(1)
        f1 = get_rigid_term(1, 2, (x, y))
        f2 = get_rigid_term(1, 2, (y, x))
        c = Context()
        trail = Trail()
        assert variant(f1, c, f2, trail)

    def test_non_variant(self):
        """f(x, x) is NOT a variant of f(x, y) — different variable structure."""
        x, y = get_variable_term(0), get_variable_term(1)
        f1 = get_rigid_term(1, 2, (x, x))
        f2 = get_rigid_term(1, 2, (x, y))
        c = Context()
        trail = Trail()
        assert not variant(f1, c, f2, trail)

    def test_constant_vs_variable(self):
        """f(a) is NOT a variant of f(x)."""
        a = get_rigid_term(2, 0)
        x = get_variable_term(0)
        f1 = get_rigid_term(1, 1, (a,))
        f2 = get_rigid_term(1, 1, (x,))
        c = Context()
        trail = Trail()
        assert not variant(f1, c, f2, trail)


class TestApplySubstitution:
    """Test apply() — building instantiated terms from context."""

    def test_apply_to_constant(self):
        """Constants produce a new identical term."""
        a = get_rigid_term(1, 0)
        c = Context()
        result = apply_substitution(a, c)
        assert result.term_ident(a)

    def test_apply_bound_variable(self):
        """Bound variable is replaced by its binding."""
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        c = Context()
        c.bind(0, a, None)
        result = apply_substitution(x, c)
        assert result.term_ident(a)

    def test_apply_unbound_variable_renamed(self):
        """Unbound variable gets renamed: multiplier * MAX_VARS + varnum."""
        x = get_variable_term(0)
        c = Context()
        result = apply_substitution(x, c)
        assert result.is_variable
        assert result.varnum == c.multiplier * 100 + 0

    def test_apply_null_context(self):
        """With None context, apply just copies the term."""
        x = get_variable_term(5)
        result = apply_substitution(x, None)
        assert result.is_variable
        assert result.varnum == 5

    def test_apply_complex(self):
        """f(x, a) with x→b becomes f(b, a)."""
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        fxa = get_rigid_term(3, 2, (x, a))
        c = Context()
        c.bind(0, b, None)
        result = apply_substitution(fxa, c)
        expected = get_rigid_term(3, 2, (b, a))
        assert result.term_ident(expected)


class TestApplySubstitute:
    """Test apply_substitute for paramodulation."""

    def test_replace_at_target(self):
        """f(a, b) with into_term=a, beta=c → f(c, b)."""
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        c_const = get_rigid_term(3, 0)
        fab = get_rigid_term(4, 2, (a, b))
        c_from = Context()
        c_into = Context()
        result = apply_substitute(fab, c_const, c_from, a, c_into)
        assert result.symnum == 4
        assert result.arg(0).term_ident(c_const)
        assert result.arg(1).term_ident(b)

    def test_replace_at_position(self):
        """f(a, b) with pos=(1,) (first arg), beta=c → f(c, b)."""
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        c_const = get_rigid_term(3, 0)
        fab = get_rigid_term(4, 2, (a, b))
        c_from = Context()
        c_into = Context()
        result = apply_substitute_at_pos(fab, c_const, c_from, (1,), c_into)
        assert result.arg(0).term_ident(c_const)
        assert result.arg(1).term_ident(b)

    def test_replace_nested_position(self):
        """f(g(a), b) with pos=(1,1) → f(g(c), b)."""
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        c_const = get_rigid_term(3, 0)
        ga = get_rigid_term(5, 1, (a,))
        fgab = get_rigid_term(4, 2, (ga, b))
        c_from = Context()
        c_into = Context()
        result = apply_substitute_at_pos(fgab, c_const, c_from, (1, 1), c_into)
        assert result.arg(0).arg(0).term_ident(c_const)
        assert result.arg(1).term_ident(b)


class TestApplyDemod:
    """Test apply_demod for demodulation."""

    def test_simple_demod(self):
        """Demod f(x) with x→a produces f(a)."""
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        fx = get_rigid_term(2, 1, (x,))
        c = Context()
        c.bind(0, a, None)
        result = apply_demod(fx, c)
        expected = get_rigid_term(2, 1, (a,))
        assert result.term_ident(expected)

    def test_demod_uninstantiated_raises(self):
        """Demod with uninstantiated variable raises."""
        x = get_variable_term(0)
        c = Context()
        with pytest.raises(ValueError, match="not instantiated"):
            apply_demod(x, c)


class TestContextUtilities:
    """Test context utility functions."""

    def test_empty_substitution_true(self):
        c = Context()
        assert empty_substitution(c)

    def test_empty_substitution_false(self):
        c = Context()
        c.bind(0, get_rigid_term(1, 0), None)
        assert not empty_substitution(c)

    def test_variable_substitution_true(self):
        """All bindings resolve to variables."""
        c = Context()
        c.bind(0, get_variable_term(1), None)
        assert variable_substitution(c)

    def test_variable_substitution_false(self):
        """A binding resolves to a constant."""
        c = Context()
        c.bind(0, get_rigid_term(1, 0), None)
        assert not variable_substitution(c)

    def test_variable_substitution_empty(self):
        """Empty substitution is trivially a variable substitution."""
        c = Context()
        assert variable_substitution(c)

    def test_subst_changes_term_yes(self):
        """Term has bound variable."""
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        f = get_rigid_term(2, 2, (x, a))
        c = Context()
        c.bind(0, get_rigid_term(3, 0), None)
        assert subst_changes_term(f, c)

    def test_subst_changes_term_no(self):
        """Term has no bound variables."""
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        f = get_rigid_term(2, 2, (x, a))
        c = Context()  # no bindings
        assert not subst_changes_term(f, c)

    def test_subst_changes_ground_term(self):
        """Ground term is never changed."""
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        f = get_rigid_term(3, 2, (a, b))
        c = Context()
        c.bind(0, get_rigid_term(4, 0), None)  # irrelevant binding
        assert not subst_changes_term(f, c)

    def test_context_to_pairs(self):
        """Convert context bindings to (var, term) pairs."""
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        c = Context()
        c.bind(0, a, None)
        c.bind(2, b, None)
        pairs = context_to_pairs({0, 1, 2}, c)
        # var 0 → a, var 2 → b; var 1 is unbound (but renamed, so still a pair)
        assert len(pairs) >= 2
        # Check that bindings are present
        pair_dict = {p[0].varnum: p[1] for p in pairs}
        assert pair_dict[0].term_ident(a)
        assert pair_dict[2].term_ident(b)
