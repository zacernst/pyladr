"""Property-based tests for unification and matching.

Verifies algebraic properties that must hold for correct unification:
- Unification is symmetric
- Successful unification produces a valid unifier
- Matching is asymmetric (one-way)
- Occur check prevents infinite terms
- Trail undo restores original state completely
"""

from __future__ import annotations

import pytest

from pyladr.core.substitution import (
    Context,
    Trail,
    apply_substitution,
    match,
    reset_multiplier,
    unify,
)
from pyladr.core.term import Term, get_rigid_term, get_variable_term


@pytest.fixture(autouse=True)
def _reset():
    reset_multiplier()


# ── Term generators ──────────────────────────────────────────────────────────

def _random_terms() -> list[tuple[Term, Term, str]]:
    """Generate pairs of terms for testing unification properties."""
    x, y, z = get_variable_term(0), get_variable_term(1), get_variable_term(2)
    a = get_rigid_term(1, 0)
    b = get_rigid_term(2, 0)
    f = lambda *args: get_rigid_term(3, len(args), tuple(args))
    g = lambda *args: get_rigid_term(4, len(args), tuple(args))

    return [
        # (term1, term2, description)
        (a, a, "identical constants"),
        (a, b, "different constants"),
        (x, a, "variable vs constant"),
        (x, y, "two variables"),
        (f(x), f(a), "f(x) vs f(a)"),
        (f(x, y), f(a, b), "f(x,y) vs f(a,b)"),
        (f(x, x), f(a, a), "f(x,x) vs f(a,a)"),
        (f(x, x), f(a, b), "f(x,x) vs f(a,b) - should fail"),
        (f(g(x)), f(g(a)), "nested: f(g(x)) vs f(g(a))"),
        (x, f(x), "occur check: x vs f(x)"),
        (f(x, a), f(b, y), "cross binding: f(x,a) vs f(b,y)"),
        (f(x), g(x), "different top symbols"),
    ]


# ── Property: Unification symmetry ─────────────────────────────────────────


class TestUnificationSymmetry:
    """Unify(t1, t2) should succeed iff unify(t2, t1) succeeds."""

    @pytest.mark.parametrize("t1,t2,desc", _random_terms())
    def test_symmetry(self, t1, t2, desc):
        c1a, c2a = Context(), Context()
        trail_a = Trail()
        result_a = unify(t1, c1a, t2, c2a, trail_a)
        trail_a.undo()

        c1b, c2b = Context(), Context()
        trail_b = Trail()
        result_b = unify(t2, c1b, t1, c2b, trail_b)
        trail_b.undo()

        assert result_a == result_b, (
            f"Symmetry violation for {desc}: "
            f"unify({t1}, {t2})={result_a} but unify({t2}, {t1})={result_b}"
        )


# ── Property: Successful unification produces a valid MGU ──────────────────


class TestUnificationCorrectness:
    """If unification succeeds, applying the substitution makes terms equal."""

    @pytest.mark.parametrize("t1,t2,desc", _random_terms())
    def test_mgu_correctness(self, t1, t2, desc):
        c1, c2 = Context(), Context()
        trail = Trail()
        if unify(t1, c1, t2, c2, trail):
            # Apply substitution to both terms
            inst1 = apply_substitution(t1, c1)
            inst2 = apply_substitution(t2, c2)
            assert inst1.term_ident(inst2), (
                f"MGU property violated for {desc}: "
                f"apply(t1)={inst1} != apply(t2)={inst2}"
            )
        trail.undo()


# ── Property: Trail undo fully restores state ───────────────────────────────


class TestTrailRestore:
    """After trail.undo(), all contexts must be fully restored."""

    @pytest.mark.parametrize("t1,t2,desc", _random_terms())
    def test_undo_restores_clean(self, t1, t2, desc):
        c1, c2 = Context(), Context()
        trail = Trail()
        unify(t1, c1, t2, c2, trail)
        trail.undo()
        # After undo, both contexts should be clean
        for i in range(10):
            assert not c1.is_bound(i), f"c1 var {i} still bound after undo"
            assert not c2.is_bound(i), f"c2 var {i} still bound after undo"


# ── Property: Occur check prevents infinite terms ─────────────────────────


class TestOccurCheckProperty:
    """Variables should never unify with terms containing themselves."""

    def test_x_vs_fx(self):
        x = get_variable_term(0)
        fx = get_rigid_term(1, 1, (x,))
        c = Context()
        trail = Trail()
        assert not unify(x, c, fx, c, trail)

    def test_x_vs_gxfx(self):
        """x cannot unify with g(x, f(x))."""
        x = get_variable_term(0)
        fx = get_rigid_term(1, 1, (x,))
        gxfx = get_rigid_term(2, 2, (x, fx))
        c = Context()
        trail = Trail()
        assert not unify(x, c, gxfx, c, trail)

    def test_deep_occur(self):
        """x cannot unify with f(f(f(x)))."""
        x = get_variable_term(0)
        t = x
        for _ in range(3):
            t = get_rigid_term(1, 1, (t,))
        c = Context()
        trail = Trail()
        assert not unify(x, c, t, c, trail)


# ── Property: Matching is one-directional ────────────────────────────────────


class TestMatchAsymmetry:
    """match(pattern, target) should NOT succeed when target has variables
    that don't appear in pattern (matching is one-way)."""

    def test_constant_matches_constant(self):
        a = get_rigid_term(1, 0)
        c = Context()
        trail = Trail()
        assert match(a, c, a, trail)

    def test_variable_matches_constant(self):
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        c = Context()
        trail = Trail()
        assert match(x, c, a, trail)

    def test_constant_does_not_match_variable(self):
        """Rigid pattern cannot match variable target."""
        a = get_rigid_term(1, 0)
        x = get_variable_term(0)
        c = Context()
        trail = Trail()
        assert not match(a, c, x, trail)

    def test_consistency_check(self):
        """match(f(x,x), f(a,b)) fails because x can't be both a and b."""
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        pattern = get_rigid_term(3, 2, (x, x))
        target = get_rigid_term(3, 2, (a, b))
        c = Context()
        trail = Trail()
        assert not match(pattern, c, target, trail)
