"""Property-based tests for term operations using Hypothesis.

These tests verify algebraic properties that must hold for any term,
regardless of specific structure. They complement unit tests by testing
with randomly generated inputs.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pyladr.core.substitution import Context, Trail, match, reset_multiplier, unify
from pyladr.core.term import Term, get_rigid_term, get_variable_term


# ── Hypothesis strategies for generating terms ──────────────────────────────

# Strategy for variable indices (0..99 per C MAX_VARS)
var_indices = st.integers(min_value=0, max_value=99)

# Strategy for symbol names (we use integer IDs internally)
symbol_ids = st.integers(min_value=1, max_value=50)

# Strategy for symbol names (for display, not used in Term construction)
symbol_names = st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True)


def _const(symnum: int) -> Term:
    return get_rigid_term(symnum, 0)


def _func(symnum: int, *args: Term) -> Term:
    return get_rigid_term(symnum, len(args), args)


@pytest.fixture(autouse=True)
def _reset():
    reset_multiplier()


@pytest.mark.property
class TestTermProperties:
    """Property-based tests for term data structure."""

    @given(idx=var_indices)
    def test_variable_is_not_ground(self, idx: int) -> None:
        """No variable term is ground."""
        v = get_variable_term(idx)
        assert not v.is_ground

    @given(sid=symbol_ids)
    def test_constant_is_ground(self, sid: int) -> None:
        """Every constant term is ground."""
        c = _const(sid)
        assert c.is_ground

    @given(idx=var_indices)
    def test_variable_depth_is_zero(self, idx: int) -> None:
        """Every variable has depth 0."""
        v = get_variable_term(idx)
        assert v.depth == 0

    @given(sid=symbol_ids)
    def test_constant_depth_is_zero(self, sid: int) -> None:
        """Every constant has depth 0."""
        c = _const(sid)
        assert c.depth == 0

    @given(idx=var_indices)
    def test_variable_symbol_count_is_one(self, idx: int) -> None:
        """Every variable has symbol count 1."""
        v = get_variable_term(idx)
        assert v.symbol_count == 1

    @given(idx=var_indices)
    def test_term_ident_reflexive(self, idx: int) -> None:
        """Every term is identical to itself."""
        v = get_variable_term(idx)
        assert v.term_ident(v)

    @given(sid=symbol_ids)
    def test_constant_term_ident_reflexive(self, sid: int) -> None:
        """Every constant is identical to itself."""
        c = _const(sid)
        assert c.term_ident(c)

    @given(sid1=symbol_ids, sid2=symbol_ids)
    def test_different_constants_not_ident(self, sid1: int, sid2: int) -> None:
        """Different constants are not identical."""
        if sid1 != sid2:
            assert not _const(sid1).term_ident(_const(sid2))

    @given(idx=var_indices)
    def test_variable_is_variable(self, idx: int) -> None:
        """get_variable_term always produces a variable."""
        v = get_variable_term(idx)
        assert v.is_variable
        assert not v.is_constant
        assert not v.is_complex

    @given(sid=symbol_ids)
    def test_constant_is_constant(self, sid: int) -> None:
        """get_rigid_term with arity 0 always produces a constant."""
        c = _const(sid)
        assert c.is_constant
        assert not c.is_variable
        assert not c.is_complex

    @given(idx=var_indices)
    def test_variable_hash_is_varnum(self, idx: int) -> None:
        """C hash of a variable is its variable number."""
        v = get_variable_term(idx)
        assert v.c_hash() == (idx & 0xFFFFFFFF)


@pytest.mark.property
class TestUnificationProperties:
    """Property-based tests for unification."""

    @given(idx=var_indices)
    def test_variable_unifies_with_self(self, idx: int) -> None:
        """A variable always unifies with itself in the same context."""
        v = get_variable_term(idx)
        ctx = Context()
        trail = Trail()
        assert unify(v, ctx, v, ctx, trail) is True

    @given(sid=symbol_ids)
    def test_constant_unifies_with_self(self, sid: int) -> None:
        """A constant always unifies with itself."""
        c = _const(sid)
        ctx1, ctx2 = Context(), Context()
        trail = Trail()
        assert unify(c, ctx1, c, ctx2, trail) is True

    @given(idx=var_indices, sid=symbol_ids)
    def test_variable_unifies_with_constant(self, idx: int, sid: int) -> None:
        """A variable always unifies with a constant."""
        v = get_variable_term(idx)
        c = _const(sid)
        ctx1, ctx2 = Context(), Context()
        trail = Trail()
        assert unify(v, ctx1, c, ctx2, trail) is True
        trail.undo()

    @given(idx1=var_indices, idx2=var_indices)
    def test_unification_symmetric(self, idx1: int, idx2: int) -> None:
        """If s unifies with t, then t unifies with s."""
        v1 = get_variable_term(idx1)
        v2 = get_variable_term(idx2)
        ctx1a, ctx2a = Context(), Context()
        trail_a = Trail()
        r1 = unify(v1, ctx1a, v2, ctx2a, trail_a)
        trail_a.undo()

        ctx1b, ctx2b = Context(), Context()
        trail_b = Trail()
        r2 = unify(v2, ctx1b, v1, ctx2b, trail_b)
        trail_b.undo()

        assert r1 == r2

    @given(sid1=symbol_ids, sid2=symbol_ids)
    def test_different_constants_dont_unify(self, sid1: int, sid2: int) -> None:
        """Different constants never unify."""
        if sid1 != sid2:
            c1, c2 = _const(sid1), _const(sid2)
            ctx1, ctx2 = Context(), Context()
            trail = Trail()
            assert unify(c1, ctx1, c2, ctx2, trail) is False


@pytest.mark.property
class TestMatchingProperties:
    """Property-based tests for pattern matching."""

    @given(sid=symbol_ids)
    def test_constant_matches_self(self, sid: int) -> None:
        """A constant matches itself."""
        c = _const(sid)
        ctx, trail = Context(), Trail()
        assert match(c, ctx, c, trail) is True

    @given(idx=var_indices, sid=symbol_ids)
    def test_variable_matches_any_ground(self, idx: int, sid: int) -> None:
        """A pattern variable matches any ground term."""
        v = get_variable_term(idx)
        c = _const(sid)
        ctx, trail = Context(), Trail()
        assert match(v, ctx, c, trail) is True
        trail.undo()

    @given(sid1=symbol_ids, sid2=symbol_ids)
    def test_different_constants_dont_match(self, sid1: int, sid2: int) -> None:
        """Different constants don't match."""
        if sid1 != sid2:
            c1, c2 = _const(sid1), _const(sid2)
            ctx, trail = Context(), Trail()
            assert match(c1, ctx, c2, trail) is False
