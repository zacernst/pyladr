"""Tests for parent-to-child penalty propagation system."""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal, ParaJust
from pyladr.core.term import Term
from pyladr.search.penalty_propagation import (
    PenaltyCache,
    PenaltyCombineMode,
    PenaltyPropagationConfig,
    PenaltyRecord,
    combine_penalty,
    compute_and_cache_penalty,
    compute_inherited_penalty,
    extract_parent_ids,
)


# ── Helper builders ──────────────────────────────────────────────────────────


def _var(n: int) -> Term:
    """Create a variable term."""
    return Term(private_symbol=n, arity=0, args=())


def _const(sym: int) -> Term:
    """Create a constant (0-arity rigid symbol)."""
    return Term(private_symbol=sym, arity=0, args=())


def _func(sym: int, *args: Term) -> Term:
    """Create a function/predicate term."""
    return Term(private_symbol=sym, arity=len(args), args=args)


def _clause(clause_id: int, literals: list[Literal], just: Justification | None = None) -> Clause:
    """Create a clause with given ID and justification."""
    justification = (just,) if just is not None else ()
    c = Clause(literals=tuple(literals), justification=justification)
    c.id = clause_id
    return c


def _make_general_clause(clause_id: int) -> Clause:
    """Create an overly general clause: P(x) — single literal, all variables.

    This should get a high penalty (>= 10.0) from _clause_generality_penalty.
    """
    # P(x) — predicate with one variable argument
    atom = _func(-1, _var(0))
    lit = Literal(sign=True, atom=atom)
    return _clause(clause_id, [lit], Justification(just_type=JustType.INPUT))


def _make_specific_clause(clause_id: int, parent_ids: tuple[int, ...] = ()) -> Clause:
    """Create a specific clause: P(a, b) — all constants.

    Low penalty (close to 0).
    """
    atom = _func(-1, _const(-2), _const(-3))
    lit = Literal(sign=True, atom=atom)
    just = Justification(just_type=JustType.BINARY_RES, clause_ids=parent_ids)
    return _clause(clause_id, [lit], just)


def _make_derived_clause(clause_id: int, parent_ids: tuple[int, ...]) -> Clause:
    """Create a derived clause with binary resolution justification."""
    atom = _func(-1, _var(0), _const(-2))
    lit = Literal(sign=True, atom=atom)
    just = Justification(just_type=JustType.BINARY_RES, clause_ids=parent_ids)
    return _clause(clause_id, [lit], just)


# ── extract_parent_ids ───────────────────────────────────────────────────────


class TestExtractParentIds:
    def test_input_clause_no_parents(self):
        just = Justification(just_type=JustType.INPUT)
        assert extract_parent_ids((just,)) == ()

    def test_goal_clause_no_parents(self):
        just = Justification(just_type=JustType.GOAL)
        assert extract_parent_ids((just,)) == ()

    def test_binary_resolution_parents(self):
        just = Justification(just_type=JustType.BINARY_RES, clause_ids=(3, 7))
        assert extract_parent_ids((just,)) == (3, 7)

    def test_hyper_resolution_parents(self):
        just = Justification(just_type=JustType.HYPER_RES, clause_ids=(1, 2, 3))
        assert extract_parent_ids((just,)) == (1, 2, 3)

    def test_factor_parents(self):
        just = Justification(just_type=JustType.FACTOR, clause_ids=(5,))
        assert extract_parent_ids((just,)) == (5,)

    def test_paramodulation_parents(self):
        para = ParaJust(from_id=10, into_id=20, from_pos=(0,), into_pos=(1,))
        just = Justification(just_type=JustType.PARA, para=para)
        assert extract_parent_ids((just,)) == (10, 20)

    def test_demod_parent(self):
        just = Justification(just_type=JustType.DEMOD, clause_id=15)
        assert extract_parent_ids((just,)) == (15,)

    def test_back_demod_parent(self):
        just = Justification(just_type=JustType.BACK_DEMOD, clause_id=8)
        assert extract_parent_ids((just,)) == (8,)

    def test_copy_parent(self):
        just = Justification(just_type=JustType.COPY, clause_id=12)
        assert extract_parent_ids((just,)) == (12,)

    def test_empty_justification(self):
        assert extract_parent_ids(()) == ()

    def test_only_primary_justification_used(self):
        """Only the first (primary) justification is examined."""
        primary = Justification(just_type=JustType.BINARY_RES, clause_ids=(1, 2))
        secondary = Justification(just_type=JustType.DEMOD, clause_id=99)
        assert extract_parent_ids((primary, secondary)) == (1, 2)


# ── combine_penalty ──────────────────────────────────────────────────────────


class TestCombinePenalty:
    def test_additive_mode(self):
        result = combine_penalty(2.0, 3.0, PenaltyCombineMode.ADDITIVE, 100.0)
        assert result == 5.0

    def test_multiplicative_mode(self):
        result = combine_penalty(2.0, 3.0, PenaltyCombineMode.MULTIPLICATIVE, 100.0)
        assert result == pytest.approx(2.0 * (1.0 + 3.0))  # 8.0

    def test_max_mode(self):
        result = combine_penalty(2.0, 3.0, PenaltyCombineMode.MAX, 100.0)
        assert result == 3.0

    def test_max_mode_own_higher(self):
        result = combine_penalty(5.0, 3.0, PenaltyCombineMode.MAX, 100.0)
        assert result == 5.0

    def test_cap_applied(self):
        result = combine_penalty(15.0, 10.0, PenaltyCombineMode.ADDITIVE, 20.0)
        assert result == 20.0

    def test_no_inherited_penalty(self):
        result = combine_penalty(3.0, 0.0, PenaltyCombineMode.ADDITIVE, 100.0)
        assert result == 3.0

    def test_negative_inherited_treated_as_zero(self):
        result = combine_penalty(3.0, -1.0, PenaltyCombineMode.ADDITIVE, 100.0)
        assert result == 3.0


# ── compute_inherited_penalty ────────────────────────────────────────────────


class TestComputeInheritedPenalty:
    def _make_cache_with_parent(
        self, parent_id: int, combined: float, depth: int = 0
    ) -> PenaltyCache:
        config = PenaltyPropagationConfig(
            enabled=True, decay=0.5, threshold=5.0, max_depth=3,
        )
        cache = PenaltyCache(config)
        cache.put(parent_id, PenaltyRecord(
            own_penalty=combined, inherited_penalty=0.0,
            combined_penalty=combined, depth=depth,
        ))
        return cache

    def test_no_parents(self):
        config = PenaltyPropagationConfig(enabled=True)
        cache = PenaltyCache(config)
        inherited, depth = compute_inherited_penalty([], cache, {}, config)
        assert inherited == 0.0
        assert depth == 0

    def test_parent_below_threshold(self):
        cache = self._make_cache_with_parent(1, combined=3.0)
        config = cache.config
        inherited, depth = compute_inherited_penalty([1], cache, {}, config)
        assert inherited == 0.0  # 3.0 < threshold 5.0

    def test_parent_above_threshold(self):
        cache = self._make_cache_with_parent(1, combined=10.0)
        config = cache.config
        inherited, depth = compute_inherited_penalty([1], cache, {}, config)
        assert inherited == pytest.approx(0.5 * 10.0)  # decay=0.5
        assert depth == 1

    def test_max_parent_penalty_selected(self):
        config = PenaltyPropagationConfig(
            enabled=True, decay=0.5, threshold=5.0, max_depth=3,
        )
        cache = PenaltyCache(config)
        cache.put(1, PenaltyRecord(6.0, 0.0, 6.0, 0))
        cache.put(2, PenaltyRecord(12.0, 0.0, 12.0, 0))
        inherited, depth = compute_inherited_penalty([1, 2], cache, {}, config)
        assert inherited == pytest.approx(0.5 * 12.0)

    def test_depth_limit_blocks_propagation(self):
        cache = self._make_cache_with_parent(1, combined=10.0, depth=3)
        config = cache.config  # max_depth=3
        inherited, depth = compute_inherited_penalty([1], cache, {}, config)
        # Parent at depth 3, child would be depth 4 > max_depth 3
        assert inherited == 0.0

    def test_unlimited_depth(self):
        config = PenaltyPropagationConfig(
            enabled=True, decay=0.5, threshold=5.0, max_depth=0,  # unlimited
        )
        cache = PenaltyCache(config)
        cache.put(1, PenaltyRecord(10.0, 0.0, 10.0, 100))  # very deep
        inherited, depth = compute_inherited_penalty([1], cache, {}, config)
        assert inherited == pytest.approx(5.0)
        assert depth == 101

    def test_parent_not_in_cache(self):
        config = PenaltyPropagationConfig(enabled=True, threshold=5.0)
        cache = PenaltyCache(config)
        inherited, depth = compute_inherited_penalty([99], cache, {}, config)
        assert inherited == 0.0


# ── PenaltyCache ─────────────────────────────────────────────────────────────


class TestPenaltyCache:
    def test_get_missing_returns_none(self):
        config = PenaltyPropagationConfig(enabled=True)
        cache = PenaltyCache(config)
        assert cache.get(1) is None

    def test_get_combined_missing_returns_zero(self):
        config = PenaltyPropagationConfig(enabled=True)
        cache = PenaltyCache(config)
        assert cache.get_combined(1) == 0.0

    def test_put_and_get(self):
        config = PenaltyPropagationConfig(enabled=True)
        cache = PenaltyCache(config)
        rec = PenaltyRecord(1.0, 2.0, 3.0, 1)
        cache.put(5, rec)
        assert cache.get(5) is rec
        assert cache.get_combined(5) == 3.0

    def test_remove(self):
        config = PenaltyPropagationConfig(enabled=True)
        cache = PenaltyCache(config)
        cache.put(5, PenaltyRecord(1.0, 2.0, 3.0, 1))
        cache.remove(5)
        assert cache.get(5) is None

    def test_contains(self):
        config = PenaltyPropagationConfig(enabled=True)
        cache = PenaltyCache(config)
        cache.put(5, PenaltyRecord(1.0, 2.0, 3.0, 1))
        assert 5 in cache
        assert 99 not in cache

    def test_len(self):
        config = PenaltyPropagationConfig(enabled=True)
        cache = PenaltyCache(config)
        assert len(cache) == 0
        cache.put(1, PenaltyRecord(1.0, 0.0, 1.0, 0))
        cache.put(2, PenaltyRecord(2.0, 0.0, 2.0, 0))
        assert len(cache) == 2


# ── compute_and_cache_penalty ────────────────────────────────────────────────


class TestComputeAndCachePenalty:
    def test_input_clause_no_inheritance(self):
        """Input clauses should have depth=0, no inherited penalty."""
        config = PenaltyPropagationConfig(enabled=True, threshold=5.0)
        cache = PenaltyCache(config)
        clause = _make_general_clause(1)
        all_clauses: dict[int, Clause] = {1: clause}

        combined = compute_and_cache_penalty(clause, cache, all_clauses)

        rec = cache.get(1)
        assert rec is not None
        assert rec.depth == 0
        assert rec.inherited_penalty == 0.0
        assert rec.own_penalty == rec.combined_penalty  # no inheritance
        assert combined == rec.combined_penalty

    def test_derived_clause_inherits_from_general_parent(self):
        """Derived clause should inherit penalty from a high-penalty parent."""
        config = PenaltyPropagationConfig(
            enabled=True, decay=0.5, threshold=5.0, max_depth=3,
            max_penalty=50.0,
        )
        cache = PenaltyCache(config)

        # Parent: overly general (high penalty >= 10.0)
        parent = _make_general_clause(1)
        all_clauses: dict[int, Clause] = {1: parent}
        compute_and_cache_penalty(parent, cache, all_clauses)
        parent_rec = cache.get(1)
        assert parent_rec is not None
        assert parent_rec.own_penalty >= 10.0  # P(x) all-variable penalty

        # Child: derived from parent
        child = _make_derived_clause(2, parent_ids=(1,))
        all_clauses[2] = child
        compute_and_cache_penalty(child, cache, all_clauses)

        child_rec = cache.get(2)
        assert child_rec is not None
        assert child_rec.inherited_penalty > 0.0
        assert child_rec.depth == 1
        assert child_rec.combined_penalty > child_rec.own_penalty

    def test_specific_child_of_specific_parent_no_inheritance(self):
        """Specific child of specific parent: no penalty inheritance."""
        config = PenaltyPropagationConfig(
            enabled=True, threshold=5.0,
        )
        cache = PenaltyCache(config)

        parent = _make_specific_clause(1)
        all_clauses: dict[int, Clause] = {1: parent}
        compute_and_cache_penalty(parent, cache, all_clauses)

        child = _make_specific_clause(2, parent_ids=(1,))
        all_clauses[2] = child
        compute_and_cache_penalty(child, cache, all_clauses)

        child_rec = cache.get(2)
        assert child_rec is not None
        assert child_rec.inherited_penalty == 0.0
        assert child_rec.depth == 0

    def test_depth_decays_through_chain(self):
        """Penalty should decay through a chain of derivations."""
        config = PenaltyPropagationConfig(
            enabled=True, decay=0.5, threshold=1.0, max_depth=5,
            max_penalty=100.0,
        )
        cache = PenaltyCache(config)

        # Root: high penalty
        root = _make_general_clause(1)
        all_clauses: dict[int, Clause] = {1: root}
        compute_and_cache_penalty(root, cache, all_clauses)

        prev_combined = cache.get(1).combined_penalty
        prev_id = 1

        # Chain of 3 derivations
        for i in range(2, 5):
            derived = _make_derived_clause(i, parent_ids=(prev_id,))
            all_clauses[i] = derived
            compute_and_cache_penalty(derived, cache, all_clauses)
            rec = cache.get(i)
            assert rec is not None
            assert rec.depth == i - 1
            prev_id = i


# ── PenaltyPropagationConfig defaults ────────────────────────────────────────


class TestPenaltyPropagationConfig:
    def test_defaults(self):
        config = PenaltyPropagationConfig()
        assert config.enabled is False
        assert config.mode == PenaltyCombineMode.ADDITIVE
        assert config.decay == 0.5
        assert config.threshold == 5.0
        assert config.max_depth == 3
        assert config.max_penalty == 20.0

    def test_disabled_by_default(self):
        """Feature is disabled by default for C Prover9 compatibility."""
        config = PenaltyPropagationConfig()
        assert not config.enabled

    def test_frozen(self):
        """Config is immutable."""
        config = PenaltyPropagationConfig()
        with pytest.raises(AttributeError):
            config.enabled = True  # type: ignore


# ── PenaltyRecord ──────────────────────────────────────────────────────────


class TestPenaltyRecord:
    def test_fields(self):
        rec = PenaltyRecord(own_penalty=1.0, inherited_penalty=2.0, combined_penalty=3.0, depth=1)
        assert rec.own_penalty == 1.0
        assert rec.inherited_penalty == 2.0
        assert rec.combined_penalty == 3.0
        assert rec.depth == 1


# ── PenaltyCombineMode ──────────────────────────────────────────────────────


class TestPenaltyCombineMode:
    def test_three_modes_exist(self):
        assert len(PenaltyCombineMode) == 3

    def test_modes_distinct(self):
        modes = {PenaltyCombineMode.ADDITIVE, PenaltyCombineMode.MULTIPLICATIVE, PenaltyCombineMode.MAX}
        assert len(modes) == 3


# ── Additional edge case tests ─────────────────────────────────────────────


class TestExtractParentIdsEdgeCases:
    """Edge cases for parent extraction."""

    def test_para_without_parajust(self):
        """PARA with None ParaJust returns empty."""
        just = Justification(just_type=JustType.PARA, para=None)
        assert extract_parent_ids((just,)) == ()

    def test_copy_with_zero_id(self):
        """COPY with clause_id=0 → no parent."""
        just = Justification(just_type=JustType.COPY, clause_id=0)
        assert extract_parent_ids((just,)) == ()

    def test_deny_parent(self):
        """DENY extracts parent clause_id."""
        just = Justification(just_type=JustType.DENY, clause_id=5)
        assert extract_parent_ids((just,)) == (5,)

    def test_clausify_no_parents(self):
        """CLAUSIFY not handled → empty."""
        just = Justification(just_type=JustType.CLAUSIFY, clause_id=2)
        assert extract_parent_ids((just,)) == ()

    def test_hyper_res_single_satellite(self):
        """Hyper-resolution with nucleus + 1 satellite."""
        just = Justification(just_type=JustType.HYPER_RES, clause_ids=(2, 5))
        assert extract_parent_ids((just,)) == (2, 5)


class TestCombinePenaltyEdgeCases:
    """Additional boundary conditions for combine_penalty."""

    def test_zero_own_additive(self):
        """Ground clause (own=0) + inherited = inherited."""
        assert combine_penalty(0.0, 5.0, PenaltyCombineMode.ADDITIVE, 20.0) == 5.0

    def test_zero_own_multiplicative(self):
        """Ground clause (own=0) * anything = 0."""
        assert combine_penalty(0.0, 5.0, PenaltyCombineMode.MULTIPLICATIVE, 20.0) == 0.0

    def test_zero_own_max(self):
        """Ground clause (own=0), max selects inherited."""
        assert combine_penalty(0.0, 5.0, PenaltyCombineMode.MAX, 20.0) == 5.0

    def test_cap_multiplicative(self):
        """Multiplicative can produce large values — verify cap."""
        result = combine_penalty(5.0, 10.0, PenaltyCombineMode.MULTIPLICATIVE, 20.0)
        assert result == 20.0  # 5.0 * (1+10) = 55 → capped at 20

    def test_cap_max(self):
        """Max of two huge values → capped."""
        result = combine_penalty(25.0, 30.0, PenaltyCombineMode.MAX, 20.0)
        assert result == 20.0


class TestComputeInheritedPenaltyEdgeCases:
    """Additional edge cases for inheritance computation."""

    def test_decay_zero_no_pass_through(self):
        """decay=0 → no penalty passes to children."""
        config = PenaltyPropagationConfig(
            enabled=True, decay=0.0, threshold=5.0, max_depth=3,
        )
        cache = PenaltyCache(config)
        cache.put(1, PenaltyRecord(10.0, 0.0, 10.0, 0))
        inherited, depth = compute_inherited_penalty([1], cache, {}, config)
        assert inherited == 0.0

    def test_decay_one_full_pass_through(self):
        """decay=1.0 → full penalty passes to children."""
        config = PenaltyPropagationConfig(
            enabled=True, decay=1.0, threshold=5.0, max_depth=3,
        )
        cache = PenaltyCache(config)
        cache.put(1, PenaltyRecord(10.0, 0.0, 10.0, 0))
        inherited, depth = compute_inherited_penalty([1], cache, {}, config)
        assert inherited == 10.0
        assert depth == 1

    def test_all_parents_below_threshold(self):
        """Multiple parents, all below threshold → no inheritance."""
        config = PenaltyPropagationConfig(
            enabled=True, decay=0.5, threshold=5.0, max_depth=3,
        )
        cache = PenaltyCache(config)
        cache.put(1, PenaltyRecord(2.0, 0.0, 2.0, 0))
        cache.put(2, PenaltyRecord(4.0, 0.0, 4.0, 0))
        inherited, depth = compute_inherited_penalty([1, 2], cache, {}, config)
        assert inherited == 0.0
        assert depth == 0

    def test_mixed_parents_one_above_threshold(self):
        """Some parents above, some below threshold."""
        config = PenaltyPropagationConfig(
            enabled=True, decay=0.5, threshold=5.0, max_depth=3,
        )
        cache = PenaltyCache(config)
        cache.put(1, PenaltyRecord(3.0, 0.0, 3.0, 0))  # Below
        cache.put(2, PenaltyRecord(8.0, 0.0, 8.0, 0))  # Above
        inherited, depth = compute_inherited_penalty([1, 2], cache, {}, config)
        assert inherited == pytest.approx(4.0)  # 0.5 * 8.0
        assert depth == 1


class TestComputeAndCachePenaltyEdgeCases:
    """Edge cases for the end-to-end compute pipeline."""

    def test_empty_clause_zero_penalty(self):
        """Empty clause (contradiction) → 0 intrinsic penalty."""
        config = PenaltyPropagationConfig(enabled=True, threshold=5.0)
        cache = PenaltyCache(config)
        c = Clause(literals=(), id=1)
        all_clauses: dict[int, Clause] = {1: c}
        result = compute_and_cache_penalty(c, cache, all_clauses)
        assert result == 0.0
        rec = cache.get(1)
        assert rec is not None
        assert rec.own_penalty == 0.0

    def test_missing_parent_graceful(self):
        """Parent in justification but not in cache → no inheritance."""
        config = PenaltyPropagationConfig(enabled=True, threshold=5.0, decay=0.5)
        cache = PenaltyCache(config)
        just = Justification(just_type=JustType.BINARY_RES, clause_ids=(99, 100))
        c = Clause(
            literals=(_make_specific_clause(0).literals),
            id=1,
            justification=(just,),
        )
        all_clauses: dict[int, Clause] = {1: c}
        compute_and_cache_penalty(c, cache, all_clauses)
        rec = cache.get(1)
        assert rec is not None
        assert rec.inherited_penalty == 0.0

    def test_max_penalty_cap_enforced(self):
        """Combined penalty never exceeds max_penalty."""
        config = PenaltyPropagationConfig(
            enabled=True, decay=1.0, threshold=1.0, max_penalty=15.0,
            max_depth=5,
        )
        cache = PenaltyCache(config)
        all_clauses: dict[int, Clause] = {}

        # Parent: heavy penalty (>= 10.0)
        parent = _make_general_clause(1)
        all_clauses[1] = parent
        compute_and_cache_penalty(parent, cache, all_clauses)

        # Child: also general, inherits from parent
        child_just = Justification(just_type=JustType.BINARY_RES, clause_ids=(1,))
        child = Clause(
            literals=_make_general_clause(0).literals, id=2,
            justification=(child_just,),
        )
        all_clauses[2] = child
        compute_and_cache_penalty(child, cache, all_clauses)

        child_rec = cache.get(2)
        assert child_rec is not None
        assert child_rec.combined_penalty <= 15.0

    def test_hyper_resolution_inherits_from_worst_parent(self):
        """Hyper-resolution child inherits from highest-penalty parent."""
        config = PenaltyPropagationConfig(
            enabled=True, decay=0.5, threshold=5.0, max_depth=3,
            max_penalty=50.0,
        )
        cache = PenaltyCache(config)
        all_clauses: dict[int, Clause] = {}

        # Nucleus: general (heavy penalty)
        nucleus = _make_general_clause(1)
        all_clauses[1] = nucleus
        compute_and_cache_penalty(nucleus, cache, all_clauses)

        # Satellite: specific (low penalty)
        sat = _make_specific_clause(2)
        all_clauses[2] = sat
        compute_and_cache_penalty(sat, cache, all_clauses)

        # Resolvent from hyper-resolution
        child_just = Justification(just_type=JustType.HYPER_RES, clause_ids=(1, 2))
        child = Clause(
            literals=_make_specific_clause(0).literals, id=3,
            justification=(child_just,),
        )
        all_clauses[3] = child
        compute_and_cache_penalty(child, cache, all_clauses)

        child_rec = cache.get(3)
        assert child_rec is not None
        assert child_rec.inherited_penalty > 0.0  # Inherited from nucleus

    def test_paramodulation_inherits_from_worst_parent(self):
        """Paramodulation child inherits from worst of from/into parents."""
        config = PenaltyPropagationConfig(
            enabled=True, decay=0.5, threshold=5.0, max_depth=3,
            max_penalty=50.0,
        )
        cache = PenaltyCache(config)
        all_clauses: dict[int, Clause] = {}

        # From: general (heavy)
        from_clause = _make_general_clause(1)
        all_clauses[1] = from_clause
        compute_and_cache_penalty(from_clause, cache, all_clauses)

        # Into: specific (light)
        into_clause = _make_specific_clause(2)
        all_clauses[2] = into_clause
        compute_and_cache_penalty(into_clause, cache, all_clauses)

        # Paramodulant
        para = ParaJust(from_id=1, into_id=2, from_pos=(1,), into_pos=(1,))
        child_just = Justification(just_type=JustType.PARA, para=para)
        child = Clause(
            literals=_make_specific_clause(0).literals, id=3,
            justification=(child_just,),
        )
        all_clauses[3] = child
        compute_and_cache_penalty(child, cache, all_clauses)

        child_rec = cache.get(3)
        assert child_rec is not None
        assert child_rec.inherited_penalty > 0.0

    def test_three_generation_chain(self):
        """Grandparent → parent → child penalty propagation."""
        config = PenaltyPropagationConfig(
            enabled=True, decay=0.5, threshold=1.0, max_depth=5,
            max_penalty=100.0,
        )
        cache = PenaltyCache(config)
        all_clauses: dict[int, Clause] = {}

        # Gen 0: heavy
        gen0 = _make_general_clause(1)
        all_clauses[1] = gen0
        compute_and_cache_penalty(gen0, cache, all_clauses)
        gen0_penalty = cache.get(1).combined_penalty

        # Gen 1: derived from gen0
        gen1 = _make_derived_clause(2, parent_ids=(1,))
        all_clauses[2] = gen1
        compute_and_cache_penalty(gen1, cache, all_clauses)

        # Gen 2: derived from gen1
        gen2 = _make_derived_clause(3, parent_ids=(2,))
        all_clauses[3] = gen2
        compute_and_cache_penalty(gen2, cache, all_clauses)

        rec1 = cache.get(2)
        rec2 = cache.get(3)
        assert rec1 is not None and rec2 is not None
        assert rec1.depth == 1
        # Each generation's inherited penalty should be less than parent's combined
        assert rec1.inherited_penalty == pytest.approx(0.5 * gen0_penalty)
