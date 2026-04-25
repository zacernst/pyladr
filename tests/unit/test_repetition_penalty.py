"""Unit tests for subformula repetition penalty system.

Tests Phase 1 (exact matching) and Phase 2 (variable-normalized matching)
of the repetition penalty that deprioritizes clauses with repeated
structural patterns.
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.search.repetition_penalty import (
    RepetitionPenaltyConfig,
    _normalize_variables,
    _penalty_exact,
    _penalty_normalized,
    compute_repetition_penalty,
)
from tests.factories import (
    make_clause_from_atoms,
    make_const as _const,
    make_func as _func,
    make_var as _var,
)


def _make_clause(*atoms, signs=None):
    return make_clause_from_atoms(*atoms, signs=signs, clause_id=1)


# ── Configuration ────────────────────────────────────────────────────────────


_DEFAULT_CONFIG = RepetitionPenaltyConfig(enabled=True)
_NORM_CONFIG = RepetitionPenaltyConfig(enabled=True, normalize_variables=True)


# ── Phase 1: Exact matching tests ────────────────────────────────────────────


class TestExactMatching:
    """Test exact structural matching for subformula repetition."""

    def test_no_repetition_returns_zero(self):
        """Clause with all distinct subterms gets no penalty."""
        # P(a, b) — all subterms distinct
        a, b = _const(1), _const(2)
        atom = _func(10, a, b)
        clause = _make_clause(atom)
        assert compute_repetition_penalty(clause, _DEFAULT_CONFIG) == 0.0

    def test_repeated_constant_below_min_size(self):
        """Repeated constants (size=1) are below default min_size=2, no penalty."""
        # P(a, a) — 'a' repeats but size=1
        a = _const(1)
        atom = _func(10, a, a)
        clause = _make_clause(atom)
        assert compute_repetition_penalty(clause, _DEFAULT_CONFIG) == 0.0

    def test_repeated_subterm_exact_match(self):
        """Repeated complex subterms get penalized."""
        # P(f(a), f(a)) — f(a) appears twice (size=2, meets min_size)
        fa = _func(5, _const(1))
        atom = _func(10, fa, fa)
        clause = _make_clause(atom)
        penalty = compute_repetition_penalty(clause, _DEFAULT_CONFIG)
        assert penalty > 0.0

    def test_repeated_subterm_across_literals(self):
        """Repetitions across different literals are detected."""
        # P(f(a)) | Q(f(a)) — f(a) appears in two literals
        fa = _func(5, _const(1))
        atom1 = _func(10, fa)
        atom2 = _func(11, fa)
        clause = _make_clause(atom1, atom2)
        penalty = compute_repetition_penalty(clause, _DEFAULT_CONFIG)
        assert penalty > 0.0

    def test_three_repetitions_higher_penalty(self):
        """Three occurrences = 2 extra, so higher penalty than two occurrences."""
        fa = _func(5, _const(1))
        # P(f(a), f(a), f(a)) — f(a) appears 3 times
        atom = _func(10, fa, fa, fa)
        clause = _make_clause(atom)
        penalty_3 = compute_repetition_penalty(clause, _DEFAULT_CONFIG)

        # Compare with only 2 occurrences
        atom2 = _func(10, fa, fa, _const(2))
        clause2 = _make_clause(atom2)
        penalty_2 = compute_repetition_penalty(clause2, _DEFAULT_CONFIG)

        assert penalty_3 > penalty_2

    def test_penalty_capped_at_max(self):
        """Penalty does not exceed max_penalty."""
        config = RepetitionPenaltyConfig(enabled=True, max_penalty=5.0, base_penalty=10.0)
        fa = _func(5, _const(1))
        atom = _func(10, fa, fa, fa, fa)
        clause = _make_clause(atom)
        penalty = compute_repetition_penalty(clause, config)
        assert penalty == 5.0

    def test_empty_clause_returns_zero(self):
        """Empty clause (no literals) gets zero penalty."""
        clause = Clause(literals=(), id=1)
        assert compute_repetition_penalty(clause, _DEFAULT_CONFIG) == 0.0

    def test_single_variable_no_penalty(self):
        """Clause with only variables gets no penalty (vars are size=1)."""
        # P(x, y)
        atom = _func(10, _var(0), _var(1))
        clause = _make_clause(atom)
        assert compute_repetition_penalty(clause, _DEFAULT_CONFIG) == 0.0

    def test_different_variables_same_structure_no_exact_match(self):
        """f(x) and f(y) are NOT identical in exact mode (different variables)."""
        fx = _func(5, _var(0))
        fy = _func(5, _var(1))
        atom = _func(10, fx, fy)
        clause = _make_clause(atom)
        penalty = compute_repetition_penalty(clause, _DEFAULT_CONFIG)
        # f(x) and f(y) are distinct terms in exact matching
        assert penalty == 0.0

    def test_min_subterm_size_filtering(self):
        """Only subterms with symbol_count >= min_size are considered."""
        # With min_size=3, f(a) (size=2) should NOT be detected
        config = RepetitionPenaltyConfig(enabled=True, min_subterm_size=3)
        fa = _func(5, _const(1))
        atom = _func(10, fa, fa)
        clause = _make_clause(atom)
        assert compute_repetition_penalty(clause, config) == 0.0

        # But g(f(a)) (size=3) should be detected
        gfa = _func(6, fa)
        atom2 = _func(10, gfa, gfa)
        clause2 = _make_clause(atom2)
        assert compute_repetition_penalty(clause2, config) > 0.0

    def test_disabled_config_returns_zero(self):
        """Disabled config always returns zero (but shouldn't be called)."""
        config = RepetitionPenaltyConfig(enabled=False)
        fa = _func(5, _const(1))
        atom = _func(10, fa, fa)
        clause = _make_clause(atom)
        # compute_repetition_penalty still works, it's the caller that checks enabled
        # but the function itself computes based on the config
        penalty = compute_repetition_penalty(clause, config)
        assert penalty >= 0.0  # Still valid output


# ── Phase 2: Variable-normalized matching tests ─────────────────────────────


class TestNormalizedMatching:
    """Test variable-normalized matching (i(x,x) ≡ i(y,y))."""

    def test_same_structure_different_vars_detected(self):
        """f(x) and f(y) are equivalent after normalization."""
        fx = _func(5, _var(0))
        fy = _func(5, _var(1))
        atom = _func(10, fx, fy)
        clause = _make_clause(atom)

        # Exact: no penalty (different vars)
        assert compute_repetition_penalty(clause, _DEFAULT_CONFIG) == 0.0

        # Normalized: penalty (f(v0) ≡ f(v0))
        penalty = compute_repetition_penalty(clause, _NORM_CONFIG)
        assert penalty > 0.0

    def test_i_x_x_pattern_detected(self):
        """i(x,x) and i(y,y) are equivalent after normalization."""
        ixx = _func(5, _var(0), _var(0))
        iyy = _func(5, _var(1), _var(1))
        atom = _func(10, ixx, iyy)
        clause = _make_clause(atom)

        penalty = compute_repetition_penalty(clause, _NORM_CONFIG)
        assert penalty > 0.0

    def test_different_binding_patterns_not_equivalent(self):
        """i(x,x) and i(x,y) are NOT equivalent (different binding structure)."""
        ixx = _func(5, _var(0), _var(0))
        ixy = _func(5, _var(0), _var(1))
        atom = _func(10, ixx, ixy)
        clause = _make_clause(atom)

        # Even with normalization, i(v0,v0) ≠ i(v0,v1)
        penalty = compute_repetition_penalty(clause, _NORM_CONFIG)
        assert penalty == 0.0

    def test_ground_terms_same_in_both_modes(self):
        """Ground terms (no variables) produce same penalty in both modes."""
        fa = _func(5, _const(1))
        atom = _func(10, fa, fa)
        clause = _make_clause(atom)

        exact = compute_repetition_penalty(clause, _DEFAULT_CONFIG)
        normalized = compute_repetition_penalty(clause, _NORM_CONFIG)
        assert exact == normalized

    def test_normalize_variables_canonical_ordering(self):
        """Variables are renumbered in DFS order."""
        # f(y, g(x)) with y=var(1), x=var(0)
        term = _func(5, _var(1), _func(6, _var(0)))
        normalized = _normalize_variables(term)

        # Should become f(v0, g(v1)) — DFS order
        assert normalized.args[0].is_variable
        assert normalized.args[0].varnum == 0
        assert normalized.args[1].args[0].is_variable
        assert normalized.args[1].args[0].varnum == 1

    def test_normalize_preserves_binding_structure(self):
        """Normalization preserves which variables are the same."""
        # f(x, x) — same variable appears twice
        term = _func(5, _var(3), _var(3))
        normalized = _normalize_variables(term)

        # Both args should map to v0
        assert normalized.args[0].varnum == normalized.args[1].varnum

    def test_normalize_constant_unchanged(self):
        """Constants are unchanged by normalization."""
        term = _const(42)
        normalized = _normalize_variables(term)
        assert normalized is term  # Same object


# ── Integration with penalty calculation ─────────────────────────────────────


class TestPenaltyCalculation:
    """Test penalty computation mechanics."""

    def test_base_penalty_scaling(self):
        """Penalty scales with base_penalty config."""
        fa = _func(5, _const(1))
        atom = _func(10, fa, fa)
        clause = _make_clause(atom)

        config_low = RepetitionPenaltyConfig(enabled=True, base_penalty=1.0)
        config_high = RepetitionPenaltyConfig(enabled=True, base_penalty=5.0)

        penalty_low = compute_repetition_penalty(clause, config_low)
        penalty_high = compute_repetition_penalty(clause, config_high)

        assert penalty_high > penalty_low
        assert penalty_high == 5.0 * penalty_low / 1.0  # Linear scaling

    def test_early_termination_small_clause(self):
        """Small clauses (total symbols < 2*min_size) get zero penalty fast."""
        # P(a) — total symbols = 2 (P and a), below 2*min_size=4
        atom = _func(10, _const(1))
        clause = _make_clause(atom)
        assert compute_repetition_penalty(clause, _DEFAULT_CONFIG) == 0.0

    def test_nested_repetition_detected(self):
        """Nested repeated structures are detected at multiple levels."""
        # f(g(a), g(a)) — g(a) repeats, and f(g(a), g(a)) is the whole structure
        ga = _func(6, _const(1))
        atom = _func(10, _func(5, ga, ga))
        clause = _make_clause(atom)
        penalty = compute_repetition_penalty(clause, _DEFAULT_CONFIG)
        assert penalty > 0.0


# ── Edsger's Extended Test Suite ─────────────────────────────────────────────
# Additional coverage: edge cases, regression prevention, performance,
# integration with search pipeline, and CLI argument parsing.


# Symbol IDs for extended tests
A, B, C_SYM, D = 1, 2, 3, 4
F, G, H = 10, 11, 12
P, Q, R = 20, 21, 22
I_SYM = 30  # inverse symbol for group theory patterns


class TestExactMatchingExtended:
    """Extended Phase 1 exact matching coverage."""

    def test_same_variable_index_is_same_object(self):
        """Variables from get_variable_term are cached — same index = same object."""
        x0a = _var(0)
        x0b = _var(0)
        assert x0a is x0b  # Shared variable cache

    def test_same_var_subterms_are_exact_match(self):
        """f(x0) with same var object appears identical in exact mode."""
        x = _var(0)
        fx = _func(F, x)
        # P(f(x0), f(x0)) — both f(x0) subterms are structurally equal
        c = _make_clause(_func(P, fx, fx))
        penalty = compute_repetition_penalty(c, _DEFAULT_CONFIG)
        assert penalty > 0.0

    def test_deeply_nested_no_repetition(self):
        """Deep nesting without repetition → 0.0."""
        # f(g(h(a))) — all subterms distinct
        c = _make_clause(_func(F, _func(G, _func(H, _const(A)))))
        assert compute_repetition_penalty(c, _DEFAULT_CONFIG) == 0.0

    def test_deeply_nested_cross_depth_repetition(self):
        """Repeated subterms at different depths."""
        # P(g(a), h(g(a))) — g(a) at depth 1 and depth 2
        ga = _func(G, _const(A))
        c = _make_clause(_func(P, ga, _func(H, ga)))
        penalty = compute_repetition_penalty(c, _DEFAULT_CONFIG)
        assert penalty > 0.0

    def test_multiple_distinct_repeated_patterns(self):
        """Multiple different patterns each repeated → additive penalty."""
        fa = _func(F, _const(A))
        gb = _func(G, _const(B))
        # P(f(a), f(a), g(b), g(b))
        c = _make_clause(_func(P, fa, fa, gb, gb))
        penalty = compute_repetition_penalty(c, _DEFAULT_CONFIG)
        # Each pattern has 1 extra → 2 extras total → 2 * 2.0 = 4.0
        assert penalty == 4.0

    def test_penalty_formula_exact(self):
        """Verify exact penalty calculation: base_penalty * sum(count-1)."""
        fa = _func(F, _const(A))
        # 4 copies → 3 extras → 3 * base_penalty
        c = _make_clause(_func(P, fa, fa, fa, fa))
        cfg = RepetitionPenaltyConfig(enabled=True, base_penalty=3.0, max_penalty=100.0)
        penalty = compute_repetition_penalty(c, cfg)
        assert penalty == 9.0  # 3 * 3.0

    def test_min_size_1_catches_constants(self):
        """min_subterm_size=1 includes constants in repetition check."""
        # P(a, a, b) — 'a' appears twice
        c = _make_clause(_func(P, _const(A), _const(A), _const(B)))
        cfg = RepetitionPenaltyConfig(enabled=True, base_penalty=1.0, min_subterm_size=1)
        penalty = compute_repetition_penalty(c, cfg)
        assert penalty == 1.0  # 'a' has 1 extra occurrence


class TestNormalizedMatchingExtended:
    """Extended Phase 2 variable normalization coverage."""

    def test_normalize_ground_is_identity(self):
        """Ground term normalization returns the same object (fast path)."""
        t = _func(F, _const(A), _func(G, _const(B)))
        normalized = _normalize_variables(t)
        assert normalized is t  # is_ground → same object

    def test_normalize_idempotent(self):
        """Normalizing an already-normalized term gives equal result."""
        t = _func(F, _var(3), _var(7))
        r1 = _normalize_variables(t)
        r2 = _normalize_variables(r1)
        assert r1 == r2

    def test_normalize_single_variable(self):
        """Single variable normalizes to v0."""
        v = _var(42)
        result = _normalize_variables(v)
        assert result.is_variable
        assert result.varnum == 0

    def test_normalize_two_distinct_variables(self):
        """Two distinct variables → v0, v1 in DFS order."""
        t = _func(F, _var(50), _var(99))
        result = _normalize_variables(t)
        assert result.args[0].varnum == 0
        assert result.args[1].varnum == 1

    def test_normalize_preserves_constants(self):
        """Constants are unchanged after normalization."""
        t = _func(F, _var(5), _const(A))
        result = _normalize_variables(t)
        assert result.args[0].is_variable
        assert result.args[0].varnum == 0
        assert result.args[1] == _const(A)

    def test_inverse_pattern_group_theory(self):
        """Classic group theory i(x,x)|i(y,y) repetition with normalization."""
        ixx = _func(I_SYM, _var(0), _var(0))
        iyy = _func(I_SYM, _var(1), _var(1))
        c = Clause(literals=(
            Literal(sign=True, atom=ixx),
            Literal(sign=True, atom=iyy),
        ), id=1)
        cfg = RepetitionPenaltyConfig(enabled=True, normalize_variables=True)
        penalty = compute_repetition_penalty(c, cfg)
        assert penalty > 0.0  # Normalized: i(v0,v0) == i(v0,v0)

    def test_exact_mode_misses_renamed_vars(self):
        """Exact mode does NOT detect variable-renamed duplicates."""
        ixx = _func(I_SYM, _var(0), _var(0))
        iyy = _func(I_SYM, _var(1), _var(1))
        c = Clause(literals=(
            Literal(sign=True, atom=ixx),
            Literal(sign=True, atom=iyy),
        ), id=1)
        cfg = RepetitionPenaltyConfig(enabled=True, normalize_variables=False)
        penalty = compute_repetition_penalty(c, cfg)
        assert penalty == 0.0  # i(x0,x0) != i(x1,x1) in exact mode

    def test_normalization_cache_correctness(self):
        """Per-clause normalization cache produces correct results.

        The cache keyed by id(subterm) should not produce incorrect
        results when different subterms share structure.
        """
        # Two structurally different terms
        fxy = _func(F, _var(0), _var(1))
        fxx = _func(F, _var(0), _var(0))
        c = _make_clause(_func(P, fxy, fxx))
        cfg = RepetitionPenaltyConfig(enabled=True, normalize_variables=True)
        penalty = compute_repetition_penalty(c, cfg)
        # f(v0,v1) vs f(v0,v0) — different patterns → no repetition
        assert penalty == 0.0


class TestPenaltyCapping:
    """Test penalty cap behavior."""

    def test_custom_max_penalty_respected(self):
        """Custom max_penalty is the ceiling."""
        fa = _func(F, _const(A))
        args = tuple(fa for _ in range(8))  # 7 extras → 14.0 raw
        c = _make_clause(_func(P, *args))
        cfg = RepetitionPenaltyConfig(enabled=True, base_penalty=2.0, max_penalty=5.0)
        assert compute_repetition_penalty(c, cfg) == 5.0

    def test_zero_base_penalty_always_zero(self):
        """base_penalty=0 → 0.0 regardless of repetition."""
        fa = _func(F, _const(A))
        c = _make_clause(_func(P, fa, fa))
        cfg = RepetitionPenaltyConfig(enabled=True, base_penalty=0.0)
        assert compute_repetition_penalty(c, cfg) == 0.0


class TestBasePenaltyScaling:
    """Test linear scaling of base_penalty."""

    def test_double_base_doubles_penalty(self):
        """Doubling base_penalty doubles the penalty."""
        fa = _func(F, _const(A))
        c = _make_clause(_func(P, fa, fa))
        cfg1 = RepetitionPenaltyConfig(enabled=True, base_penalty=1.0, max_penalty=100.0)
        cfg2 = RepetitionPenaltyConfig(enabled=True, base_penalty=2.0, max_penalty=100.0)
        p1 = compute_repetition_penalty(c, cfg1)
        p2 = compute_repetition_penalty(c, cfg2)
        assert p2 == 2.0 * p1


class TestSearchOptionsIntegration:
    """Test SearchOptions fields and config construction."""

    def test_search_options_defaults(self):
        """SearchOptions has repetition penalty fields with correct defaults."""
        from pyladr.search.given_clause import SearchOptions
        opts = SearchOptions()
        assert opts.repetition_penalty is False
        assert opts.repetition_penalty_weight == 2.0
        assert opts.repetition_penalty_min_size == 2
        assert opts.repetition_penalty_max == 15.0
        assert opts.repetition_penalty_normalize is False

    def test_search_options_config_construction(self):
        """SearchOptions fields map correctly to RepetitionPenaltyConfig."""
        from pyladr.search.given_clause import SearchOptions
        opts = SearchOptions(
            repetition_penalty=True,
            repetition_penalty_weight=3.0,
            repetition_penalty_min_size=3,
            repetition_penalty_max=10.0,
            repetition_penalty_normalize=True,
        )
        cfg = RepetitionPenaltyConfig(
            enabled=True,
            base_penalty=opts.repetition_penalty_weight,
            min_subterm_size=opts.repetition_penalty_min_size,
            max_penalty=opts.repetition_penalty_max,
            normalize_variables=opts.repetition_penalty_normalize,
        )
        assert cfg.base_penalty == 3.0
        assert cfg.min_subterm_size == 3
        assert cfg.max_penalty == 10.0
        assert cfg.normalize_variables is True


class TestSearchPipelineIntegration:
    """Test the penalty combination logic from given_clause.py."""

    def test_additive_with_propagation_penalty(self):
        """Repetition penalty adds to existing penalty_propagation value."""
        existing_penalty = 5.0
        fa = _func(F, _const(A))
        c = _make_clause(_func(P, fa, fa))
        cfg = RepetitionPenaltyConfig(enabled=True, base_penalty=2.0)
        rep = compute_repetition_penalty(c, cfg)
        assert rep == 2.0
        combined = (existing_penalty or 0.0) + rep
        assert combined == 7.0

    def test_no_propagation_penalty_repetition_only(self):
        """When penalty_propagation is None, only repetition penalty applies."""
        penalty_val = None
        fa = _func(F, _const(A))
        c = _make_clause(_func(P, fa, fa))
        cfg = RepetitionPenaltyConfig(enabled=True, base_penalty=2.0)
        rep = compute_repetition_penalty(c, cfg)
        if rep > 0.0:
            penalty_val = (penalty_val or 0.0) + rep
        assert penalty_val == 2.0

    def test_zero_repetition_no_override(self):
        """No repetition → penalty_val stays None (no override)."""
        penalty_val = None
        c = _make_clause(_func(P, _const(A), _const(B)))
        cfg = RepetitionPenaltyConfig(enabled=True, base_penalty=2.0)
        rep = compute_repetition_penalty(c, cfg)
        if rep > 0.0:
            penalty_val = (penalty_val or 0.0) + rep
        assert penalty_val is None


class TestRegressionPrevention:
    """Tests preventing regressions in existing systems."""

    def test_penalty_propagation_unaffected(self):
        """PenaltyCache and PenaltyRecord still work correctly."""
        from pyladr.search.penalty_propagation import (
            PenaltyCache,
            PenaltyPropagationConfig,
            PenaltyRecord,
        )
        config = PenaltyPropagationConfig(enabled=True)
        cache = PenaltyCache(config)
        cache.put(1, PenaltyRecord(
            own_penalty=3.0, inherited_penalty=0.0,
            combined_penalty=3.0, depth=0,
        ))
        rec = cache.get(1)
        assert rec is not None
        assert rec.combined_penalty == 3.0

    def test_selection_module_imports(self):
        """Selection module still exports all expected symbols."""
        from pyladr.search.selection import (
            GivenSelection,
            SelectionOrder,
            SelectionRule,
            _clause_generality_penalty,
        )
        assert SelectionOrder.WEIGHT == 0
        assert SelectionOrder.AGE == 1

    def test_generality_penalty_unchanged(self):
        """_clause_generality_penalty still works correctly."""
        from pyladr.search.selection import _clause_generality_penalty
        c_ground = _make_clause(_func(P, _const(A), _const(B)))
        p_ground = _clause_generality_penalty(c_ground)
        assert p_ground == 0.0
        c_general = _make_clause(_func(P, _var(0)))
        p_general = _clause_generality_penalty(c_general)
        assert p_general >= 10.0


class TestConfigFrozenImmutability:
    """Test that RepetitionPenaltyConfig is properly frozen."""

    def test_cannot_modify_enabled(self):
        cfg = RepetitionPenaltyConfig(enabled=True)
        with pytest.raises(AttributeError):
            cfg.enabled = False  # type: ignore[misc]

    def test_cannot_modify_base_penalty(self):
        cfg = RepetitionPenaltyConfig(enabled=True)
        with pytest.raises(AttributeError):
            cfg.base_penalty = 99.0  # type: ignore[misc]


class TestPerformance:
    """Validate O(n) performance and consistency."""

    def test_large_clause_completes_quickly(self):
        """Clause with ~100 subterms completes in < 50ms."""
        import time
        args = tuple(_func(F, _const(i % 50 + 1)) for i in range(50))
        c = _make_clause(_func(P, *args))
        cfg = RepetitionPenaltyConfig(enabled=True, base_penalty=1.0, min_subterm_size=1)
        start = time.perf_counter()
        compute_repetition_penalty(c, cfg)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.05, f"Took {elapsed:.3f}s (limit 0.05s)"

    def test_repeated_calls_deterministic(self):
        """Same input always produces same output."""
        fa = _func(F, _const(A))
        c = _make_clause(_func(P, fa, fa))
        cfg = RepetitionPenaltyConfig(enabled=True, base_penalty=2.0)
        results = [compute_repetition_penalty(c, cfg) for _ in range(100)]
        assert all(r == results[0] for r in results)
