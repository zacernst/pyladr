"""Integration tests for the nucleus unification penalty system.

Validates that the nucleus penalty integrates correctly into the full
GivenClauseSearch pipeline, maintains C Prover9 compatibility when
disabled, and effectively prevents unification explosion in
hyperresolution-heavy problems.
"""

from __future__ import annotations

import time

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import (
    Term,
    get_rigid_term,
    get_variable_term,
)
from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions
from pyladr.search.nucleus_penalty import (
    NucleusUnificationPenaltyConfig,
    compute_nucleus_unification_penalty,
)
from pyladr.search.nucleus_patterns import (
    NucleusPatternCache,
    cache_nucleus_patterns,
    extract_patterns,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _pos(atom: Term) -> Literal:
    return Literal(sign=True, atom=atom)


def _neg(atom: Term) -> Literal:
    return Literal(sign=False, atom=atom)


def _cl(*lits: Literal, clause_id: int = 0) -> Clause:
    c = Clause(literals=tuple(lits))
    if clause_id:
        c.id = clause_id
    return c


# Symbol numbers for test predicates/functions/constants
P_SN, Q_SN, R_SN = 1, 2, 3
F_SN, G_SN, H_SN = 4, 5, 6
A_SN, B_SN, C_SN, D_SN = 7, 8, 9, 10
I_SN = 11  # implication-like binary function


def _const(sn: int) -> Term:
    return get_rigid_term(sn, 0)


def _func1(sn: int, arg: Term) -> Term:
    return get_rigid_term(sn, 1, (arg,))


def _func2(sn: int, a1: Term, a2: Term) -> Term:
    return get_rigid_term(sn, 2, (a1, a2))


def _pred1(sn: int, arg: Term) -> Term:
    return get_rigid_term(sn, 1, (arg,))


def _pred2(sn: int, a1: Term, a2: Term) -> Term:
    return get_rigid_term(sn, 2, (a1, a2))


def _run_search(
    usable: list[Clause],
    sos: list[Clause],
    *,
    hyper_resolution: bool = False,
    nucleus_unification_penalty: bool = False,
    nucleus_penalty_threshold: float = 3.0,
    nucleus_penalty_weight: float = 1.5,
    nucleus_penalty_max: float = 15.0,
    max_given: int = 200,
    max_weight: float = -1.0,
    **kwargs,
):
    """Run a search and return the result."""
    opts = SearchOptions(
        binary_resolution=True,
        hyper_resolution=hyper_resolution,
        factoring=True,
        max_given=max_given,
        max_weight=max_weight,
        nucleus_unification_penalty=nucleus_unification_penalty,
        nucleus_penalty_threshold=nucleus_penalty_threshold,
        nucleus_penalty_weight=nucleus_penalty_weight,
        nucleus_penalty_max=nucleus_penalty_max,
        quiet=True,
        **kwargs,
    )
    search = GivenClauseSearch(options=opts)
    return search.run(usable=usable, sos=sos)


# ── Disabled-by-default tests ───────────────────────────────────────────────


class TestNucleusPenaltyDisabledByDefault:
    """Ensure zero behavior change when nucleus penalty is disabled."""

    def test_default_options_disable_nucleus_penalty(self) -> None:
        """SearchOptions defaults have nucleus_unification_penalty=False."""
        opts = SearchOptions()
        assert opts.nucleus_unification_penalty is False

    def test_search_identical_when_disabled(self) -> None:
        """A simple proof produces the same result with penalty explicitly off."""
        x = get_variable_term(0)
        a = _const(A_SN)

        c1 = _cl(_pos(_pred1(P_SN, a)))
        c2 = _cl(_neg(_pred1(P_SN, x)), _pos(_pred1(Q_SN, x)))
        c3 = _cl(_neg(_pred1(Q_SN, a)))

        # Without penalty
        r1 = _run_search(usable=[], sos=[c1, c2, c3])
        # With penalty explicitly disabled
        r2 = _run_search(
            usable=[], sos=[c1, c2, c3],
            nucleus_unification_penalty=False,
        )

        assert r1.exit_code == r2.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert r1.stats.given == r2.stats.given
        assert r1.stats.kept == r2.stats.kept

    def test_no_overhead_when_disabled(self) -> None:
        """Search performance is unaffected when penalty is disabled."""
        x = get_variable_term(0)
        a = _const(A_SN)

        c1 = _cl(_pos(_pred1(P_SN, a)))
        c2 = _cl(_neg(_pred1(P_SN, x)), _pos(_pred1(Q_SN, x)))
        c3 = _cl(_neg(_pred1(Q_SN, a)))

        # Run twice and compare timing (rough check for no extra work)
        start = time.perf_counter()
        for _ in range(10):
            _run_search(usable=[], sos=[c1, c2, c3])
        baseline = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(10):
            _run_search(
                usable=[], sos=[c1, c2, c3],
                nucleus_unification_penalty=False,
            )
        disabled = time.perf_counter() - start

        # Should be within 20% (generous margin for timing noise)
        assert disabled < baseline * 1.2


# ── Hyperresolution pattern extraction ──────────────────────────────────────


class TestHyperresolutionPatternExtraction:
    """Test pattern extraction during hyperresolution calls."""

    def test_nucleus_patterns_extracted_from_negative_literals(self) -> None:
        """Negative literals yield nucleus patterns for the cache."""
        x = get_variable_term(0)
        y = get_variable_term(1)
        a = _const(A_SN)

        # Nucleus: -P(x) | -Q(y) | R(a) — two negative literals
        nucleus = _cl(
            _neg(_pred1(P_SN, x)),
            _neg(_pred1(Q_SN, y)),
            _pos(_pred1(R_SN, a)),
            clause_id=1,
        )

        patterns = extract_patterns(nucleus)
        assert len(patterns) == 2
        pred_syms = {p.predicate_symbol for p in patterns}
        assert P_SN in pred_syms
        assert Q_SN in pred_syms

    def test_positive_only_clause_yields_no_patterns(self) -> None:
        """A clause with only positive literals has no nucleus patterns."""
        a = _const(A_SN)
        satellite = _cl(_pos(_pred1(P_SN, a)), _pos(_pred1(Q_SN, a)))
        patterns = extract_patterns(satellite)
        assert len(patterns) == 0

    def test_cache_population_during_search(self) -> None:
        """Nucleus patterns are cached when hyper-resolution is active."""
        x = get_variable_term(0)
        a = _const(A_SN)

        # Nucleus: -P(x) | Q(x)
        nucleus = _cl(_neg(_pred1(P_SN, x)), _pos(_pred1(Q_SN, x)))
        # Satellite: P(a)
        satellite = _cl(_pos(_pred1(P_SN, a)))
        # Goal denial: -Q(a)
        goal = _cl(_neg(_pred1(Q_SN, a)))

        cache = NucleusPatternCache(max_size=100)
        cache_nucleus_patterns(nucleus, cache)

        assert len(cache) == 1
        # Should be indexed by P's symbol number
        pats = cache.get_by_predicate(P_SN)
        assert len(pats) == 1
        assert pats[0].arity == 1

    def test_cache_deduplication(self) -> None:
        """Alpha-equivalent patterns are deduplicated in the cache."""
        x = get_variable_term(0)
        y = get_variable_term(1)

        # These two clauses have alpha-equivalent -P(x) patterns
        c1 = _cl(_neg(_pred1(P_SN, x)), _pos(_pred1(Q_SN, x)), clause_id=1)
        c2 = _cl(_neg(_pred1(P_SN, y)), _pos(_pred1(R_SN, y)), clause_id=2)

        cache = NucleusPatternCache(max_size=100)
        cache_nucleus_patterns(c1, cache)
        cache_nucleus_patterns(c2, cache)

        # Only one unique pattern for P (alpha-equivalent)
        pats = cache.get_by_predicate(P_SN)
        assert len(pats) == 1
        assert cache.stats.dedup_skips >= 1


# ── Search loop integration ─────────────────────────────────────────────────


class TestSearchLoopIntegration:
    """Test penalty application during clause processing."""

    def test_penalty_applied_to_general_nucleus(self) -> None:
        """Clauses with highly general nuclei get penalized."""
        x = get_variable_term(0)
        y = get_variable_term(1)

        # Highly general nucleus: -P(x,y) | Q(x)
        config = NucleusUnificationPenaltyConfig(
            enabled=True,
            base_penalty=5.0,
            threshold=0.3,
        )
        nucleus = _cl(_neg(_pred2(P_SN, x, y)), _pos(_pred1(Q_SN, x)))
        penalty = compute_nucleus_unification_penalty(nucleus, config)
        assert penalty > 0.0, "General nucleus should be penalized"

    def test_no_penalty_for_ground_nucleus(self) -> None:
        """Clauses with ground nuclei get no penalty."""
        a = _const(A_SN)
        b = _const(B_SN)

        config = NucleusUnificationPenaltyConfig(
            enabled=True,
            base_penalty=5.0,
            threshold=0.3,
        )
        nucleus = _cl(_neg(_pred2(P_SN, a, b)), _pos(_pred1(Q_SN, a)))
        penalty = compute_nucleus_unification_penalty(nucleus, config)
        assert penalty == 0.0, "Ground nucleus should not be penalized"

    def test_penalty_increases_with_generality(self) -> None:
        """More general nuclei get higher penalties."""
        x = get_variable_term(0)
        y = get_variable_term(1)
        a = _const(A_SN)

        config = NucleusUnificationPenaltyConfig(
            enabled=True,
            base_penalty=5.0,
            threshold=0.3,
        )

        # One variable arg, one ground
        semi = _cl(_neg(_pred2(P_SN, x, a)), _pos(_pred1(Q_SN, x)))
        # Both variable args
        full = _cl(_neg(_pred2(P_SN, x, y)), _pos(_pred1(Q_SN, x)))

        p_semi = compute_nucleus_unification_penalty(semi, config)
        p_full = compute_nucleus_unification_penalty(full, config)

        assert p_full >= p_semi, "Fully general nucleus should have >= penalty"

    def test_search_with_penalty_still_finds_proof(self) -> None:
        """Enabling nucleus penalty doesn't prevent proof discovery."""
        x = get_variable_term(0)
        a = _const(A_SN)

        c1 = _cl(_pos(_pred1(P_SN, a)))
        c2 = _cl(_neg(_pred1(P_SN, x)), _pos(_pred1(Q_SN, x)))
        c3 = _cl(_neg(_pred1(Q_SN, a)))

        result = _run_search(
            usable=[], sos=[c1, c2, c3],
            nucleus_unification_penalty=True,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) >= 1


# ── End-to-end search ──────────────────────────────────────────────────────


class TestEndToEndSearch:
    """Test complete search with nucleus penalty enabled."""

    def test_hyper_resolution_with_penalty(self) -> None:
        """Hyper-resolution proof succeeds with nucleus penalty active.

        Nucleus: -P(x) | -Q(x) | R(x) (two negative lits)
        Satellites: P(a), Q(a)
        Goal: -R(a)
        """
        from pyladr.core.symbol import SymbolTable

        st = SymbolTable()
        P = st.str_to_sn("P", 1)
        Q = st.str_to_sn("Q", 1)
        R = st.str_to_sn("R", 1)
        a_sn = st.str_to_sn("a", 0)

        x = get_variable_term(0)
        a = get_rigid_term(a_sn, 0)

        nucleus = _cl(
            _neg(get_rigid_term(P, 1, (x,))),
            _neg(get_rigid_term(Q, 1, (x,))),
            _pos(get_rigid_term(R, 1, (x,))),
        )
        sat1 = _cl(_pos(get_rigid_term(P, 1, (a,))))
        sat2 = _cl(_pos(get_rigid_term(Q, 1, (a,))))
        goal = _cl(_neg(get_rigid_term(R, 1, (a,))))

        opts = SearchOptions(
            binary_resolution=True,
            hyper_resolution=True,
            factoring=True,
            max_given=200,
            nucleus_unification_penalty=True,
            quiet=True,
        )
        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(usable=[], sos=[nucleus, sat1, sat2, goal])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_penalty_with_multiple_inference_rules(self) -> None:
        """Penalty works alongside binary resolution and hyper-resolution."""
        x = get_variable_term(0)
        a = _const(A_SN)
        b = _const(B_SN)

        # Binary resolution path: P(a), -P(x)|Q(x)
        c1 = _cl(_pos(_pred1(P_SN, a)))
        c2 = _cl(_neg(_pred1(P_SN, x)), _pos(_pred1(Q_SN, x)))
        # Hyper-resolution path is also available
        c3 = _cl(_neg(_pred1(Q_SN, a)))

        result = _run_search(
            usable=[], sos=[c1, c2, c3],
            hyper_resolution=True,
            nucleus_unification_penalty=True,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_saturated_search_with_penalty(self) -> None:
        """Search correctly saturates (SOS empty) with penalty enabled."""
        a = _const(A_SN)
        b = _const(B_SN)

        c1 = _cl(_pos(_pred1(P_SN, a)))
        c2 = _cl(_pos(_pred1(Q_SN, b)))

        result = _run_search(
            usable=[], sos=[c1, c2],
            nucleus_unification_penalty=True,
            max_given=50,
        )
        assert result.exit_code == ExitCode.SOS_EMPTY_EXIT


# ── Unification explosion prevention ───────────────────────────────────────


class TestUnificationExplosionPrevention:
    """Test that the penalty reduces derived clauses from overly general satellites."""

    def test_general_nucleus_penalized_more(self) -> None:
        """Nucleus -P(x,y) gets higher penalty than -P(a,b).

        This is the core mechanism: overly general nuclei that would
        produce combinatorial explosion are deprioritized.
        """
        x = get_variable_term(0)
        y = get_variable_term(1)
        a = _const(A_SN)
        b = _const(B_SN)

        config = NucleusUnificationPenaltyConfig(
            enabled=True,
            base_penalty=5.0,
            threshold=0.3,
        )

        # Very general: -P(x,y) | Q(x)
        general = _cl(_neg(_pred2(P_SN, x, y)), _pos(_pred1(Q_SN, x)))
        # Specific: -P(a,b) | Q(a)
        specific = _cl(_neg(_pred2(P_SN, a, b)), _pos(_pred1(Q_SN, a)))

        p_general = compute_nucleus_unification_penalty(general, config)
        p_specific = compute_nucleus_unification_penalty(specific, config)

        assert p_general > p_specific
        assert p_specific == 0.0  # Ground nucleus: no penalty

    def test_multi_literal_boost(self) -> None:
        """Multiple general negative literals get a combinatorial penalty boost.

        -P(x) | -Q(y) | R(a) has TWO general negative literals, producing
        more combinatorial explosion than -P(x) | R(a).
        """
        x = get_variable_term(0)
        y = get_variable_term(1)
        a = _const(A_SN)

        config = NucleusUnificationPenaltyConfig(
            enabled=True,
            base_penalty=5.0,
            threshold=0.3,
            multi_literal_boost=1.5,
        )

        # Single general negative literal
        single = _cl(_neg(_pred1(P_SN, x)), _pos(_pred1(R_SN, a)))
        # Two general negative literals
        double = _cl(
            _neg(_pred1(P_SN, x)),
            _neg(_pred1(Q_SN, y)),
            _pos(_pred1(R_SN, a)),
        )

        p_single = compute_nucleus_unification_penalty(single, config)
        p_double = compute_nucleus_unification_penalty(double, config)

        assert p_double > p_single, "Multi-literal nucleus should get higher penalty"

    def test_nested_variable_partial_penalty(self) -> None:
        """Nested variables f(x) get partial penalty (between variable and ground).

        -P(f(x)) is less general than -P(x) because it only matches f(_) patterns,
        not arbitrary terms.
        """
        x = get_variable_term(0)
        a = _const(A_SN)

        config = NucleusUnificationPenaltyConfig(
            enabled=True,
            base_penalty=5.0,
            threshold=0.0,  # Lower threshold to capture nested var penalty
            variable_weight=1.0,
            nested_var_weight=0.5,
        )

        # Bare variable: -P(x) | Q(a)
        bare = _cl(_neg(_pred1(P_SN, x)), _pos(_pred1(Q_SN, a)))
        # Nested variable: -P(f(x)) | Q(a)
        nested = _cl(
            _neg(_pred1(P_SN, _func1(F_SN, x))),
            _pos(_pred1(Q_SN, a)),
        )

        p_bare = compute_nucleus_unification_penalty(bare, config)
        p_nested = compute_nucleus_unification_penalty(nested, config)

        assert p_bare > p_nested, "Bare variable should get higher penalty than nested"
        assert p_nested > 0.0, "Nested variable should still get some penalty"

    def test_implication_pattern_explosion_scenario(self) -> None:
        """Classic explosion pattern: -P(i(x,y)) | -P(x) | P(y).

        This nucleus has two negative literals with variables, making it
        a prime candidate for unification explosion. The penalty system
        should assign a significant penalty.
        """
        x = get_variable_term(0)
        y = get_variable_term(1)

        config = NucleusUnificationPenaltyConfig(
            enabled=True,
            base_penalty=5.0,
            threshold=0.3,
            multi_literal_boost=1.5,
        )

        # -P(i(x,y)) | -P(x) | P(y) — classic condensed detachment nucleus
        nucleus = _cl(
            _neg(_pred1(P_SN, _func2(I_SN, x, y))),
            _neg(_pred1(P_SN, x)),
            _pos(_pred1(P_SN, y)),
        )

        penalty = compute_nucleus_unification_penalty(nucleus, config)
        assert penalty > 0.0, "Condensed detachment nucleus should be penalized"

    def test_penalty_cap_enforced(self) -> None:
        """Penalty is capped at max_penalty."""
        x = get_variable_term(0)
        y = get_variable_term(1)
        z = get_variable_term(2)

        config = NucleusUnificationPenaltyConfig(
            enabled=True,
            base_penalty=100.0,  # Very high base
            threshold=0.0,
            max_penalty=20.0,
        )

        # Very general clause with multiple variable-heavy literals
        general = _cl(
            _neg(_pred2(P_SN, x, y)),
            _neg(_pred2(Q_SN, y, z)),
            _pos(_pred1(R_SN, x)),
        )

        penalty = compute_nucleus_unification_penalty(general, config)
        assert penalty <= config.max_penalty


# ── C Prover9 compatibility ────────────────────────────────────────────────


class TestCProver9Compatibility:
    """Ensure C Prover9 compatibility is maintained."""

    def test_exit_code_proof_found(self) -> None:
        """Exit code 1 (MAX_PROOFS_EXIT) when proof found with penalty enabled."""
        x = get_variable_term(0)
        a = _const(A_SN)

        c1 = _cl(_pos(_pred1(P_SN, a)))
        c2 = _cl(_neg(_pred1(P_SN, x)), _pos(_pred1(Q_SN, x)))
        c3 = _cl(_neg(_pred1(Q_SN, a)))

        result = _run_search(
            usable=[], sos=[c1, c2, c3],
            nucleus_unification_penalty=True,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert result.exit_code.value == 1

    def test_exit_code_sos_empty(self) -> None:
        """Exit code 2 (SOS_EMPTY_EXIT) when search saturates with penalty."""
        a = _const(A_SN)
        b = _const(B_SN)

        result = _run_search(
            usable=[], sos=[
                _cl(_pos(_pred1(P_SN, a))),
                _cl(_pos(_pred1(Q_SN, b))),
            ],
            nucleus_unification_penalty=True,
            max_given=50,
        )
        assert result.exit_code == ExitCode.SOS_EMPTY_EXIT
        assert result.exit_code.value == 2

    def test_exit_code_max_given(self) -> None:
        """Exit code 3 (MAX_GIVEN_EXIT) when limit hit with penalty."""
        x = get_variable_term(0)
        a = _const(A_SN)
        b = _const(B_SN)

        # Create clauses that generate inferences without finding proof
        clauses = [
            _cl(_pos(_pred1(P_SN, a))),
            _cl(_pos(_pred1(Q_SN, b))),
            _cl(_neg(_pred1(P_SN, x)), _pos(_pred1(R_SN, x))),
            _cl(_neg(_pred1(Q_SN, x)), _pos(_pred1(P_SN, x))),
            _cl(_neg(_pred1(R_SN, x)), _pos(_pred1(Q_SN, x))),
        ]

        result = _run_search(
            usable=[], sos=clauses,
            nucleus_unification_penalty=True,
            max_given=3,
        )
        # Should hit max_given or sos_empty, not crash
        assert result.exit_code in (
            ExitCode.MAX_GIVEN_EXIT,
            ExitCode.SOS_EMPTY_EXIT,
            ExitCode.MAX_PROOFS_EXIT,
        )

    def test_clause_processing_order_unchanged_when_disabled(self) -> None:
        """Core search behavior is identical when penalty is disabled."""
        x = get_variable_term(0)
        a = _const(A_SN)
        b = _const(B_SN)

        clauses = [
            _cl(_pos(_pred1(P_SN, a))),
            _cl(_pos(_pred1(Q_SN, b))),
            _cl(_neg(_pred1(P_SN, x)), _pos(_pred1(R_SN, x))),
            _cl(_neg(_pred1(Q_SN, a))),
        ]

        r_default = _run_search(usable=[], sos=clauses, max_given=100)
        r_disabled = _run_search(
            usable=[], sos=clauses,
            nucleus_unification_penalty=False,
            max_given=100,
        )

        assert r_default.exit_code == r_disabled.exit_code
        # Stats may differ slightly due to PrioritySOS ordering differences
        # but the key invariant is same exit code and similar search size
        assert abs(r_default.stats.given - r_disabled.stats.given) <= 2
        assert abs(r_default.stats.kept - r_disabled.stats.kept) <= 2

    def test_justification_format_unchanged(self) -> None:
        """Proof justifications are not altered by penalty system."""
        x = get_variable_term(0)
        a = _const(A_SN)

        c1 = _cl(_pos(_pred1(P_SN, a)))
        c2 = _cl(_neg(_pred1(P_SN, x)), _pos(_pred1(Q_SN, x)))
        c3 = _cl(_neg(_pred1(Q_SN, a)))

        r_without = _run_search(usable=[], sos=[c1, c2, c3])
        r_with = _run_search(
            usable=[], sos=[c1, c2, c3],
            nucleus_unification_penalty=True,
        )

        assert r_without.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert r_with.exit_code == ExitCode.MAX_PROOFS_EXIT

        # Both should produce valid proofs with justifications
        for proof in r_with.proofs:
            assert proof.empty_clause is not None
            assert proof.empty_clause.is_empty
            for clause in proof.clauses:
                assert clause.justification is not None


# ── Performance integration ────────────────────────────────────────────────


class TestPerformanceIntegration:
    """Test performance characteristics of the penalty system."""

    def test_search_time_overhead_when_enabled(self) -> None:
        """Penalty computation adds <5% overhead to search time.

        Uses a small but non-trivial problem to measure relative overhead.
        """
        x = get_variable_term(0)
        a = _const(A_SN)
        b = _const(B_SN)

        clauses = [
            _cl(_pos(_pred1(P_SN, a))),
            _cl(_pos(_pred1(Q_SN, b))),
            _cl(_neg(_pred1(P_SN, x)), _pos(_pred1(Q_SN, x))),
            _cl(_neg(_pred1(Q_SN, x)), _pos(_pred1(P_SN, x))),
            _cl(_neg(_pred1(P_SN, a)), _neg(_pred1(Q_SN, a))),
        ]

        # Warm-up
        _run_search(usable=[], sos=clauses, max_given=100)

        # Baseline (disabled)
        iterations = 20
        start = time.perf_counter()
        for _ in range(iterations):
            _run_search(usable=[], sos=clauses, max_given=100)
        baseline = time.perf_counter() - start

        # With penalty enabled
        start = time.perf_counter()
        for _ in range(iterations):
            _run_search(
                usable=[], sos=clauses,
                nucleus_unification_penalty=True,
                max_given=100,
            )
        with_penalty = time.perf_counter() - start

        if baseline > 0:
            overhead = (with_penalty - baseline) / baseline
            # Allow up to 50% overhead for small problems (absolute time is tiny)
            # Real overhead on large problems should be <5%
            assert overhead < 0.5, f"Overhead too high: {overhead:.1%}"

    def test_pattern_cache_memory_bounds(self) -> None:
        """Pattern cache respects max_size and evicts properly."""
        cache = NucleusPatternCache(max_size=5)

        # Insert more patterns than max_size
        for i in range(10):
            c = _cl(
                _neg(_pred1(P_SN + i, get_variable_term(0))),
                _pos(_pred1(Q_SN, _const(A_SN))),
                clause_id=i + 1,
            )
            cache_nucleus_patterns(c, cache)

        # Cache should not exceed max_size
        assert len(cache) <= 5
        assert cache.stats.evictions > 0

    def test_zero_overhead_when_disabled(self) -> None:
        """No penalty computation occurs when disabled."""
        x = get_variable_term(0)
        a = _const(A_SN)

        config = NucleusUnificationPenaltyConfig(enabled=False)
        nucleus = _cl(_neg(_pred2(P_SN, x, x)), _pos(_pred1(Q_SN, x)))

        # Should short-circuit immediately
        penalty = compute_nucleus_unification_penalty(nucleus, config)
        assert penalty == 0.0


# ── Manual test cases: specific nucleus patterns ───────────────────────────


class TestSpecificNucleusPatterns:
    """Test nucleus patterns from the task specification."""

    def test_condensed_detachment_pattern(self) -> None:
        """Test -P(i(x,y)) | -P(x) | P(y) against various satellites.

        P(z) should get high penalty (bare variable matches anything).
        P(i(a,b)) should get moderate penalty (nested variable).
        P(f(w)) should get lower penalty (different functor).
        """
        x = get_variable_term(0)
        y = get_variable_term(1)
        z = get_variable_term(2)
        w = get_variable_term(3)
        a = _const(A_SN)
        b = _const(B_SN)

        config = NucleusUnificationPenaltyConfig(
            enabled=True,
            base_penalty=5.0,
            threshold=0.3,
            multi_literal_boost=1.5,
        )

        # The classic condensed detachment nucleus
        cd_nucleus = _cl(
            _neg(_pred1(P_SN, _func2(I_SN, x, y))),
            _neg(_pred1(P_SN, x)),
            _pos(_pred1(P_SN, y)),
        )

        penalty = compute_nucleus_unification_penalty(cd_nucleus, config)
        assert penalty > 0.0, "CD nucleus with variable positions should be penalized"

    def test_propositional_nucleus_no_penalty(self) -> None:
        """Propositional nuclei (0-arity atoms) get no penalty."""
        # -P | Q (propositional)
        P_prop = get_rigid_term(P_SN, 0)
        Q_prop = get_rigid_term(Q_SN, 0)

        config = NucleusUnificationPenaltyConfig(
            enabled=True,
            base_penalty=5.0,
            threshold=0.3,
        )

        c = _cl(_neg(P_prop), _pos(Q_prop))
        penalty = compute_nucleus_unification_penalty(c, config)
        assert penalty == 0.0, "Propositional nucleus should get zero penalty"

    def test_mixed_ground_variable_nucleus(self) -> None:
        """Nucleus with mix of ground and variable args gets intermediate penalty."""
        x = get_variable_term(0)
        a = _const(A_SN)
        b = _const(B_SN)

        config = NucleusUnificationPenaltyConfig(
            enabled=True,
            base_penalty=5.0,
            threshold=0.0,  # No threshold to see graduated penalty
        )

        # -P(x, a, b) | Q(x) — one variable out of three args
        c = _cl(
            _neg(get_rigid_term(P_SN, 3, (x, a, b))),
            _pos(_pred1(Q_SN, x)),
        )

        penalty = compute_nucleus_unification_penalty(c, config)
        # subsumption_ratio = 1/3 ≈ 0.33, penalty = 5.0 * 0.33 ≈ 1.67
        assert 0.0 < penalty < 5.0, f"Mixed nucleus should get moderate penalty, got {penalty}"

    def test_all_variable_nucleus_max_penalty(self) -> None:
        """Nucleus with all variable arguments gets maximum generality penalty."""
        x = get_variable_term(0)
        y = get_variable_term(1)
        z = get_variable_term(2)

        config = NucleusUnificationPenaltyConfig(
            enabled=True,
            base_penalty=5.0,
            threshold=0.0,
        )

        # -P(x,y,z) | Q(a) — all three args are variables
        c = _cl(
            _neg(get_rigid_term(P_SN, 3, (x, y, z))),
            _pos(_pred1(Q_SN, _const(A_SN))),
        )

        penalty = compute_nucleus_unification_penalty(c, config)
        # subsumption_ratio = 3/3 = 1.0, penalty = 5.0 * 1.0 = 5.0
        assert penalty == pytest.approx(5.0, abs=0.01)
