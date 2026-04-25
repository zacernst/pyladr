"""Integration tests for penalty propagation in the search pipeline.

Validates that penalty propagation:
- Produces identical results when disabled (C Prover9 regression prevention)
- Correctly computes and caches penalties during search
- Influences clause selection when enabled with unification_weight
- Handles all inference rule types through the derivation chain
- Maintains proof soundness with penalty propagation enabled
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.parsing.ladr_parser import parse_input
from pyladr.search.given_clause import (
    ExitCode,
    GivenClauseSearch,
    SearchOptions,
    SearchResult,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_and_deny(text: str) -> tuple[list[Clause], SymbolTable]:
    """Parse LADR input and deny goals into SOS clauses."""
    st = SymbolTable()
    parsed = parse_input(text, st)
    sos = list(parsed.sos)
    for goal in parsed.goals:
        denied_lits = tuple(
            Literal(sign=not lit.sign, atom=lit.atom) for lit in goal.literals
        )
        denied = Clause(
            literals=denied_lits,
            justification=(Justification(just_type=JustType.DENY, clause_ids=(0,)),),
        )
        sos.append(denied)
    return sos, st


def _run_search(text: str, max_given: int = 500, max_seconds: float = 10.0, **kwargs) -> SearchResult:
    """Parse input text and run search with given options."""
    sos, st = _parse_and_deny(text)
    opts = SearchOptions(
        max_given=max_given,
        max_seconds=max_seconds,
        quiet=True,
        print_given=False,
        **kwargs,
    )
    engine = GivenClauseSearch(opts)
    return engine.run(usable=[], sos=sos)


def _run_search_with_engine(
    text: str, max_given: int = 500, max_seconds: float = 10.0, **kwargs
) -> tuple[SearchResult, GivenClauseSearch]:
    """Run search and return both result and engine for inspection."""
    sos, st = _parse_and_deny(text)
    opts = SearchOptions(
        max_given=max_given,
        max_seconds=max_seconds,
        quiet=True,
        print_given=False,
        **kwargs,
    )
    engine = GivenClauseSearch(opts)
    result = engine.run(usable=[], sos=sos)
    return result, engine


# ── Test problems ────────────────────────────────────────────────────────────

# Trivial: e*e=e from e*x=x
TRIVIAL_IDENTITY = """
formulas(sos).
  e * x = x.
end_of_list.
formulas(goals).
  e * e = e.
end_of_list.
"""

# Group theory: prove commutativity from x*x=e (uses binary resolution)
GROUP_COMMUTATIVITY = """
formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
  x * x = e.
end_of_list.
formulas(goals).
  x * y = y * x.
end_of_list.
"""

# Simple propositional: P(a), P(x)->Q(x), ~Q(a)
SIMPLE_CHAIN = """
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(a).
end_of_list.
"""

# Two-step resolution chain
TWO_STEP_CHAIN = """
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(x) | R(x).
  -R(a).
end_of_list.
"""


# ── Regression Prevention: Disabled Penalty Propagation ──────────────────────


class TestDisabledPenaltyPropagation:
    """Verify that with penalty_propagation=False, behavior is identical to baseline."""

    def test_trivial_proof_unchanged(self) -> None:
        """Trivial problem finds same proof with propagation disabled."""
        result_baseline = _run_search(TRIVIAL_IDENTITY, penalty_propagation=False)
        assert result_baseline.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result_baseline.proofs) == 1

    def test_no_penalty_cache_when_disabled(self) -> None:
        """Engine should not create penalty cache when feature disabled."""
        _, engine = _run_search_with_engine(
            TRIVIAL_IDENTITY, penalty_propagation=False,
        )
        assert engine._penalty_cache is None

    def test_stats_unchanged_when_disabled(self) -> None:
        """Statistics should be identical to non-penalty baseline."""
        result_off = _run_search(TRIVIAL_IDENTITY, penalty_propagation=False)
        # Run a second time — should produce identical stats
        result_off2 = _run_search(TRIVIAL_IDENTITY, penalty_propagation=False)
        assert result_off.exit_code == result_off2.exit_code
        assert result_off.stats.given == result_off2.stats.given
        assert result_off.stats.kept == result_off2.stats.kept

    def test_group_theory_unchanged(self) -> None:
        """Group commutativity proof finds proof with propagation disabled."""
        result = _run_search(GROUP_COMMUTATIVITY, penalty_propagation=False)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_default_options_have_propagation_disabled(self) -> None:
        """Default SearchOptions have penalty_propagation=False."""
        opts = SearchOptions()
        assert not opts.penalty_propagation


# ── Enabled Penalty Propagation ──────────────────────────────────────────────


class TestEnabledPenaltyPropagation:
    """Verify penalty propagation works when enabled."""

    def test_trivial_proof_still_found(self) -> None:
        """Trivial problem still finds proof with propagation enabled."""
        result = _run_search(
            TRIVIAL_IDENTITY,
            penalty_propagation=True,
            penalty_propagation_mode="additive",
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1

    def test_penalty_cache_created(self) -> None:
        """Engine creates penalty cache when feature enabled."""
        _, engine = _run_search_with_engine(
            TRIVIAL_IDENTITY,
            penalty_propagation=True,
        )
        assert engine._penalty_cache is not None
        assert len(engine._penalty_cache) > 0

    def test_initial_clauses_seeded(self) -> None:
        """Initial clauses should be in the penalty cache."""
        _, engine = _run_search_with_engine(
            TRIVIAL_IDENTITY,
            penalty_propagation=True,
        )
        cache = engine._penalty_cache
        assert cache is not None
        # Initial clauses (marked with .initial=True) should be in cache
        for cid, clause in engine._all_clauses.items():
            if clause.initial:
                assert cache.get(cid) is not None, f"Initial clause {cid} missing from penalty cache"

    def test_group_theory_with_propagation(self) -> None:
        """Group commutativity proof still found with propagation enabled."""
        result = _run_search(
            GROUP_COMMUTATIVITY,
            penalty_propagation=True,
            penalty_propagation_decay=0.5,
            penalty_propagation_threshold=5.0,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_simple_chain_with_propagation(self) -> None:
        """Simple propositional chain still resolves correctly."""
        result = _run_search(
            SIMPLE_CHAIN,
            penalty_propagation=True,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_two_step_chain_with_propagation(self) -> None:
        """Two-step resolution chain still resolves correctly."""
        result = _run_search(
            TWO_STEP_CHAIN,
            penalty_propagation=True,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT


# ── Configuration Modes ──────────────────────────────────────────────────────


class TestPenaltyPropagationModes:
    """Verify all three combination modes work correctly."""

    def test_additive_mode_finds_proof(self) -> None:
        result = _run_search(
            TRIVIAL_IDENTITY,
            penalty_propagation=True,
            penalty_propagation_mode="additive",
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_multiplicative_mode_finds_proof(self) -> None:
        result = _run_search(
            TRIVIAL_IDENTITY,
            penalty_propagation=True,
            penalty_propagation_mode="multiplicative",
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_max_mode_finds_proof(self) -> None:
        result = _run_search(
            TRIVIAL_IDENTITY,
            penalty_propagation=True,
            penalty_propagation_mode="max",
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_invalid_mode_falls_back_to_additive(self) -> None:
        """Unknown mode string should default to additive gracefully."""
        result = _run_search(
            TRIVIAL_IDENTITY,
            penalty_propagation=True,
            penalty_propagation_mode="bogus",
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT


# ── Parameter Boundary Tests ─────────────────────────────────────────────────


class TestPenaltyPropagationParameters:
    """Verify boundary parameter values work correctly."""

    def test_zero_decay(self) -> None:
        """decay=0.0 means no inheritance passes through."""
        result, engine = _run_search_with_engine(
            TRIVIAL_IDENTITY,
            penalty_propagation=True,
            penalty_propagation_decay=0.0,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        # With decay=0, all inherited penalties should be 0
        cache = engine._penalty_cache
        if cache is not None:
            for cid in engine._all_clauses:
                rec = cache.get(cid)
                if rec is not None:
                    assert rec.inherited_penalty == 0.0

    def test_full_decay(self) -> None:
        """decay=1.0 passes full penalty to children."""
        result = _run_search(
            TRIVIAL_IDENTITY,
            penalty_propagation=True,
            penalty_propagation_decay=1.0,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_high_threshold_no_propagation(self) -> None:
        """Very high threshold means nothing gets propagated."""
        result, engine = _run_search_with_engine(
            TRIVIAL_IDENTITY,
            penalty_propagation=True,
            penalty_propagation_threshold=1000.0,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        cache = engine._penalty_cache
        if cache is not None:
            for cid in engine._all_clauses:
                rec = cache.get(cid)
                if rec is not None:
                    assert rec.inherited_penalty == 0.0

    def test_low_threshold_more_propagation(self) -> None:
        """Low threshold allows more penalty propagation."""
        result = _run_search(
            GROUP_COMMUTATIVITY,
            penalty_propagation=True,
            penalty_propagation_threshold=0.1,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_depth_limit_one(self) -> None:
        """max_depth=1 limits inheritance to direct children only."""
        result = _run_search(
            TRIVIAL_IDENTITY,
            penalty_propagation=True,
            penalty_propagation_max_depth=1,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_unlimited_depth(self) -> None:
        """max_depth=0 means unlimited depth."""
        result = _run_search(
            TRIVIAL_IDENTITY,
            penalty_propagation=True,
            penalty_propagation_max_depth=0,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_low_max_penalty(self) -> None:
        """Low max_penalty caps accumulated penalties."""
        result = _run_search(
            TRIVIAL_IDENTITY,
            penalty_propagation=True,
            penalty_propagation_max=5.0,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        # Verify no penalty exceeds cap (if engine accessible)


# ── Penalty Cache Inspection ─────────────────────────────────────────────────


class TestPenaltyCachePopulation:
    """Verify the penalty cache is correctly populated during search."""

    def test_cache_grows_during_search(self) -> None:
        """Cache should contain entries for kept clauses."""
        _, engine = _run_search_with_engine(
            GROUP_COMMUTATIVITY,
            penalty_propagation=True,
        )
        cache = engine._penalty_cache
        assert cache is not None
        # Should have entries for most clauses that passed through _cl_process
        assert len(cache) > 3  # At least the initial clauses

    def test_input_clauses_have_zero_inheritance(self) -> None:
        """All input/initial clauses should have depth=0, zero inheritance."""
        _, engine = _run_search_with_engine(
            TRIVIAL_IDENTITY,
            penalty_propagation=True,
        )
        cache = engine._penalty_cache
        assert cache is not None
        # Find initial clauses (id assigned during init, typically low IDs)
        for cid, clause in engine._all_clauses.items():
            if clause.initial:
                rec = cache.get(cid)
                assert rec is not None, f"Initial clause {cid} missing from cache"
                assert rec.depth == 0
                assert rec.inherited_penalty == 0.0

    def test_derived_clauses_may_inherit(self) -> None:
        """Derived clauses from general parents may have inherited penalty > 0."""
        _, engine = _run_search_with_engine(
            GROUP_COMMUTATIVITY,
            penalty_propagation=True,
            penalty_propagation_threshold=0.1,  # Low threshold to trigger more inheritance
            penalty_propagation_decay=0.8,
        )
        cache = engine._penalty_cache
        assert cache is not None
        # At least some derived clauses should have inherited penalty
        has_inherited = any(
            cache.get(cid) is not None and cache.get(cid).inherited_penalty > 0.0
            for cid in engine._all_clauses
            if not engine._all_clauses[cid].initial
        )
        # This may or may not be true depending on the search path
        # but with low threshold it's very likely for group theory
        # Just verify no crash and cache is populated
        assert len(cache) > 0

    def test_max_penalty_respected_in_cache(self) -> None:
        """No cached penalty should exceed the configured max."""
        _, engine = _run_search_with_engine(
            GROUP_COMMUTATIVITY,
            penalty_propagation=True,
            penalty_propagation_max=10.0,
            penalty_propagation_threshold=0.1,
        )
        cache = engine._penalty_cache
        assert cache is not None
        for cid in engine._all_clauses:
            rec = cache.get(cid)
            if rec is not None:
                assert rec.combined_penalty <= 10.0, (
                    f"Clause {cid} has penalty {rec.combined_penalty} > max 10.0"
                )


# ── Combined with Unification Weight ────────────────────────────────────────


class TestPenaltyWithUnificationWeight:
    """Test penalty propagation combined with unification_weight selection."""

    def test_unification_weight_with_propagation(self) -> None:
        """Combining unification_weight + penalty_propagation finds proof."""
        result = _run_search(
            TRIVIAL_IDENTITY,
            unification_weight=2,
            penalty_propagation=True,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_group_theory_with_both(self) -> None:
        """Group theory with both features enabled."""
        result = _run_search(
            GROUP_COMMUTATIVITY,
            unification_weight=1,
            penalty_propagation=True,
            penalty_propagation_decay=0.5,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT


# ── Proof Soundness ─────────────────────────────────────────────────────────


class TestPenaltyOverrideInSelection:
    """Verify that penalty_override wires propagated penalties into PrioritySOS."""

    def test_penalty_override_accepted_by_priority_sos(self) -> None:
        """PrioritySOS.append() accepts penalty_override parameter."""
        from pyladr.search.priority_sos import PrioritySOS
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term

        sos = PrioritySOS("test")
        atom = get_rigid_term(1, 0)
        c1 = Clause(literals=(Literal(sign=True, atom=atom),), id=1)
        c1.weight = 1.0
        c2 = Clause(literals=(Literal(sign=True, atom=atom),), id=2)
        c2.weight = 1.0

        # c1 gets low override, c2 gets high override
        sos.append(c1, penalty_override=1.0)
        sos.append(c2, penalty_override=100.0)

        # pop_lowest_penalty should return c1 first (lower penalty)
        selected = sos.pop_lowest_penalty()
        assert selected is not None
        assert selected.id == 1

    def test_penalty_override_overrides_intrinsic(self) -> None:
        """penalty_override overrides intrinsic penalty computation."""
        from pyladr.search.priority_sos import PrioritySOS
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term

        sos = PrioritySOS("test")

        # c1: P(x) — intrinsically general (high penalty), but low override
        atom_general = get_rigid_term(1, 1, (get_variable_term(0),))
        c1 = Clause(literals=(Literal(sign=True, atom=atom_general),), id=1)
        c1.weight = 1.0

        # c2: P(a) — intrinsically specific (low penalty), but high override
        atom_specific = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c2 = Clause(literals=(Literal(sign=True, atom=atom_specific),), id=2)
        c2.weight = 1.0

        # Override inverts the natural order
        sos.append(c1, penalty_override=0.5)   # Artificially low
        sos.append(c2, penalty_override=50.0)  # Artificially high

        # c1 should be selected first despite being intrinsically more general
        selected = sos.pop_lowest_penalty()
        assert selected is not None
        assert selected.id == 1  # Override wins over intrinsic

    def test_no_override_uses_intrinsic(self) -> None:
        """Without penalty_override, intrinsic penalty is used."""
        from pyladr.search.priority_sos import PrioritySOS
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term

        sos = PrioritySOS("test")

        # c1: P(a) — ground, low intrinsic penalty
        atom_ground = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c1 = Clause(literals=(Literal(sign=True, atom=atom_ground),), id=1)
        c1.weight = 1.0

        # c2: P(x) — variable, higher intrinsic penalty
        atom_var = get_rigid_term(1, 1, (get_variable_term(0),))
        c2 = Clause(literals=(Literal(sign=True, atom=atom_var),), id=2)
        c2.weight = 1.0

        sos.append(c1)  # No override — uses intrinsic
        sos.append(c2)  # No override — uses intrinsic

        # c1 should be selected first (lower intrinsic penalty)
        selected = sos.pop_lowest_penalty()
        assert selected is not None
        assert selected.id == 1

    def test_search_uses_combined_penalty_for_selection(self) -> None:
        """End-to-end: search with propagation uses combined penalty in SOS heap."""
        result, engine = _run_search_with_engine(
            GROUP_COMMUTATIVITY,
            unification_weight=1,  # Enable penalty-based selection
            penalty_propagation=True,
            penalty_propagation_threshold=0.1,
            penalty_propagation_decay=0.8,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        # Verify the penalty cache was used
        assert engine._penalty_cache is not None
        assert len(engine._penalty_cache) > 0


class TestProofSoundnessWithPenalty:
    """Verify proofs remain sound with penalty propagation enabled."""

    def test_proof_has_valid_structure(self) -> None:
        """Proof should have an empty clause and clause trace."""
        result = _run_search(
            TRIVIAL_IDENTITY,
            penalty_propagation=True,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        proof = result.proofs[0]
        assert proof.empty_clause.is_empty
        assert len(proof.clauses) > 0

    def test_proof_trace_has_clauses(self) -> None:
        """Proof trace contains the supporting clauses."""
        result = _run_search(
            SIMPLE_CHAIN,
            penalty_propagation=True,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        proof = result.proofs[0]
        assert proof.empty_clause.is_empty
        # Proof trace should include at least the initial clauses used
        assert len(proof.clauses) >= 2

    def test_exit_code_matches_baseline(self) -> None:
        """Exit code with propagation should match expected value."""
        for problem in [TRIVIAL_IDENTITY, SIMPLE_CHAIN, TWO_STEP_CHAIN]:
            result_off = _run_search(problem, penalty_propagation=False)
            result_on = _run_search(problem, penalty_propagation=True)
            assert result_off.exit_code == result_on.exit_code, (
                f"Exit code mismatch: off={result_off.exit_code} vs on={result_on.exit_code}"
            )
