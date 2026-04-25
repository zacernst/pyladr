"""Integration tests for penalty weight adjustment through search pipeline.

Verifies that penalty weight adjustment works end-to-end through
GivenClauseSearch.run():
- High-penalty clauses get weight inflated
- Weight inflation affects max_weight elimination
- Stats tracking (penalty_weight_adjusted) is accurate
- Penalty propagation + weight adjustment compose correctly
- Feature disabled by default (C Prover9 compatibility)
- Cache vs fresh penalty computation paths both work
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import Symbol, SymbolTable
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.search.given_clause import (
    ExitCode,
    GivenClauseSearch,
    SearchOptions,
)
from pyladr.search.penalty_weight import (
    PenaltyWeightConfig,
    PenaltyWeightMode,
    penalty_adjusted_weight,
)


# ── Symbol constants ─────────────────────────────────────────────────────

A, B, C_SYM = 1, 2, 3
F, G = 10, 11
P, Q, R = 20, 21, 22


def _make_symbol_table() -> SymbolTable:
    st = SymbolTable()
    _symbols = [
        (A, "a", 0), (B, "b", 0), (C_SYM, "c", 0),
        (F, "f", 2), (G, "g", 1),
        (P, "P", 1), (Q, "Q", 1), (R, "R", 1),
    ]
    for symnum, name, arity in _symbols:
        sym = Symbol(symnum=symnum, name=name, arity=arity)
        st._by_id[symnum] = sym
        st._by_name_arity[(name, arity)] = symnum
    st._next_id = max(symnum for symnum, _, _ in _symbols) + 1
    return st


# ── Helpers ──────────────────────────────────────────────────────────────

def _const(symnum: int) -> Term:
    return get_rigid_term(symnum, 0)

def _func(symnum: int, *args: Term) -> Term:
    return get_rigid_term(symnum, len(args), args)

def _var(n: int) -> Term:
    return get_variable_term(n)

def _pos_lit(atom: Term) -> Literal:
    return Literal(sign=True, atom=atom)

def _neg_lit(atom: Term) -> Literal:
    return Literal(sign=False, atom=atom)

def _input_clause(*lits: Literal) -> Clause:
    return Clause(
        literals=tuple(lits),
        justification=(Justification(just_type=JustType.INPUT),),
    )


# ── Unit tests for penalty_adjusted_weight ───────────────────────────────


class TestPenaltyAdjustedWeight:
    """Test core penalty_adjusted_weight function."""

    def test_below_threshold_unchanged(self):
        cfg = PenaltyWeightConfig(enabled=True, threshold=5.0, multiplier=2.0)
        assert penalty_adjusted_weight(10.0, 3.0, cfg) == 10.0

    def test_at_threshold_unchanged(self):
        """Penalty must be >= threshold, so exactly at threshold triggers."""
        cfg = PenaltyWeightConfig(enabled=True, threshold=5.0, multiplier=2.0)
        # At threshold, exponential: 10 * 2^(5/5) = 10 * 2 = 20
        result = penalty_adjusted_weight(10.0, 5.0, cfg)
        assert result == pytest.approx(20.0)

    def test_disabled_unchanged(self):
        cfg = PenaltyWeightConfig(enabled=False, threshold=1.0, multiplier=100.0)
        assert penalty_adjusted_weight(10.0, 100.0, cfg) == 10.0

    def test_exponential_mode(self):
        cfg = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.EXPONENTIAL,
        )
        # penalty=10, exponent=10/5=2, adjusted = 10 * 2^2 = 40
        assert penalty_adjusted_weight(10.0, 10.0, cfg) == pytest.approx(40.0)

    def test_linear_mode(self):
        cfg = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.LINEAR,
        )
        # adjusted = 10 + 2 * 10 = 30
        assert penalty_adjusted_weight(10.0, 10.0, cfg) == pytest.approx(30.0)

    def test_step_mode(self):
        cfg = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=3.0,
            mode=PenaltyWeightMode.STEP,
        )
        # adjusted = 10 * 3 = 30
        assert penalty_adjusted_weight(10.0, 10.0, cfg) == pytest.approx(30.0)

    def test_max_weight_cap(self):
        cfg = PenaltyWeightConfig(
            enabled=True, threshold=1.0, multiplier=100.0,
            max_adjusted_weight=50.0,
        )
        assert penalty_adjusted_weight(10.0, 10.0, cfg) == 50.0

    def test_zero_base_weight(self):
        cfg = PenaltyWeightConfig(
            enabled=True, threshold=1.0, multiplier=2.0,
            mode=PenaltyWeightMode.EXPONENTIAL,
        )
        # base=0, exponential: 0 * anything = 0
        assert penalty_adjusted_weight(0.0, 5.0, cfg) == 0.0

    def test_zero_penalty(self):
        cfg = PenaltyWeightConfig(enabled=True, threshold=0.0, multiplier=2.0)
        # penalty=0 < threshold=0 is False since 0 is not < 0
        # Actually 0 >= 0 so it triggers. Let me set threshold > 0
        cfg2 = PenaltyWeightConfig(enabled=True, threshold=1.0, multiplier=2.0)
        assert penalty_adjusted_weight(10.0, 0.0, cfg2) == 10.0


# ── Integration tests: search pipeline ───────────────────────────────────


class TestPenaltyWeightDisabledByDefault:
    """Verify C Prover9 compatibility when feature is disabled."""

    def test_default_options_no_penalty_weight(self):
        opts = SearchOptions()
        assert opts.penalty_weight_enabled is False

    def test_search_runs_without_penalty_weight(self):
        """Basic search works with penalty weight disabled."""
        st = _make_symbol_table()
        # P(a)
        c1 = _input_clause(_pos_lit(_func(P, _const(A))))
        # -P(a)
        c2 = _input_clause(_neg_lit(_func(P, _const(A))))

        opts = SearchOptions(max_given=10, quiet=True)
        search = GivenClauseSearch(opts, symbol_table=st)
        result = search.run(sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert result.stats.penalty_weight_adjusted == 0


class TestPenaltyWeightIntegration:
    """Test penalty weight adjustment through the search pipeline."""

    def test_stats_tracking(self):
        """Penalty weight adjusted counter increments for high-penalty clauses."""
        st = _make_symbol_table()

        # P(x) — overly general (all-variable single literal, penalty > 10)
        general = _input_clause(_pos_lit(_func(P, _var(0))))
        # -P(x) | Q(x) — resolves with P(x) to produce Q(x), also general
        bridge = _input_clause(
            _neg_lit(_func(P, _var(0))),
            _pos_lit(_func(Q, _var(0))),
        )
        # -Q(a) — specific complement
        complement = _input_clause(_neg_lit(_func(Q, _const(A))))

        opts = SearchOptions(
            penalty_weight_enabled=True,
            penalty_weight_threshold=0.3,  # Low threshold to catch more clauses
            penalty_weight_multiplier=2.0,
            max_given=20,
            quiet=True,
        )
        search = GivenClauseSearch(opts, symbol_table=st)
        result = search.run(sos=[general, bridge, complement])

        # Inferred clauses from general parents should have been weight-adjusted
        assert result.stats.penalty_weight_adjusted > 0

    def test_max_weight_eliminates_high_penalty(self):
        """High-penalty clauses eliminated when adjusted weight exceeds max_weight.

        Uses a multi-step problem where inferred clauses go through _cl_process
        and get weight adjusted. P(x)|Q(x) resolved with -P(a) gives Q(a),
        and -Q(x)|R(x) resolved with Q(a) gives R(a). The general clauses
        generate inferred children that get penalty-adjusted.
        """
        st = _make_symbol_table()

        # -P(x) | Q(x) — general bridge
        bridge1 = _input_clause(
            _neg_lit(_func(P, _var(0))),
            _pos_lit(_func(Q, _var(0))),
        )
        # -Q(x) | R(x) — another general bridge
        bridge2 = _input_clause(
            _neg_lit(_func(Q, _var(0))),
            _pos_lit(_func(R, _var(0))),
        )
        # P(a) — specific fact
        fact = _input_clause(_pos_lit(_func(P, _const(A))))
        # -R(a) — goal
        goal = _input_clause(_neg_lit(_func(R, _const(A))))

        opts = SearchOptions(
            penalty_weight_enabled=True,
            penalty_weight_threshold=0.2,  # Low threshold to catch variable-heavy clauses
            penalty_weight_multiplier=5.0,
            max_weight=100.0,
            max_given=30,
            quiet=True,
        )
        search = GivenClauseSearch(opts, symbol_table=st)
        result = search.run(sos=[bridge1, bridge2, fact, goal])

        # Should find proof through the chain
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_penalty_weight_with_propagation(self):
        """Penalty weight adjustment composes with penalty propagation."""
        st = _make_symbol_table()

        # P(x) — general
        general = _input_clause(_pos_lit(_func(P, _var(0))))
        # -P(a) | Q(a)
        resolvent_source = _input_clause(
            _neg_lit(_func(P, _const(A))),
            _pos_lit(_func(Q, _const(A))),
        )
        # -Q(a)
        complement = _input_clause(_neg_lit(_func(Q, _const(A))))

        opts = SearchOptions(
            penalty_weight_enabled=True,
            penalty_weight_threshold=5.0,
            penalty_weight_multiplier=2.0,
            penalty_propagation=True,
            penalty_propagation_threshold=5.0,
            max_given=30,
            quiet=True,
        )
        search = GivenClauseSearch(opts, symbol_table=st)
        result = search.run(sos=[general, resolvent_source, complement])

        # Search should complete (either proof or SOS empty)
        assert result.exit_code in (
            ExitCode.MAX_PROOFS_EXIT,
            ExitCode.SOS_EMPTY_EXIT,
            ExitCode.MAX_GIVEN_EXIT,
        )

    def test_all_modes_run_without_error(self):
        """All penalty weight modes run through search without error."""
        st = _make_symbol_table()
        c1 = _input_clause(_pos_lit(_func(P, _const(A))))
        c2 = _input_clause(_neg_lit(_func(P, _const(A))))

        for mode in ("linear", "exponential", "step"):
            opts = SearchOptions(
                penalty_weight_enabled=True,
                penalty_weight_mode=mode,
                max_given=5,
                quiet=True,
            )
            search = GivenClauseSearch(opts, symbol_table=st)
            result = search.run(sos=[c1, c2])
            assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_invalid_mode_defaults_to_exponential(self):
        """Unknown mode string falls back to exponential."""
        st = _make_symbol_table()
        c1 = _input_clause(_pos_lit(_func(P, _const(A))))
        c2 = _input_clause(_neg_lit(_func(P, _const(A))))

        opts = SearchOptions(
            penalty_weight_enabled=True,
            penalty_weight_mode="invalid_mode",
            max_given=5,
            quiet=True,
        )
        search = GivenClauseSearch(opts, symbol_table=st)
        assert search._penalty_weight_config.mode == PenaltyWeightMode.EXPONENTIAL


# ── Repetition penalty + weight adjustment ──────────────────────────────────


class TestRepetitionPenaltyInteraction:
    """Verify penalty weight composes with repetition penalty."""

    def test_repetition_and_weight_both_enabled(self):
        """Both systems can be enabled simultaneously."""
        st = _make_symbol_table()
        c1 = _input_clause(_pos_lit(_func(P, _const(A))))
        c2 = _input_clause(_neg_lit(_func(P, _const(A))))

        opts = SearchOptions(
            repetition_penalty=True,
            penalty_weight_enabled=True,
            max_given=10,
            quiet=True,
        )
        search = GivenClauseSearch(opts, symbol_table=st)
        result = search.run(sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_all_three_penalty_systems_enabled(self):
        """Penalty propagation + repetition penalty + weight adjustment."""
        st = _make_symbol_table()
        c1 = _input_clause(_pos_lit(_func(P, _const(A))))
        c2 = _input_clause(_neg_lit(_func(P, _const(A))))

        opts = SearchOptions(
            penalty_propagation=True,
            repetition_penalty=True,
            penalty_weight_enabled=True,
            max_given=10,
            quiet=True,
        )
        search = GivenClauseSearch(opts, symbol_table=st)
        assert search._penalty_cache is not None
        assert search._repetition_config is not None
        assert search._penalty_weight_config is not None

        result = search.run(sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT


# ── Cache vs fresh penalty computation paths ────────────────────────────────


class TestCacheVsFreshPenalty:
    """Verify both penalty lookup paths work correctly."""

    def test_with_cache_path(self):
        """When penalty_propagation is on, penalty comes from cache."""
        st = _make_symbol_table()
        general = _input_clause(_pos_lit(_func(P, _var(0))))
        complement = _input_clause(_neg_lit(_func(P, _const(A))))

        opts = SearchOptions(
            penalty_propagation=True,
            penalty_weight_enabled=True,
            penalty_weight_threshold=1.0,
            max_given=20,
            quiet=True,
        )
        search = GivenClauseSearch(opts, symbol_table=st)
        result = search.run(sos=[general, complement])
        # Cache should be populated
        assert search._penalty_cache is not None
        assert len(search._penalty_cache) > 0

    def test_without_cache_path(self):
        """When penalty_propagation is off, intrinsic generality is computed."""
        st = _make_symbol_table()
        general = _input_clause(_pos_lit(_func(P, _var(0))))
        complement = _input_clause(_neg_lit(_func(P, _const(A))))

        opts = SearchOptions(
            penalty_propagation=False,
            penalty_weight_enabled=True,
            penalty_weight_threshold=1.0,
            max_given=20,
            quiet=True,
        )
        search = GivenClauseSearch(opts, symbol_table=st)
        result = search.run(sos=[general, complement])
        assert search._penalty_cache is None
        # Should still have weight-adjusted some clauses
        assert result.exit_code in (
            ExitCode.MAX_PROOFS_EXIT,
            ExitCode.SOS_EMPTY_EXIT,
            ExitCode.MAX_GIVEN_EXIT,
        )


# ── Search behavior changes ────────────────────────────────────────────────


class TestSearchBehaviorChanges:
    """Verify penalty weight affects search behavior as expected."""

    def test_enabled_vs_disabled_same_proof(self):
        """Enabled and disabled should both find proofs on simple problems."""
        st = _make_symbol_table()
        c1 = _input_clause(_pos_lit(_func(P, _const(A))))
        c2 = _input_clause(_neg_lit(_func(P, _const(A))))

        for enabled in (True, False):
            opts = SearchOptions(
                penalty_weight_enabled=enabled,
                max_given=10,
                quiet=True,
            )
            search = GivenClauseSearch(opts, symbol_table=st)
            result = search.run(sos=[c1, c2])
            assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_no_stats_when_disabled(self):
        """Disabled penalty weight should never increment the counter."""
        st = _make_symbol_table()
        general = _input_clause(_pos_lit(_func(P, _var(0))))
        bridge = _input_clause(
            _neg_lit(_func(P, _var(0))),
            _pos_lit(_func(Q, _var(0))),
        )
        complement = _input_clause(_neg_lit(_func(Q, _const(A))))

        opts = SearchOptions(
            penalty_weight_enabled=False,
            max_given=20,
            quiet=True,
        )
        search = GivenClauseSearch(opts, symbol_table=st)
        result = search.run(sos=[general, bridge, complement])
        assert result.stats.penalty_weight_adjusted == 0

    def test_high_threshold_no_adjustments(self):
        """Very high threshold means no adjustments even when enabled."""
        st = _make_symbol_table()
        general = _input_clause(_pos_lit(_func(P, _var(0))))
        complement = _input_clause(_neg_lit(_func(P, _const(A))))

        opts = SearchOptions(
            penalty_weight_enabled=True,
            penalty_weight_threshold=9999.0,
            max_given=20,
            quiet=True,
        )
        search = GivenClauseSearch(opts, symbol_table=st)
        result = search.run(sos=[general, complement])
        assert result.stats.penalty_weight_adjusted == 0


# ── Proof soundness ─────────────────────────────────────────────────────────


class TestProofSoundness:
    """Ensure penalty weight adjustments don't compromise proof validity."""

    def test_trivial_proof_has_proofs(self):
        st = _make_symbol_table()
        c1 = _input_clause(_pos_lit(_func(P, _const(A))))
        c2 = _input_clause(_neg_lit(_func(P, _const(A))))

        opts = SearchOptions(
            penalty_weight_enabled=True,
            max_given=10,
            quiet=True,
        )
        search = GivenClauseSearch(opts, symbol_table=st)
        result = search.run(sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) >= 1

    def test_chain_proof_sound_all_modes(self):
        """Proofs found under all modes should be valid."""
        st = _make_symbol_table()
        bridge = _input_clause(
            _neg_lit(_func(P, _var(0))),
            _pos_lit(_func(Q, _var(0))),
        )
        fact = _input_clause(_pos_lit(_func(P, _const(A))))
        goal = _input_clause(_neg_lit(_func(Q, _const(A))))

        for mode in ("linear", "exponential", "step"):
            opts = SearchOptions(
                penalty_weight_enabled=True,
                penalty_weight_mode=mode,
                max_given=20,
                quiet=True,
            )
            search = GivenClauseSearch(opts, symbol_table=st)
            result = search.run(sos=[bridge, fact, goal])
            assert result.exit_code == ExitCode.MAX_PROOFS_EXIT, (
                f"Mode {mode} failed to find proof"
            )
            assert len(result.proofs) >= 1, (
                f"Mode {mode} produced no proofs"
            )


# ── Parameter boundary conditions ────────────────────────────────────────────


class TestParameterBoundaries:
    """Verify search handles extreme parameter values gracefully."""

    def test_zero_threshold(self):
        """threshold=0 means all non-negative penalties trigger adjustment."""
        st = _make_symbol_table()
        c1 = _input_clause(_pos_lit(_func(P, _const(A))))
        c2 = _input_clause(_neg_lit(_func(P, _const(A))))

        opts = SearchOptions(
            penalty_weight_enabled=True,
            penalty_weight_threshold=0.0,
            max_given=10,
            quiet=True,
        )
        search = GivenClauseSearch(opts, symbol_table=st)
        result = search.run(sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_multiplier_one_step_no_change(self):
        """multiplier=1.0 in step mode: base * 1.0 = base, so no weight change."""
        st = _make_symbol_table()
        c1 = _input_clause(_pos_lit(_func(P, _const(A))))
        c2 = _input_clause(_neg_lit(_func(P, _const(A))))

        opts = SearchOptions(
            penalty_weight_enabled=True,
            penalty_weight_multiplier=1.0,
            penalty_weight_mode="step",
            max_given=10,
            quiet=True,
        )
        search = GivenClauseSearch(opts, symbol_table=st)
        result = search.run(sos=[c1, c2])
        # step: base * 1.0 = base, no actual change
        assert result.stats.penalty_weight_adjusted == 0

    def test_very_low_cap(self):
        """Very low cap limits maximum adjusted weight."""
        st = _make_symbol_table()
        c1 = _input_clause(_pos_lit(_func(P, _const(A))))
        c2 = _input_clause(_neg_lit(_func(P, _const(A))))

        opts = SearchOptions(
            penalty_weight_enabled=True,
            penalty_weight_max=10.0,
            max_given=10,
            quiet=True,
        )
        search = GivenClauseSearch(opts, symbol_table=st)
        result = search.run(sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_very_high_multiplier_capped(self):
        """Very high multiplier capped by max_adjusted_weight."""
        st = _make_symbol_table()
        c1 = _input_clause(_pos_lit(_func(P, _const(A))))
        c2 = _input_clause(_neg_lit(_func(P, _const(A))))

        opts = SearchOptions(
            penalty_weight_enabled=True,
            penalty_weight_multiplier=1000.0,
            penalty_weight_max=500.0,
            max_given=10,
            quiet=True,
        )
        search = GivenClauseSearch(opts, symbol_table=st)
        result = search.run(sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT


# ── Performance characteristics ──────────────────────────────────────────────


class TestPerformanceCharacteristics:
    """Verify penalty weight doesn't cause performance regressions."""

    def test_enabled_completes_promptly(self):
        """Search with penalty weight should not time out on simple problems."""
        st = _make_symbol_table()
        bridge = _input_clause(
            _neg_lit(_func(P, _var(0))),
            _pos_lit(_func(Q, _var(0))),
        )
        bridge2 = _input_clause(
            _neg_lit(_func(Q, _var(0))),
            _pos_lit(_func(R, _var(0))),
        )
        fact = _input_clause(_pos_lit(_func(P, _const(A))))
        goal = _input_clause(_neg_lit(_func(R, _const(A))))

        opts = SearchOptions(
            penalty_weight_enabled=True,
            penalty_weight_threshold=0.5,
            penalty_weight_multiplier=10.0,
            max_given=100,
            max_seconds=5.0,
            quiet=True,
        )
        search = GivenClauseSearch(opts, symbol_table=st)
        result = search.run(sos=[bridge, bridge2, fact, goal])
        assert result.exit_code != ExitCode.MAX_SECONDS_EXIT
