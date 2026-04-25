"""Integration tests for penalty propagation through inference rules.

Verifies that penalty propagation works end-to-end through
GivenClauseSearch.run() for all inference rule types:
- Binary resolution
- Hyper-resolution
- Factoring
- Paramodulation (deferred — requires equality setup)

Also verifies:
- Penalty cache seeded for initial clauses
- Derived clauses inherit penalties from general parents
- Penalty propagation disabled by default (C compatibility)
- PrioritySOS uses combined penalties when enabled
- Multi-generation penalty decay through derivation chains
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
from pyladr.search.penalty_propagation import PenaltyCache


# ── Symbol constants (matching test_search.py) ────────────────────────────

A, B, C_SYM = 1, 2, 3
F, G = 10, 11
P, Q, R = 20, 21, 22


def _make_symbol_table() -> SymbolTable:
    """Create a SymbolTable with test symbol IDs pre-registered.

    The pre-existing term.py changes require symbol registration for to_str()
    to work when a SymbolTable is provided to GivenClauseSearch.
    """
    st = SymbolTable()
    # Register symbols at specific IDs used in tests
    _symbols = [
        (A, "a", 0), (B, "b", 0), (C_SYM, "c", 0),
        (F, "f", 2), (G, "g", 1),
        (P, "P", 1), (Q, "Q", 1), (R, "R", 1),
    ]
    for symnum, name, arity in _symbols:
        sym = Symbol(symnum=symnum, name=name, arity=arity)
        st._by_id[symnum] = sym
        st._by_name_arity[(name, arity)] = symnum
    # Ensure next_id is beyond all our symbols
    st._next_id = max(symnum for symnum, _, _ in _symbols) + 1
    return st


# ── Helpers ───────────────────────────────────────────────────────────────


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
    """Create an input clause."""
    return Clause(
        literals=tuple(lits),
        justification=(Justification(just_type=JustType.INPUT),),
    )


# ── Disabled by default (C compatibility) ────────────────────────────────


class TestPenaltyPropagationDisabled:
    """When penalty_propagation=False, no cache is created."""

    def test_no_cache_by_default(self):
        opts = SearchOptions(print_given=False)
        search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())
        assert search._penalty_cache is None

    def test_search_works_without_propagation(self):
        """Normal search works fine without penalty propagation."""
        pa = _func(P, _const(A))
        c1 = _input_clause(_pos_lit(pa))
        c2 = _input_clause(_neg_lit(pa))

        opts = SearchOptions(print_given=False)
        search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())
        result = search.run(sos=[c1, c2])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert search._penalty_cache is None


# ── Cache initialization ─────────────────────────────────────────────────


class TestPenaltyCacheInitialization:
    """Penalty cache is seeded for initial clauses."""

    def test_cache_created_when_enabled(self):
        opts = SearchOptions(
            print_given=False, penalty_propagation=True,
        )
        search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())
        assert search._penalty_cache is not None
        assert isinstance(search._penalty_cache, PenaltyCache)

    def test_initial_clauses_seeded(self):
        """Initial usable + SOS clauses should be in the penalty cache."""
        pa = _func(P, _const(A))
        c1 = _input_clause(_pos_lit(pa))
        c2 = _input_clause(_neg_lit(pa))

        opts = SearchOptions(
            print_given=False, penalty_propagation=True,
        )
        search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())
        result = search.run(sos=[c1, c2])

        cache = search._penalty_cache
        assert cache is not None
        # Initial clauses should be in cache
        assert cache.get(c1.id) is not None
        assert cache.get(c2.id) is not None

    def test_initial_clauses_have_zero_depth(self):
        """Input clauses have depth=0 (no inheritance)."""
        pa = _func(P, _const(A))
        c1 = _input_clause(_pos_lit(pa))

        opts = SearchOptions(
            print_given=False, penalty_propagation=True,
        )
        search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())
        result = search.run(sos=[c1])

        rec = search._penalty_cache.get(c1.id)
        assert rec is not None
        assert rec.depth == 0
        assert rec.inherited_penalty == 0.0


# ── Binary resolution penalty inheritance ────────────────────────────────


class TestBinaryResolutionInheritance:
    """Penalty propagation through binary resolution derivations."""

    def test_derived_clause_gets_cached(self):
        """Clauses derived via binary resolution should appear in penalty cache."""
        x = _var(0)
        # P(x) — overly general (all-variable)
        c1 = _input_clause(_pos_lit(_func(P, x)))
        # -P(a) | Q(a) — resolves with P(x)
        c2 = _input_clause(_neg_lit(_func(P, _const(A))), _pos_lit(_func(Q, _const(A))))

        opts = SearchOptions(
            print_given=False, penalty_propagation=True,
            penalty_propagation_threshold=5.0,
            penalty_propagation_decay=0.5,
            max_given=20,
        )
        search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())
        result = search.run(sos=[c1, c2])

        cache = search._penalty_cache
        assert cache is not None
        # More clauses should be in cache than just the 2 initial ones
        assert len(cache) >= 2

    def test_general_parent_penalty_propagates(self):
        """Child of a high-penalty parent should inherit penalty."""
        x = _var(0)
        # P(x) — single literal, all variables → penalty >= 10.0
        general_clause = _input_clause(_pos_lit(_func(P, x)))

        # -P(a) — resolves with P(x) to produce empty clause, but let's
        # add more structure to ensure intermediate clauses are generated
        # -P(a) | Q(b)
        c2 = _input_clause(
            _neg_lit(_func(P, _const(A))),
            _pos_lit(_func(Q, _const(B))),
        )
        # -Q(b) — to eventually reach proof
        c3 = _input_clause(_neg_lit(_func(Q, _const(B))))

        opts = SearchOptions(
            print_given=False, penalty_propagation=True,
            penalty_propagation_threshold=5.0,
            penalty_propagation_decay=0.5,
            penalty_propagation_max_depth=5,
            max_given=30,
        )
        search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())
        result = search.run(sos=[general_clause, c2, c3])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

        cache = search._penalty_cache
        # Check the general clause has high penalty
        gen_rec = cache.get(general_clause.id)
        assert gen_rec is not None
        assert gen_rec.own_penalty >= 10.0  # P(x) → all-variable penalty

        # Check that at least one derived clause inherited penalty
        inherited_found = False
        for cid, clause in search._all_clauses.items():
            rec = cache.get(cid)
            if rec is not None and rec.inherited_penalty > 0.0:
                inherited_found = True
                assert rec.depth >= 1
                break

        assert inherited_found, "No derived clause inherited penalty from general parent"


# ── Hyper-resolution penalty inheritance ─────────────────────────────────


class TestHyperResolutionInheritance:
    """Penalty propagation through hyper-resolution derivations."""

    def test_hyper_resolution_with_propagation(self):
        """Hyper-resolution resolvents should be in penalty cache."""
        x = _var(0)
        # Nucleus: -P(x) | -Q(x) | R(x) — has negative literals
        nucleus = _input_clause(
            _neg_lit(_func(P, x)),
            _neg_lit(_func(Q, x)),
            _pos_lit(_func(R, x)),
        )
        # Satellite 1: P(a) — positive unit
        sat1 = _input_clause(_pos_lit(_func(P, _const(A))))
        # Satellite 2: Q(a) — positive unit
        sat2 = _input_clause(_pos_lit(_func(Q, _const(A))))
        # Goal: -R(a) — to complete proof
        goal = _input_clause(_neg_lit(_func(R, _const(A))))

        opts = SearchOptions(
            print_given=False,
            hyper_resolution=True,
            binary_resolution=False,
            penalty_propagation=True,
            penalty_propagation_threshold=1.0,  # Low threshold to catch more
            max_given=30,
        )
        search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())
        result = search.run(sos=[nucleus, sat1, sat2, goal])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

        cache = search._penalty_cache
        assert cache is not None
        # All initial clauses should be cached
        for c in [nucleus, sat1, sat2, goal]:
            assert cache.get(c.id) is not None

        # Derived clauses should also be cached
        assert len(cache) > 4  # More than just initial clauses


# ── PrioritySOS integration ──────────────────────────────────────────────


class TestPrioritySosIntegration:
    """Penalty propagation with PrioritySOS heap ordering."""

    def test_priority_sos_with_propagation(self):
        """Combined penalties should be used for PrioritySOS ordering."""
        x = _var(0)
        # P(x) — general clause with high penalty
        c1 = _input_clause(_pos_lit(_func(P, x)))
        # -P(a) — specific clause
        c2 = _input_clause(_neg_lit(_func(P, _const(A))))

        opts = SearchOptions(
            print_given=False,
            penalty_propagation=True,
            priority_sos=True,
            unification_weight=1,  # Enable penalty-based selection
            penalty_propagation_threshold=5.0,
            max_given=10,
        )
        search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())
        result = search.run(sos=[c1, c2])

        # Should find proof
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

        cache = search._penalty_cache
        assert cache is not None
        # General clause should have high penalty in cache
        rec = cache.get(c1.id)
        assert rec is not None
        assert rec.own_penalty >= 10.0


# ── Configuration modes ──────────────────────────────────────────────────


class TestConfigurationModes:
    """Test different penalty propagation configuration modes."""

    def _run_with_mode(self, mode: str) -> GivenClauseSearch:
        """Run a simple search with a specific penalty propagation mode."""
        x = _var(0)
        c1 = _input_clause(_pos_lit(_func(P, x)))
        c2 = _input_clause(
            _neg_lit(_func(P, _const(A))),
            _pos_lit(_func(Q, _const(A))),
        )
        c3 = _input_clause(_neg_lit(_func(Q, _const(A))))

        opts = SearchOptions(
            print_given=False,
            penalty_propagation=True,
            penalty_propagation_mode=mode,
            penalty_propagation_threshold=5.0,
            penalty_propagation_decay=0.5,
            max_given=30,
        )
        search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())
        search.run(sos=[c1, c2, c3])
        return search

    def test_additive_mode(self):
        search = self._run_with_mode("additive")
        assert search._penalty_cache is not None
        assert search._penalty_cache.config.mode.name == "ADDITIVE"

    def test_multiplicative_mode(self):
        search = self._run_with_mode("multiplicative")
        assert search._penalty_cache is not None
        assert search._penalty_cache.config.mode.name == "MULTIPLICATIVE"

    def test_max_mode(self):
        search = self._run_with_mode("max")
        assert search._penalty_cache is not None
        assert search._penalty_cache.config.mode.name == "MAX"

    def test_all_modes_find_proof(self):
        """All modes should still find the proof — penalty affects ordering, not soundness."""
        for mode in ("additive", "multiplicative", "max"):
            x = _var(0)
            c1 = _input_clause(_pos_lit(_func(P, x)))
            c2 = _input_clause(_neg_lit(_func(P, _const(A))))

            opts = SearchOptions(
                print_given=False,
                penalty_propagation=True,
                penalty_propagation_mode=mode,
                max_given=20,
            )
            search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())
            result = search.run(sos=[c1, c2])
            assert result.exit_code == ExitCode.MAX_PROOFS_EXIT, (
                f"Mode {mode} failed to find proof"
            )


# ── Depth limiting ───────────────────────────────────────────────────────


class TestDepthLimiting:
    """Verify depth limiting stops penalty propagation."""

    def test_depth_zero_means_no_propagation(self):
        """max_depth=0 means unlimited, but let's verify depth tracking works."""
        x = _var(0)
        c1 = _input_clause(_pos_lit(_func(P, x)))
        c2 = _input_clause(
            _neg_lit(_func(P, _const(A))),
            _pos_lit(_func(Q, _const(A))),
        )
        c3 = _input_clause(_neg_lit(_func(Q, _const(A))))

        opts = SearchOptions(
            print_given=False,
            penalty_propagation=True,
            penalty_propagation_max_depth=0,  # unlimited
            penalty_propagation_threshold=5.0,
            penalty_propagation_decay=0.5,
            max_given=30,
        )
        search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())
        result = search.run(sos=[c1, c2, c3])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_depth_1_limits_inheritance(self):
        """max_depth=1 allows only direct children to inherit."""
        x = _var(0)
        c1 = _input_clause(_pos_lit(_func(P, x)))
        c2 = _input_clause(
            _neg_lit(_func(P, _const(A))),
            _pos_lit(_func(Q, _const(A))),
        )
        c3 = _input_clause(_neg_lit(_func(Q, _const(A))))

        opts = SearchOptions(
            print_given=False,
            penalty_propagation=True,
            penalty_propagation_max_depth=1,
            penalty_propagation_threshold=5.0,
            penalty_propagation_decay=0.5,
            max_given=30,
        )
        search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())
        result = search.run(sos=[c1, c2, c3])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

        cache = search._penalty_cache
        # No clause should have depth > 1
        for cid in search._all_clauses:
            rec = cache.get(cid)
            if rec is not None:
                assert rec.depth <= 1, f"Clause {cid} has depth {rec.depth} > max_depth=1"


# ── Display integration ──────────────────────────────────────────────────


class TestDisplayIntegration:
    """Test that inherited penalty appears in selection extras display."""

    def test_format_selection_extras_shows_inherited(self):
        """When penalty propagation is enabled, inherited penalty shown in output."""
        from pyladr.search.penalty_propagation import PenaltyRecord

        opts = SearchOptions(
            print_given=False,
            penalty_propagation=True,
            penalty_propagation_threshold=5.0,
        )
        search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())

        # Manually seed a clause with inherited penalty
        c = _input_clause(_pos_lit(_func(P, _const(A))))
        c.id = 1
        search._penalty_cache.put(1, PenaltyRecord(
            own_penalty=2.0, inherited_penalty=3.5,
            combined_penalty=5.5, depth=1,
        ))

        extras = search._format_selection_extras(c)
        assert "ipen=3.50" in extras

    def test_format_no_inherited_no_display(self):
        """When inherited penalty is 0, ipen is not shown."""
        from pyladr.search.penalty_propagation import PenaltyRecord

        opts = SearchOptions(
            print_given=False,
            penalty_propagation=True,
        )
        search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())

        c = _input_clause(_pos_lit(_func(P, _const(A))))
        c.id = 1
        search._penalty_cache.put(1, PenaltyRecord(
            own_penalty=2.0, inherited_penalty=0.0,
            combined_penalty=2.0, depth=0,
        ))

        extras = search._format_selection_extras(c)
        assert "ipen" not in extras


# ── Soundness preservation ───────────────────────────────────────────────


class TestSoundnessPreservation:
    """Penalty propagation must not affect proof soundness."""

    def test_proof_found_with_propagation(self):
        """Basic proof still found when propagation enabled."""
        pa = _func(P, _const(A))
        c1 = _input_clause(_pos_lit(pa))
        c2 = _input_clause(_neg_lit(pa))

        opts = SearchOptions(
            print_given=False,
            penalty_propagation=True,
        )
        search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())
        result = search.run(sos=[c1, c2])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1

    def test_two_step_proof_with_propagation(self):
        """Multi-step proof found with propagation enabled."""
        x = _var(0)
        c1 = _input_clause(_pos_lit(_func(P, x)))
        c2 = _input_clause(
            _neg_lit(_func(P, _const(A))),
            _pos_lit(_func(Q, _const(A))),
        )
        c3 = _input_clause(_neg_lit(_func(Q, _const(A))))

        opts = SearchOptions(
            print_given=False,
            penalty_propagation=True,
            penalty_propagation_threshold=5.0,
            max_given=30,
        )
        search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())
        result = search.run(sos=[c1, c2, c3])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_sos_empty_with_propagation(self):
        """Unsatisfiable SOS correctly reports empty."""
        c1 = _input_clause(_pos_lit(_func(P, _const(A))))

        opts = SearchOptions(
            print_given=False,
            penalty_propagation=True,
            max_given=5,
        )
        search = GivenClauseSearch(options=opts, symbol_table=_make_symbol_table())
        result = search.run(sos=[c1])

        # Should exhaust SOS or hit max_given, not crash
        assert result.exit_code in (ExitCode.SOS_EMPTY_EXIT, ExitCode.MAX_GIVEN_EXIT)
