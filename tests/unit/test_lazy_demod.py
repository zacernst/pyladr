"""Tests for lazy demodulation prototype."""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.inference.demodulation import DemodType, DemodulatorIndex
from pyladr.search.lazy_demod import LazyDemodState


def _make_eq_clause(lhs, rhs, sym_table, id=1):
    """Make a unit equality clause lhs = rhs."""
    eq_sym = sym_table.lookup_or_create("=", 2)
    atom = get_rigid_term(eq_sym, 2, (lhs, rhs))
    return Clause(
        literals=(Literal(sign=True, atom=atom),),
        id=id,
        justification=(Justification(just_type=JustType.INPUT),),
    )


class TestLazyDemodState:
    """Test LazyDemodState version tracking."""

    def test_initial_version(self):
        state = LazyDemodState()
        assert state.version == 0

    def test_bump_version(self):
        state = LazyDemodState()
        v = state.bump_version()
        assert v == 1
        assert state.version == 1

    def test_needs_reduction_unmarked(self):
        state = LazyDemodState()
        state.bump_version()  # version = 1
        c = Clause(literals=(), id=1)
        # Unmarked clause needs reduction when version > 0
        assert state.needs_reduction(c)

    def test_needs_reduction_partially_reduced(self):
        state = LazyDemodState()
        state.bump_version()
        c = Clause(literals=(), id=1)
        state.mark_partially_reduced(c)
        # Partially reduced (-1) still needs reduction
        assert state.needs_reduction(c)

    def test_fully_reduced_no_need(self):
        state = LazyDemodState()
        state.bump_version()
        c = Clause(literals=(), id=1)
        state.mark_fully_reduced(c)
        # Fully reduced at current version
        assert not state.needs_reduction(c)

    def test_new_demod_invalidates(self):
        state = LazyDemodState()
        c = Clause(literals=(), id=1)
        state.bump_version()
        state.mark_fully_reduced(c)
        assert not state.needs_reduction(c)
        # New demodulator bumps version
        state.bump_version()
        assert state.needs_reduction(c)

    def test_stats_tracking(self):
        state = LazyDemodState()
        assert state.stats.deferred_demods == 0
        assert state.stats.selection_demods == 0
        assert state.stats.already_reduced == 0


class TestLazyDemodIntegration:
    """Test lazy demod with actual demodulation."""

    def test_ensure_fully_reduced_empty_index(self):
        state = LazyDemodState()
        state.bump_version()
        idx = DemodulatorIndex()
        st = SymbolTable()
        c = Clause(literals=(), id=1)
        result = state.ensure_fully_reduced(c, idx, st)
        assert result is c  # No change (empty index → early return)
        # With empty index, marks as fully reduced without demodulating
        assert not state.needs_reduction(c)

    def test_ensure_fully_reduced_already_current(self):
        state = LazyDemodState()
        idx = DemodulatorIndex()
        st = SymbolTable()
        c = Clause(literals=(), id=1)
        # Version 0, clause at default -1, but version is 0 so -1 < 0 is True
        # Need to mark as fully reduced first to test the "already current" path
        state.mark_fully_reduced(c)
        assert not state.needs_reduction(c)
        result = state.ensure_fully_reduced(c, idx, st)
        assert result is c
        assert state.stats.already_reduced == 1

    def test_forget_clause(self):
        state = LazyDemodState()
        c = Clause(literals=(), id=1)
        state.mark_fully_reduced(c)
        state.forget(c)
        state.bump_version()
        # After forget, clause defaults to -1 which is < version 1
        assert state.needs_reduction(c)


class TestLazyDemodSearchIntegration:
    """Test lazy demod through the search engine."""

    def test_search_with_lazy_demod_disabled(self):
        """Baseline: search works with lazy_demod=False."""
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        st = SymbolTable()
        a_id = st.str_to_sn("a", 0)
        p_id = st.str_to_sn("P", 1)
        a = get_rigid_term(a_id, 0)
        x = get_variable_term(0)
        Pa = get_rigid_term(p_id, 1, (a,))
        Px = get_rigid_term(p_id, 1, (x,))

        c1 = Clause(
            literals=(Literal(sign=True, atom=Px),),
            justification=(Justification(just_type=JustType.INPUT),),
        )
        c2 = Clause(
            literals=(Literal(sign=False, atom=Pa),),
            justification=(Justification(just_type=JustType.INPUT),),
        )

        opts = SearchOptions(lazy_demod=False)
        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(usable=[], sos=[c1, c2])
        assert len(result.proofs) == 1

    def test_search_with_lazy_demod_enabled(self):
        """Search with lazy_demod=True finds same proof."""
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        st = SymbolTable()
        a_id = st.str_to_sn("a", 0)
        p_id = st.str_to_sn("P", 1)
        a = get_rigid_term(a_id, 0)
        x = get_variable_term(0)
        Pa = get_rigid_term(p_id, 1, (a,))
        Px = get_rigid_term(p_id, 1, (x,))

        c1 = Clause(
            literals=(Literal(sign=True, atom=Px),),
            justification=(Justification(just_type=JustType.INPUT),),
        )
        c2 = Clause(
            literals=(Literal(sign=False, atom=Pa),),
            justification=(Justification(just_type=JustType.INPUT),),
        )

        opts = SearchOptions(lazy_demod=True)
        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(usable=[], sos=[c1, c2])
        assert len(result.proofs) == 1
        assert search._lazy_demod is not None

    def test_lazy_demod_state_initialized(self):
        """Verify lazy demod state is created when enabled."""
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        opts = SearchOptions(lazy_demod=True)
        search = GivenClauseSearch(options=opts)
        assert search._lazy_demod is not None
        assert search._lazy_demod.version == 0

    def test_lazy_demod_not_initialized_when_disabled(self):
        """Verify lazy demod state is None when disabled."""
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        opts = SearchOptions(lazy_demod=False)
        search = GivenClauseSearch(options=opts)
        assert search._lazy_demod is None
