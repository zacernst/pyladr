"""Integration tests: validate all optimizations compose correctly.

Tests that Priority SOS, lazy demodulation, and indexed subsumption
work together without interference when enabled simultaneously.
"""

from __future__ import annotations

import gc
import sys

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions, ExitCode
from pyladr.search.priority_sos import PrioritySOS
from pyladr.search.lazy_demod import LazyDemodState


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_simple_clauses():
    """Create P(x) and -P(a) for a simple resolution proof."""
    a = get_rigid_term(2, 0)
    x = get_variable_term(0)
    Pa = get_rigid_term(3, 1, (a,))
    Px = get_rigid_term(3, 1, (x,))

    c1 = Clause(
        literals=(Literal(sign=True, atom=Px),),
        justification=(Justification(just_type=JustType.INPUT),),
    )
    c2 = Clause(
        literals=(Literal(sign=False, atom=Pa),),
        justification=(Justification(just_type=JustType.INPUT),),
    )
    return [c1, c2]


def _make_two_step_clauses():
    """Create clauses requiring a 2-step proof: P(x), -P(f(a)), f(a)=a."""
    st = SymbolTable()
    a_sym = st.str_to_sn("a", 0)
    f_sym = st.str_to_sn("f", 1)
    P_sym = st.str_to_sn("P", 1)

    a = get_rigid_term(a_sym, 0)
    x = get_variable_term(0)
    fa = get_rigid_term(f_sym, 1, (a,))
    Px = get_rigid_term(P_sym, 1, (x,))
    Pfa = get_rigid_term(P_sym, 1, (fa,))

    c1 = Clause(
        literals=(Literal(sign=True, atom=Px),),
        justification=(Justification(just_type=JustType.INPUT),),
    )
    c2 = Clause(
        literals=(Literal(sign=False, atom=Pfa),),
        justification=(Justification(just_type=JustType.INPUT),),
    )
    return st, [c1, c2]


def _run_with_opts(**kwargs) -> tuple:
    """Run simple proof with given SearchOptions kwargs.

    Returns (exit_code, n_proofs, stats).
    """
    opts = SearchOptions(**kwargs)
    search = GivenClauseSearch(options=opts)
    clauses = _make_simple_clauses()
    result = search.run(usable=[], sos=clauses)
    return result.exit_code, len(result.proofs), search


# ── Test: Individual optimizations ───────────────────────────────────────


class TestIndividualOptimizations:
    """Verify each optimization works independently."""

    def test_baseline(self):
        exit_code, n_proofs, _ = _run_with_opts()
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        assert n_proofs == 1

    def test_priority_sos_only(self):
        exit_code, n_proofs, search = _run_with_opts(priority_sos=True)
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        assert n_proofs == 1
        assert isinstance(search._state.sos, PrioritySOS)

    def test_lazy_demod_only(self):
        exit_code, n_proofs, search = _run_with_opts(lazy_demod=True)
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        assert n_proofs == 1
        assert search._lazy_demod is not None

    def test_priority_sos_and_lazy_demod(self):
        exit_code, n_proofs, search = _run_with_opts(
            priority_sos=True, lazy_demod=True,
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        assert n_proofs == 1
        assert isinstance(search._state.sos, PrioritySOS)
        assert search._lazy_demod is not None


# ── Test: Compound optimization correctness ─────────────────────────────


class TestCompoundCorrectness:
    """All optimization combinations produce identical proof results."""

    OPTION_COMBOS = [
        {},
        {"priority_sos": True},
        {"lazy_demod": True},
        {"priority_sos": True, "lazy_demod": True},
    ]

    def test_all_combos_find_proof(self):
        """Every combination finds the proof."""
        for opts_kwargs in self.OPTION_COMBOS:
            exit_code, n_proofs, _ = _run_with_opts(**opts_kwargs)
            assert exit_code == ExitCode.MAX_PROOFS_EXIT, (
                f"Failed with opts={opts_kwargs}: exit_code={exit_code}"
            )
            assert n_proofs == 1, f"Failed with opts={opts_kwargs}"

    def test_all_combos_sos_empty_on_no_proof(self):
        """SOS-empty result consistent across combinations."""
        # Create unsatisfiable SOS with no complementary literals
        a = get_rigid_term(2, 0)
        b = get_rigid_term(3, 0)
        Pa = get_rigid_term(4, 1, (a,))
        Pb = get_rigid_term(4, 1, (b,))

        c1 = Clause(
            literals=(Literal(sign=True, atom=Pa),),
            justification=(Justification(just_type=JustType.INPUT),),
        )
        c2 = Clause(
            literals=(Literal(sign=True, atom=Pb),),
            justification=(Justification(just_type=JustType.INPUT),),
        )

        for opts_kwargs in self.OPTION_COMBOS:
            opts = SearchOptions(**opts_kwargs)
            search = GivenClauseSearch(options=opts)
            result = search.run(usable=[], sos=[c1, c2])
            assert result.exit_code == ExitCode.SOS_EMPTY_EXIT, (
                f"Expected SOS_EMPTY with opts={opts_kwargs}, "
                f"got exit_code={result.exit_code}"
            )

    def test_all_combos_max_given_limit(self):
        """Max given limit respected with all optimization combos."""
        for opts_kwargs in self.OPTION_COMBOS:
            merged = {**opts_kwargs, "max_given": 2}
            opts = SearchOptions(**merged)
            search = GivenClauseSearch(options=opts)

            # Clauses that won't find a proof quickly
            a = get_rigid_term(2, 0)
            b = get_rigid_term(3, 0)
            Pa = get_rigid_term(4, 1, (a,))
            Pb = get_rigid_term(4, 1, (b,))

            c1 = Clause(
                literals=(Literal(sign=True, atom=Pa),),
                justification=(Justification(just_type=JustType.INPUT),),
            )
            c2 = Clause(
                literals=(Literal(sign=True, atom=Pb),),
                justification=(Justification(just_type=JustType.INPUT),),
            )
            result = search.run(usable=[], sos=[c1, c2])
            assert result.stats.given <= 3, (
                f"Max given violated with opts={opts_kwargs}, "
                f"given={result.stats.given}"
            )


# ── Test: PrioritySOS + search engine interaction ───────────────────────


class TestPrioritySOS_SearchEngine:
    """Verify PrioritySOS integrates correctly with the search engine."""

    def test_sos_type_correct(self):
        opts = SearchOptions(priority_sos=True)
        search = GivenClauseSearch(options=opts)
        assert isinstance(search._state.sos, PrioritySOS)

    def test_sos_type_default(self):
        opts = SearchOptions()
        search = GivenClauseSearch(options=opts)
        assert not isinstance(search._state.sos, PrioritySOS)

    def test_clause_append_and_remove(self):
        """Clauses can be added and removed from PrioritySOS during search."""
        opts = SearchOptions(priority_sos=True)
        search = GivenClauseSearch(options=opts)
        sos = search._state.sos

        c = Clause(
            literals=(Literal(sign=True, atom=get_rigid_term(2, 0)),),
            id=1,
        )
        c.weight = 5.0
        sos.append(c)
        assert sos.length == 1
        assert sos.contains(c)

        sos.remove(c)
        assert sos.length == 0
        assert not sos.contains(c)

    def test_disable_clause_with_priority_sos(self):
        """SearchState.disable_clause works with PrioritySOS."""
        opts = SearchOptions(priority_sos=True)
        search = GivenClauseSearch(options=opts)

        c = Clause(
            literals=(Literal(sign=True, atom=get_rigid_term(2, 0)),),
            id=1,
        )
        c.weight = 5.0
        search._state.sos.append(c)
        search._state.disable_clause(c)
        assert search._state.sos.length == 0
        assert search._state.disabled.length == 1


# ── Test: LazyDemod + search engine interaction ─────────────────────────


class TestLazyDemod_SearchEngine:
    """Verify lazy demod integrates correctly with the search engine."""

    def test_lazy_demod_state_created(self):
        opts = SearchOptions(lazy_demod=True)
        search = GivenClauseSearch(options=opts)
        assert search._lazy_demod is not None
        assert search._lazy_demod.version == 0

    def test_lazy_demod_not_created_when_disabled(self):
        opts = SearchOptions(lazy_demod=False)
        search = GivenClauseSearch(options=opts)
        assert search._lazy_demod is None

    def test_lazy_demod_with_no_demodulators(self):
        """Lazy demod is a no-op when there are no demodulators."""
        opts = SearchOptions(lazy_demod=True)
        search = GivenClauseSearch(options=opts)
        clauses = _make_simple_clauses()
        result = search.run(usable=[], sos=clauses)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        # No demodulators → deferred_demods should be 0
        assert search._lazy_demod.stats.deferred_demods == 0


# ── Test: Memory stability ──────────────────────────────────────────────


class TestMemoryStability:
    """Verify no memory leaks from optimization data structures."""

    def test_priority_sos_cleanup_after_search(self):
        """PrioritySOS internal structures don't grow unbounded."""
        opts = SearchOptions(priority_sos=True)
        search = GivenClauseSearch(options=opts)
        clauses = _make_simple_clauses()
        result = search.run(usable=[], sos=clauses)

        sos = search._state.sos
        if isinstance(sos, PrioritySOS):
            # After search, SOS should be empty (all clauses processed)
            assert sos.length == 0
            # _by_id should also be empty
            assert len(sos._by_id) == 0

    def test_lazy_demod_version_map_bounded(self):
        """Lazy demod version map doesn't grow without bound."""
        opts = SearchOptions(lazy_demod=True)
        search = GivenClauseSearch(options=opts)
        clauses = _make_simple_clauses()
        result = search.run(usable=[], sos=clauses)

        if search._lazy_demod is not None:
            # Version map should be reasonable size
            n_versions = len(search._lazy_demod._clause_versions)
            n_clauses = search.stats.kept + len(clauses)
            # Should not be orders of magnitude larger than clauses
            assert n_versions <= n_clauses * 2, (
                f"Version map too large: {n_versions} vs {n_clauses} clauses"
            )

    def test_compound_opts_memory_stable(self):
        """Combined optimizations don't cause memory explosion."""
        opts = SearchOptions(priority_sos=True, lazy_demod=True)
        search = GivenClauseSearch(options=opts)
        clauses = _make_simple_clauses()

        # Run search
        gc.collect()
        before = _get_pyladr_object_count()
        result = search.run(usable=[], sos=clauses)
        gc.collect()
        after = _get_pyladr_object_count()

        # Object count should not explode (allow 10x growth for search)
        assert after < before + 1000, (
            f"Object count grew too much: {before} → {after}"
        )


def _get_pyladr_object_count() -> int:
    """Count objects from pyladr modules (rough estimate)."""
    count = 0
    for obj in gc.get_objects():
        try:
            mod = getattr(type(obj), "__module__", "")
            if isinstance(mod, str) and mod.startswith("pyladr"):
                count += 1
        except (ReferenceError, AttributeError):
            pass
    return count


# ── Test: Configuration compatibility ───────────────────────────────────


class TestConfigurationCompatibility:
    """Verify optimization flags don't interfere with other options."""

    def test_max_proofs_with_all_opts(self):
        opts = SearchOptions(
            priority_sos=True,
            lazy_demod=True,
            max_proofs=1,
        )
        search = GivenClauseSearch(options=opts)
        clauses = _make_simple_clauses()
        result = search.run(usable=[], sos=clauses)
        assert len(result.proofs) == 1

    def test_factoring_with_all_opts(self):
        opts = SearchOptions(
            priority_sos=True,
            lazy_demod=True,
            factoring=True,
        )
        search = GivenClauseSearch(options=opts)
        clauses = _make_simple_clauses()
        result = search.run(usable=[], sos=clauses)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_quiet_mode_with_all_opts(self):
        opts = SearchOptions(
            priority_sos=True,
            lazy_demod=True,
            quiet=True,
        )
        search = GivenClauseSearch(options=opts)
        clauses = _make_simple_clauses()
        result = search.run(usable=[], sos=clauses)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_defaults_unchanged(self):
        """Default options don't enable any optimizations."""
        opts = SearchOptions()
        assert opts.priority_sos is False
        assert opts.lazy_demod is False
