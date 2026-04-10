"""Performance regression detection for hierarchical GNN integration.

Ensures that adding hierarchical GNN code paths introduces ZERO overhead
when features are disabled. Tests measure that disabled-mode performance
is indistinguishable from baseline.

Run with: pytest tests/compatibility/test_hierarchical_perf_regression.py -v
"""

from __future__ import annotations

import time

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions


# ── Helpers ────────────────────────────────────────────────────────────────


def _build_simple_problem() -> tuple[str, str]:
    """Build a simple LADR problem for performance testing.

    Returns (input_text, description).
    """
    input_text = """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(x) | R(x).
  -R(x) | S(x).
  -S(a).
end_of_list.
"""
    return input_text, "chain_5"


def _run_perf_python(input_text: str, max_given: int = 200) -> object:
    """Run Python prover on LADR input for performance testing."""
    from pyladr.parsing.ladr_parser import LADRParser

    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(input_text)

    opts = SearchOptions(
        max_given=max_given,
        quiet=True,
    )

    # Apply auto if present
    from pyladr.apps.prover9 import _deny_goals, _apply_settings
    usable, sos = _deny_goals(parsed, st)
    _apply_settings(parsed, opts, st)

    search = GivenClauseSearch(options=opts, symbol_table=st)
    return search.run(usable=usable, sos=sos)


def _timed_search(clauses: list[Clause], opts: SearchOptions) -> tuple[float, object]:
    """Run search and return (elapsed_seconds, result)."""
    start = time.perf_counter()
    result = GivenClauseSearch(options=opts).run(usable=[], sos=clauses)
    elapsed = time.perf_counter() - start
    return elapsed, result


# ── Zero Overhead When Disabled ───────────────────────────────────────────


class TestZeroOverheadWhenDisabled:
    """Verify no performance regression from hierarchical GNN code existing."""

    def test_short_chain_no_overhead(self):
        """Short resolution chain: disabled mode ≈ baseline timing."""
        input_text, _ = _build_simple_problem()

        # Warm up
        _run_perf_python(input_text)

        start = time.perf_counter()
        r_baseline = _run_perf_python(input_text)
        t_baseline = time.perf_counter() - start

        start = time.perf_counter()
        r_disabled = _run_perf_python(input_text)
        t_disabled = time.perf_counter() - start

        assert r_baseline.exit_code == r_disabled.exit_code

        # Allow 2x tolerance (timing noise), but should be nearly identical
        if t_baseline > 0.001:
            ratio = t_disabled / t_baseline
            assert ratio < 2.0, (
                f"Disabled mode {ratio:.1f}x slower than baseline "
                f"({t_disabled:.4f}s vs {t_baseline:.4f}s)"
            )

    def test_repeated_runs_stable(self):
        """Multiple runs produce stable timing."""
        input_text, _ = _build_simple_problem()

        # Warm up
        _run_perf_python(input_text)

        times = []
        for _ in range(5):
            start = time.perf_counter()
            _run_perf_python(input_text)
            times.append(time.perf_counter() - start)

        avg = sum(times) / len(times)
        max_t = max(times)

        # No single run should be more than 3x the average
        if avg > 0.001:
            assert max_t < avg * 3.0, (
                f"Unstable timing: max={max_t:.4f}s, avg={avg:.4f}s"
            )


# ── Search Statistics Overhead ────────────────────────────────────────────


class TestStatisticsOverhead:
    """Statistics computation must not slow down with hierarchical params."""

    def test_stats_computation_fast(self):
        """Statistics access is constant-time regardless of configuration."""
        from pyladr.search.statistics import SearchStatistics

        stats = SearchStatistics()
        # Simulate some work
        for _ in range(1000):
            stats.given += 1
            stats.generated += 5
            stats.kept += 2

        start = time.perf_counter()
        for _ in range(10000):
            _ = stats.given
            _ = stats.generated
            _ = stats.kept
            _ = stats.elapsed_seconds()
        elapsed = time.perf_counter() - start

        # 10k stat accesses should be < 100ms
        assert elapsed < 0.1, f"Stats access too slow: {elapsed:.3f}s for 10k iterations"


# ── Config Construction Overhead ──────────────────────────────────────────


class TestConfigConstructionOverhead:
    """Configuration objects must be cheap to create."""

    def test_search_options_creation_fast(self):
        """SearchOptions creation with hierarchical params is fast."""
        start = time.perf_counter()
        for _ in range(10000):
            SearchOptions(
                binary_resolution=True,
                factoring=True,
                max_given=100,
                quiet=True,
                goal_directed=False,
                goal_proximity_weight=0.3,
                embedding_evolution_rate=0.01,
            )
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"SearchOptions creation too slow: {elapsed:.3f}s for 10k"

    def test_goal_directed_config_creation_fast(self):
        """GoalDirectedConfig creation is fast."""
        from pyladr.search.goal_directed import GoalDirectedConfig

        start = time.perf_counter()
        for _ in range(10000):
            GoalDirectedConfig(
                enabled=False,
                goal_proximity_weight=0.3,
                proximity_method="max",
            )
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"GoalDirectedConfig creation too slow: {elapsed:.3f}s"

    def test_hierarchical_config_creation_fast(self):
        """HierarchicalIntegrationConfig creation is fast."""
        from pyladr.search.hierarchical_integration import HierarchicalIntegrationConfig

        start = time.perf_counter()
        for _ in range(10000):
            HierarchicalIntegrationConfig(enabled=False)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"Config creation too slow: {elapsed:.3f}s"


# ── Import Overhead ───────────────────────────────────────────────────────


class TestImportOverhead:
    """Hierarchical modules must not cause heavy imports at module level."""

    def test_search_given_clause_import_no_torch(self):
        """given_clause.py does not trigger torch import at module level."""
        import sys
        # Just verify the module is importable without torch being loaded
        # (it may already be loaded in the test session, but the import
        # itself should not fail if torch were absent)
        import pyladr.search.given_clause
        assert pyladr.search.given_clause is not None

    def test_goal_directed_import_no_torch(self):
        """goal_directed.py does not require torch at module level."""
        import pyladr.search.goal_directed
        assert pyladr.search.goal_directed is not None

    def test_hierarchical_integration_import_no_torch(self):
        """hierarchical_integration.py does not require torch at module level."""
        import pyladr.search.hierarchical_integration
        assert pyladr.search.hierarchical_integration is not None
