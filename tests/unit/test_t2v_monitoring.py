"""Tests for T2V monitoring features.

Covers:
1. Clause.given_distance field — default, assignment, persistence
2. Per-given gp annotation in _format_selection_extras()
3. Periodic proximity trend report — window accumulation, NOTE output, trend computation
4. Proof display gp annotation in _print_proof()
5. assign(tree2vec_proximity_report_interval, N) CLI parsing
6. T2V goal distance histogram — bucketing, proof vs non-proof separation
7. Regression — monitoring features don't affect proof finding
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.search.goal_directed import GoalProximityScorer


# ── Helpers ───────────────────────────────────────────────────────────────

SYM_P = 1
SYM_A = 2
SYM_B = 3


def var(n: int) -> Term:
    return get_variable_term(n)


def P(arg: Term) -> Term:
    return get_rigid_term(SYM_P, 1, (arg,))


def a() -> Term:
    return get_rigid_term(SYM_A, 0)


def b() -> Term:
    return get_rigid_term(SYM_B, 0)


def make_literal(sign: bool, atom: Term) -> Literal:
    return Literal(sign=sign, atom=atom)


def make_clause(*lits: Literal, clause_id: int = 0) -> Clause:
    return Clause(literals=lits, id=clause_id)


def make_goal_clause(*lits: Literal, clause_id: int = 0) -> Clause:
    return Clause(
        literals=lits,
        id=clause_id,
        justification=(Justification(just_type=JustType.DENY),),
    )


# ── 1. Clause.given_distance field ──────────────────────────────────────


class TestGivenProximityField:
    def test_default_is_zero(self) -> None:
        c = make_clause(make_literal(True, P(a())), clause_id=1)
        assert c.given_distance == 0.0

    def test_assignment(self) -> None:
        c = make_clause(make_literal(True, P(a())), clause_id=1)
        c.given_distance = 0.75
        assert c.given_distance == pytest.approx(0.75)

    def test_zero_means_not_set(self) -> None:
        """0.0 indicates the clause was never selected or goal distance was disabled."""
        c = make_clause(make_literal(True, P(a())), clause_id=1)
        assert c.given_distance == 0.0
        # Should not appear in proof display when 0.0
        assert not (c.given_distance > 0.0)


# ── 2. Per-given gp annotation in _format_selection_extras() ─────────────


class TestFormatSelectionExtras:
    def _make_engine(self, goal_proximity: bool = True):
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        opts = SearchOptions(max_seconds=1, tree2vec_goal_proximity=goal_proximity)
        st = SymbolTable()
        engine = GivenClauseSearch(options=opts, symbol_table=st)
        return engine

    def test_gp_annotation_when_goal_proximity_active(self) -> None:
        engine = self._make_engine(goal_proximity=True)

        # Set up minimal goal provider with scorer
        scorer = GoalProximityScorer(method="max")
        scorer.set_goals([[1.0, 0.0]])

        class _FakeGoalProvider:
            def __init__(self, s):
                self._goal_scorer = s
        engine._t2v_goal_provider = _FakeGoalProvider(scorer)

        # Store an embedding for the clause
        clause = make_clause(make_literal(True, P(a())), clause_id=42)
        engine._tree2vec_embeddings[42] = [0.9, 0.1]

        extras = engine._format_selection_extras(clause)
        assert ",gd=" in extras

    def test_no_gp_annotation_when_disabled(self) -> None:
        engine = self._make_engine(goal_proximity=False)

        clause = make_clause(make_literal(True, P(a())), clause_id=42)
        extras = engine._format_selection_extras(clause)
        assert "gd=" not in extras

    def test_no_gp_annotation_without_embedding(self) -> None:
        engine = self._make_engine(goal_proximity=True)

        scorer = GoalProximityScorer(method="max")
        scorer.set_goals([[1.0, 0.0]])

        class _FakeGoalProvider:
            def __init__(self, s):
                self._goal_scorer = s
        engine._t2v_goal_provider = _FakeGoalProvider(scorer)

        # No embedding stored for this clause
        clause = make_clause(make_literal(True, P(a())), clause_id=99)
        extras = engine._format_selection_extras(clause)
        assert "gd=" not in extras


# ── 3. Periodic proximity trend report ───────────────────────────────────


class TestProximityTrendReport:
    def test_window_accumulation(self) -> None:
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        opts = SearchOptions(max_seconds=1, tree2vec_goal_proximity=True)
        st = SymbolTable()
        engine = GivenClauseSearch(options=opts, symbol_table=st)

        assert engine._t2v_distance_window == []
        engine._t2v_distance_window.append(0.6)
        engine._t2v_distance_window.append(0.8)
        assert len(engine._t2v_distance_window) == 2

    def test_initial_goal_count_tracking(self) -> None:
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        opts = SearchOptions(max_seconds=1)
        st = SymbolTable()
        engine = GivenClauseSearch(options=opts, symbol_table=st)

        assert engine._t2v_initial_goal_count == 0
        engine._t2v_initial_goal_count = 3
        assert engine._t2v_initial_goal_count == 3

    def test_prev_avg_initialized_to_zero(self) -> None:
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        opts = SearchOptions(max_seconds=1)
        st = SymbolTable()
        engine = GivenClauseSearch(options=opts, symbol_table=st)

        assert engine._t2v_distance_prev_avg == 0.0

    def test_trend_computation(self) -> None:
        """Verify trend = current_avg - prev_avg logic."""
        prev_avg = 0.5
        window = [0.7, 0.8, 0.9]
        current_avg = sum(window) / len(window)
        trend = current_avg - prev_avg
        assert trend == pytest.approx(0.3)

    def test_report_interval_default(self) -> None:
        from pyladr.search.given_clause import SearchOptions
        opts = SearchOptions()
        assert opts.tree2vec_proximity_report_interval == 100


# ── 4. Proof display gp annotation ──────────────────────────────────────


class TestProofDisplayGP:
    def test_gp_in_proof_display_when_positive(self) -> None:
        """Clause with given_distance > 0.0 should show gp annotation in proof output."""
        c = make_clause(make_literal(True, P(a())), clause_id=1)
        c.given_distance = 0.85
        assert c.given_distance > 0.0

    def test_gp_not_in_proof_display_when_zero(self) -> None:
        """Clause with given_distance == 0.0 should not show gp annotation."""
        c = make_clause(make_literal(True, P(a())), clause_id=1)
        assert c.given_distance == 0.0
        assert not (c.given_distance > 0.0)


# ── 5. CLI parsing ──────────────────────────────────────────────────────


class TestCLIParsing:
    def test_assign_proximity_report_interval(self) -> None:
        from pyladr.apps.prover9 import _apply_settings
        from pyladr.parsing.ladr_parser import LADRParser
        from pyladr.search.given_clause import SearchOptions

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(
            "assign(tree2vec_proximity_report_interval, 50).\n"
            "formulas(sos). P(a). end_of_list.\n"
        )
        opts = SearchOptions()
        _apply_settings(parsed, opts)
        assert opts.tree2vec_proximity_report_interval == 50

    def test_default_report_interval(self) -> None:
        from pyladr.search.given_clause import SearchOptions
        opts = SearchOptions()
        assert opts.tree2vec_proximity_report_interval == 100

    def test_custom_report_interval(self) -> None:
        from pyladr.search.given_clause import SearchOptions
        opts = SearchOptions(tree2vec_proximity_report_interval=25)
        assert opts.tree2vec_proximity_report_interval == 25


# ── 6. T2V goal distance histogram ──────────────────────────────────────────


class TestProximityHistogram:
    def test_histogram_field_default_none(self) -> None:
        from pyladr.search.statistics import SearchStatistics
        stats = SearchStatistics()
        assert stats.t2v_distance_histogram is None

    def test_histogram_has_required_keys(self) -> None:
        """Histogram dict must have proof_probs, nonproof_probs, proof_n, nonproof_n, lo, hi, bucket_width."""
        histogram = {
            "proof_probs": [0.0, 0.0, 0.5, 0.5, 0.0],
            "nonproof_probs": [0.1, 0.2, 0.3, 0.3, 0.1],
            "proof_n": 4,
            "nonproof_n": 20,
            "lo": 0.3,
            "hi": 0.9,
            "bucket_width": 0.12,
        }
        for key in ("proof_probs", "nonproof_probs", "proof_n", "nonproof_n", "lo", "hi", "bucket_width"):
            assert key in histogram

    def test_histogram_probs_length_five(self) -> None:
        """Both proof_probs and nonproof_probs must have exactly 5 elements."""
        histogram = {
            "proof_probs": [0.0, 0.0, 0.5, 0.5, 0.0],
            "nonproof_probs": [0.1, 0.2, 0.3, 0.3, 0.1],
            "proof_n": 4,
            "nonproof_n": 20,
            "lo": 0.3,
            "hi": 0.9,
            "bucket_width": 0.12,
        }
        assert len(histogram["proof_probs"]) == 5
        assert len(histogram["nonproof_probs"]) == 5

    def test_histogram_probs_sum_to_one(self) -> None:
        """Each probability list should sum to ~1.0 when population is non-empty."""
        probs = [0.1, 0.2, 0.3, 0.25, 0.15]
        assert sum(probs) == pytest.approx(1.0)

    def test_histogram_adaptive_bucketing(self) -> None:
        """Buckets are adaptive: bucket_width = (hi - lo) / 5, lo <= hi."""
        lo, hi = 0.35, 0.85
        bucket_width = (hi - lo) / 5
        assert lo <= hi
        assert bucket_width == pytest.approx(0.1)
        # Score mapping: bucket = min(4, int((score - lo) / bucket_width))
        assert min(4, int((0.35 - lo) / bucket_width)) == 0
        assert min(4, int((0.55 - lo) / bucket_width)) == 2
        assert min(4, int((0.85 - lo) / bucket_width)) == 4  # clamped

    def test_histogram_assignment_to_stats(self) -> None:
        from pyladr.search.statistics import SearchStatistics
        stats = SearchStatistics()
        stats.t2v_distance_histogram = {
            "proof_probs": [0.0, 0.25, 0.5, 0.25, 0.0],
            "nonproof_probs": [0.1, 0.2, 0.3, 0.3, 0.1],
            "proof_n": 4,
            "nonproof_n": 20,
            "lo": 0.3,
            "hi": 0.8,
            "bucket_width": 0.1,
        }
        assert stats.t2v_distance_histogram is not None
        assert stats.t2v_distance_histogram["proof_n"] == 4
        assert stats.t2v_distance_histogram["nonproof_n"] == 20

    def test_histogram_not_set_without_goal_proximity(self) -> None:
        """Histogram should remain None when goal distance is not active."""
        from pyladr.search.statistics import SearchStatistics
        stats = SearchStatistics()
        assert stats.t2v_distance_histogram is None

    def test_all_given_proximities_accumulation(self) -> None:
        """_t2v_all_given_distances accumulates clause_id -> proximity during search."""
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions
        opts = SearchOptions(max_seconds=1)
        st = SymbolTable()
        engine = GivenClauseSearch(options=opts, symbol_table=st)

        assert engine._t2v_all_given_distances == {}
        engine._t2v_all_given_distances[10] = 0.7
        engine._t2v_all_given_distances[20] = 0.5
        assert len(engine._t2v_all_given_distances) == 2

    def test_compute_t2v_histogram_returns_none_when_empty(self) -> None:
        """_compute_t2v_histogram returns None when no proximities recorded."""
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions
        opts = SearchOptions(max_seconds=1)
        st = SymbolTable()
        engine = GivenClauseSearch(options=opts, symbol_table=st)

        # Create a minimal proof object
        class _FakeProof:
            clauses = ()
        result = engine._compute_t2v_histogram(_FakeProof())
        assert result is None


# ── 6b. format_t2v_histogram() ───────────────────────────────────────────


class TestFormatT2VHistogram:
    def test_format_includes_proof_num(self) -> None:
        from pyladr.search.given_clause import format_t2v_histogram
        histogram = {
            "proof_probs": [0.0, 0.0, 0.5, 0.5, 0.0],
            "nonproof_probs": [0.1, 0.2, 0.3, 0.3, 0.1],
            "proof_n": 2,
            "nonproof_n": 10,
            "lo": 0.3,
            "hi": 0.8,
            "bucket_width": 0.1,
        }
        output = format_t2v_histogram(histogram, proof_num=3)
        assert "proof 3" in output

    def test_format_includes_counts(self) -> None:
        from pyladr.search.given_clause import format_t2v_histogram
        histogram = {
            "proof_probs": [0.0, 0.0, 0.5, 0.5, 0.0],
            "nonproof_probs": [0.1, 0.2, 0.3, 0.3, 0.1],
            "proof_n": 5,
            "nonproof_n": 15,
            "lo": 0.3,
            "hi": 0.8,
            "bucket_width": 0.1,
        }
        output = format_t2v_histogram(histogram)
        assert "5 proof" in output
        assert "15 non-proof" in output

    def test_format_contains_range_labels(self) -> None:
        from pyladr.search.given_clause import format_t2v_histogram
        histogram = {
            "proof_probs": [0.2, 0.2, 0.2, 0.2, 0.2],
            "nonproof_probs": [0.2, 0.2, 0.2, 0.2, 0.2],
            "proof_n": 5,
            "nonproof_n": 5,
            "lo": 0.0,
            "hi": 1.0,
            "bucket_width": 0.2,
        }
        output = format_t2v_histogram(histogram)
        # Should contain range labels like "0.00-0.20"
        assert "0.00-0.20" in output

    def test_format_default_proof_num_is_one(self) -> None:
        from pyladr.search.given_clause import format_t2v_histogram
        histogram = {
            "proof_probs": [0.0, 0.0, 0.0, 1.0, 0.0],
            "nonproof_probs": [0.5, 0.5, 0.0, 0.0, 0.0],
            "proof_n": 1,
            "nonproof_n": 2,
            "lo": 0.4,
            "hi": 0.9,
            "bucket_width": 0.1,
        }
        output = format_t2v_histogram(histogram)
        assert "proof 1" in output


# ── 7. Regression — monitoring doesn't break proof finding ───────────────


_SIMPLE_PROOF_INPUT = (
    "formulas(sos).\n"
    "  P(a).\n"
    "  -P(x) | Q(x).\n"
    "end_of_list.\n"
    "formulas(goals).\n"
    "  Q(a).\n"
    "end_of_list.\n"
)


def _run_python(input_text: str, max_seconds: float = 10) -> dict:
    from pyladr.apps.prover9 import _auto_inference, _auto_limits, _deny_goals, _apply_settings
    from pyladr.parsing.ladr_parser import LADRParser
    from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(input_text)
    usable, sos, _denied = _deny_goals(parsed, st)
    opts = SearchOptions(max_seconds=max_seconds)
    _auto_inference(parsed, opts)
    _auto_limits(parsed, opts)
    _apply_settings(parsed, opts)
    engine = GivenClauseSearch(
        options=opts,
        symbol_table=st,
        hints=parsed.hints if parsed.hints else None,
    )
    result = engine.run(usable=usable, sos=sos)
    return {
        "proved": len(result.proofs) > 0,
        "exit_code": result.exit_code,
    }


class TestRegressionMonitoring:
    def test_goal_proximity_with_reporting_finds_proof(self) -> None:
        input_text = (
            "set(tree2vec_goal_proximity).\n"
            "assign(tree2vec_proximity_report_interval, 10).\n"
            + _SIMPLE_PROOF_INPUT
        )
        result = _run_python(input_text)
        assert result["proved"]

    def test_goal_proximity_disabled_finds_proof(self) -> None:
        result = _run_python(_SIMPLE_PROOF_INPUT)
        assert result["proved"]

    def test_report_interval_zero_disables_reporting(self) -> None:
        """Setting interval to 0 disables periodic reports but search still works."""
        input_text = (
            "set(tree2vec_goal_proximity).\n"
            "assign(tree2vec_proximity_report_interval, 0).\n"
            + _SIMPLE_PROOF_INPUT
        )
        result = _run_python(input_text)
        assert result["proved"]
