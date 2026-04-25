"""Unit tests for per-given-clause inference tracking.

Tests the new inference tracking functionality added to SearchStatistics:
- begin_given() / record_generated() / get_given_inference_count() / top_given_clauses()
- given_inference_counts dict initialization and behavior
- Integration with existing statistics (zero regression)
- Edge cases (zero inferences, missing IDs, large counts)
- End-to-end tracking through GivenClauseSearch
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.search.statistics import SearchStatistics


# ── Helpers ──────────────────────────────────────────────────────────────────


def _const(symnum: int) -> Term:
    return get_rigid_term(symnum, 0)


def _func(symnum: int, *args: Term) -> Term:
    return get_rigid_term(symnum, len(args), args)


def _pos_lit(atom: Term) -> tuple[bool, Term]:
    return (True, atom)


def _neg_lit(atom: Term) -> tuple[bool, Term]:
    return (False, atom)


def _make_clause_from_lits(*lits: tuple[bool, Term], id: int = 0) -> Clause:
    return Clause(
        literals=tuple(Literal(sign=s, atom=a) for s, a in lits),
        id=id,
        justification=(Justification(just_type=JustType.INPUT),),
    )


# Symbol IDs
A, B, C_SYM = 1, 2, 3
P, Q = 20, 21


# ── SearchStatistics Inference Tracking Tests ───────────────────────────────


class TestInferenceTrackingInit:
    """Test initialization of inference tracking fields."""

    def test_given_inference_counts_default_empty(self) -> None:
        """given_inference_counts starts as empty dict."""
        stats = SearchStatistics()
        assert stats.given_inference_counts == {}
        assert isinstance(stats.given_inference_counts, dict)

    def test_current_given_id_default_zero(self) -> None:
        """_current_given_id starts at 0."""
        stats = SearchStatistics()
        assert stats._current_given_id == 0

    def test_separate_instances_have_independent_dicts(self) -> None:
        """Each instance gets its own dict (default_factory correctness)."""
        stats1 = SearchStatistics()
        stats2 = SearchStatistics()
        stats1.given_inference_counts[1] = 5
        assert 1 not in stats2.given_inference_counts

    def test_slots_compatibility(self) -> None:
        """New fields work with slots=True dataclass."""
        stats = SearchStatistics()
        # Should be able to set/get both new fields without AttributeError
        stats.given_inference_counts[42] = 10
        stats._current_given_id = 42
        assert stats.given_inference_counts[42] == 10
        assert stats._current_given_id == 42


class TestBeginGiven:
    """Test begin_given() method."""

    def test_sets_current_given_id(self) -> None:
        stats = SearchStatistics()
        stats.begin_given(7)
        assert stats._current_given_id == 7

    def test_initializes_counter_to_zero(self) -> None:
        stats = SearchStatistics()
        stats.begin_given(7)
        assert stats.given_inference_counts[7] == 0

    def test_multiple_begins_track_separately(self) -> None:
        stats = SearchStatistics()
        stats.begin_given(1)
        stats.begin_given(2)
        stats.begin_given(3)
        assert 1 in stats.given_inference_counts
        assert 2 in stats.given_inference_counts
        assert 3 in stats.given_inference_counts

    def test_begin_overwrites_previous_count(self) -> None:
        """If begin_given is called again for the same ID, it resets to 0."""
        stats = SearchStatistics()
        stats.begin_given(5)
        stats.given_inference_counts[5] = 42
        stats.begin_given(5)
        assert stats.given_inference_counts[5] == 0


class TestRecordGenerated:
    """Test record_generated() method."""

    def test_increments_global_generated(self) -> None:
        stats = SearchStatistics()
        stats.begin_given(1)
        stats.record_generated()
        assert stats.generated == 1

    def test_increments_per_given_count(self) -> None:
        stats = SearchStatistics()
        stats.begin_given(1)
        stats.record_generated()
        stats.record_generated()
        stats.record_generated()
        assert stats.given_inference_counts[1] == 3

    def test_tracks_across_multiple_givens(self) -> None:
        stats = SearchStatistics()

        stats.begin_given(1)
        stats.record_generated()
        stats.record_generated()

        stats.begin_given(2)
        stats.record_generated()
        stats.record_generated()
        stats.record_generated()

        stats.begin_given(3)
        # No inferences for given 3

        assert stats.given_inference_counts[1] == 2
        assert stats.given_inference_counts[2] == 3
        assert stats.given_inference_counts[3] == 0
        assert stats.generated == 5

    def test_no_current_given_still_increments_global(self) -> None:
        """If _current_given_id is 0, global generated still increments."""
        stats = SearchStatistics()
        stats.record_generated()
        assert stats.generated == 1
        # No per-given tracking when no given is active
        assert stats.given_inference_counts == {}

    def test_large_count(self) -> None:
        stats = SearchStatistics()
        stats.begin_given(1)
        for _ in range(10_000):
            stats.record_generated()
        assert stats.given_inference_counts[1] == 10_000
        assert stats.generated == 10_000


class TestGetGivenInferenceCount:
    """Test get_given_inference_count() method."""

    def test_returns_count_for_tracked_clause(self) -> None:
        stats = SearchStatistics()
        stats.begin_given(5)
        stats.record_generated()
        stats.record_generated()
        assert stats.get_given_inference_count(5) == 2

    def test_returns_zero_for_unknown_clause(self) -> None:
        stats = SearchStatistics()
        assert stats.get_given_inference_count(999) == 0

    def test_returns_zero_for_given_with_no_inferences(self) -> None:
        stats = SearchStatistics()
        stats.begin_given(1)
        assert stats.get_given_inference_count(1) == 0


class TestTopGivenClauses:
    """Test top_given_clauses() method."""

    def test_returns_sorted_descending(self) -> None:
        stats = SearchStatistics()
        stats.begin_given(1)
        stats.record_generated()

        stats.begin_given(2)
        for _ in range(5):
            stats.record_generated()

        stats.begin_given(3)
        for _ in range(3):
            stats.record_generated()

        top = stats.top_given_clauses(10)
        assert top[0] == (2, 5)
        assert top[1] == (3, 3)
        assert top[2] == (1, 1)

    def test_respects_n_limit(self) -> None:
        stats = SearchStatistics()
        for i in range(1, 20):
            stats.begin_given(i)
            for _ in range(i):
                stats.record_generated()

        top3 = stats.top_given_clauses(3)
        assert len(top3) == 3
        assert top3[0] == (19, 19)
        assert top3[1] == (18, 18)
        assert top3[2] == (17, 17)

    def test_empty_stats(self) -> None:
        stats = SearchStatistics()
        assert stats.top_given_clauses(10) == []

    def test_default_n_is_10(self) -> None:
        stats = SearchStatistics()
        for i in range(1, 20):
            stats.begin_given(i)
            stats.record_generated()
        top = stats.top_given_clauses()
        assert len(top) == 10


# ── Regression: Existing SearchStatistics Behavior Unchanged ────────────────


class TestExistingStatisticsRegression:
    """Ensure existing SearchStatistics behavior is completely unchanged."""

    def test_initial_zeros(self) -> None:
        stats = SearchStatistics()
        assert stats.given == 0
        assert stats.generated == 0
        assert stats.kept == 0
        assert stats.subsumed == 0
        assert stats.proofs == 0
        assert stats.back_subsumed == 0
        assert stats.demodulated == 0
        assert stats.back_demodulated == 0
        assert stats.new_demodulators == 0
        assert stats.unit_conflicts == 0
        assert stats.penalty_weight_adjusted == 0

    def test_counter_increment(self) -> None:
        stats = SearchStatistics()
        stats.given += 1
        stats.kept += 5
        assert stats.given == 1
        assert stats.kept == 5

    def test_report_format_unchanged(self) -> None:
        """Report format must match C fprint_all_stats exactly."""
        stats = SearchStatistics()
        stats.given = 12
        stats.generated = 118
        stats.kept = 23
        stats.proofs = 1
        report = stats.report()
        assert "given=12" in report
        assert "generated=118" in report
        assert "kept=23" in report
        assert "proofs=1" in report
        # Verify report does NOT include inference tracking data
        assert "given_inference" not in report
        assert "current_given" not in report

    def test_report_format_no_new_fields(self) -> None:
        """Report should only contain the original C-compatible fields."""
        stats = SearchStatistics()
        stats.begin_given(1)
        for _ in range(50):
            stats.record_generated()
        report = stats.report()
        # Should have exactly the C-compatible format
        parts = report.split(", ")
        field_names = [p.split("=")[0] for p in parts]
        assert field_names == [
            "given", "generated", "kept", "subsumed",
            "back_subsumed", "proofs", "time",
        ]

    def test_timing(self) -> None:
        stats = SearchStatistics()
        stats.start()
        assert stats.elapsed_seconds() >= 0.0
        assert stats.search_seconds() >= 0.0

    def test_record_generated_keeps_global_count_accurate(self) -> None:
        """record_generated() must maintain exact same global count as old stats.generated += 1."""
        stats = SearchStatistics()
        stats.begin_given(1)
        for _ in range(100):
            stats.record_generated()
        assert stats.generated == 100


# ── Integration: GivenClauseSearch Inference Tracking ───────────────────────


from pyladr.search.given_clause import (
    ExitCode,
    GivenClauseSearch,
    SearchOptions,
)


class TestInferenceTrackingIntegration:
    """Test inference tracking through actual GivenClauseSearch runs."""

    def test_simple_proof_tracks_inferences(self) -> None:
        """P(a) and -P(a) should track inferences per given clause."""
        a = _const(A)
        pa = _func(P, a)
        c1 = _make_clause_from_lits(_pos_lit(pa))
        c2 = _make_clause_from_lits(_neg_lit(pa))

        opts = SearchOptions(print_given=False, print_kept=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1, c2])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        # Verify inference counts were tracked
        assert len(result.stats.given_inference_counts) > 0
        # Total per-given counts should equal global generated
        total_per_given = sum(result.stats.given_inference_counts.values())
        assert total_per_given == result.stats.generated

    def test_two_step_proof_tracks_inferences(self) -> None:
        """Multi-step proof should track inferences for each given clause."""
        a = _const(A)
        x = get_variable_term(0)
        p_x = _func(P, x)
        p_a = _func(P, a)
        q_x = _func(Q, x)
        q_a = _func(Q, a)

        c1 = _make_clause_from_lits(_pos_lit(p_a))
        c2 = _make_clause_from_lits(_neg_lit(p_x), _pos_lit(q_x))
        c3 = _make_clause_from_lits(_neg_lit(q_a))

        opts = SearchOptions(print_given=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1, c2, c3])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        # Each given clause should have an entry
        assert len(result.stats.given_inference_counts) == result.stats.given
        # Global generated == sum of per-given
        total_per_given = sum(result.stats.given_inference_counts.values())
        assert total_per_given == result.stats.generated

    def test_sos_empty_tracks_all_givens(self) -> None:
        """Non-provable problem: all given clauses should be tracked."""
        a = _const(A)
        b = _const(B)
        pa = _func(P, a)
        pb = _func(P, b)

        c1 = _make_clause_from_lits(_pos_lit(pa))
        c2 = _make_clause_from_lits(_pos_lit(pb))

        opts = SearchOptions(max_given=10, print_given=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1, c2])

        # All given clauses tracked
        assert len(result.stats.given_inference_counts) == result.stats.given
        # Invariant: sum of per-given == global generated
        total_per_given = sum(result.stats.given_inference_counts.values())
        assert total_per_given == result.stats.generated

    def test_top_given_clauses_after_search(self) -> None:
        """top_given_clauses() returns valid data after a search run."""
        a = _const(A)
        x = get_variable_term(0)
        p_x = _func(P, x)
        p_a = _func(P, a)
        q_x = _func(Q, x)
        q_a = _func(Q, a)

        c1 = _make_clause_from_lits(_pos_lit(p_a))
        c2 = _make_clause_from_lits(_neg_lit(p_x), _pos_lit(q_x))
        c3 = _make_clause_from_lits(_neg_lit(q_a))

        opts = SearchOptions(print_given=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1, c2, c3])

        top = result.stats.top_given_clauses(5)
        assert len(top) > 0
        # Should be sorted descending by count
        for i in range(len(top) - 1):
            assert top[i][1] >= top[i + 1][1]

    def test_exit_codes_unchanged(self) -> None:
        """Exit codes must remain identical to pre-enhancement behavior."""
        # SOS_EMPTY
        search = GivenClauseSearch()
        result = search.run(usable=[], sos=[])
        assert result.exit_code == ExitCode.SOS_EMPTY_EXIT

        # MAX_PROOFS
        a = _const(A)
        pa = _func(P, a)
        c1 = _make_clause_from_lits(_pos_lit(pa))
        c2 = _make_clause_from_lits(_neg_lit(pa))
        opts = SearchOptions(print_given=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_proof_clauses_unchanged(self) -> None:
        """Proof structure must be identical to pre-enhancement behavior."""
        a = _const(A)
        pa = _func(P, a)
        c1 = _make_clause_from_lits(_pos_lit(pa))
        c2 = _make_clause_from_lits(_neg_lit(pa))

        opts = SearchOptions(print_given=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1, c2])

        assert len(result.proofs) == 1
        proof = result.proofs[0]
        proof_ids = {c.id for c in proof.clauses}
        assert c1.id in proof_ids
        assert c2.id in proof_ids

    def test_immediate_empty_clause_no_tracking(self) -> None:
        """Empty clause in SOS should find proof before any inferences."""
        empty = Clause(
            literals=(),
            justification=(Justification(just_type=JustType.INPUT),),
        )
        search = GivenClauseSearch()
        result = search.run(sos=[empty])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        # No given clauses were processed for inferences
        assert result.stats.generated == 0

    def test_factoring_inferences_tracked(self) -> None:
        """Factoring inferences should be attributed to the given clause."""
        from pyladr.core.symbol import SymbolTable

        st = SymbolTable()
        p_sn = st.str_to_sn("P", 1)
        a_sn = st.str_to_sn("a", 0)

        a = get_rigid_term(a_sn, 0)
        x = get_variable_term(0)
        y = get_variable_term(1)
        p_x = get_rigid_term(p_sn, 1, (x,))
        p_y = get_rigid_term(p_sn, 1, (y,))
        p_a = get_rigid_term(p_sn, 1, (a,))

        c1 = _make_clause_from_lits(_pos_lit(p_x), _pos_lit(p_y))
        c2 = _make_clause_from_lits(_neg_lit(p_a))

        opts = SearchOptions(factoring=True, print_given=False)
        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(sos=[c1, c2])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        # Invariant: sum == global
        total = sum(result.stats.given_inference_counts.values())
        assert total == result.stats.generated
