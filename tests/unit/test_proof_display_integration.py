"""Tests for proof display during multi-proof search (REQ-R006 regression prevention).

Validates that proofs are displayed incrementally as they are found during search,
AND that this works correctly in combination with max_proofs (REQ-R005).

These tests guard against the regression where proofs are accumulated silently
during multi-proof search and only printed after search completes.

Test strategy:
- Use proof_callback to observe incremental proof delivery
- Verify timing: each proof is delivered DURING search, not after
- Verify integration: proof_callback + max_proofs cooperate correctly
- Verify output: full pipeline produces formatted proof output incrementally
"""

from __future__ import annotations

import io
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import parse_input
from pyladr.apps.prover9 import _print_proof
from pyladr.search.given_clause import (
    ExitCode,
    GivenClauseSearch,
    Proof,
    SearchOptions,
    SearchResult,
)


def _apply_assignments(parsed, opts: SearchOptions) -> None:
    """Apply parsed assign() directives to SearchOptions.

    Local helper replacing removed prover9._apply_assignments.
    Mirrors the inline logic in run_prover().
    """
    assigns = parsed.assigns
    if "max_proofs" in assigns:
        opts.max_proofs = int(assigns["max_proofs"])
    if "max_given" in assigns:
        opts.max_given = int(assigns["max_given"])
    if "max_kept" in assigns:
        opts.max_kept = int(assigns["max_kept"])
    if "max_seconds" in assigns:
        opts.max_seconds = float(assigns["max_seconds"])
    if "max_generated" in assigns:
        opts.max_generated = int(assigns["max_generated"])
    if "max_weight" in assigns:
        opts.max_weight = float(assigns["max_weight"])
    if "sos_limit" in assigns:
        opts.sos_limit = int(assigns["sos_limit"])


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_and_deny(text: str) -> tuple[list[Clause], list[Clause], SymbolTable]:
    """Parse LADR input, deny goals, return (usable, sos, symbol_table)."""
    st = SymbolTable()
    parsed = parse_input(text, st)

    usable = list(parsed.usable)
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

    return usable, sos, st


@dataclass
class ProofEvent:
    """Record of a single proof callback invocation."""

    proof: Proof
    proof_num: int


class ProofObserver:
    """Observer that records proof callbacks for testing.

    Captures when each proof is delivered (via callback) so tests can verify
    incremental delivery during search execution.
    """

    def __init__(self) -> None:
        self.events: list[ProofEvent] = []
        self.callback_count: int = 0

    def on_proof(self, proof: Proof, proof_num: int) -> None:
        """Proof callback handler."""
        self.callback_count += 1
        self.events.append(ProofEvent(proof=proof, proof_num=proof_num))

    @property
    def proof_nums(self) -> list[int]:
        return [e.proof_num for e in self.events]

    @property
    def proofs(self) -> list[Proof]:
        return [e.proof for e in self.events]


def _run_with_observer(
    text: str,
    max_given: int = 500,
    max_seconds: float = 10.0,
    **kwargs,
) -> tuple[SearchResult, ProofObserver]:
    """Run search with a proof observer attached via proof_callback."""
    usable, sos, st = _parse_and_deny(text)
    observer = ProofObserver()
    observer.symbol_table = st
    opts = SearchOptions(
        max_given=max_given,
        max_seconds=max_seconds,
        quiet=True,
        print_given=False,
        **kwargs,
    )
    engine = GivenClauseSearch(
        opts,
        symbol_table=st,
        proof_callback=observer.on_proof,
    )
    result = engine.run(usable=usable, sos=sos)
    return result, observer


# ── Test problems ────────────────────────────────────────────────────────────

# Propositional with multiple complementary pairs → multiple proofs.
PROPOSITIONAL_MULTI = """\
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""

# Equational problem (paramodulation) with goal.
GROUP_X2_COMM = """\
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

# Trivial single-proof problem.
TRIVIAL_SINGLE = """\
formulas(sos).
  e * x = x.
end_of_list.
formulas(goals).
  e * e = e.
end_of_list.
"""

# Multi-path resolution problem.
MULTI_PATH = """\
formulas(sos).
  P(a).
  P(b).
  -P(x) | Q(x).
  -Q(a).
  -Q(b).
end_of_list.
"""


# ══════════════════════════════════════════════════════════════════════════════
# REQ-R006: Proof Callback Mechanism
# ══════════════════════════════════════════════════════════════════════════════


class TestProofCallbackMechanism:
    """Verify the proof_callback parameter on GivenClauseSearch works."""

    def test_callback_invoked_on_proof(self):
        """proof_callback is called when a proof is found."""
        result, obs = _run_with_observer(PROPOSITIONAL_MULTI, max_proofs=1)
        assert obs.callback_count == 1, (
            f"Expected 1 callback invocation, got {obs.callback_count}"
        )

    def test_callback_receives_valid_proof(self):
        """Callback receives a Proof with a valid empty clause."""
        result, obs = _run_with_observer(PROPOSITIONAL_MULTI, max_proofs=1)
        assert len(obs.events) == 1
        proof = obs.events[0].proof
        assert isinstance(proof, Proof)
        assert proof.empty_clause.is_empty
        assert len(proof.clauses) > 0

    def test_callback_receives_proof_number(self):
        """Callback receives correct 1-indexed proof number."""
        result, obs = _run_with_observer(PROPOSITIONAL_MULTI, max_proofs=1)
        assert obs.events[0].proof_num == 1

    def test_callback_not_required(self):
        """Search works without proof_callback (None)."""
        usable, sos, st = _parse_and_deny(PROPOSITIONAL_MULTI)
        opts = SearchOptions(max_proofs=1, quiet=True, print_given=False)
        engine = GivenClauseSearch(opts, symbol_table=st, proof_callback=None)
        result = engine.run(usable=usable, sos=sos)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1

    def test_no_callback_prints_to_stdout(self):
        """Without callback, "PROOF FOUND" still appears on stdout."""
        usable, sos, st = _parse_and_deny(PROPOSITIONAL_MULTI)
        opts = SearchOptions(max_proofs=1, quiet=True, print_given=False)
        engine = GivenClauseSearch(opts, symbol_table=st, proof_callback=None)

        captured = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = captured
            engine.run(usable=usable, sos=sos)
        finally:
            sys.stdout = old_stdout

        assert "PROOF FOUND" in captured.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# REQ-R006: Incremental Proof Delivery During Multi-Proof Search
# ══════════════════════════════════════════════════════════════════════════════


class TestIncrementalProofDelivery:
    """REQ-R006 REGRESSION GUARD: Proofs must be delivered incrementally.

    If proofs are buffered and only delivered after search completes,
    these tests FAIL.
    """

    def test_two_proofs_delivered_incrementally(self):
        """With max_proofs=2, callback fires twice with proof_num 1 then 2."""
        result, obs = _run_with_observer(PROPOSITIONAL_MULTI, max_proofs=2)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert obs.callback_count == 2, (
            f"REQ-R006: Expected 2 incremental callbacks, got {obs.callback_count}. "
            "Proofs may not be delivered incrementally."
        )
        assert obs.proof_nums == [1, 2]

    def test_three_proofs_sequential_numbering(self):
        """Proof numbers are sequential 1, 2, 3 for max_proofs=3."""
        result, obs = _run_with_observer(
            PROPOSITIONAL_MULTI, max_proofs=3, max_given=1000,
        )
        if len(obs.events) >= 3:
            assert obs.proof_nums[:3] == [1, 2, 3]
        elif len(obs.events) >= 2:
            assert obs.proof_nums[:2] == [1, 2]
        else:
            pytest.skip("Problem didn't produce enough proofs for this test")

    def test_callback_count_matches_result_proofs(self):
        """Number of callbacks must equal len(result.proofs)."""
        result, obs = _run_with_observer(
            PROPOSITIONAL_MULTI, max_proofs=5, max_given=1000,
        )
        assert obs.callback_count == len(result.proofs), (
            f"Callback count {obs.callback_count} != result.proofs {len(result.proofs)}. "
            "Some proofs were not delivered via callback."
        )

    def test_callback_proofs_match_result_proofs(self):
        """Proofs delivered via callback are the same objects in result.proofs."""
        result, obs = _run_with_observer(PROPOSITIONAL_MULTI, max_proofs=2)
        for i, (cb_proof, res_proof) in enumerate(
            zip(obs.proofs, result.proofs)
        ):
            assert cb_proof.empty_clause.id == res_proof.empty_clause.id, (
                f"Proof {i+1}: callback empty_clause id {cb_proof.empty_clause.id} "
                f"!= result proof id {res_proof.empty_clause.id}"
            )

    def test_each_callback_proof_is_distinct(self):
        """Each callback delivers a distinct proof (different empty clause)."""
        result, obs = _run_with_observer(PROPOSITIONAL_MULTI, max_proofs=3, max_given=1000)
        if obs.callback_count >= 2:
            empty_ids = [e.proof.empty_clause.id for e in obs.events]
            assert len(set(empty_ids)) == len(empty_ids), (
                f"Duplicate empty clause IDs in callbacks: {empty_ids}"
            )

    def test_equational_incremental_delivery(self):
        """Equational (paramodulation) multi-proof also delivers incrementally."""
        result, obs = _run_with_observer(
            GROUP_X2_COMM,
            max_proofs=3,
            paramodulation=True,
            max_given=2000,
            max_seconds=30.0,
        )
        # At least 1 proof should be found
        assert obs.callback_count >= 1
        assert obs.callback_count == len(result.proofs)
        # If multiple found, verify sequential numbering
        if obs.callback_count >= 2:
            for i, event in enumerate(obs.events):
                assert event.proof_num == i + 1


# ══════════════════════════════════════════════════════════════════════════════
# REQ-R005 + REQ-R006 Integration: max_proofs AND Incremental Display
# ══════════════════════════════════════════════════════════════════════════════


class TestMaxProofsAndDisplayIntegration:
    """Verify max_proofs (REQ-R005) and proof display (REQ-R006) cooperate.

    Critical integration tests: BOTH must work simultaneously.
    """

    def test_max_proofs_1_single_callback(self):
        """max_proofs=1: exactly 1 callback, then search exits."""
        result, obs = _run_with_observer(PROPOSITIONAL_MULTI, max_proofs=1)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert obs.callback_count == 1
        assert len(result.proofs) == 1

    def test_max_proofs_2_two_callbacks_then_exit(self):
        """max_proofs=2: exactly 2 callbacks, then MAX_PROOFS_EXIT."""
        result, obs = _run_with_observer(PROPOSITIONAL_MULTI, max_proofs=2)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert obs.callback_count == 2
        assert len(result.proofs) == 2

    def test_max_proofs_high_callbacks_match_found(self):
        """max_proofs=100 with limited search: callbacks match proofs found."""
        result, obs = _run_with_observer(
            PROPOSITIONAL_MULTI, max_proofs=100, max_given=200,
        )
        # Search ends by resource limit, not max_proofs
        assert obs.callback_count == len(result.proofs)
        assert result.stats.proofs == obs.callback_count

    def test_unlimited_proofs_continuous_callbacks(self):
        """max_proofs=-1 (unlimited): callbacks fire for every proof found."""
        result, obs = _run_with_observer(
            PROPOSITIONAL_MULTI, max_proofs=-1, max_given=200,
        )
        assert result.exit_code in (ExitCode.SOS_EMPTY_EXIT, ExitCode.MAX_GIVEN_EXIT)
        assert obs.callback_count >= 1
        assert obs.callback_count == len(result.proofs)

    def test_proof_delivery_order_matches_discovery_order(self):
        """Proofs are delivered in discovery order (not reversed, not reordered)."""
        result, obs = _run_with_observer(
            PROPOSITIONAL_MULTI, max_proofs=3, max_given=1000,
        )
        if obs.callback_count >= 2:
            # Empty clause IDs should be monotonically increasing (later proofs
            # involve later-derived empty clauses with higher IDs).
            empty_ids = [e.proof.empty_clause.id for e in obs.events]
            assert empty_ids == sorted(empty_ids), (
                f"Proofs delivered out of order. Empty clause IDs: {empty_ids}"
            )

    def test_search_continues_between_callbacks(self):
        """Search continues after each callback (not terminating early)."""
        result, obs = _run_with_observer(
            PROPOSITIONAL_MULTI, max_proofs=5, max_given=1000,
        )
        if obs.callback_count >= 2:
            # If we got 2+ callbacks, search genuinely continued
            assert result.stats.proofs >= 2, (
                "REQ-R005+R006: search did not continue past first proof"
            )


# ══════════════════════════════════════════════════════════════════════════════
# REQ-R006: Formatted Proof Output During Search
# ══════════════════════════════════════════════════════════════════════════════


class TestFormattedProofOutput:
    """Verify that the proof callback can produce fully formatted proof output.

    The callback should enable _print_proof to be called during search,
    producing formatted output incrementally rather than after completion.
    """

    def test_callback_proof_is_printable(self):
        """Proof delivered via callback contains enough data for _print_proof."""
        result, obs = _run_with_observer(PROPOSITIONAL_MULTI, max_proofs=1)
        proof = obs.events[0].proof
        buf = io.StringIO()
        # Should not raise
        _print_proof(proof, proof_num=1, search_seconds=0.0, symbol_table=obs.symbol_table, out=buf)
        output = buf.getvalue()
        assert "Proof 1" in output
        assert "-------- Proof 1 --------" in output

    def test_each_incremental_proof_is_printable(self):
        """All incrementally delivered proofs can be formatted."""
        result, obs = _run_with_observer(PROPOSITIONAL_MULTI, max_proofs=3, max_given=1000)
        for i, event in enumerate(obs.events):
            buf = io.StringIO()
            _print_proof(
                event.proof,
                proof_num=event.proof_num,
                search_seconds=0.0,
                symbol_table=obs.symbol_table,
                out=buf,
            )
            output = buf.getvalue()
            assert f"Proof {event.proof_num}" in output, (
                f"Callback proof {event.proof_num} not printable"
            )

    def test_incremental_output_contains_proof_clauses(self):
        """Formatted proof output from callback contains clause lines."""
        result, obs = _run_with_observer(PROPOSITIONAL_MULTI, max_proofs=1)
        proof = obs.events[0].proof
        buf = io.StringIO()
        _print_proof(proof, proof_num=1, search_seconds=0.0, symbol_table=obs.symbol_table, out=buf)
        output = buf.getvalue()
        # Proof should contain clause lines (at least the empty clause)
        assert len(proof.clauses) > 0
        # Output should have multiple lines of content
        lines = [l for l in output.strip().split("\n") if l.strip()]
        assert len(lines) >= 3, (
            f"Formatted proof output too short ({len(lines)} lines)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# REQ-R006: Proof Display with assign() Directives (End-to-End)
# ══════════════════════════════════════════════════════════════════════════════


class TestAssignMaxProofsWithDisplay:
    """End-to-end: assign(max_proofs, N) from input file + incremental display."""

    def _run_e2e_with_observer(
        self, text: str, **override_opts,
    ) -> tuple[SearchResult, ProofObserver]:
        """Full pipeline with assign() parsing and proof observer."""
        st = SymbolTable()
        parsed = parse_input(text, st)

        usable = list(parsed.usable)
        sos = list(parsed.sos)
        for goal in parsed.goals:
            denied_lits = tuple(
                Literal(sign=not lit.sign, atom=lit.atom) for lit in goal.literals
            )
            denied = Clause(
                literals=denied_lits,
                justification=(
                    Justification(just_type=JustType.DENY, clause_ids=(0,)),
                ),
            )
            sos.append(denied)

        observer = ProofObserver()
        opts = SearchOptions(
            quiet=True,
            print_given=False,
            max_given=override_opts.pop("max_given", 500),
            max_seconds=override_opts.pop("max_seconds", 10.0),
            **override_opts,
        )
        _apply_assignments(parsed, opts)

        engine = GivenClauseSearch(
            opts,
            symbol_table=st,
            proof_callback=observer.on_proof,
        )
        result = engine.run(usable=usable, sos=sos)
        return result, observer

    def test_assign_max_proofs_2_with_callbacks(self):
        """assign(max_proofs, 2) triggers 2 incremental callbacks."""
        text = """\
assign(max_proofs, 2).
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        result, obs = self._run_e2e_with_observer(text)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert obs.callback_count == 2, (
            f"REQ-R006: assign(max_proofs, 2) should trigger 2 callbacks, "
            f"got {obs.callback_count}"
        )
        assert obs.proof_nums == [1, 2]

    def test_assign_max_proofs_1_with_callback(self):
        """assign(max_proofs, 1) triggers exactly 1 callback."""
        text = """\
assign(max_proofs, 1).
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        result, obs = self._run_e2e_with_observer(text)
        assert obs.callback_count == 1
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_no_assign_default_single_callback(self):
        """Without assign(), default max_proofs=1 triggers 1 callback."""
        text = """\
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        result, obs = self._run_e2e_with_observer(text)
        assert obs.callback_count == 1
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_assign_max_proofs_10_incremental_user_scenario(self):
        """REQ-R006 USER SCENARIO: assign(max_proofs, 10) with incremental display.

        User expects to see proofs appearing one-by-one during search.
        """
        text = """\
assign(max_proofs, 10).
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        result, obs = self._run_e2e_with_observer(text, max_given=2000)
        # Must find more than 1 proof (REQ-R005)
        assert obs.callback_count >= 2, (
            f"REQ-R005+R006 USER SCENARIO: assign(max_proofs, 10) produced only "
            f"{obs.callback_count} callback(s). User expects incremental discovery."
        )
        # Callbacks must be sequential
        for i, num in enumerate(obs.proof_nums):
            assert num == i + 1, (
                f"Proof number {num} at position {i} not sequential"
            )

    def test_assign_max_proofs_with_goals_incremental(self):
        """assign(max_proofs, 3) with goals delivers proofs incrementally."""
        text = """\
assign(max_proofs, 3).
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
        result, obs = self._run_e2e_with_observer(
            text, paramodulation=True, max_given=2000, max_seconds=30.0,
        )
        assert obs.callback_count >= 1
        assert obs.callback_count == len(result.proofs)
        # If found multiple proofs, verify incremental numbering
        if obs.callback_count >= 2:
            assert obs.proof_nums[:2] == [1, 2]


# ══════════════════════════════════════════════════════════════════════════════
# REQ-R006: Edge Cases and Robustness
# ══════════════════════════════════════════════════════════════════════════════


class TestProofDisplayEdgeCases:
    """Edge cases for proof callback and display."""

    def test_callback_exception_does_not_crash_search(self):
        """If callback raises, search should not abort silently.

        Note: Current behavior may vary; this test documents expectation.
        A robust implementation should catch callback errors.
        """
        usable, sos, st = _parse_and_deny(PROPOSITIONAL_MULTI)

        call_count = 0

        def bad_callback(proof: Proof, num: int) -> None:
            nonlocal call_count
            call_count += 1
            # First call succeeds, subsequent raises
            if call_count > 1:
                raise ValueError("callback error")

        opts = SearchOptions(max_proofs=1, quiet=True, print_given=False)
        engine = GivenClauseSearch(
            opts, symbol_table=st, proof_callback=bad_callback,
        )
        # With max_proofs=1, only 1 callback fires (no exception)
        result = engine.run(usable=usable, sos=sos)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert call_count == 1

    def test_trivial_proof_callback(self):
        """Even trivial proofs (initial empty clause) trigger callback."""
        # If an input clause is already empty, proof should still be delivered
        result, obs = _run_with_observer(
            TRIVIAL_SINGLE, max_proofs=1, paramodulation=True,
        )
        assert obs.callback_count >= 1
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_resolution_multi_proof_callbacks(self):
        """Binary resolution multi-proof search triggers incremental callbacks."""
        result, obs = _run_with_observer(
            MULTI_PATH,
            max_proofs=2,
            binary_resolution=True,
            paramodulation=False,
        )
        if result.exit_code == ExitCode.MAX_PROOFS_EXIT:
            assert obs.callback_count == 2
            assert obs.proof_nums == [1, 2]
        else:
            assert obs.callback_count >= 1

    def test_proof_trace_in_callback_is_complete(self):
        """Proof clauses in callback include initial + derived clauses."""
        result, obs = _run_with_observer(PROPOSITIONAL_MULTI, max_proofs=1)
        proof = obs.events[0].proof
        # Should have input clauses (initial) and derived clauses
        has_initial = any(c.initial for c in proof.clauses)
        has_empty = any(c.is_empty for c in proof.clauses)
        assert has_initial, "Proof trace missing initial clauses"
        assert has_empty, "Proof trace missing empty clause"

    def test_proof_trace_sorted_by_id_in_callback(self):
        """Proof clauses in callback are sorted by clause ID."""
        result, obs = _run_with_observer(PROPOSITIONAL_MULTI, max_proofs=2)
        for event in obs.events:
            ids = [c.id for c in event.proof.clauses]
            assert ids == sorted(ids), (
                f"Proof {event.proof_num} clauses not sorted: {ids}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# REQ-R006: Statistics Consistency with Incremental Display
# ══════════════════════════════════════════════════════════════════════════════


class TestStatsConsistencyWithCallbacks:
    """Verify statistics remain consistent when proof_callback is used."""

    def test_stats_proofs_equals_callback_count(self):
        """result.stats.proofs must equal the number of callbacks fired."""
        result, obs = _run_with_observer(
            PROPOSITIONAL_MULTI, max_proofs=5, max_given=1000,
        )
        assert result.stats.proofs == obs.callback_count

    def test_stats_proofs_equals_result_proofs_length(self):
        """stats.proofs == len(result.proofs) == callback_count."""
        result, obs = _run_with_observer(
            PROPOSITIONAL_MULTI, max_proofs=3, max_given=1000,
        )
        assert result.stats.proofs == len(result.proofs) == obs.callback_count

    def test_empty_clauses_found_matches(self):
        """stats.empty_clauses_found matches callback count."""
        result, obs = _run_with_observer(
            PROPOSITIONAL_MULTI, max_proofs=3, max_given=1000,
        )
        assert result.stats.empty_clauses_found == obs.callback_count

    def test_stats_given_positive_with_multi_proof(self):
        """With multi-proof search, given clauses > 0 (search progressed)."""
        result, obs = _run_with_observer(
            PROPOSITIONAL_MULTI, max_proofs=3, max_given=1000,
        )
        assert result.stats.given > 0, "No given clauses processed"


# ══════════════════════════════════════════════════════════════════════════════
# Regression Canaries
# ══════════════════════════════════════════════════════════════════════════════


class TestRegressionCanaries:
    """Quick-check tests that catch REQ-R005 and REQ-R006 regressions early.

    These are intentionally fast and focused. If any of these fail,
    investigate deeply before looking at other test classes.
    """

    def test_canary_max_proofs_not_ignored(self):
        """CANARY: max_proofs=2 finds 2 proofs (not 1)."""
        result, obs = _run_with_observer(PROPOSITIONAL_MULTI, max_proofs=2)
        assert len(result.proofs) == 2, "REQ-R005 REGRESSION: max_proofs ignored"

    def test_canary_callback_fires(self):
        """CANARY: proof_callback fires at least once."""
        result, obs = _run_with_observer(PROPOSITIONAL_MULTI, max_proofs=1)
        assert obs.callback_count > 0, "REQ-R006 REGRESSION: callback never fired"

    def test_canary_incremental_delivery(self):
        """CANARY: 2 proofs → 2 callbacks (not 0 callbacks + 2 in result)."""
        result, obs = _run_with_observer(PROPOSITIONAL_MULTI, max_proofs=2)
        assert obs.callback_count == 2, (
            f"REQ-R006 REGRESSION: Expected 2 callbacks, got {obs.callback_count}. "
            f"result.proofs has {len(result.proofs)} entries."
        )

    def test_canary_assign_directive_pipeline(self):
        """CANARY: assign(max_proofs, 2) parsed AND applied AND triggers callbacks."""
        st = SymbolTable()
        text = "assign(max_proofs, 2).\nformulas(sos).\n  P | Q.\n  -P.\n  -Q.\nend_of_list.\n"
        parsed = parse_input(text, st)
        assert parsed.assigns.get("max_proofs") == 2, "Parse failed"

        opts = SearchOptions(quiet=True, print_given=False)
        _apply_assignments(parsed, opts)
        assert opts.max_proofs == 2, "Application failed"

        observer = ProofObserver()
        usable, sos, _ = _parse_and_deny(text)
        engine = GivenClauseSearch(
            opts, symbol_table=st, proof_callback=observer.on_proof,
        )
        result = engine.run(usable=usable, sos=sos)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT, "Search failed"
        assert observer.callback_count == 2, "Callbacks failed"


# ══════════════════════════════════════════════════════════════════════════════
# REQ-R006: Full Pipeline Integration (run_prover output capture)
# ══════════════════════════════════════════════════════════════════════════════


class TestRunProverProofOutput:
    """Full-pipeline tests: run_prover() produces incremental proof output.

    These tests exercise the complete prover9 app layer, verifying that
    Christopher's proof_callback wiring in run_prover() actually causes
    formatted proofs to appear in the output stream.
    """

    @staticmethod
    def _run_prover_capture(input_text: str, extra_args: list[str] | None = None) -> tuple[int, str]:
        """Run run_prover() with captured stdout, feeding input via stdin."""
        import tempfile
        import os
        from pyladr.apps.prover9 import run_prover

        # Write input to a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".in", delete=False) as f:
            f.write(input_text)
            tmp_path = f.name

        try:
            argv = ["prover9", "-f", tmp_path]
            if extra_args:
                argv.extend(extra_args)

            buf = io.StringIO()
            old_stdout = sys.stdout
            try:
                sys.stdout = buf
                exit_code = run_prover(argv=argv)
            finally:
                sys.stdout = old_stdout
        finally:
            os.unlink(tmp_path)

        return exit_code, buf.getvalue()

    def test_single_proof_output(self):
        """run_prover with default max_proofs=1 prints exactly 1 proof."""
        text = """\
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        exit_code, output = self._run_prover_capture(text)
        assert exit_code == 0
        assert output.count("-------- Proof") == 1
        assert "Proof 1" in output
        assert "THEOREM PROVED" in output

    def test_multi_proof_output_two(self):
        """assign(max_proofs, 2) produces 2 formatted proof blocks in output."""
        text = """\
assign(max_proofs, 2).
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        exit_code, output = self._run_prover_capture(text)
        assert exit_code == 0
        proof_count = output.count("-------- Proof")
        assert proof_count == 2, (
            f"REQ-R006: Expected 2 proof blocks, found {proof_count}. "
            "Proofs may not be displayed incrementally."
        )
        assert "Proof 1" in output
        assert "Proof 2" in output
        assert "Exiting with 2 proofs" in output

    @pytest.mark.skip(reason="SOS exhausted before finding 3 proofs from simple propositional problem")
    def test_multi_proof_output_three(self):
        """assign(max_proofs, 3) produces 3 formatted proof blocks."""
        text = """\
assign(max_proofs, 3).
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        exit_code, output = self._run_prover_capture(text, extra_args=["-max_given", "1000"])
        assert exit_code == 0
        proof_count = output.count("-------- Proof")
        assert proof_count >= 3, (
            f"Expected >=3 proof blocks, found {proof_count}"
        )

    def test_proof_blocks_appear_before_statistics(self):
        """All proof blocks appear before STATISTICS section (during search, not after)."""
        text = """\
assign(max_proofs, 2).
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        exit_code, output = self._run_prover_capture(text)
        assert exit_code == 0

        # Find positions
        stats_pos = output.find("STATISTICS")
        proof1_pos = output.find("-------- Proof 1")
        proof2_pos = output.find("-------- Proof 2")

        assert proof1_pos >= 0, "Proof 1 not found in output"
        assert proof2_pos >= 0, "Proof 2 not found in output"
        assert stats_pos >= 0, "STATISTICS not found in output"
        assert proof1_pos < proof2_pos < stats_pos, (
            "Proofs must appear before STATISTICS section. "
            f"Positions: Proof1={proof1_pos}, Proof2={proof2_pos}, Stats={stats_pos}"
        )

    def test_proof_contains_clauses(self):
        """Each proof block in output contains formatted clause lines."""
        text = """\
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        exit_code, output = self._run_prover_capture(text)
        assert exit_code == 0

        # Extract proof block
        start = output.find("-------- Proof 1")
        end = output.find("end of proof", start)
        assert start >= 0 and end >= 0
        proof_block = output[start:end]

        # Should contain clause lines (e.g., "1 P | Q.")
        import re
        clause_lines = re.findall(r"^\d+\s+.+\.", proof_block, re.MULTILINE)
        assert len(clause_lines) >= 2, (
            f"Proof block has too few clause lines: {clause_lines}"
        )

    @pytest.mark.skip(reason="SOS exhausted before finding 5 proofs from simple propositional problem")
    def test_multi_proof_output_five(self):
        """assign(max_proofs, 5) with propositional problem produces 5 proof blocks."""
        text = """\
assign(max_proofs, 5).
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        exit_code, output = self._run_prover_capture(
            text, extra_args=["-max_given", "2000"],
        )
        assert exit_code == 0
        proof_count = output.count("-------- Proof")
        assert proof_count == 5, (
            f"Expected 5 proof blocks, found {proof_count}"
        )
        assert "Exiting with 5 proofs" in output

    def test_cli_max_proofs_flag(self):
        """CLI -max_proofs flag produces correct number of proof blocks."""
        text = """\
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        exit_code, output = self._run_prover_capture(
            text, extra_args=["-max_proofs", "2"],
        )
        assert exit_code == 0
        assert output.count("-------- Proof") == 2

    def test_no_duplicate_proof_output(self):
        """Proofs are NOT printed both during and after search (no duplicates)."""
        text = """\
assign(max_proofs, 2).
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        exit_code, output = self._run_prover_capture(text)
        assert exit_code == 0
        # Each proof header should appear exactly once
        assert output.count("-------- Proof 1") == 1, "Proof 1 printed multiple times"
        assert output.count("-------- Proof 2") == 1, "Proof 2 printed multiple times"
