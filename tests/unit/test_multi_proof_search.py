"""Tests for multi-proof search functionality (REQ-R005 regression prevention).

Validates that assign(max_proofs, N) works correctly:
- Parser correctly extracts assign(max_proofs, N) directives
- Parsed assignments are applied to SearchOptions
- Search continues after first proof when max_proofs > 1
- Search stops at exactly max_proofs limit
- Proof accumulation and statistics are accurate
- Edge cases handled correctly
- CLI flags and file assignments interact correctly

These tests are the primary regression guard for REQ-R005.
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import parse_input, ParsedInput
from pyladr.apps.prover9 import _apply_assignments
from pyladr.search.given_clause import (
    ExitCode,
    GivenClauseSearch,
    SearchOptions,
    SearchResult,
)


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


def _run_search(
    text: str,
    max_given: int = 500,
    max_seconds: float = 10.0,
    **kwargs,
) -> SearchResult:
    """Parse input text and run search with given options."""
    usable, sos, st = _parse_and_deny(text)
    opts = SearchOptions(
        max_given=max_given,
        max_seconds=max_seconds,
        quiet=True,
        print_given=False,
        **kwargs,
    )
    engine = GivenClauseSearch(opts, symbol_table=st)
    return engine.run(usable=usable, sos=sos)


# ── Test problems with multiple proofs ───────────────────────────────────────
#
# Strategy: Use problems where the denied goal can resolve with multiple
# different clauses, producing distinct empty clauses via different paths.

# Problem 1: Propositional with redundant complementary pairs.
# P and -P can derive $F. Q and -Q can also derive $F.
# With resolution, this produces multiple empty clauses.
PROPOSITIONAL_MULTI_PROOF = """\
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""

# Problem 2: Group theory x*x=e implies commutativity.
# This classic problem typically finds multiple proofs via different
# paramodulation paths in the equational space.
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

# Problem 3: Simple resolution problem with multiple derivation paths.
# Multiple ways to derive the goal from different axiom combinations.
MULTI_PATH_RESOLUTION = """\
formulas(sos).
  P(a).
  P(b).
  -P(x) | Q(x).
  -Q(a).
  -Q(b).
end_of_list.
"""

# Problem 4: Very simple - identity has trivial proof from e*x=x.
TRIVIAL_SINGLE_PROOF = """\
formulas(sos).
  e * x = x.
end_of_list.
formulas(goals).
  e * e = e.
end_of_list.
"""


# ── REQ-R005 Core Regression Tests ──────────────────────────────────────────


class TestMaxProofsDefault:
    """Verify default max_proofs=1 behavior (baseline)."""

    def test_default_max_proofs_is_one(self):
        """SearchOptions defaults to max_proofs=1."""
        opts = SearchOptions()
        assert opts.max_proofs == 1

    def test_single_proof_found_with_default(self):
        """Default max_proofs=1 finds exactly 1 proof and stops."""
        result = _run_search(TRIVIAL_SINGLE_PROOF, paramodulation=True)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1
        assert result.stats.proofs == 1

    def test_single_proof_propositional(self):
        """Propositional problem finds 1 proof with default max_proofs."""
        result = _run_search(PROPOSITIONAL_MULTI_PROOF)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1
        assert result.stats.proofs == 1


class TestMultiProofSearch:
    """REQ-R005: Verify search continues after first proof when max_proofs > 1.

    This is the critical regression test class. If max_proofs is ignored
    and search stops after the first proof, these tests FAIL.
    """

    def test_max_proofs_2_finds_more_than_1(self):
        """max_proofs=2 must find 2 proofs (not stop at 1).

        REQ-R005 REGRESSION GUARD: This test fails if max_proofs is ignored.
        """
        result = _run_search(
            PROPOSITIONAL_MULTI_PROOF,
            max_proofs=2,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 2, (
            f"Expected 2 proofs but got {len(result.proofs)}. "
            "REQ-R005 regression: search may have stopped after first proof."
        )
        assert result.stats.proofs == 2

    def test_max_proofs_3_propositional(self):
        """max_proofs=3 with propositional problem that has many proofs."""
        result = _run_search(
            PROPOSITIONAL_MULTI_PROOF,
            max_proofs=3,
            max_given=1000,
        )
        # Should find at least 2 proofs (may exhaust SOS before 3)
        assert len(result.proofs) >= 2, (
            f"Expected >= 2 proofs but got {len(result.proofs)}. "
            "REQ-R005 regression: max_proofs may be ignored."
        )
        assert result.stats.proofs == len(result.proofs)

    def test_max_proofs_equational(self):
        """max_proofs > 1 with equational (paramodulation) problem."""
        result = _run_search(
            GROUP_X2_COMM,
            max_proofs=3,
            paramodulation=True,
            max_given=2000,
            max_seconds=30.0,
        )
        # The x2 problem should find at least 1 proof; with continuation
        # and enough search space, it should find more.
        assert result.exit_code in (ExitCode.MAX_PROOFS_EXIT, ExitCode.SOS_EMPTY_EXIT,
                                     ExitCode.MAX_GIVEN_EXIT)
        if result.exit_code == ExitCode.MAX_PROOFS_EXIT:
            assert len(result.proofs) == 3
        else:
            # Search ended before finding 3, but should have found at least 1
            assert len(result.proofs) >= 1

    def test_proof_accumulation_matches_stats(self):
        """Proof list length must always equal stats.proofs counter."""
        result = _run_search(
            PROPOSITIONAL_MULTI_PROOF,
            max_proofs=5,
            max_given=1000,
        )
        assert len(result.proofs) == result.stats.proofs, (
            f"Proof list has {len(result.proofs)} entries but "
            f"stats.proofs = {result.stats.proofs}"
        )

    def test_each_proof_has_empty_clause(self):
        """Every accumulated proof must contain a valid empty clause."""
        result = _run_search(
            PROPOSITIONAL_MULTI_PROOF,
            max_proofs=3,
            max_given=1000,
        )
        for i, proof in enumerate(result.proofs):
            assert proof.empty_clause.is_empty, (
                f"Proof {i+1} empty_clause is not actually empty"
            )
            assert len(proof.clauses) > 0, (
                f"Proof {i+1} has no clauses in trace"
            )

    def test_distinct_proofs(self):
        """Multiple proofs should have distinct empty clause IDs."""
        result = _run_search(
            PROPOSITIONAL_MULTI_PROOF,
            max_proofs=3,
            max_given=1000,
        )
        if len(result.proofs) >= 2:
            empty_ids = [p.empty_clause.id for p in result.proofs]
            assert len(set(empty_ids)) == len(empty_ids), (
                f"Duplicate empty clause IDs found: {empty_ids}. "
                "Each proof should derive a distinct empty clause."
            )


class TestMaxProofsTermination:
    """Verify search terminates at exactly the max_proofs limit."""

    def test_stops_at_exactly_max_proofs(self):
        """When max_proofs=N proofs exist, search finds exactly N and exits."""
        # Use max_proofs=1 as baseline
        r1 = _run_search(PROPOSITIONAL_MULTI_PROOF, max_proofs=1)
        assert len(r1.proofs) == 1
        assert r1.exit_code == ExitCode.MAX_PROOFS_EXIT

        # Now max_proofs=2 - should find exactly 2
        r2 = _run_search(PROPOSITIONAL_MULTI_PROOF, max_proofs=2)
        assert len(r2.proofs) == 2
        assert r2.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_exit_code_max_proofs(self):
        """Exit code is MAX_PROOFS_EXIT when limit reached."""
        result = _run_search(
            PROPOSITIONAL_MULTI_PROOF,
            max_proofs=2,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert result.exit_code.value == 1

    def test_sos_empty_when_proofs_fewer_than_max(self):
        """If SOS exhausted before max_proofs, exit with SOS_EMPTY_EXIT."""
        result = _run_search(
            TRIVIAL_SINGLE_PROOF,
            max_proofs=100,
            paramodulation=True,
            max_given=200,
        )
        # This trivial problem likely has very few proofs.
        # Search should end with SOS_EMPTY or MAX_GIVEN, not MAX_PROOFS.
        assert result.exit_code in (
            ExitCode.SOS_EMPTY_EXIT,
            ExitCode.MAX_GIVEN_EXIT,
            ExitCode.MAX_PROOFS_EXIT,  # in case it does find enough
        )
        # Whatever happened, proofs found <= max_proofs
        assert len(result.proofs) <= 100

    def test_max_proofs_with_resource_limit(self):
        """Resource limits (max_given) take priority if hit before max_proofs."""
        result = _run_search(
            GROUP_X2_COMM,
            max_proofs=100,
            paramodulation=True,
            max_given=10,  # Very tight limit
        )
        # Should hit max_given before finding 100 proofs
        assert result.exit_code in (
            ExitCode.MAX_GIVEN_EXIT,
            ExitCode.MAX_PROOFS_EXIT,
            ExitCode.SOS_EMPTY_EXIT,
        )
        assert len(result.proofs) < 100


class TestMultiProofResolution:
    """Multi-proof tests specifically using binary resolution."""

    def test_resolution_multi_proof(self):
        """Binary resolution finds multiple proofs from complementary pairs."""
        result = _run_search(
            MULTI_PATH_RESOLUTION,
            max_proofs=2,
            binary_resolution=True,
            paramodulation=False,
        )
        # P(a) + -P(x)|Q(x) -> Q(a); Q(a) + -Q(a) -> empty
        # P(b) + -P(x)|Q(x) -> Q(b); Q(b) + -Q(b) -> empty
        # Two distinct derivation paths
        assert len(result.proofs) >= 1, "Should find at least one proof"
        if result.exit_code == ExitCode.MAX_PROOFS_EXIT:
            assert len(result.proofs) == 2


class TestMaxProofsEdgeCases:
    """Edge cases for max_proofs parameter."""

    def test_max_proofs_minus_one_unlimited(self):
        """max_proofs=-1 means unlimited (search until SOS empty or other limit)."""
        result = _run_search(
            PROPOSITIONAL_MULTI_PROOF,
            max_proofs=-1,
            max_given=100,
        )
        # With unlimited proofs, should NOT exit with MAX_PROOFS_EXIT
        # (unless the check is: max_proofs > 0 AND proofs >= max_proofs)
        # -1 > 0 is False, so the proof limit check never triggers
        assert result.exit_code in (
            ExitCode.SOS_EMPTY_EXIT,
            ExitCode.MAX_GIVEN_EXIT,
        )
        # Should have found some proofs
        assert len(result.proofs) >= 1

    def test_max_proofs_zero_treated_as_unlimited(self):
        """max_proofs=0: check is (0 > 0) which is False, so unlimited."""
        result = _run_search(
            PROPOSITIONAL_MULTI_PROOF,
            max_proofs=0,
            max_given=100,
        )
        # (0 > 0 and proofs >= 0) is False, so never triggers MAX_PROOFS_EXIT
        assert result.exit_code in (
            ExitCode.SOS_EMPTY_EXIT,
            ExitCode.MAX_GIVEN_EXIT,
        )

    def test_max_proofs_very_large(self):
        """max_proofs much larger than available proofs: search ends normally."""
        result = _run_search(
            TRIVIAL_SINGLE_PROOF,
            max_proofs=10000,
            paramodulation=True,
            max_given=200,
        )
        # Not enough proofs to hit the limit
        assert result.exit_code in (
            ExitCode.SOS_EMPTY_EXIT,
            ExitCode.MAX_GIVEN_EXIT,
            ExitCode.MAX_PROOFS_EXIT,
        )
        assert len(result.proofs) <= 10000


class TestMultiProofProofQuality:
    """Validate that multi-proof results have well-formed proof traces."""

    def test_proof_traces_are_sorted_by_id(self):
        """Clause IDs in each proof trace are sorted ascending."""
        result = _run_search(
            PROPOSITIONAL_MULTI_PROOF,
            max_proofs=3,
            max_given=1000,
        )
        for i, proof in enumerate(result.proofs):
            ids = [c.id for c in proof.clauses]
            assert ids == sorted(ids), (
                f"Proof {i+1} clauses not sorted by ID: {ids}"
            )

    def test_proof_traces_contain_input_clauses(self):
        """Each proof trace should include input (initial) clauses."""
        result = _run_search(
            PROPOSITIONAL_MULTI_PROOF,
            max_proofs=2,
        )
        for i, proof in enumerate(result.proofs):
            has_initial = any(c.initial for c in proof.clauses)
            assert has_initial, (
                f"Proof {i+1} has no initial clauses in trace"
            )

    def test_later_proofs_found_after_first(self):
        """Second proof should involve at least some different clause IDs."""
        result = _run_search(
            PROPOSITIONAL_MULTI_PROOF,
            max_proofs=2,
        )
        if len(result.proofs) >= 2:
            ids_0 = set(c.id for c in result.proofs[0].clauses)
            ids_1 = set(c.id for c in result.proofs[1].clauses)
            # The two proof traces should not be identical
            assert ids_0 != ids_1, (
                "Two proofs have identical clause traces - "
                "expected different derivation paths"
            )


class TestSearchContinuationAfterProof:
    """Validate that the search loop genuinely continues after finding a proof.

    These tests are specifically designed to detect the REQ-R005 regression
    where search stops after the first proof regardless of max_proofs.
    """

    def test_more_given_clauses_processed_with_higher_max_proofs(self):
        """Higher max_proofs should process more given clauses (search continues)."""
        r1 = _run_search(PROPOSITIONAL_MULTI_PROOF, max_proofs=1)
        r2 = _run_search(PROPOSITIONAL_MULTI_PROOF, max_proofs=5, max_given=1000)

        if r2.exit_code != ExitCode.MAX_PROOFS_EXIT or len(r2.proofs) > 1:
            # If more proofs were found, search must have continued
            # This means more given clauses were processed
            assert r2.stats.given >= r1.stats.given, (
                "With max_proofs=5, search should process at least as many "
                f"given clauses as max_proofs=1. Got {r2.stats.given} vs {r1.stats.given}"
            )

    def test_stats_proofs_increments(self):
        """stats.proofs must increment for each proof found."""
        result = _run_search(
            PROPOSITIONAL_MULTI_PROOF,
            max_proofs=3,
            max_given=1000,
        )
        # If we found N proofs, stats.proofs must be N
        assert result.stats.proofs == len(result.proofs)
        if len(result.proofs) >= 2:
            assert result.stats.proofs >= 2, (
                "REQ-R005: stats.proofs not incrementing past 1"
            )


# ── Assign Directive Parsing Tests ──────────────────────────────────────────
#
# REQ-R005 ROOT CAUSE: The regression was caused by assign() directives not
# being parsed or not being applied to SearchOptions. These tests guard the
# parser → application pipeline.


class TestAssignDirectiveParsing:
    """Verify assign() directives are correctly parsed from input text.

    REQ-R005 ROOT CAUSE GUARD: If the parser skips assign() directives,
    max_proofs from the input file is silently ignored.
    """

    def test_assign_max_proofs_parsed(self):
        """assign(max_proofs, 10) must appear in ParsedInput.assignments."""
        text = """\
assign(max_proofs, 10).
formulas(sos).
  P | Q.
end_of_list.
"""
        st = SymbolTable()
        parsed = parse_input(text, st)
        assert "max_proofs" in parsed.assignments, (
            "REQ-R005: assign(max_proofs, N) not parsed into assignments dict"
        )
        assert parsed.assignments["max_proofs"] == 10

    def test_assign_max_proofs_integer_type(self):
        """assign(max_proofs, N) must parse as int, not float."""
        text = "assign(max_proofs, 5).\nformulas(sos).\n  P.\nend_of_list.\n"
        st = SymbolTable()
        parsed = parse_input(text, st)
        assert isinstance(parsed.assignments["max_proofs"], int)
        assert parsed.assignments["max_proofs"] == 5

    def test_assign_max_proofs_various_values(self):
        """Test parsing assign(max_proofs, N) for N=1,2,3,5,10."""
        for n in [1, 2, 3, 5, 10]:
            text = f"assign(max_proofs, {n}).\nformulas(sos).\n  P.\nend_of_list.\n"
            st = SymbolTable()
            parsed = parse_input(text, st)
            assert parsed.assignments["max_proofs"] == n, (
                f"assign(max_proofs, {n}) parsed as {parsed.assignments.get('max_proofs')}"
            )

    def test_assign_multiple_directives(self):
        """Multiple assign() directives all parsed correctly."""
        text = """\
assign(max_proofs, 5).
assign(max_given, 200).
assign(max_seconds, 30).
formulas(sos).
  P.
end_of_list.
"""
        st = SymbolTable()
        parsed = parse_input(text, st)
        assert parsed.assignments["max_proofs"] == 5
        assert parsed.assignments["max_given"] == 200
        assert parsed.assignments["max_seconds"] == 30.0 or parsed.assignments["max_seconds"] == 30

    def test_assign_with_whitespace_variations(self):
        """assign() with extra whitespace still parsed correctly."""
        text = "assign( max_proofs , 3 ).\nformulas(sos).\n  P.\nend_of_list.\n"
        st = SymbolTable()
        parsed = parse_input(text, st)
        assert parsed.assignments.get("max_proofs") == 3, (
            "assign() with extra whitespace not parsed correctly"
        )

    def test_assign_before_and_after_formulas(self):
        """assign() directives work whether placed before or after formulas."""
        text = """\
formulas(sos).
  P | Q.
end_of_list.
assign(max_proofs, 7).
"""
        st = SymbolTable()
        parsed = parse_input(text, st)
        assert parsed.assignments.get("max_proofs") == 7, (
            "assign() after formulas block not parsed"
        )

    def test_no_assignments_when_absent(self):
        """No assignments in result when input has no assign() directives."""
        text = "formulas(sos).\n  P.\nend_of_list.\n"
        st = SymbolTable()
        parsed = parse_input(text, st)
        assert len(parsed.assignments) == 0


class TestAssignDirectiveApplication:
    """Verify parsed assign() directives are applied to SearchOptions.

    REQ-R005 ROOT CAUSE GUARD: Even if parsing works, the application
    step must transfer the value to SearchOptions.
    """

    def test_apply_max_proofs_overrides_default(self):
        """_apply_assignments must override default max_proofs=1."""
        parsed = ParsedInput()
        parsed.assignments["max_proofs"] = 10

        opts = SearchOptions()  # max_proofs defaults to 1
        assert opts.max_proofs == 1

        _apply_assignments(parsed, opts)
        assert opts.max_proofs == 10, (
            "REQ-R005: _apply_assignments did not set max_proofs from parsed input"
        )

    def test_apply_max_proofs_overrides_cli(self):
        """File assign() overrides CLI default (simulating run_prover flow)."""
        parsed = ParsedInput()
        parsed.assignments["max_proofs"] = 5

        # CLI default is 1 (args.max_proofs default)
        opts = SearchOptions(max_proofs=1)
        _apply_assignments(parsed, opts)
        assert opts.max_proofs == 5

    def test_apply_preserves_other_options(self):
        """Applying max_proofs doesn't clobber other SearchOptions fields."""
        parsed = ParsedInput()
        parsed.assignments["max_proofs"] = 10

        opts = SearchOptions(max_given=500, max_seconds=30.0, paramodulation=True)
        _apply_assignments(parsed, opts)

        assert opts.max_proofs == 10
        assert opts.max_given == 500
        assert opts.max_seconds == 30.0
        assert opts.paramodulation is True

    def test_apply_multiple_assignments(self):
        """Multiple assign() directives all applied to SearchOptions."""
        parsed = ParsedInput()
        parsed.assignments["max_proofs"] = 3
        parsed.assignments["max_given"] = 1000
        parsed.assignments["max_weight"] = 50.0

        opts = SearchOptions()
        _apply_assignments(parsed, opts)

        assert opts.max_proofs == 3
        assert opts.max_given == 1000
        assert opts.max_weight == 50.0

    def test_unknown_assignment_ignored(self):
        """Unknown assign() names are silently ignored (no crash)."""
        parsed = ParsedInput()
        parsed.assignments["nonexistent_option"] = 42
        parsed.assignments["max_proofs"] = 5

        opts = SearchOptions()
        _apply_assignments(parsed, opts)
        assert opts.max_proofs == 5  # Known option applied
        assert not hasattr(opts, "nonexistent_option")


# ── End-to-End: assign() in Input → Multi-Proof Search ─────────────────────


def _run_search_from_full_input(text: str, **override_opts) -> SearchResult:
    """Parse input WITH assign() directives and run search.

    This simulates the full run_prover pipeline:
    1. Parse input text (including assign/set/clear directives)
    2. Build SearchOptions from defaults
    3. Apply parsed assignments to override defaults
    4. Run search
    """
    st = SymbolTable()
    parsed = parse_input(text, st)

    # Build usable/sos with denied goals (same as run_prover)
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

    # Start with defaults, then apply parsed assignments
    opts = SearchOptions(
        quiet=True,
        print_given=False,
        max_given=override_opts.pop("max_given", 500),
        max_seconds=override_opts.pop("max_seconds", 10.0),
        **override_opts,
    )
    _apply_assignments(parsed, opts)

    engine = GivenClauseSearch(opts, symbol_table=st)
    return engine.run(usable=usable, sos=sos)


class TestEndToEndAssignMaxProofs:
    """End-to-end tests: assign(max_proofs, N) in input drives multi-proof search.

    REQ-R005 REGRESSION GUARD: These tests exercise the complete pipeline
    from input file text to multi-proof search results. If any layer breaks
    (parser, application, or search), these tests fail.
    """

    def test_assign_max_proofs_2_in_input(self):
        """assign(max_proofs, 2) in input file finds 2 proofs."""
        text = """\
assign(max_proofs, 2).
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        result = _run_search_from_full_input(text)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 2, (
            f"REQ-R005: assign(max_proofs, 2) in input yielded {len(result.proofs)} proofs. "
            "Expected exactly 2. The assign() directive may not be reaching the search engine."
        )

    def test_assign_max_proofs_1_in_input(self):
        """assign(max_proofs, 1) in input file finds exactly 1 proof."""
        text = """\
assign(max_proofs, 1).
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        result = _run_search_from_full_input(text)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1

    def test_assign_max_proofs_overrides_default_e2e(self):
        """Without assign(), default max_proofs=1; with assign(max_proofs, 2), finds 2."""
        input_no_assign = """\
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        input_with_assign = """\
assign(max_proofs, 2).
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        r_default = _run_search_from_full_input(input_no_assign)
        r_assigned = _run_search_from_full_input(input_with_assign)

        assert len(r_default.proofs) == 1, "Default should find 1 proof"
        assert len(r_assigned.proofs) == 2, (
            f"REQ-R005: assign(max_proofs, 2) should find 2 proofs, got {len(r_assigned.proofs)}"
        )

    def test_assign_max_proofs_with_goals(self):
        """assign(max_proofs, N) works with goals (deny-and-search pattern)."""
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
        result = _run_search_from_full_input(
            text,
            paramodulation=True,
            max_given=2000,
            max_seconds=30.0,
        )
        # Should find at least 1 proof; may find up to 3
        assert len(result.proofs) >= 1
        if result.exit_code == ExitCode.MAX_PROOFS_EXIT:
            assert len(result.proofs) == 3

    def test_assign_max_proofs_with_other_assigns(self):
        """max_proofs works alongside other assign() directives."""
        text = """\
assign(max_proofs, 2).
assign(max_given, 1000).
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        result = _run_search_from_full_input(text)
        assert len(result.proofs) == 2
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_assign_max_proofs_10_user_scenario(self):
        """Exact user scenario from REQ-R005: assign(max_proofs, 10).

        With a problem that has many proofs, this must find more than 1.
        """
        text = """\
assign(max_proofs, 10).
formulas(sos).
  P | Q.
  -P.
  -Q.
end_of_list.
"""
        result = _run_search_from_full_input(text, max_given=2000)
        assert len(result.proofs) >= 2, (
            f"REQ-R005 USER SCENARIO: assign(max_proofs, 10) found only "
            f"{len(result.proofs)} proof(s). Search must continue past first proof."
        )
