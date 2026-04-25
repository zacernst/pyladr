"""Unit tests for hints-guided clause selection.

Tests that formulas(hints) blocks are parsed, hint-matching clauses
receive reduced weight via subsumption checking, and hint-guided
search finds valid proofs.
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.inference.subsumption import subsumes
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions
from pyladr.search.selection import default_clause_weight
from tests.factories import (
    make_clause_from_atoms as _make_clause,
    make_const as _const,
    make_func as _func,
    make_var as _var,
)


# Symbol IDs
A, B, C_SYM = 1, 2, 3
F, G = 10, 11
P, Q = 20, 21


def _run_python(input_text: str, max_seconds: float = 10) -> dict:
    """Run Python prover on input text, return result dict."""
    from pyladr.apps.prover9 import _auto_inference, _auto_limits, _deny_goals

    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(input_text)
    usable, sos, _denied = _deny_goals(parsed, st)
    opts = SearchOptions(max_seconds=max_seconds)
    _auto_inference(parsed, opts)
    _auto_limits(parsed, opts)
    if "hint_wt" in parsed.assigns:
        opts.hint_wt = float(parsed.assigns["hint_wt"])
    engine = GivenClauseSearch(
        options=opts,
        symbol_table=st,
        hints=parsed.hints if parsed.hints else None,
    )
    result = engine.run(usable=usable, sos=sos)
    return {
        "proved": len(result.proofs) > 0,
        "exit_code": result.exit_code,
        "given": result.stats.given,
        "generated": result.stats.generated,
        "kept": result.stats.kept,
    }


# ── Parsing Tests ────────────────────────────────────────────────────────────


class TestHintsParsing:
    """Test that formulas(hints) blocks populate ParsedInput.hints."""

    def test_hints_block_parsed(self) -> None:
        """formulas(hints) block produces hints list."""
        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(
            "formulas(hints). P(x). end_of_list."
        )
        assert len(parsed.hints) == 1

    def test_hints_block_empty(self) -> None:
        """Empty hints block produces empty list."""
        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(
            "formulas(hints). end_of_list."
        )
        assert parsed.hints == []

    def test_multiple_hints(self) -> None:
        """Multiple formulas in hints block all captured."""
        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(
            "formulas(hints). P(x). Q(x,y). end_of_list."
        )
        assert len(parsed.hints) == 2

    def test_hints_separate_from_sos(self) -> None:
        """Hints and sos are distinct lists."""
        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(
            "formulas(sos). A(x). end_of_list.\n"
            "formulas(hints). B(x). end_of_list."
        )
        assert len(parsed.sos) == 1
        assert len(parsed.hints) == 1

    def test_no_hints_block_defaults_empty(self) -> None:
        """No hints block means empty hints list."""
        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(
            "formulas(sos). P(x). end_of_list."
        )
        assert parsed.hints == []


# ── Weight Adjustment Tests ──────────────────────────────────────────────────


class TestHintWeightAdjustment:
    """Test that hint-matching clauses get reduced weight."""

    def test_matching_clause_gets_hint_weight(self) -> None:
        """A clause subsumed by a hint gets weight = min(original, hint_wt)."""
        # Hint: P(x) subsumes P(a) for any constant a
        hint = _make_clause(_func(P, _var(0)))
        clause = _make_clause(_func(P, _const(A)))
        clause.weight = default_clause_weight(clause)
        original_weight = clause.weight

        opts = SearchOptions(hint_wt=1.0)
        engine = GivenClauseSearch(options=opts, hints=[hint])
        engine._apply_hint_weight(clause)

        assert clause.weight == min(original_weight, 1.0)

    def test_non_matching_clause_keeps_weight(self) -> None:
        """A clause not subsumed by any hint keeps its original weight."""
        hint = _make_clause(_func(Q, _var(0)))  # Q(x)
        clause = _make_clause(_func(P, _const(A)))  # P(a) — different predicate
        clause.weight = default_clause_weight(clause)
        original_weight = clause.weight

        opts = SearchOptions(hint_wt=1.0)
        engine = GivenClauseSearch(options=opts, hints=[hint])
        engine._apply_hint_weight(clause)

        assert clause.weight == original_weight

    def test_hint_wt_respected(self) -> None:
        """Custom hint_wt value is used for matched clauses."""
        hint = _make_clause(_func(P, _var(0)))
        # Create a heavier clause: P(f(a))
        clause = _make_clause(_func(P, _func(F, _const(A))))
        clause.weight = default_clause_weight(clause)
        assert clause.weight > 0.5  # should be > 1

        opts = SearchOptions(hint_wt=0.5)
        engine = GivenClauseSearch(options=opts, hints=[hint])
        engine._apply_hint_weight(clause)

        assert clause.weight == 0.5

    def test_hint_wt_uses_min(self) -> None:
        """If clause weight is already below hint_wt, weight unchanged."""
        hint = _make_clause(_func(P, _var(0)))
        clause = _make_clause(_func(P, _const(A)))
        clause.weight = 0.5  # Artificially low

        opts = SearchOptions(hint_wt=10.0)
        engine = GivenClauseSearch(options=opts, hints=[hint])
        engine._apply_hint_weight(clause)

        assert clause.weight == 0.5  # min(0.5, 10.0)

    def test_multiple_hints_first_match_wins(self) -> None:
        """With multiple hints, first matching hint triggers weight reduction."""
        hint1 = _make_clause(_func(P, _var(0)))  # P(x)
        hint2 = _make_clause(_func(Q, _var(0)))  # Q(x)
        clause = _make_clause(_func(P, _const(A)))  # P(a) — matches hint1
        clause.weight = default_clause_weight(clause)

        opts = SearchOptions(hint_wt=1.0)
        engine = GivenClauseSearch(options=opts, hints=[hint1, hint2])
        engine._apply_hint_weight(clause)

        assert clause.weight == 1.0


# ── No-Hints Fallback Tests ─────────────────────────────────────────────────


class TestNoHintsFallback:
    """Test that empty hints = identical behavior to no hints."""

    def test_no_hints_no_weight_change(self) -> None:
        """With no hints, _apply_hint_weight is a no-op."""
        clause = _make_clause(_func(P, _const(A)))
        clause.weight = default_clause_weight(clause)
        original = clause.weight

        engine = GivenClauseSearch(options=SearchOptions())
        engine._apply_hint_weight(clause)

        assert clause.weight == original

    def test_empty_hints_list_no_weight_change(self) -> None:
        """Explicit empty hints list = no weight change."""
        clause = _make_clause(_func(P, _const(A)))
        clause.weight = default_clause_weight(clause)
        original = clause.weight

        engine = GivenClauseSearch(options=SearchOptions(), hints=[])
        engine._apply_hint_weight(clause)

        assert clause.weight == original

    def test_none_hints_no_weight_change(self) -> None:
        """hints=None = no weight change."""
        clause = _make_clause(_func(P, _const(A)))
        clause.weight = default_clause_weight(clause)
        original = clause.weight

        engine = GivenClauseSearch(options=SearchOptions(), hints=None)
        engine._apply_hint_weight(clause)

        assert clause.weight == original


# ── Subsumption-Based Matching Tests ─────────────────────────────────────────


class TestHintSubsumptionMatching:
    """Test that hint matching uses subsumption correctly."""

    def test_variable_hint_matches_ground(self) -> None:
        """Hint P(x) subsumes ground clause P(a)."""
        hint = _make_clause(_func(P, _var(0)))
        ground = _make_clause(_func(P, _const(A)))
        assert subsumes(hint, ground)

    def test_ground_hint_does_not_match_variable(self) -> None:
        """Ground hint P(a) does not subsume P(x)."""
        hint = _make_clause(_func(P, _const(A)))
        general = _make_clause(_func(P, _var(0)))
        assert not subsumes(hint, general)

    def test_hint_different_predicate_no_match(self) -> None:
        """Hint P(x) does not subsume Q(a)."""
        hint = _make_clause(_func(P, _var(0)))
        clause = _make_clause(_func(Q, _const(A)))
        assert not subsumes(hint, clause)

    def test_multi_literal_hint(self) -> None:
        """Hint P(x) | Q(x) subsumes P(a) | Q(a)."""
        hint = _make_clause(_func(P, _var(0)), _func(Q, _var(0)))
        clause = _make_clause(_func(P, _const(A)), _func(Q, _const(A)))
        assert subsumes(hint, clause)


# ── Integration: Search Correctness ──────────────────────────────────────────


class TestHintsSearchIntegration:
    """Test that hint-guided search finds valid proofs."""

    def test_proof_with_hints(self) -> None:
        """Hints guide search to a valid proof."""
        input_text = (
            "formulas(sos).\n"
            "  P(a).\n"
            "  -P(x) | Q(x).\n"
            "end_of_list.\n"
            "formulas(goals).\n"
            "  Q(a).\n"
            "end_of_list.\n"
            "formulas(hints).\n"
            "  Q(x).\n"
            "end_of_list.\n"
        )
        result = _run_python(input_text)
        assert result["proved"]

    def test_proof_without_hints_same_result(self) -> None:
        """Same problem without hints still finds a proof."""
        input_text = (
            "formulas(sos).\n"
            "  P(a).\n"
            "  -P(x) | Q(x).\n"
            "end_of_list.\n"
            "formulas(goals).\n"
            "  Q(a).\n"
            "end_of_list.\n"
        )
        result = _run_python(input_text)
        assert result["proved"]

    def test_hint_wt_assign_parsed(self) -> None:
        """assign(hint_wt, N) directive is respected."""
        input_text = (
            "assign(hint_wt, 0.5).\n"
            "formulas(sos).\n"
            "  P(a).\n"
            "  -P(x) | Q(x).\n"
            "end_of_list.\n"
            "formulas(goals).\n"
            "  Q(a).\n"
            "end_of_list.\n"
            "formulas(hints).\n"
            "  Q(x).\n"
            "end_of_list.\n"
        )
        result = _run_python(input_text)
        assert result["proved"]

    def test_irrelevant_hints_do_not_prevent_proof(self) -> None:
        """Hints that don't match anything don't break search."""
        input_text = (
            "formulas(sos).\n"
            "  P(a).\n"
            "  -P(x) | Q(x).\n"
            "end_of_list.\n"
            "formulas(goals).\n"
            "  Q(a).\n"
            "end_of_list.\n"
            "formulas(hints).\n"
            "  R(x,y).\n"
            "end_of_list.\n"
        )
        result = _run_python(input_text)
        assert result["proved"]
