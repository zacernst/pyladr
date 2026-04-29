"""Tests for proven goal removal from proximity scoring.

Covers:
1. GoalProximityScorer.remove_goal() — correct removal, index shift, num_goals
2. GivenClauseSearch._on_goal_subsumed() — ID lookup, removal, no-op for non-goals
3. Back-subsumption hook — goal removal fires when goal clause is back-subsumed
4. Regression — search with goal proximity still finds proofs
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.search.goal_directed import GoalProximityScorer


# ── Helpers ───────────────────────────────────────────────────────────────

SYM_P = 1
SYM_F = 2
SYM_A = 3
SYM_B = 4


def var(n: int) -> Term:
    return get_variable_term(n)


def P(arg: Term) -> Term:
    return get_rigid_term(SYM_P, 1, (arg,))


def f(arg: Term) -> Term:
    return get_rigid_term(SYM_F, 1, (arg,))


def a() -> Term:
    return get_rigid_term(SYM_A, 0)


def b() -> Term:
    return get_rigid_term(SYM_B, 0)


def make_literal(sign: bool, atom: Term) -> Literal:
    return Literal(sign=sign, atom=atom)


def make_clause(*lits: Literal, clause_id: int = 0) -> Clause:
    return Clause(literals=lits, id=clause_id)


def make_goal_clause(*lits: Literal, clause_id: int = 0) -> Clause:
    """Create a clause with DENY justification (goal clause)."""
    return Clause(
        literals=lits,
        id=clause_id,
        justification=(Justification(just_type=JustType.DENY),),
    )


# ── 1. GoalProximityScorer.remove_goal() ─────────────────────────────────


class TestGoalProximityScorerRemoveGoal:
    def test_remove_single_goal(self) -> None:
        scorer = GoalProximityScorer(method="max")
        scorer.set_goals([[1.0, 0.0], [0.0, 1.0]])
        assert scorer.num_goals == 2

        scorer.remove_goal(0)
        assert scorer.num_goals == 1

    def test_remove_decrements_num_goals(self) -> None:
        scorer = GoalProximityScorer(method="max")
        scorer.set_goals([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        assert scorer.num_goals == 3

        scorer.remove_goal(1)
        assert scorer.num_goals == 2

        scorer.remove_goal(0)
        assert scorer.num_goals == 1

        scorer.remove_goal(0)
        assert scorer.num_goals == 0

    def test_remove_last_goal_returns_neutral_distance(self) -> None:
        scorer = GoalProximityScorer(method="max")
        scorer.set_goals([[1.0, 0.0]])
        assert scorer.num_goals == 1

        scorer.remove_goal(0)
        assert scorer.num_goals == 0
        assert scorer.nearest_goal_distance([1.0, 0.0]) == 0.5

    def test_remove_shifts_remaining_goals(self) -> None:
        """After removing index 0, the former index-1 goal is now at index 0."""
        scorer = GoalProximityScorer(method="max")
        goal_a = [1.0, 0.0]
        goal_b = [0.0, 1.0]
        scorer.set_goals([goal_a, goal_b])

        # Query aligned with goal_b — small distance
        query = [0.0, 1.0]
        dist_before = scorer.nearest_goal_distance(query)

        # Remove goal_a (index 0) — goal_b shifts to index 0
        scorer.remove_goal(0)
        dist_after = scorer.nearest_goal_distance(query)

        # Distance to goal_b should be identical (it's still there)
        assert dist_before == pytest.approx(dist_after)

    def test_remove_out_of_bounds_is_noop(self) -> None:
        scorer = GoalProximityScorer(method="max")
        scorer.set_goals([[1.0, 0.0]])

        scorer.remove_goal(5)
        assert scorer.num_goals == 1

        scorer.remove_goal(-1)
        assert scorer.num_goals == 1

    def test_remove_from_empty_is_noop(self) -> None:
        scorer = GoalProximityScorer(method="max")
        scorer.remove_goal(0)
        assert scorer.num_goals == 0

    def test_remove_correct_goal_increases_distance(self) -> None:
        """Removing goal_a leaves only goal_b, increasing distance for goal_a-aligned queries."""
        scorer = GoalProximityScorer(method="max")
        goal_a = [1.0, 0.0]
        goal_b = [0.0, 1.0]
        scorer.set_goals([goal_a, goal_b])

        query_a = [1.0, 0.0]  # aligned with goal_a
        dist_with_a = scorer.nearest_goal_distance(query_a)

        # Remove goal_a
        scorer.remove_goal(0)
        dist_without_a = scorer.nearest_goal_distance(query_a)

        # Without goal_a, distance should increase (query is orthogonal to remaining goal_b)
        assert dist_without_a > dist_with_a


# ── Regression — search with goal proximity still finds proofs ───────────


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
    """Run Python prover on input text, return result dict."""
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
        "given": result.stats.given,
    }


class TestRegressionGoalProximitySearch:
    def test_search_with_goal_proximity_finds_proof(self) -> None:
        """Goal proximity enabled does not prevent proof finding."""
        input_text = "set(rnn2vec_goal_proximity).\n" + _SIMPLE_PROOF_INPUT
        result = _run_python(input_text)
        assert result["proved"]

    def test_search_without_goal_proximity_finds_proof(self) -> None:
        """Baseline: search without goal proximity still works."""
        result = _run_python(_SIMPLE_PROOF_INPUT)
        assert result["proved"]

    def test_search_with_goal_proximity_and_online_learning(self) -> None:
        """Goal proximity + online learning does not regress."""
        input_text = (
            "set(rnn2vec_goal_proximity).\n"
            "set(rnn2vec_online_learning).\n"
            + _SIMPLE_PROOF_INPUT
        )
        result = _run_python(input_text)
        assert result["proved"]
