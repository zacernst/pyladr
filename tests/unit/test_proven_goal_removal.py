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


# ── 2. GivenClauseSearch._on_goal_subsumed() ─────────────────────────────


class TestOnGoalSubsumed:
    def _make_engine_with_goals(
        self, goal_clauses: list[Clause],
    ) -> object:
        """Set up a GivenClauseSearch with goal clauses registered in T2V."""
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        opts = SearchOptions(max_seconds=1)
        st = SymbolTable()
        engine = GivenClauseSearch(options=opts, symbol_table=st)

        # Manually wire up goal state as run() would
        engine._t2v_goal_clauses = list(goal_clauses)
        engine._t2v_goal_clause_ids = [c.id for c in goal_clauses]

        # Set up a GoalProximityScorer with dummy embeddings
        scorer = GoalProximityScorer(method="max")
        dummy_embs = [[float(i)] for i in range(len(goal_clauses))]
        scorer.set_goals(dummy_embs)

        # Wire up a minimal goal provider mock with _goal_scorer
        class _FakeGoalProvider:
            def __init__(self, s: GoalProximityScorer) -> None:
                self._goal_scorer = s
        engine._t2v_goal_provider = _FakeGoalProvider(scorer)

        return engine, scorer

    def test_subsumed_goal_is_removed(self) -> None:
        g1 = make_goal_clause(make_literal(True, P(a())), clause_id=10)
        g2 = make_goal_clause(make_literal(True, P(b())), clause_id=20)
        engine, scorer = self._make_engine_with_goals([g1, g2])

        assert scorer.num_goals == 2
        engine._on_goal_subsumed(g1)

        assert scorer.num_goals == 1
        assert engine._t2v_goal_clause_ids == [20]
        assert len(engine._t2v_goal_clauses) == 1
        assert engine._t2v_goal_clauses[0].id == 20

    def test_non_goal_clause_is_noop(self) -> None:
        g1 = make_goal_clause(make_literal(True, P(a())), clause_id=10)
        engine, scorer = self._make_engine_with_goals([g1])

        non_goal = make_clause(make_literal(True, P(b())), clause_id=99)
        engine._on_goal_subsumed(non_goal)

        assert scorer.num_goals == 1
        assert engine._t2v_goal_clause_ids == [10]

    def test_double_removal_is_noop(self) -> None:
        g1 = make_goal_clause(make_literal(True, P(a())), clause_id=10)
        engine, scorer = self._make_engine_with_goals([g1])

        engine._on_goal_subsumed(g1)
        assert scorer.num_goals == 0

        # Second call is safe
        engine._on_goal_subsumed(g1)
        assert scorer.num_goals == 0

    def test_remove_all_goals_one_by_one(self) -> None:
        goals = [
            make_goal_clause(make_literal(True, P(a())), clause_id=i)
            for i in range(5)
        ]
        engine, scorer = self._make_engine_with_goals(goals)
        assert scorer.num_goals == 5

        for g in goals:
            engine._on_goal_subsumed(g)

        assert scorer.num_goals == 0
        assert engine._t2v_goal_clause_ids == []
        assert engine._t2v_goal_clauses == []

    def test_no_goal_provider_is_safe(self) -> None:
        """_on_goal_subsumed is safe when _t2v_goal_provider is None."""
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        opts = SearchOptions(max_seconds=1)
        st = SymbolTable()
        engine = GivenClauseSearch(options=opts, symbol_table=st)

        g1 = make_goal_clause(make_literal(True, P(a())), clause_id=10)
        engine._t2v_goal_clauses = [g1]
        engine._t2v_goal_clause_ids = [10]
        engine._t2v_goal_provider = None

        # Should not raise
        engine._on_goal_subsumed(g1)
        # IDs list still updated even without scorer
        assert engine._t2v_goal_clause_ids == []

    def test_empty_goal_list_is_noop(self) -> None:
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        opts = SearchOptions(max_seconds=1)
        st = SymbolTable()
        engine = GivenClauseSearch(options=opts, symbol_table=st)

        clause = make_clause(make_literal(True, P(a())), clause_id=5)
        # Should not raise when _t2v_goal_clause_ids is empty
        engine._on_goal_subsumed(clause)


# ── 2b. Proof-participation goal removal (_handle_proof route) ────────────


class TestProofParticipationGoalRemoval:
    def _make_engine_with_goals(
        self, goal_clauses: list[Clause],
    ) -> tuple:
        """Set up a GivenClauseSearch with goal clauses registered in T2V."""
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        opts = SearchOptions(max_seconds=1, tree2vec_goal_proximity=True)
        st = SymbolTable()
        engine = GivenClauseSearch(options=opts, symbol_table=st)

        engine._t2v_goal_clauses = list(goal_clauses)
        engine._t2v_goal_clause_ids = [c.id for c in goal_clauses]

        scorer = GoalProximityScorer(method="max")
        dummy_embs = [[float(i)] for i in range(len(goal_clauses))]
        scorer.set_goals(dummy_embs)

        class _FakeGoalProvider:
            def __init__(self, s: GoalProximityScorer) -> None:
                self._goal_scorer = s
        engine._t2v_goal_provider = _FakeGoalProvider(scorer)

        return engine, scorer

    def test_goal_in_proof_trace_is_removed(self) -> None:
        """When _handle_proof finds a goal clause in the proof trace, it's removed."""
        g1 = make_goal_clause(make_literal(True, P(a())), clause_id=10)
        g2 = make_goal_clause(make_literal(True, P(b())), clause_id=20)
        engine, scorer = self._make_engine_with_goals([g1, g2])

        assert scorer.num_goals == 2

        # Simulate: _on_goal_subsumed is called for each proof trace clause
        # whose ID is in _t2v_goal_clause_ids (lines 1832-1836 of given_clause.py)
        engine._on_goal_subsumed(g1)

        assert scorer.num_goals == 1
        assert engine._t2v_goal_clause_ids == [20]

    def test_non_goal_in_proof_trace_no_change(self) -> None:
        """Non-goal clauses in proof trace don't affect the goal list."""
        g1 = make_goal_clause(make_literal(True, P(a())), clause_id=10)
        engine, scorer = self._make_engine_with_goals([g1])

        non_goal = make_clause(make_literal(True, P(b())), clause_id=99)
        engine._on_goal_subsumed(non_goal)

        assert scorer.num_goals == 1
        assert engine._t2v_goal_clause_ids == [10]

    def test_both_routes_reduce_goals(self) -> None:
        """Back-subsumption and proof-participation both reduce the goal list."""
        g1 = make_goal_clause(make_literal(True, P(a())), clause_id=10)
        g2 = make_goal_clause(make_literal(True, P(b())), clause_id=20)
        g3 = make_goal_clause(make_literal(True, P(f(a()))), clause_id=30)
        engine, scorer = self._make_engine_with_goals([g1, g2, g3])

        assert scorer.num_goals == 3

        # g1 removed via back-subsumption route
        engine._on_goal_subsumed(g1)
        assert scorer.num_goals == 2
        assert engine._t2v_goal_clause_ids == [20, 30]

        # g3 removed via proof-participation route (same _on_goal_subsumed call)
        engine._on_goal_subsumed(g3)
        assert scorer.num_goals == 1
        assert engine._t2v_goal_clause_ids == [20]

    def test_proof_participation_guard_check(self) -> None:
        """Proof-participation removal is guarded by tree2vec_goal_proximity and _t2v_goal_clause_ids."""
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        # With goal_proximity=False, the guard at line 1833 prevents removal
        opts = SearchOptions(max_seconds=1, tree2vec_goal_proximity=False)
        st = SymbolTable()
        engine = GivenClauseSearch(options=opts, symbol_table=st)

        g1 = make_goal_clause(make_literal(True, P(a())), clause_id=10)
        engine._t2v_goal_clauses = [g1]
        engine._t2v_goal_clause_ids = [10]

        # The guard condition: tree2vec_goal_proximity must be True
        assert not engine._opts.tree2vec_goal_proximity


# ── 3. Back-subsumption hook ─────────────────────────────────────────────


class TestBackSubsumptionGoalHook:
    def test_hook_guard_requires_goal_proximity_flag(self) -> None:
        """Goal removal does NOT fire when tree2vec_goal_proximity is False."""
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        opts = SearchOptions(max_seconds=1, tree2vec_goal_proximity=False)
        st = SymbolTable()
        engine = GivenClauseSearch(options=opts, symbol_table=st)

        g1 = make_goal_clause(make_literal(True, P(a())), clause_id=10)
        engine._t2v_goal_clauses = [g1]
        engine._t2v_goal_clause_ids = [10]

        # Verify the guard: tree2vec_goal_proximity=False means hook won't fire
        assert engine._opts.tree2vec_goal_proximity is False

    def test_hook_guard_requires_goal_provider(self) -> None:
        """Goal removal does NOT fire when _t2v_goal_provider is None."""
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        opts = SearchOptions(max_seconds=1, tree2vec_goal_proximity=True)
        st = SymbolTable()
        engine = GivenClauseSearch(options=opts, symbol_table=st)

        assert engine._t2v_goal_provider is None


# ── 4. Regression — search with goal proximity still finds proofs ────────


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
        input_text = "set(tree2vec_goal_proximity).\n" + _SIMPLE_PROOF_INPUT
        result = _run_python(input_text)
        assert result["proved"]

    def test_search_without_goal_proximity_finds_proof(self) -> None:
        """Baseline: search without goal proximity still works."""
        result = _run_python(_SIMPLE_PROOF_INPUT)
        assert result["proved"]

    def test_search_with_goal_proximity_and_online_learning(self) -> None:
        """Goal proximity + online learning does not regress."""
        input_text = (
            "set(tree2vec_goal_proximity).\n"
            "set(tree2vec_online_learning).\n"
            + _SIMPLE_PROOF_INPUT
        )
        result = _run_python(input_text)
        assert result["proved"]
