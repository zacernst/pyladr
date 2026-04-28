"""Tests for goal_directed.py deskolemization and scoring direction.

Verifies:
- _deskolemize_clause: signs forced positive, constants replaced by variables
- GoalDistanceScorer: cosine-based distance [0, 1] with correct semantics
- _enhance_embedding: goal-close clauses get smaller norms (more promising)
"""

from __future__ import annotations

import math

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.search.goal_directed import (
    GoalDirectedConfig,
    GoalDirectedEmbeddingProvider,
    GoalDistanceScorer,
    _deskolemize_clause,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _const(symnum: int) -> Term:
    return get_rigid_term(symnum, 0)


def _func(symnum: int, *args: Term) -> Term:
    return get_rigid_term(symnum, len(args), args)


def _make_clause(literals: list[Literal], cid: int = 0) -> Clause:
    c = Clause(literals=tuple(literals), id=cid)
    return c


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


class MockBaseProvider:
    """Minimal EmbeddingProvider-compatible mock. No torch required."""

    def __init__(self, fixed_embedding: list[float] | None = None, dim: int = 3):
        self._fixed = fixed_embedding
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def get_embedding(self, clause: Clause) -> list[float] | None:
        return list(self._fixed) if self._fixed is not None else None

    def get_embeddings_batch(
        self, clauses: list[Clause],
    ) -> list[list[float] | None]:
        return [self.get_embedding(c) for c in clauses]


class PerClauseProvider:
    """Mock provider that returns different embeddings based on clause id."""

    def __init__(self, embeddings: dict[int, list[float]], dim: int = 3):
        self._embeddings = embeddings
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def get_embedding(self, clause: Clause) -> list[float] | None:
        return list(self._embeddings.get(clause.id, []))  or None

    def get_embeddings_batch(
        self, clauses: list[Clause],
    ) -> list[list[float] | None]:
        return [self.get_embedding(c) for c in clauses]


# ── _deskolemize_clause tests ────────────────────────────────────────────────


class TestDeskolemizeClause:
    """_deskolemize_clause: strip signs, replace constants with variables."""

    def test_signs_forced_positive(self) -> None:
        """All output literals must have sign=True."""
        # Build clause: -P(a) | -Q(b)
        p_a = _func(1, _const(10))       # P(a)
        q_b = _func(2, _const(11))       # Q(b)
        clause = _make_clause([
            Literal(sign=False, atom=p_a),
            Literal(sign=False, atom=q_b),
        ])
        result = _deskolemize_clause(clause)
        for lit in result.literals:
            assert lit.sign is True, f"Expected positive literal, got sign={lit.sign}"

    def test_constants_replaced_by_variables(self) -> None:
        """All constants in the clause should become variables (private_symbol >= 0)."""
        # Build clause: -P(a, b)   (a=const10, b=const11)
        atom = _func(1, _const(10), _const(11))
        clause = _make_clause([Literal(sign=False, atom=atom)])
        result = _deskolemize_clause(clause)

        # The atom's args should now be variables
        result_atom = result.literals[0].atom
        for arg in result_atom.args:
            assert arg.is_variable, f"Expected variable, got {arg!r}"

    def test_distinct_constants_get_distinct_variables(self) -> None:
        """Each distinct constant maps to a distinct variable number."""
        # P(a, b, a)  — a and b are distinct constants
        atom = _func(1, _const(10), _const(11), _const(10))
        clause = _make_clause([Literal(sign=True, atom=atom)])
        result = _deskolemize_clause(clause)

        args = result.literals[0].atom.args
        # a→var0, b→var1, a→var0 again
        assert args[0].varnum == args[2].varnum, "Same constant should map to same variable"
        assert args[0].varnum != args[1].varnum, "Distinct constants should map to distinct variables"

    def test_existing_variables_preserved(self) -> None:
        """Variables already in the clause remain as variables."""
        # P(x, a)  where x is variable 0, a is constant 10
        var_x = get_variable_term(0)
        atom = _func(1, var_x, _const(10))
        clause = _make_clause([Literal(sign=True, atom=atom)])
        result = _deskolemize_clause(clause)

        result_atom = result.literals[0].atom
        # First arg (was variable) should still be variable
        assert result_atom.args[0].is_variable
        # Second arg (was constant) should now be variable
        assert result_atom.args[1].is_variable

    def test_nested_constants_replaced(self) -> None:
        """Constants nested inside function terms are also replaced."""
        # P(f(a, b))  where f=symnum2, a=const10, b=const11
        inner = _func(2, _const(10), _const(11))
        atom = _func(1, inner)
        clause = _make_clause([Literal(sign=False, atom=atom)])
        result = _deskolemize_clause(clause)

        # Dig into nested structure
        result_inner = result.literals[0].atom.args[0]
        for arg in result_inner.args:
            assert arg.is_variable, f"Nested constant should be replaced: {arg!r}"

    def test_clause_id_preserved(self) -> None:
        """The result clause retains the original clause's id."""
        atom = _func(1, _const(10))
        clause = _make_clause([Literal(sign=False, atom=atom)], cid=42)
        result = _deskolemize_clause(clause)
        assert result.id == 42

    def test_empty_clause(self) -> None:
        """Empty clause (no literals) should work fine."""
        clause = _make_clause([])
        result = _deskolemize_clause(clause)
        assert result.literals == ()

    def test_multi_literal_clause(self) -> None:
        """Multiple literals with mixed signs all get sign=True."""
        lit1 = Literal(sign=True, atom=_func(1, _const(10)))
        lit2 = Literal(sign=False, atom=_func(2, _const(11)))
        lit3 = Literal(sign=False, atom=_func(3, _const(12)))
        clause = _make_clause([lit1, lit2, lit3])
        result = _deskolemize_clause(clause)

        assert len(result.literals) == 3
        for lit in result.literals:
            assert lit.sign is True


# ── GoalDistanceScorer tests ─────────────────────────────────────────────────


class TestGoalDistanceScorer:
    """GoalDistanceScorer: cosine-distance semantics for goal-directed search."""

    def test_identical_embedding_distance_zero(self) -> None:
        """Embedding identical to goal → distance 0.0."""
        scorer = GoalDistanceScorer()
        scorer.set_goals([[1.0, 0.0, 0.0]])
        assert scorer.nearest_goal_distance([1.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_orthogonal_embedding_distance_half(self) -> None:
        """Embedding orthogonal to goal → distance 0.5."""
        scorer = GoalDistanceScorer()
        scorer.set_goals([[1.0, 0.0, 0.0]])
        assert scorer.nearest_goal_distance([0.0, 1.0, 0.0]) == pytest.approx(0.5)

    def test_opposite_embedding_distance_one(self) -> None:
        """Embedding opposite to goal → distance 1.0."""
        scorer = GoalDistanceScorer()
        scorer.set_goals([[1.0, 0.0, 0.0]])
        assert scorer.nearest_goal_distance([-1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_no_goals_returns_neutral(self) -> None:
        """No goals registered → neutral distance 0.5."""
        scorer = GoalDistanceScorer()
        assert scorer.nearest_goal_distance([1.0, 0.0, 0.0]) == pytest.approx(0.5)

    def test_none_embedding_returns_neutral(self) -> None:
        """None embedding → neutral distance 0.5."""
        scorer = GoalDistanceScorer()
        scorer.set_goals([[1.0, 0.0, 0.0]])
        assert scorer.nearest_goal_distance(None) == pytest.approx(0.5)

    def test_multiple_goals_uses_nearest(self) -> None:
        """With multiple goals, returns distance to the nearest one."""
        scorer = GoalDistanceScorer()
        scorer.set_goals([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # [1, 0, 0] is identical to first goal → distance 0.0
        assert scorer.nearest_goal_distance([1.0, 0.0, 0.0]) == pytest.approx(0.0)
        # [0, 0, 1] is orthogonal to both → distance 0.5
        assert scorer.nearest_goal_distance([0.0, 0.0, 1.0]) == pytest.approx(0.5)

    def test_clear_removes_all_goals(self) -> None:
        """After clear(), scorer returns neutral distance."""
        scorer = GoalDistanceScorer()
        scorer.set_goals([[1.0, 0.0, 0.0]])
        scorer.clear()
        assert scorer.nearest_goal_distance([1.0, 0.0, 0.0]) == pytest.approx(0.5)

    def test_farthest_goal_distance(self) -> None:
        """farthest_goal_distance uses the least similar goal."""
        scorer = GoalDistanceScorer()
        scorer.set_goals([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        # [1,0,0] is identical to first (sim=1) and opposite to second (sim=-1)
        # farthest uses min sim = -1, distance = (1 - (-1)) / 2 = 1.0
        assert scorer.farthest_goal_distance([1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_num_goals(self) -> None:
        scorer = GoalDistanceScorer()
        assert scorer.num_goals == 0
        scorer.set_goals([[1.0, 0.0], [0.0, 1.0]])
        assert scorer.num_goals == 2

    def test_remove_goal(self) -> None:
        scorer = GoalDistanceScorer()
        scorer.set_goals([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        scorer.remove_goal(0)
        assert scorer.num_goals == 1


# ── _enhance_embedding direction invariant tests ─────────────────────────────


class TestEnhanceEmbeddingDirection:
    """Goal-close clauses get smaller norms → higher proof potential score."""

    def test_goal_close_clause_has_smaller_norm(self) -> None:
        """A clause with embedding identical to goal should produce a smaller
        output norm than one with an orthogonal embedding."""
        goal_emb = [1.0, 0.0, 0.0]

        # Provider returns goal-identical embedding for clause 1,
        # orthogonal for clause 2
        provider = PerClauseProvider(
            embeddings={1: [1.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0]},
            dim=3,
        )
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(enabled=True, goal_proximity_weight=0.3),
        )
        # Register the goal directly via scorer (bypassing deskolemize which
        # needs a real clause → we just set the embedding directly)
        gdp._goal_scorer.set_goals([goal_emb])

        clause_close = Clause(literals=(), id=1)
        clause_ortho = Clause(literals=(), id=2)

        emb_close = gdp.get_embedding(clause_close)
        emb_ortho = gdp.get_embedding(clause_ortho)

        assert emb_close is not None
        assert emb_ortho is not None

        norm_close = _norm(emb_close)
        norm_ortho = _norm(emb_ortho)

        assert norm_close < norm_ortho, (
            f"Goal-close clause should have smaller norm ({norm_close}) "
            f"than orthogonal clause ({norm_ortho})"
        )

    def test_opposite_clause_has_largest_norm(self) -> None:
        """A clause opposite to the goal should have the largest norm."""
        goal_emb = [1.0, 0.0, 0.0]

        provider = PerClauseProvider(
            embeddings={
                1: [1.0, 0.0, 0.0],   # identical to goal
                2: [0.0, 1.0, 0.0],   # orthogonal
                3: [-1.0, 0.0, 0.0],  # opposite
            },
            dim=3,
        )
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(enabled=True, goal_proximity_weight=0.3),
        )
        gdp._goal_scorer.set_goals([goal_emb])

        emb_close = gdp.get_embedding(Clause(literals=(), id=1))
        emb_ortho = gdp.get_embedding(Clause(literals=(), id=2))
        emb_opp = gdp.get_embedding(Clause(literals=(), id=3))

        norm_close = _norm(emb_close)
        norm_ortho = _norm(emb_ortho)
        norm_opp = _norm(emb_opp)

        assert norm_close < norm_ortho < norm_opp, (
            f"Expected norm_close ({norm_close:.4f}) < norm_ortho ({norm_ortho:.4f}) "
            f"< norm_opposite ({norm_opp:.4f})"
        )

    def test_disabled_is_passthrough(self) -> None:
        """With enabled=False, get_embedding returns the base embedding unchanged."""
        base_emb = [1.0, 2.0, 3.0]
        provider = MockBaseProvider(fixed_embedding=base_emb, dim=3)
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(enabled=False),
        )
        gdp._goal_scorer.set_goals([[1.0, 0.0, 0.0]])

        clause = Clause(literals=(), id=1)
        result = gdp.get_embedding(clause)
        assert result == base_emb

    def test_no_goals_no_modulation(self) -> None:
        """With goals enabled but no goals registered, distance=0.5 → partial scale."""
        base_emb = [1.0, 0.0, 0.0]
        provider = MockBaseProvider(fixed_embedding=base_emb, dim=3)
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(enabled=True, goal_proximity_weight=0.3),
        )
        # No goals registered — distance will be 0.5
        clause = Clause(literals=(), id=1)
        result = gdp.get_embedding(clause)
        assert result is not None

        # With distance=0.5, scale = 1.0 - 0.3 * (1.0 - 0.5) = 0.85
        expected_scale = 1.0 - 0.3 * 0.5
        assert result[0] == pytest.approx(1.0 * expected_scale)
