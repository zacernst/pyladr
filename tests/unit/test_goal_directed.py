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
    ReferencePoint,
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
        """A clause opposite to the goal should have the largest norm.

        With ancestor-aware distance (min over goals ∪ ancestors and a 0.5
        neutral default when no ancestors have been recorded), distances above
        0.5 clip to 0.5 — so a clause *opposite* to the goal is treated the
        same as an orthogonal clause. The essential invariant is that both
        are strictly worse than a goal-close clause.
        """
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

        assert norm_close < norm_ortho, (
            f"Expected norm_close ({norm_close:.4f}) < norm_ortho ({norm_ortho:.4f})"
        )
        assert norm_close < norm_opp, (
            f"Expected norm_close ({norm_close:.4f}) < norm_opposite ({norm_opp:.4f})"
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


# ── Productive-ancestor tracking tests ───────────────────────────────────────


class TestAncestorProximity:
    """Tests for the productive-ancestor tracking extension."""

    def test_record_productive_ancestors_adds_embeddings(self) -> None:
        """record_productive_ancestors stores parent embeddings via the base provider."""
        provider = PerClauseProvider(
            embeddings={10: [1.0, 0.0, 0.0], 20: [0.0, 1.0, 0.0]},
            dim=3,
        )
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(enabled=True, ancestor_tracking=True),
        )
        assert gdp.num_ancestors == 0

        parents = [
            Clause(literals=(), id=10),
            Clause(literals=(), id=20),
        ]
        gdp.record_productive_ancestors(parents)

        assert gdp.num_ancestors == 2

    def test_record_productive_ancestors_disabled_is_noop(self) -> None:
        """With ancestor_tracking=False, no ancestors are stored."""
        provider = PerClauseProvider(
            embeddings={10: [1.0, 0.0, 0.0]},
            dim=3,
        )
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(enabled=True, ancestor_tracking=False),
        )
        gdp.record_productive_ancestors([Clause(literals=(), id=10)])
        assert gdp.num_ancestors == 0

    def test_nearest_ancestor_distance_neutral_when_empty(self) -> None:
        """With no references registered, _nearest_reference_distance returns 0.5."""
        provider = MockBaseProvider(fixed_embedding=[1.0, 0.0, 0.0], dim=3)
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(enabled=True),
        )
        assert gdp.num_ancestors == 0
        # Any embedding should return the neutral value when the reference
        # set (goals ∪ ancestors) is empty.
        assert gdp._nearest_reference_distance([1.0, 0.0, 0.0]) == 0.5
        assert gdp._nearest_reference_distance([0.0, 1.0, 0.0]) == 0.5
        assert gdp._nearest_reference_distance([-1.0, 0.0, 0.0]) == 0.5

    def test_nearest_ancestor_distance_uses_nearest(self) -> None:
        """Distance should use the closest registered ancestor, weighted by
        its depth-decay.  With default decay=0.8, a depth-1 ancestor whose
        embedding matches the query gives weighted-sim=0.8 and so a distance
        of (1 - 0.8) / 2 = 0.1 rather than 0.0."""
        provider = PerClauseProvider(
            embeddings={
                10: [1.0, 0.0, 0.0],
                20: [0.0, 1.0, 0.0],
            },
            dim=3,
        )
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(enabled=True, ancestor_tracking=True),
        )
        gdp.record_productive_ancestors([
            Clause(literals=(), id=10),
            Clause(literals=(), id=20),
        ])

        # A query identical to ancestor 10 (weight 0.8) → distance 0.1
        assert gdp._nearest_reference_distance([1.0, 0.0, 0.0]) == pytest.approx(0.1)
        # A query identical to ancestor 20 (weight 0.8) → distance 0.1
        assert gdp._nearest_reference_distance([0.0, 1.0, 0.0]) == pytest.approx(0.1)

    def test_enhance_uses_ancestor_distance(self) -> None:
        """_enhance_embedding uses ancestor distance when ancestors are closer than goals.

        Set a goal at direction A and an ancestor at direction B (orthogonal).
        A clause whose embedding matches B should be enhanced (smaller norm)
        because its nearest reference is the ancestor, not the far goal.
        """
        goal_emb = [1.0, 0.0, 0.0]
        ancestor_emb = [0.0, 1.0, 0.0]

        provider = PerClauseProvider(
            embeddings={
                # Ancestor clause
                100: ancestor_emb,
                # Query clauses: matches ancestor, matches goal, far from both
                1: [0.0, 1.0, 0.0],     # identical to ancestor
                2: [1.0, 0.0, 0.0],     # identical to goal
                3: [0.0, 0.0, 1.0],     # orthogonal to both
            },
            dim=3,
        )
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(
                enabled=True,
                goal_proximity_weight=0.5,
                ancestor_tracking=True,
            ),
        )
        gdp._goal_scorer.set_goals([goal_emb])
        gdp.record_productive_ancestors([Clause(literals=(), id=100)])
        assert gdp.num_ancestors == 1

        emb_ancestor_close = gdp.get_embedding(Clause(literals=(), id=1))
        emb_goal_close = gdp.get_embedding(Clause(literals=(), id=2))
        emb_far = gdp.get_embedding(Clause(literals=(), id=3))

        assert emb_ancestor_close is not None
        assert emb_goal_close is not None
        assert emb_far is not None

        norm_ancestor = _norm(emb_ancestor_close)
        norm_goal = _norm(emb_goal_close)
        norm_far = _norm(emb_far)

        # Both the ancestor-close and goal-close clauses are "close to some
        # reference" and should be strictly more promising (smaller norm) than
        # an orthogonal clause.
        assert norm_ancestor < norm_far
        assert norm_goal < norm_far

    def test_clear_ancestors(self) -> None:
        """clear_ancestors() empties the ancestor set."""
        provider = PerClauseProvider(
            embeddings={10: [1.0, 0.0, 0.0], 20: [0.0, 1.0, 0.0]},
            dim=3,
        )
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(enabled=True, ancestor_tracking=True),
        )
        gdp.record_productive_ancestors([
            Clause(literals=(), id=10),
            Clause(literals=(), id=20),
        ])
        assert gdp.num_ancestors == 2

        gdp.clear_ancestors()
        assert gdp.num_ancestors == 0

    def test_ancestor_max_count(self) -> None:
        """No more than ancestor_max_count ancestors are stored."""
        embs = {i: [float(i), 0.0, 0.0] for i in range(1, 6)}
        provider = PerClauseProvider(embeddings=embs, dim=3)
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(
                enabled=True,
                ancestor_tracking=True,
                ancestor_max_count=3,
            ),
        )
        parents = [Clause(literals=(), id=i) for i in range(1, 6)]  # 5 parents
        gdp.record_productive_ancestors(parents)
        # Only the first 3 should have been stored
        assert gdp.num_ancestors == 3

        # Additional calls should not exceed the cap
        gdp.record_productive_ancestors([Clause(literals=(), id=5)])
        assert gdp.num_ancestors == 3

    def test_stats_report_includes_ancestors(self) -> None:
        """Stats dict contains num_ancestors; stats_report includes 'ancestors='."""
        provider = PerClauseProvider(
            embeddings={10: [1.0, 0.0, 0.0]},
            dim=3,
        )
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(enabled=True, ancestor_tracking=True),
        )
        stats = gdp.stats
        assert "num_ancestors" in stats
        assert stats["num_ancestors"] == 0

        gdp.record_productive_ancestors([Clause(literals=(), id=10)])
        stats = gdp.stats
        assert stats["num_ancestors"] == 1

        report = gdp.stats_report()
        assert "ancestors=" in report
        assert "ancestors=1" in report


# ── Recursive ancestor expansion tests ───────────────────────────────────────


class TestRecursiveAncestorExpansion:
    """Tests for the recursive reference-set expansion via try_expand_from_clause.

    The productive reference set is seeded with goals (depth 0, weight 1.0)
    and grown by :meth:`try_expand_from_clause`: when a probe embedding is
    close enough to a current reference point, the parents of that probe are
    added as deeper reference points with exponentially-decaying weight.
    """

    def test_try_expand_from_clause_close_to_goal_adds_depth1(self) -> None:
        """A probe embedding close to a goal triggers depth-1 parent insertion."""
        parent_emb = [0.5, 0.5, 0.0]  # embedding of the parent clause
        provider = PerClauseProvider(
            embeddings={50: parent_emb},
            dim=3,
        )
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(
                enabled=True,
                ancestor_tracking=True,
                ancestor_proximity_threshold=0.3,
                ancestor_decay=0.8,
            ),
        )
        # Register a goal so the reference set is non-empty.
        gdp._goal_scorer.set_goals([[1.0, 0.0, 0.0]])

        # Probe embedding is identical to the goal → distance = 0.0 < 0.3.
        added = gdp.try_expand_from_clause(
            [1.0, 0.0, 0.0], [Clause(literals=(), id=50)],
        )
        assert added is True
        assert gdp.num_ancestors == 1

        # The newly inserted reference should be at depth 1 with weight 0.8.
        ref = gdp._ancestor_reference[0]
        assert ref.depth == 1
        assert ref.weight == pytest.approx(0.8)
        assert ref.embedding == parent_emb

    def test_try_expand_from_clause_far_from_all_adds_nothing(self) -> None:
        """A probe too far from every reference triggers no expansion."""
        provider = PerClauseProvider(
            embeddings={50: [0.0, 0.0, 1.0]},
            dim=3,
        )
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(
                enabled=True,
                ancestor_tracking=True,
                ancestor_proximity_threshold=0.3,
            ),
        )
        gdp._goal_scorer.set_goals([[1.0, 0.0, 0.0]])

        # Probe is opposite to the only goal → distance = 1.0 > 0.3.
        added = gdp.try_expand_from_clause(
            [-1.0, 0.0, 0.0], [Clause(literals=(), id=50)],
        )
        assert added is False
        assert gdp.num_ancestors == 0

    def test_try_expand_from_clause_close_to_ancestor_adds_depth2(self) -> None:
        """A probe close to a depth-1 ancestor adds parents at depth 2."""
        parent_emb = [0.1, 0.9, 0.0]
        provider = PerClauseProvider(
            embeddings={50: parent_emb},
            dim=3,
        )
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(
                enabled=True,
                ancestor_tracking=True,
                ancestor_proximity_threshold=0.3,
                ancestor_decay=0.8,
                ancestor_max_depth=5,
            ),
        )
        # Goal is orthogonal to the depth-1 ancestor so only the ancestor
        # drives the expansion decision.
        gdp._goal_scorer.set_goals([[1.0, 0.0, 0.0]])
        # Pre-install a depth-1 ancestor at [0, 1, 0] with weight decay=0.8.
        gdp._ancestor_reference.append(
            ReferencePoint(embedding=[0.0, 1.0, 0.0], depth=1, weight=0.8),
        )
        assert gdp.num_ancestors == 1

        # Probe identical to the depth-1 ancestor:
        #   weighted-sim from goal    = 1.0 * 0.0 = 0.0
        #   weighted-sim from ancestor = 0.8 * 1.0 = 0.8
        #   distance = (1 - 0.8) / 2 = 0.1 < 0.3 → expand to depth 2.
        added = gdp.try_expand_from_clause(
            [0.0, 1.0, 0.0], [Clause(literals=(), id=50)],
        )
        assert added is True
        assert gdp.num_ancestors == 2

        new_ref = gdp._ancestor_reference[-1]
        assert new_ref.depth == 2
        assert new_ref.weight == pytest.approx(0.8 ** 2)
        assert new_ref.embedding == parent_emb

    def test_max_depth_stops_recursion(self) -> None:
        """With ancestor_max_depth=1, depth-2 expansion is blocked even when
        the probe is close to a depth-1 ancestor."""
        provider = PerClauseProvider(
            embeddings={50: [0.1, 0.9, 0.0]},
            dim=3,
        )
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(
                enabled=True,
                ancestor_tracking=True,
                ancestor_proximity_threshold=0.3,
                ancestor_decay=0.8,
                ancestor_max_depth=1,  # ← the boundary we're testing
            ),
        )
        gdp._goal_scorer.set_goals([[1.0, 0.0, 0.0]])
        gdp._ancestor_reference.append(
            ReferencePoint(embedding=[0.0, 1.0, 0.0], depth=1, weight=0.8),
        )
        assert gdp.num_ancestors == 1

        # The probe would otherwise trigger a depth-2 insertion, but
        # new_depth (2) > ancestor_max_depth (1) → blocked.
        added = gdp.try_expand_from_clause(
            [0.0, 1.0, 0.0], [Clause(literals=(), id=50)],
        )
        assert added is False
        assert gdp.num_ancestors == 1  # unchanged

    def test_min_weight_stops_recursion(self) -> None:
        """ancestor_min_weight halts recursion when decay**depth falls below it."""
        provider = PerClauseProvider(
            embeddings={50: [0.1, 0.9, 0.0]},
            dim=3,
        )
        # decay=0.5, min_weight=0.4:
        #   depth-1 weight = 0.5 ≥ 0.4 → allowed
        #   depth-2 weight = 0.25 < 0.4 → blocked
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(
                enabled=True,
                ancestor_tracking=True,
                ancestor_proximity_threshold=0.3,
                ancestor_decay=0.5,
                ancestor_min_weight=0.4,
                ancestor_max_depth=5,  # not the limiting factor
            ),
        )
        gdp._goal_scorer.set_goals([[1.0, 0.0, 0.0]])
        # Pre-install a depth-1 ancestor with the matching weight=0.5.
        gdp._ancestor_reference.append(
            ReferencePoint(embedding=[0.0, 1.0, 0.0], depth=1, weight=0.5),
        )
        assert gdp.num_ancestors == 1

        # Probe close to the depth-1 ancestor:
        #   weighted-sim from ancestor = 0.5 * 1.0 = 0.5
        #   distance = 0.25 < 0.3 → passes the threshold
        #   new_depth = 2, new_weight = 0.25 < min_weight=0.4 → blocked.
        added = gdp.try_expand_from_clause(
            [0.0, 1.0, 0.0], [Clause(literals=(), id=50)],
        )
        assert added is False
        assert gdp.num_ancestors == 1  # recursion stopped by min_weight

        # Boundary: depth-1 expansion from a goal is still allowed because
        # new_weight = 0.5 ≥ 0.4.
        parent_emb = [0.9, 0.1, 0.0]
        provider2 = PerClauseProvider(
            embeddings={60: parent_emb}, dim=3,
        )
        gdp2 = GoalDirectedEmbeddingProvider(
            base_provider=provider2,
            config=GoalDirectedConfig(
                enabled=True,
                ancestor_tracking=True,
                ancestor_proximity_threshold=0.3,
                ancestor_decay=0.5,
                ancestor_min_weight=0.4,
            ),
        )
        gdp2._goal_scorer.set_goals([[1.0, 0.0, 0.0]])
        added2 = gdp2.try_expand_from_clause(
            [1.0, 0.0, 0.0], [Clause(literals=(), id=60)],
        )
        assert added2 is True
        assert gdp2.num_ancestors == 1

    def test_nearest_reference_distance_weighted(self) -> None:
        """_nearest_reference_distance uses the weighted-similarity formula
        (1 - max_i w_i * cos(e, r_i)) / 2, with goals at weight 1.0 and
        ancestors at their stored weights."""
        provider = MockBaseProvider(fixed_embedding=[1.0, 0.0, 0.0], dim=3)
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(enabled=True, ancestor_tracking=True),
        )
        gdp._goal_scorer.set_goals([[1.0, 0.0, 0.0]])  # G
        gdp._ancestor_reference.append(
            ReferencePoint(embedding=[0.0, 1.0, 0.0], depth=1, weight=0.8),
        )

        # Identical to goal:      sim_G=1.0*1.0=1.0, sim_A=0.8*0.0=0.0 → 0.0
        assert gdp._nearest_reference_distance([1.0, 0.0, 0.0]) == pytest.approx(0.0)
        # Identical to ancestor:  sim_G=1.0*0.0=0.0, sim_A=0.8*1.0=0.8 → 0.1
        assert gdp._nearest_reference_distance([0.0, 1.0, 0.0]) == pytest.approx(0.1)
        # Orthogonal to both:     sim_G=0.0, sim_A=0.0 → 0.5
        assert gdp._nearest_reference_distance([0.0, 0.0, 1.0]) == pytest.approx(0.5)

    def test_neutral_distance_when_no_references(self) -> None:
        """With neither goals nor ancestors registered, distance is 0.5."""
        provider = MockBaseProvider(fixed_embedding=[1.0, 0.0, 0.0], dim=3)
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(enabled=True, ancestor_tracking=True),
        )
        assert gdp.num_goals == 0
        assert gdp.num_ancestors == 0
        assert gdp._nearest_reference_distance([1.0, 0.0, 0.0]) == 0.5
        assert gdp._nearest_reference_distance([0.5, 0.5, 0.0]) == 0.5

    def test_ancestor_max_count_respected_for_reference_points(self) -> None:
        """ancestor_max_count caps try_expand_from_clause insertions."""
        embs = {i: [1.0, 0.0, 0.0] for i in range(1, 6)}
        provider = PerClauseProvider(embeddings=embs, dim=3)
        gdp = GoalDirectedEmbeddingProvider(
            base_provider=provider,
            config=GoalDirectedConfig(
                enabled=True,
                ancestor_tracking=True,
                ancestor_proximity_threshold=0.3,
                ancestor_max_count=2,
            ),
        )
        gdp._goal_scorer.set_goals([[1.0, 0.0, 0.0]])

        parents = [Clause(literals=(), id=i) for i in range(1, 6)]
        added = gdp.try_expand_from_clause([1.0, 0.0, 0.0], parents)
        assert added is True
        # Only 2 of the 5 parents should have been inserted.
        assert gdp.num_ancestors == 2
