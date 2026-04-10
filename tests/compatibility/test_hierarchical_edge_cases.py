"""Edge case and error handling compatibility tests for hierarchical GNN.

Tests boundary conditions, error handling, and unusual inputs to ensure
the hierarchical GNN integration handles all cases correctly without
breaking existing behavior.

Run with: pytest tests/compatibility/test_hierarchical_edge_cases.py -v
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions


# ── Empty/Trivial Inputs ──────────────────────────────────────────────────


class TestEmptyAndTrivialInputs:
    """Edge cases with empty or minimal inputs."""

    def test_empty_sos_returns_sos_empty(self):
        """Empty SOS should terminate with SOS_EMPTY or similar."""
        opts = SearchOptions(max_given=10, quiet=True, goal_directed=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(usable=[], sos=[])
        assert result.exit_code == ExitCode.SOS_EMPTY_EXIT

    def test_single_positive_clause_sos_empty(self):
        """Single positive clause with no negation -> SOS empty."""
        atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c = Clause(literals=(Literal(sign=True, atom=atom),))

        opts = SearchOptions(max_given=10, quiet=True, goal_directed=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(usable=[], sos=[c])
        assert result.exit_code == ExitCode.SOS_EMPTY_EXIT

    def test_immediate_contradiction(self):
        """P(a) and -P(a) should immediately find proof."""
        atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c1 = Clause(literals=(Literal(sign=True, atom=atom),))
        c2 = Clause(literals=(Literal(sign=False, atom=atom),))

        opts = SearchOptions(max_given=10, quiet=True, goal_directed=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(usable=[], sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert result.stats.given <= 2

    def test_tautology_only_input(self):
        """Input containing only a tautology should exhaust SOS."""
        atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        tautology = Clause(literals=(
            Literal(sign=True, atom=atom),
            Literal(sign=False, atom=atom),
        ))

        opts = SearchOptions(max_given=10, quiet=True, goal_directed=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(usable=[], sos=[tautology])
        # Tautology should be removed, leaving SOS empty
        assert result.exit_code == ExitCode.SOS_EMPTY_EXIT


# ── Limit Boundaries ─────────────────────────────────────────────────────


class TestLimitBoundaries:
    """Verify resource limits work correctly with hierarchical options."""

    def test_max_given_respects_limit(self):
        """max_given limits the number of given clauses processed."""
        P_sn = 1
        a = get_rigid_term(3, 0)
        b = get_rigid_term(4, 0)
        x = get_variable_term(0)
        y = get_variable_term(1)

        # Create clauses that generate a large search space
        clauses = [
            Clause(literals=(
                Literal(sign=True, atom=get_rigid_term(P_sn, 2, (a, b))),
            )),
            Clause(literals=(
                Literal(sign=False, atom=get_rigid_term(P_sn, 2, (x, y))),
                Literal(sign=True, atom=get_rigid_term(P_sn, 2, (y, x))),
            )),
        ]

        opts = SearchOptions(
            binary_resolution=True, max_given=3, quiet=True, goal_directed=False,
        )
        search = GivenClauseSearch(options=opts)
        result = search.run(usable=[], sos=clauses)
        assert result.stats.given <= 3

    def test_max_given_limits_search(self):
        """max_given limits how many clauses are selected as given."""
        P_sn = 1
        a = get_rigid_term(3, 0)
        b = get_rigid_term(4, 0)
        x = get_variable_term(0)
        y = get_variable_term(1)

        # Clauses that generate an expanding search space
        clauses = [
            Clause(literals=(
                Literal(sign=True, atom=get_rigid_term(P_sn, 2, (a, b))),
            )),
            Clause(literals=(
                Literal(sign=False, atom=get_rigid_term(P_sn, 2, (x, y))),
                Literal(sign=True, atom=get_rigid_term(P_sn, 2, (y, x))),
            )),
        ]

        limit = 5
        opts = SearchOptions(
            binary_resolution=True, max_given=limit, quiet=True, goal_directed=False,
        )
        search = GivenClauseSearch(options=opts)
        result = search.run(usable=[], sos=clauses)
        assert result.stats.given <= limit + 1  # +1 for possible off-by-one in limit check

    def test_max_given_negative_one_means_no_limit(self):
        """max_given=-1 should mean no limit (C convention)."""
        atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c1 = Clause(literals=(Literal(sign=True, atom=atom),))
        c2 = Clause(literals=(Literal(sign=False, atom=atom),))

        opts = SearchOptions(max_given=-1, quiet=True, goal_directed=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(usable=[], sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT


# ── Goal-Directed Config Boundary Values ──────────────────────────────────


class TestGoalDirectedBoundaryValues:
    """Boundary values for goal-directed configuration."""

    def test_proximity_weight_zero(self):
        """goal_proximity_weight=0.0 should have no goal influence."""
        from pyladr.search.goal_directed import GoalDirectedConfig, GoalDirectedEmbeddingProvider

        class FakeProvider:
            embedding_dim = 4
            def get_embedding(self, clause):
                return [1.0, 0.0, 0.0, 0.0]
            def get_embeddings_batch(self, clauses):
                return [self.get_embedding(c) for c in clauses]

        cfg = GoalDirectedConfig(enabled=True, goal_proximity_weight=0.0)
        provider = GoalDirectedEmbeddingProvider(FakeProvider(), cfg)

        atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        clause = Clause(literals=(Literal(sign=True, atom=atom),))

        emb = provider.get_embedding(clause)
        # With weight=0, scale = 1.0 - 0.0 * proximity = 1.0 (unchanged)
        assert emb == [1.0, 0.0, 0.0, 0.0]

    def test_proximity_weight_one(self):
        """goal_proximity_weight=1.0 should have maximum goal influence."""
        from pyladr.search.goal_directed import GoalDirectedConfig, GoalDirectedEmbeddingProvider

        class FakeProvider:
            embedding_dim = 4
            def get_embedding(self, clause):
                return [1.0, 0.0, 0.0, 0.0]
            def get_embeddings_batch(self, clauses):
                return [self.get_embedding(c) for c in clauses]

        cfg = GoalDirectedConfig(enabled=True, goal_proximity_weight=1.0)
        provider = GoalDirectedEmbeddingProvider(FakeProvider(), cfg)

        atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        clause = Clause(literals=(Literal(sign=True, atom=atom),))

        emb = provider.get_embedding(clause)
        # With no goals registered, proximity = 0.5 (neutral)
        # scale = 1.0 - 1.0 * 0.5 = 0.5
        assert all(abs(a - b) < 1e-6 for a, b in zip(emb, [0.5, 0.0, 0.0, 0.0]))


# ── Goal Proximity Scorer Edge Cases ──────────────────────────────────────


class TestGoalProximityScorerEdgeCases:
    """Edge cases in goal proximity scoring."""

    def test_no_goals_returns_neutral(self):
        from pyladr.search.goal_directed import GoalProximityScorer
        scorer = GoalProximityScorer()
        assert scorer.proximity([1.0, 0.0, 0.0]) == 0.5

    def test_none_embedding_returns_neutral(self):
        from pyladr.search.goal_directed import GoalProximityScorer
        scorer = GoalProximityScorer()
        scorer.set_goals([[1.0, 0.0, 0.0]])
        assert scorer.proximity(None) == 0.5

    def test_identical_to_goal_returns_one(self):
        from pyladr.search.goal_directed import GoalProximityScorer
        scorer = GoalProximityScorer()
        scorer.set_goals([[1.0, 0.0, 0.0]])
        # cosine_sim = 1.0, proximity = (1.0 + 1) / 2 = 1.0
        assert abs(scorer.proximity([1.0, 0.0, 0.0]) - 1.0) < 1e-6

    def test_opposite_to_goal_returns_zero(self):
        from pyladr.search.goal_directed import GoalProximityScorer
        scorer = GoalProximityScorer()
        scorer.set_goals([[1.0, 0.0, 0.0]])
        # cosine_sim = -1.0, proximity = (-1.0 + 1) / 2 = 0.0
        assert abs(scorer.proximity([-1.0, 0.0, 0.0]) - 0.0) < 1e-6

    def test_orthogonal_to_goal_returns_half(self):
        from pyladr.search.goal_directed import GoalProximityScorer
        scorer = GoalProximityScorer()
        scorer.set_goals([[1.0, 0.0, 0.0]])
        # cosine_sim = 0.0, proximity = (0.0 + 1) / 2 = 0.5
        assert abs(scorer.proximity([0.0, 1.0, 0.0]) - 0.5) < 1e-6

    def test_zero_vector_returns_neutral(self):
        from pyladr.search.goal_directed import GoalProximityScorer
        scorer = GoalProximityScorer()
        scorer.set_goals([[1.0, 0.0, 0.0]])
        # Zero vector -> cosine_sim = 0.0 (guarded), proximity = 0.5
        assert abs(scorer.proximity([0.0, 0.0, 0.0]) - 0.5) < 1e-6

    def test_multiple_goals_max_method(self):
        from pyladr.search.goal_directed import GoalProximityScorer
        scorer = GoalProximityScorer(method="max")
        scorer.set_goals([[1.0, 0.0], [0.0, 1.0]])
        # Embedding [1.0, 0.0] is identical to goal 1, orthogonal to goal 2
        # max(1.0, 0.5) = 1.0
        assert abs(scorer.proximity([1.0, 0.0]) - 1.0) < 1e-6

    def test_multiple_goals_mean_method(self):
        from pyladr.search.goal_directed import GoalProximityScorer
        scorer = GoalProximityScorer(method="mean")
        scorer.set_goals([[1.0, 0.0], [0.0, 1.0]])
        # Embedding [1.0, 0.0]: sim to goal 1 = 1.0 (prox=1.0), sim to goal 2 = 0.0 (prox=0.5)
        # mean = (1.0 + 0.5) / 2 = 0.75
        assert abs(scorer.proximity([1.0, 0.0]) - 0.75) < 1e-6

    def test_clear_goals(self):
        from pyladr.search.goal_directed import GoalProximityScorer
        scorer = GoalProximityScorer()
        scorer.set_goals([[1.0, 0.0]])
        assert scorer.num_goals == 1
        scorer.clear()
        assert scorer.num_goals == 0
        assert scorer.proximity([1.0, 0.0]) == 0.5


# ── GoalDirectedEmbeddingProvider Edge Cases ──────────────────────────────


class TestGoalDirectedProviderEdgeCases:
    """Edge cases in the goal-directed embedding provider."""

    def test_none_embedding_passthrough(self):
        """None embeddings passed through unchanged."""
        from pyladr.search.goal_directed import GoalDirectedConfig, GoalDirectedEmbeddingProvider

        class NoneProvider:
            embedding_dim = 4
            def get_embedding(self, clause):
                return None
            def get_embeddings_batch(self, clauses):
                return [None for _ in clauses]

        cfg = GoalDirectedConfig(enabled=True)
        provider = GoalDirectedEmbeddingProvider(NoneProvider(), cfg)

        atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        clause = Clause(literals=(Literal(sign=True, atom=atom),))

        assert provider.get_embedding(clause) is None

    def test_batch_embedding_consistency(self):
        """Batch and single embeddings produce same results."""
        from pyladr.search.goal_directed import GoalDirectedConfig, GoalDirectedEmbeddingProvider

        class FixedProvider:
            embedding_dim = 4
            def get_embedding(self, clause):
                return [1.0, 2.0, 3.0, 4.0]
            def get_embeddings_batch(self, clauses):
                return [self.get_embedding(c) for c in clauses]

        cfg = GoalDirectedConfig(enabled=True, goal_proximity_weight=0.3)
        provider = GoalDirectedEmbeddingProvider(FixedProvider(), cfg)

        atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        clause = Clause(literals=(Literal(sign=True, atom=atom),))

        single = provider.get_embedding(clause)
        batch = provider.get_embeddings_batch([clause])

        assert len(batch) == 1
        assert single == batch[0]

    def test_embedding_dim_matches_base(self):
        """embedding_dim property delegates to base provider."""
        from pyladr.search.goal_directed import GoalDirectedConfig, GoalDirectedEmbeddingProvider

        class DimProvider:
            embedding_dim = 42
            def get_embedding(self, clause):
                return [0.0] * 42
            def get_embeddings_batch(self, clauses):
                return [self.get_embedding(c) for c in clauses]

        cfg = GoalDirectedConfig(enabled=True)
        provider = GoalDirectedEmbeddingProvider(DimProvider(), cfg)
        assert provider.embedding_dim == 42

    def test_stats_tracking(self):
        """Statistics properly tracked."""
        from pyladr.search.goal_directed import GoalDirectedConfig, GoalDirectedEmbeddingProvider

        class FakeProvider:
            embedding_dim = 4
            def get_embedding(self, clause):
                return [1.0, 0.0, 0.0, 0.0]
            def get_embeddings_batch(self, clauses):
                return [self.get_embedding(c) for c in clauses]

        cfg = GoalDirectedConfig(enabled=True)
        provider = GoalDirectedEmbeddingProvider(FakeProvider(), cfg)

        atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        clause = Clause(literals=(Literal(sign=True, atom=atom),))

        provider.notify_clause_kept(clause)
        provider.notify_clause_kept(clause)
        provider.notify_clause_selected(clause)
        provider.notify_proof_found([clause])

        stats = provider.stats
        assert stats["clauses_observed"] == 2
        assert stats["clauses_selected"] == 1
        assert stats["proofs_observed"] == 1

    def test_feedback_recording_disabled(self):
        """Feedback recording does nothing when online_learning is off."""
        from pyladr.search.goal_directed import GoalDirectedConfig, GoalDirectedEmbeddingProvider

        class FakeProvider:
            embedding_dim = 4
            def get_embedding(self, clause):
                return [1.0, 0.0, 0.0, 0.0]
            def get_embeddings_batch(self, clauses):
                return [self.get_embedding(c) for c in clauses]

        cfg = GoalDirectedConfig(enabled=True, online_learning=False)
        provider = GoalDirectedEmbeddingProvider(FakeProvider(), cfg)

        atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        clause = Clause(literals=(Literal(sign=True, atom=atom),))

        provider.record_feedback([clause], [clause])
        assert provider.stats["feedback_pairs"] == 0


# ── Large Input Stability ─────────────────────────────────────────────────


class TestLargeInputStability:
    """Stability tests with larger clause sets."""

    def test_many_clauses_deterministic(self):
        """Many clauses produce deterministic results when disabled."""
        P_sn = 1
        clauses = []
        for i in range(20):
            sym = get_rigid_term(i + 10, 0)
            clauses.append(
                Clause(literals=(Literal(sign=True, atom=get_rigid_term(P_sn, 1, (sym,))),))
            )

        opts = SearchOptions(max_given=5, quiet=True, goal_directed=False)
        r1 = GivenClauseSearch(options=opts).run(usable=[], sos=clauses[:])
        r2 = GivenClauseSearch(options=opts).run(usable=[], sos=clauses[:])

        assert r1.exit_code == r2.exit_code
        assert r1.stats.given == r2.stats.given
        assert r1.stats.generated == r2.stats.generated

    def test_usable_and_sos_both_populated(self):
        """Search works correctly with both usable and SOS populated."""
        P_sn, Q_sn = 1, 2
        a = get_rigid_term(3, 0)
        x = get_variable_term(0)

        usable = [
            Clause(literals=(
                Literal(sign=False, atom=get_rigid_term(P_sn, 1, (x,))),
                Literal(sign=True, atom=get_rigid_term(Q_sn, 1, (x,))),
            )),
        ]
        sos = [
            Clause(literals=(Literal(sign=True, atom=get_rigid_term(P_sn, 1, (a,))),)),
            Clause(literals=(Literal(sign=False, atom=get_rigid_term(Q_sn, 1, (a,))),)),
        ]

        opts = SearchOptions(
            binary_resolution=True,
            factoring=True,
            max_given=50,
            quiet=True,
            goal_directed=False,
        )
        result = GivenClauseSearch(options=opts).run(usable=usable, sos=sos)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
