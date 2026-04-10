"""Tests for GoalDirectedEmbeddingProvider.

Verifies:
- EmbeddingProvider protocol compliance (drop-in replacement)
- Goal proximity scoring: distance from goal clause embeddings
- Backward compatibility: disabled = identical passthrough
- Incremental embedding updates during proof search
- Online contrastive learning from search feedback
- Caching behavior and invalidation
- Thread safety for concurrent access
"""

from __future__ import annotations

import math
import threading

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.term import get_rigid_term
from pyladr.search.ml_selection import (
    EmbeddingProvider,
    EmbeddingEnhancedSelection,
    MLSelectionConfig,
)
from pyladr.search.selection import SelectionOrder, SelectionRule
from pyladr.search.state import ClauseList


# ── Helpers ──────────────────────────────────────────────────────────────────


def _const(symnum: int) -> "Term":
    return get_rigid_term(symnum, 0)


def _func(symnum: int, *args) -> "Term":
    return get_rigid_term(symnum, len(args), args)


def _make_clause(*atoms, signs=None, cid: int = 0) -> Clause:
    if signs is None:
        signs = (True,) * len(atoms)
    lits = tuple(Literal(sign=s, atom=a) for s, a in zip(signs, atoms))
    c = Clause(literals=lits)
    if cid:
        c.id = cid
    return c


def _make_weighted_clause(weight: float, cid: int = 0) -> Clause:
    c = _make_clause(_const(1))
    c.weight = weight
    c.id = cid
    return c


def _make_goal_clause(cid: int = 0) -> Clause:
    """Create a clause with DENY justification (negated goal)."""
    c = _make_clause(_const(2), cid=cid)
    c._justification = (Justification(just_type=JustType.DENY, clause_ids=(0,)),)
    return c


def _make_sos(clauses: list[Clause]) -> ClauseList:
    cl = ClauseList("sos")
    for c in clauses:
        cl.append(c)
    return cl


class MockEmbeddingProvider:
    """Mock base provider for testing goal-directed wrapper."""

    def __init__(
        self,
        embeddings: dict[int, list[float]] | None = None,
        dim: int = 4,
    ):
        self._embeddings = embeddings or {}
        self._dim = dim
        self.call_count = 0
        self.batch_call_count = 0

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def get_embedding(self, clause: Clause) -> list[float] | None:
        self.call_count += 1
        return self._embeddings.get(clause.id)

    def get_embeddings_batch(
        self, clauses: list[Clause],
    ) -> list[list[float] | None]:
        self.batch_call_count += 1
        return [self.get_embedding(c) for c in clauses]


# ── Import the module under test ─────────────────────────────────────────────

from pyladr.search.goal_directed import (
    GoalDirectedConfig,
    GoalDirectedEmbeddingProvider,
    GoalProximityScorer,
)


# ── Protocol compliance tests ────────────────────────────────────────────────


class TestProtocolCompliance:
    """GoalDirectedEmbeddingProvider must satisfy EmbeddingProvider protocol."""

    def test_satisfies_embedding_provider_protocol(self) -> None:
        base = MockEmbeddingProvider(dim=4)
        provider = GoalDirectedEmbeddingProvider(base_provider=base)
        assert isinstance(provider, EmbeddingProvider)

    def test_has_embedding_dim(self) -> None:
        base = MockEmbeddingProvider(dim=8)
        provider = GoalDirectedEmbeddingProvider(base_provider=base)
        assert provider.embedding_dim == 8

    def test_get_embedding_returns_list_or_none(self) -> None:
        base = MockEmbeddingProvider(
            embeddings={1: [1.0, 0.0, 0.0, 0.0]}, dim=4,
        )
        provider = GoalDirectedEmbeddingProvider(base_provider=base)
        c = _make_weighted_clause(1.0, cid=1)
        result = provider.get_embedding(c)
        assert isinstance(result, list)
        assert len(result) == 4

    def test_get_embedding_returns_none_when_base_returns_none(self) -> None:
        base = MockEmbeddingProvider(dim=4)
        provider = GoalDirectedEmbeddingProvider(base_provider=base)
        c = _make_weighted_clause(1.0, cid=99)
        result = provider.get_embedding(c)
        assert result is None

    def test_get_embeddings_batch_returns_parallel_list(self) -> None:
        base = MockEmbeddingProvider(
            embeddings={1: [1.0, 0.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0, 0.0]},
            dim=4,
        )
        provider = GoalDirectedEmbeddingProvider(base_provider=base)
        c1 = _make_weighted_clause(1.0, cid=1)
        c2 = _make_weighted_clause(2.0, cid=2)
        results = provider.get_embeddings_batch([c1, c2])
        assert len(results) == 2
        assert results[0] is not None
        assert results[1] is not None

    def test_get_embeddings_batch_empty_input(self) -> None:
        base = MockEmbeddingProvider(dim=4)
        provider = GoalDirectedEmbeddingProvider(base_provider=base)
        results = provider.get_embeddings_batch([])
        assert results == []


# ── Disabled mode (passthrough) tests ────────────────────────────────────────


class TestDisabledPassthrough:
    """When disabled, provider must be an exact passthrough."""

    def test_disabled_returns_base_embedding_unchanged(self) -> None:
        emb = [1.0, 2.0, 3.0, 4.0]
        base = MockEmbeddingProvider(embeddings={1: emb}, dim=4)
        config = GoalDirectedConfig(enabled=False)
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )
        c = _make_weighted_clause(1.0, cid=1)
        result = provider.get_embedding(c)
        assert result == emb

    def test_disabled_batch_returns_base_unchanged(self) -> None:
        base = MockEmbeddingProvider(
            embeddings={1: [1.0, 0.0], 2: [0.0, 1.0]}, dim=2,
        )
        config = GoalDirectedConfig(enabled=False)
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )
        c1 = _make_weighted_clause(1.0, cid=1)
        c2 = _make_weighted_clause(2.0, cid=2)
        results = provider.get_embeddings_batch([c1, c2])
        assert results == [[1.0, 0.0], [0.0, 1.0]]

    def test_disabled_does_not_modify_selection_behavior(self) -> None:
        """When disabled, selection should be identical to using base provider."""
        embs = {1: [0.5, 0.5, 0.0, 0.0], 2: [0.0, 0.0, 0.5, 0.5]}
        base = MockEmbeddingProvider(embeddings=embs, dim=4)
        config = GoalDirectedConfig(enabled=False)
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )

        ml_config = MLSelectionConfig(
            enabled=True, ml_weight=0.5, min_sos_for_ml=1,
        )
        sel = EmbeddingEnhancedSelection(
            embedding_provider=provider, ml_config=ml_config,
        )
        c1 = _make_weighted_clause(1.0, cid=1)
        c2 = _make_weighted_clause(1.0, cid=2)
        sos = _make_sos([c1, c2])

        c, name = sel.select_given(sos, 0)
        assert c is not None  # Just verify it works without error


# ── Goal registration tests ──────────────────────────────────────────────────


class TestGoalRegistration:
    """Test registering goal clauses for proximity scoring."""

    def test_register_goals(self) -> None:
        base = MockEmbeddingProvider(
            embeddings={10: [1.0, 0.0, 0.0, 0.0]}, dim=4,
        )
        provider = GoalDirectedEmbeddingProvider(base_provider=base)
        goal = _make_weighted_clause(1.0, cid=10)
        provider.register_goals([goal])
        assert provider.num_goals == 1

    def test_register_multiple_goals(self) -> None:
        base = MockEmbeddingProvider(
            embeddings={
                10: [1.0, 0.0, 0.0, 0.0],
                11: [0.0, 1.0, 0.0, 0.0],
            },
            dim=4,
        )
        provider = GoalDirectedEmbeddingProvider(base_provider=base)
        g1 = _make_weighted_clause(1.0, cid=10)
        g2 = _make_weighted_clause(1.0, cid=11)
        provider.register_goals([g1, g2])
        assert provider.num_goals == 2

    def test_register_goals_skips_unembeddable(self) -> None:
        """Goals that can't be embedded are silently skipped."""
        base = MockEmbeddingProvider(dim=4)  # no embeddings
        provider = GoalDirectedEmbeddingProvider(base_provider=base)
        goal = _make_weighted_clause(1.0, cid=10)
        provider.register_goals([goal])
        assert provider.num_goals == 0

    def test_clear_goals(self) -> None:
        base = MockEmbeddingProvider(
            embeddings={10: [1.0, 0.0, 0.0, 0.0]}, dim=4,
        )
        provider = GoalDirectedEmbeddingProvider(base_provider=base)
        goal = _make_weighted_clause(1.0, cid=10)
        provider.register_goals([goal])
        provider.clear_goals()
        assert provider.num_goals == 0


# ── Goal proximity scoring tests ─────────────────────────────────────────────


class TestGoalProximityScoring:
    """Test goal proximity distance computation."""

    def test_identical_to_goal_has_max_proximity(self) -> None:
        base = MockEmbeddingProvider(
            embeddings={
                1: [1.0, 0.0, 0.0, 0.0],
                10: [1.0, 0.0, 0.0, 0.0],  # goal
            },
            dim=4,
        )
        config = GoalDirectedConfig(enabled=True, goal_proximity_weight=1.0)
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )
        goal = _make_weighted_clause(1.0, cid=10)
        provider.register_goals([goal])

        scorer = provider.goal_scorer
        c = _make_weighted_clause(1.0, cid=1)
        emb = base.get_embedding(c)
        proximity = scorer.proximity(emb)
        assert proximity == pytest.approx(1.0)

    def test_orthogonal_to_goal_has_low_proximity(self) -> None:
        base = MockEmbeddingProvider(
            embeddings={
                1: [0.0, 0.0, 1.0, 0.0],  # orthogonal to goal
                10: [1.0, 0.0, 0.0, 0.0],  # goal
            },
            dim=4,
        )
        config = GoalDirectedConfig(enabled=True, goal_proximity_weight=1.0)
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )
        goal = _make_weighted_clause(1.0, cid=10)
        provider.register_goals([goal])

        scorer = provider.goal_scorer
        c = _make_weighted_clause(1.0, cid=1)
        emb = base.get_embedding(c)
        proximity = scorer.proximity(emb)
        assert proximity == pytest.approx(0.5)  # neutral for orthogonal

    def test_opposite_to_goal_has_min_proximity(self) -> None:
        base = MockEmbeddingProvider(
            embeddings={
                1: [-1.0, 0.0, 0.0, 0.0],  # opposite of goal
                10: [1.0, 0.0, 0.0, 0.0],  # goal
            },
            dim=4,
        )
        config = GoalDirectedConfig(enabled=True, goal_proximity_weight=1.0)
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )
        goal = _make_weighted_clause(1.0, cid=10)
        provider.register_goals([goal])

        scorer = provider.goal_scorer
        c = _make_weighted_clause(1.0, cid=1)
        emb = base.get_embedding(c)
        proximity = scorer.proximity(emb)
        assert proximity == pytest.approx(0.0)

    def test_proximity_with_multiple_goals_uses_max(self) -> None:
        """Proximity is max similarity to any goal (closest goal matters)."""
        base = MockEmbeddingProvider(
            embeddings={
                1: [0.0, 1.0, 0.0, 0.0],
                10: [1.0, 0.0, 0.0, 0.0],  # goal A - far
                11: [0.0, 1.0, 0.0, 0.0],  # goal B - close
            },
            dim=4,
        )
        config = GoalDirectedConfig(enabled=True)
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )
        g1 = _make_weighted_clause(1.0, cid=10)
        g2 = _make_weighted_clause(1.0, cid=11)
        provider.register_goals([g1, g2])

        scorer = provider.goal_scorer
        emb = base.get_embedding(_make_weighted_clause(1.0, cid=1))
        proximity = scorer.proximity(emb)
        # Should be close to 1.0 because of goal B
        assert proximity == pytest.approx(1.0)

    def test_no_goals_returns_neutral(self) -> None:
        """Without registered goals, proximity is 0.5 (neutral)."""
        base = MockEmbeddingProvider(
            embeddings={1: [1.0, 0.0, 0.0, 0.0]}, dim=4,
        )
        provider = GoalDirectedEmbeddingProvider(base_provider=base)
        scorer = provider.goal_scorer
        proximity = scorer.proximity([1.0, 0.0, 0.0, 0.0])
        assert proximity == pytest.approx(0.5)


# ── Goal-enhanced embedding tests ────────────────────────────────────────────


class TestGoalEnhancedEmbeddings:
    """Test that embeddings are enhanced with goal proximity info when enabled."""

    def test_enabled_modifies_embeddings(self) -> None:
        """When enabled with goals, embeddings should differ from base."""
        base_emb = [1.0, 0.0, 0.0, 0.0]
        base = MockEmbeddingProvider(
            embeddings={
                1: base_emb.copy(),
                10: [0.0, 1.0, 0.0, 0.0],
            },
            dim=4,
        )
        config = GoalDirectedConfig(
            enabled=True, goal_proximity_weight=0.5,
        )
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )
        goal = _make_weighted_clause(1.0, cid=10)
        provider.register_goals([goal])

        c = _make_weighted_clause(1.0, cid=1)
        result = provider.get_embedding(c)
        assert result is not None
        # Should differ from base embedding due to goal proximity modulation
        assert result != base_emb

    def test_goal_proximity_boosts_close_clauses(self) -> None:
        """Clauses close to goals should get smaller norms (higher proof potential).

        The proof_potential_score in ml_selection.py rewards smaller norms
        (inverse sigmoid). Goal-proximate clauses get scaled down → smaller
        norm → higher proof potential score.
        """
        base = MockEmbeddingProvider(
            embeddings={
                1: [0.9, 0.1, 0.0, 0.0],  # close to goal
                2: [0.0, 0.0, 0.9, 0.1],  # far from goal
                10: [1.0, 0.0, 0.0, 0.0],  # goal
            },
            dim=4,
        )
        config = GoalDirectedConfig(
            enabled=True, goal_proximity_weight=0.8,
        )
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )
        goal = _make_weighted_clause(1.0, cid=10)
        provider.register_goals([goal])

        c_close = _make_weighted_clause(1.0, cid=1)
        c_far = _make_weighted_clause(1.0, cid=2)

        emb_close = provider.get_embedding(c_close)
        emb_far = provider.get_embedding(c_far)
        assert emb_close is not None
        assert emb_far is not None

        # Close clause should have SMALLER norm (more scaled down by proximity)
        # → higher proof_potential_score in ml_selection.py
        norm_close = math.sqrt(sum(x * x for x in emb_close))
        norm_far = math.sqrt(sum(x * x for x in emb_far))
        assert norm_close < norm_far


# ── Incremental update tests ─────────────────────────────────────────────────


class TestIncrementalUpdates:
    """Test incremental embedding updates during proof search."""

    def test_notify_clause_kept_updates_cache(self) -> None:
        base = MockEmbeddingProvider(
            embeddings={1: [1.0, 0.0, 0.0, 0.0]}, dim=4,
        )
        config = GoalDirectedConfig(enabled=True)
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )
        c = _make_weighted_clause(1.0, cid=1)
        # Should not raise
        provider.notify_clause_kept(c)
        assert provider.stats["clauses_observed"] >= 1

    def test_notify_clause_selected_tracks_given(self) -> None:
        base = MockEmbeddingProvider(
            embeddings={1: [1.0, 0.0, 0.0, 0.0]}, dim=4,
        )
        config = GoalDirectedConfig(enabled=True)
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )
        c = _make_weighted_clause(1.0, cid=1)
        provider.notify_clause_selected(c)
        assert provider.stats["clauses_selected"] >= 1

    def test_notify_proof_found_records_proof_clauses(self) -> None:
        base = MockEmbeddingProvider(
            embeddings={1: [1.0, 0.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0, 0.0]},
            dim=4,
        )
        config = GoalDirectedConfig(enabled=True)
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )
        c1 = _make_weighted_clause(1.0, cid=1)
        c2 = _make_weighted_clause(1.0, cid=2)
        provider.notify_proof_found([c1, c2])
        assert provider.stats["proofs_observed"] >= 1


# ── Online contrastive learning tests ────────────────────────────────────────


class TestOnlineLearning:
    """Test online contrastive learning from search feedback."""

    def test_learning_disabled_by_default(self) -> None:
        base = MockEmbeddingProvider(dim=4)
        config = GoalDirectedConfig(enabled=True, online_learning=False)
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )
        assert not provider.is_learning_enabled

    def test_learning_can_be_enabled(self) -> None:
        base = MockEmbeddingProvider(dim=4)
        config = GoalDirectedConfig(enabled=True, online_learning=True)
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )
        assert provider.is_learning_enabled

    def test_feedback_collection_when_learning_enabled(self) -> None:
        base = MockEmbeddingProvider(
            embeddings={1: [1.0, 0.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0, 0.0]},
            dim=4,
        )
        config = GoalDirectedConfig(enabled=True, online_learning=True)
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )
        c1 = _make_weighted_clause(1.0, cid=1)
        c2 = _make_weighted_clause(1.0, cid=2)
        # Record that c1 was productive, c2 was not
        provider.record_feedback(productive=[c1], unproductive=[c2])
        assert provider.stats["feedback_pairs"] >= 1


# ── Configuration tests ──────────────────────────────────────────────────────


class TestGoalDirectedConfig:
    """Test configuration validation and defaults."""

    def test_default_config_is_disabled(self) -> None:
        config = GoalDirectedConfig()
        assert not config.enabled

    def test_goal_proximity_weight_bounds(self) -> None:
        config = GoalDirectedConfig(goal_proximity_weight=0.5)
        assert 0.0 <= config.goal_proximity_weight <= 1.0

    def test_config_frozen(self) -> None:
        config = GoalDirectedConfig()
        with pytest.raises(AttributeError):
            config.enabled = True  # type: ignore[misc]


# ── Statistics tests ─────────────────────────────────────────────────────────


class TestStats:
    """Test statistics tracking."""

    def test_initial_stats_are_zero(self) -> None:
        base = MockEmbeddingProvider(dim=4)
        provider = GoalDirectedEmbeddingProvider(base_provider=base)
        stats = provider.stats
        assert stats["clauses_observed"] == 0
        assert stats["clauses_selected"] == 0
        assert stats["proofs_observed"] == 0

    def test_stats_report_contains_key_metrics(self) -> None:
        base = MockEmbeddingProvider(dim=4)
        provider = GoalDirectedEmbeddingProvider(base_provider=base)
        report = provider.stats_report()
        assert "goal_directed" in report


# ── Thread safety tests ──────────────────────────────────────────────────────


class TestThreadSafety:
    """Test concurrent access patterns."""

    def test_concurrent_get_embedding(self) -> None:
        """Multiple threads can safely call get_embedding concurrently."""
        base = MockEmbeddingProvider(
            embeddings={i: [float(i), 0.0, 0.0, 0.0] for i in range(100)},
            dim=4,
        )
        config = GoalDirectedConfig(enabled=True)
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )

        results = [None] * 100
        errors = []

        def worker(idx: int) -> None:
            try:
                c = _make_weighted_clause(1.0, cid=idx)
                results[idx] = provider.get_embedding(c)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(r is not None for r in results)

    def test_concurrent_register_and_read(self) -> None:
        """Goal registration and embedding reads don't deadlock."""
        base = MockEmbeddingProvider(
            embeddings={i: [float(i), 0.0, 0.0, 0.0] for i in range(110)},
            dim=4,
        )
        config = GoalDirectedConfig(enabled=True)
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )

        errors = []

        def reader() -> None:
            try:
                for i in range(50):
                    c = _make_weighted_clause(1.0, cid=i)
                    provider.get_embedding(c)
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                goals = [_make_weighted_clause(1.0, cid=100 + i) for i in range(10)]
                provider.register_goals(goals)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ── Integration with EmbeddingEnhancedSelection tests ────────────────────────


class TestSelectionIntegration:
    """Test that GoalDirectedEmbeddingProvider works with EmbeddingEnhancedSelection."""

    def test_goal_directed_as_selection_provider(self) -> None:
        """GoalDirectedEmbeddingProvider can be used as selection provider."""
        base = MockEmbeddingProvider(
            embeddings={
                1: [0.9, 0.1, 0.0, 0.0],
                2: [0.0, 0.0, 0.9, 0.1],
                10: [1.0, 0.0, 0.0, 0.0],
            },
            dim=4,
        )
        config = GoalDirectedConfig(
            enabled=True, goal_proximity_weight=0.5,
        )
        provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )
        goal = _make_weighted_clause(1.0, cid=10)
        provider.register_goals([goal])

        ml_config = MLSelectionConfig(
            enabled=True,
            ml_weight=0.8,
            diversity_weight=0.0,
            proof_potential_weight=1.0,
            min_sos_for_ml=1,
        )
        sel = EmbeddingEnhancedSelection(
            rules=[SelectionRule("W", SelectionOrder.WEIGHT, part=1)],
            embedding_provider=provider,
            ml_config=ml_config,
        )

        c1 = _make_weighted_clause(1.0, cid=1)  # close to goal
        c2 = _make_weighted_clause(1.0, cid=2)  # far from goal
        sos = _make_sos([c1, c2])

        c, name = sel.select_given(sos, 0)
        assert c is not None
        # With goal proximity and proof potential scoring, c1 should be preferred
        assert c is c1
        assert "+ML" in name

    def test_disabled_goal_directed_selection_unchanged(self) -> None:
        """With goal-directed disabled, selection identical to base provider."""
        embs = {1: [0.5, 0.5, 0.0, 0.0], 2: [0.0, 0.0, 0.5, 0.5]}
        base = MockEmbeddingProvider(embeddings=embs, dim=4)
        config = GoalDirectedConfig(enabled=False)
        gd_provider = GoalDirectedEmbeddingProvider(
            base_provider=base, config=config,
        )

        ml_config = MLSelectionConfig(
            enabled=True, ml_weight=0.5, min_sos_for_ml=1,
        )

        w_rules = [SelectionRule("W", SelectionOrder.WEIGHT, part=1)]
        # Selection with base provider
        sel_base = EmbeddingEnhancedSelection(
            rules=w_rules,
            embedding_provider=base, ml_config=ml_config,
        )
        c1a = _make_weighted_clause(3.0, cid=1)
        c2a = _make_weighted_clause(1.0, cid=2)
        sos_base = _make_sos([c1a, c2a])
        result_base, _ = sel_base.select_given(sos_base, 0)

        # Selection with disabled goal-directed provider
        sel_gd = EmbeddingEnhancedSelection(
            rules=w_rules,
            embedding_provider=gd_provider, ml_config=ml_config,
        )
        c1b = _make_weighted_clause(3.0, cid=1)
        c2b = _make_weighted_clause(1.0, cid=2)
        sos_gd = _make_sos([c1b, c2b])
        result_gd, _ = sel_gd.select_given(sos_gd, 0)

        # Both should select the same clause (by weight)
        assert result_base is not None
        assert result_gd is not None
        assert result_base.weight == result_gd.weight
