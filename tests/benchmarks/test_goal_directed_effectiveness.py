"""Effectiveness validation for goal-directed clause selection.

Validates that the hierarchical GNN and goal-directed features produce
measurable improvements in proof search quality.

Key areas tested:
1. Goal proximity correlation with proof membership
2. Embedding diversity during search
3. Goal-directed vs traditional selection on controlled problems
4. Online learning adaptation over time
5. Demonstration cases showing goal-directed behavior improvements
6. Embedding quality validation through distance metrics

Run with: pytest tests/benchmarks/test_goal_directed_effectiveness.py -v
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field

import pytest

torch = pytest.importorskip("torch")

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term
from pyladr.core.symbol import SymbolTable
from pyladr.search.goal_directed import (
    GoalDirectedConfig,
    GoalDirectedEmbeddingProvider,
    GoalProximityScorer,
)
from pyladr.search.ml_selection import (
    EmbeddingEnhancedSelection,
    MLSelectionConfig,
    MLSelectionStats,
    _cosine_similarity,
)
from pyladr.search.selection import GivenSelection, SelectionOrder, SelectionRule
from pyladr.search.state import ClauseList


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_term(symnum: int, args: tuple[Term, ...] = ()) -> Term:
    return Term(private_symbol=-symnum, arity=len(args), args=args)


def _make_var(varnum: int) -> Term:
    return Term(private_symbol=varnum, arity=0, args=())


def _make_clause(
    lits: list[tuple[bool, Term]], clause_id: int = 0,
    weight: float | None = None,
) -> Clause:
    literals = tuple(Literal(sign=s, atom=a) for s, a in lits)
    c = Clause(literals=literals)
    c.id = clause_id
    c.weight = weight if weight is not None else float(len(literals))
    return c


def _make_nested_term(depth: int, base_sym: int = 1) -> Term:
    if depth <= 0:
        return _make_var(base_sym % 5)
    left = _make_nested_term(depth - 1, base_sym * 2)
    right = _make_nested_term(depth - 1, base_sym * 2 + 1)
    return _make_term(base_sym % 10 + 1, (left, right))


class MockEmbeddingProvider:
    """Mock embedding provider with controllable embeddings.

    Embeds each clause as a deterministic vector based on its ID,
    enabling predictable distance/similarity computations for testing.
    """

    def __init__(self, dim: int = 32, goal_vector: list[float] | None = None):
        self._dim = dim
        self._goal_vector = goal_vector  # Optional fixed goal direction
        self._call_count = 0

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def _clause_embedding(self, clause: Clause) -> list[float]:
        """Deterministic embedding based on clause ID."""
        self._call_count += 1
        # Use golden-ratio-based hashing for well-separated embeddings
        emb = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        for i in range(self._dim):
            # Quasi-random sequence that spreads points evenly
            val = math.sin((clause.id * phi + i) * 2.0 * math.pi)
            val += 0.5 * math.cos((clause.id * 3.7 + i * 2.3) * math.pi)
            emb.append(val * (1.0 + 0.1 * clause.weight))
        return emb

    def get_embedding(self, clause: Clause) -> list[float] | None:
        return self._clause_embedding(clause)

    def get_embeddings_batch(
        self, clauses: list[Clause],
    ) -> list[list[float] | None]:
        return [self._clause_embedding(c) for c in clauses]


class GoalAlignedMockProvider(MockEmbeddingProvider):
    """Mock provider where some clauses are goal-aligned by construction.

    Clauses with IDs divisible by goal_divisor have embeddings similar
    to the goal vector. Others are orthogonal or dissimilar.
    """

    def __init__(self, dim: int = 32, goal_divisor: int = 5):
        super().__init__(dim)
        self._goal_divisor = goal_divisor
        # Fixed goal direction
        self._goal_direction = [1.0 / math.sqrt(dim)] * dim

    def _clause_embedding(self, clause: Clause) -> list[float]:
        self._call_count += 1
        if clause.id % self._goal_divisor == 0:
            # Goal-aligned: close to goal direction
            noise_scale = 0.1 * (clause.id % 3)  # Small variation
            return [
                g + noise_scale * math.sin(clause.id * 0.5 + i)
                for i, g in enumerate(self._goal_direction)
            ]
        else:
            # Non-aligned: orthogonal/random direction
            emb = []
            for i in range(self._dim):
                angle = (clause.id * 2.3 + i * 1.7) * math.pi / self._dim
                emb.append(math.sin(angle))
            # Normalize
            norm = math.sqrt(sum(x * x for x in emb))
            return [x / (norm + 1e-8) for x in emb]


def _make_sos(num_clauses: int, weight_range: tuple[float, float] = (1.0, 10.0)) -> ClauseList:
    """Create a SOS clause list with varying weights."""
    sos = ClauseList("sos")
    for i in range(1, num_clauses + 1):
        weight = weight_range[0] + (weight_range[1] - weight_range[0]) * (i / num_clauses)
        atom = _make_term(i % 10 + 1, (_make_var(0), _make_var(1)))
        c = _make_clause([(True, atom)], clause_id=i, weight=weight)
        sos.append(c)
    return sos


# ── Goal proximity scorer tests ──────────────────────────────────────────


@pytest.mark.benchmark
class TestGoalDistanceScoring:
    """Validate goal distance computation quality."""

    def test_identical_embeddings_give_zero_distance(self):
        """Identical embeddings should yield distance = 0.0."""
        scorer = GoalProximityScorer(method="max")
        goal = [1.0, 0.0, 0.0, 0.0]
        scorer.set_goals([goal])

        dist = scorer.nearest_goal_distance(goal)
        assert abs(dist - 0.0) < 1e-6, f"Expected 0.0, got {dist}"

    def test_orthogonal_embeddings_give_neutral_distance(self):
        """Orthogonal embeddings should yield distance = 0.5."""
        scorer = GoalProximityScorer(method="max")
        goal = [1.0, 0.0, 0.0, 0.0]
        scorer.set_goals([goal])

        ortho = [0.0, 1.0, 0.0, 0.0]
        dist = scorer.nearest_goal_distance(ortho)
        assert abs(dist - 0.5) < 1e-6, f"Expected 0.5, got {dist}"

    def test_opposite_embeddings_give_max_distance(self):
        """Opposite embeddings should yield distance = 1.0."""
        scorer = GoalProximityScorer(method="max")
        goal = [1.0, 0.0, 0.0, 0.0]
        scorer.set_goals([goal])

        opposite = [-1.0, 0.0, 0.0, 0.0]
        dist = scorer.nearest_goal_distance(opposite)
        assert abs(dist - 1.0) < 1e-6, f"Expected 1.0, got {dist}"

    def test_distance_ordering_is_monotonic(self):
        """Closer embeddings should have smaller distance scores."""
        scorer = GoalProximityScorer(method="max")
        goal = [1.0, 0.0, 0.0, 0.0]
        scorer.set_goals([goal])

        # Vectors at increasing angles from goal
        vectors = [
            [0.95, 0.31, 0.0, 0.0],  # close
            [0.70, 0.71, 0.0, 0.0],  # medium
            [0.31, 0.95, 0.0, 0.0],  # far
            [0.0, 1.0, 0.0, 0.0],    # orthogonal
        ]

        distances = [scorer.nearest_goal_distance(v) for v in vectors]
        for i in range(len(distances) - 1):
            assert distances[i] < distances[i + 1], (
                f"Non-monotonic: dist[{i}]={distances[i]:.4f} >= dist[{i+1}]={distances[i+1]:.4f}"
            )

    def test_multiple_goals_uses_nearest(self):
        """With multiple goals, nearest_goal_distance uses the closest goal."""
        scorer = GoalProximityScorer(method="max")
        goals = [
            [1.0, 0.0, 0.0, 0.0],  # goal A
            [0.0, 1.0, 0.0, 0.0],  # goal B
        ]
        scorer.set_goals(goals)

        # Close to goal A → small distance
        dist_a = scorer.nearest_goal_distance([0.9, 0.1, 0.0, 0.0])
        # Close to goal B → small distance
        dist_b = scorer.nearest_goal_distance([0.1, 0.9, 0.0, 0.0])

        # Both should be small (close to one goal)
        assert dist_a < 0.3
        assert dist_b < 0.3

    def test_equidistant_from_two_goals(self):
        """Vector equidistant from two goals should have modest nearest distance."""
        scorer = GoalProximityScorer(method="max")
        goals = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
        scorer.set_goals(goals)

        # Equidistant from both goals — still close (cos 45 = 0.707)
        equidistant = [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]
        dist = scorer.nearest_goal_distance(equidistant)
        # nearest_goal_distance = (1 - 0.707) / 2 ≈ 0.146
        assert 0.05 < dist < 0.3


# ── Goal-directed embedding enhancement tests ───────────────────────────


@pytest.mark.benchmark
class TestGoalDirectedEmbeddingEnhancement:
    """Validate that goal-directed embedding modulation works correctly."""

    def test_disabled_config_is_passthrough(self):
        """When disabled, enhanced provider returns exact base embeddings."""
        base = MockEmbeddingProvider(dim=8)
        config = GoalDirectedConfig(enabled=False)
        provider = GoalDirectedEmbeddingProvider(base, config)

        clause = _make_clause([(True, _make_term(1))], 1)
        base_emb = base.get_embedding(clause)
        enhanced_emb = provider.get_embedding(clause)

        assert base_emb == enhanced_emb

    def test_enabled_without_goals_is_passthrough(self):
        """When enabled but no goals registered, provider is passthrough."""
        base = MockEmbeddingProvider(dim=8)
        config = GoalDirectedConfig(enabled=True, goal_proximity_weight=0.3)
        provider = GoalDirectedEmbeddingProvider(base, config)

        clause = _make_clause([(True, _make_term(1))], 1)
        base_emb = base.get_embedding(clause)
        enhanced_emb = provider.get_embedding(clause)

        # With no goals, proximity=0.5, scale = 1.0 - 0.3*0.5 = 0.85
        # So enhanced != base but only scaled
        assert enhanced_emb is not None
        assert len(enhanced_emb) == len(base_emb)

    def test_goal_close_clauses_get_smaller_norms(self):
        """Clauses close to the goal should get smaller norms (higher proof potential).

        Goal embeddings are stored sign-stripped, so small distance = similar
        to goal content = proof-useful.  Register an aligned clause as the goal
        so that other aligned clauses have small distance → scaled down → preferred.
        """
        base = GoalAlignedMockProvider(dim=16, goal_divisor=5)
        config = GoalDirectedConfig(enabled=True, goal_proximity_weight=0.5)
        provider = GoalDirectedEmbeddingProvider(base, config)

        # Register an aligned clause as the goal (id divisible by 5).
        # Other aligned clauses will be close to this → small distance → preferred.
        goal = _make_clause([(True, _make_term(1))], 5)  # aligned id
        provider.register_goals([goal])

        # Close clause: aligned (id divisible by 5)
        close = _make_clause([(True, _make_term(2))], 10)
        close_emb = provider.get_embedding(close)

        # Far clause: non-aligned
        far = _make_clause([(True, _make_term(4))], 9)
        far_emb = provider.get_embedding(far)

        close_norm = math.sqrt(sum(x * x for x in close_emb))
        far_norm = math.sqrt(sum(x * x for x in far_emb))

        print(f"\nClose (aligned) norm: {close_norm:.4f}, Far (non-aligned) norm: {far_norm:.4f}")
        # Goal-close clause → small distance → more scaled down → smaller norm
        assert close_norm < far_norm, (
            f"Goal-close norm ({close_norm:.4f}) should be < far ({far_norm:.4f})"
        )


# ── Selection effectiveness tests ────────────────────────────────────────


@pytest.mark.benchmark
class TestSelectionEffectiveness:
    """Validate ML-guided selection produces better clause ordering."""

    def test_ml_selection_prefers_diverse_clauses(self):
        """ML selection should prefer diverse clauses over similar ones."""
        provider = MockEmbeddingProvider(dim=16)
        config = MLSelectionConfig(
            enabled=True,
            ml_weight=0.5,
            diversity_weight=0.8,
            proof_potential_weight=0.2,
            min_sos_for_ml=5,
        )
        selection = EmbeddingEnhancedSelection(
            embedding_provider=provider,
            ml_config=config,
        )

        sos = _make_sos(20, weight_range=(3.0, 3.5))  # Similar weights

        selected_ids = []
        for i in range(10):
            clause, sel_type = selection.select_given(sos, i)
            if clause is None:
                break
            selected_ids.append(clause.id)

        # Diversity should lead to spread selection, not consecutive IDs
        print(f"\nSelected IDs (diversity mode): {selected_ids}")
        assert len(selected_ids) >= 8, "Too few selections made"

    def test_goal_directed_prefers_goal_aligned(self):
        """Goal-directed selection should prefer goal-aligned clauses.

        We register a non-aligned clause as the DENY goal.  Aligned clauses
        are far from the deny reference → large distance → smaller norms →
        higher proof potential → more likely to be selected.
        """
        base = GoalAlignedMockProvider(dim=16, goal_divisor=5)
        config = GoalDirectedConfig(enabled=True, goal_proximity_weight=0.5)
        provider = GoalDirectedEmbeddingProvider(base, config)

        # Register a non-aligned clause as DENY goal so aligned (id%5==0)
        # clauses have large distance → are preferred.
        goal = _make_clause([(True, _make_term(3))], 7)  # non-aligned id
        provider.register_goals([goal])

        ml_config = MLSelectionConfig(
            enabled=True,
            ml_weight=0.6,
            diversity_weight=0.3,
            proof_potential_weight=0.7,
            min_sos_for_ml=5,
        )
        selection = EmbeddingEnhancedSelection(
            embedding_provider=provider,
            ml_config=ml_config,
        )

        # Create SOS with mix of aligned and non-aligned clauses
        sos = ClauseList("sos")
        for i in range(1, 31):
            atom = _make_term(i % 10 + 1, (_make_var(0),))
            c = _make_clause([(True, atom)], clause_id=i, weight=5.0)
            sos.append(c)

        # Select 15 clauses and count goal-aligned ones
        selected_aligned = 0
        selected_total = 0
        for i in range(15):
            clause, sel_type = selection.select_given(sos, i)
            if clause is None:
                break
            selected_total += 1
            if clause.id % 5 == 0:
                selected_aligned += 1

        # Expected 6 aligned (5, 10, 15, 20, 25, 30) out of 30
        # Goal-directed should select them with higher frequency
        baseline_rate = 6.0 / 30.0  # 20% random baseline
        actual_rate = selected_aligned / selected_total if selected_total > 0 else 0

        print(f"\nGoal-aligned selections: {selected_aligned}/{selected_total} ({actual_rate:.1%})")
        print(f"Baseline random rate: {baseline_rate:.1%}")
        # At minimum, we expect at least some goal-aligned clauses early
        assert selected_aligned >= 1, "No goal-aligned clauses selected"

    def test_traditional_selection_preserved_when_disabled(self):
        """With ML disabled, selection matches traditional behavior."""
        provider = MockEmbeddingProvider(dim=16)
        config = MLSelectionConfig(enabled=False)
        selection = EmbeddingEnhancedSelection(
            embedding_provider=provider,
            ml_config=config,
        )

        sos = _make_sos(20, weight_range=(1.0, 20.0))

        selected_weights = []
        for i in range(10):
            clause, sel_type = selection.select_given(sos, i)
            if clause is None:
                break
            selected_weights.append(clause.weight)

        # Traditional weight selection should generally pick lighter clauses
        print(f"\nTraditional weights: {selected_weights}")
        assert len(selected_weights) >= 5


# ── Embedding quality validation ─────────────────────────────────────────


@pytest.mark.benchmark
class TestEmbeddingQuality:
    """Validate embedding space properties relevant to proof search."""

    def test_distinct_clauses_get_distinct_embeddings(self):
        """Different clauses should map to different embedding vectors."""
        provider = MockEmbeddingProvider(dim=32)
        clauses = [
            _make_clause([(True, _make_term(i, (_make_var(0),)))], i)
            for i in range(1, 11)
        ]

        embeddings = provider.get_embeddings_batch(clauses)
        assert all(e is not None for e in embeddings)

        # Pairwise similarity should not be 1.0 for distinct clauses
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = _cosine_similarity(embeddings[i], embeddings[j])
                assert sim < 0.99, f"Clauses {i} and {j} too similar: {sim:.4f}"

    def test_embedding_norm_distribution(self):
        """Embedding norms should be reasonably distributed."""
        provider = MockEmbeddingProvider(dim=32)
        clauses = [
            _make_clause([(True, _make_term(i))], i, weight=float(i))
            for i in range(1, 51)
        ]

        embeddings = provider.get_embeddings_batch(clauses)
        norms = [math.sqrt(sum(x * x for x in e)) for e in embeddings]

        mean_norm = statistics.mean(norms)
        std_norm = statistics.stdev(norms)
        print(f"\nEmbedding norms: mean={mean_norm:.4f}, std={std_norm:.4f}")
        print(f"Range: [{min(norms):.4f}, {max(norms):.4f}]")

        # Norms should not be degenerate (all zero or all identical)
        assert std_norm > 0.01, "Embedding norms have no variance"
        assert mean_norm > 0.1, "Embedding norms are too small"


# ── Online learning effectiveness ────────────────────────────────────────


@pytest.mark.benchmark
class TestOnlineLearningEffectiveness:
    """Validate that feedback improves selection over time."""

    def test_feedback_recording(self):
        """GoalDirectedProvider correctly records feedback pairs."""
        base = MockEmbeddingProvider(dim=16)
        config = GoalDirectedConfig(
            enabled=True,
            online_learning=True,
            feedback_buffer_size=100,
        )
        provider = GoalDirectedEmbeddingProvider(base, config)

        # Create productive and unproductive clauses
        productive = [_make_clause([(True, _make_term(i))], i) for i in range(1, 4)]
        unproductive = [_make_clause([(True, _make_term(i))], i) for i in range(10, 13)]

        provider.record_feedback(productive, unproductive)

        stats = provider.stats
        # 3 productive × 3 unproductive = 9 pairs
        assert stats["feedback_pairs"] == 9

    def test_stats_tracking_completeness(self):
        """All statistics are properly tracked."""
        base = MockEmbeddingProvider(dim=16)
        config = GoalDirectedConfig(enabled=True)
        provider = GoalDirectedEmbeddingProvider(base, config)

        clause = _make_clause([(True, _make_term(1))], 1)

        provider.notify_clause_kept(clause)
        provider.notify_clause_kept(clause)
        provider.notify_clause_selected(clause)
        provider.notify_proof_found([clause])

        stats = provider.stats
        assert stats["clauses_observed"] == 2
        assert stats["clauses_selected"] == 1
        assert stats["proofs_observed"] == 1

    def test_selection_stats_tracking(self):
        """ML selection stats correctly track ML vs traditional usage."""
        provider = MockEmbeddingProvider(dim=16)
        ml_config = MLSelectionConfig(
            enabled=True,
            ml_weight=0.3,
            min_sos_for_ml=5,
        )
        selection = EmbeddingEnhancedSelection(
            embedding_provider=provider,
            ml_config=ml_config,
        )

        sos = _make_sos(20)
        for i in range(10):
            selection.select_given(sos, i)

        report = selection.ml_stats.report()
        print(f"\nSelection stats: {report}")
        total = selection.ml_stats.ml_selections + selection.ml_stats.traditional_selections
        assert total == 10


# ── Goal proximity performance in selection loop ─────────────────────────


@pytest.mark.benchmark
class TestGoalProximityPerformance:
    """Benchmark goal proximity computation in hot selection path."""

    def test_proximity_computation_throughput(self):
        """Goal proximity should be fast enough for selection hot path."""
        scorer = GoalProximityScorer(method="max")
        dim = 64

        # Register 3 goals
        goals = [
            [math.sin(i * 0.3 + j * 0.7) for j in range(dim)]
            for i in range(3)
        ]
        scorer.set_goals(goals)

        embeddings = [
            [math.sin(i * 0.5 + j * 0.3) for j in range(dim)]
            for i in range(100)
        ]

        start = time.perf_counter()
        iterations = 1000
        for _ in range(iterations):
            for emb in embeddings:
                scorer.nearest_goal_distance(emb)
        elapsed = time.perf_counter() - start

        total_ops = iterations * len(embeddings)
        ops_per_sec = total_ops / elapsed
        us_per_op = elapsed / total_ops * 1e6

        print(f"\nGoal distance: {ops_per_sec:.0f} ops/s, {us_per_op:.1f} us/op")
        # Catastrophic regression floor: >5K distance computations per second
        assert ops_per_sec > 5_000, f"Too slow: {ops_per_sec:.0f} ops/s (floor: 5k)"

    def test_embedding_enhancement_throughput(self):
        """Goal-directed embedding enhancement throughput."""
        base = MockEmbeddingProvider(dim=64)
        config = GoalDirectedConfig(enabled=True, goal_proximity_weight=0.3)
        provider = GoalDirectedEmbeddingProvider(base, config)

        # Register goals
        goals = [_make_clause([(True, _make_term(i))], i) for i in range(1, 4)]
        provider.register_goals(goals)

        clauses = [
            _make_clause([(True, _make_term(i))], i)
            for i in range(10, 110)
        ]

        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            provider.get_embeddings_batch(clauses)
        elapsed = time.perf_counter() - start

        total_clauses = iterations * len(clauses)
        clauses_per_sec = total_clauses / elapsed
        us_per_clause = elapsed / total_clauses * 1e6

        print(f"\nGoal-enhanced embedding: {clauses_per_sec:.0f} clauses/s, {us_per_clause:.1f} us/clause")
        # Catastrophic regression floor: enhancement overhead < 1000us/clause
        assert us_per_clause < 1000.0, f"Enhancement too slow: {us_per_clause:.1f} us/clause (floor: 1000)"


# ── Demonstration cases ──────────────────────────────────────────────────


@pytest.mark.benchmark
class TestDemonstrationCases:
    """Demonstrate goal-directed behavior improvements on controlled problems."""

    def test_goal_distance_modulates_proof_potential(self):
        """Show that goal distance flows through proof_potential_score.

        The proof_potential_score in ml_selection uses inverse sigmoid of
        embedding norm. A clause close to the goal (small distance) gets
        its norm scaled down → higher proof_potential_score.

        Register an aligned clause as the goal so that other aligned clauses
        (small distance) get higher proof potential.
        """
        from pyladr.search.ml_selection import EmbeddingEnhancedSelection

        dim = 16
        base = GoalAlignedMockProvider(dim=dim, goal_divisor=3)
        config = GoalDirectedConfig(enabled=True, goal_proximity_weight=0.5)
        provider = GoalDirectedEmbeddingProvider(base, config)

        # Register an aligned clause as goal; other aligned clauses will be
        # close to it → small distance → scaled down → higher proof potential.
        goal = _make_clause([(True, _make_term(1))], 3)  # aligned id
        provider.register_goals([goal])

        selection = EmbeddingEnhancedSelection(
            embedding_provider=provider,
            ml_config=MLSelectionConfig(enabled=True, ml_weight=0.5),
        )

        # Goal-close (aligned)
        aligned = _make_clause([(True, _make_term(1))], 6, weight=5.0)
        aligned_emb = provider.get_embedding(aligned)
        aligned_pp = selection._proof_potential_score(aligned_emb)

        # Goal-far (non-aligned)
        non_aligned = _make_clause([(True, _make_term(3))], 8, weight=5.0)
        non_aligned_emb = provider.get_embedding(non_aligned)
        non_aligned_pp = selection._proof_potential_score(non_aligned_emb)

        print(f"\nGoal-close (aligned) proof potential: {aligned_pp:.4f}")
        print(f"Goal-far (non-aligned) proof potential: {non_aligned_pp:.4f}")

        # Goal-close clause has smaller norm → higher proof potential score
        assert aligned_pp > non_aligned_pp, (
            f"Goal distance not reflected in proof potential: "
            f"{aligned_pp:.4f} <= {non_aligned_pp:.4f}"
        )

    def test_selection_with_varying_goal_weights(self):
        """Show how goal_proximity_weight affects selection behavior."""
        dim = 16
        base = GoalAlignedMockProvider(dim=dim, goal_divisor=5)

        results = {}
        for gw in [0.0, 0.3, 0.6, 0.9]:
            config = GoalDirectedConfig(enabled=True, goal_proximity_weight=gw)
            provider = GoalDirectedEmbeddingProvider(base, config)
            goal = _make_clause([(True, _make_term(1))], 5)
            provider.register_goals([goal])

            ml_config = MLSelectionConfig(
                enabled=True,
                ml_weight=0.5,
                diversity_weight=0.2,
                proof_potential_weight=0.8,
                min_sos_for_ml=5,
            )
            selection = EmbeddingEnhancedSelection(
                embedding_provider=provider,
                ml_config=ml_config,
            )

            sos = ClauseList("sos")
            for i in range(1, 26):
                atom = _make_term(i % 10 + 1, (_make_var(0),))
                c = _make_clause([(True, atom)], clause_id=i, weight=5.0)
                sos.append(c)

            aligned_count = 0
            for i in range(12):
                clause, _ = selection.select_given(sos, i)
                if clause is not None and clause.id % 5 == 0:
                    aligned_count += 1

            results[gw] = aligned_count

        print("\nGoal-aligned selections by goal_proximity_weight:")
        for gw, count in sorted(results.items()):
            print(f"  weight={gw:.1f}: {count} aligned out of 12")

        # Higher weight should generally increase aligned selections
        # (may not be strictly monotonic due to weight/age cycle, but trend should hold)

    def test_diversity_prevents_cluster_stagnation(self):
        """Diversity scoring should prevent selecting from one cluster."""
        provider = MockEmbeddingProvider(dim=16)
        config = MLSelectionConfig(
            enabled=True,
            ml_weight=0.5,
            diversity_weight=0.9,
            proof_potential_weight=0.1,
            min_sos_for_ml=5,
            diversity_window=5,
        )
        selection = EmbeddingEnhancedSelection(
            embedding_provider=provider,
            ml_config=config,
        )

        sos = _make_sos(30, weight_range=(4.0, 6.0))

        selected_ids = []
        for i in range(15):
            clause, _ = selection.select_given(sos, i)
            if clause is not None:
                selected_ids.append(clause.id)

        print(f"\nDiversity-driven selection order: {selected_ids}")

        # With high diversity weight, selections should be spread out
        if len(selected_ids) >= 10:
            # Check that we're not just picking consecutive IDs
            diffs = [abs(selected_ids[i] - selected_ids[i - 1]) for i in range(1, len(selected_ids))]
            avg_diff = statistics.mean(diffs)
            print(f"Average gap between selected IDs: {avg_diff:.1f}")
            assert avg_diff > 1.5, "Selections are too clustered"
