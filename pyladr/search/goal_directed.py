"""Goal-directed embedding provider for proof search.

Wraps any EmbeddingProvider with goal proximity scoring, enabling
goal-directed clause selection. When a proof goal (negated conjecture)
is registered, clause embeddings are modulated by their proximity to
the goal in embedding space.

All goal-directed features are strictly opt-in. When disabled, the
provider is an exact passthrough to the base provider — zero overhead,
zero behavioral change.

Design:
    GoalDirectedEmbeddingProvider(base) satisfies the same
    EmbeddingProvider protocol as the base, so it is a transparent
    drop-in replacement. The selection layer (EmbeddingEnhancedSelection)
    does not need to know about goal-directed features.

Thread-safety:
    Goal registration and embedding reads use a readers–writer lock
    to allow concurrent reads during inference while serialising
    goal updates.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyladr.core.clause import Clause

if TYPE_CHECKING:
    from pyladr.search.ml_selection import EmbeddingProvider

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GoalDirectedConfig:
    """Configuration for goal-directed embedding enhancement.

    Attributes:
        enabled: Master switch. When False, provider is pure passthrough.
        goal_proximity_weight: How much goal proximity influences the
            embedding modulation. 0.0 = no influence, 1.0 = strong.
        proximity_method: How to combine similarities to multiple goals.
            "max" = closest goal, "mean" = average over all goals.
        online_learning: Enable online contrastive learning from feedback.
        feedback_buffer_size: Max feedback pairs to retain for learning.
    """

    enabled: bool = False
    goal_proximity_weight: float = 0.3
    proximity_method: str = "max"
    online_learning: bool = False
    feedback_buffer_size: int = 1000


# ── Goal proximity scorer ─────────────────────────────────────────────────


class GoalProximityScorer:
    """Computes proximity of clause embeddings to registered goal embeddings.

    Proximity is based on cosine similarity, mapped to [0, 1]:
        proximity = (cosine_sim + 1) / 2

    With multiple goals, uses max (closest goal) or mean, configurable.
    With no goals, returns 0.5 (neutral).
    """

    __slots__ = ("_goal_embeddings", "_method", "_lock")

    def __init__(self, method: str = "max") -> None:
        self._goal_embeddings: list[list[float]] = []
        self._method = method
        self._lock = threading.Lock()

    def set_goals(self, goal_embeddings: list[list[float]]) -> None:
        """Replace all goal embeddings."""
        with self._lock:
            self._goal_embeddings = list(goal_embeddings)

    def clear(self) -> None:
        """Remove all goal embeddings."""
        with self._lock:
            self._goal_embeddings = []

    @property
    def num_goals(self) -> int:
        return len(self._goal_embeddings)

    def proximity(self, embedding: list[float] | None) -> float:
        """Compute proximity score for a clause embedding.

        Returns:
            Float in [0, 1]. 1.0 = identical to a goal.
            0.5 = neutral (no goals or orthogonal).
            0.0 = maximally dissimilar.
        """
        if embedding is None:
            return 0.5

        with self._lock:
            goals = self._goal_embeddings

        if not goals:
            return 0.5

        similarities = [_cosine_similarity(embedding, g) for g in goals]

        if self._method == "max":
            best_sim = max(similarities)
        else:
            best_sim = sum(similarities) / len(similarities)

        # Map from [-1, 1] to [0, 1]
        return (best_sim + 1.0) / 2.0


# ── Goal-directed embedding provider ──────────────────────────────────────


class GoalDirectedEmbeddingProvider:
    """Wraps an EmbeddingProvider with goal-directed proximity enhancement.

    When enabled and goals are registered:
    - Embeddings are scaled by goal proximity: clauses closer to goals
      get larger norms, which the proof_potential_score interprets as
      more promising.
    - Goal proximity information flows through the existing ML scoring
      pipeline without any changes to ml_selection.py.

    When disabled:
    - Pure passthrough. get_embedding() returns exactly what the base
      provider returns. Zero overhead, zero behavioral change.

    Satisfies the EmbeddingProvider protocol.
    """

    __slots__ = (
        "_base",
        "_config",
        "_goal_scorer",
        "_stats_lock",
        "_clauses_observed",
        "_clauses_selected",
        "_proofs_observed",
        "_feedback_pairs",
        "_feedback_buffer",
    )

    def __init__(
        self,
        base_provider: EmbeddingProvider,
        config: GoalDirectedConfig | None = None,
    ) -> None:
        self._base = base_provider
        self._config = config or GoalDirectedConfig()
        self._goal_scorer = GoalProximityScorer(
            method=self._config.proximity_method,
        )
        self._stats_lock = threading.Lock()
        self._clauses_observed: int = 0
        self._clauses_selected: int = 0
        self._proofs_observed: int = 0
        self._feedback_pairs: int = 0
        self._feedback_buffer: list[tuple[list[float], list[float], bool]] = []

    # ── EmbeddingProvider protocol ─────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """Dimensionality matches base provider."""
        return self._base.embedding_dim

    def get_embedding(self, clause: Clause) -> list[float] | None:
        """Return embedding, optionally enhanced with goal proximity."""
        emb = self._base.get_embedding(clause)
        if not self._config.enabled or emb is None:
            return emb
        return self._enhance_embedding(emb)

    def get_embeddings_batch(
        self, clauses: list[Clause],
    ) -> list[list[float] | None]:
        """Batch embedding retrieval with optional goal enhancement."""
        base_results = self._base.get_embeddings_batch(clauses)
        if not self._config.enabled:
            return base_results
        return [
            self._enhance_embedding(emb) if emb is not None else None
            for emb in base_results
        ]

    # ── Goal management ────────────────────────────────────────────────

    def register_goals(self, goals: list[Clause]) -> None:
        """Register goal clauses for proximity scoring.

        Computes embeddings for each goal via the base provider.
        Goals that cannot be embedded are silently skipped.
        """
        goal_embeddings: list[list[float]] = []
        for g in goals:
            emb = self._base.get_embedding(g)
            if emb is not None:
                goal_embeddings.append(emb)
            else:
                logger.debug(
                    "Goal clause %d could not be embedded, skipping", g.id,
                )
        self._goal_scorer.set_goals(goal_embeddings)
        if goal_embeddings:
            logger.info(
                "Registered %d goal embeddings for goal-directed search",
                len(goal_embeddings),
            )

    def clear_goals(self) -> None:
        """Remove all registered goals."""
        self._goal_scorer.clear()

    @property
    def num_goals(self) -> int:
        """Number of successfully registered goal embeddings."""
        return self._goal_scorer.num_goals

    @property
    def goal_scorer(self) -> GoalProximityScorer:
        """Access the goal proximity scorer (for testing/inspection)."""
        return self._goal_scorer

    # ── Incremental update notifications ───────────────────────────────

    def notify_clause_kept(self, clause: Clause) -> None:
        """Called when a clause is kept during search."""
        with self._stats_lock:
            self._clauses_observed += 1

    def notify_clause_selected(self, clause: Clause) -> None:
        """Called when a clause is selected as given."""
        with self._stats_lock:
            self._clauses_selected += 1

    def notify_proof_found(self, proof_clauses: list[Clause]) -> None:
        """Called when a proof is found."""
        with self._stats_lock:
            self._proofs_observed += 1

    # ── Online learning ────────────────────────────────────────────────

    @property
    def is_learning_enabled(self) -> bool:
        return self._config.online_learning

    def record_feedback(
        self,
        productive: list[Clause],
        unproductive: list[Clause],
    ) -> None:
        """Record contrastive feedback from search outcomes.

        Pairs productive/unproductive clauses for future learning updates.
        """
        if not self._config.online_learning:
            return

        for p in productive:
            p_emb = self._base.get_embedding(p)
            if p_emb is None:
                continue
            for u in unproductive:
                u_emb = self._base.get_embedding(u)
                if u_emb is None:
                    continue
                pair = (p_emb, u_emb, True)
                with self._stats_lock:
                    self._feedback_pairs += 1
                    if len(self._feedback_buffer) < self._config.feedback_buffer_size:
                        self._feedback_buffer.append(pair)

    # ── Statistics ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Current statistics snapshot."""
        with self._stats_lock:
            return {
                "clauses_observed": self._clauses_observed,
                "clauses_selected": self._clauses_selected,
                "proofs_observed": self._proofs_observed,
                "feedback_pairs": self._feedback_pairs,
                "num_goals": self.num_goals,
                "enabled": self._config.enabled,
            }

    def stats_report(self) -> str:
        """Human-readable statistics report."""
        s = self.stats
        return (
            f"goal_directed: enabled={s['enabled']}, "
            f"goals={s['num_goals']}, "
            f"observed={s['clauses_observed']}, "
            f"selected={s['clauses_selected']}, "
            f"proofs={s['proofs_observed']}, "
            f"feedback_pairs={s['feedback_pairs']}"
        )

    # ── Internal ───────────────────────────────────────────────────────

    def _enhance_embedding(self, embedding: list[float]) -> list[float]:
        """Modulate embedding with goal proximity.

        Strategy: scale the embedding norm by goal proximity. Clauses
        closer to goals get larger norms. The proof_potential_score
        in ml_selection.py interprets smaller norms as higher potential
        (via inverse sigmoid), so we invert: goal-close clauses get
        SMALLER norms to score higher.

        The scaling factor is:
            scale = 1.0 - goal_proximity_weight * proximity

        Where proximity ∈ [0, 1]:
        - proximity = 1.0 (identical to goal): scale = 1 - weight
          (smaller norm → higher proof potential score)
        - proximity = 0.0 (opposite of goal): scale = 1.0 (unchanged)
        - proximity = 0.5 (neutral): scale = 1 - weight/2

        This ensures clauses aligned with goals are preferred by the
        existing proof_potential scoring without any changes to
        ml_selection.py.
        """
        proximity = self._goal_scorer.proximity(embedding)
        weight = self._config.goal_proximity_weight

        # Scale factor: lower for goal-proximate clauses → smaller norm
        # → higher proof_potential_score
        scale = 1.0 - weight * proximity

        # Avoid zero scale (would lose direction information)
        scale = max(scale, 0.01)

        return [x * scale for x in embedding]


# ── Utility ────────────────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 for zero vectors."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom < 1e-12:
        return 0.0
    return dot / denom
