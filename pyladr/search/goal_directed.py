"""Goal-directed embedding provider for proof search.

Wraps any EmbeddingProvider with goal-distance scoring, enabling
goal-directed clause selection. When a proof goal (negated conjecture)
is registered, clause embeddings are modulated by their cosine distance
to the goal in embedding space.

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

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term, get_variable_term

if TYPE_CHECKING:
    from pyladr.search.ml_selection import EmbeddingProvider

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GoalDirectedConfig:
    """Configuration for goal-directed embedding enhancement.

    Attributes:
        enabled: Master switch. When False, provider is pure passthrough.
        goal_proximity_weight: How much goal distance influences the
            embedding modulation. 0.0 = no influence, 1.0 = strong.
        proximity_method: Unused. Kept for backwards compatibility.
        online_learning: Enable online contrastive learning from feedback.
        feedback_buffer_size: Max feedback pairs to retain for learning.
    """

    enabled: bool = False
    goal_proximity_weight: float = 0.3
    proximity_method: str = "max"  # unused; kept for backwards compatibility
    online_learning: bool = False
    feedback_buffer_size: int = 1000


# ── Goal clause normalisation ──────────────────────────────────────────────


def _replace_constants_with_vars(term: Term, const_to_var: dict[int, int]) -> Term:
    """Recursively replace constants with variables in a term tree.

    Each distinct constant symnum gets its own fresh variable number,
    allocated in order of first appearance in ``const_to_var``.
    """
    if term.is_variable:
        return term
    if term.is_constant:
        symnum = term.symnum
        if symnum not in const_to_var:
            const_to_var[symnum] = len(const_to_var)
        return get_variable_term(const_to_var[symnum])
    # Complex term: recurse into args
    new_args = tuple(_replace_constants_with_vars(a, const_to_var) for a in term.args)
    if new_args == term.args:
        return term
    from pyladr.core.term import get_rigid_term
    return get_rigid_term(term.symnum, term.arity, new_args)


def _deskolemize_clause(clause: Clause) -> Clause:
    """Return a copy of *clause* with all constants replaced by variables.

    Goal clauses are produced by Skolemizing the negated conjecture, so every
    constant in them is a Skolem constant.  Replacing them with variables
    makes the embedding represent pure structural shape (e.g. P(i(x,y)))
    rather than specific constant identities, giving a more meaningful
    distance measure for arbitrary derived clauses.

    Signs are also forced to True (caller should already have done this, but
    we do it here as well for safety).
    """
    const_to_var: dict[int, int] = {}
    new_lits = tuple(
        Literal(sign=True, atom=_replace_constants_with_vars(lit.atom, const_to_var))
        for lit in clause.literals
    )
    return Clause(literals=new_lits, id=clause.id)


# ── Goal distance scorer ───────────────────────────────────────────────────


class GoalDistanceScorer:
    """Computes cosine distance of clause embeddings to registered goal embeddings.

    "Distance" is (1 - cosine_similarity) / 2, mapping to [0, 1]:
        distance = 0.0  →  identical to goal
        distance = 0.5  →  orthogonal to goal
        distance = 1.0  →  maximally dissimilar to goal

    With no goals, returns 0.5 (neutral).

    Goals are typically DENY-justified (negated-conjecture) clauses.
    A large distance from the DENY reference means the clause is close
    to the actual (un-negated) goal, making it promising for proof search.
    """

    __slots__ = ("_goal_embeddings", "_lock")

    def __init__(self) -> None:
        self._goal_embeddings: list[list[float]] = []
        self._lock = threading.Lock()

    def set_goals(self, goal_embeddings: list[list[float]]) -> None:
        """Replace all goal embeddings."""
        with self._lock:
            self._goal_embeddings = list(goal_embeddings)

    def clear(self) -> None:
        """Remove all goal embeddings."""
        with self._lock:
            self._goal_embeddings = []

    def remove_goal(self, index: int) -> None:
        """Remove the goal embedding at the given index."""
        with self._lock:
            if 0 <= index < len(self._goal_embeddings):
                self._goal_embeddings.pop(index)

    @property
    def num_goals(self) -> int:
        return len(self._goal_embeddings)

    def nearest_goal_distance(self, embedding: list[float] | None) -> float:
        """Distance to the nearest (most similar) goal.

        Returns (1 - max_cosine_similarity) / 2.  Lower = closer to some goal.
        Use with argmin to select the clause nearest to any goal.

        Returns:
            Float in [0, 1]. 0.0 = identical to some goal.
            0.5 = orthogonal to all goals. 1.0 = opposite to all goals.
        """
        if embedding is None:
            return 0.5

        with self._lock:
            goals = self._goal_embeddings

        if not goals:
            return 0.5

        best_sim = max(_cosine_similarity(embedding, g) for g in goals)
        return (1.0 - best_sim) / 2.0

    def farthest_goal_distance(self, embedding: list[float] | None) -> float:
        """Distance to the farthest (least similar) goal.

        Returns (1 - min_cosine_similarity) / 2.  Lower = all goals are close.
        Use with argmin to implement minimax-distance clause selection: pick
        the clause whose maximum distance to any goal is minimised, favouring
        clauses relevant to every goal rather than just one.

        Returns:
            Float in [0, 1]. 0.0 = identical to all goals.
            0.5 = orthogonal to all goals. 1.0 = opposite to some goal.
        """
        if embedding is None:
            return 0.5

        with self._lock:
            goals = self._goal_embeddings

        if not goals:
            return 0.5

        worst_sim = min(_cosine_similarity(embedding, g) for g in goals)
        return (1.0 - worst_sim) / 2.0


def GoalProximityScorer(method: str = "max") -> GoalDistanceScorer:  # type: ignore[misc]
    """Backwards-compatibility factory. The ``method`` parameter is ignored."""
    return GoalDistanceScorer()


# ── Goal-directed embedding provider ──────────────────────────────────────


class GoalDirectedEmbeddingProvider:
    """Wraps an EmbeddingProvider with goal-directed distance-based enhancement.

    When enabled and goals are registered:
    - Embeddings are scaled by goal distance: clauses far from the
      DENY-justified reference (i.e. close to the actual goal) get
      smaller norms, which the proof_potential_score interprets as
      more promising (via inverse sigmoid on norm).
    - Goal distance information flows through the existing ML scoring
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
        self._goal_scorer = GoalDistanceScorer()
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
        """Return embedding, optionally enhanced with goal-distance modulation."""
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
        """Register goal clauses for distance scoring.

        Normalises each goal before embedding:
        1. Literal signs are forced to positive so the embedding points
           toward the goal's structural content, not away from it.
        2. Skolem constants are replaced by variables so the embedding
           captures pure structural shape (e.g. P(i(x,y)) instead of
           P(i(c1,c2))).  This makes distance meaningful for derived
           clauses that share the same predicate/function structure but
           contain different constants or variables.

        Goals that cannot be embedded are silently skipped.
        """
        goal_embeddings: list[list[float]] = []
        for g in goals:
            # Normalise: strip signs and replace Skolem constants with
            # variables so the embedding captures structural shape only.
            normalised = _deskolemize_clause(g)
            emb = self._base.get_embedding(normalised)
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
    def goal_scorer(self) -> GoalDistanceScorer:
        """Access the goal distance scorer (for testing/inspection)."""
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
        """Modulate embedding norm by distance to the nearest goal.

        Goal embeddings are stored sign-stripped (positive), so a small
        distance means the clause is structurally similar to the goal content
        and is therefore proof-useful.

        Strategy: reduce the embedding norm for goal-close (proof-useful)
        clauses. The proof_potential_score in ml_selection.py interprets
        smaller norms as higher potential (via inverse sigmoid), so a smaller
        scale → higher score → more likely to be selected.

        The scaling factor is:
            scale = 1.0 - goal_proximity_weight * (1.0 - distance)

        Where distance = nearest_goal_distance ∈ [0, 1]:
        - distance = 0.0 (clause identical to goal content, very proof-useful):
          scale = 1 - weight  (smallest norm → highest proof potential score)
        - distance = 1.0 (clause opposite to goal, not proof-useful):
          scale = 1.0 (unchanged)
        - distance = 0.5 (neutral): scale = 1 - weight/2
        """
        distance = self._goal_scorer.nearest_goal_distance(embedding)
        weight = self._config.goal_proximity_weight

        # Scale factor: lower for goal-close (proof-useful) clauses
        # → smaller norm → higher proof_potential_score
        scale = 1.0 - weight * (1.0 - distance)

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
