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

Recursive ancestor tracking:
    The productive reference set is built recursively.  Goals sit at
    depth 0 with weight 1.0.  When any kept clause is found close to a
    current reference point (goal or existing ancestor), its parents are
    added as new reference points at ``depth+1`` with weight
    ``ancestor_decay**(depth+1)``.  Expansion halts when the new depth
    exceeds ``ancestor_max_depth`` or the new weight falls below
    ``ancestor_min_weight``.

    Distance to the combined reference set uses the weighted formula
    ``(1 - max_i w_i * cos(e, r_i)) / 2`` so that orthogonality maps to
    the neutral value 0.5, identity with a depth-0 goal to 0.0, and
    identity with a deep ancestor to a value increasing gently with
    depth.

Thread-safety:
    Goal registration and embedding reads use a readers–writer lock
    to allow concurrent reads during inference while serialising
    goal updates.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term, get_variable_term

if TYPE_CHECKING:
    from pyladr.search.ml_selection import EmbeddingProvider

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class ReferencePoint:
    """A depth-tagged entry in the productive reference set.

    Attributes:
        embedding: The clause's embedding vector.
        depth: Number of hops from a goal. 0 = goal, 1 = parent of a
            goal-proximate clause, etc.
        weight: ``ancestor_decay**depth``, pre-computed at insertion and
            used to down-weight similarity contributions from deeper
            ancestors when scoring.
    """

    embedding: list[float]
    depth: int
    weight: float


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
        ancestor_tracking: Enable recursive productive-ancestor expansion.
        ancestor_proximity_threshold: Reference distance below which a
            clause's parents are eligible for expansion.
        ancestor_max_count: Cap on total retained reference points.
        ancestor_decay: Weight multiplier per depth level. ``0.8`` means
            depth-1 ancestors contribute at 80% of the goal's influence,
            depth-2 at 64%, and so on.
        ancestor_min_weight: Stop recursion when
            ``ancestor_decay**depth`` falls below this cutoff.
        ancestor_max_depth: Hard cap on recursion depth (CLI-exposed).
    """

    enabled: bool = False
    goal_proximity_weight: float = 0.3
    proximity_method: str = "max"  # unused; kept for backwards compatibility
    online_learning: bool = False
    feedback_buffer_size: int = 1000
    ancestor_tracking: bool = True
    ancestor_proximity_threshold: float = 0.3  # ref dist below which parents are expanded
    ancestor_max_count: int = 500              # max reference points to retain
    ancestor_decay: float = 0.8                # weight multiplier per depth level
    ancestor_min_weight: float = 0.1           # stop recursion when decay^depth < this
    ancestor_max_depth: int = 5                # hard cap on depth (CLI-exposed)


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

    Goals are typically DENY-justified (negated-conjecture) clauses,
    deskolemized so that small distance = structurally similar to the
    goal = proof-useful.
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
    - Embeddings are scaled by goal distance: clauses CLOSE to the
      deskolemized goal reference get smaller norms, which the
      proof_potential_score interprets as more promising (via inverse
      sigmoid on norm).  Lower norm → higher score → more likely selected.
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
        "_ancestor_reference",
        "_ancestor_lock",
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
        self._ancestor_reference: deque[ReferencePoint] = deque(
            maxlen=self._config.ancestor_max_count
        )
        self._ancestor_lock = threading.Lock()

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

        Each goal is normalised before embedding:
        1. Literal signs are forced to positive so the embedding points
           toward the goal's structural content rather than away from it.
        2. Constants are replaced by variables (deskolemization) so that
           any derived clause whose variables can be substituted to match
           the goal's constants appears at distance 0 — exactly the clauses
           that can directly resolve with the goal literal.

        Note: alpha-equivalent clauses produce identical walk-token sequences
        and therefore identical embeddings regardless of whether the result
        is served from the embedding cache or recomputed, so the cache does
        not distort the goal embeddings.

        Goals that cannot be embedded are silently skipped.
        """
        goal_embeddings: list[list[float]] = []
        for g in goals:
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

    # ── Productive ancestor tracking ───────────────────────────────────

    def try_expand_from_clause(
        self,
        embedding: list[float],
        parent_clauses: list[Clause],
    ) -> bool:
        """Recursively expand the productive reference set.

        Checks if ``embedding`` is close to any current reference point
        (goal at depth 0 weight 1.0, or any recorded ancestor at its
        stored depth/weight). If the weighted distance is below
        ``ancestor_proximity_threshold``, adds the parent clauses as new
        reference points at ``best_depth + 1`` with weight
        ``ancestor_decay**(best_depth+1)``, subject to
        ``ancestor_max_depth`` and ``ancestor_min_weight`` stopping
        conditions and the ``ancestor_max_count`` cap.

        Returns True if any ancestor was actually added.
        """
        if not self._config.ancestor_tracking:
            return False

        # Snapshot the reference set.
        with self._goal_scorer._lock:
            goal_embs = list(self._goal_scorer._goal_embeddings)
        with self._ancestor_lock:
            refs = list(self._ancestor_reference)

        if not goal_embs and not refs:
            return False

        # Find the closest reference point by weighted similarity.
        best_weighted_sim = -math.inf
        best_depth = 0
        for g in goal_embs:
            sim = _cosine_similarity(embedding, g)
            if sim > best_weighted_sim:
                best_weighted_sim = sim
                best_depth = 0
        for r in refs:
            ws = r.weight * _cosine_similarity(embedding, r.embedding)
            if ws > best_weighted_sim:
                best_weighted_sim = ws
                best_depth = r.depth

        dist = (1.0 - best_weighted_sim) / 2.0
        if dist >= self._config.ancestor_proximity_threshold:
            return False  # not close enough to any reference

        new_depth = best_depth + 1
        new_weight = self._config.ancestor_decay ** new_depth

        # Stopping conditions.
        if new_depth > self._config.ancestor_max_depth:
            return False
        if new_weight < self._config.ancestor_min_weight:
            return False

        expanded = False
        for parent in parent_clauses:
            emb = self._base.get_embedding(parent)
            if emb is None:
                continue
            with self._ancestor_lock:
                self._ancestor_reference.append(
                    ReferencePoint(embedding=emb, depth=new_depth, weight=new_weight)
                )
            expanded = True

        return expanded

    def record_productive_ancestors(self, parent_clauses: list[Clause]) -> None:
        """Backward-compatible wrapper; inserts parents at depth 1.

        Older call sites (and tests) use this API directly without first
        probing the reference set. It unconditionally inserts the given
        parents at depth 1 with weight ``ancestor_decay``, respecting
        the ``ancestor_min_weight`` / ``ancestor_max_depth`` /
        ``ancestor_max_count`` stops.  Prefer
        :meth:`try_expand_from_clause` for the full recursive behaviour.
        """
        if not self._config.ancestor_tracking:
            return
        new_depth = 1
        new_weight = self._config.ancestor_decay ** new_depth
        if new_depth > self._config.ancestor_max_depth:
            return
        if new_weight < self._config.ancestor_min_weight:
            return
        for parent in parent_clauses:
            emb = self._base.get_embedding(parent)
            if emb is None:
                continue
            with self._ancestor_lock:
                self._ancestor_reference.append(
                    ReferencePoint(embedding=emb, depth=new_depth, weight=new_weight)
                )

    def clear_ancestors(self) -> None:
        with self._ancestor_lock:
            self._ancestor_reference.clear()

    @property
    def num_ancestors(self) -> int:
        with self._ancestor_lock:
            return len(self._ancestor_reference)

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
                "num_ancestors": self.num_ancestors,
                "enabled": self._config.enabled,
            }

    def stats_report(self) -> str:
        """Human-readable statistics report."""
        s = self.stats
        return (
            f"goal_directed: enabled={s['enabled']}, "
            f"goals={s['num_goals']}, "
            f"ancestors={s['num_ancestors']}, "
            f"observed={s['clauses_observed']}, "
            f"selected={s['clauses_selected']}, "
            f"proofs={s['proofs_observed']}, "
            f"feedback_pairs={s['feedback_pairs']}"
        )

    # ── Internal ───────────────────────────────────────────────────────

    def _nearest_reference_distance(self, embedding: list[float]) -> float:
        """Distance to the nearest reference point (goal or ancestor).

        Uses the weighted similarity formula
        ``(1 - max_i w_i * cos(embedding, r_i)) / 2`` where goals have
        weight 1.0 and ancestors use their pre-computed ``decay**depth``
        weight.  Returns 0.5 (neutral) when the reference set is empty,
        0.0 when ``embedding`` is identical to a depth-0 goal, and
        increases gently with depth when identical to a deeper
        ancestor.  Never exceeds 1.0.
        """
        with self._goal_scorer._lock:
            goal_embs = list(self._goal_scorer._goal_embeddings)
        with self._ancestor_lock:
            refs = list(self._ancestor_reference)

        if not goal_embs and not refs:
            return 0.5

        best_weighted_sim = max(
            [1.0 * _cosine_similarity(embedding, g) for g in goal_embs] +
            [r.weight * _cosine_similarity(embedding, r.embedding) for r in refs],
            default=0.0,
        )
        return (1.0 - best_weighted_sim) / 2.0

    def _enhance_embedding(self, embedding: list[float]) -> list[float]:
        """Modulate embedding norm by distance to the nearest reference.

        The reference set is the union of registered goal embeddings
        (depth 0, weight 1.0) and recursively-collected productive
        ancestor embeddings (each tagged with its own depth and
        ``decay**depth`` weight). See :meth:`_nearest_reference_distance`
        for the weighted-similarity formula.

        Strategy: reduce the embedding norm for reference-close (proof-useful)
        clauses. The proof_potential_score in ml_selection.py interprets
        smaller norms as higher potential (via inverse sigmoid), so a smaller
        scale → higher score → more likely to be selected.

        The scaling factor is:
            scale = 1.0 - goal_proximity_weight * (1.0 - distance)
        """
        distance = self._nearest_reference_distance(embedding)
        weight = self._config.goal_proximity_weight
        scale = 1.0 - weight * (1.0 - distance)
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
