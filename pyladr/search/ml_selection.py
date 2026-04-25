"""Embedding-enhanced clause selection for ML-guided search.

Extends the standard GivenSelection with optional ML-based scoring
that blends clause embeddings with traditional weight/age metrics.

All ML features are strictly opt-in. When disabled or when embeddings
are unavailable, behavior is identical to the original GivenSelection.
"""

from __future__ import annotations

import logging
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyladr.core.clause import Clause
from pyladr.protocols import EmbeddingProvider
from pyladr.search.selection import (
    GivenSelection,
    SelectionOrder,
    SelectionRule,
    default_clause_weight,
)
from pyladr.search.state import ClauseList

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ── Embedding provider protocol ─────────────────────────────────────────────

# Re-exported from pyladr.protocols for backward compatibility.
# All existing ``from pyladr.search.ml_selection import EmbeddingProvider``
# imports continue to work. New code should import from pyladr.protocols.
__all__ = ["EmbeddingProvider"]  # noqa: F811 — re-export


# ── Selection configuration ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MLSelectionConfig:
    """Configuration for embedding-enhanced selection.

    Attributes:
        enabled: Master switch for ML-enhanced selection.
        ml_weight: Blending weight for ML score vs traditional score.
            0.0 = pure traditional, 1.0 = pure ML. Default 0.3.
        diversity_weight: Weight of diversity component in ML score.
        proof_potential_weight: Weight of proof-potential component.
        diversity_window: Number of recent givens tracked for diversity.
        min_sos_for_ml: Minimum SOS size before ML selection activates.
            Below this threshold, traditional selection is used.
        fallback_on_error: Fall back to traditional on any ML error.
        log_selections: Log ML vs traditional selection decisions.
        complexity_normalization: Normalize ML scores by clause complexity
            to prevent pathological clause growth with high ml_weight.
    """

    enabled: bool = False
    ml_weight: float = 0.3
    diversity_weight: float = 0.5
    proof_potential_weight: float = 0.5
    diversity_window: int = 20
    min_sos_for_ml: int = 10
    fallback_on_error: bool = True
    log_selections: bool = True
    complexity_normalization: bool = True


# ── ML selection statistics ──────────────────────────────────────────────────


@dataclass(slots=True)
class MLSelectionStats:
    """Tracks ML selection effectiveness for monitoring."""

    ml_selections: int = 0
    traditional_selections: int = 0
    fallback_count: int = 0
    embedding_miss_count: int = 0
    avg_ml_score: float = 0.0
    _ml_score_sum: float = 0.0

    def record_ml_selection(self, score: float) -> None:
        self.ml_selections += 1
        self._ml_score_sum += score
        self.avg_ml_score = self._ml_score_sum / self.ml_selections

    def record_traditional(self) -> None:
        self.traditional_selections += 1

    def record_fallback(self) -> None:
        self.fallback_count += 1
        self.traditional_selections += 1

    def record_embedding_miss(self) -> None:
        self.embedding_miss_count += 1

    def report(self) -> str:
        total = self.ml_selections + self.traditional_selections
        if total == 0:
            return "ml_selection: no selections made"
        ml_pct = 100.0 * self.ml_selections / total
        return (
            f"ml_selection: {self.ml_selections}/{total} ML ({ml_pct:.1f}%), "
            f"fallbacks={self.fallback_count}, "
            f"embedding_misses={self.embedding_miss_count}, "
            f"avg_ml_score={self.avg_ml_score:.4f}"
        )


# ── Embedding-enhanced selection ─────────────────────────────────────────────


@dataclass(slots=True)
class EmbeddingEnhancedSelection(GivenSelection):
    """Clause selection with optional ML-guided scoring.

    Extends GivenSelection by blending traditional weight/age selection
    with embedding-based diversity and proof-potential scores. Falls back
    to pure traditional selection when ML is disabled or unavailable.

    The selection algorithm:
    1. Determine if this selection step should use ML (based on ratio cycle
       and config). Weight-based steps may use ML; age-based steps always
       use traditional ordering to maintain fairness.
    2. For ML-enhanced steps, score each SOS clause with a blend of:
       - Traditional weight score (normalized, lower is better)
       - Diversity score (cosine distance from recent given embeddings)
       - Proof-potential score (if provider supports it)
    3. Select the highest-scoring clause.

    Usage:
        provider = MyEmbeddingProvider(model)
        config = MLSelectionConfig(enabled=True, ml_weight=0.3)
        selection = EmbeddingEnhancedSelection(
            embedding_provider=provider,
            ml_config=config,
        )
        search = GivenClauseSearch(options, selection=selection)
    """

    embedding_provider: EmbeddingProvider | None = None
    ml_config: MLSelectionConfig = field(default_factory=MLSelectionConfig)
    ml_stats: MLSelectionStats = field(default_factory=MLSelectionStats)

    # Recent given clause embeddings for diversity scoring
    _recent_embeddings: deque[list[float]] = field(
        default_factory=lambda: deque(maxlen=20),
    )

    # Optional repetition tracking (set externally if used)
    repetition_tracker: object | None = None
    repetition_penalty: float = 0.0

    def __post_init__(self) -> None:
        # Explicit parent call — super() doesn't work with slots=True
        # dataclass inheritance in Python 3.10+
        GivenSelection.__post_init__(self)
        # Sync diversity window from config
        if self.ml_config.diversity_window != self._recent_embeddings.maxlen:
            self._recent_embeddings = deque(
                maxlen=self.ml_config.diversity_window,
            )

    def select_given(
        self,
        sos: ClauseList,
        given_count: int,
    ) -> tuple[Clause | None, str]:
        """Select the next given clause, optionally using ML scoring.

        For age-based selection steps, always uses traditional FIFO to
        maintain fairness guarantees. For weight-based steps, may blend
        ML scoring when conditions are met.

        Returns:
            (clause, selection_type) or (None, "") if SOS is empty.
        """
        if sos.is_empty:
            return None, ""

        # Use parent's ratio-cycle rule selection
        rule = self._get_current_rule()
        self._advance_cycle()

        # Only apply ML scoring on weight-based selection steps.
        # Age-based selection must remain FIFO for fairness.
        if (
            rule.order == SelectionOrder.WEIGHT
            and self._should_use_ml(sos)
        ):
            selected, score = self._ml_select(sos)
            if selected is not None:
                sos.remove(selected)
                rule.selected += 1
                self.ml_stats.record_ml_selection(score)
                self._record_embedding(selected)
                sel_type = f"{rule.name}+ML"
                if self.ml_config.log_selections:
                    logger.info(
                        "ML selection: id=%d, score=%.4f, weight=%.1f",
                        selected.id, score, selected.weight,
                    )
                return selected, sel_type

            # ML failed — fall back to traditional
            self.ml_stats.record_fallback()

        # Traditional selection (original behavior)
        selected = self._select_by_order(sos, rule.order)
        if selected is None:
            return None, ""

        sos.remove(selected)
        rule.selected += 1
        self.ml_stats.record_traditional()
        self._record_embedding(selected)
        return selected, rule.name

    def _should_use_ml(self, sos: ClauseList) -> bool:
        """Check if ML selection should be attempted."""
        return (
            self.ml_config.enabled
            and self.embedding_provider is not None
            and sos.length >= self.ml_config.min_sos_for_ml
        )

    def _ml_select(
        self, sos: ClauseList,
    ) -> tuple[Clause | None, float]:
        """Score all SOS clauses with blended ML + traditional metrics.

        Returns (best_clause, best_score) or (None, 0.0) on failure.
        """
        try:
            return self._ml_select_inner(sos)
        except Exception:
            if self.ml_config.fallback_on_error:
                logger.debug("ML selection failed, falling back", exc_info=True)
                return None, 0.0
            raise

    def _ml_select_inner(
        self, sos: ClauseList,
    ) -> tuple[Clause | None, float]:
        """Inner ML selection logic (may raise on provider errors)."""
        provider = self.embedding_provider
        assert provider is not None

        clauses = list(sos)
        if not clauses:
            return None, 0.0

        # Collect embeddings via batch API
        embeddings = provider.get_embeddings_batch(clauses)

        # Compute weight statistics in a single pass
        min_w = math.inf
        max_w = -math.inf
        for c in clauses:
            w = c.weight
            if w < min_w:
                min_w = w
            if w > max_w:
                max_w = w
        w_range = max_w - min_w if max_w > min_w else 1.0
        inv_w_range = 1.0 / w_range

        best_clause: Clause | None = None
        best_score = -math.inf

        ml_weight = self.ml_config.ml_weight
        use_complexity_norm = self.ml_config.complexity_normalization
        one_minus_ml = 1.0 - ml_weight

        # Repetition bias integration: apply penalty to ML scoring too
        rep_tracker = self.repetition_tracker
        rep_penalty = self.repetition_penalty if rep_tracker is not None else 0.0

        # Pre-compute progressive blending constant
        weight_bias_strength = one_minus_ml
        neutral_contrib = (1.0 - weight_bias_strength) * 0.5

        for i, clause in enumerate(clauses):
            emb = embeddings[i]

            # Progressive traditional score: blend weight preference with neutrality
            base_weight_score = 1.0 - (clause.weight - min_w) * inv_w_range
            trad_score = weight_bias_strength * base_weight_score + neutral_contrib

            if emb is None:
                # No embedding available — use adaptive traditional score
                self.ml_stats.record_embedding_miss()
                score = trad_score
            else:
                # ML score with weight exploration that increases with ml_weight
                raw_ml_score = self._compute_ml_score(emb, clause.weight, ml_weight)

                # Scale [0,1] ML scores to roughly match traditional weight-based
                # score range (~0-10)
                ml_score = raw_ml_score * 10.0

                # Divide by sqrt(complexity) to avoid penalizing complex-but-promising
                # clauses too heavily; superlinear depth penalty in complexity already
                # handles pathological cases
                if use_complexity_norm:
                    complexity = _clause_complexity(clause)
                    ml_score = ml_score / math.sqrt(complexity)

                # Simple linear blending
                score = one_minus_ml * trad_score + ml_weight * ml_score

            # Apply repetition penalty: reduce score for repetitious clauses
            if rep_tracker is not None and rep_penalty > 0:
                rep_score = rep_tracker.repetition_score(clause)
                score *= (1.0 - rep_penalty * rep_score)

            if score > best_score:
                best_score = score
                best_clause = clause

        return best_clause, best_score

    def _compute_ml_score(self, embedding: list[float], clause_weight: float = 0.0, ml_weight: float = 0.0) -> float:
        """Compute ML component score from embedding.

        Blends diversity (distance from recent givens) with proof-potential
        and progressive weight exploration as ML influence increases.
        """
        div_w = self.ml_config.diversity_weight
        pp_w = self.ml_config.proof_potential_weight

        # Add progressive weight exploration component
        # As ml_weight increases, add strong preference for heavier clauses
        # This allows ML to override traditional weight bias progressively
        weight_exploration = ml_weight * 1.0  # More aggressive scale factor

        total_w = div_w + pp_w + weight_exploration
        if total_w == 0:
            return 0.0

        diversity = self._diversity_score(embedding)
        proof_potential = self._proof_potential_score(embedding)

        # Weight exploration score: favor heavier clauses progressively
        # Higher weight → higher exploration score when ML influence is high
        if clause_weight > 0:
            # Use log scale but make it more impactful
            weight_exploration_score = math.log(1 + clause_weight) / 3.0  # More aggressive scaling
        else:
            weight_exploration_score = 0.0

        return (div_w * diversity + pp_w * proof_potential + weight_exploration * weight_exploration_score) / total_w

    def _diversity_score(self, embedding: list[float]) -> float:
        """Score based on cosine distance from recent given clause embeddings.

        Returns a value in [0, 1] where 1 means maximally diverse
        (dissimilar to all recent givens). If no recent embeddings exist,
        returns 0.5 (neutral).
        """
        if not self._recent_embeddings:
            return 0.5

        # Average cosine distance to recent embeddings
        total_dist = 0.0
        count = 0
        for recent in self._recent_embeddings:
            sim = _cosine_similarity(embedding, recent)
            total_dist += 1.0 - sim  # distance = 1 - similarity
            count += 1

        avg_dist = total_dist / count if count > 0 else 0.5
        # Clamp to [0, 1]
        return max(0.0, min(1.0, avg_dist))

    def _proof_potential_score(self, embedding: list[float]) -> float:
        """Estimate proof potential from embedding structure.

        FIXED: Now aligns with training objective where productive clauses
        have SMALLER embedding norms (trained with 0.1× loss coefficient).

        Rewards smaller, more focused embeddings that the model learned
        to associate with productive clauses during contrastive training.

        Returns a value in [0, 1].
        """
        # Compute squared L2 norm (avoid sqrt, use directly in exp)
        norm_sq = sum(x * x for x in embedding)
        norm = math.sqrt(norm_sq)

        # FIXED: Reward SMALLER norms (aligning with training)
        # Use inverse sigmoid: smaller norm → higher score
        return 2.0 / (1.0 + math.exp(norm)) - 1.0

    def _record_embedding(self, clause: Clause) -> None:
        """Record the embedding of a selected given clause for diversity."""
        if self.embedding_provider is None:
            return
        try:
            emb = self.embedding_provider.get_embedding(clause)
            if emb is not None:
                self._recent_embeddings.append(emb)
        except Exception:
            pass  # Non-critical — diversity tracking is best-effort


# ── Complexity normalization ──────────────────────────────────────────────────


def _clause_complexity(clause: Clause) -> float:
    """Compute structural complexity of a clause for ML score normalization.

    Combines literal count, maximum term depth, and variable diversity
    into a single complexity measure. Higher complexity → higher divisor
    → lower normalized ML score, preventing pathological clause growth.

    Returns a value >= 1.0 (minimum complexity for a unit ground clause).
    """
    literal_count = len(clause.literals)
    if literal_count == 0:
        return 1.0

    # Single traversal: collect max depth and variable set together
    max_depth = 0
    var_count = 0
    _seen_vars: set[int] = set()

    for lit in clause.literals:
        # Walk term tree iteratively to get depth + variables
        stack: list[tuple[object, int]] = [(lit.atom, 0)]
        while stack:
            t, d = stack.pop()
            if d > max_depth:
                max_depth = d
            if t.is_variable:
                vn = t.varnum
                if vn not in _seen_vars:
                    _seen_vars.add(vn)
                    var_count += 1
            else:
                for a in t.args:
                    stack.append((a, d + 1))

    # Depth penalty: superlinear to strongly penalize deeply nested terms
    depth_penalty = max_depth ** 1.5

    # Variable diversity bonus: more unique variables → lower complexity
    # (diverse variables suggest a general, useful clause)
    var_bonus = 1.0 / (var_count + 1)

    complexity = literal_count + depth_penalty + var_bonus
    return max(complexity, 1.0)


# ── Utility functions ────────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns value in [-1, 1]. Returns 0.0 for zero vectors.
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    denom = norm_a * norm_b  # avoid two sqrt calls
    if denom < 1e-24:
        return 0.0
    return dot / math.sqrt(denom)
