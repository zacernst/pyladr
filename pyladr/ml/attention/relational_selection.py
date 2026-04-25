"""Relational clause selection with cross-clause attention.

Extends EmbeddingEnhancedSelection with cross-clause attention scoring that
models inter-clause relationships during given-clause selection. The relational
score is blended with the existing ML score, preserving full backward compatibility.

When cross-clause attention is disabled (default), behavior is identical to
EmbeddingEnhancedSelection. When enabled, the selection process becomes:

1. Collect embeddings for all SOS clauses (via existing EmbeddingProvider)
2. Run cross-clause attention to compute relational scores
3. Blend relational scores with existing ML + traditional scores
4. Select the highest-scoring clause

Integration with existing systems:
- Reuses EmbeddingProvider protocol (no new provider needed)
- Reuses ClauseList/PrioritySOS iteration
- Preserves age-based fairness (attention only on weight-based steps)
- Falls back cleanly on any attention error

Thread Safety:
- CrossClauseAttentionScorer is stateless under torch.no_grad()
- _recent_embeddings deque is per-instance (no sharing)
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from pyladr.core.clause import Clause
from pyladr.protocols import EmbeddingProvider
from pyladr.search.ml_selection import (
    EmbeddingEnhancedSelection,
    MLSelectionConfig,
    MLSelectionStats,
    _clause_complexity,
    _cosine_similarity,
)
from pyladr.search.state import ClauseList

logger = logging.getLogger(__name__)

# Guard ML imports
try:
    import torch
    from pyladr.ml.attention.cross_clause import (
        CrossClauseAttentionConfig,
        CrossClauseAttentionScorer,
    )
    _ATTENTION_AVAILABLE = True
except ImportError:
    _ATTENTION_AVAILABLE = False


@dataclass(frozen=True, slots=True)
class RelationalSelectionConfig:
    """Configuration for relational clause selection.

    Extends MLSelectionConfig with cross-clause attention parameters.

    Attributes:
        attention_weight: Blending weight for relational score vs ML score.
            0.0 = pure ML scoring (no attention), 1.0 = pure relational.
            Default 0.3 — mild relational influence.
        attention_config: Cross-clause attention configuration.
        min_sos_for_attention: Minimum SOS size before attention activates.
            Below this, relational scoring adds little value.
        attention_interval: Apply attention every N weight-based selections.
            Attention is O(N^2) in SOS size; this amortizes the cost.
        fallback_on_attention_error: Fall back to non-relational ML scoring
            if attention fails.
    """

    attention_weight: float = 0.3
    attention_config: object = None  # CrossClauseAttentionConfig when available
    min_sos_for_attention: int = 20
    attention_interval: int = 1
    fallback_on_attention_error: bool = True


@dataclass(slots=True)
class RelationalSelectionStats:
    """Tracks cross-clause attention effectiveness."""

    attention_selections: int = 0
    attention_fallbacks: int = 0
    avg_attention_score: float = 0.0
    _attention_score_sum: float = 0.0

    def record_attention(self, score: float) -> None:
        self.attention_selections += 1
        self._attention_score_sum += score
        self.avg_attention_score = (
            self._attention_score_sum / self.attention_selections
        )

    def record_fallback(self) -> None:
        self.attention_fallbacks += 1

    def report(self) -> str:
        total = self.attention_selections + self.attention_fallbacks
        if total == 0:
            return "relational_selection: no attention selections"
        att_pct = 100.0 * self.attention_selections / total
        return (
            f"relational_selection: {self.attention_selections}/{total} "
            f"attention ({att_pct:.1f}%), "
            f"fallbacks={self.attention_fallbacks}, "
            f"avg_score={self.avg_attention_score:.4f}"
        )


@dataclass(slots=True)
class RelationalEnhancedSelection(EmbeddingEnhancedSelection):
    """Clause selection with cross-clause attention scoring.

    Extends EmbeddingEnhancedSelection by adding a relational scoring step
    that models inter-clause dependencies via multi-head attention. The
    relational score captures complementarity, diversity, and proof-path
    potential across the entire SOS.

    When attention is disabled or unavailable, behavior is identical to
    EmbeddingEnhancedSelection (full backward compatibility).

    Usage:
        from pyladr.ml.attention.cross_clause import CrossClauseAttentionConfig
        from pyladr.ml.attention.relational_selection import (
            RelationalEnhancedSelection,
            RelationalSelectionConfig,
        )

        rel_config = RelationalSelectionConfig(
            attention_weight=0.3,
            attention_config=CrossClauseAttentionConfig(
                enabled=True,
                embedding_dim=512,
                num_heads=8,
            ),
        )
        selection = RelationalEnhancedSelection(
            embedding_provider=provider,
            ml_config=MLSelectionConfig(enabled=True),
            relational_config=rel_config,
        )
    """

    relational_config: RelationalSelectionConfig = field(
        default_factory=RelationalSelectionConfig,
    )
    relational_stats: RelationalSelectionStats = field(
        default_factory=RelationalSelectionStats,
    )
    _attention_scorer: object = field(default=None, repr=False)
    _attention_step: int = 0

    def __post_init__(self) -> None:
        # Explicit parent call — super() doesn't work with slots=True
        # dataclass inheritance in Python 3.10+
        EmbeddingEnhancedSelection.__post_init__(self)
        self._init_attention_scorer()

    def _init_attention_scorer(self) -> None:
        """Lazily initialize the attention scorer if configured and available."""
        if not _ATTENTION_AVAILABLE:
            return

        cfg = self.relational_config.attention_config
        if cfg is None or not getattr(cfg, "enabled", False):
            return

        try:
            self._attention_scorer = CrossClauseAttentionScorer(cfg)
            self._attention_scorer.eval()
            logger.info(
                "Cross-clause attention initialized: heads=%d, dim=%d",
                cfg.num_heads,
                cfg.embedding_dim,
            )
        except Exception:
            logger.warning(
                "Failed to initialize cross-clause attention scorer",
                exc_info=True,
            )
            self._attention_scorer = None

    def _ml_select_inner(
        self, sos: ClauseList
    ) -> tuple[Clause | None, float]:
        """Override ML selection to blend relational attention scores.

        When cross-clause attention is active, this method:
        1. Collects embeddings for all SOS clauses
        2. Runs cross-clause attention to get relational scores
        3. Blends relational + ML + traditional scores
        4. Returns the best clause

        Falls back to parent implementation if attention is not applicable.
        """
        # Check if we should use attention on this step
        if not self._should_use_attention(sos):
            return super()._ml_select_inner(sos)

        self._attention_step += 1

        provider = self.embedding_provider
        assert provider is not None

        clauses = list(sos)
        if not clauses:
            return None, 0.0

        # Collect embeddings
        embeddings = provider.get_embeddings_batch(clauses)

        # Separate clauses with and without embeddings
        valid_indices = []
        valid_embeddings = []
        for i, emb in enumerate(embeddings):
            if emb is not None:
                valid_indices.append(i)
                valid_embeddings.append(emb)

        # Need enough clauses with embeddings for attention to be meaningful
        if len(valid_indices) < self.relational_config.min_sos_for_attention:
            return super()._ml_select_inner(sos)

        # Compute relational scores via cross-clause attention
        try:
            relational_scores = self._compute_relational_scores(
                clauses, valid_indices, valid_embeddings
            )
        except Exception:
            if self.relational_config.fallback_on_attention_error:
                logger.debug(
                    "Cross-clause attention failed, falling back", exc_info=True
                )
                self.relational_stats.record_fallback()
                return super()._ml_select_inner(sos)
            raise

        # Blend relational scores with ML + traditional scores
        return self._blend_and_select(
            clauses, embeddings, valid_indices, relational_scores
        )

    def _should_use_attention(self, sos: ClauseList) -> bool:
        """Check if cross-clause attention should be applied."""
        if self._attention_scorer is None:
            return False
        if sos.length < self.relational_config.min_sos_for_attention:
            return False
        interval = self.relational_config.attention_interval
        if interval > 1 and (self._attention_step % interval) != 0:
            return False
        return True

    def _compute_relational_scores(
        self,
        clauses: list[Clause],
        valid_indices: list[int],
        valid_embeddings: list[list[float]],
    ) -> dict[int, float]:
        """Compute relational scores using cross-clause attention.

        Returns a dict mapping clause list index → relational score.
        """
        import torch

        scorer = self._attention_scorer
        assert scorer is not None

        # Build embedding tensor
        emb_tensor = torch.tensor(valid_embeddings, dtype=torch.float32)

        # Build clause ID tensor for relative position bias
        clause_ids = torch.tensor(
            [clauses[i].id for i in valid_indices], dtype=torch.long
        )

        # Score via attention
        scores = scorer.score_clauses(emb_tensor, clause_ids=clause_ids)

        # Map back to clause indices
        return {valid_indices[i]: scores[i] for i in range(len(valid_indices))}

    def _blend_and_select(
        self,
        clauses: list[Clause],
        embeddings: list[list[float] | None],
        valid_indices: list[int],
        relational_scores: dict[int, float],
    ) -> tuple[Clause | None, float]:
        """Blend relational, ML, and traditional scores to select best clause.

        Three-way blending:
          final = (1 - ml_w) * trad + ml_w * [(1 - att_w) * ml + att_w * relational]

        where ml_w is the ML weight and att_w is the attention weight.
        """
        # Weight statistics for traditional scoring
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

        ml_weight = self.ml_config.ml_weight
        att_weight = self.relational_config.attention_weight
        one_minus_ml = 1.0 - ml_weight
        one_minus_att = 1.0 - att_weight
        use_complexity_norm = self.ml_config.complexity_normalization

        # Normalize relational scores to [0, 1]
        if relational_scores:
            rel_vals = list(relational_scores.values())
            rel_min = min(rel_vals)
            rel_max = max(rel_vals)
            rel_range = rel_max - rel_min if rel_max > rel_min else 1.0
            norm_relational = {
                k: (v - rel_min) / rel_range
                for k, v in relational_scores.items()
            }
        else:
            norm_relational = {}

        # Repetition tracking
        rep_tracker = self.repetition_tracker
        rep_penalty = self.repetition_penalty if rep_tracker is not None else 0.0

        # Progressive blending constant
        weight_bias_strength = one_minus_ml
        neutral_contrib = (1.0 - weight_bias_strength) * 0.5

        best_clause: Clause | None = None
        best_score = -math.inf

        for i, clause in enumerate(clauses):
            emb = embeddings[i]

            # Traditional score
            base_weight_score = 1.0 - (clause.weight - min_w) * inv_w_range
            trad_score = weight_bias_strength * base_weight_score + neutral_contrib

            if emb is None:
                self.ml_stats.record_embedding_miss()
                score = trad_score
            else:
                # ML score (same as parent)
                raw_ml_score = self._compute_ml_score(
                    emb, clause.weight, ml_weight
                )
                ml_score = raw_ml_score * 10.0

                if use_complexity_norm:
                    complexity = _clause_complexity(clause)
                    ml_score = ml_score / math.sqrt(complexity)

                # Blend ML with relational
                if i in norm_relational:
                    rel_score = norm_relational[i] * 10.0  # scale to match ML
                    blended_ml = one_minus_att * ml_score + att_weight * rel_score
                    self.relational_stats.record_attention(rel_score)
                else:
                    blended_ml = ml_score

                # Final blend: traditional + blended ML
                score = one_minus_ml * trad_score + ml_weight * blended_ml

            # Repetition penalty
            if rep_tracker is not None and rep_penalty > 0:
                rep_score = rep_tracker.repetition_score(clause)
                score *= 1.0 - rep_penalty * rep_score

            if score > best_score:
                best_score = score
                best_clause = clause

        return best_clause, best_score
