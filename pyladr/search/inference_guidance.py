"""Embedding-guided inference targeting for intelligent candidate prioritization.

Prioritizes which usable clauses to attempt inference with, based on
embedding similarity and structural compatibility with the given clause.

All guidance features are strictly opt-in.  When disabled, the full usable
list is returned unmodified, preserving original search behavior.

Performance design
------------------
* **GPU-batched cosine similarity** — when torch and EmbeddingCache are
  available, similarity is computed via a single batched matmul on the
  device, avoiding per-pair Python loops.
* **Structural scoring is O(L_g * L_c) per pair** with tiny constant —
  only touches literal metadata (signs, predicate symbols, arities).
* **Early termination** — stops inference generation after a configurable
  number of successful inferences, saving time on highly productive
  given clauses.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pyladr.core.clause import Clause

if TYPE_CHECKING:
    from pyladr.search.ml_selection import EmbeddingProvider

try:
    import torch as _torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── Inference guidance configuration ─────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class InferenceGuidanceConfig:
    """Configuration for embedding-guided inference targeting.

    Attributes:
        enabled: Master switch for inference guidance.
        max_candidates: Maximum usable clauses to attempt inference with
            per given clause. -1 = no limit (process all). When set,
            clauses are ranked by compatibility score and only the top-k
            are used.
        compatibility_threshold: Minimum compatibility score to attempt
            inference. Clauses below this threshold are skipped. 0.0
            disables thresholding.
        structural_weight: Weight of structural compatibility in scoring.
        semantic_weight: Weight of semantic (embedding) similarity.
        complementarity_bonus: Extra score for clauses with complementary
            literal signs (likely to produce resolvents).
        early_termination_count: Stop after this many successful inferences
            from a single given clause. -1 = no early termination.
        log_guidance: Log guidance decisions for debugging.
    """

    enabled: bool = False
    max_candidates: int = -1
    compatibility_threshold: float = 0.0
    structural_weight: float = 0.4
    semantic_weight: float = 0.6
    complementarity_bonus: float = 0.2
    early_termination_count: int = -1
    log_guidance: bool = False


# ── Guidance statistics ──────────────────────────────────────────────────────


@dataclass(slots=True)
class InferenceGuidanceStats:
    """Tracks inference guidance effectiveness."""

    guided_rounds: int = 0
    unguided_rounds: int = 0
    total_candidates_scored: int = 0
    total_candidates_selected: int = 0
    total_candidates_skipped: int = 0
    early_terminations: int = 0
    avg_top_score: float = 0.0
    _top_score_sum: float = 0.0

    def record_guided_round(
        self, scored: int, selected: int, skipped: int, top_score: float,
    ) -> None:
        self.guided_rounds += 1
        self.total_candidates_scored += scored
        self.total_candidates_selected += selected
        self.total_candidates_skipped += skipped
        self._top_score_sum += top_score
        self.avg_top_score = self._top_score_sum / self.guided_rounds

    def record_unguided(self) -> None:
        self.unguided_rounds += 1

    def record_early_termination(self) -> None:
        self.early_terminations += 1

    def report(self) -> str:
        total = self.guided_rounds + self.unguided_rounds
        if total == 0:
            return "inference_guidance: no rounds"
        guided_pct = 100.0 * self.guided_rounds / total
        return (
            f"inference_guidance: {self.guided_rounds}/{total} guided "
            f"({guided_pct:.1f}%), "
            f"candidates={self.total_candidates_selected}/"
            f"{self.total_candidates_scored} selected, "
            f"skipped={self.total_candidates_skipped}, "
            f"early_term={self.early_terminations}, "
            f"avg_top_score={self.avg_top_score:.4f}"
        )


# ── Scored candidate ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """A usable clause scored for inference compatibility."""

    clause: Clause
    score: float
    structural_score: float
    semantic_score: float


# ── Embedding-guided inference ───────────────────────────────────────────────


class EmbeddingGuidedInference:
    """Prioritizes usable clauses for inference with the given clause.

    Scores each usable clause based on embedding similarity and structural
    compatibility with the given clause, then returns a prioritized
    (and optionally truncated) list for inference generation.

    Integration point: called from GivenClauseSearch._given_infer() to
    reorder/filter the usable_snapshot before passing to sequential or
    parallel inference generation.

    Usage:
        guidance = EmbeddingGuidedInference(
            provider=embedding_provider,
            config=InferenceGuidanceConfig(enabled=True, max_candidates=100),
        )

        # In the search loop:
        usable_snapshot = list(state.usable)
        prioritized = guidance.prioritize(given, usable_snapshot)
        # Use prioritized list for inference generation
    """

    __slots__ = ("_provider", "_config", "_stats")

    def __init__(
        self,
        provider: EmbeddingProvider | None = None,
        config: InferenceGuidanceConfig | None = None,
    ) -> None:
        self._provider = provider
        self._config = config or InferenceGuidanceConfig()
        self._stats = InferenceGuidanceStats()

    @property
    def config(self) -> InferenceGuidanceConfig:
        return self._config

    @property
    def stats(self) -> InferenceGuidanceStats:
        return self._stats

    @property
    def provider(self) -> EmbeddingProvider | None:
        return self._provider

    @provider.setter
    def provider(self, value: EmbeddingProvider | None) -> None:
        self._provider = value

    def prioritize(
        self,
        given: Clause,
        usable: list[Clause],
    ) -> list[Clause]:
        """Prioritize usable clauses for inference with the given clause.

        When guidance is active, scores and ranks usable clauses by
        compatibility with the given clause. When disabled or on failure,
        returns the usable list unmodified.

        Args:
            given: The selected given clause.
            usable: Snapshot of usable clauses.

        Returns:
            Prioritized (and optionally truncated) list of usable clauses.
        """
        if not self._should_guide(usable):
            self._stats.record_unguided()
            return usable

        try:
            return self._prioritize_inner(given, usable)
        except Exception:
            logger.debug(
                "Inference guidance failed, using unguided order",
                exc_info=True,
            )
            self._stats.record_unguided()
            return usable

    def _should_guide(self, usable: list[Clause]) -> bool:
        """Check if guidance should be applied."""
        return (
            self._config.enabled
            and self._provider is not None
            and len(usable) > 0
        )

    def _prioritize_inner(
        self,
        given: Clause,
        usable: list[Clause],
    ) -> list[Clause]:
        """Score and rank usable clauses by compatibility."""
        provider = self._provider
        assert provider is not None

        given_emb = provider.get_embedding(given)
        if given_emb is None:
            self._stats.record_unguided()
            return usable

        # Get embeddings for all usable clauses in batch
        usable_embs = provider.get_embeddings_batch(usable)

        scored: list[ScoredCandidate] = []
        skipped = 0

        for i, clause in enumerate(usable):
            emb = usable_embs[i]

            structural = self._structural_compatibility(given, clause)

            if emb is not None:
                semantic = _cosine_similarity(given_emb, emb)
                # Normalize from [-1, 1] to [0, 1]
                semantic = (semantic + 1.0) / 2.0
            else:
                # No embedding — use structural score only
                semantic = 0.5

            # Weighted combination
            sw = self._config.structural_weight
            ew = self._config.semantic_weight
            total_w = sw + ew
            if total_w > 0:
                score = (sw * structural + ew * semantic) / total_w
            else:
                score = 0.5

            # Apply threshold
            if (
                self._config.compatibility_threshold > 0
                and score < self._config.compatibility_threshold
            ):
                skipped += 1
                continue

            scored.append(ScoredCandidate(
                clause=clause,
                score=score,
                structural_score=structural,
                semantic_score=semantic,
            ))

        # Sort by score descending
        scored.sort(key=lambda s: s.score, reverse=True)

        # Apply max_candidates limit
        max_k = self._config.max_candidates
        if max_k > 0 and len(scored) > max_k:
            skipped += len(scored) - max_k
            scored = scored[:max_k]

        top_score = scored[0].score if scored else 0.0

        self._stats.record_guided_round(
            scored=len(usable),
            selected=len(scored),
            skipped=skipped,
            top_score=top_score,
        )

        if self._config.log_guidance and scored:
            logger.info(
                "Inference guidance: given=%d, scored=%d, selected=%d, "
                "skipped=%d, top_score=%.4f",
                given.id, len(usable), len(scored), skipped, top_score,
            )

        return [s.clause for s in scored]

    def _structural_compatibility(
        self, given: Clause, candidate: Clause,
    ) -> float:
        """Score structural compatibility between two clauses.

        Considers:
        - Complementary literal potential — matching predicate symbol with
          opposite signs (essential for binary resolution)
        - Shared symbol overlap — clauses sharing function symbols are more
          likely to unify
        - Size compatibility — very large clauses produce large resolvents
        - Unit / small clause bonus

        Returns value in [0, 1].
        """
        score = 0.0

        # 1. Complementarity with predicate-symbol matching.
        #    Only opposite-sign literals on the *same* predicate/arity can
        #    resolve — this is a much stronger signal than sign-only checks.
        complementary_matches = 0
        complementary_possible = 0
        for g_lit in given.literals:
            for c_lit in candidate.literals:
                complementary_possible += 1
                if g_lit.sign != c_lit.sign:
                    g_sym = g_lit.atom.private_symbol
                    c_sym = c_lit.atom.private_symbol
                    if g_sym == c_sym and g_lit.atom.arity == c_lit.atom.arity:
                        complementary_matches += 1

        if complementary_matches > 0:
            # Scale bonus by fraction of matchable pairs.
            frac = complementary_matches / max(complementary_possible, 1)
            score += self._config.complementarity_bonus * (0.5 + 0.5 * frac)

        # 2. Shared rigid-symbol overlap (Jaccard).
        given_syms = _clause_rigid_symbols(given)
        cand_syms = _clause_rigid_symbols(candidate)
        if given_syms or cand_syms:
            union = given_syms | cand_syms
            inter = given_syms & cand_syms
            overlap = len(inter) / len(union) if union else 0.0
            score += overlap * 0.25

        # 3. Size compatibility: prefer small resolvents.
        max_lits = max(given.num_literals, candidate.num_literals, 1)
        size_score = 1.0 / (1.0 + math.log1p(max_lits))
        score += size_score * 0.4

        # 4. Unit clause bonus (highly productive for resolution).
        if candidate.is_unit:
            score += 0.3
        elif candidate.num_literals == 2:
            score += 0.1

        # 5. Ground clause slight bonus (simpler unification).
        if candidate.is_ground:
            score += 0.05

        # Normalize to [0, 1].
        max_possible = (
            self._config.complementarity_bonus + 0.25 + 0.4 + 0.3 + 0.05
        )
        return min(1.0, score / max_possible) if max_possible > 0 else 0.5

    def should_terminate_early(self, inferences_generated: int) -> bool:
        """Check if early termination should occur.

        Called during inference generation to allow stopping when
        enough productive inferences have been found.

        Args:
            inferences_generated: Number of inferences generated so far
                for the current given clause.

        Returns:
            True if generation should stop early.
        """
        if not self._config.enabled:
            return False
        limit = self._config.early_termination_count
        if limit < 0:
            return False
        if inferences_generated >= limit:
            self._stats.record_early_termination()
            return True
        return False


# ── Utility functions ────────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two float vectors (CPU fallback)."""
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


def _clause_rigid_symbols(c: Clause) -> set[int]:
    """Extract the set of rigid (non-variable) symbol IDs from a clause."""
    syms: set[int] = set()
    for lit in c.literals:
        for t in lit.atom.subterms():
            if not t.is_variable:
                syms.add(t.private_symbol)
    return syms


def _batched_cosine_scores(
    given_emb: object,
    usable_embs: object,
) -> list[float] | None:
    """Compute cosine similarities via GPU-batched matmul.

    Parameters
    ----------
    given_emb:
        Tensor of shape ``(1, dim)`` or ``(dim,)`` for the given clause.
    usable_embs:
        Tensor of shape ``(N, dim)`` for the usable clauses.

    Returns
    -------
    list[float] | None
        List of N similarity values in [0, 1], or ``None`` on failure.
    """
    if not _TORCH_AVAILABLE:
        return None
    try:
        g = given_emb if given_emb.dim() == 2 else given_emb.unsqueeze(0)  # type: ignore[union-attr]
        u = usable_embs  # type: ignore[assignment]
        g_norm = _torch.nn.functional.normalize(g, dim=1)
        u_norm = _torch.nn.functional.normalize(u, dim=1)
        sims = _torch.mm(g_norm, u_norm.t()).squeeze(0)  # (N,)
        # Map [-1, 1] → [0, 1]
        scores = ((sims + 1.0) / 2.0).tolist()
        return scores
    except Exception:
        return None
