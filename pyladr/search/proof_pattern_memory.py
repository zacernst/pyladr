"""Proof-guided clause selection via proof pattern memory.

Learns from successful proofs by storing embeddings of clauses that
contributed to a proof. Future clause selection is guided toward
candidates similar to successful proof patterns (exploitation) while
maintaining diversity via exploration.

Algorithm:
  1. Maintain a memory of embeddings from clauses in successful proofs.
  2. For each candidate clause, compute cosine similarity to the most
     relevant proof pattern (max-similarity).
  3. Blend exploitation (similarity to proof patterns) with exploration
     (diversity from recent selections) using a configurable ratio.
  4. Apply exponential decay to older patterns so the memory adapts
     to evolving search dynamics.

Thread-safety: ProofPatternMemory uses no internal locks but is safe
for single-threaded search loops (the standard use case). External
synchronization is required for concurrent access.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyladr.core.clause import Clause


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ProofGuidedConfig:
    """Configuration for proof-guided selection.

    Attributes:
        enabled: Master switch for proof-guided selection.
        exploitation_ratio: Blend weight for exploitation vs exploration.
            1.0 = pure exploitation (only follow proof patterns),
            0.0 = pure exploration (ignore proof patterns).
        max_patterns: Maximum number of proof pattern embeddings retained.
            Oldest patterns are evicted when this limit is reached.
        decay_rate: Exponential decay factor applied per proof event.
            Each time a new proof is recorded, existing pattern weights
            are multiplied by this factor. Range (0, 1].
            1.0 = no decay, 0.9 = 10% decay per proof event.
        min_similarity_threshold: Minimum cosine similarity for a pattern
            to be considered relevant. Below this, the pattern contributes
            zero exploitation signal.
        warmup_proofs: Number of proofs required before proof-guided
            scoring activates. Until then, returns neutral scores.
    """

    enabled: bool = True
    exploitation_ratio: float = 0.7
    max_patterns: int = 500
    decay_rate: float = 0.95
    min_similarity_threshold: float = 0.1
    warmup_proofs: int = 1


# ── Proof pattern memory ─────────────────────────────────────────────────────


@dataclass(slots=True)
class _PatternEntry:
    """A single proof pattern embedding with its associated weight."""

    embedding: list[float]
    weight: float


@dataclass(slots=True)
class ProofPatternMemory:
    """Stores and manages embeddings from successful proof clauses.

    Patterns are stored with weights that decay over time so that
    recent proof patterns have stronger influence than older ones.

    Usage:
        memory = ProofPatternMemory(config=ProofGuidedConfig())

        # After a proof is found, record contributing clause embeddings
        memory.record_proof(proof_embeddings)

        # During selection, score a candidate clause embedding
        exploitation_score = memory.exploitation_score(candidate_embedding)
    """

    config: ProofGuidedConfig = field(default_factory=ProofGuidedConfig)
    _patterns: deque[_PatternEntry] = field(init=False)
    _proof_count: int = field(default=0, init=False)
    _centroid: list[float] | None = field(default=None, init=False)
    _centroid_dirty: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        self._patterns = deque(maxlen=self.config.max_patterns)

    @property
    def proof_count(self) -> int:
        """Number of proofs recorded so far."""
        return self._proof_count

    @property
    def pattern_count(self) -> int:
        """Number of pattern embeddings currently stored."""
        return len(self._patterns)

    @property
    def is_warmed_up(self) -> bool:
        """Whether enough proofs have been recorded to activate scoring."""
        return self._proof_count >= self.config.warmup_proofs

    def record_proof(self, embeddings: list[list[float]]) -> None:
        """Record embeddings from clauses that contributed to a proof.

        Applies decay to existing patterns before adding new ones.

        Args:
            embeddings: List of embedding vectors from proof clauses.
                Each vector should be L2-normalized for best results.
        """
        if not embeddings:
            return

        # Decay existing pattern weights
        decay = self.config.decay_rate
        if decay < 1.0:
            for entry in self._patterns:
                entry.weight *= decay

        # Add new patterns with weight 1.0
        for emb in embeddings:
            self._patterns.append(_PatternEntry(embedding=emb, weight=1.0))

        self._proof_count += 1
        self._centroid_dirty = True

    def exploitation_score(self, embedding: list[float]) -> float:
        """Compute exploitation score: similarity to known proof patterns.

        Uses weighted max-similarity: finds the most similar proof pattern
        and returns the similarity scaled by the pattern's weight.

        Returns a value in [0, 1] where 1 means perfect match to a
        high-weight proof pattern. Returns 0.5 (neutral) if not warmed up.

        Args:
            embedding: The candidate clause embedding to score.
        """
        if not self.is_warmed_up or not self._patterns:
            return 0.5  # Neutral — no signal yet

        threshold = self.config.min_similarity_threshold
        best_score = 0.0

        for entry in self._patterns:
            sim = _cosine_similarity(embedding, entry.embedding)
            if sim < threshold:
                continue
            # Weight-scaled similarity: recent patterns (weight≈1) dominate
            score = sim * entry.weight
            if score > best_score:
                best_score = score

        # Clamp to [0, 1]
        return min(best_score, 1.0)

    def centroid_score(self, embedding: list[float]) -> float:
        """Compute similarity to the weighted centroid of all patterns.

        This is cheaper than exploitation_score (O(d) vs O(n*d)) and
        provides a smooth, averaged signal. Useful as a secondary metric
        or for large pattern memories.

        Returns a value in [0, 1]. Returns 0.5 if not warmed up.
        """
        if not self.is_warmed_up or not self._patterns:
            return 0.5

        centroid = self._get_centroid()
        if centroid is None:
            return 0.5

        sim = _cosine_similarity(embedding, centroid)
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))  # Map [-1,1] → [0,1]

    def _get_centroid(self) -> list[float] | None:
        """Compute or return cached weighted centroid of patterns."""
        if not self._centroid_dirty and self._centroid is not None:
            return self._centroid

        if not self._patterns:
            self._centroid = None
            self._centroid_dirty = False
            return None

        dim = len(self._patterns[0].embedding)
        centroid = [0.0] * dim
        total_weight = 0.0

        for entry in self._patterns:
            w = entry.weight
            total_weight += w
            for j in range(dim):
                centroid[j] += entry.embedding[j] * w

        if total_weight < 1e-12:
            self._centroid = None
        else:
            inv_w = 1.0 / total_weight
            self._centroid = [c * inv_w for c in centroid]

        self._centroid_dirty = False
        return self._centroid

    def clear(self) -> None:
        """Clear all stored patterns and reset state."""
        self._patterns.clear()
        self._proof_count = 0
        self._centroid = None
        self._centroid_dirty = True


# ── Proof-guided scoring function ────────────────────────────────────────────


def proof_guided_score(
    embedding: list[float],
    memory: ProofPatternMemory,
    diversity_score: float,
    config: ProofGuidedConfig,
) -> float:
    """Compute blended proof-guided selection score.

    Blends exploitation (similarity to successful proof patterns) with
    exploration (diversity from recent selections) according to the
    configured exploitation ratio.

    The formula:
        score = α × exploitation + (1 - α) × exploration

    where α = config.exploitation_ratio.

    Before warmup is complete, returns the diversity score unmodified.

    Args:
        embedding: Candidate clause embedding vector.
        memory: Proof pattern memory with recorded proof embeddings.
        diversity_score: Pre-computed diversity score in [0, 1] from the
            existing diversity scoring mechanism.
        config: Proof-guided configuration.

    Returns:
        Blended score in [0, 1].
    """
    if not config.enabled or not memory.is_warmed_up:
        return diversity_score

    exploitation = memory.exploitation_score(embedding)
    alpha = config.exploitation_ratio
    return alpha * exploitation + (1.0 - alpha) * diversity_score


# ── Utility ──────────────────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns value in [-1, 1]. Returns 0.0 for zero vectors.
    Operates in O(d) where d is the vector dimension.
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    denom = norm_a * norm_b
    if denom < 1e-24:
        return 0.0
    return dot / math.sqrt(denom)
