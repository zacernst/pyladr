"""Contrastive loss functions optimized for real-time online learning.

The offline ``ContrastiveLoss`` in ``contrastive.py`` expects pre-batched
3-D negatives tensors (batch, n_neg, dim) and is designed for the standard
DataLoader training loop.  Online learning during theorem proving has
different constraints:

* **Paired format** — ``ExperienceBuffer.sample_contrastive_batch()``
  yields flat (positive, negative) outcome pairs, not anchor/positive/
  negative triplets with variable-length negative lists.
* **In-batch negatives** — with small online batches, every non-matching
  example in the batch can serve as a negative, dramatically improving
  sample efficiency without extra encoding cost.
* **Stability** — gradient clipping, temperature annealing, and loss
  magnitude tracking are critical when updating a live model.
* **Low allocation** — avoid creating intermediate tensors that would
  pressure the GC during search.

This module provides drop-in loss functions for
``OnlineLearningManager._gradient_step()`` and any future real-time
training loops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OnlineLossConfig:
    """Configuration for online contrastive losses.

    Attributes:
        temperature: InfoNCE softmax temperature.  Lower → sharper.
        temperature_min: Floor for temperature annealing.
        temperature_decay: Multiplicative decay per update step
            (1.0 = no annealing).
        margin: Triplet margin for the margin-based loss variant.
        use_in_batch_negatives: Mine additional negatives from within
            the batch (every other anchor's positive becomes a negative).
        label_smoothing: Smooth the cross-entropy target to prevent
            over-confident predictions (0.0 = hard labels).
        loss_ema_decay: Exponential moving average decay for loss
            tracking (used in ``LossStatistics``).
    """

    temperature: float = 0.07
    temperature_min: float = 0.01
    temperature_decay: float = 1.0
    margin: float = 0.5
    use_in_batch_negatives: bool = True
    label_smoothing: float = 0.0
    loss_ema_decay: float = 0.95


_DEFAULT_CONFIG = OnlineLossConfig()


# ── Loss statistics ────────────────────────────────────────────────────────


@dataclass(slots=True)
class LossStatistics:
    """Running statistics for online loss monitoring.

    Updated after every forward pass.  Useful for detecting divergence
    (``ema_loss`` exploding) or saturation (``mean_positive_sim`` ≈
    ``mean_negative_sim``).
    """

    total_steps: int = 0
    ema_loss: float = 0.0
    last_loss: float = 0.0
    mean_positive_sim: float = 0.0
    mean_negative_sim: float = 0.0
    similarity_gap: float = 0.0
    current_temperature: float = 0.07

    def snapshot(self) -> dict[str, float | int]:
        """Return a plain-dict copy suitable for logging / JSON."""
        return {
            "total_steps": self.total_steps,
            "ema_loss": round(self.ema_loss, 6),
            "last_loss": round(self.last_loss, 6),
            "mean_positive_sim": round(self.mean_positive_sim, 4),
            "mean_negative_sim": round(self.mean_negative_sim, 4),
            "similarity_gap": round(self.similarity_gap, 4),
            "current_temperature": round(self.current_temperature, 6),
        }


# ── Online InfoNCE loss ───────────────────────────────────────────────────


class OnlineInfoNCELoss(nn.Module):
    """InfoNCE loss optimized for the online paired-example format.

    Accepts flat ``(batch, dim)`` tensors for anchors, positives, and
    negatives — the natural output shape from
    ``ExperienceBuffer.sample_contrastive_batch()``.

    When ``use_in_batch_negatives`` is enabled (the default), every
    other example's positive embedding is used as an additional negative,
    turning a batch of *B* pairs into *B* × *(B−1)* negative
    comparisons at zero extra encoding cost.

    Supports optional temperature annealing and label smoothing.
    """

    def __init__(self, config: OnlineLossConfig | None = None) -> None:
        super().__init__()
        self._config = config or _DEFAULT_CONFIG
        self._temperature = self._config.temperature
        self._stats = LossStatistics(current_temperature=self._temperature)

    @property
    def stats(self) -> LossStatistics:
        return self._stats

    @property
    def temperature(self) -> float:
        return self._temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute online InfoNCE loss.

        Args:
            anchor: Anchor embeddings, shape ``(B, D)``.
            positive: Positive embeddings, shape ``(B, D)``.
            negative: Explicit negative embeddings, shape ``(B, D)``.
                One negative per anchor (paired format).
            weights: Optional per-example importance weights ``(B,)``.

        Returns:
            Scalar loss tensor with grad attached.
        """
        # L2 normalize
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        tau = self._temperature

        if self._config.use_in_batch_negatives and anchor.size(0) > 1:
            loss, pos_sim_mean, neg_sim_mean = self._forward_in_batch(
                anchor, positive, negative, tau, weights,
            )
        else:
            loss, pos_sim_mean, neg_sim_mean = self._forward_paired(
                anchor, positive, negative, tau, weights,
            )

        # Update statistics (detached — no grad impact)
        self._update_stats(loss.detach().item(), pos_sim_mean, neg_sim_mean)
        self._anneal_temperature()

        return loss

    # ── Internal forward paths ────────────────────────────────────────

    def _forward_paired(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        tau: float,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, float, float]:
        """Simple paired InfoNCE: 1 positive + 1 explicit negative."""
        pos_sim = torch.sum(anchor * positive, dim=-1) / tau   # (B,)
        neg_sim = torch.sum(anchor * negative, dim=-1) / tau   # (B,)

        logits = torch.stack([pos_sim, neg_sim], dim=-1)        # (B, 2)
        labels = torch.zeros(
            logits.size(0), dtype=torch.long, device=logits.device,
        )

        loss = F.cross_entropy(
            logits, labels,
            reduction="none",
            label_smoothing=self._config.label_smoothing,
        )

        if weights is not None:
            loss = loss * weights
            loss = loss.sum() / weights.sum().clamp(min=1e-8)
        else:
            loss = loss.mean()

        with torch.no_grad():
            ps = (pos_sim * tau).mean().item()
            ns = (neg_sim * tau).mean().item()

        return loss, ps, ns

    def _forward_in_batch(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        tau: float,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, float, float]:
        """InfoNCE with in-batch negatives for better sample efficiency.

        For each anchor *i*, the positive is its own ``positive[i]``.
        Negatives include:
          1. The explicit ``negative[i]``.
          2. Every ``positive[j]`` for *j ≠ i* (in-batch negatives).

        This gives ``B`` negatives per anchor instead of 1.
        """
        B = anchor.size(0)

        # Positive similarity: diagonal of anchor @ positive^T
        pos_sim = torch.sum(anchor * positive, dim=-1) / tau   # (B,)

        # Explicit negative similarity
        explicit_neg_sim = torch.sum(
            anchor * negative, dim=-1,
        ) / tau                                                 # (B,)

        # In-batch negatives: anchor @ positive^T  (full matrix)
        # Shape: (B, B)
        all_pos_sim = torch.mm(anchor, positive.t()) / tau

        # Mask out the diagonal (those are the positives, not negatives)
        mask = torch.eye(B, dtype=torch.bool, device=anchor.device)
        in_batch_neg_sim = all_pos_sim.masked_fill(mask, float("-inf"))

        # Logits: [pos_sim, explicit_neg, in_batch_neg_0, ..., in_batch_neg_{B-2}]
        # Shape: (B, 1 + 1 + B-1) = (B, B+1)
        logits = torch.cat([
            pos_sim.unsqueeze(-1),          # (B, 1) — positive
            explicit_neg_sim.unsqueeze(-1),  # (B, 1) — explicit neg
            in_batch_neg_sim,                # (B, B) — in-batch (diagonal masked)
        ], dim=-1)

        labels = torch.zeros(B, dtype=torch.long, device=anchor.device)

        loss = F.cross_entropy(
            logits, labels,
            reduction="none",
            label_smoothing=self._config.label_smoothing,
        )

        if weights is not None:
            loss = loss * weights
            loss = loss.sum() / weights.sum().clamp(min=1e-8)
        else:
            loss = loss.mean()

        with torch.no_grad():
            ps = (pos_sim * tau).mean().item()
            # Mean over explicit negatives only for cleaner monitoring
            ns = (explicit_neg_sim * tau).mean().item()

        return loss, ps, ns

    # ── Statistics and annealing ──────────────────────────────────────

    def _update_stats(
        self, loss_val: float, pos_sim: float, neg_sim: float,
    ) -> None:
        s = self._stats
        decay = self._config.loss_ema_decay
        if s.total_steps == 0:
            s.ema_loss = loss_val
        else:
            s.ema_loss = decay * s.ema_loss + (1.0 - decay) * loss_val
        s.last_loss = loss_val
        s.mean_positive_sim = pos_sim
        s.mean_negative_sim = neg_sim
        s.similarity_gap = pos_sim - neg_sim
        s.total_steps += 1
        s.current_temperature = self._temperature

    def _anneal_temperature(self) -> None:
        if self._config.temperature_decay < 1.0:
            self._temperature = max(
                self._config.temperature_min,
                self._temperature * self._config.temperature_decay,
            )


# ── Online triplet margin loss ─────────────────────────────────────────────


class OnlineTripletLoss(nn.Module):
    """Margin-based triplet loss for the online paired format.

    Directly enforces ``d(anchor, positive) + margin < d(anchor, negative)``
    using cosine distance.  Useful when the batch is very small (< 4) and
    InfoNCE's softmax denominator is unreliable.
    """

    def __init__(self, config: OnlineLossConfig | None = None) -> None:
        super().__init__()
        self._config = config or _DEFAULT_CONFIG
        self._stats = LossStatistics(current_temperature=self._config.temperature)

    @property
    def stats(self) -> LossStatistics:
        return self._stats

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute triplet margin loss.

        Args:
            anchor: ``(B, D)``
            positive: ``(B, D)``
            negative: ``(B, D)``
            weights: optional ``(B,)``

        Returns:
            Scalar loss.
        """
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        pos_dist = 1.0 - torch.sum(anchor * positive, dim=-1)
        neg_dist = 1.0 - torch.sum(anchor * negative, dim=-1)

        loss = F.relu(pos_dist - neg_dist + self._config.margin)

        if weights is not None:
            loss = loss * weights
            loss = loss.sum() / weights.sum().clamp(min=1e-8)
        else:
            loss = loss.mean()

        with torch.no_grad():
            ps = (1.0 - pos_dist).mean().item()
            ns = (1.0 - neg_dist).mean().item()

        self._update_stats(loss.detach().item(), ps, ns)
        return loss

    def _update_stats(
        self, loss_val: float, pos_sim: float, neg_sim: float,
    ) -> None:
        s = self._stats
        decay = self._config.loss_ema_decay
        if s.total_steps == 0:
            s.ema_loss = loss_val
        else:
            s.ema_loss = decay * s.ema_loss + (1.0 - decay) * loss_val
        s.last_loss = loss_val
        s.mean_positive_sim = pos_sim
        s.mean_negative_sim = neg_sim
        s.similarity_gap = pos_sim - neg_sim
        s.total_steps += 1


# ── Combined online loss ───────────────────────────────────────────────────


class CombinedOnlineLoss(nn.Module):
    """Blends InfoNCE and triplet margin losses for online training.

    Using both losses together often stabilizes early training: InfoNCE
    provides strong gradients from the softmax denominator while the
    triplet margin prevents embedding collapse.

    The blend ratio is configurable via ``infonce_weight``.
    """

    def __init__(
        self,
        config: OnlineLossConfig | None = None,
        infonce_weight: float = 0.7,
    ) -> None:
        super().__init__()
        self._infonce = OnlineInfoNCELoss(config)
        self._triplet = OnlineTripletLoss(config)
        self._infonce_weight = infonce_weight

    @property
    def infonce(self) -> OnlineInfoNCELoss:
        return self._infonce

    @property
    def triplet(self) -> OnlineTripletLoss:
        return self._triplet

    @property
    def stats(self) -> LossStatistics:
        """Return the InfoNCE branch stats (primary signal)."""
        return self._infonce.stats

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute blended loss.

        Args:
            anchor: ``(B, D)``
            positive: ``(B, D)``
            negative: ``(B, D)``
            weights: optional ``(B,)``

        Returns:
            Scalar loss (weighted sum of InfoNCE + triplet margin).
        """
        w = self._infonce_weight
        loss_i = self._infonce(anchor, positive, negative, weights)
        loss_t = self._triplet(anchor, positive, negative, weights)
        return w * loss_i + (1.0 - w) * loss_t
