"""Bridge connecting DerivationContext to the temporal attention pipeline.

Provides adapters that convert DerivationContext metadata into the tensor
formats expected by TemporalCrossClauseAttention and enrich the attention
system with full inference chain embeddings from InferenceChainEncoder.

Two integration modes:
  1. **Lightweight** (DerivationAttentionAdapter): Extracts per-clause
     (depth, primary_rule, parent_count) tensors for TemporalPositionEncoder.
     Zero additional parameters, negligible overhead.

  2. **Rich** (ChainEnhancedAttentionAdapter): Adds InferenceChainEncoder
     embeddings as an additional attention bias, capturing the full
     derivation sequence structure beyond what depth/rule alone conveys.

Thread Safety:
  DerivationContext is lock-protected; adapters hold a reference but
  never mutate it. Tensor construction is allocation-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from pyladr.core.clause import Clause

from .derivation_context import DerivationContext
from .inference_chain_encoder import InferenceChainConfig, InferenceChainEncoder


@dataclass(frozen=True, slots=True)
class TemporalMetadata:
    """Packed tensor representation of derivation metadata for a clause batch.

    All tensors have shape (N,) and are ready to pass directly to
    TemporalCrossClauseAttention.forward().
    """

    derivation_depths: torch.Tensor   # (N,) long
    inference_types: torch.Tensor     # (N,) long
    parent_counts: torch.Tensor       # (N,) long
    clause_ids: torch.Tensor          # (N,) long


class DerivationAttentionAdapter:
    """Lightweight adapter: DerivationContext → temporal attention tensors.

    Extracts per-clause metadata from the DerivationContext and packs it
    into the tensor format expected by TemporalCrossClauseAttention.

    Usage::

        ctx = DerivationContext()
        adapter = DerivationAttentionAdapter(ctx)

        # During selection (clauses = list of SOS candidates)
        meta = adapter.extract_metadata(clauses)

        scores = temporal_attention.forward(
            base_embeddings=embs,
            derivation_depths=meta.derivation_depths,
            inference_types=meta.inference_types,
            parent_counts=meta.parent_counts,
            clause_ids=meta.clause_ids,
        )
    """

    def __init__(self, context: DerivationContext) -> None:
        self._ctx = context

    def extract_metadata(
        self,
        clauses: list[Clause],
        device: torch.device | None = None,
    ) -> TemporalMetadata:
        """Extract temporal metadata tensors for a batch of clauses.

        Clauses not registered in the context get depth=0, type=0 (INPUT),
        parents=0 — equivalent to axiom treatment (graceful degradation).

        Args:
            clauses: List of clauses to extract metadata for.
            device: Target device for tensors.

        Returns:
            TemporalMetadata with packed tensors.
        """
        N = len(clauses)
        depths = torch.zeros(N, dtype=torch.long)
        types = torch.zeros(N, dtype=torch.long)
        parents = torch.zeros(N, dtype=torch.long)
        ids = torch.zeros(N, dtype=torch.long)

        for i, clause in enumerate(clauses):
            ids[i] = clause.id
            info = self._ctx.get(clause.id)
            if info is not None:
                depths[i] = info.depth
                types[i] = info.primary_rule
                parents[i] = len(info.parent_ids)

        if device is not None:
            depths = depths.to(device)
            types = types.to(device)
            parents = parents.to(device)
            ids = ids.to(device)

        return TemporalMetadata(
            derivation_depths=depths,
            inference_types=types,
            parent_counts=parents,
            clause_ids=ids,
        )


class ChainEnhancedAttentionAdapter(nn.Module):
    """Rich adapter: adds inference chain embeddings as attention bias.

    Computes a per-clause temporal embedding from the full inference chain
    and projects it to a bias vector that is added to clause embeddings
    before cross-clause attention. This captures sequence-level patterns
    (e.g., "resolution followed by repeated demodulation") that the simpler
    depth/rule/parent features cannot represent.

    The chain embeddings are combined with the base clause embeddings via
    a learned gating mechanism so the model can learn how much temporal
    context to incorporate.

    Usage::

        ctx = DerivationContext()
        adapter = ChainEnhancedAttentionAdapter(ctx)

        # Enrich clause embeddings with chain context before attention
        enriched = adapter(clauses, base_embeddings)
        # enriched has same shape as base_embeddings
    """

    def __init__(
        self,
        context: DerivationContext,
        embedding_dim: int = 512,
        chain_config: InferenceChainConfig | None = None,
    ) -> None:
        super().__init__()
        self._ctx = context
        self._lightweight = DerivationAttentionAdapter(context)

        # Inference chain encoder
        cfg = chain_config or InferenceChainConfig(output_dim=128)
        self.chain_encoder = InferenceChainEncoder(cfg)

        # Project chain embedding to match clause embedding dimension
        self.chain_proj = nn.Linear(cfg.output_dim, embedding_dim)

        # Learned gate: how much chain context to blend in
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        clauses: list[Clause],
        base_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Enrich clause embeddings with inference chain context.

        Args:
            clauses: List of N clauses.
            base_embeddings: (N, D) base clause embeddings.

        Returns:
            (N, D) enriched clause embeddings (same shape as input).
        """
        device = base_embeddings.device

        # Extract inference chains
        chains = [
            self._ctx.get_inference_chain(c.id, max_length=self.chain_encoder.config.max_chain_length)
            for c in clauses
        ]

        # Pad and encode
        padded, lengths = self.chain_encoder.pad_chains(chains, device=device)
        chain_embs = self.chain_encoder(padded, lengths)  # (N, chain_output_dim)

        # Project to embedding dimension
        chain_projected = self.chain_proj(chain_embs)  # (N, D)

        # Gated fusion
        gate_input = torch.cat([base_embeddings, chain_projected], dim=-1)
        g = self.gate(gate_input)  # (N, D)

        enriched = g * base_embeddings + (1 - g) * chain_projected
        return self.norm(enriched)

    def extract_metadata(
        self,
        clauses: list[Clause],
        device: torch.device | None = None,
    ) -> TemporalMetadata:
        """Delegate to lightweight adapter for basic temporal tensors."""
        return self._lightweight.extract_metadata(clauses, device)
