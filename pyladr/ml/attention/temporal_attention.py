"""Temporal and multi-source attention for integrated clause scoring.

Extends the base cross-clause attention with temporal awareness from derivation
history and multi-source feature fusion from hierarchical message passing and
property-invariant embeddings. This module is the integration bridge between
all four ML enhancements:

  1. Hierarchical message passing → multi-level clause features
  2. Property-invariant embeddings → canonical (symbol-independent) features
  3. Derivation history → temporal context and inference chain features
  4. Cross-clause attention → relational scoring (this package)

Architecture:
  - TemporalPositionEncoder: Encodes derivation depth and inference chain
    metadata as position-like signals for attention bias
  - MultiSourceFusion: Fuses features from hierarchical, invariant, and
    flat embedding sources into a unified representation
  - TemporalCrossClauseAttention: Full pipeline combining temporal encoding,
    multi-source fusion, and cross-clause attention

All components are opt-in. When a source is unavailable, the corresponding
branch is gracefully skipped (zero contribution, not error).

Thread Safety:
  Stateless nn.Modules under torch.no_grad() — safe for concurrent inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyladr.ml.attention.cross_clause import (
    CrossClauseAttentionConfig,
    CrossClauseAttentionScorer,
    MultiHeadClauseAttention,
)


# ── Configuration ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TemporalAttentionConfig:
    """Configuration for temporal and multi-source attention.

    Attributes:
        base_config: Cross-clause attention configuration.
        use_temporal_encoding: Enable derivation-based temporal position encoding.
        max_derivation_depth: Maximum derivation depth for position encoding.
        num_inference_types: Number of distinct JustType values for type embeddings.
        temporal_dim: Dimension of temporal position encodings.
        use_multi_source_fusion: Enable fusion of multiple embedding sources.
        hierarchical_dim: Expected dimension from hierarchical embeddings (0 = disabled).
        invariant_dim: Expected dimension from property-invariant embeddings (0 = disabled).
        fusion_method: How to fuse multiple sources: "gate", "attention", "concat_project".
    """

    base_config: CrossClauseAttentionConfig = field(
        default_factory=CrossClauseAttentionConfig,
    )
    use_temporal_encoding: bool = True
    max_derivation_depth: int = 100
    num_inference_types: int = 22  # matches JustType enum count
    temporal_dim: int = 64
    use_multi_source_fusion: bool = True
    hierarchical_dim: int = 0  # 0 = hierarchical embeddings not available
    invariant_dim: int = 0  # 0 = invariant embeddings not available
    fusion_method: str = "gate"  # "gate", "attention", "concat_project"


# ── Temporal Position Encoder ─────────────────────────────────────────────


class TemporalPositionEncoder(nn.Module):
    """Encodes derivation history metadata as temporal position signals.

    Converts derivation depth, inference rule type, and parent relationships
    into a dense vector that biases attention to account for the temporal
    structure of the proof search.

    Features encoded:
    - Derivation depth (sinusoidal, like standard positional encoding)
    - Primary inference rule type (learned embedding)
    - Number of parents (normalized scalar)
    """

    def __init__(self, config: TemporalAttentionConfig):
        super().__init__()
        self.config = config
        d = config.temporal_dim

        # Sinusoidal depth encoding (like transformer positional encoding)
        max_depth = config.max_derivation_depth
        pe = torch.zeros(max_depth + 1, d // 2)
        position = torch.arange(0, max_depth + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d // 2, 2, dtype=torch.float) * (-math.log(10000.0) / (d // 2))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("depth_pe", pe)

        # Learned inference type embedding
        self.inference_type_embed = nn.Embedding(
            config.num_inference_types, d // 4
        )

        # Parent count feature
        self.parent_proj = nn.Linear(1, d // 4)

        # Final projection to temporal_dim
        self.output_proj = nn.Linear(d // 2 + d // 4 + d // 4, d)
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        derivation_depths: torch.Tensor,
        inference_types: torch.Tensor,
        parent_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Encode temporal position for each clause.

        Args:
            derivation_depths: (N,) integer derivation depths.
            inference_types: (N,) integer JustType indices.
            parent_counts: (N,) integer number of parent clauses.

        Returns:
            (N, temporal_dim) temporal position encodings.
        """
        # Depth encoding (sinusoidal lookup)
        depths_clamped = derivation_depths.clamp(0, self.config.max_derivation_depth)
        depth_enc = self.depth_pe[depths_clamped]  # (N, d//2)

        # Inference type embedding
        types_clamped = inference_types.clamp(0, self.config.num_inference_types - 1)
        type_enc = self.inference_type_embed(types_clamped)  # (N, d//4)

        # Parent count feature
        parent_feat = parent_counts.float().unsqueeze(-1)  # (N, 1)
        parent_enc = self.parent_proj(parent_feat)  # (N, d//4)

        # Concatenate and project
        combined = torch.cat([depth_enc, type_enc, parent_enc], dim=-1)
        return self.norm(self.output_proj(combined))


# ── Multi-Source Feature Fusion ───────────────────────────────────────────


class MultiSourceFusion(nn.Module):
    """Fuses embeddings from multiple sources into a unified representation.

    Combines up to three embedding sources:
    - Base/flat GNN embeddings (always available)
    - Hierarchical message passing embeddings (optional)
    - Property-invariant embeddings (optional)

    Fusion methods:
    - "gate": Learned gating weights per source (default, efficient)
    - "attention": Source-level attention (more expressive, higher cost)
    - "concat_project": Concatenate all + linear projection (simple baseline)
    """

    def __init__(self, config: TemporalAttentionConfig):
        super().__init__()
        self.config = config
        base_dim = config.base_config.embedding_dim
        hier_dim = config.hierarchical_dim
        inv_dim = config.invariant_dim

        # Count active sources
        self.has_hierarchical = hier_dim > 0
        self.has_invariant = inv_dim > 0
        self.num_sources = 1 + int(self.has_hierarchical) + int(self.has_invariant)

        method = config.fusion_method

        if method == "gate":
            self._build_gated_fusion(base_dim, hier_dim, inv_dim)
        elif method == "attention":
            self._build_attention_fusion(base_dim, hier_dim, inv_dim)
        elif method == "concat_project":
            self._build_concat_fusion(base_dim, hier_dim, inv_dim)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

        self.method = method
        self.output_dim = base_dim  # output always matches base dim
        self.norm = nn.LayerNorm(base_dim)

    def _build_gated_fusion(self, base_dim, hier_dim, inv_dim):
        """Build gated fusion: learned per-source gates."""
        # Project each source to base_dim
        self.base_proj = nn.Identity()  # base is already base_dim
        if self.has_hierarchical:
            self.hier_proj = nn.Linear(hier_dim, base_dim)
        if self.has_invariant:
            self.inv_proj = nn.Linear(inv_dim, base_dim)

        # Gate network: takes concatenated projected sources, outputs per-source weights
        self.gate_net = nn.Sequential(
            nn.Linear(base_dim * self.num_sources, self.num_sources),
            nn.Softmax(dim=-1),
        )

    def _build_attention_fusion(self, base_dim, hier_dim, inv_dim):
        """Build attention-based fusion: source-level attention."""
        # Project all to base_dim
        self.base_proj = nn.Identity()
        if self.has_hierarchical:
            self.hier_proj = nn.Linear(hier_dim, base_dim)
        if self.has_invariant:
            self.inv_proj = nn.Linear(inv_dim, base_dim)

        # Attention over sources
        self.source_query = nn.Linear(base_dim, base_dim)
        self.source_key = nn.Linear(base_dim, base_dim)
        self.source_value = nn.Linear(base_dim, base_dim)
        self._attn_scale = 1.0 / math.sqrt(base_dim)

    def _build_concat_fusion(self, base_dim, hier_dim, inv_dim):
        """Build concatenation + projection fusion."""
        total_dim = base_dim + (hier_dim if self.has_hierarchical else 0) + (inv_dim if self.has_invariant else 0)
        self.concat_proj = nn.Sequential(
            nn.Linear(total_dim, base_dim),
            nn.GELU(),
            nn.Linear(base_dim, base_dim),
        )

    def forward(
        self,
        base_embeddings: torch.Tensor,
        hierarchical_embeddings: Optional[torch.Tensor] = None,
        invariant_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse multiple embedding sources.

        Args:
            base_embeddings: (N, base_dim) base/flat GNN embeddings.
            hierarchical_embeddings: Optional (N, hier_dim) hierarchical embeddings.
            invariant_embeddings: Optional (N, inv_dim) property-invariant embeddings.

        Returns:
            (N, base_dim) fused embeddings.
        """
        if self.method == "gate":
            return self._gated_forward(base_embeddings, hierarchical_embeddings, invariant_embeddings)
        elif self.method == "attention":
            return self._attention_forward(base_embeddings, hierarchical_embeddings, invariant_embeddings)
        else:
            return self._concat_forward(base_embeddings, hierarchical_embeddings, invariant_embeddings)

    def _gated_forward(self, base, hier, inv):
        N = base.shape[0]
        projected = [self.base_proj(base)]

        if self.has_hierarchical and hier is not None:
            projected.append(self.hier_proj(hier))
        elif self.has_hierarchical:
            projected.append(torch.zeros_like(base))

        if self.has_invariant and inv is not None:
            projected.append(self.inv_proj(inv))
        elif self.has_invariant:
            projected.append(torch.zeros_like(base))

        # Compute gates
        concat = torch.cat(projected, dim=-1)  # (N, base_dim * num_sources)
        gates = self.gate_net(concat)  # (N, num_sources)

        # Weighted sum
        stacked = torch.stack(projected, dim=1)  # (N, num_sources, base_dim)
        fused = (stacked * gates.unsqueeze(-1)).sum(dim=1)  # (N, base_dim)

        return self.norm(fused)

    def _attention_forward(self, base, hier, inv):
        projected = [self.base_proj(base)]

        if self.has_hierarchical and hier is not None:
            projected.append(self.hier_proj(hier))
        elif self.has_hierarchical:
            projected.append(torch.zeros_like(base))

        if self.has_invariant and inv is not None:
            projected.append(self.inv_proj(inv))
        elif self.has_invariant:
            projected.append(torch.zeros_like(base))

        # Stack: (N, S, D) where S = num_sources
        stacked = torch.stack(projected, dim=1)

        # Attention: query from base, keys/values from all sources
        q = self.source_query(base).unsqueeze(1)  # (N, 1, D)
        k = self.source_key(stacked)  # (N, S, D)
        v = self.source_value(stacked)  # (N, S, D)

        attn = torch.bmm(q, k.transpose(1, 2)) * self._attn_scale  # (N, 1, S)
        attn = F.softmax(attn, dim=-1)
        fused = torch.bmm(attn, v).squeeze(1)  # (N, D)

        return self.norm(fused)

    def _concat_forward(self, base, hier, inv):
        parts = [base]
        if self.has_hierarchical:
            parts.append(hier if hier is not None else torch.zeros(base.shape[0], self.config.hierarchical_dim, device=base.device))
        if self.has_invariant:
            parts.append(inv if inv is not None else torch.zeros(base.shape[0], self.config.invariant_dim, device=base.device))

        concat = torch.cat(parts, dim=-1)
        return self.norm(self.concat_proj(concat))


# ── Temporal Cross-Clause Attention ───────────────────────────────────────


class TemporalCrossClauseAttention(nn.Module):
    """Full integrated pipeline: temporal encoding + multi-source fusion + attention.

    This is the integration module for Task #5, combining all four ML enhancements:

    Pipeline:
    1. Multi-source fusion: Combine flat, hierarchical, and invariant embeddings
    2. Temporal encoding: Add derivation history position signals
    3. Cross-clause attention: Relational scoring over fused + temporal representations

    Usage:
        config = TemporalAttentionConfig(
            base_config=CrossClauseAttentionConfig(enabled=True, embedding_dim=512),
            use_temporal_encoding=True,
            use_multi_source_fusion=True,
            hierarchical_dim=512,
            invariant_dim=256,
        )
        model = TemporalCrossClauseAttention(config)

        scores = model.score_clauses(
            base_embeddings=flat_embs,       # from GNNEmbeddingProvider
            hierarchical_embeddings=hier_embs,  # from Task #2
            invariant_embeddings=inv_embs,      # from Task #3
            derivation_depths=depths,            # from Task #4
            inference_types=types,               # from Task #4
            parent_counts=parents,               # from Task #4
        )
    """

    def __init__(self, config: TemporalAttentionConfig):
        super().__init__()
        self.config = config
        base_dim = config.base_config.embedding_dim

        # Multi-source fusion
        if config.use_multi_source_fusion and (config.hierarchical_dim > 0 or config.invariant_dim > 0):
            self.fusion = MultiSourceFusion(config)
        else:
            self.fusion = None

        # Temporal position encoder
        if config.use_temporal_encoding:
            self.temporal_encoder = TemporalPositionEncoder(config)
            # Project temporal encoding to add to clause embeddings
            self.temporal_proj = nn.Linear(config.temporal_dim, base_dim)
        else:
            self.temporal_encoder = None
            self.temporal_proj = None

        # Cross-clause attention scorer (from base module)
        self.attention_scorer = CrossClauseAttentionScorer(config.base_config)

    def forward(
        self,
        base_embeddings: torch.Tensor,
        hierarchical_embeddings: Optional[torch.Tensor] = None,
        invariant_embeddings: Optional[torch.Tensor] = None,
        derivation_depths: Optional[torch.Tensor] = None,
        inference_types: Optional[torch.Tensor] = None,
        parent_counts: Optional[torch.Tensor] = None,
        goal_context: Optional[torch.Tensor] = None,
        clause_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute integrated relational scores.

        Args:
            base_embeddings: (N, D) base clause embeddings (required).
            hierarchical_embeddings: Optional (N, H) hierarchical features.
            invariant_embeddings: Optional (N, I) property-invariant features.
            derivation_depths: Optional (N,) derivation depths.
            inference_types: Optional (N,) JustType indices.
            parent_counts: Optional (N,) parent clause counts.
            goal_context: Optional goal context vector.
            clause_ids: Optional clause IDs for position bias.

        Returns:
            (N,) relational scores.
        """
        # Step 1: Multi-source fusion
        if self.fusion is not None:
            fused = self.fusion(base_embeddings, hierarchical_embeddings, invariant_embeddings)
        else:
            fused = base_embeddings

        # Step 2: Add temporal encoding
        if self.temporal_encoder is not None and derivation_depths is not None:
            # Default temporal features if not all provided
            N = fused.shape[0]
            device = fused.device
            if inference_types is None:
                inference_types = torch.zeros(N, dtype=torch.long, device=device)
            if parent_counts is None:
                parent_counts = torch.zeros(N, dtype=torch.long, device=device)

            temporal = self.temporal_encoder(derivation_depths, inference_types, parent_counts)
            fused = fused + self.temporal_proj(temporal)

        # Step 3: Cross-clause attention scoring
        return self.attention_scorer(fused, goal_context, clause_ids)

    @torch.no_grad()
    def score_clauses(
        self,
        base_embeddings: torch.Tensor,
        hierarchical_embeddings: Optional[torch.Tensor] = None,
        invariant_embeddings: Optional[torch.Tensor] = None,
        derivation_depths: Optional[torch.Tensor] = None,
        inference_types: Optional[torch.Tensor] = None,
        parent_counts: Optional[torch.Tensor] = None,
        goal_context: Optional[torch.Tensor] = None,
        clause_ids: Optional[torch.Tensor] = None,
    ) -> list[float]:
        """Inference-mode scoring returning Python list.

        Convenience method for selection integration.
        """
        was_training = self.training
        self.eval()
        scores = self.forward(
            base_embeddings, hierarchical_embeddings, invariant_embeddings,
            derivation_depths, inference_types, parent_counts,
            goal_context, clause_ids,
        )
        if was_training:
            self.train()
        return scores.tolist()
