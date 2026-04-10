"""Clause-level message passing network for hierarchical clause embeddings.

Implements the clause level of the hierarchical message passing architecture:
  1. Clause structure attention — multi-head attention over literal representations
     within each clause, producing a clause-level summary
  2. Inference rule potential — scores pairs of clauses for their potential as
     resolution/paramodulation partners
  3. Literal-to-clause composition — aggregates literal embeddings into a clause
     representation, gated by clause structural properties
  4. Inter-clause message passing — optional GNN layer over clause adjacency
     (e.g., clauses that share symbols or are unification candidates)

Designed to receive literal-level representations from LiteralLevelMPN and
produce clause-level representations consumed by ProofLevelMPN.

No modifications to core PyLADR data structures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from pyladr.core.clause import Clause


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ClauseLevelConfig:
    """Hyperparameters for the clause-level message passing network.

    Attributes:
        hidden_dim: Hidden dimension for representations.
        num_attention_heads: Number of heads in multi-head attention.
        num_layers: Number of inter-clause message passing layers.
        dropout: Dropout probability.
        clause_feature_dim: Dimension of clause property features
            (num_literals, is_unit, is_horn, is_positive, is_negative,
             is_ground, weight).
    """

    hidden_dim: int = 256
    num_attention_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    clause_feature_dim: int = 7


# ── Clause Property Encoder ───────────────────────────────────────────────


class ClausePropertyEncoder(nn.Module):
    """Encodes clause structural properties into a hidden representation.

    Takes a feature vector of clause properties and projects it to hidden_dim.
    """

    def __init__(self, clause_feature_dim: int = 7, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clause_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Encode clause property features.

        Args:
            features: (N, clause_feature_dim) clause property vectors.

        Returns:
            (N, hidden_dim) encoded clause properties.
        """
        return self.net(features)

    @staticmethod
    def extract_features(clauses: list[Clause]) -> torch.Tensor:
        """Extract feature vectors from Clause objects.

        Features: [num_literals, is_unit, is_horn, is_positive, is_negative,
                   is_ground, weight]

        Args:
            clauses: List of PyLADR Clause objects.

        Returns:
            Tensor of shape (len(clauses), 7).
        """
        features = []
        for c in clauses:
            features.append([
                float(c.num_literals),
                float(c.is_unit),
                float(c.is_horn),
                float(c.is_positive),
                float(c.is_negative),
                float(c.is_ground),
                float(c.weight),
            ])
        return torch.tensor(features, dtype=torch.float32)


# ── Inference Rule Potential ──────────────────────────────────────────────


class InferenceRulePotential(nn.Module):
    """Scores pairs of clauses for inference potential.

    Given two clause representations, predicts how productive an inference
    (resolution or paramodulation) between them would be. Output is a
    probability in [0, 1].

    When symmetric=True, the score is invariant to argument order
    (achieved by using a symmetric combination: sum + element-wise product).
    """

    def __init__(self, hidden_dim: int = 256, symmetric: bool = False):
        super().__init__()
        self.symmetric = symmetric

        if symmetric:
            # Symmetric: use sum + product (both order-invariant)
            self.net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        else:
            # Asymmetric: concatenate + product
            self.net = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

    def forward(
        self, clause_a: torch.Tensor, clause_b: torch.Tensor
    ) -> torch.Tensor:
        """Score inference potential between clause pairs.

        Args:
            clause_a: (N, hidden_dim) first clause representations.
            clause_b: (N, hidden_dim) second clause representations.

        Returns:
            (N,) probability scores in [0, 1].
        """
        if self.symmetric:
            combined = torch.cat([clause_a + clause_b, clause_a * clause_b], dim=-1)
        else:
            combined = torch.cat([clause_a, clause_b, clause_a * clause_b], dim=-1)

        return self.net(combined).squeeze(-1)


# ── Clause-Level Message Passing Network ─────────────────────────────────


class ClauseLevelMPN(nn.Module):
    """Clause-level message passing network.

    Aggregates literal-level representations into clause representations
    using multi-head attention, then optionally refines them through
    inter-clause message passing.

    Architecture:
      1. ClausePropertyEncoder encodes clause structural features
      2. Multi-head attention over literal representations (clause structure attention)
      3. Gated fusion of attention output with clause property encoding
      4. Optional inter-clause message passing via adjacency graph
      5. Layer norm + residual connections

    Input:
      - literal_reprs: (batch, max_literals, hidden_dim) — from LiteralLevelMPN
      - clause_features: (batch, clause_feature_dim) — structural properties
      - mask: (batch, max_literals) — True for valid literal positions
      - adjacency: (2, num_edges) — optional inter-clause edge index

    Output:
      - (batch, hidden_dim) — clause-level representations
    """

    def __init__(self, config: ClauseLevelConfig | None = None):
        super().__init__()
        self.config = config or ClauseLevelConfig()
        c = self.config

        # Clause property encoder
        self.property_encoder = ClausePropertyEncoder(
            clause_feature_dim=c.clause_feature_dim,
            hidden_dim=c.hidden_dim,
        )

        # Multi-head attention for clause structure
        self.structure_attention = nn.MultiheadAttention(
            embed_dim=c.hidden_dim,
            num_heads=c.num_attention_heads,
            dropout=c.dropout,
            batch_first=True,
        )
        self.attn_layer_norm = nn.LayerNorm(c.hidden_dim)

        # Gated fusion of attention output with clause properties
        self.fusion_gate = nn.Sequential(
            nn.Linear(c.hidden_dim * 2, c.hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_transform = nn.Linear(c.hidden_dim * 2, c.hidden_dim)
        self.fusion_layer_norm = nn.LayerNorm(c.hidden_dim)

        # Inter-clause message passing layers
        self.inter_clause_layers = nn.ModuleList()
        self.inter_clause_norms = nn.ModuleList()
        for _ in range(c.num_layers):
            self.inter_clause_layers.append(
                nn.Linear(c.hidden_dim * 2, c.hidden_dim)
            )
            self.inter_clause_norms.append(nn.LayerNorm(c.hidden_dim))

        self.dropout = nn.Dropout(c.dropout)

    def clause_structure_attention(
        self,
        literal_reprs: torch.Tensor,
        clause_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-head attention over literal representations within clauses.

        The clause property encoding serves as the query, while literal
        representations are keys and values. This allows the clause to
        attend to its literals in a property-aware manner.

        Args:
            literal_reprs: (batch, max_literals, hidden_dim).
            clause_features: (batch, clause_feature_dim).
            mask: (batch, max_literals) bool, True for valid positions.

        Returns:
            (batch, hidden_dim) attended clause representations.
        """
        batch_size = literal_reprs.shape[0]

        # Encode clause properties as query
        clause_encoding = self.property_encoder(clause_features)  # (B, H)
        query = clause_encoding.unsqueeze(1)  # (B, 1, H)

        # Build attention mask: True means IGNORE in PyTorch MHA
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # Invert: True = ignore

            # Check for fully-masked clauses (empty clauses)
            all_masked = key_padding_mask.all(dim=1)  # (B,)
            if all_masked.any():
                # For fully-masked clauses, use clause encoding only
                # Unmask one position to avoid NaN in attention
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False

        # Multi-head attention: clause queries attend to literal keys/values
        attn_output, _ = self.structure_attention(
            query, literal_reprs, literal_reprs,
            key_padding_mask=key_padding_mask,
        )
        attn_output = attn_output.squeeze(1)  # (B, H)

        # For fully-masked clauses, zero out the attention contribution
        if mask is not None:
            all_masked = (~mask).all(dim=1)  # (B,)
            if all_masked.any():
                attn_output = attn_output.clone()
                attn_output[all_masked] = 0.0

        # Residual connection with clause encoding
        output = self.attn_layer_norm(attn_output + clause_encoding)
        return output

    def compose_literals_to_clause(
        self,
        literal_reprs: torch.Tensor,
        clause_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate literal representations into clause representation.

        Uses attention-based aggregation followed by gated fusion with
        clause property features. Includes a mean-pooling residual path
        from literal representations to ensure gradient flow.

        Args:
            literal_reprs: (batch, max_literals, hidden_dim).
            clause_features: (batch, clause_feature_dim).
            mask: (batch, max_literals) bool, True for valid positions.

        Returns:
            (batch, hidden_dim) composed clause representations.
        """
        # Get attention-based aggregation
        attn_repr = self.clause_structure_attention(
            literal_reprs, clause_features, mask=mask
        )

        # Mean-pooling residual: direct path from literal representations
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)
            lit_sum = (literal_reprs * mask_expanded).sum(dim=1)  # (B, H)
            lit_count = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            lit_mean = lit_sum / lit_count
        else:
            lit_mean = literal_reprs.mean(dim=1)  # (B, H)

        # Combine attention output with mean pooling residual
        attn_repr = attn_repr + lit_mean

        # Gated fusion with clause property encoding
        clause_encoding = self.property_encoder(clause_features)
        combined = torch.cat([attn_repr, clause_encoding], dim=-1)
        gate = self.fusion_gate(combined)
        transform = self.fusion_transform(combined)
        fused = gate * attn_repr + (1 - gate) * transform

        return self.fusion_layer_norm(fused)

    def _inter_clause_message_pass(
        self,
        clause_reprs: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Message passing between clauses via adjacency graph.

        Args:
            clause_reprs: (N, hidden_dim) clause representations.
            adjacency: (2, E) edge index for inter-clause connections.

        Returns:
            (N, hidden_dim) updated clause representations.
        """
        x = clause_reprs
        src, dst = adjacency[0], adjacency[1]

        for layer, norm in zip(self.inter_clause_layers, self.inter_clause_norms):
            # Gather neighbor messages
            neighbor_reprs = x[src]  # (E, H)

            # Aggregate messages per destination node (mean)
            num_nodes = x.shape[0]
            agg = torch.zeros_like(x)  # (N, H)
            count = torch.zeros(num_nodes, 1, device=x.device)
            agg.index_add_(0, dst, neighbor_reprs)
            count.index_add_(0, dst, torch.ones(src.shape[0], 1, device=x.device))
            count = count.clamp(min=1)
            agg = agg / count

            # Combine self representation with aggregated neighbors
            combined = torch.cat([x, agg], dim=-1)
            update = F.relu(layer(combined))
            update = self.dropout(update)

            # Residual + layer norm
            x = norm(x + update)

        return x

    def forward(
        self,
        literal_reprs: torch.Tensor,
        clause_features: torch.Tensor,
        mask: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full clause-level forward pass.

        1. Compose literal representations into clause representations
        2. Optionally refine via inter-clause message passing

        Args:
            literal_reprs: (batch, max_literals, hidden_dim) from literal level.
            clause_features: (batch, clause_feature_dim) structural properties.
            mask: (batch, max_literals) bool, True for valid positions.
            adjacency: (2, E) optional inter-clause edge index.

        Returns:
            (batch, hidden_dim) clause-level representations.
        """
        # Step 1: Compose literals into clause representations
        clause_reprs = self.compose_literals_to_clause(
            literal_reprs, clause_features, mask=mask
        )

        # Step 2: Inter-clause message passing (if adjacency provided)
        if adjacency is not None and adjacency.shape[1] > 0:
            clause_reprs = self._inter_clause_message_pass(clause_reprs, adjacency)

        return clause_reprs
