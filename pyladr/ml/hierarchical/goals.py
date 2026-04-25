"""Goal-directed components for hierarchical clause selection.

Implements:
  1. GoalEncoder: Encodes goal clauses into a goal context representation
  2. GoalDirectedAttention: Biases level embeddings toward goal-relevant features
  3. DistanceComputer: Computes distance between clause and goal embeddings
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .architecture import HierarchyLevel

if TYPE_CHECKING:
    from pyladr.core.clause import Clause


# ── Goal Encoder ──────────────────────────────────────────────────────────


class GoalEncoder(nn.Module):
    """Encodes goal clauses into a fixed-dimensional goal context tensor.

    Produces a goal representation by:
    1. Extracting structural features from goal clauses
    2. Projecting to goal_embedding_dim
    3. Mean-pooling across multiple goals
    """

    def __init__(self, goal_embedding_dim: int = 128):
        super().__init__()
        self.goal_embedding_dim = goal_embedding_dim

        # Goal feature dim: [num_literals, is_unit, is_horn, is_positive,
        #                     is_negative, is_ground, weight]
        self.feature_dim = 7

        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, goal_embedding_dim),
            nn.ReLU(),
            nn.Linear(goal_embedding_dim, goal_embedding_dim),
        )
        self.norm = nn.LayerNorm(goal_embedding_dim)

    def forward(
        self,
        goal_clauses: List[Clause],
        goal_embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """Encode goal clauses into goal context.

        Args:
            goal_clauses: List of goal Clause objects.
            goal_embeddings: Pre-computed embeddings (unused if empty).

        Returns:
            (1, goal_embedding_dim) goal context tensor.
        """
        if not goal_clauses:
            device = next(self.parameters()).device
            return torch.zeros(1, self.goal_embedding_dim, device=device)

        # Extract features from goal clauses
        features = []
        for c in goal_clauses:
            features.append([
                float(c.num_literals),
                float(c.is_unit),
                float(c.is_horn),
                float(c.is_positive),
                float(c.is_negative),
                float(c.is_ground),
                float(c.weight),
            ])

        device = next(self.parameters()).device
        feat_tensor = torch.tensor(features, dtype=torch.float32, device=device)

        # Encode and pool
        encoded = self.encoder(feat_tensor)  # (G, goal_embedding_dim)
        encoded = self.norm(encoded)
        pooled = encoded.mean(dim=0, keepdim=True)  # (1, goal_embedding_dim)

        return pooled


# ── Goal-Directed Attention ──────────────────────────────────────────────


class GoalDirectedAttention(nn.Module):
    """Biases hierarchical embeddings toward goal-relevant features.

    Takes level embeddings and a goal context, and modulates each level's
    representations to emphasize features relevant to the proof goal.
    Uses a cross-attention mechanism where the goal context serves as query
    and level representations serve as keys/values.
    """

    def __init__(self, hidden_dim: int = 256, goal_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.goal_dim = goal_dim

        # Project goal to hidden dim for attention
        self.goal_proj = nn.Linear(goal_dim, hidden_dim)

        # Per-level goal attention
        self.level_attention = nn.ModuleDict()
        self.level_gates = nn.ModuleDict()
        self.level_norms = nn.ModuleDict()

        for level in HierarchyLevel:
            self.level_attention[level.name] = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True,
            )
            self.level_gates[level.name] = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            self.level_norms[level.name] = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        level_embeddings: Dict[HierarchyLevel, torch.Tensor],
        goal_context: torch.Tensor,
    ) -> Dict[HierarchyLevel, torch.Tensor]:
        """Apply goal-directed attention to level embeddings.

        Args:
            level_embeddings: {level: (N, hidden_dim)} per-level features.
            goal_context: (1, goal_dim) goal context tensor.

        Returns:
            Updated level_embeddings with goal-directed bias.
        """
        result = {}
        # Pool multi-element goal contexts to a single vector, then project
        if goal_context.dim() == 2 and goal_context.shape[0] > 1:
            goal_context = goal_context.mean(dim=0, keepdim=True)  # (1, goal_dim)
        goal_hidden = self.goal_proj(goal_context)  # (1, hidden_dim)

        for level, x in level_embeddings.items():
            if level.name not in self.level_attention or x.shape[0] == 0:
                result[level] = x
                continue

            # Cross-attention: goal queries attend to level representations
            # goal_hidden as query (1, 1, H), x as key/value (1, N, H)
            x_unsq = x.unsqueeze(0)  # (1, N, H)
            goal_unsq = goal_hidden.unsqueeze(0) if goal_hidden.dim() == 2 else goal_hidden

            attn_out, _ = self.level_attention[level.name](
                goal_unsq, x_unsq, x_unsq
            )  # (1, 1, H)

            # Broadcast goal-modulated context to all nodes
            context = attn_out.squeeze(0).expand_as(x)  # (N, H)

            # Gated fusion
            gate = self.level_gates[level.name](torch.cat([x, context], dim=-1))
            updated = gate * x + (1 - gate) * context
            result[level] = self.level_norms[level.name](updated)

        return result


# ── Distance Computer ────────────────────────────────────────────────────


class DistanceComputer(nn.Module):
    """Computes distance between clause embeddings and goal embeddings.

    Supports multiple distance metrics:
    - cosine: 1 - cosine_similarity
    - euclidean: L2 distance
    - learned: MLP-based learned distance function
    """

    def __init__(self, embedding_dim: int = 512, metric: str = "cosine"):
        super().__init__()
        self.metric = metric

        if metric == "learned":
            self.distance_net = nn.Sequential(
                nn.Linear(embedding_dim * 3, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        clause_emb: torch.Tensor,
        goal_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distance between clause and goal embeddings.

        Args:
            clause_emb: (N, D) or (D,) clause embedding(s).
            goal_emb: (M, D) or (D,) goal embedding(s).

        Returns:
            (N,) or scalar distance value(s) in [0, 1].
        """
        if clause_emb.dim() == 1:
            clause_emb = clause_emb.unsqueeze(0)
        if goal_emb.dim() == 1:
            goal_emb = goal_emb.unsqueeze(0)

        # Pool multiple goals
        if goal_emb.shape[0] > 1:
            goal_emb = goal_emb.mean(dim=0, keepdim=True)

        if self.metric == "cosine":
            sim = F.cosine_similarity(clause_emb, goal_emb.expand_as(clause_emb), dim=-1)
            return (1 - sim) / 2  # Normalize to [0, 1]

        elif self.metric == "euclidean":
            dist = torch.norm(clause_emb - goal_emb.expand_as(clause_emb), dim=-1)
            return torch.sigmoid(dist)  # Squash to [0, 1]

        elif self.metric == "learned":
            goal_expanded = goal_emb.expand_as(clause_emb)
            combined = torch.cat([
                clause_emb,
                goal_expanded,
                clause_emb * goal_expanded,
            ], dim=-1)
            return self.distance_net(combined).squeeze(-1)

        else:
            raise ValueError(f"Unknown distance metric: {self.metric}")
