"""Hierarchical message passing modules for multi-level aggregation.

Implements the three core message passing components used by HierarchicalClauseGNN:
  1. IntraLevelMP: Message passing within a single hierarchy level
  2. InterLevelMP: Bidirectional message passing between adjacent levels
  3. CrossLevelAttention: Attention between non-adjacent hierarchy levels

These compose the hierarchical message passing rounds in HierarchicalClauseGNN:
  for each round:
      intra-level → inter-level → cross-level
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .architecture import HierarchyLevel


# ── Intra-Level Message Passing ────────────────────────────────────────────


class IntraLevelMP(nn.Module):
    """Message passing within a single hierarchy level.

    Performs SAGEConv-style message passing between nodes at the same
    hierarchy level using the edge indices extracted from the clause graph.

    For levels with few or no intra-level edges (e.g., CLAUSE), this
    degenerates to a self-transform with layer norm.
    """

    def __init__(self, level: HierarchyLevel, hidden_dim: int):
        super().__init__()
        self.level = level
        self.hidden_dim = hidden_dim

        # Message computation: neighbor aggregation + self transform
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update: combine self with aggregated neighbors
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Intra-level message passing.

        Args:
            x: (N, hidden_dim) node features at this level.
            edge_index: (2, E) edges between nodes at this level.

        Returns:
            (N, hidden_dim) updated node features.
        """
        num_nodes = x.shape[0]

        if edge_index.shape[1] == 0 or num_nodes == 0:
            # No edges: just apply self-transform with residual
            return self.norm(self.message_mlp(x) + x)

        src, dst = edge_index[0], edge_index[1]

        # Compute messages from source nodes
        messages = self.message_mlp(x[src])  # (E, H)

        # Aggregate messages per destination (mean)
        agg = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
        count = torch.zeros(num_nodes, 1, device=x.device)
        agg.index_add_(0, dst, messages)
        count.index_add_(0, dst, torch.ones(src.shape[0], 1, device=x.device))
        count = count.clamp(min=1)
        agg = agg / count

        # Update with residual
        combined = torch.cat([x, agg], dim=-1)
        updated = self.update_mlp(combined)
        updated = self.dropout(updated)

        return self.norm(updated + x)


# ── Inter-Level Message Passing ─────────────────────────────────────────────


class InterLevelMP(nn.Module):
    """Bidirectional message passing between adjacent hierarchy levels.

    Implements both bottom-up (lower → upper) and top-down (upper → lower)
    information flow between adjacent levels in the hierarchy.

    Bottom-up: aggregates child representations to enrich parent level.
    Top-down: propagates parent context back to child level.
    """

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Bottom-up: lower → upper
        self.bottom_up_msg = nn.Linear(hidden_dim, hidden_dim)
        self.bottom_up_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.bottom_up_norm = nn.LayerNorm(output_dim)

        # Top-down: upper → lower
        self.top_down_msg = nn.Linear(hidden_dim, hidden_dim)
        self.top_down_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.top_down_norm = nn.LayerNorm(output_dim)

        # Gating for bidirectional fusion
        self.bottom_up_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Sigmoid(),
        )
        self.top_down_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        lower_x: torch.Tensor,
        upper_x: torch.Tensor,
        bottom_up_edges: torch.Tensor,
        top_down_edges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bidirectional inter-level message passing.

        Args:
            lower_x: (N_lower, hidden_dim) lower level features.
            upper_x: (N_upper, hidden_dim) upper level features.
            bottom_up_edges: (2, E) edges from lower to upper nodes.
                Row 0 = lower node indices, Row 1 = upper node indices.
            top_down_edges: (2, E) edges from upper to lower nodes.
                Row 0 = upper node indices, Row 1 = lower node indices.

        Returns:
            Tuple of (updated_lower, updated_upper).
        """
        updated_lower = lower_x
        updated_upper = upper_x

        # Bottom-up: lower → upper
        if bottom_up_edges.numel() > 0 and bottom_up_edges.shape[1] > 0:
            src, dst = bottom_up_edges[0], bottom_up_edges[1]
            messages = self.bottom_up_msg(lower_x[src])

            # Aggregate per upper node
            agg = torch.zeros_like(upper_x)
            count = torch.zeros(upper_x.shape[0], 1, device=upper_x.device)
            agg.index_add_(0, dst, messages)
            count.index_add_(0, dst, torch.ones(src.shape[0], 1, device=upper_x.device))
            count = count.clamp(min=1)
            agg = agg / count

            # Gated update
            gate_input = torch.cat([upper_x, agg], dim=-1)
            gate = self.bottom_up_gate(gate_input)
            update = self.bottom_up_update(gate_input)
            update = self.dropout(update)
            updated_upper = self.bottom_up_norm(gate * upper_x + (1 - gate) * update)

        # Top-down: upper → lower
        if top_down_edges.numel() > 0 and top_down_edges.shape[1] > 0:
            src, dst = top_down_edges[0], top_down_edges[1]
            messages = self.top_down_msg(upper_x[src])

            # Aggregate per lower node
            agg = torch.zeros_like(lower_x)
            count = torch.zeros(lower_x.shape[0], 1, device=lower_x.device)
            agg.index_add_(0, dst, messages)
            count.index_add_(0, dst, torch.ones(src.shape[0], 1, device=lower_x.device))
            count = count.clamp(min=1)
            agg = agg / count

            # Gated update
            gate_input = torch.cat([lower_x, agg], dim=-1)
            gate = self.top_down_gate(gate_input)
            update = self.top_down_update(gate_input)
            update = self.dropout(update)
            updated_lower = self.top_down_norm(gate * lower_x + (1 - gate) * update)

        return updated_lower, updated_upper


# ── Cross-Level Attention ───────────────────────────────────────────────────


class CrossLevelAttention(nn.Module):
    """Attention mechanism between non-adjacent hierarchy levels.

    Enables direct information flow between levels that are more than
    one step apart (e.g., SYMBOL → CLAUSE skip connection). This allows
    higher levels to directly attend to fine-grained features without
    information loss through intermediate levels.

    Uses multi-head attention where each level can attend to all other
    levels. A level mask prevents self-attention (handled by IntraLevelMP)
    and optionally restricts to non-adjacent pairs only.
    """

    def __init__(
        self,
        levels: List[HierarchyLevel],
        hidden_dim: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.levels = levels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Per-level query/key/value projections
        self.query_proj = nn.ModuleDict({
            level.name: nn.Linear(hidden_dim, hidden_dim)
            for level in levels
        })
        self.key_proj = nn.ModuleDict({
            level.name: nn.Linear(hidden_dim, hidden_dim)
            for level in levels
        })
        self.value_proj = nn.ModuleDict({
            level.name: nn.Linear(hidden_dim, hidden_dim)
            for level in levels
        })

        # Output projection per level
        self.output_proj = nn.ModuleDict({
            level.name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for level in levels
        })

        self.norms = nn.ModuleDict({
            level.name: nn.LayerNorm(hidden_dim)
            for level in levels
        })

        self.scale = (hidden_dim // num_heads) ** -0.5
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        level_embeddings: Dict[HierarchyLevel, torch.Tensor],
        cross_indices: Dict[str, torch.Tensor],
    ) -> Dict[HierarchyLevel, torch.Tensor]:
        """Cross-level attention between non-adjacent hierarchy levels.

        For each level, computes attention over representations from all
        other levels (pooled to a single vector per level), then applies
        a gated residual update.

        Args:
            level_embeddings: {level: (N_level, hidden_dim)} per-level features.
            cross_indices: Currently unused; reserved for sparse attention patterns.

        Returns:
            Updated level_embeddings dict.
        """
        result = {}

        # Pool each level to a single representative vector
        level_summaries: Dict[HierarchyLevel, torch.Tensor] = {}
        for level, x in level_embeddings.items():
            if x.shape[0] > 0:
                level_summaries[level] = x.mean(dim=0, keepdim=True)  # (1, H)

        if len(level_summaries) < 2:
            return level_embeddings

        for target_level, target_x in level_embeddings.items():
            if target_level.name not in self.query_proj:
                result[target_level] = target_x
                continue

            if target_x.shape[0] == 0:
                result[target_level] = target_x
                continue

            # Compute query from target level (use pooled representation)
            target_summary = level_summaries.get(target_level)
            if target_summary is None:
                result[target_level] = target_x
                continue

            q = self.query_proj[target_level.name](target_summary)  # (1, H)

            # Build keys and values from non-adjacent levels
            kv_list = []
            for source_level, source_summary in level_summaries.items():
                if source_level == target_level:
                    continue  # Skip self (handled by IntraLevelMP)
                k = self.key_proj[source_level.name](source_summary)  # (1, H)
                v = self.value_proj[source_level.name](source_summary)  # (1, H)
                kv_list.append((k, v))

            if not kv_list:
                result[target_level] = target_x
                continue

            # Stack keys and values: (num_sources, H)
            keys = torch.cat([kv[0] for kv in kv_list], dim=0)
            values = torch.cat([kv[1] for kv in kv_list], dim=0)

            # Scaled dot-product attention
            attn_scores = torch.matmul(q, keys.T) * self.scale  # (1, num_sources)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Weighted combination of values
            context = torch.matmul(attn_weights, values)  # (1, H)

            # Project and broadcast to all nodes at target level
            update = self.output_proj[target_level.name](context)  # (1, H)
            update = update.expand_as(target_x)  # (N_target, H)

            # Residual + norm
            result[target_level] = self.norms[target_level.name](target_x + update)

        return result
