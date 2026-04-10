"""Core hierarchical GNN architecture for PyLADR.

This module defines the main HierarchicalClauseGNN model that extends the existing
heterogeneous GNN with 5-level hierarchical message passing, cross-level attention,
and goal-directed capabilities.

The architecture maintains full backward compatibility with the existing
HeterogeneousClauseGNN while adding powerful hierarchical features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from pyladr.ml.graph.clause_encoder import GNNConfig, HeterogeneousClauseGNN
from pyladr.ml.graph.clause_graph import NodeType


class HierarchyLevel(Enum):
    """Hierarchy levels for the hierarchical GNN architecture."""

    SYMBOL = 0      # Foundation: symbols and variables
    TERM = 1        # Compositional: terms and subterms
    LITERAL = 2     # Logical: literals and equations
    CLAUSE = 3      # Unit: complete clauses
    PROOF = 4       # Global: proof context and goals

    @property
    def node_types(self) -> List[str]:
        """Get the node types associated with this hierarchy level."""
        mapping = {
            HierarchyLevel.SYMBOL: [NodeType.SYMBOL.value, NodeType.VARIABLE.value],
            HierarchyLevel.TERM: [NodeType.TERM.value],
            HierarchyLevel.LITERAL: [NodeType.LITERAL.value],
            HierarchyLevel.CLAUSE: [NodeType.CLAUSE.value],
            HierarchyLevel.PROOF: ["proof"],  # New node type for proof-level context
        }
        return mapping[self]

    def is_adjacent_to(self, other: 'HierarchyLevel') -> bool:
        """Check if this level is adjacent to another in the hierarchy."""
        return abs(self.value - other.value) == 1


@dataclass(frozen=True, slots=True)
class HierarchicalGNNConfig:
    """Configuration for the hierarchical GNN architecture.

    Extends the base GNNConfig with hierarchical-specific parameters while
    maintaining backward compatibility.
    """

    # Base GNN configuration (backward compatible)
    base_config: GNNConfig = field(default_factory=GNNConfig)

    # Hierarchical architecture parameters
    hierarchy_levels: int = 5
    intra_level_layers: int = 3
    inter_level_rounds: int = 2
    cross_level_enabled: bool = True
    cross_level_heads: int = 4

    # Goal-directed features
    goal_attention_enabled: bool = True
    goal_embedding_dim: int = 128
    distance_metric: str = "cosine"  # "cosine", "euclidean", "learned"

    # Incremental update features
    incremental_enabled: bool = True
    update_batch_size: int = 32
    staleness_threshold: float = 0.1

    # Performance optimization
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    gradient_checkpointing: bool = False

    @property
    def hidden_dim(self) -> int:
        """Get hidden dimension from base config."""
        return self.base_config.hidden_dim

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension from base config."""
        return self.base_config.embedding_dim

    @property
    def dropout(self) -> float:
        """Get dropout rate from base config."""
        return self.base_config.dropout


class HierarchicalClauseGNN(nn.Module):
    """Hierarchical GNN with 5-level message passing and goal guidance.

    This model extends HeterogeneousClauseGNN with:
    - 5-level hierarchical message passing (Symbol → Term → Literal → Clause → Proof)
    - Cross-level attention for direct non-adjacent communication
    - Goal-directed attention mechanisms
    - Incremental embedding updates
    - Backward compatibility with existing EmbeddingProvider protocol

    Architecture Flow:
    1. Input projection (same as base GNN)
    2. Multiple rounds of hierarchical message passing:
       a. Intra-level message passing within each level
       b. Inter-level message passing between adjacent levels
       c. Cross-level attention between non-adjacent levels
    3. Goal-directed attention (if goal context provided)
    4. Final aggregation and projection (maintains compatibility)
    """

    def __init__(self, config: HierarchicalGNNConfig):
        super().__init__()
        self.config = config

        # Backward compatibility: embed base GNN for fallback
        self.base_gnn = HeterogeneousClauseGNN(config.base_config)

        # Input projections (reuse from base GNN)
        self.input_projections = self.base_gnn.input_projections
        self.symbol_embedding = self.base_gnn.symbol_embedding
        self.symbol_combine = self.base_gnn.symbol_combine

        # Hierarchical message passing modules
        self._build_hierarchical_layers()

        # Goal-directed components
        if config.goal_attention_enabled:
            self._build_goal_components()

        # Distance computation
        self._build_distance_computer()

        # Output projection (maintains compatibility)
        self.output_projection = self.base_gnn.output_projection

        # Incremental update components (lazy initialization)
        self._incremental_updater = None

    def _build_hierarchical_layers(self):
        """Build hierarchical message passing layers."""
        from .message_passing import IntraLevelMP, InterLevelMP, CrossLevelAttention

        # Intra-level message passing
        self.intra_level_mp = nn.ModuleDict({
            level.name: IntraLevelMP(level, self.config.hidden_dim)
            for level in HierarchyLevel
        })

        # Inter-level message passing
        self.inter_level_mp = nn.ModuleDict()
        for i in range(len(HierarchyLevel) - 1):
            lower = HierarchyLevel(i)
            upper = HierarchyLevel(i + 1)
            key = f"{lower.name}_to_{upper.name}"
            self.inter_level_mp[key] = InterLevelMP(
                self.config.hidden_dim, self.config.hidden_dim
            )

        # Cross-level attention
        if self.config.cross_level_enabled:
            self.cross_attention = CrossLevelAttention(
                list(HierarchyLevel), self.config.hidden_dim, self.config.cross_level_heads
            )

    def _build_goal_components(self):
        """Build goal-directed components."""
        from .goals import GoalEncoder, GoalDirectedAttention

        self.goal_encoder = GoalEncoder(self.config.goal_embedding_dim)
        self.goal_attention = GoalDirectedAttention(
            self.config.hidden_dim, self.config.goal_embedding_dim
        )

    def _build_distance_computer(self):
        """Build distance computation module."""
        from .goals import DistanceComputer

        self.distance_computer = DistanceComputer(
            self.config.embedding_dim, self.config.distance_metric
        )

    @property
    def incremental_updater(self):
        """Lazy initialization of incremental updater."""
        if self._incremental_updater is None and self.config.incremental_enabled:
            from .incremental import IncrementalUpdater
            self._incremental_updater = IncrementalUpdater(self.config)
        return self._incremental_updater

    def forward(self, data: HeteroData, goal_context: Optional[torch.Tensor] = None,
                use_hierarchical: bool = True) -> torch.Tensor:
        """Forward pass with optional hierarchical processing.

        Args:
            data: HeteroData graph from clause_to_heterograph
            goal_context: Optional goal context for goal-directed attention
            use_hierarchical: If False, falls back to base GNN (for compatibility)

        Returns:
            Clause embeddings of shape (num_clauses, embedding_dim)
        """
        # Fallback to base GNN if hierarchical features are disabled
        if not use_hierarchical:
            return self.base_gnn.forward(data)

        # Step 1: Input projections (same as base GNN)
        level_embeddings = self._project_inputs(data)

        # Step 2: Multiple rounds of hierarchical message passing
        for round_idx in range(self.config.inter_level_rounds):
            # Intra-level message passing
            level_embeddings = self._intra_level_propagation(level_embeddings, data)

            # Inter-level message passing
            level_embeddings = self._inter_level_propagation(level_embeddings, data)

            # Cross-level attention
            if self.config.cross_level_enabled:
                level_embeddings = self._cross_level_attention(level_embeddings, data)

        # Step 3: Goal-directed attention (if available)
        if goal_context is not None and self.config.goal_attention_enabled:
            level_embeddings = self.goal_attention(level_embeddings, goal_context)

        # Step 4: Final clause-level aggregation and projection
        clause_embeddings = level_embeddings.get(HierarchyLevel.CLAUSE)
        if clause_embeddings is None:
            # Fallback if no clause nodes
            return torch.zeros(1, self.config.embedding_dim, device=self._device())

        return self.output_projection(clause_embeddings)

    def _project_inputs(self, data: HeteroData) -> Dict[HierarchyLevel, torch.Tensor]:
        """Project input node features to hidden dimension."""
        level_embeddings = {}

        # Use base GNN projection logic
        for level in HierarchyLevel:
            level_features = []

            for node_type in level.node_types:
                if node_type in data.node_types:
                    store = data[node_type]
                    if store.num_nodes > 0 and hasattr(store, 'x') and store.x is not None:
                        # Project using existing input projections
                        if node_type in self.input_projections:
                            projected = torch.relu(self.input_projections[node_type](store.x))
                            level_features.append(projected)

                        # Special handling for symbols (with learned embeddings)
                        if node_type == NodeType.SYMBOL.value and projected is not None:
                            raw_features = store.x
                            sym_ids = raw_features[:, 0].long().clamp(
                                0, self.config.base_config.symbol_vocab_size - 1
                            )
                            sym_embed = self.symbol_embedding(sym_ids)
                            enhanced = torch.relu(
                                self.symbol_combine(torch.cat([projected, sym_embed], dim=-1))
                            )
                            level_features[-1] = enhanced

            # Combine features from all node types at this level
            if level_features:
                if len(level_features) == 1:
                    level_embeddings[level] = level_features[0]
                else:
                    # Concatenate different node types at the same level
                    level_embeddings[level] = torch.cat(level_features, dim=0)

        return level_embeddings

    def _intra_level_propagation(self, level_embeddings: Dict[HierarchyLevel, torch.Tensor],
                               data: HeteroData) -> Dict[HierarchyLevel, torch.Tensor]:
        """Perform intra-level message passing."""
        updated = {}

        for level, x in level_embeddings.items():
            if level.name in self.intra_level_mp:
                edge_index = self._get_intra_level_edges(data, level)
                updated[level] = self.intra_level_mp[level.name](x, edge_index)
            else:
                updated[level] = x

        return updated

    def _inter_level_propagation(self, level_embeddings: Dict[HierarchyLevel, torch.Tensor],
                               data: HeteroData) -> Dict[HierarchyLevel, torch.Tensor]:
        """Perform inter-level message passing between adjacent levels."""
        updated = level_embeddings.copy()

        for i in range(len(HierarchyLevel) - 1):
            lower = HierarchyLevel(i)
            upper = HierarchyLevel(i + 1)

            if lower in updated and upper in updated:
                key = f"{lower.name}_to_{upper.name}"
                if key in self.inter_level_mp:
                    # Get hierarchical edge indices
                    bottom_up_idx, top_down_idx = self._get_inter_level_edges(data, lower, upper)

                    # Perform bidirectional message passing
                    updated_lower, updated_upper = self.inter_level_mp[key](
                        updated[lower], updated[upper], bottom_up_idx, top_down_idx
                    )

                    updated[lower] = updated_lower
                    updated[upper] = updated_upper

        return updated

    def _cross_level_attention(self, level_embeddings: Dict[HierarchyLevel, torch.Tensor],
                             data: HeteroData) -> Dict[HierarchyLevel, torch.Tensor]:
        """Perform cross-level attention between non-adjacent levels."""
        if not hasattr(self, 'cross_attention'):
            return level_embeddings

        cross_indices = self._build_cross_level_indices(data)
        return self.cross_attention(level_embeddings, cross_indices)

    def _get_intra_level_edges(self, data: HeteroData, level: HierarchyLevel) -> torch.Tensor:
        """Get edge indices for intra-level message passing."""
        # This is a placeholder - actual implementation would extract
        # relevant edges for nodes at the specified hierarchy level
        edges = []
        for edge_type in data.edge_types:
            # Include edges that connect nodes within the same level
            src_type, relation, dst_type = edge_type
            if (src_type in level.node_types and dst_type in level.node_types):
                edges.append(data[edge_type].edge_index)

        if edges:
            return torch.cat(edges, dim=1)
        else:
            return torch.empty(2, 0, dtype=torch.long, device=self._device())

    def _get_inter_level_edges(self, data: HeteroData,
                             lower: HierarchyLevel, upper: HierarchyLevel) -> tuple[torch.Tensor, torch.Tensor]:
        """Get edge indices for inter-level message passing."""
        # This is a placeholder - actual implementation would build
        # hierarchical connections between levels
        bottom_up = torch.empty(0, dtype=torch.long, device=self._device())
        top_down = torch.empty(0, dtype=torch.long, device=self._device())
        return bottom_up, top_down

    def _build_cross_level_indices(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Build index mappings for cross-level attention."""
        # This is a placeholder - actual implementation would build
        # mappings between non-adjacent hierarchy levels
        return {}

    def _device(self) -> torch.device:
        """Get the device of model parameters."""
        return next(self.parameters()).device

    def embed_clause(self, data: HeteroData, goal_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convenience method for inference (no gradients)."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            result = self.forward(data, goal_context)
        if was_training:
            self.train()
        return result.detach()

    def compute_goal_distance(self, clause_emb: torch.Tensor,
                            goal_emb: torch.Tensor) -> torch.Tensor:
        """Compute goal-directed distance between clause and goal embeddings."""
        return self.distance_computer(clause_emb, goal_emb)

    def incremental_update(self, new_clauses, context):
        """Incrementally update embeddings for new clauses."""
        if self.incremental_updater is not None:
            return self.incremental_updater.update(new_clauses, context)
        else:
            # Fallback to full computation
            from pyladr.ml.graph.clause_graph import batch_clauses_to_heterograph
            graphs = batch_clauses_to_heterograph(new_clauses)
            if graphs:
                data = graphs[0]  # Simplified for placeholder
                return self.forward(data)
            return torch.empty(0, self.config.embedding_dim)

    def get_hierarchical_embedding(self, data: HeteroData,
                                 level: HierarchyLevel) -> torch.Tensor:
        """Get embeddings at a specific hierarchy level."""
        level_embeddings = self._project_inputs(data)

        # Run partial forward pass up to the specified level
        for round_idx in range(self.config.inter_level_rounds):
            level_embeddings = self._intra_level_propagation(level_embeddings, data)
            level_embeddings = self._inter_level_propagation(level_embeddings, data)

            if level.value <= round_idx:
                break

        return level_embeddings.get(level, torch.empty(0, self.config.hidden_dim))