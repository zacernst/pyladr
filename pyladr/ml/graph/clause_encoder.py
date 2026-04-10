"""Heterogeneous Graph Neural Network for clause embeddings.

Implements a multi-layer heterogeneous GNN that operates on the clause graphs
produced by clause_graph.py. Produces fixed-dimensional clause embeddings
suitable for downstream tasks (selection scoring, inference guidance).

Architecture:
  1. Per-type input projections (different feature dims → hidden dim)
  2. N layers of heterogeneous message passing (HeteroConv with SAGEConv)
  3. Global mean pooling over clause-level aggregation
  4. Projection head for task-specific outputs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool

from pyladr.ml.graph.clause_graph import EdgeType, NodeType


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GNNConfig:
    """Hyperparameter configuration for the clause GNN.

    Attributes:
        hidden_dim: Hidden dimension for all GNN layers.
        embedding_dim: Output clause embedding dimension.
        num_layers: Number of heterogeneous message-passing layers.
        dropout: Dropout probability applied after each layer.
        symbol_vocab_size: Size of learnable symbol embedding table.
        symbol_embed_dim: Dimension of symbol embeddings (before projection).
        node_feature_dims: Input feature dimensions per node type.
            Defaults match clause_graph.py feature extractors.
    """

    hidden_dim: int = 256
    embedding_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    symbol_vocab_size: int = 10000
    symbol_embed_dim: int = 64
    node_feature_dims: dict[str, int] = field(default_factory=lambda: {
        NodeType.CLAUSE.value: 7,
        NodeType.LITERAL.value: 3,
        NodeType.TERM.value: 8,
        NodeType.SYMBOL.value: 6,
        NodeType.VARIABLE.value: 1,
    })


_DEFAULT_CONFIG = GNNConfig()

# Forward edge types (from clause_graph.py)
_FORWARD_EDGE_TYPES: list[tuple[str, str, str]] = [
    (NodeType.CLAUSE.value, EdgeType.CONTAINS_LITERAL.value, NodeType.LITERAL.value),
    (NodeType.LITERAL.value, EdgeType.HAS_ATOM.value, NodeType.TERM.value),
    (NodeType.TERM.value, EdgeType.HAS_ARG.value, NodeType.TERM.value),
    (NodeType.TERM.value, EdgeType.SYMBOL_OF.value, NodeType.SYMBOL.value),
    (NodeType.VARIABLE.value, EdgeType.VAR_OCCURRENCE.value, NodeType.TERM.value),
    (NodeType.VARIABLE.value, EdgeType.SHARED_VARIABLE.value, NodeType.VARIABLE.value),
]

# Reverse edges allow message passing back up the hierarchy
# (e.g., literal→clause so clause nodes receive aggregated info)
_REVERSE_EDGE_TYPES: list[tuple[str, str, str]] = [
    (NodeType.LITERAL.value, "rev_contains_literal", NodeType.CLAUSE.value),
    (NodeType.TERM.value, "rev_has_atom", NodeType.LITERAL.value),
    (NodeType.TERM.value, "rev_has_arg", NodeType.TERM.value),
    (NodeType.SYMBOL.value, "rev_symbol_of", NodeType.TERM.value),
    (NodeType.TERM.value, "rev_var_occurrence", NodeType.VARIABLE.value),
]

_ALL_EDGE_TYPES = _FORWARD_EDGE_TYPES + _REVERSE_EDGE_TYPES


# ── Model ──────────────────────────────────────────────────────────────────


class HeterogeneousClauseGNN(nn.Module):
    """Multi-layer heterogeneous GNN producing clause embeddings.

    Processes the heterogeneous graphs from clause_to_heterograph() and
    outputs a fixed-dimensional embedding vector for each clause.

    The architecture:
      1. Per-type linear projections map varying input feature dims to hidden_dim
      2. Symbol embedding lookup enriches symbol nodes
      3. N HeteroConv layers with SAGEConv per edge type
      4. Global pooling aggregates clause-level representations
      5. A final projection maps to embedding_dim
    """

    def __init__(self, config: GNNConfig | None = None):
        super().__init__()
        self.config = config or _DEFAULT_CONFIG
        c = self.config

        # Per-type input projections
        self.input_projections = nn.ModuleDict()
        for node_type, feat_dim in c.node_feature_dims.items():
            self.input_projections[node_type] = nn.Linear(feat_dim, c.hidden_dim)

        # Symbol embedding lookup (learned)
        self.symbol_embedding = nn.Embedding(c.symbol_vocab_size, c.symbol_embed_dim)
        # Projection to combine symbol features + learned embedding
        self.symbol_combine = nn.Linear(c.hidden_dim + c.symbol_embed_dim, c.hidden_dim)

        # Heterogeneous conv layers
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(c.num_layers):
            conv_dict = {}
            for edge_triplet in _ALL_EDGE_TYPES:
                conv_dict[edge_triplet] = SAGEConv(
                    (c.hidden_dim, c.hidden_dim), c.hidden_dim
                )
            self.conv_layers.append(HeteroConv(conv_dict, aggr="sum"))

            # Per-type layer norms
            norms = nn.ModuleDict()
            for nt in NodeType:
                norms[nt.value] = nn.LayerNorm(c.hidden_dim)
            self.layer_norms.append(norms)

        self.dropout = nn.Dropout(c.dropout)

        # Final projection: hidden_dim → embedding_dim
        self.output_projection = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.ReLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, c.embedding_dim),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Compute clause embeddings from a heterogeneous graph.

        Args:
            data: A HeteroData (single or batched) from clause_to_heterograph.

        Returns:
            Tensor of shape (num_clauses, embedding_dim).
        """
        x_dict: dict[str, torch.Tensor] = {}

        # Step 1: Project input features to hidden_dim
        for node_type in NodeType:
            nt = node_type.value
            if nt not in data.node_types:
                continue
            store = data[nt]
            if store.num_nodes == 0 or not hasattr(store, "x") or store.x is None:
                continue
            x_dict[nt] = torch.relu(self.input_projections[nt](store.x))

        # Step 2: Enrich symbol nodes with learned embeddings
        if NodeType.SYMBOL.value in x_dict:
            sym_x = x_dict[NodeType.SYMBOL.value]
            # Use the first feature (capped symnum) as embedding index
            raw_features = data[NodeType.SYMBOL.value].x
            sym_ids = raw_features[:, 0].long().clamp(
                0, self.config.symbol_vocab_size - 1
            )
            sym_embed = self.symbol_embedding(sym_ids)
            x_dict[NodeType.SYMBOL.value] = torch.relu(
                self.symbol_combine(torch.cat([sym_x, sym_embed], dim=-1))
            )

        # Step 3: Build edge index dict with forward + reverse edges
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}

        # Map forward edge types to their reverse counterparts
        _REVERSE_MAP = {
            (s, e, d): (d, r, s)
            for (s, e, d), (d2, r, s2) in zip(
                _FORWARD_EDGE_TYPES, _REVERSE_EDGE_TYPES
            )
        }

        for edge_type in data.edge_types:
            fwd = tuple(edge_type)
            if fwd in {tuple(et) for et in _FORWARD_EDGE_TYPES}:
                ei = data[edge_type].edge_index
                edge_index_dict[fwd] = ei
                # Add reverse edge (flip src/dst)
                if fwd in _REVERSE_MAP:
                    rev_key = _REVERSE_MAP[fwd]
                    edge_index_dict[rev_key] = ei.flip(0)

        # Self-loops (shared_variable) — already bidirectional
        sv_key = (
            NodeType.VARIABLE.value,
            EdgeType.SHARED_VARIABLE.value,
            NodeType.VARIABLE.value,
        )
        if sv_key in {tuple(et) for et in data.edge_types}:
            edge_index_dict[sv_key] = data[sv_key].edge_index

        # Step 4: Message passing layers
        for layer_idx, conv in enumerate(self.conv_layers):
            if not edge_index_dict:
                break

            # Run heterogeneous convolution
            out_dict = conv(x_dict, edge_index_dict)

            # Apply layer norm, residual connection, and dropout
            norms = self.layer_norms[layer_idx]
            for nt, out in out_dict.items():
                if nt in x_dict:
                    out = norms[nt](out + x_dict[nt])  # residual + norm
                    out = self.dropout(out)
                    x_dict[nt] = out

        # Step 5: Global pooling — aggregate clause node representations
        clause_nt = NodeType.CLAUSE.value
        if clause_nt not in x_dict:
            # Fallback: return zeros if no clause nodes
            return torch.zeros(1, self.config.embedding_dim, device=self._device())

        clause_x = x_dict[clause_nt]

        # If batched data has a batch vector for clause nodes, use it
        if hasattr(data[clause_nt], "batch") and data[clause_nt].batch is not None:
            batch = data[clause_nt].batch
            pooled = global_mean_pool(clause_x, batch)
        else:
            # Single graph — clause_x is already the clause embeddings
            pooled = clause_x

        # Step 6: Project to final embedding dimension
        return self.output_projection(pooled)

    def _device(self) -> torch.device:
        """Get the device of model parameters."""
        return next(self.parameters()).device

    def embed_clause(self, data: HeteroData) -> torch.Tensor:
        """Convenience method: compute embeddings in eval mode, no grad.

        Args:
            data: HeteroData from clause_to_heterograph.

        Returns:
            Detached embedding tensor of shape (num_clauses, embedding_dim).
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            result = self.forward(data)
        if was_training:
            self.train()
        return result.detach()


# ── Projection heads for downstream tasks ──────────────────────────────────


class SelectionHead(nn.Module):
    """Scoring head for clause selection (given-clause algorithm).

    Takes clause embeddings and produces a scalar score indicating
    how promising the clause is for selection.
    """

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Score clause embeddings.

        Args:
            embeddings: (N, embedding_dim) clause embeddings.

        Returns:
            (N,) scalar scores.
        """
        return self.net(embeddings).squeeze(-1)


class InferenceGuidanceHead(nn.Module):
    """Scoring head for inference candidate prioritization.

    Takes a pair of clause embeddings and predicts how productive
    their inference (resolution/paramodulation) will be.
    """

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        # Input: concatenation of two clause embeddings + element-wise product
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, emb_a: torch.Tensor, emb_b: torch.Tensor
    ) -> torch.Tensor:
        """Score an inference pair.

        Args:
            emb_a: (N, embedding_dim) first clause embeddings.
            emb_b: (N, embedding_dim) second clause embeddings.

        Returns:
            (N,) probability scores in [0, 1].
        """
        combined = torch.cat([emb_a, emb_b, emb_a * emb_b], dim=-1)
        return self.net(combined).squeeze(-1)


# ── Model persistence ─────────────────────────────────────────────────────


def save_model(
    model: HeterogeneousClauseGNN,
    path: str | Path,
    config: GNNConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model weights and config to disk.

    Args:
        model: The GNN model to save.
        path: File path for the checkpoint.
        config: Config to save alongside weights. Uses model.config if None.
        metadata: Optional metadata (training stats, etc).
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config or model.config,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)


def load_model(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[HeterogeneousClauseGNN, dict[str, Any]]:
    """Load a saved model from disk.

    Args:
        path: Path to the saved checkpoint.
        device: Device to load the model onto.

    Returns:
        Tuple of (model, metadata dict).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = HeterogeneousClauseGNN(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint.get("metadata", {})
