"""CLAUSE node feature augmentation with derivation history.

Extends the existing 7-dimensional CLAUSE node features in the graph builder
with 13-dimensional derivation features, producing a 20-dimensional CLAUSE
feature vector.  This gives the GNN (flat or hierarchical) access to
temporal/ancestry information during message passing.

Design:
  - Non-invasive: wraps the existing graph builder, never modifies it.
  - Opt-in: controlled by DerivationGraphConfig.enabled (default False).
  - Compatible: the augmented feature vector is a strict superset of the
    original 7-dim features.  Models trained without derivation features
    simply see zeros in positions [7:20] when the feature is disabled.

Integration with hierarchical message passing (Task #2):
  Marcus's HierarchicalClauseGNN uses GNNConfig.node_feature_dims to
  determine input projection sizes.  When derivation augmentation is
  enabled, the CLAUSE feature dim changes from 7 to 20 — the config
  must be updated accordingly (see augmented_gnn_config()).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch_geometric.data import HeteroData

if TYPE_CHECKING:
    from pyladr.core.clause import Clause
    from pyladr.core.symbol import SymbolTable

from pyladr.ml.graph.clause_graph import (
    ClauseGraphConfig,
    NodeType,
    batch_clauses_to_heterograph,
    clause_to_heterograph,
)

from .derivation_context import DerivationContext
from .derivation_features import (
    DERIVATION_FEATURE_DIM,
    DerivationFeatureConfig,
    DerivationFeatureExtractor,
)


@dataclass(frozen=True, slots=True)
class DerivationGraphConfig:
    """Configuration for derivation-augmented graph construction.

    Attributes:
        enabled: Master switch for derivation augmentation.
        graph_config: Base graph construction configuration.
        feature_config: Derivation feature extraction configuration.
    """

    enabled: bool = False
    graph_config: ClauseGraphConfig | None = None
    feature_config: DerivationFeatureConfig | None = None


# Total CLAUSE feature dimension when augmented
AUGMENTED_CLAUSE_FEATURE_DIM = 7 + DERIVATION_FEATURE_DIM  # 7 + 13 = 20


def clause_to_heterograph_augmented(
    clause: Clause,
    context: DerivationContext,
    symbol_table: SymbolTable | None = None,
    config: DerivationGraphConfig | None = None,
) -> HeteroData:
    """Convert a clause to a HeteroData graph with augmented CLAUSE features.

    Builds the standard heterogeneous graph, then concatenates the 13-dim
    derivation features to each CLAUSE node's feature vector.

    Args:
        clause: The clause to convert.
        context: DerivationContext with registered derivation info.
        symbol_table: Optional symbol table for symbol metadata.
        config: Augmentation configuration.

    Returns:
        HeteroData with CLAUSE node features of dimension 20 (7 + 13).
    """
    cfg = config or DerivationGraphConfig(enabled=True)
    graph_cfg = cfg.graph_config
    feat_cfg = cfg.feature_config

    # Build base graph
    data = clause_to_heterograph(clause, symbol_table, graph_cfg)

    if not cfg.enabled:
        return data

    # Extract derivation features
    extractor = DerivationFeatureExtractor(
        context, feat_cfg or DerivationFeatureConfig()
    )
    deriv_feats = extractor.extract(clause)
    deriv_tensor = torch.tensor(
        [deriv_feats.features], dtype=torch.float32
    )  # (1, 13)

    # Augment CLAUSE node features
    clause_nt = NodeType.CLAUSE.value
    if clause_nt in data.node_types and data[clause_nt].num_nodes > 0:
        existing = data[clause_nt].x  # (num_clause_nodes, 7)
        # Expand deriv_tensor to match clause node count (usually 1)
        num_nodes = existing.shape[0]
        deriv_expanded = deriv_tensor.expand(num_nodes, -1)
        data[clause_nt].x = torch.cat([existing, deriv_expanded], dim=-1)

    return data


def batch_clauses_to_heterograph_augmented(
    clauses: list[Clause],
    context: DerivationContext,
    symbol_table: SymbolTable | None = None,
    config: DerivationGraphConfig | None = None,
) -> list[HeteroData]:
    """Convert multiple clauses to individual augmented HeteroData graphs.

    Args:
        clauses: List of clauses.
        context: DerivationContext with registered derivation info.
        symbol_table: Optional symbol table.
        config: Augmentation configuration.

    Returns:
        List of HeteroData graphs with augmented CLAUSE features.
    """
    cfg = config or DerivationGraphConfig(enabled=True)

    if not cfg.enabled:
        return batch_clauses_to_heterograph(clauses, symbol_table, cfg.graph_config)

    # Build base graphs
    graphs = batch_clauses_to_heterograph(clauses, symbol_table, cfg.graph_config)

    # Augment each graph with derivation features
    extractor = DerivationFeatureExtractor(
        context, cfg.feature_config or DerivationFeatureConfig()
    )

    for graph, clause in zip(graphs, clauses):
        deriv_feats = extractor.extract(clause)
        deriv_tensor = torch.tensor(
            [deriv_feats.features], dtype=torch.float32
        )

        clause_nt = NodeType.CLAUSE.value
        if clause_nt in graph.node_types and graph[clause_nt].num_nodes > 0:
            existing = graph[clause_nt].x
            num_nodes = existing.shape[0]
            deriv_expanded = deriv_tensor.expand(num_nodes, -1)
            graph[clause_nt].x = torch.cat([existing, deriv_expanded], dim=-1)

    return graphs


def augmented_gnn_config(
    base_node_feature_dims: dict[str, int] | None = None,
) -> dict[str, int]:
    """Return node_feature_dims dict with augmented CLAUSE dimension.

    Use this when constructing a GNNConfig or HierarchicalGNNConfig to
    account for the additional 13 derivation features on CLAUSE nodes.

    Args:
        base_node_feature_dims: Existing dims dict to modify. If None,
            uses the standard defaults.

    Returns:
        Updated dict with clause dim = 20 (7 + 13).

    Example::

        from pyladr.ml.graph.clause_encoder import GNNConfig
        from pyladr.ml.derivation.graph_augmentation import augmented_gnn_config

        config = GNNConfig(
            node_feature_dims=augmented_gnn_config(),
        )
    """
    from pyladr.ml.graph.clause_graph import NodeType

    defaults = {
        NodeType.CLAUSE.value: 7,
        NodeType.LITERAL.value: 3,
        NodeType.TERM.value: 8,
        NodeType.SYMBOL.value: 6,
        NodeType.VARIABLE.value: 1,
    }

    dims = dict(base_node_feature_dims) if base_node_feature_dims else defaults
    dims[NodeType.CLAUSE.value] = AUGMENTED_CLAUSE_FEATURE_DIM
    return dims
