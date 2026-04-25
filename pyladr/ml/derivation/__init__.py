"""Derivation history embeddings for inference chain encoding.

This package provides temporal and structural features derived from clause
justification chains, enabling the GNN to reason about *how* a clause was
derived — not just its syntactic structure.

Key Components:
- DerivationFeatureExtractor: Extracts fixed-size derivation features from
  justification chains (depth, rule distribution, ancestry statistics).
- InferenceChainEncoder: Learned sequence encoder that maps variable-length
  inference chains to fixed-size embeddings using JustType embeddings and
  positional encoding.
- DerivationContext: Maintains a clause-id → derivation-info mapping that
  tracks the full derivation DAG during search.

Design Principles:
- Opt-in: disabled by default; no impact when not used.
- Protocol-compatible: produces features that augment CLAUSE node features
  in the existing graph builder without modifying core structures.
- Thread-safe: DerivationContext uses read-write locking for concurrent
  access during search.
"""

from __future__ import annotations

from .derivation_features import (
    DerivationFeatureExtractor,
    DerivationFeatureConfig,
    DerivationFeatures,
)
from .inference_chain_encoder import (
    InferenceChainEncoder,
    InferenceChainConfig,
)
from .derivation_context import (
    DerivationContext,
    DerivationInfo,
)

__all__ = [
    "DerivationFeatureExtractor",
    "DerivationFeatureConfig",
    "DerivationFeatures",
    "InferenceChainEncoder",
    "InferenceChainConfig",
    "DerivationContext",
    "DerivationInfo",
]

# Integration bridges (require torch)
try:
    from .attention_bridge import (
        DerivationAttentionAdapter,
        ChainEnhancedAttentionAdapter,
        TemporalMetadata,
    )
    from .graph_augmentation import (
        clause_to_heterograph_augmented,
        batch_clauses_to_heterograph_augmented,
        augmented_gnn_config,
        DerivationGraphConfig,
        AUGMENTED_CLAUSE_FEATURE_DIM,
    )
    __all__ += [
        "DerivationAttentionAdapter",
        "ChainEnhancedAttentionAdapter",
        "TemporalMetadata",
        "clause_to_heterograph_augmented",
        "batch_clauses_to_heterograph_augmented",
        "augmented_gnn_config",
        "DerivationGraphConfig",
        "AUGMENTED_CLAUSE_FEATURE_DIM",
    ]
except ImportError:
    pass  # torch/torch_geometric not available
