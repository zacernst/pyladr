"""Hierarchical GNN architecture for PyLADR.

This package implements a 5-level hierarchical Graph Neural Network for
goal-directed clause selection in automated theorem proving.

Key Components:
- HierarchicalClauseGNN: Main GNN model with hierarchical message passing
- HierarchicalEmbeddingProvider: Enhanced embedding provider with hierarchical features
- GoalDirectedSelection: Selection mechanism with goal-oriented guidance
- IncrementalUpdater: Real-time embedding updates during search

Architecture Levels:
1. SYMBOL: Function/predicate symbols and variables
2. TERM: Complex terms and subterm structures
3. LITERAL: Signed atoms and equations
4. CLAUSE: Complete logical clauses
5. PROOF: Global proof context and goals

The architecture maintains full backward compatibility with the existing
EmbeddingProvider protocol while adding powerful hierarchical capabilities.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

# Core hierarchical components (always available)
from .architecture import (
    HierarchicalClauseGNN,
    HierarchicalGNNConfig,
    HierarchyLevel,
)

# Enhanced embedding provider (always available)
from .provider import (
    HierarchicalEmbeddingProvider,
    HierarchicalEmbeddingProviderConfig,
)

__all__ = [
    # Core architecture
    "HierarchicalClauseGNN",
    "HierarchicalGNNConfig",
    "HierarchyLevel",

    # Enhanced provider
    "HierarchicalEmbeddingProvider",
    "HierarchicalEmbeddingProviderConfig",
]

# Optional components — these modules are planned but not yet implemented.
# Import them conditionally so the package remains importable without them.

try:
    from .goals import (
        GoalEncoder,
        GoalDirectedAttention,
        DistanceComputer,
    )
    __all__ += ["GoalEncoder", "GoalDirectedAttention", "DistanceComputer"]
except ImportError:
    _logger.debug("hierarchical.goals not available (not yet implemented)")

try:
    from .message_passing import (
        IntraLevelMP,
        InterLevelMP,
        CrossLevelAttention,
    )
    __all__ += ["IntraLevelMP", "InterLevelMP", "CrossLevelAttention"]
except ImportError:
    _logger.debug("hierarchical.message_passing not available (not yet implemented)")

try:
    from .incremental import (
        IncrementalUpdater,
        IncrementalContext,
        StructuralChangeDetector,
    )
    __all__ += ["IncrementalUpdater", "IncrementalContext", "StructuralChangeDetector"]
except ImportError:
    _logger.debug("hierarchical.incremental not available (not yet implemented)")

try:
    from .selection import (
        HierarchicalSelection,
        HierarchicalSelectionConfig,
    )
    __all__ += ["HierarchicalSelection", "HierarchicalSelectionConfig"]
except ImportError:
    _logger.debug("hierarchical.selection not available (not yet implemented)")

try:
    from .factory import (
        create_hierarchical_embedding_provider,
        create_hierarchical_selection,
    )
    __all__ += ["create_hierarchical_embedding_provider", "create_hierarchical_selection"]
except ImportError:
    _logger.debug("hierarchical.factory not available (not yet implemented)")

# Version information
__version__ = "0.1.0"
__author__ = "PyLADR Hierarchical GNN Team"

# Feature flags for gradual rollout
FEATURES = {
    "hierarchical_message_passing": True,
    "cross_level_attention": True,
    "goal_directed_selection": True,
    "incremental_updates": True,
    "backward_compatibility": True,
}
