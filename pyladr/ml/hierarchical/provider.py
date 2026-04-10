"""Enhanced embedding provider with hierarchical GNN support.

This module provides HierarchicalEmbeddingProvider that maintains full backward
compatibility with the existing EmbeddingProvider protocol while adding
hierarchical GNN features like goal-directed selection and incremental updates.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from pyladr.core.clause import Clause
from pyladr.core.symbol import SymbolTable
from pyladr.ml.embedding_provider import (
    EmbeddingProvider,
    EmbeddingProviderConfig,
    GNNEmbeddingProvider,
    NoOpEmbeddingProvider,
)
from pyladr.ml.graph.clause_graph import ClauseGraphConfig

from .architecture import HierarchicalClauseGNN, HierarchicalGNNConfig, HierarchyLevel

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HierarchicalEmbeddingProviderConfig:
    """Configuration for the hierarchical embedding provider.

    Extends EmbeddingProviderConfig with hierarchical-specific options while
    maintaining full backward compatibility.
    """

    # Base configuration (backward compatible)
    base_config: EmbeddingProviderConfig = field(default_factory=EmbeddingProviderConfig)

    # Hierarchical GNN configuration
    hierarchical_config: HierarchicalGNNConfig = field(default_factory=HierarchicalGNNConfig)

    # Feature flags
    use_hierarchical_features: bool = False
    fallback_to_base_on_error: bool = True

    # Goal-directed features
    enable_goal_directed_selection: bool = False
    goal_context_size: int = 50
    goal_distance_threshold: float = 0.7
    goal_recompute_interval: int = 100

    # Incremental update features
    enable_incremental_updates: bool = False
    incremental_cache_size: int = 10000
    staleness_threshold: float = 0.1

    # Performance tuning
    hierarchical_batch_size: int = 16
    max_hierarchy_depth: int = 5


class HierarchicalEmbeddingProvider:
    """Enhanced embedding provider with hierarchical GNN support.

    Maintains full backward compatibility with EmbeddingProvider protocol
    while adding hierarchical features:
    - Multi-level embeddings at different hierarchy levels
    - Goal-directed embedding computation
    - Incremental updates during search
    - Cross-level attention mechanisms

    The provider automatically falls back to the base GNNEmbeddingProvider
    when hierarchical features are disabled or encounter errors.

    Thread Safety:
    - Inherits thread-safety from base GNNEmbeddingProvider
    - Adds additional locks for hierarchical state management
    - Safe model hot-swapping with hierarchical features
    """

    def __init__(
        self,
        hierarchical_model: HierarchicalClauseGNN,
        config: HierarchicalEmbeddingProviderConfig,
        symbol_table: Optional[SymbolTable] = None,
    ):
        self.config = config
        self.hierarchical_model = hierarchical_model
        self.symbol_table = symbol_table

        # Base provider for backward compatibility and fallback
        self.base_provider = GNNEmbeddingProvider.create(
            symbol_table=symbol_table,
            config=config.base_config,
            gnn_config=config.hierarchical_config.base_config,
        )

        # Hierarchical-specific state
        self._hierarchical_lock = threading.RLock()
        self._goal_context: Optional[torch.Tensor] = None
        self._goal_context_version = 0

        # Incremental update state
        if config.enable_incremental_updates:
            from .incremental import IncrementalContext
            self.incremental_context = IncrementalContext()
        else:
            self.incremental_context = None

        # Performance tracking
        self._hierarchical_calls = 0
        self._fallback_calls = 0

    # ─── EmbeddingProvider Protocol (Backward Compatible) ────────────────

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of produced embeddings (backward compatible)."""
        return self.base_provider.embedding_dim

    def get_embedding(self, clause: Clause) -> List[float] | None:
        """Get embedding for a single clause (backward compatible).

        Falls back to base provider on any hierarchical processing error
        to ensure system reliability.
        """
        if not self.config.use_hierarchical_features:
            return self.base_provider.get_embedding(clause)

        try:
            result = self._get_hierarchical_embedding(clause)
            if result is not None:
                self._hierarchical_calls += 1
                return result
        except Exception:
            logger.debug(
                "Hierarchical embedding failed for clause %d, falling back to base",
                clause.id, exc_info=True
            )

        # Fallback to base provider
        self._fallback_calls += 1
        return self.base_provider.get_embedding(clause)

    def get_embeddings_batch(
        self, clauses: List[Clause]
    ) -> List[List[float] | None]:
        """Get embeddings for a batch of clauses (backward compatible)."""
        if not self.config.use_hierarchical_features:
            return self.base_provider.get_embeddings_batch(clauses)

        try:
            result = self._get_hierarchical_embeddings_batch(clauses)
            if result is not None:
                self._hierarchical_calls += len(clauses)
                return result
        except Exception:
            logger.debug(
                "Hierarchical batch embedding failed for %d clauses, falling back",
                len(clauses), exc_info=True
            )

        # Fallback to base provider
        self._fallback_calls += len(clauses)
        return self.base_provider.get_embeddings_batch(clauses)

    # ─── Enhanced Hierarchical Methods ─────────────────────────────────────

    def get_hierarchical_embedding(
        self,
        clause: Clause,
        level: HierarchyLevel = HierarchyLevel.CLAUSE,
        goal_context: Optional[List[Clause]] = None,
    ) -> List[float] | None:
        """Get embedding at specific hierarchy level with optional goal context.

        Args:
            clause: The clause to embed
            level: Hierarchy level for embedding extraction
            goal_context: Optional goal clauses for goal-directed embedding

        Returns:
            Embedding vector at the specified level, or None on failure
        """
        try:
            with self._hierarchical_lock:
                # Build graph representation
                from pyladr.ml.graph.clause_graph import clause_to_heterograph
                graph = clause_to_heterograph(clause, self.symbol_table)

                # Encode goal context if provided
                goal_tensor = None
                if goal_context and self.config.enable_goal_directed_selection:
                    goal_tensor = self._encode_goal_context(goal_context)

                # Get hierarchical embedding
                embedding = self.hierarchical_model.get_hierarchical_embedding(
                    graph, level
                )

                # Apply goal-directed attention if available
                if goal_tensor is not None and hasattr(self.hierarchical_model, 'goal_attention'):
                    level_embeddings = {level: embedding}
                    updated = self.hierarchical_model.goal_attention(level_embeddings, goal_tensor)
                    embedding = updated[level]

                return embedding.flatten().tolist()

        except Exception:
            logger.debug(
                "Hierarchical embedding failed for clause %d at level %s",
                clause.id, level.name, exc_info=True
            )
            return None

    def get_goal_directed_embedding(
        self,
        clause: Clause,
        goal_clauses: List[Clause],
        distance_threshold: Optional[float] = None,
    ) -> List[float] | None:
        """Get goal-directed embedding with distance-based filtering.

        Args:
            clause: The clause to embed
            goal_clauses: Goal clauses for direction
            distance_threshold: Optional distance threshold for filtering

        Returns:
            Goal-directed embedding vector, or None if too distant from goals
        """
        if not self.config.enable_goal_directed_selection:
            return self.get_embedding(clause)

        try:
            # Get base embedding
            base_embedding = self.get_hierarchical_embedding(clause, goal_context=goal_clauses)
            if base_embedding is None:
                return None

            # Compute goal distance if threshold specified
            if distance_threshold is not None:
                distance = self.compute_goal_distance(clause, goal_clauses)
                if distance > distance_threshold:
                    return None  # Too distant from goals

            return base_embedding

        except Exception:
            logger.debug(
                "Goal-directed embedding failed for clause %d", clause.id, exc_info=True
            )
            return self.get_embedding(clause)  # Fallback

    def compute_goal_distance(self, clause: Clause, goal_clauses: List[Clause]) -> float:
        """Compute distance between clause and goal context.

        Args:
            clause: The clause to measure
            goal_clauses: Goal context clauses

        Returns:
            Distance score (lower = more relevant to goals)
        """
        try:
            with self._hierarchical_lock:
                # Get clause embedding
                clause_emb = self.get_embedding(clause)
                if clause_emb is None:
                    return float('inf')

                # Encode goal context
                goal_emb = self._encode_goal_context(goal_clauses)
                if goal_emb is None:
                    return float('inf')

                # Compute distance using hierarchical model
                clause_tensor = torch.tensor(clause_emb, dtype=torch.float32)
                distance = self.hierarchical_model.compute_goal_distance(
                    clause_tensor, goal_emb
                )

                return distance.item()

        except Exception:
            logger.debug(
                "Goal distance computation failed for clause %d", clause.id, exc_info=True
            )
            return float('inf')

    def incremental_update(self, new_clauses: List[Clause]) -> None:
        """Trigger incremental update for new clauses.

        Args:
            new_clauses: Newly derived clauses to update embeddings for
        """
        if not self.config.enable_incremental_updates or self.incremental_context is None:
            return

        try:
            with self._hierarchical_lock:
                self.hierarchical_model.incremental_update(
                    new_clauses, self.incremental_context
                )
        except Exception:
            logger.debug(
                "Incremental update failed for %d clauses", len(new_clauses), exc_info=True
            )

    def set_goal_context(self, goal_clauses: List[Clause]) -> None:
        """Set global goal context for subsequent embeddings.

        Args:
            goal_clauses: Goal clauses to use for goal-directed attention
        """
        if not self.config.enable_goal_directed_selection:
            return

        try:
            with self._hierarchical_lock:
                self._goal_context = self._encode_goal_context(goal_clauses)
                self._goal_context_version += 1
        except Exception:
            logger.debug("Failed to set goal context", exc_info=True)

    def clear_goal_context(self) -> None:
        """Clear the current goal context."""
        with self._hierarchical_lock:
            self._goal_context = None
            self._goal_context_version += 1

    # ─── Internal Implementation ─────────────────────────────────────────────

    def _get_hierarchical_embedding(self, clause: Clause) -> List[float] | None:
        """Internal hierarchical embedding computation."""
        from pyladr.ml.graph.clause_graph import clause_to_heterograph

        # Build graph
        graph = clause_to_heterograph(clause, self.symbol_table)

        # Use current goal context if available
        goal_context = None
        if self._goal_context is not None:
            goal_context = self._goal_context

        # Compute embedding
        with self._hierarchical_lock:
            embedding = self.hierarchical_model.embed_clause(graph, goal_context)

        return embedding.flatten().tolist()

    def _get_hierarchical_embeddings_batch(
        self, clauses: List[Clause]
    ) -> List[List[float] | None]:
        """Internal batch hierarchical embedding computation."""
        from pyladr.ml.graph.clause_graph import batch_clauses_to_heterograph
        from torch_geometric.data import Batch

        # Build batch of graphs
        graphs = batch_clauses_to_heterograph(clauses, self.symbol_table)
        if not graphs:
            return [None] * len(clauses)

        # Batch graphs
        batch = Batch.from_data_list(graphs)

        # Use current goal context if available
        goal_context = None
        if self._goal_context is not None:
            goal_context = self._goal_context

        # Compute embeddings
        with self._hierarchical_lock:
            embeddings = self.hierarchical_model.embed_clause(batch, goal_context)

        # Convert to list format
        return [row.tolist() for row in embeddings.unbind(0)]

    def _encode_goal_context(self, goal_clauses: List[Clause]) -> torch.Tensor | None:
        """Encode goal clauses into goal context tensor."""
        if not goal_clauses or not hasattr(self.hierarchical_model, 'goal_encoder'):
            return None

        try:
            # Limit goal context size for performance
            limited_goals = goal_clauses[:self.config.goal_context_size]

            # Use goal encoder from hierarchical model
            return self.hierarchical_model.goal_encoder(limited_goals, [])

        except Exception:
            logger.debug("Goal context encoding failed", exc_info=True)
            return None

    # ─── Model Management (extends base provider) ──────────────────────────

    def swap_hierarchical_model(self, new_model: HierarchicalClauseGNN) -> int:
        """Replace the hierarchical model and invalidate caches."""
        with self._hierarchical_lock:
            self.hierarchical_model = new_model

            # Clear incremental context
            if self.incremental_context is not None:
                self.incremental_context = type(self.incremental_context)()

        # Also update base model if compatible
        if hasattr(new_model, 'base_gnn'):
            return self.base_provider.swap_model(new_model.base_gnn)

        return 0

    @property
    def stats(self) -> Dict:
        """Combined statistics from base and hierarchical components."""
        base_stats = self.base_provider.stats
        hierarchical_stats = {
            'hierarchical_calls': self._hierarchical_calls,
            'fallback_calls': self._fallback_calls,
            'goal_context_version': self._goal_context_version,
            'hierarchical_enabled': self.config.use_hierarchical_features,
        }

        if self.incremental_context is not None:
            hierarchical_stats['incremental_cache_size'] = len(
                self.incremental_context.cached_embeddings
            )

        return {**base_stats, **hierarchical_stats}

    # ─── Compatibility Delegation ────────────────────────────────────────────

    def __getattr__(self, name):
        """Delegate unknown attributes to base provider for compatibility."""
        return getattr(self.base_provider, name)


# ─── Factory Functions ─────────────────────────────────────────────────────

def create_hierarchical_embedding_provider(
    symbol_table: Optional[SymbolTable] = None,
    config: Optional[HierarchicalEmbeddingProviderConfig] = None,
) -> EmbeddingProvider:
    """Create hierarchical embedding provider with automatic fallback.

    Returns HierarchicalEmbeddingProvider if all dependencies are available,
    otherwise falls back to standard GNNEmbeddingProvider or NoOpEmbeddingProvider.

    Args:
        symbol_table: Symbol table for graph construction
        config: Hierarchical configuration

    Returns:
        Best available embedding provider
    """
    cfg = config or HierarchicalEmbeddingProviderConfig()

    try:
        # Check if hierarchical features are requested and available
        if not cfg.use_hierarchical_features:
            # User explicitly disabled hierarchical features
            from pyladr.ml.embedding_provider import create_embedding_provider
            return create_embedding_provider(symbol_table, cfg.base_config)

        # Try to create hierarchical model
        hierarchical_model = HierarchicalClauseGNN(cfg.hierarchical_config)

        return HierarchicalEmbeddingProvider(
            hierarchical_model=hierarchical_model,
            config=cfg,
            symbol_table=symbol_table,
        )

    except ImportError:
        logger.info(
            "Hierarchical ML dependencies not available, falling back to base provider"
        )
        from pyladr.ml.embedding_provider import create_embedding_provider
        return create_embedding_provider(symbol_table, cfg.base_config)

    except Exception:
        logger.warning(
            "Failed to create HierarchicalEmbeddingProvider, falling back to base",
            exc_info=True
        )
        from pyladr.ml.embedding_provider import create_embedding_provider
        return create_embedding_provider(symbol_table, cfg.base_config)


# ─── Backward Compatibility Alias ───────────────────────────────────────────

# For drop-in replacement in existing code
HierarchicalEmbeddingProvider.create = create_hierarchical_embedding_provider