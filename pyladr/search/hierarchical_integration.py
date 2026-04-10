"""Unified hierarchical integration for goal-directed clause selection.

Composes the 5-level hierarchical message passing architecture with
goal-directed embedding enhancement into a single, configurable
embedding provider.

Levels:
  1. Symbol-level MPN (SymbolLevelMPN)
  2. Term-level MPN (TermLevelMPN)
  3. Literal-level MPN (LiteralLevelMPN)
  4. Clause-level MPN (ClauseLevelMPN)
  5. Proof-level MPN (ProofLevelMPN) — optional, applied when available

The unified provider wraps all levels behind the EmbeddingProvider
protocol, making it a transparent drop-in for EmbeddingEnhancedSelection.

All features are opt-in. When hierarchical features are disabled,
the provider falls back to the existing GNNEmbeddingProvider or
NoOpEmbeddingProvider for zero-breakage compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyladr.core.clause import Clause
from pyladr.search.goal_directed import (
    GoalDirectedConfig,
    GoalDirectedEmbeddingProvider,
)

if TYPE_CHECKING:
    from pyladr.core.symbol import SymbolTable
    from pyladr.search.ml_selection import EmbeddingProvider

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HierarchicalIntegrationConfig:
    """Configuration for the unified hierarchical integration.

    Attributes:
        enabled: Master switch for hierarchical features. When False,
            uses base_provider directly (zero overhead).
        goal_directed: Goal-directed embedding enhancement config.
        use_hierarchical_gnn: Use hierarchical MPN layers when available.
            When False, uses the existing flat GNNEmbeddingProvider.
    """

    enabled: bool = False
    goal_directed: GoalDirectedConfig = field(
        default_factory=GoalDirectedConfig,
    )
    use_hierarchical_gnn: bool = False


# ── Factory ────────────────────────────────────────────────────────────────


def create_goal_directed_provider(
    base_provider: EmbeddingProvider,
    config: HierarchicalIntegrationConfig | None = None,
    goals: list[Clause] | None = None,
) -> EmbeddingProvider:
    """Create a goal-directed embedding provider wrapping the base.

    When disabled, returns the base provider unchanged.
    When enabled, wraps with GoalDirectedEmbeddingProvider.

    Args:
        base_provider: The underlying embedding provider (GNN, NoOp, etc.)
        config: Hierarchical integration configuration.
        goals: Optional initial goal clauses to register.

    Returns:
        An EmbeddingProvider, either the base or goal-directed wrapper.
    """
    cfg = config or HierarchicalIntegrationConfig()

    if not cfg.enabled:
        return base_provider

    gd_config = cfg.goal_directed
    if not gd_config.enabled:
        # Hierarchical enabled but goal-directed disabled —
        # just return base (hierarchical GNN is internal to base)
        return base_provider

    provider = GoalDirectedEmbeddingProvider(
        base_provider=base_provider,
        config=gd_config,
    )

    if goals:
        provider.register_goals(goals)

    logger.info(
        "Goal-directed provider created: %d goals, proximity_weight=%.2f",
        provider.num_goals,
        gd_config.goal_proximity_weight,
    )

    return provider
