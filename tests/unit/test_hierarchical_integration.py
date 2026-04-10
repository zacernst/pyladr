"""Tests for unified hierarchical integration.

Verifies:
- Factory creates correct provider type based on config
- Disabled config returns base provider unchanged
- Goal registration works through factory
- Integration with EmbeddingEnhancedSelection
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import get_rigid_term
from pyladr.search.goal_directed import GoalDirectedConfig, GoalDirectedEmbeddingProvider
from pyladr.search.hierarchical_integration import (
    HierarchicalIntegrationConfig,
    create_goal_directed_provider,
)
from pyladr.search.ml_selection import EmbeddingProvider


# ── Helpers ──────────────────────────────────────────────────────────────────


def _const(symnum: int):
    return get_rigid_term(symnum, 0)


def _make_clause(cid: int = 0) -> Clause:
    c = Clause(literals=(Literal(sign=True, atom=_const(1)),))
    c.id = cid
    c.weight = 1.0
    return c


class MockProvider:
    """Mock base embedding provider."""

    def __init__(self, dim: int = 4, embeddings: dict[int, list[float]] | None = None):
        self._dim = dim
        self._embeddings = embeddings or {}

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def get_embedding(self, clause: Clause) -> list[float] | None:
        return self._embeddings.get(clause.id)

    def get_embeddings_batch(self, clauses: list[Clause]) -> list[list[float] | None]:
        return [self.get_embedding(c) for c in clauses]


# ── Tests ────────────────────────────────────────────────────────────────────


class TestFactory:
    """Test create_goal_directed_provider factory."""

    def test_disabled_returns_base(self) -> None:
        base = MockProvider()
        config = HierarchicalIntegrationConfig(enabled=False)
        result = create_goal_directed_provider(base, config)
        assert result is base

    def test_enabled_but_goal_directed_disabled_returns_base(self) -> None:
        base = MockProvider()
        config = HierarchicalIntegrationConfig(
            enabled=True,
            goal_directed=GoalDirectedConfig(enabled=False),
        )
        result = create_goal_directed_provider(base, config)
        assert result is base

    def test_enabled_with_goal_directed_wraps(self) -> None:
        base = MockProvider()
        config = HierarchicalIntegrationConfig(
            enabled=True,
            goal_directed=GoalDirectedConfig(enabled=True),
        )
        result = create_goal_directed_provider(base, config)
        assert isinstance(result, GoalDirectedEmbeddingProvider)

    def test_factory_with_goals(self) -> None:
        base = MockProvider(
            embeddings={10: [1.0, 0.0, 0.0, 0.0]},
        )
        config = HierarchicalIntegrationConfig(
            enabled=True,
            goal_directed=GoalDirectedConfig(enabled=True),
        )
        goal = _make_clause(cid=10)
        result = create_goal_directed_provider(base, config, goals=[goal])
        assert isinstance(result, GoalDirectedEmbeddingProvider)
        assert result.num_goals == 1

    def test_factory_default_config_returns_base(self) -> None:
        base = MockProvider()
        result = create_goal_directed_provider(base)
        assert result is base

    def test_result_satisfies_protocol(self) -> None:
        base = MockProvider(embeddings={1: [1.0, 0.0, 0.0, 0.0]})
        config = HierarchicalIntegrationConfig(
            enabled=True,
            goal_directed=GoalDirectedConfig(enabled=True),
        )
        result = create_goal_directed_provider(base, config)
        assert isinstance(result, EmbeddingProvider)

        c = _make_clause(cid=1)
        emb = result.get_embedding(c)
        assert emb is not None
        assert len(emb) == 4


class TestConfigDefaults:
    """Test configuration defaults."""

    def test_default_disabled(self) -> None:
        config = HierarchicalIntegrationConfig()
        assert not config.enabled

    def test_frozen(self) -> None:
        config = HierarchicalIntegrationConfig()
        with pytest.raises(AttributeError):
            config.enabled = True  # type: ignore[misc]
