"""Tests for hierarchical message passing modules.

Tests IntraLevelMP, InterLevelMP, and CrossLevelAttention components
that implement multi-level aggregation for the hierarchical GNN.
"""

from __future__ import annotations

import pytest

# Guard: these tests require torch
torch = pytest.importorskip("torch")

import torch.nn as nn

from pyladr.ml.hierarchical.message_passing import (
    CrossLevelAttention,
    InterLevelMP,
    IntraLevelMP,
)
from pyladr.ml.hierarchical.architecture import HierarchyLevel


HIDDEN_DIM = 32


# ── IntraLevelMP Tests ───────────────────────────────────────────────────────


class TestIntraLevelMP:
    """Test intra-level (within same hierarchy level) message passing."""

    @pytest.fixture
    def module(self) -> IntraLevelMP:
        m = IntraLevelMP(level=HierarchyLevel.TERM, hidden_dim=HIDDEN_DIM)
        m.eval()
        return m

    def test_output_shape(self, module):
        """Output shape matches input shape."""
        x = torch.randn(10, HIDDEN_DIM)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        out = module(x, edge_index)
        assert out.shape == x.shape

    def test_no_edges_still_produces_output(self, module):
        """With no edges, self-transform with residual is applied."""
        x = torch.randn(5, HIDDEN_DIM)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        out = module(x, edge_index)
        assert out.shape == x.shape

    def test_empty_input(self, module):
        """Zero nodes returns empty tensor."""
        x = torch.zeros(0, HIDDEN_DIM)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        out = module(x, edge_index)
        assert out.shape == (0, HIDDEN_DIM)

    def test_single_node_no_edges(self, module):
        """Single node with no edges works via self-transform."""
        x = torch.randn(1, HIDDEN_DIM)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        out = module(x, edge_index)
        assert out.shape == (1, HIDDEN_DIM)

    def test_self_loop(self, module):
        """Self-loop edges are handled correctly."""
        x = torch.randn(3, HIDDEN_DIM)
        edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
        out = module(x, edge_index)
        assert out.shape == x.shape

    def test_gradients_flow(self, module):
        """Gradients flow through the module."""
        module.train()
        x = torch.randn(5, HIDDEN_DIM, requires_grad=True)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out = module(x, edge_index)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_different_levels(self):
        """Module works for different hierarchy levels."""
        for level in HierarchyLevel:
            m = IntraLevelMP(level=level, hidden_dim=HIDDEN_DIM)
            m.eval()
            x = torch.randn(3, HIDDEN_DIM)
            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            out = m(x, edge_index)
            assert out.shape == x.shape

    def test_deterministic_in_eval(self, module):
        """Eval mode produces deterministic output."""
        x = torch.randn(5, HIDDEN_DIM)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        out1 = module(x, edge_index)
        out2 = module(x, edge_index)
        assert torch.allclose(out1, out2)


# ── InterLevelMP Tests ───────────────────────────────────────────────────────


class TestInterLevelMP:
    """Test inter-level (between adjacent levels) message passing."""

    @pytest.fixture
    def module(self) -> InterLevelMP:
        m = InterLevelMP(hidden_dim=HIDDEN_DIM, output_dim=HIDDEN_DIM)
        m.eval()
        return m

    def test_output_shapes(self, module):
        """Both outputs match input shapes."""
        lower = torch.randn(8, HIDDEN_DIM)
        upper = torch.randn(3, HIDDEN_DIM)
        bottom_up = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 1, 1, 1, 2, 2]], dtype=torch.long)
        top_down = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2], [0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
        out_lower, out_upper = module(lower, upper, bottom_up, top_down)
        assert out_lower.shape == lower.shape
        assert out_upper.shape == upper.shape

    def test_empty_edges(self, module):
        """Empty edges return inputs unchanged."""
        lower = torch.randn(5, HIDDEN_DIM)
        upper = torch.randn(2, HIDDEN_DIM)
        empty = torch.zeros(2, 0, dtype=torch.long)
        out_lower, out_upper = module(lower, upper, empty, empty)
        assert torch.equal(out_lower, lower)
        assert torch.equal(out_upper, upper)

    def test_bottom_up_only(self, module):
        """Bottom-up edges update upper, lower stays unchanged."""
        lower = torch.randn(4, HIDDEN_DIM)
        upper = torch.randn(2, HIDDEN_DIM)
        bottom_up = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long)
        empty = torch.zeros(2, 0, dtype=torch.long)
        out_lower, out_upper = module(lower, upper, bottom_up, empty)
        assert torch.equal(out_lower, lower)  # unchanged
        assert not torch.equal(out_upper, upper)  # updated

    def test_top_down_only(self, module):
        """Top-down edges update lower, upper stays unchanged."""
        lower = torch.randn(4, HIDDEN_DIM)
        upper = torch.randn(2, HIDDEN_DIM)
        empty = torch.zeros(2, 0, dtype=torch.long)
        top_down = torch.tensor([[0, 0, 1, 1], [0, 1, 2, 3]], dtype=torch.long)
        out_lower, out_upper = module(lower, upper, empty, top_down)
        assert torch.equal(out_upper, upper)  # unchanged
        assert not torch.equal(out_lower, lower)  # updated

    def test_gradients_flow(self, module):
        """Gradients flow through both directions."""
        module.train()
        lower = torch.randn(4, HIDDEN_DIM, requires_grad=True)
        upper = torch.randn(2, HIDDEN_DIM, requires_grad=True)
        bottom_up = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long)
        top_down = torch.tensor([[0, 0, 1, 1], [0, 1, 2, 3]], dtype=torch.long)
        out_lower, out_upper = module(lower, upper, bottom_up, top_down)
        loss = out_lower.sum() + out_upper.sum()
        loss.backward()
        assert lower.grad is not None
        assert upper.grad is not None

    def test_deterministic_in_eval(self, module):
        """Eval mode produces deterministic output."""
        lower = torch.randn(4, HIDDEN_DIM)
        upper = torch.randn(2, HIDDEN_DIM)
        edges = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        out1 = module(lower, upper, edges, edges)
        out2 = module(lower, upper, edges, edges)
        assert torch.allclose(out1[0], out2[0])
        assert torch.allclose(out1[1], out2[1])


# ── CrossLevelAttention Tests ────────────────────────────────────────────────


class TestCrossLevelAttention:
    """Test cross-level attention between non-adjacent hierarchy levels."""

    @pytest.fixture
    def levels(self) -> list[HierarchyLevel]:
        return [HierarchyLevel.SYMBOL, HierarchyLevel.TERM, HierarchyLevel.LITERAL, HierarchyLevel.CLAUSE]

    @pytest.fixture
    def module(self, levels) -> CrossLevelAttention:
        m = CrossLevelAttention(
            levels=levels,
            hidden_dim=HIDDEN_DIM,
            num_heads=4,
        )
        m.eval()
        return m

    def test_output_shapes_match_input(self, module, levels):
        """Each level's output shape matches its input."""
        level_embeddings = {
            HierarchyLevel.SYMBOL: torch.randn(10, HIDDEN_DIM),
            HierarchyLevel.TERM: torch.randn(8, HIDDEN_DIM),
            HierarchyLevel.LITERAL: torch.randn(5, HIDDEN_DIM),
            HierarchyLevel.CLAUSE: torch.randn(3, HIDDEN_DIM),
        }
        result = module(level_embeddings, {})
        for level, x in level_embeddings.items():
            assert result[level].shape == x.shape

    def test_single_level_returns_unchanged(self, module):
        """With only one level, returns input unchanged."""
        level_embeddings = {
            HierarchyLevel.TERM: torch.randn(5, HIDDEN_DIM),
        }
        result = module(level_embeddings, {})
        assert torch.equal(result[HierarchyLevel.TERM], level_embeddings[HierarchyLevel.TERM])

    def test_two_levels_cross_attention(self, module):
        """With two levels, cross-attention modifies embeddings."""
        emb_a = torch.randn(5, HIDDEN_DIM)
        emb_b = torch.randn(3, HIDDEN_DIM)
        level_embeddings = {
            HierarchyLevel.SYMBOL: emb_a,
            HierarchyLevel.CLAUSE: emb_b,
        }
        result = module(level_embeddings, {})
        # Both should be updated
        assert not torch.equal(result[HierarchyLevel.SYMBOL], emb_a)
        assert not torch.equal(result[HierarchyLevel.CLAUSE], emb_b)

    def test_empty_level_handled(self, module):
        """Empty level (0 nodes) returns empty tensor."""
        level_embeddings = {
            HierarchyLevel.SYMBOL: torch.randn(5, HIDDEN_DIM),
            HierarchyLevel.TERM: torch.zeros(0, HIDDEN_DIM),
            HierarchyLevel.CLAUSE: torch.randn(3, HIDDEN_DIM),
        }
        result = module(level_embeddings, {})
        assert result[HierarchyLevel.TERM].shape == (0, HIDDEN_DIM)

    def test_gradients_flow(self, module):
        """Gradients flow through cross-level attention."""
        module.train()
        embs = {
            HierarchyLevel.SYMBOL: torch.randn(5, HIDDEN_DIM, requires_grad=True),
            HierarchyLevel.CLAUSE: torch.randn(3, HIDDEN_DIM, requires_grad=True),
        }
        result = module(embs, {})
        loss = sum(v.sum() for v in result.values())
        loss.backward()
        assert embs[HierarchyLevel.SYMBOL].grad is not None
        assert embs[HierarchyLevel.CLAUSE].grad is not None

    def test_deterministic_in_eval(self, module):
        """Eval mode produces deterministic output."""
        embs = {
            HierarchyLevel.SYMBOL: torch.randn(5, HIDDEN_DIM),
            HierarchyLevel.CLAUSE: torch.randn(3, HIDDEN_DIM),
        }
        r1 = module(embs, {})
        r2 = module(embs, {})
        for level in embs:
            assert torch.allclose(r1[level], r2[level])

    def test_scale_factor(self, levels):
        """Scale factor computed correctly from dimensions."""
        m = CrossLevelAttention(levels=levels, hidden_dim=64, num_heads=8)
        expected_scale = (64 // 8) ** -0.5
        assert abs(m.scale - expected_scale) < 1e-6
