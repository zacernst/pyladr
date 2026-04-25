"""Tests for cross-clause attention mechanisms.

Tests the multi-head attention, relational scoring head, and the combined
scorer pipeline. All tests use small dimensions for speed and determinism.
"""

from __future__ import annotations

import math

import pytest
import torch

from pyladr.ml.attention.cross_clause import (
    CrossClauseAttentionConfig,
    CrossClauseAttentionScorer,
    MultiHeadClauseAttention,
    RelationalScoringHead,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def small_config() -> CrossClauseAttentionConfig:
    """Small config for fast unit tests."""
    return CrossClauseAttentionConfig(
        enabled=True,
        embedding_dim=32,
        num_heads=4,
        head_dim=8,
        dropout=0.0,  # deterministic
        use_relative_position=True,
        max_clauses=64,
        temperature=1.0,
        use_goal_conditioning=False,
        scoring_hidden_dim=16,
    )


@pytest.fixture
def goal_config() -> CrossClauseAttentionConfig:
    """Config with goal conditioning enabled."""
    return CrossClauseAttentionConfig(
        enabled=True,
        embedding_dim=32,
        num_heads=4,
        head_dim=8,
        dropout=0.0,
        use_relative_position=True,
        max_clauses=64,
        temperature=1.0,
        use_goal_conditioning=True,
        goal_dim=16,
        scoring_hidden_dim=16,
    )


def _random_embeddings(n: int, d: int, seed: int = 42) -> torch.Tensor:
    """Generate deterministic random embeddings."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=gen)


# ── MultiHeadClauseAttention tests ───────────────────────────────────────


class TestMultiHeadClauseAttention:
    def test_output_shape(self, small_config):
        attn = MultiHeadClauseAttention(small_config)
        attn.eval()
        x = _random_embeddings(10, 32)
        out = attn(x)
        assert out.shape == (10, 32)

    def test_single_clause(self, small_config):
        """Attention with a single clause should produce valid output."""
        attn = MultiHeadClauseAttention(small_config)
        attn.eval()
        x = _random_embeddings(1, 32)
        out = attn(x)
        assert out.shape == (1, 32)
        assert torch.isfinite(out).all()

    def test_residual_connection(self, small_config):
        """Output should not be identical to input (attention adds information)."""
        attn = MultiHeadClauseAttention(small_config)
        attn.eval()
        x = _random_embeddings(5, 32)
        out = attn(x)
        # With random weights, residual + attention should differ from input
        assert not torch.allclose(out, x, atol=1e-6)

    def test_deterministic_eval(self, small_config):
        """Eval mode with dropout=0 should be deterministic."""
        attn = MultiHeadClauseAttention(small_config)
        attn.eval()
        x = _random_embeddings(8, 32)
        out1 = attn(x)
        out2 = attn(x)
        assert torch.allclose(out1, out2)

    def test_with_clause_ids(self, small_config):
        """Clause IDs should affect output via relative position bias."""
        attn = MultiHeadClauseAttention(small_config)
        attn.eval()
        x = _random_embeddings(5, 32)

        ids_sequential = torch.tensor([0, 1, 2, 3, 4])
        ids_reversed = torch.tensor([4, 3, 2, 1, 0])

        out_seq = attn(x, clause_ids=ids_sequential)
        out_rev = attn(x, clause_ids=ids_reversed)

        # Different orderings should produce different outputs
        assert not torch.allclose(out_seq, out_rev, atol=1e-5)

    def test_no_relative_position(self):
        """Without relative position bias, clause IDs should not matter."""
        cfg = CrossClauseAttentionConfig(
            enabled=True,
            embedding_dim=32,
            num_heads=4,
            head_dim=8,
            dropout=0.0,
            use_relative_position=False,
            scoring_hidden_dim=16,
        )
        attn = MultiHeadClauseAttention(cfg)
        attn.eval()
        x = _random_embeddings(5, 32)

        out_none = attn(x, clause_ids=None)
        out_ids = attn(x, clause_ids=torch.tensor([10, 20, 30, 40, 50]))

        # Without relative position bias, IDs should not affect output
        assert torch.allclose(out_none, out_ids, atol=1e-5)

    def test_goal_conditioning(self, goal_config):
        """Goal context should modulate attention output."""
        attn = MultiHeadClauseAttention(goal_config)
        attn.eval()
        x = _random_embeddings(5, 32)
        goal = torch.randn(16)

        out_no_goal = attn(x)
        out_with_goal = attn(x, goal_context=goal)

        # Goal conditioning should change the output
        assert not torch.allclose(out_no_goal, out_with_goal, atol=1e-5)

    def test_large_clause_set(self, small_config):
        """Should handle the maximum clause set size."""
        attn = MultiHeadClauseAttention(small_config)
        attn.eval()
        x = _random_embeddings(64, 32)
        out = attn(x)
        assert out.shape == (64, 32)
        assert torch.isfinite(out).all()


# ── RelationalScoringHead tests ──────────────────────────────────────────


class TestRelationalScoringHead:
    def test_output_shape(self, small_config):
        head = RelationalScoringHead(small_config)
        head.eval()
        enriched = _random_embeddings(10, 32)
        original = _random_embeddings(10, 32, seed=99)
        scores = head(enriched, original)
        assert scores.shape == (10,)

    def test_single_clause(self, small_config):
        head = RelationalScoringHead(small_config)
        head.eval()
        enriched = _random_embeddings(1, 32)
        original = _random_embeddings(1, 32, seed=99)
        scores = head(enriched, original)
        assert scores.shape == (1,)
        assert torch.isfinite(scores).all()

    def test_different_inputs_different_scores(self, small_config):
        """Different enriched embeddings should produce different scores."""
        head = RelationalScoringHead(small_config)
        head.eval()
        original = _random_embeddings(5, 32)
        enriched1 = _random_embeddings(5, 32, seed=1)
        enriched2 = _random_embeddings(5, 32, seed=2)

        scores1 = head(enriched1, original)
        scores2 = head(enriched2, original)

        assert not torch.allclose(scores1, scores2)


# ── CrossClauseAttentionScorer tests ─────────────────────────────────────


class TestCrossClauseAttentionScorer:
    def test_output_shape(self, small_config):
        scorer = CrossClauseAttentionScorer(small_config)
        scorer.eval()
        x = _random_embeddings(10, 32)
        scores = scorer(x)
        assert scores.shape == (10,)

    def test_score_clauses_returns_list(self, small_config):
        scorer = CrossClauseAttentionScorer(small_config)
        x = _random_embeddings(10, 32)
        scores = scorer.score_clauses(x)
        assert isinstance(scores, list)
        assert len(scores) == 10
        assert all(isinstance(s, float) for s in scores)

    def test_truncation_beyond_max_clauses(self):
        """Clauses beyond max_clauses should be truncated."""
        cfg = CrossClauseAttentionConfig(
            enabled=True,
            embedding_dim=16,
            num_heads=2,
            head_dim=8,
            dropout=0.0,
            use_relative_position=False,
            max_clauses=5,
            scoring_hidden_dim=8,
        )
        scorer = CrossClauseAttentionScorer(cfg)
        scorer.eval()

        # Input has 10 clauses, max is 5
        x = _random_embeddings(10, 16)
        scores = scorer(x)
        assert scores.shape == (5,)  # truncated to last 5

    def test_with_goal_context(self, goal_config):
        scorer = CrossClauseAttentionScorer(goal_config)
        scorer.eval()
        x = _random_embeddings(8, 32)
        goal = torch.randn(16)
        scores = scorer(x, goal_context=goal)
        assert scores.shape == (8,)
        assert torch.isfinite(scores).all()

    def test_gradient_flow(self, small_config):
        """Gradients should flow through the entire pipeline."""
        scorer = CrossClauseAttentionScorer(small_config)
        scorer.train()
        x = _random_embeddings(5, 32).requires_grad_(True)
        scores = scorer(x)
        loss = scores.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_no_nan_with_identical_embeddings(self, small_config):
        """Identical embeddings should not cause NaN (softmax edge case)."""
        scorer = CrossClauseAttentionScorer(small_config)
        scorer.eval()
        # All clauses have the same embedding
        x = torch.ones(5, 32)
        scores = scorer(x)
        assert torch.isfinite(scores).all()

    def test_no_nan_with_zero_embeddings(self, small_config):
        """Zero embeddings should not cause NaN."""
        scorer = CrossClauseAttentionScorer(small_config)
        scorer.eval()
        x = torch.zeros(5, 32)
        scores = scorer(x)
        assert torch.isfinite(scores).all()


# ── Config tests ─────────────────────────────────────────────────────────


class TestCrossClauseAttentionConfig:
    def test_effective_head_dim_auto(self):
        cfg = CrossClauseAttentionConfig(embedding_dim=256, num_heads=8, head_dim=0)
        assert cfg.effective_head_dim == 32

    def test_effective_head_dim_explicit(self):
        cfg = CrossClauseAttentionConfig(embedding_dim=256, num_heads=8, head_dim=64)
        assert cfg.effective_head_dim == 64

    def test_default_disabled(self):
        cfg = CrossClauseAttentionConfig()
        assert not cfg.enabled
