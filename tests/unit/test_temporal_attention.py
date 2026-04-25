"""Tests for temporal and multi-source attention integration.

Tests the TemporalPositionEncoder, MultiSourceFusion, and the combined
TemporalCrossClauseAttention pipeline. Uses small dimensions for speed.
"""

from __future__ import annotations

import pytest
import torch

from pyladr.ml.attention.cross_clause import CrossClauseAttentionConfig
from pyladr.ml.attention.temporal_attention import (
    MultiSourceFusion,
    TemporalAttentionConfig,
    TemporalCrossClauseAttention,
    TemporalPositionEncoder,
)


def _rand(n, d, seed=42):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=gen)


@pytest.fixture
def base_attn_config():
    return CrossClauseAttentionConfig(
        enabled=True,
        embedding_dim=32,
        num_heads=4,
        head_dim=8,
        dropout=0.0,
        use_relative_position=False,
        max_clauses=64,
        scoring_hidden_dim=16,
    )


# ── TemporalPositionEncoder tests ────────────────────────────────────────


class TestTemporalPositionEncoder:
    @pytest.fixture
    def config(self, base_attn_config):
        return TemporalAttentionConfig(
            base_config=base_attn_config,
            temporal_dim=32,
            max_derivation_depth=50,
            num_inference_types=22,
        )

    def test_output_shape(self, config):
        enc = TemporalPositionEncoder(config)
        enc.eval()
        depths = torch.tensor([0, 1, 5, 10, 50])
        types = torch.tensor([0, 5, 3, 1, 7])
        parents = torch.tensor([0, 1, 2, 2, 3])
        out = enc(depths, types, parents)
        assert out.shape == (5, 32)

    def test_different_depths_different_encodings(self, config):
        enc = TemporalPositionEncoder(config)
        enc.eval()
        depths = torch.tensor([0, 50])
        types = torch.tensor([0, 0])
        parents = torch.tensor([0, 0])
        out = enc(depths, types, parents)
        assert not torch.allclose(out[0], out[1], atol=1e-4)

    def test_different_types_different_encodings(self, config):
        enc = TemporalPositionEncoder(config)
        enc.eval()
        depths = torch.tensor([5, 5])
        types = torch.tensor([0, 10])
        parents = torch.tensor([1, 1])
        out = enc(depths, types, parents)
        assert not torch.allclose(out[0], out[1], atol=1e-4)

    def test_depth_clamping(self, config):
        """Depths beyond max should be clamped, not crash."""
        enc = TemporalPositionEncoder(config)
        enc.eval()
        depths = torch.tensor([0, 100, 999])  # 999 > max_derivation_depth=50
        types = torch.tensor([0, 0, 0])
        parents = torch.tensor([0, 0, 0])
        out = enc(depths, types, parents)
        assert torch.isfinite(out).all()
        # 100 and 999 should both clamp to 50, so same encoding
        assert torch.allclose(out[1], out[2])

    def test_single_clause(self, config):
        enc = TemporalPositionEncoder(config)
        enc.eval()
        out = enc(torch.tensor([3]), torch.tensor([1]), torch.tensor([2]))
        assert out.shape == (1, 32)
        assert torch.isfinite(out).all()


# ── MultiSourceFusion tests ──────────────────────────────────────────────


class TestMultiSourceFusion:
    def test_base_only_gate(self, base_attn_config):
        """With no extra sources, fusion should pass through base."""
        cfg = TemporalAttentionConfig(
            base_config=base_attn_config,
            hierarchical_dim=0,
            invariant_dim=0,
            fusion_method="gate",
        )
        fusion = MultiSourceFusion(cfg)
        fusion.eval()
        base = _rand(5, 32)
        out = fusion(base)
        assert out.shape == (5, 32)
        assert torch.isfinite(out).all()

    def test_with_hierarchical(self, base_attn_config):
        cfg = TemporalAttentionConfig(
            base_config=base_attn_config,
            hierarchical_dim=16,
            invariant_dim=0,
            fusion_method="gate",
        )
        fusion = MultiSourceFusion(cfg)
        fusion.eval()
        base = _rand(5, 32)
        hier = _rand(5, 16, seed=99)
        out = fusion(base, hierarchical_embeddings=hier)
        assert out.shape == (5, 32)

    def test_with_all_sources(self, base_attn_config):
        cfg = TemporalAttentionConfig(
            base_config=base_attn_config,
            hierarchical_dim=16,
            invariant_dim=24,
            fusion_method="gate",
        )
        fusion = MultiSourceFusion(cfg)
        fusion.eval()
        base = _rand(5, 32)
        hier = _rand(5, 16, seed=99)
        inv = _rand(5, 24, seed=77)
        out = fusion(base, hierarchical_embeddings=hier, invariant_embeddings=inv)
        assert out.shape == (5, 32)

    def test_missing_optional_source(self, base_attn_config):
        """Missing optional source should use zeros, not crash."""
        cfg = TemporalAttentionConfig(
            base_config=base_attn_config,
            hierarchical_dim=16,
            invariant_dim=0,
            fusion_method="gate",
        )
        fusion = MultiSourceFusion(cfg)
        fusion.eval()
        base = _rand(5, 32)
        # Don't pass hierarchical — should use zero fallback
        out = fusion(base)
        assert out.shape == (5, 32)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("method", ["gate", "attention", "concat_project"])
    def test_all_fusion_methods(self, base_attn_config, method):
        cfg = TemporalAttentionConfig(
            base_config=base_attn_config,
            hierarchical_dim=16,
            invariant_dim=24,
            fusion_method=method,
        )
        fusion = MultiSourceFusion(cfg)
        fusion.eval()
        base = _rand(5, 32)
        hier = _rand(5, 16, seed=99)
        inv = _rand(5, 24, seed=77)
        out = fusion(base, hierarchical_embeddings=hier, invariant_embeddings=inv)
        assert out.shape == (5, 32)
        assert torch.isfinite(out).all()

    def test_hierarchical_changes_output(self, base_attn_config):
        """Adding hierarchical features should change the output."""
        cfg = TemporalAttentionConfig(
            base_config=base_attn_config,
            hierarchical_dim=16,
            invariant_dim=0,
            fusion_method="gate",
        )
        fusion = MultiSourceFusion(cfg)
        fusion.eval()
        base = _rand(5, 32)
        hier = _rand(5, 16, seed=99)
        out_base_only = fusion(base)
        out_with_hier = fusion(base, hierarchical_embeddings=hier)
        assert not torch.allclose(out_base_only, out_with_hier, atol=1e-5)


# ── TemporalCrossClauseAttention tests ───────────────────────────────────


class TestTemporalCrossClauseAttention:
    @pytest.fixture
    def full_config(self, base_attn_config):
        return TemporalAttentionConfig(
            base_config=base_attn_config,
            use_temporal_encoding=True,
            temporal_dim=32,
            max_derivation_depth=50,
            num_inference_types=22,
            use_multi_source_fusion=True,
            hierarchical_dim=16,
            invariant_dim=24,
            fusion_method="gate",
        )

    @pytest.fixture
    def base_only_config(self, base_attn_config):
        return TemporalAttentionConfig(
            base_config=base_attn_config,
            use_temporal_encoding=False,
            use_multi_source_fusion=False,
            hierarchical_dim=0,
            invariant_dim=0,
        )

    def test_full_pipeline_output_shape(self, full_config):
        model = TemporalCrossClauseAttention(full_config)
        model.eval()

        N = 8
        base = _rand(N, 32)
        hier = _rand(N, 16, seed=99)
        inv = _rand(N, 24, seed=77)
        depths = torch.randint(0, 50, (N,))
        types = torch.randint(0, 22, (N,))
        parents = torch.randint(0, 5, (N,))

        scores = model(
            base, hierarchical_embeddings=hier, invariant_embeddings=inv,
            derivation_depths=depths, inference_types=types, parent_counts=parents,
        )
        assert scores.shape == (N,)
        assert torch.isfinite(scores).all()

    def test_base_only_pipeline(self, base_only_config):
        """Without temporal/fusion, should behave like base CrossClauseAttention."""
        model = TemporalCrossClauseAttention(base_only_config)
        model.eval()
        base = _rand(8, 32)
        scores = model(base)
        assert scores.shape == (8,)
        assert torch.isfinite(scores).all()

    def test_score_clauses_returns_list(self, full_config):
        model = TemporalCrossClauseAttention(full_config)
        N = 5
        base = _rand(N, 32)
        depths = torch.randint(0, 50, (N,))
        scores = model.score_clauses(base, derivation_depths=depths)
        assert isinstance(scores, list)
        assert len(scores) == N
        assert all(isinstance(s, float) for s in scores)

    def test_temporal_affects_scores(self, full_config):
        """Temporal encoding should change scores compared to no temporal."""
        model = TemporalCrossClauseAttention(full_config)
        model.eval()
        N = 5
        base = _rand(N, 32)
        hier = _rand(N, 16, seed=99)
        inv = _rand(N, 24, seed=77)

        # With temporal
        depths = torch.tensor([0, 10, 20, 30, 40])
        types = torch.tensor([0, 5, 3, 1, 7])
        parents = torch.tensor([0, 1, 2, 2, 3])
        scores_with = model(
            base, hier, inv,
            derivation_depths=depths, inference_types=types, parent_counts=parents,
        )

        # Without temporal (depths=None triggers skip)
        scores_without = model(base, hier, inv)

        assert not torch.allclose(scores_with, scores_without, atol=1e-5)

    def test_gradient_flow_through_pipeline(self, full_config):
        model = TemporalCrossClauseAttention(full_config)
        model.train()
        N = 4
        base = _rand(N, 32).requires_grad_(True)
        hier = _rand(N, 16, seed=99)
        inv = _rand(N, 24, seed=77)
        depths = torch.randint(0, 50, (N,))
        types = torch.randint(0, 22, (N,))
        parents = torch.randint(0, 5, (N,))

        scores = model(base, hier, inv, depths, types, parents)
        loss = scores.sum()
        loss.backward()
        assert base.grad is not None
        assert torch.isfinite(base.grad).all()

    def test_single_clause(self, full_config):
        model = TemporalCrossClauseAttention(full_config)
        model.eval()
        base = _rand(1, 32)
        hier = _rand(1, 16, seed=99)
        inv = _rand(1, 24, seed=77)
        depths = torch.tensor([5])
        types = torch.tensor([3])
        parents = torch.tensor([2])
        scores = model(base, hier, inv, depths, types, parents)
        assert scores.shape == (1,)
        assert torch.isfinite(scores).all()

    def test_partial_optional_inputs(self, full_config):
        """Should work with only some optional inputs provided."""
        model = TemporalCrossClauseAttention(full_config)
        model.eval()
        N = 5
        base = _rand(N, 32)

        # Only hierarchical, no invariant or temporal
        hier = _rand(N, 16, seed=99)
        scores = model(base, hierarchical_embeddings=hier)
        assert scores.shape == (N,)
        assert torch.isfinite(scores).all()
