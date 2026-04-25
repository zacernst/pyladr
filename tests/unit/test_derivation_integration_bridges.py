"""Tests for derivation history integration bridges.

Validates:
1. DerivationAttentionAdapter — metadata extraction for temporal attention
2. ChainEnhancedAttentionAdapter — rich chain embeddings for attention bias
3. Graph augmentation — CLAUSE node feature extension
4. End-to-end: DerivationContext → attention pipeline integration
"""

from __future__ import annotations

import pytest
import torch

from pyladr.core.clause import Clause, Justification, JustType, Literal, ParaJust
from pyladr.core.term import Term
from pyladr.ml.derivation.derivation_context import DerivationContext
from pyladr.ml.derivation.attention_bridge import (
    ChainEnhancedAttentionAdapter,
    DerivationAttentionAdapter,
    TemporalMetadata,
)
from pyladr.ml.derivation.graph_augmentation import (
    AUGMENTED_CLAUSE_FEATURE_DIM,
    DerivationGraphConfig,
    augmented_gnn_config,
    batch_clauses_to_heterograph_augmented,
    clause_to_heterograph_augmented,
)
from pyladr.ml.derivation.inference_chain_encoder import InferenceChainConfig
from pyladr.ml.graph.clause_graph import NodeType


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_term(symnum: int = -1) -> Term:
    return Term(private_symbol=symnum, arity=0, args=())


def _make_literal(sign: bool = True, symnum: int = -1) -> Literal:
    return Literal(sign=sign, atom=_make_term(symnum))


def _make_clause(
    clause_id: int,
    just_type: JustType = JustType.INPUT,
    clause_ids: tuple[int, ...] = (),
    para: ParaJust | None = None,
) -> Clause:
    primary = Justification(
        just_type=just_type,
        clause_ids=clause_ids,
        para=para,
    )
    return Clause(
        literals=(_make_literal(),),
        id=clause_id,
        justification=(primary,),
    )


def _build_derivation_chain() -> tuple[DerivationContext, list[Clause]]:
    """Build a test derivation: INPUT(1,2) → RES(3) → HYPER(4) → PARA(5)."""
    ctx = DerivationContext()
    clauses = [
        _make_clause(1, JustType.INPUT),
        _make_clause(2, JustType.GOAL),
        _make_clause(3, JustType.BINARY_RES, clause_ids=(1, 2)),
        _make_clause(4, JustType.HYPER_RES, clause_ids=(3, 1)),
        _make_clause(5, JustType.PARA, para=ParaJust(4, 2, (0,), (0,))),
    ]
    for c in clauses:
        ctx.register(c)
    return ctx, clauses


# ── DerivationAttentionAdapter Tests ─────────────────────────────────────────


class TestDerivationAttentionAdapter:
    """Tests for the lightweight temporal metadata adapter."""

    def test_extract_metadata_shapes(self):
        ctx, clauses = _build_derivation_chain()
        adapter = DerivationAttentionAdapter(ctx)
        meta = adapter.extract_metadata(clauses)

        assert isinstance(meta, TemporalMetadata)
        assert meta.derivation_depths.shape == (5,)
        assert meta.inference_types.shape == (5,)
        assert meta.parent_counts.shape == (5,)
        assert meta.clause_ids.shape == (5,)

    def test_extract_metadata_values(self):
        ctx, clauses = _build_derivation_chain()
        adapter = DerivationAttentionAdapter(ctx)
        meta = adapter.extract_metadata(clauses)

        # Clause 1: INPUT, depth 0, 0 parents
        assert meta.derivation_depths[0].item() == 0
        assert meta.inference_types[0].item() == int(JustType.INPUT)
        assert meta.parent_counts[0].item() == 0
        assert meta.clause_ids[0].item() == 1

        # Clause 3: BINARY_RES, depth 1, 2 parents
        assert meta.derivation_depths[2].item() == 1
        assert meta.inference_types[2].item() == int(JustType.BINARY_RES)
        assert meta.parent_counts[2].item() == 2

        # Clause 5: PARA, depth 3, 2 parents
        assert meta.derivation_depths[4].item() == 3
        assert meta.inference_types[4].item() == int(JustType.PARA)
        assert meta.parent_counts[4].item() == 2

    def test_unregistered_clauses_get_defaults(self):
        ctx = DerivationContext()
        clauses = [_make_clause(99, JustType.INPUT)]
        # Don't register — adapter should handle gracefully
        adapter = DerivationAttentionAdapter(ctx)
        meta = adapter.extract_metadata(clauses)

        assert meta.derivation_depths[0].item() == 0
        assert meta.inference_types[0].item() == 0
        assert meta.parent_counts[0].item() == 0

    def test_device_placement(self):
        ctx, clauses = _build_derivation_chain()
        adapter = DerivationAttentionAdapter(ctx)
        meta = adapter.extract_metadata(clauses, device=torch.device("cpu"))

        assert meta.derivation_depths.device.type == "cpu"

    def test_empty_clause_list(self):
        ctx = DerivationContext()
        adapter = DerivationAttentionAdapter(ctx)
        meta = adapter.extract_metadata([])

        assert meta.derivation_depths.shape == (0,)

    def test_compatible_with_temporal_attention_input(self):
        """Verify output tensors have correct dtypes for TemporalPositionEncoder."""
        ctx, clauses = _build_derivation_chain()
        adapter = DerivationAttentionAdapter(ctx)
        meta = adapter.extract_metadata(clauses)

        assert meta.derivation_depths.dtype == torch.long
        assert meta.inference_types.dtype == torch.long
        assert meta.parent_counts.dtype == torch.long
        assert meta.clause_ids.dtype == torch.long


# ── ChainEnhancedAttentionAdapter Tests ──────────────────────────────────────


class TestChainEnhancedAttentionAdapter:
    """Tests for the rich chain-embedding adapter."""

    @pytest.fixture
    def adapter_and_data(self):
        ctx, clauses = _build_derivation_chain()
        adapter = ChainEnhancedAttentionAdapter(
            context=ctx,
            embedding_dim=64,
            chain_config=InferenceChainConfig(
                chain_embed_dim=16,
                output_dim=32,
                max_chain_length=8,
            ),
        )
        return adapter, ctx, clauses

    def test_output_shape(self, adapter_and_data):
        adapter, ctx, clauses = adapter_and_data
        adapter.eval()
        base_embs = torch.randn(5, 64)

        enriched = adapter(clauses, base_embs)
        assert enriched.shape == (5, 64)

    def test_enriched_differs_from_base(self, adapter_and_data):
        adapter, ctx, clauses = adapter_and_data
        adapter.eval()
        base_embs = torch.randn(5, 64)

        enriched = adapter(clauses, base_embs)
        # Enriched should differ from base (chain context added)
        assert not torch.allclose(enriched, base_embs, atol=1e-3)

    def test_gradients_flow(self, adapter_and_data):
        adapter, ctx, clauses = adapter_and_data
        adapter.train()
        base_embs = torch.randn(5, 64, requires_grad=True)

        enriched = adapter(clauses, base_embs)
        loss = enriched.sum()
        loss.backward()

        assert base_embs.grad is not None
        assert base_embs.grad.abs().sum() > 0

        # Chain encoder parameters also got gradients
        proj_params = list(adapter.chain_proj.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in proj_params)

    def test_extract_metadata_delegates(self, adapter_and_data):
        adapter, ctx, clauses = adapter_and_data
        meta = adapter.extract_metadata(clauses)
        assert isinstance(meta, TemporalMetadata)
        assert meta.derivation_depths.shape == (5,)

    def test_single_clause(self, adapter_and_data):
        adapter, ctx, clauses = adapter_and_data
        adapter.eval()
        base_embs = torch.randn(1, 64)

        enriched = adapter([clauses[0]], base_embs)
        assert enriched.shape == (1, 64)


# ── Graph Augmentation Tests ─────────────────────────────────────────────────


class TestGraphAugmentation:
    """Tests for CLAUSE node feature augmentation."""

    def test_augmented_clause_features_dim(self):
        ctx, clauses = _build_derivation_chain()
        config = DerivationGraphConfig(enabled=True)

        data = clause_to_heterograph_augmented(clauses[0], ctx, config=config)
        clause_features = data[NodeType.CLAUSE.value].x
        assert clause_features.shape[1] == AUGMENTED_CLAUSE_FEATURE_DIM
        assert clause_features.shape[1] == 20  # 7 + 13

    def test_disabled_preserves_original_dim(self):
        ctx, clauses = _build_derivation_chain()
        config = DerivationGraphConfig(enabled=False)

        data = clause_to_heterograph_augmented(clauses[0], ctx, config=config)
        clause_features = data[NodeType.CLAUSE.value].x
        assert clause_features.shape[1] == 7  # original

    def test_batch_augmentation(self):
        ctx, clauses = _build_derivation_chain()
        config = DerivationGraphConfig(enabled=True)

        graphs = batch_clauses_to_heterograph_augmented(clauses, ctx, config=config)
        assert len(graphs) == 5

        for graph in graphs:
            clause_features = graph[NodeType.CLAUSE.value].x
            assert clause_features.shape[1] == AUGMENTED_CLAUSE_FEATURE_DIM

    def test_original_features_preserved(self):
        """First 7 features should match non-augmented graph."""
        ctx, clauses = _build_derivation_chain()
        from pyladr.ml.graph.clause_graph import clause_to_heterograph

        # Non-augmented
        base = clause_to_heterograph(clauses[2])
        base_feats = base[NodeType.CLAUSE.value].x  # (1, 7)

        # Augmented
        config = DerivationGraphConfig(enabled=True)
        aug = clause_to_heterograph_augmented(clauses[2], ctx, config=config)
        aug_feats = aug[NodeType.CLAUSE.value].x  # (1, 20)

        # First 7 dims should match exactly
        assert torch.allclose(aug_feats[:, :7], base_feats)

    def test_derivation_features_non_zero_for_derived(self):
        """Derived clauses should have non-zero derivation features."""
        ctx, clauses = _build_derivation_chain()
        config = DerivationGraphConfig(enabled=True)

        # Clause 3 is BINARY_RES — should have non-zero derivation features
        data = clause_to_heterograph_augmented(clauses[2], ctx, config=config)
        deriv_feats = data[NodeType.CLAUSE.value].x[:, 7:]  # derivation part
        assert deriv_feats.abs().sum() > 0

    def test_augmented_gnn_config(self):
        dims = augmented_gnn_config()
        assert dims[NodeType.CLAUSE.value] == 20
        assert dims[NodeType.LITERAL.value] == 3  # unchanged
        assert dims[NodeType.TERM.value] == 8  # unchanged

    def test_augmented_gnn_config_from_existing(self):
        existing = {
            NodeType.CLAUSE.value: 7,
            NodeType.LITERAL.value: 3,
        }
        dims = augmented_gnn_config(existing)
        assert dims[NodeType.CLAUSE.value] == 20
        assert dims[NodeType.LITERAL.value] == 3


# ── End-to-End Integration Tests ─────────────────────────────────────────────


class TestEndToEndIntegration:
    """Tests verifying the full derivation → attention pipeline."""

    def test_metadata_to_temporal_attention_compatible(self):
        """TemporalCrossClauseAttention accepts adapter output."""
        from pyladr.ml.attention.temporal_attention import (
            TemporalAttentionConfig,
            TemporalCrossClauseAttention,
        )
        from pyladr.ml.attention.cross_clause import CrossClauseAttentionConfig

        ctx, clauses = _build_derivation_chain()
        adapter = DerivationAttentionAdapter(ctx)
        meta = adapter.extract_metadata(clauses)

        config = TemporalAttentionConfig(
            base_config=CrossClauseAttentionConfig(
                enabled=True,
                embedding_dim=64,
                num_heads=4,
                max_clauses=32,
            ),
            use_temporal_encoding=True,
            temporal_dim=32,
        )
        model = TemporalCrossClauseAttention(config)
        model.eval()

        base_embs = torch.randn(5, 64)
        scores = model.score_clauses(
            base_embeddings=base_embs,
            derivation_depths=meta.derivation_depths,
            inference_types=meta.inference_types,
            parent_counts=meta.parent_counts,
            clause_ids=meta.clause_ids,
        )

        assert len(scores) == 5
        assert all(isinstance(s, float) for s in scores)

    def test_chain_enhanced_with_temporal_attention(self):
        """ChainEnhancedAdapter output feeds into TemporalCrossClauseAttention."""
        from pyladr.ml.attention.temporal_attention import (
            TemporalAttentionConfig,
            TemporalCrossClauseAttention,
        )
        from pyladr.ml.attention.cross_clause import CrossClauseAttentionConfig

        ctx, clauses = _build_derivation_chain()

        # Chain-enhanced adapter
        chain_adapter = ChainEnhancedAttentionAdapter(
            context=ctx,
            embedding_dim=64,
            chain_config=InferenceChainConfig(
                chain_embed_dim=16, output_dim=32, max_chain_length=8,
            ),
        )
        chain_adapter.eval()

        # Lightweight adapter for metadata
        meta_adapter = DerivationAttentionAdapter(ctx)
        meta = meta_adapter.extract_metadata(clauses)

        # Enrich base embeddings with chain context
        base_embs = torch.randn(5, 64)
        enriched = chain_adapter(clauses, base_embs)

        # Feed enriched embeddings to temporal attention
        config = TemporalAttentionConfig(
            base_config=CrossClauseAttentionConfig(
                enabled=True,
                embedding_dim=64,
                num_heads=4,
                max_clauses=32,
            ),
            use_temporal_encoding=True,
            temporal_dim=32,
        )
        model = TemporalCrossClauseAttention(config)
        model.eval()

        scores = model.score_clauses(
            base_embeddings=enriched,
            derivation_depths=meta.derivation_depths,
            inference_types=meta.inference_types,
            parent_counts=meta.parent_counts,
            clause_ids=meta.clause_ids,
        )

        assert len(scores) == 5

    def test_augmented_graph_with_gnn(self):
        """Augmented graph can be processed by GNN with updated config."""
        from pyladr.ml.graph.clause_encoder import GNNConfig, HeterogeneousClauseGNN
        from torch_geometric.data import Batch

        ctx, clauses = _build_derivation_chain()
        graph_config = DerivationGraphConfig(enabled=True)

        graphs = batch_clauses_to_heterograph_augmented(
            clauses, ctx, config=graph_config
        )

        # Build GNN with augmented clause feature dim
        gnn_config = GNNConfig(
            hidden_dim=32,
            embedding_dim=64,
            num_layers=2,
            node_feature_dims=augmented_gnn_config(),
        )
        model = HeterogeneousClauseGNN(gnn_config)
        model.eval()

        # Process each graph individually
        for graph in graphs:
            emb = model.embed_clause(graph)
            assert emb.shape[1] == 64
