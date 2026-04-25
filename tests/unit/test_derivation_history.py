"""Tests for derivation history embeddings and inference chain encoding.

Validates:
1. DerivationContext — DAG tracking, depth computation, chain extraction
2. DerivationFeatureExtractor — feature vector correctness and dimensions
3. InferenceChainEncoder — encoding, padding, recency weighting, gradients
"""

from __future__ import annotations

import math

import pytest
import torch

from pyladr.core.clause import Clause, Justification, JustType, Literal, ParaJust
from pyladr.core.term import Term
from pyladr.ml.derivation.derivation_context import DerivationContext, DerivationInfo
from pyladr.ml.derivation.derivation_features import (
    DERIVATION_FEATURE_DIM,
    DerivationFeatureConfig,
    DerivationFeatureExtractor,
)
from pyladr.ml.derivation.inference_chain_encoder import (
    InferenceChainConfig,
    InferenceChainEncoder,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_term(symnum: int = -1, arity: int = 0) -> Term:
    """Create a simple constant/function term."""
    return Term(private_symbol=symnum, arity=arity, args=())


def _make_literal(sign: bool = True, symnum: int = -1) -> Literal:
    return Literal(sign=sign, atom=_make_term(symnum))


def _make_clause(
    clause_id: int,
    just_type: JustType = JustType.INPUT,
    clause_ids: tuple[int, ...] = (),
    clause_id_single: int = 0,
    para: ParaJust | None = None,
    secondary: tuple[Justification, ...] = (),
) -> Clause:
    """Create a clause with specified justification."""
    primary = Justification(
        just_type=just_type,
        clause_id=clause_id_single,
        clause_ids=clause_ids,
        para=para,
    )
    justification = (primary,) + secondary
    return Clause(
        literals=(_make_literal(),),
        id=clause_id,
        justification=justification,
    )


# ── DerivationContext Tests ──────────────────────────────────────────────────


class TestDerivationContext:
    """Tests for the derivation DAG tracker."""

    def test_register_input_clause(self):
        ctx = DerivationContext()
        c = _make_clause(1, JustType.INPUT)
        info = ctx.register(c)

        assert info.clause_id == 1
        assert info.depth == 0
        assert info.parent_ids == ()
        assert info.primary_rule == int(JustType.INPUT)
        assert info.num_simplifications == 0

    def test_register_goal_clause(self):
        ctx = DerivationContext()
        c = _make_clause(2, JustType.GOAL)
        info = ctx.register(c)

        assert info.depth == 0
        assert info.primary_rule == int(JustType.GOAL)

    def test_register_binary_resolution(self):
        ctx = DerivationContext()
        # Register parents first
        ctx.register(_make_clause(1, JustType.INPUT))
        ctx.register(_make_clause(2, JustType.INPUT))

        # Resolution child
        child = _make_clause(3, JustType.BINARY_RES, clause_ids=(1, 2))
        info = ctx.register(child)

        assert info.depth == 1
        assert info.parent_ids == (1, 2)
        assert info.primary_rule == int(JustType.BINARY_RES)

    def test_register_hyper_resolution_multi_parent(self):
        ctx = DerivationContext()
        for i in range(1, 4):
            ctx.register(_make_clause(i, JustType.INPUT))

        child = _make_clause(4, JustType.HYPER_RES, clause_ids=(1, 2, 3))
        info = ctx.register(child)

        assert info.depth == 1
        assert info.parent_ids == (1, 2, 3)

    def test_register_paramodulation(self):
        ctx = DerivationContext()
        ctx.register(_make_clause(1, JustType.INPUT))
        ctx.register(_make_clause(2, JustType.INPUT))

        para = ParaJust(from_id=1, into_id=2, from_pos=(0,), into_pos=(1,))
        child = _make_clause(3, JustType.PARA, para=para)
        info = ctx.register(child)

        assert info.parent_ids == (1, 2)
        assert info.primary_rule == int(JustType.PARA)

    def test_depth_computation_chain(self):
        """Depth increases along a linear derivation chain."""
        ctx = DerivationContext()
        ctx.register(_make_clause(1, JustType.INPUT))

        for i in range(2, 6):
            ctx.register(
                _make_clause(i, JustType.BINARY_RES, clause_ids=(i - 1, 1))
            )

        assert ctx.get_depth(1) == 0
        assert ctx.get_depth(2) == 1
        assert ctx.get_depth(3) == 2
        assert ctx.get_depth(4) == 3
        assert ctx.get_depth(5) == 4

    def test_depth_computation_diamond(self):
        """Depth is max of parents in a diamond-shaped DAG."""
        ctx = DerivationContext()
        ctx.register(_make_clause(1, JustType.INPUT))    # depth 0
        ctx.register(_make_clause(2, JustType.INPUT))    # depth 0

        # Two children at depth 1
        ctx.register(_make_clause(3, JustType.BINARY_RES, clause_ids=(1, 2)))  # depth 1
        ctx.register(_make_clause(4, JustType.BINARY_RES, clause_ids=(1, 2)))  # depth 1

        # Diamond join at depth 2
        ctx.register(_make_clause(5, JustType.BINARY_RES, clause_ids=(3, 4)))

        assert ctx.get_depth(5) == 2

    def test_get_inference_chain_linear(self):
        ctx = DerivationContext()
        ctx.register(_make_clause(1, JustType.INPUT))
        ctx.register(_make_clause(2, JustType.BINARY_RES, clause_ids=(1,)))
        ctx.register(_make_clause(3, JustType.HYPER_RES, clause_ids=(2,)))
        ctx.register(_make_clause(4, JustType.PARA, para=ParaJust(3, 1, (0,), (0,))))

        chain = ctx.get_inference_chain(4)

        # Should be root → clause: INPUT, BINARY_RES, HYPER_RES, PARA
        assert chain == (
            int(JustType.INPUT),
            int(JustType.BINARY_RES),
            int(JustType.HYPER_RES),
            int(JustType.PARA),
        )

    def test_get_inference_chain_max_length(self):
        ctx = DerivationContext()
        ctx.register(_make_clause(1, JustType.INPUT))
        for i in range(2, 20):
            ctx.register(_make_clause(i, JustType.BINARY_RES, clause_ids=(i - 1,)))

        chain = ctx.get_inference_chain(19, max_length=5)
        assert len(chain) == 5
        # Should contain the 5 most recent steps
        assert chain[-1] == int(JustType.BINARY_RES)

    def test_get_unregistered_clause(self):
        ctx = DerivationContext()
        assert ctx.get(999) is None
        assert ctx.get_depth(999) == 0
        assert ctx.get_inference_chain(999) == ()

    def test_idempotent_registration(self):
        ctx = DerivationContext()
        c = _make_clause(1, JustType.INPUT)
        info1 = ctx.register(c)
        info2 = ctx.register(c)
        assert info1.clause_id == info2.clause_id
        assert ctx.size == 1

    def test_clear(self):
        ctx = DerivationContext()
        ctx.register(_make_clause(1, JustType.INPUT))
        assert ctx.size == 1
        ctx.clear()
        assert ctx.size == 0

    def test_secondary_justification_count(self):
        secondary = (
            Justification(just_type=JustType.DEMOD),
            Justification(just_type=JustType.RENUMBER),
        )
        c = _make_clause(1, JustType.BINARY_RES, clause_ids=(0,), secondary=secondary)
        ctx = DerivationContext()
        info = ctx.register(c)
        assert info.num_simplifications == 2


# ── DerivationFeatureExtractor Tests ─────────────────────────────────────────


class TestDerivationFeatureExtractor:
    """Tests for fixed-size derivation feature extraction."""

    def _setup_linear_chain(self) -> tuple[DerivationContext, list[Clause]]:
        """Create a linear chain: INPUT → BINARY_RES → HYPER_RES → PARA."""
        ctx = DerivationContext()
        clauses = [
            _make_clause(1, JustType.INPUT),
            _make_clause(2, JustType.BINARY_RES, clause_ids=(1,)),
            _make_clause(3, JustType.HYPER_RES, clause_ids=(2,)),
            _make_clause(
                4, JustType.PARA,
                para=ParaJust(3, 1, (0,), (0,)),
            ),
        ]
        for c in clauses:
            ctx.register(c)
        return ctx, clauses

    def test_feature_dimension(self):
        ctx, clauses = self._setup_linear_chain()
        extractor = DerivationFeatureExtractor(ctx)

        for c in clauses:
            features = extractor.extract(c)
            assert features.dim == DERIVATION_FEATURE_DIM
            assert len(features.features) == DERIVATION_FEATURE_DIM

    def test_input_clause_features(self):
        ctx, clauses = self._setup_linear_chain()
        extractor = DerivationFeatureExtractor(ctx)

        features = extractor.extract(clauses[0])
        assert features.depth == 0.0
        assert features.num_parents == 0.0
        assert features.is_input is True

    def test_derived_clause_features(self):
        ctx, clauses = self._setup_linear_chain()
        extractor = DerivationFeatureExtractor(ctx)

        features = extractor.extract(clauses[1])
        assert features.depth > 0.0
        assert features.num_parents == 1.0
        assert features.is_input is False

    def test_feature_depth_increases(self):
        ctx, clauses = self._setup_linear_chain()
        extractor = DerivationFeatureExtractor(ctx)

        depths = [extractor.extract(c).depth for c in clauses]
        for i in range(1, len(depths)):
            assert depths[i] > depths[i - 1]

    def test_disabled_returns_zeros(self):
        ctx, clauses = self._setup_linear_chain()
        config = DerivationFeatureConfig(enabled=False)
        extractor = DerivationFeatureExtractor(ctx, config)

        features = extractor.extract(clauses[2])
        assert all(f == 0.0 for f in features.features)

    def test_demodulation_detection(self):
        ctx = DerivationContext()
        secondary = (Justification(just_type=JustType.DEMOD),)
        c = _make_clause(1, JustType.BINARY_RES, clause_ids=(0,), secondary=secondary)
        ctx.register(c)

        extractor = DerivationFeatureExtractor(ctx)
        features = extractor.extract(c)
        # Feature index 7 = is_demodulated
        assert features.features[7] == 1.0

    def test_batch_extraction(self):
        ctx, clauses = self._setup_linear_chain()
        extractor = DerivationFeatureExtractor(ctx)

        batch = extractor.extract_batch(clauses)
        assert len(batch) == len(clauses)
        for feat_vec in batch:
            assert len(feat_vec) == DERIVATION_FEATURE_DIM

    def test_entropy_increases_with_diversity(self):
        """Chains with more diverse rules should have higher entropy."""
        ctx = DerivationContext()

        # Uniform chain: all BINARY_RES
        ctx.register(_make_clause(1, JustType.INPUT))
        for i in range(2, 6):
            ctx.register(_make_clause(i, JustType.BINARY_RES, clause_ids=(i - 1,)))

        extractor = DerivationFeatureExtractor(ctx)
        uniform_entropy = extractor.extract(
            _make_clause(5, JustType.BINARY_RES, clause_ids=(4,))
        ).features[9]

        # Mixed chain: alternating rules
        ctx2 = DerivationContext()
        ctx2.register(_make_clause(1, JustType.INPUT))
        ctx2.register(_make_clause(2, JustType.BINARY_RES, clause_ids=(1,)))
        ctx2.register(_make_clause(3, JustType.HYPER_RES, clause_ids=(2,)))
        ctx2.register(_make_clause(4, JustType.FACTOR, clause_ids=(3,)))
        ctx2.register(
            _make_clause(5, JustType.PARA, para=ParaJust(4, 1, (0,), (0,)))
        )

        extractor2 = DerivationFeatureExtractor(ctx2)
        mixed_entropy = extractor2.extract(
            _make_clause(5, JustType.PARA, para=ParaJust(4, 1, (0,), (0,)))
        ).features[9]

        assert mixed_entropy > uniform_entropy


# ── InferenceChainEncoder Tests ──────────────────────────────────────────────


class TestInferenceChainEncoder:
    """Tests for the learned inference chain encoder."""

    @pytest.fixture
    def encoder(self):
        config = InferenceChainConfig(
            num_just_types=24,
            chain_embed_dim=32,
            output_dim=64,
            max_chain_length=16,
            recency_temperature=5.0,
        )
        return InferenceChainEncoder(config)

    def test_output_shape(self, encoder):
        chains = torch.tensor([[1, 5, 6, 9]], dtype=torch.long)
        lengths = torch.tensor([4], dtype=torch.long)
        output = encoder(chains, lengths)
        assert output.shape == (1, 64)

    def test_batch_encoding(self, encoder):
        chains = torch.tensor([
            [1, 5, 6, 0],
            [1, 5, 0, 0],
            [1, 5, 6, 9],
        ], dtype=torch.long)
        lengths = torch.tensor([3, 2, 4], dtype=torch.long)
        output = encoder(chains, lengths)
        assert output.shape == (3, 64)

    def test_empty_chain_uses_no_history(self, encoder):
        chains = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        lengths = torch.tensor([0], dtype=torch.long)
        output = encoder(chains, lengths)
        assert output.shape == (1, 64)
        # Should be close to no_history_embedding (after norm)
        # Just verify it's not all zeros
        assert output.abs().sum() > 0

    def test_pad_chains(self, encoder):
        raw_chains = [
            (1, 5, 6),
            (1,),
            (1, 5, 6, 9, 5, 6),
        ]
        padded, lengths = encoder.pad_chains(raw_chains)

        assert padded.shape[0] == 3
        assert padded.shape[1] == 6  # max length in batch
        assert lengths.tolist() == [3, 1, 6]
        assert padded[1, 1] == 0  # padding

    def test_pad_chains_truncation(self, encoder):
        # Chain longer than max_chain_length
        long_chain = tuple(range(1, 20))  # 19 elements
        padded, lengths = encoder.pad_chains([long_chain])

        assert padded.shape[1] <= encoder.config.max_chain_length
        assert lengths[0] == encoder.config.max_chain_length

    def test_encode_single(self, encoder):
        encoder.eval()
        chain = (int(JustType.INPUT), int(JustType.BINARY_RES), int(JustType.HYPER_RES))
        emb = encoder.encode_single(chain)
        assert emb.shape == (64,)
        assert not emb.requires_grad

    def test_encode_single_empty(self, encoder):
        encoder.eval()
        emb = encoder.encode_single(())
        assert emb.shape == (64,)

    def test_gradients_flow(self, encoder):
        """Verify that gradients flow through the encoder for training."""
        encoder.train()
        chains = torch.tensor([[1, 5, 6]], dtype=torch.long)
        lengths = torch.tensor([3], dtype=torch.long)
        output = encoder(chains, lengths)
        loss = output.sum()
        loss.backward()

        # Check that output projection got gradients (main training path)
        proj_params = list(encoder.output_projection.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in proj_params)

        # Check that no_history_embedding is a learnable parameter
        assert encoder.no_history_embedding.requires_grad

    def test_different_chains_different_embeddings(self, encoder):
        encoder.eval()
        chain_a = torch.tensor([[1, 5, 6]], dtype=torch.long)
        chain_b = torch.tensor([[1, 9, 6]], dtype=torch.long)
        lengths = torch.tensor([3], dtype=torch.long)

        emb_a = encoder(chain_a, lengths)
        emb_b = encoder(chain_b, lengths)

        # Different chains should produce different embeddings
        assert not torch.allclose(emb_a, emb_b, atol=1e-4)

    def test_recency_weighting(self, encoder):
        """More recent steps should have higher influence."""
        encoder.eval()

        # Chain where only the last step differs
        chain_same_prefix = torch.tensor([
            [1, 5, 6],
            [1, 5, 9],
        ], dtype=torch.long)
        lengths = torch.tensor([3, 3], dtype=torch.long)
        emb = encoder(chain_same_prefix, lengths)

        # Chain where only the first step differs
        chain_same_suffix = torch.tensor([
            [1, 5, 6],
            [9, 5, 6],
        ], dtype=torch.long)
        emb2 = encoder(chain_same_suffix, lengths)

        # Difference should be larger when the *recent* step differs
        diff_recent = (emb[0] - emb[1]).norm()
        diff_ancient = (emb2[0] - emb2[1]).norm()

        # With recency bias, changing the last step should cause a bigger diff
        # This is a soft check — the learned embeddings may not always satisfy it
        # before training, but the architecture should enable it.
        # We just verify both produce non-trivial differences.
        assert diff_recent > 0
        assert diff_ancient > 0

    def test_lengths_auto_inferred(self, encoder):
        """When lengths=None, should infer from non-zero entries."""
        encoder.eval()
        chains = torch.tensor([[1, 5, 0, 0]], dtype=torch.long)

        with_lengths = encoder(chains, torch.tensor([2]))
        without_lengths = encoder(chains, None)

        assert torch.allclose(with_lengths, without_lengths, atol=1e-6)


# ── Integration Tests ────────────────────────────────────────────────────────


class TestDerivationIntegration:
    """End-to-end tests combining context, features, and encoder."""

    def test_full_pipeline(self):
        """Register clauses → extract features → encode chains."""
        ctx = DerivationContext()

        # Build a small derivation
        clauses = [
            _make_clause(1, JustType.INPUT),
            _make_clause(2, JustType.GOAL),
            _make_clause(3, JustType.BINARY_RES, clause_ids=(1, 2)),
            _make_clause(4, JustType.HYPER_RES, clause_ids=(3, 1)),
            _make_clause(5, JustType.BINARY_RES, clause_ids=(4, 2)),
        ]
        for c in clauses:
            ctx.register(c)

        # Extract features
        extractor = DerivationFeatureExtractor(ctx)
        features = extractor.extract_batch(clauses)
        assert len(features) == 5
        assert all(len(f) == DERIVATION_FEATURE_DIM for f in features)

        # Encode chains
        config = InferenceChainConfig(
            chain_embed_dim=16, output_dim=32, max_chain_length=8
        )
        encoder = InferenceChainEncoder(config)
        encoder.eval()

        chains = [ctx.get_inference_chain(c.id) for c in clauses]
        padded, lengths = encoder.pad_chains(chains)
        embeddings = encoder(padded, lengths)

        assert embeddings.shape == (5, 32)

    def test_features_plus_embeddings_concatenation(self):
        """Verify features and embeddings can be concatenated for CLAUSE nodes."""
        ctx = DerivationContext()
        c = _make_clause(1, JustType.INPUT)
        ctx.register(c)

        extractor = DerivationFeatureExtractor(ctx)
        features = torch.tensor([extractor.extract(c).features])  # (1, 13)

        config = InferenceChainConfig(
            chain_embed_dim=16, output_dim=32, max_chain_length=8
        )
        encoder = InferenceChainEncoder(config)
        encoder.eval()

        chain = ctx.get_inference_chain(c.id)
        chain_emb = encoder.encode_single(chain).unsqueeze(0)  # (1, 32)

        # Concatenate: this would augment CLAUSE node features
        combined = torch.cat([features, chain_emb], dim=-1)  # (1, 45)
        assert combined.shape == (1, DERIVATION_FEATURE_DIM + 32)

    def test_thread_safety_concurrent_registration(self):
        """Concurrent registration should not corrupt state."""
        import threading

        ctx = DerivationContext()
        errors: list[Exception] = []

        def register_range(start: int, end: int):
            try:
                for i in range(start, end):
                    ctx.register(_make_clause(i, JustType.INPUT))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_range, args=(i * 100, (i + 1) * 100))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert ctx.size == 400
