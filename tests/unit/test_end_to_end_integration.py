"""End-to-end integration validation for all ML enhancements.

Tests every combination of the three completed enhancements working together:
  1. Property-invariant embeddings (Sophia)
  2. Derivation history embeddings (Leonhard)
  3. Cross-clause attention (Ashish)

Validates:
- Each enhancement pair integrates correctly
- Triple combination produces valid results
- Graceful degradation when any subset is missing
- Invariance properties hold end-to-end
- Temporal weighting works through the full pipeline
- Performance stays within overhead budget
- Error handling and fallback paths

All tests use real PyLADR data structures (Clause, Literal, Term, JustType)
to validate genuine integration, not just tensor shapes.
"""

from __future__ import annotations

import math
import time

import pytest
import torch
import torch.nn as nn

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.term import Term

# Invariant embeddings (Sophia)
from pyladr.ml.invariant.canonicalization import canonicalize_clause
from pyladr.ml.invariant.invariant_features import (
    InvariantFeatureExtractor,
    invariant_clause_structural_hash,
)

# Derivation history (Leonhard)
from pyladr.ml.derivation.derivation_context import DerivationContext
from pyladr.ml.derivation.derivation_features import (
    DERIVATION_FEATURE_DIM,
    DerivationFeatureExtractor,
)
from pyladr.ml.derivation.attention_bridge import (
    DerivationAttentionAdapter,
    TemporalMetadata,
)

# Cross-clause attention (Ashish)
from pyladr.ml.attention.cross_clause import (
    CrossClauseAttentionConfig,
    CrossClauseAttentionScorer,
)
from pyladr.ml.attention.temporal_attention import (
    MultiSourceFusion,
    TemporalAttentionConfig,
    TemporalCrossClauseAttention,
    TemporalPositionEncoder,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def make_term(symnum: int, *args: Term) -> Term:
    return Term(private_symbol=-symnum, arity=len(args), args=tuple(args))


def make_var(varnum: int) -> Term:
    return Term(private_symbol=varnum)


def make_literal(sign: bool, atom: Term) -> Literal:
    return Literal(sign=sign, atom=atom)


def make_clause(
    literals: list[Literal],
    clause_id: int = 1,
    justification: tuple[Justification, ...] = (),
) -> Clause:
    return Clause(
        id=clause_id,
        literals=tuple(literals),
        justification=justification,
    )


def _small_attn_config() -> CrossClauseAttentionConfig:
    return CrossClauseAttentionConfig(
        enabled=True,
        embedding_dim=32,
        num_heads=4,
        head_dim=8,
        dropout=0.0,
        use_relative_position=True,
        max_clauses=64,
        scoring_hidden_dim=16,
    )


def _random_embs(n: int, d: int, seed: int = 42) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=gen)


def _build_derivation_chain():
    """Build a realistic derivation chain of clauses with justifications.

    Returns a list of clauses forming a derivation DAG:
      clause 1: P(a) [input]
      clause 2: -P(x) | Q(x) [input]
      clause 3: Q(a) [binary_res from 1, 2]
      clause 4: -Q(x) | R(x) [input]
      clause 5: R(a) [binary_res from 3, 4]
    """
    a = make_term(1)  # constant a
    x = make_var(0)

    # Clause 1: P(a)  [input]
    Pa = make_term(10, a)
    c1 = make_clause(
        [make_literal(True, Pa)],
        clause_id=1,
        justification=(Justification(just_type=JustType.INPUT),),
    )

    # Clause 2: -P(x) | Q(x)  [input]
    Px = make_term(10, x)
    Qx = make_term(20, x)
    c2 = make_clause(
        [make_literal(False, Px), make_literal(True, Qx)],
        clause_id=2,
        justification=(Justification(just_type=JustType.INPUT),),
    )

    # Clause 3: Q(a)  [binary_res from 1, 2]
    Qa = make_term(20, a)
    c3 = make_clause(
        [make_literal(True, Qa)],
        clause_id=3,
        justification=(Justification(
            just_type=JustType.BINARY_RES, clause_ids=(1, 2)
        ),),
    )

    # Clause 4: -Q(x) | R(x)  [input]
    Rx = make_term(30, x)
    c4 = make_clause(
        [make_literal(False, Qx), make_literal(True, Rx)],
        clause_id=4,
        justification=(Justification(just_type=JustType.INPUT),),
    )

    # Clause 5: R(a)  [binary_res from 3, 4]
    Ra = make_term(30, a)
    c5 = make_clause(
        [make_literal(True, Ra)],
        clause_id=5,
        justification=(Justification(
            just_type=JustType.BINARY_RES, clause_ids=(3, 4)
        ),),
    )

    return [c1, c2, c3, c4, c5]


def _build_renamed_derivation_chain():
    """Same derivation structure as above but with all symbols renamed.

    Same structure, different symbol IDs:
      clause 6: S(b) [input]           (was P(a))
      clause 7: -S(x) | T(x) [input]  (was -P(x) | Q(x))
      clause 8: T(b) [binary_res]      (was Q(a))
      clause 9: -T(x) | U(x) [input]  (was -Q(x) | R(x))
      clause 10: U(b) [binary_res]     (was R(a))
    """
    b = make_term(100)  # constant b (was a=1)
    x = make_var(0)

    Sb = make_term(110, b)  # was P=10
    c6 = make_clause(
        [make_literal(True, Sb)],
        clause_id=6,
        justification=(Justification(just_type=JustType.INPUT),),
    )

    Sx = make_term(110, x)
    Tx = make_term(120, x)  # was Q=20
    c7 = make_clause(
        [make_literal(False, Sx), make_literal(True, Tx)],
        clause_id=7,
        justification=(Justification(just_type=JustType.INPUT),),
    )

    Tb = make_term(120, b)
    c8 = make_clause(
        [make_literal(True, Tb)],
        clause_id=8,
        justification=(Justification(
            just_type=JustType.BINARY_RES, clause_ids=(6, 7)
        ),),
    )

    Ux = make_term(130, x)  # was R=30
    c9 = make_clause(
        [make_literal(False, Tx), make_literal(True, Ux)],
        clause_id=9,
        justification=(Justification(just_type=JustType.INPUT),),
    )

    Ub = make_term(130, b)
    c10 = make_clause(
        [make_literal(True, Ub)],
        clause_id=10,
        justification=(Justification(
            just_type=JustType.BINARY_RES, clause_ids=(8, 9)
        ),),
    )

    return [c6, c7, c8, c9, c10]


# ── Enhancement Pair: Invariant + Derivation History ─────────────────────


class TestInvariantPlusDerivation:
    """Validate invariant embeddings working with derivation history features."""

    def test_invariant_hash_stable_across_derivation_depths(self):
        """Invariant hash captures structure, not symbol names or derivation."""
        chain = _build_derivation_chain()
        # P(a) and Q(a) are structurally identical (unary pred + constant)
        # so invariant hash correctly treats them as equivalent
        h1 = invariant_clause_structural_hash(chain[0])  # P(a)
        h3 = invariant_clause_structural_hash(chain[2])  # Q(a)
        assert h1 == h3  # same structure: unary predicate applied to constant

        # Multi-literal vs single-literal should differ
        h2 = invariant_clause_structural_hash(chain[1])  # -P(x) | Q(x)
        assert h1 != h2  # different structure: 1 literal vs 2 literals

    def test_renamed_chain_pairs_have_matching_hashes(self):
        """Structurally identical clauses in renamed chains should match."""
        chain1 = _build_derivation_chain()
        chain2 = _build_renamed_derivation_chain()

        for c1, c2 in zip(chain1, chain2):
            h1 = invariant_clause_structural_hash(c1)
            h2 = invariant_clause_structural_hash(c2)
            assert h1 == h2, (
                f"Clause pair ({c1.id}, {c2.id}) should have matching "
                f"invariant hash but got {h1} != {h2}"
            )

    def test_derivation_features_independent_of_invariant_hash(self):
        """Derivation features capture history, not structure —
        they should differ even when invariant hashes match."""
        chain1 = _build_derivation_chain()
        chain2 = _build_renamed_derivation_chain()

        ctx1 = DerivationContext()
        ctx2 = DerivationContext()
        for c in chain1:
            ctx1.register(c)
        for c in chain2:
            ctx2.register(c)

        ext1 = DerivationFeatureExtractor(ctx1)
        ext2 = DerivationFeatureExtractor(ctx2)

        # Corresponding clauses (same position in chain) should have
        # same derivation features (same derivation structure)
        for c1, c2 in zip(chain1, chain2):
            f1 = ext1.extract(c1)
            f2 = ext2.extract(c2)
            # Depth should match
            assert f1.depth == f2.depth, f"Depth mismatch for pair ({c1.id}, {c2.id})"
            # is_input should match
            assert f1.is_input == f2.is_input
            # num_parents should match
            assert f1.num_parents == f2.num_parents

    def test_derivation_depth_progression(self):
        """Derivation depths should increase along the chain."""
        chain = _build_derivation_chain()
        ctx = DerivationContext()
        for c in chain:
            ctx.register(c)

        assert ctx.get_depth(1) == 0  # input
        assert ctx.get_depth(2) == 0  # input
        assert ctx.get_depth(3) == 1  # resolved from inputs
        assert ctx.get_depth(4) == 0  # input
        assert ctx.get_depth(5) == 2  # resolved from depth-1 + input


# ── Enhancement Pair: Derivation History + Attention ─────────────────────


class TestDerivationPlusAttention:
    """Validate derivation history working with cross-clause attention."""

    @pytest.fixture
    def pipeline_config(self):
        return TemporalAttentionConfig(
            base_config=_small_attn_config(),
            use_temporal_encoding=True,
            temporal_dim=32,
            max_derivation_depth=50,
            num_inference_types=22,
            use_multi_source_fusion=False,
            hierarchical_dim=0,
            invariant_dim=0,
        )

    def test_real_derivation_metadata_flows_through_attention(self, pipeline_config):
        """Real DerivationContext metadata should flow through the temporal
        attention pipeline and produce valid scores."""
        chain = _build_derivation_chain()
        ctx = DerivationContext()
        for c in chain:
            ctx.register(c)

        adapter = DerivationAttentionAdapter(ctx)
        meta = adapter.extract_metadata(chain)

        model = TemporalCrossClauseAttention(pipeline_config)
        model.eval()

        N = len(chain)
        base_embs = _random_embs(N, 32)

        scores = model.score_clauses(
            base_embs,
            derivation_depths=meta.derivation_depths,
            inference_types=meta.inference_types,
            parent_counts=meta.parent_counts,
            clause_ids=meta.clause_ids,
        )

        assert len(scores) == N
        assert all(math.isfinite(s) for s in scores)

    def test_temporal_metadata_tensor_shapes(self):
        """TemporalMetadata tensors should have correct shapes."""
        chain = _build_derivation_chain()
        ctx = DerivationContext()
        for c in chain:
            ctx.register(c)

        adapter = DerivationAttentionAdapter(ctx)
        meta = adapter.extract_metadata(chain)

        N = len(chain)
        assert meta.derivation_depths.shape == (N,)
        assert meta.inference_types.shape == (N,)
        assert meta.parent_counts.shape == (N,)
        assert meta.clause_ids.shape == (N,)

        # Check actual values
        assert meta.derivation_depths[0].item() == 0  # input clause
        assert meta.derivation_depths[2].item() == 1  # first resolution
        assert meta.derivation_depths[4].item() == 2  # second resolution
        assert meta.inference_types[0].item() == int(JustType.INPUT)
        assert meta.inference_types[2].item() == int(JustType.BINARY_RES)
        assert meta.parent_counts[0].item() == 0  # input: no parents
        assert meta.parent_counts[2].item() == 2  # resolved from 2 parents

    def test_temporal_encoding_differentiates_derivation_stages(self, pipeline_config):
        """Clauses at different derivation stages should receive different
        temporal scores when other features are identical."""
        model = TemporalCrossClauseAttention(pipeline_config)
        model.eval()

        N = 4
        base_embs = _random_embs(N, 32)

        # All inputs (depth 0)
        depths_all_input = torch.tensor([0, 0, 0, 0])
        types_all_input = torch.tensor([0, 0, 0, 0])  # INPUT
        parents_none = torch.tensor([0, 0, 0, 0])

        # Mixed depths (realistic derivation)
        depths_mixed = torch.tensor([0, 0, 1, 2])
        types_mixed = torch.tensor([0, 0, 5, 5])  # INPUT, INPUT, BINARY_RES, BINARY_RES
        parents_mixed = torch.tensor([0, 0, 2, 2])

        scores_input = model.score_clauses(
            base_embs, derivation_depths=depths_all_input,
            inference_types=types_all_input, parent_counts=parents_none,
        )
        scores_mixed = model.score_clauses(
            base_embs, derivation_depths=depths_mixed,
            inference_types=types_mixed, parent_counts=parents_mixed,
        )

        # Different temporal context → different scores
        assert scores_input != scores_mixed

    def test_unregistered_clauses_get_default_metadata(self):
        """Clauses not in the derivation context should get safe defaults."""
        ctx = DerivationContext()
        adapter = DerivationAttentionAdapter(ctx)

        # Clause not registered
        x = make_var(0)
        atom = make_term(1, x)
        unregistered = make_clause([make_literal(True, atom)], clause_id=999)

        meta = adapter.extract_metadata([unregistered])
        assert meta.derivation_depths[0].item() == 0
        assert meta.inference_types[0].item() == 0
        assert meta.parent_counts[0].item() == 0


# ── Enhancement Pair: Invariant + Attention ──────────────────────────────


class TestInvariantPlusAttention:
    """Validate invariant embeddings with cross-clause attention scoring."""

    def test_invariant_embeddings_produce_consistent_attention(self):
        """If invariant provider produces identical embeddings for renamed
        clauses, attention scores must also be identical."""
        config = _small_attn_config()
        scorer = CrossClauseAttentionScorer(config)
        scorer.eval()

        # Simulate: two sets of 5 clauses where structurally identical pairs
        # get the same embeddings from InvariantEmbeddingProvider
        embs = _random_embs(5, 32)

        scores1 = scorer.score_clauses(embs)
        scores2 = scorer.score_clauses(embs)  # identical input

        assert scores1 == scores2, (
            "Identical invariant embeddings must produce identical attention scores"
        )

    def test_canonical_features_match_across_renamed_sets(self):
        """Verify that renamed clause sets produce matching canonical features."""
        chain1 = _build_derivation_chain()
        chain2 = _build_renamed_derivation_chain()

        for c1, c2 in zip(chain1, chain2):
            m1 = canonicalize_clause(c1)
            m2 = canonicalize_clause(c2)
            assert m1.next_id == m2.next_id, (
                f"Canonical ID count mismatch for ({c1.id}, {c2.id})"
            )


# ── Triple Enhancement: Invariant + Derivation + Attention ───────────────


class TestTripleEnhancementIntegration:
    """Test all three completed enhancements working together."""

    @pytest.fixture
    def full_config(self):
        return TemporalAttentionConfig(
            base_config=_small_attn_config(),
            use_temporal_encoding=True,
            temporal_dim=32,
            max_derivation_depth=50,
            num_inference_types=22,
            use_multi_source_fusion=True,
            hierarchical_dim=0,
            invariant_dim=32,
            fusion_method="gate",
        )

    def test_complete_pipeline_with_real_clauses(self, full_config):
        """Full pipeline: real clauses → invariant features + derivation
        metadata → temporal attention → relational scores."""
        chain = _build_derivation_chain()
        N = len(chain)

        # Derivation context
        ctx = DerivationContext()
        for c in chain:
            ctx.register(c)
        adapter = DerivationAttentionAdapter(ctx)
        meta = adapter.extract_metadata(chain)

        # Simulate embeddings (would come from providers in real system)
        base_embs = _random_embs(N, 32)
        invariant_embs = _random_embs(N, 32, seed=99)

        model = TemporalCrossClauseAttention(full_config)
        model.eval()

        scores = model.score_clauses(
            base_embs,
            invariant_embeddings=invariant_embs,
            derivation_depths=meta.derivation_depths,
            inference_types=meta.inference_types,
            parent_counts=meta.parent_counts,
            clause_ids=meta.clause_ids,
        )

        assert len(scores) == N
        assert all(math.isfinite(s) for s in scores)

    def test_renamed_chains_get_same_scores_with_same_embeddings(self, full_config):
        """If invariant provider gives identical embeddings for renamed chains,
        and derivation structure matches, scores should be identical."""
        chain1 = _build_derivation_chain()
        chain2 = _build_renamed_derivation_chain()
        N = len(chain1)

        # Register both chains in separate contexts
        ctx1 = DerivationContext()
        ctx2 = DerivationContext()
        for c in chain1:
            ctx1.register(c)
        for c in chain2:
            ctx2.register(c)

        adapter1 = DerivationAttentionAdapter(ctx1)
        adapter2 = DerivationAttentionAdapter(ctx2)
        meta1 = adapter1.extract_metadata(chain1)
        meta2 = adapter2.extract_metadata(chain2)

        # Derivation depths should match
        assert torch.equal(meta1.derivation_depths, meta2.derivation_depths)
        assert torch.equal(meta1.inference_types, meta2.inference_types)
        assert torch.equal(meta1.parent_counts, meta2.parent_counts)

        # Use same embeddings (simulating invariant provider output)
        base_embs = _random_embs(N, 32)
        invariant_embs = _random_embs(N, 32, seed=99)

        model = TemporalCrossClauseAttention(full_config)
        model.eval()

        # Clause IDs differ but we're testing with same relative positions
        scores1 = model.score_clauses(
            base_embs, invariant_embeddings=invariant_embs,
            derivation_depths=meta1.derivation_depths,
            inference_types=meta1.inference_types,
            parent_counts=meta1.parent_counts,
        )
        scores2 = model.score_clauses(
            base_embs, invariant_embeddings=invariant_embs,
            derivation_depths=meta2.derivation_depths,
            inference_types=meta2.inference_types,
            parent_counts=meta2.parent_counts,
        )

        # Same embeddings + same derivation structure → same scores
        assert scores1 == scores2

    def test_each_enhancement_contributes(self, full_config):
        """Verify that each enhancement actually changes scores (not ignored)."""
        N = 5
        base_embs = _random_embs(N, 32)
        invariant_embs = _random_embs(N, 32, seed=99)
        depths = torch.tensor([0, 0, 1, 2, 3])
        types = torch.tensor([0, 0, 5, 5, 5])
        parents = torch.tensor([0, 0, 2, 2, 2])

        model = TemporalCrossClauseAttention(full_config)
        model.eval()

        # Base only (no invariant, no temporal)
        scores_base = model.score_clauses(base_embs)

        # Base + invariant
        scores_inv = model.score_clauses(
            base_embs, invariant_embeddings=invariant_embs,
        )

        # Base + temporal
        scores_temp = model.score_clauses(
            base_embs,
            derivation_depths=depths, inference_types=types, parent_counts=parents,
        )

        # Base + invariant + temporal
        scores_all = model.score_clauses(
            base_embs, invariant_embeddings=invariant_embs,
            derivation_depths=depths, inference_types=types, parent_counts=parents,
        )

        # Each should produce different results
        assert scores_base != scores_inv, "Invariant should change scores"
        assert scores_base != scores_temp, "Temporal should change scores"
        assert scores_inv != scores_all, "Adding temporal to invariant should change scores"
        assert scores_temp != scores_all, "Adding invariant to temporal should change scores"


# ── Graceful Degradation ─────────────────────────────────────────────────


class TestGracefulDegradation:
    """Test that the system degrades gracefully when enhancements are missing."""

    @pytest.fixture
    def config(self):
        return TemporalAttentionConfig(
            base_config=_small_attn_config(),
            use_temporal_encoding=True,
            temporal_dim=32,
            use_multi_source_fusion=True,
            hierarchical_dim=16,
            invariant_dim=32,
            fusion_method="gate",
        )

    def test_missing_all_optional_inputs(self, config):
        """System should work with only base embeddings."""
        model = TemporalCrossClauseAttention(config)
        model.eval()
        base = _random_embs(5, 32)
        scores = model.score_clauses(base)
        assert len(scores) == 5
        assert all(math.isfinite(s) for s in scores)

    def test_missing_invariant_only(self, config):
        """Should work without invariant embeddings."""
        model = TemporalCrossClauseAttention(config)
        model.eval()
        base = _random_embs(5, 32)
        depths = torch.tensor([0, 1, 2, 3, 4])
        scores = model.score_clauses(base, derivation_depths=depths)
        assert len(scores) == 5
        assert all(math.isfinite(s) for s in scores)

    def test_missing_temporal_only(self, config):
        """Should work without temporal metadata."""
        model = TemporalCrossClauseAttention(config)
        model.eval()
        base = _random_embs(5, 32)
        inv = _random_embs(5, 32, seed=99)
        scores = model.score_clauses(base, invariant_embeddings=inv)
        assert len(scores) == 5
        assert all(math.isfinite(s) for s in scores)

    def test_single_clause_all_enhancements(self, config):
        """Single clause should work with all enhancements."""
        model = TemporalCrossClauseAttention(config)
        model.eval()
        base = _random_embs(1, 32)
        inv = _random_embs(1, 32, seed=99)
        depths = torch.tensor([3])
        types = torch.tensor([5])
        parents = torch.tensor([2])
        scores = model.score_clauses(
            base, invariant_embeddings=inv,
            derivation_depths=depths, inference_types=types, parent_counts=parents,
        )
        assert len(scores) == 1
        assert math.isfinite(scores[0])

    def test_empty_derivation_context_safe(self):
        """Empty derivation context should not crash the pipeline."""
        ctx = DerivationContext()
        adapter = DerivationAttentionAdapter(ctx)

        x = make_var(0)
        atom = make_term(1, x)
        clauses = [make_clause([make_literal(True, atom)], clause_id=i) for i in range(5)]
        meta = adapter.extract_metadata(clauses)

        # All defaults (depth=0, type=0, parents=0)
        assert (meta.derivation_depths == 0).all()
        assert (meta.inference_types == 0).all()
        assert (meta.parent_counts == 0).all()


# ── Derivation Feature Quality ───────────────────────────────────────────


class TestDerivationFeatureQuality:
    """Validate derivation feature extraction produces meaningful values."""

    def test_feature_dimensions(self):
        chain = _build_derivation_chain()
        ctx = DerivationContext()
        for c in chain:
            ctx.register(c)
        ext = DerivationFeatureExtractor(ctx)

        for c in chain:
            f = ext.extract(c)
            assert f.dim == DERIVATION_FEATURE_DIM == 13

    def test_input_clause_features(self):
        chain = _build_derivation_chain()
        ctx = DerivationContext()
        for c in chain:
            ctx.register(c)
        ext = DerivationFeatureExtractor(ctx)

        f = ext.extract(chain[0])  # P(a), input clause
        assert f.is_input
        assert f.depth == 0.0
        assert f.num_parents == 0.0

    def test_derived_clause_features(self):
        chain = _build_derivation_chain()
        ctx = DerivationContext()
        for c in chain:
            ctx.register(c)
        ext = DerivationFeatureExtractor(ctx)

        f = ext.extract(chain[2])  # Q(a), resolved from 1 and 2
        assert not f.is_input
        assert f.depth > 0.0
        assert f.num_parents == 2.0

    def test_batch_extraction_consistency(self):
        chain = _build_derivation_chain()
        ctx = DerivationContext()
        for c in chain:
            ctx.register(c)
        ext = DerivationFeatureExtractor(ctx)

        batch = ext.extract_batch(chain)
        individual = [ext.extract(c).features for c in chain]

        assert len(batch) == len(individual)
        for b, i in zip(batch, individual):
            assert b == i


# ── Performance Validation ───────────────────────────────────────────────


class TestPerformanceValidation:
    """Validate that combined system stays within overhead budget."""

    def test_triple_enhancement_pipeline_latency(self):
        """Full triple-enhancement pipeline should complete within 100ms
        for typical SOS sizes."""
        config = TemporalAttentionConfig(
            base_config=CrossClauseAttentionConfig(
                enabled=True, embedding_dim=64, num_heads=8, head_dim=8,
                dropout=0.0, max_clauses=512, scoring_hidden_dim=32,
            ),
            use_temporal_encoding=True,
            temporal_dim=64,
            use_multi_source_fusion=True,
            hierarchical_dim=0,
            invariant_dim=64,
            fusion_method="gate",
        )
        model = TemporalCrossClauseAttention(config)
        model.eval()

        for n_clauses in [50, 100, 200]:
            base = _random_embs(n_clauses, 64)
            inv = _random_embs(n_clauses, 64, seed=99)
            depths = torch.randint(0, 50, (n_clauses,))
            types = torch.randint(0, 22, (n_clauses,))
            parents = torch.randint(0, 5, (n_clauses,))

            # Warm up
            model.score_clauses(base, inv, derivation_depths=depths,
                              inference_types=types, parent_counts=parents)

            start = time.perf_counter()
            iterations = 5
            for _ in range(iterations):
                model.score_clauses(base, inv, derivation_depths=depths,
                                  inference_types=types, parent_counts=parents)
            elapsed = (time.perf_counter() - start) / iterations

            assert elapsed < 0.1, (
                f"Triple-enhancement pipeline too slow for {n_clauses} clauses: "
                f"{elapsed*1000:.1f}ms (limit: 100ms)"
            )

    def test_derivation_context_registration_speed(self):
        """Registering 1000 clauses should be fast."""
        ctx = DerivationContext()
        x = make_var(0)

        clauses = []
        for i in range(1000):
            atom = make_term(1, x)
            just = ()
            if i > 0:
                just = (Justification(
                    just_type=JustType.BINARY_RES,
                    clause_ids=(max(0, i - 1), max(0, i - 2)),
                ),)
            clauses.append(make_clause([make_literal(True, atom)], clause_id=i, justification=just))

        start = time.perf_counter()
        for c in clauses:
            ctx.register(c)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, (
            f"Registering 1000 clauses took {elapsed*1000:.1f}ms (limit: 500ms)"
        )
        assert ctx.size == 1000
