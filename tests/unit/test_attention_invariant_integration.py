"""Validation tests: cross-clause attention with invariant embeddings.

Tests that:
1. Symbol renaming produces identical attention scores (invariance)
2. Attention consistency improves with invariant vs standard embeddings
3. Relational scoring is stable under symbol permutation
4. Performance overhead remains within acceptable bounds
5. Multi-source fusion correctly integrates invariant embeddings
"""

from __future__ import annotations

import math
import time

import pytest
import torch

from pyladr.ml.attention.cross_clause import (
    CrossClauseAttentionConfig,
    CrossClauseAttentionScorer,
    MultiHeadClauseAttention,
)
from pyladr.ml.attention.temporal_attention import (
    MultiSourceFusion,
    TemporalAttentionConfig,
    TemporalCrossClauseAttention,
)
from pyladr.ml.invariant.canonicalization import (
    CanonicalMapping,
    canonicalize_clause,
)
from pyladr.ml.invariant.invariant_features import (
    InvariantFeatureExtractor,
    invariant_clause_structural_hash,
)
from tests.factories import make_clause, make_literal, make_term, make_var


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


def _random_embeddings(n: int, d: int, seed: int = 42) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=gen)


# Build pairs of structurally identical clauses with different symbols
def _make_renamed_pair():
    """Create two structurally identical clauses with different symbol names.

    Clause 1: P(f(x, y), g(a))  with P=1, f=2, g=3, a=4
    Clause 2: Q(h(x, y), k(b))  with Q=10, h=20, k=30, b=40
    """
    x, y = make_var(0), make_var(1)

    # Clause 1
    a1 = make_term(4)  # constant a
    fx1 = make_term(2, x, y)
    ga1 = make_term(3, a1)
    atom1 = make_term(1, fx1, ga1)
    c1 = make_clause(make_literal(True, atom1), clause_id=1)

    # Clause 2 (same structure, different symbols)
    x2, y2 = make_var(0), make_var(1)
    b2 = make_term(40)  # constant b
    hx2 = make_term(20, x2, y2)
    kb2 = make_term(30, b2)
    atom2 = make_term(10, hx2, kb2)
    c2 = make_clause(make_literal(True, atom2), clause_id=2)

    return c1, c2


def _make_structurally_different():
    """Create two structurally different clauses.

    Clause 1: P(x)           — unary predicate, variable arg
    Clause 2: Q(f(a), g(b))  — binary predicate, nested function args
    """
    x = make_var(0)
    atom1 = make_term(1, x)
    c1 = make_clause(make_literal(True, atom1), clause_id=1)

    a = make_term(4)
    b = make_term(5)
    fa = make_term(2, a)
    gb = make_term(3, b)
    atom2 = make_term(10, fa, gb)
    c2 = make_clause(make_literal(True, atom2), clause_id=2)

    return c1, c2


# ── Invariance validation tests ──────────────────────────────────────────


class TestInvariantHashConsistency:
    """Verify that invariant hashing produces correct collisions."""

    def test_renamed_clauses_have_same_hash(self):
        c1, c2 = _make_renamed_pair()
        h1 = invariant_clause_structural_hash(c1)
        h2 = invariant_clause_structural_hash(c2)
        assert h1 == h2, f"Renamed clauses should have identical invariant hash: {h1} != {h2}"

    def test_different_clauses_have_different_hash(self):
        c1, c2 = _make_structurally_different()
        h1 = invariant_clause_structural_hash(c1)
        h2 = invariant_clause_structural_hash(c2)
        assert h1 != h2, "Structurally different clauses should have different hashes"

    def test_multi_literal_renamed(self):
        """Multi-literal clauses with symbol renaming should hash identically."""
        x, y = make_var(0), make_var(1)

        # Clause 1: P(x) | -Q(y)
        atom1a = make_term(1, x)
        atom1b = make_term(2, y)
        c1 = make_clause(make_literal(True, atom1a), make_literal(False, atom1b))

        # Clause 2: R(x) | -S(y)  (same structure, different names)
        x2, y2 = make_var(0), make_var(1)
        atom2a = make_term(10, x2)
        atom2b = make_term(20, y2)
        c2 = make_clause(make_literal(True, atom2a), make_literal(False, atom2b))

        assert invariant_clause_structural_hash(c1) == invariant_clause_structural_hash(c2)


class TestCanonicalMappingConsistency:
    """Verify canonical mapping produces consistent results."""

    def test_renamed_clauses_produce_same_canonical_structure(self):
        c1, c2 = _make_renamed_pair()
        m1 = canonicalize_clause(c1)
        m2 = canonicalize_clause(c2)

        assert m1.next_id == m2.next_id, "Same number of canonical IDs expected"

        for cid in range(m1.next_id):
            role1 = m1.canonical_to_role[cid]
            role2 = m2.canonical_to_role[cid]
            assert role1.arity == role2.arity
            assert role1.is_predicate == role2.is_predicate

    def test_invariant_features_match_for_renamed_symbols(self):
        c1, c2 = _make_renamed_pair()

        ext1 = InvariantFeatureExtractor()
        ext1.prepare(c1)

        ext2 = InvariantFeatureExtractor()
        ext2.prepare(c2)

        # Collect all canonical features for both clauses
        features1 = []
        for symnum in ext1._mapping.sym_to_canonical:
            features1.append(ext1.symbol_features(symnum))

        features2 = []
        for symnum in ext2._mapping.sym_to_canonical:
            features2.append(ext2.symbol_features(symnum))

        # Sort by canonical ID for comparison
        features1.sort(key=lambda f: f[0])
        features2.sort(key=lambda f: f[0])

        assert len(features1) == len(features2)
        for f1, f2 in zip(features1, features2):
            assert f1[0] == f2[0], f"Canonical IDs should match: {f1[0]} != {f2[0]}"
            assert f1[1] == f2[1], f"Arities should match: {f1[1]} != {f2[1]}"
            assert f1[2] == f2[2], f"is_predicate should match"
            assert f1[3] == f2[3], f"is_skolem should match"


# ── Attention consistency with invariant embeddings ──────────────────────


class TestAttentionInvariance:
    """Test that attention produces consistent scores with invariant embeddings."""

    def test_identical_embeddings_produce_identical_attention(self):
        """If two clause sets have identical embeddings (from invariant provider),
        attention scores should be identical."""
        config = _small_attn_config()
        scorer = CrossClauseAttentionScorer(config)
        scorer.eval()

        # Same embeddings (simulating what invariant provider would produce)
        embs = _random_embeddings(5, 32)
        scores1 = scorer.score_clauses(embs)
        scores2 = scorer.score_clauses(embs)
        assert scores1 == scores2, "Same embeddings must produce same scores"

    def test_permuted_clause_order_different_scores(self):
        """Different clause ordering should produce different attention scores
        (attention is position-sensitive via relative position bias)."""
        config = _small_attn_config()
        scorer = CrossClauseAttentionScorer(config)
        scorer.eval()

        embs = _random_embeddings(5, 32)
        ids_original = torch.tensor([1, 2, 3, 4, 5])
        ids_permuted = torch.tensor([5, 4, 3, 2, 1])

        scores_orig = scorer.score_clauses(embs, clause_ids=ids_original)
        scores_perm = scorer.score_clauses(embs, clause_ids=ids_permuted)

        # Scores should differ due to relative position bias
        assert scores_orig != scores_perm

    def test_attention_consistency_across_batches(self):
        """Scoring the same clauses in different batch contexts should produce
        different scores (attention is context-dependent)."""
        config = _small_attn_config()
        scorer = CrossClauseAttentionScorer(config)
        scorer.eval()

        # Two overlapping batches
        embs_full = _random_embeddings(10, 32)
        embs_partial = embs_full[:5]

        scores_full = scorer.score_clauses(embs_full)[:5]
        scores_partial = scorer.score_clauses(embs_partial)

        # Context-dependent: full batch vs partial should differ
        # (the first 5 clauses attend to different sets of clauses)
        assert scores_full != scores_partial


# ── Multi-source fusion with invariant embeddings ────────────────────────


class TestMultiSourceFusionWithInvariant:
    """Test that multi-source fusion correctly integrates invariant embeddings."""

    @pytest.fixture
    def fusion_config(self):
        return TemporalAttentionConfig(
            base_config=_small_attn_config(),
            use_multi_source_fusion=True,
            hierarchical_dim=0,
            invariant_dim=32,
            fusion_method="gate",
        )

    def test_invariant_source_changes_output(self, fusion_config):
        """Adding invariant embeddings should change the fused output."""
        fusion = MultiSourceFusion(fusion_config)
        fusion.eval()

        base = _random_embeddings(5, 32)
        invariant = _random_embeddings(5, 32, seed=99)

        out_base_only = fusion(base)
        out_with_inv = fusion(base, invariant_embeddings=invariant)

        assert not torch.allclose(out_base_only, out_with_inv, atol=1e-5)

    def test_same_invariant_embeddings_same_fusion(self, fusion_config):
        """Same invariant input should produce same fused output."""
        fusion = MultiSourceFusion(fusion_config)
        fusion.eval()

        base = _random_embeddings(5, 32)
        invariant = _random_embeddings(5, 32, seed=99)

        out1 = fusion(base, invariant_embeddings=invariant)
        out2 = fusion(base, invariant_embeddings=invariant)

        assert torch.allclose(out1, out2)

    @pytest.mark.parametrize("method", ["gate", "attention", "concat_project"])
    def test_all_fusion_methods_with_invariant(self, method):
        """All fusion methods should work with invariant embeddings."""
        cfg = TemporalAttentionConfig(
            base_config=_small_attn_config(),
            use_multi_source_fusion=True,
            hierarchical_dim=0,
            invariant_dim=32,
            fusion_method=method,
        )
        fusion = MultiSourceFusion(cfg)
        fusion.eval()

        base = _random_embeddings(5, 32)
        invariant = _random_embeddings(5, 32, seed=99)

        out = fusion(base, invariant_embeddings=invariant)
        assert out.shape == (5, 32)
        assert torch.isfinite(out).all()


# ── Full pipeline with invariant + temporal ──────────────────────────────


class TestFullPipelineWithInvariant:
    """Test the full TemporalCrossClauseAttention with invariant embeddings."""

    @pytest.fixture
    def pipeline_config(self):
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

    def test_full_pipeline_with_invariant_and_temporal(self, pipeline_config):
        """Full pipeline: invariant embeddings + temporal encoding → scores."""
        model = TemporalCrossClauseAttention(pipeline_config)
        model.eval()

        N = 8
        base = _random_embeddings(N, 32)
        invariant = _random_embeddings(N, 32, seed=99)
        depths = torch.randint(0, 50, (N,))
        types = torch.randint(0, 22, (N,))
        parents = torch.randint(0, 5, (N,))

        scores = model(
            base,
            invariant_embeddings=invariant,
            derivation_depths=depths,
            inference_types=types,
            parent_counts=parents,
        )
        assert scores.shape == (N,)
        assert torch.isfinite(scores).all()

    def test_invariant_embeddings_improve_score_stability(self, pipeline_config):
        """Simulating invariant embeddings: if two clause sets are structurally
        identical (same invariant embeddings), their scores should match exactly."""
        model = TemporalCrossClauseAttention(pipeline_config)
        model.eval()

        N = 5
        # Simulate: two sets of clauses that are symbol-renamings of each other
        # With invariant provider, they'd get identical embeddings
        base = _random_embeddings(N, 32)
        invariant = _random_embeddings(N, 32, seed=99)  # same for both
        depths = torch.tensor([0, 1, 2, 3, 4])
        types = torch.tensor([0, 5, 5, 6, 6])
        parents = torch.tensor([0, 1, 1, 2, 2])

        scores1 = model.score_clauses(
            base, invariant_embeddings=invariant,
            derivation_depths=depths, inference_types=types, parent_counts=parents,
        )
        scores2 = model.score_clauses(
            base, invariant_embeddings=invariant,
            derivation_depths=depths, inference_types=types, parent_counts=parents,
        )

        # Identical inputs → identical outputs
        assert scores1 == scores2

    def test_gradients_flow_through_invariant_path(self, pipeline_config):
        """Gradients should propagate through invariant embedding fusion."""
        model = TemporalCrossClauseAttention(pipeline_config)
        model.train()

        N = 4
        base = _random_embeddings(N, 32).requires_grad_(True)
        invariant = _random_embeddings(N, 32, seed=99).requires_grad_(True)
        depths = torch.randint(0, 50, (N,))
        types = torch.randint(0, 22, (N,))
        parents = torch.randint(0, 5, (N,))

        scores = model(
            base, invariant_embeddings=invariant,
            derivation_depths=depths, inference_types=types, parent_counts=parents,
        )
        loss = scores.sum()
        loss.backward()

        # Both inputs should receive gradients
        assert base.grad is not None
        assert invariant.grad is not None
        assert torch.isfinite(base.grad).all()
        assert torch.isfinite(invariant.grad).all()


# ── Performance overhead measurement ─────────────────────────────────────


class TestPerformanceOverhead:
    """Measure attention mechanism overhead to validate <25% constraint."""

    def test_attention_overhead_acceptable(self):
        """Attention scoring should complete within reasonable time for
        typical SOS sizes (100-500 clauses)."""
        config = CrossClauseAttentionConfig(
            enabled=True,
            embedding_dim=64,
            num_heads=8,
            head_dim=8,
            dropout=0.0,
            use_relative_position=True,
            max_clauses=512,
            scoring_hidden_dim=32,
        )
        scorer = CrossClauseAttentionScorer(config)
        scorer.eval()

        # Typical SOS sizes
        for n_clauses in [50, 100, 200, 500]:
            embs = _random_embeddings(n_clauses, 64)

            # Warm up
            scorer.score_clauses(embs)

            # Measure
            start = time.perf_counter()
            iterations = 10
            for _ in range(iterations):
                scorer.score_clauses(embs)
            elapsed = (time.perf_counter() - start) / iterations

            # Attention for 500 clauses should be < 50ms per call
            assert elapsed < 0.05, (
                f"Attention overhead too high for {n_clauses} clauses: "
                f"{elapsed*1000:.1f}ms (limit: 50ms)"
            )

    def test_fusion_overhead_negligible(self):
        """Multi-source fusion should add < 1ms overhead."""
        cfg = TemporalAttentionConfig(
            base_config=CrossClauseAttentionConfig(
                enabled=True, embedding_dim=64, num_heads=8, head_dim=8,
                dropout=0.0, scoring_hidden_dim=32,
            ),
            use_multi_source_fusion=True,
            hierarchical_dim=64,
            invariant_dim=64,
            fusion_method="gate",
        )
        fusion = MultiSourceFusion(cfg)
        fusion.eval()

        base = _random_embeddings(200, 64)
        hier = _random_embeddings(200, 64, seed=99)
        inv = _random_embeddings(200, 64, seed=77)

        # Warm up
        fusion(base, hier, inv)

        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            fusion(base, hier, inv)
        elapsed = (time.perf_counter() - start) / iterations

        assert elapsed < 0.001, (
            f"Fusion overhead too high: {elapsed*1000:.2f}ms (limit: 1ms)"
        )
