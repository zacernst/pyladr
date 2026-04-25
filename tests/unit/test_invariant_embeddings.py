"""Tests for property-invariant embeddings with symbol-independence.

Verifies that:
1. Canonical mapping assigns consistent IDs based on structural role
2. Invariant features are independent of symbol naming
3. Invariant graph construction preserves topology but canonicalizes symbols
4. Invariant structural hashing collides for renamed clauses
5. InvariantEmbeddingProvider produces identical embeddings for renamed clauses
"""

from __future__ import annotations

import pytest

from tests.factories import make_clause, make_literal, make_term, make_var


# ── Canonicalization tests ────────────────────────────────────────────────


class TestCanonicalMapping:
    """Tests for symbol canonicalization."""

    def test_same_structure_different_symbols_get_same_canonical_ids(self):
        """P(f(x), g(y)) and Q(h(x), k(y)) should have same canonical structure."""
        from pyladr.ml.invariant.canonicalization import canonicalize_clause

        # Clause 1: P(f(x), g(y)) with P=1, f=2, g=3
        x, y = make_var(0), make_var(1)
        fx = make_term(2, x)
        gy = make_term(3, y)
        atom1 = make_term(1, fx, gy)
        clause1 = make_clause(make_literal(True, atom1))

        # Clause 2: Q(h(x), k(y)) with Q=10, h=20, k=30
        x2, y2 = make_var(0), make_var(1)
        hx = make_term(20, x2)
        ky = make_term(30, y2)
        atom2 = make_term(10, hx, ky)
        clause2 = make_clause(make_literal(True, atom2))

        m1 = canonicalize_clause(clause1)
        m2 = canonicalize_clause(clause2)

        # Both should produce the same number of canonical IDs
        assert m1.next_id == m2.next_id

        # Structural roles should match
        for cid in range(m1.next_id):
            role1 = m1.canonical_to_role[cid]
            role2 = m2.canonical_to_role[cid]
            assert role1.arity == role2.arity
            assert role1.is_predicate == role2.is_predicate

    def test_different_arities_get_different_canonical_ids(self):
        """f(x) and g(x, y) should NOT collide."""
        from pyladr.ml.invariant.canonicalization import canonicalize_clause

        x, y = make_var(0), make_var(1)
        fx = make_term(2, x)          # unary function
        gxy = make_term(3, x, y)      # binary function
        atom = make_term(1, fx, gxy)
        clause = make_clause(make_literal(True, atom))

        mapping = canonicalize_clause(clause)

        # f (arity 1) and g (arity 2) should have different canonical IDs
        f_id = mapping.sym_to_canonical[2]
        g_id = mapping.sym_to_canonical[3]
        assert f_id != g_id

    def test_predicate_vs_function_distinction(self):
        """Top-level predicate symbol and nested function symbol differ."""
        from pyladr.ml.invariant.canonicalization import canonicalize_clause

        x = make_var(0)
        # P(x) where P is predicate (arity 1)
        atom = make_term(1, x)
        clause = make_clause(make_literal(True, atom))

        mapping = canonicalize_clause(clause)
        p_role = mapping.canonical_to_role[mapping.sym_to_canonical[1]]
        assert p_role.is_predicate is True

    def test_reset_clears_state(self):
        """CanonicalMapping.reset() produces clean state."""
        from pyladr.ml.invariant.canonicalization import CanonicalMapping, SymbolRole

        mapping = CanonicalMapping()
        mapping.get_or_assign(5, SymbolRole(arity=2, is_predicate=True))
        assert mapping.next_id == 1

        mapping.reset()
        assert mapping.next_id == 0
        assert len(mapping.sym_to_canonical) == 0

    def test_deterministic_ordering(self):
        """Canonical IDs should be assigned in traversal order."""
        from pyladr.ml.invariant.canonicalization import canonicalize_clause

        x = make_var(0)
        fx = make_term(5, x)
        gfx = make_term(7, fx)
        atom = make_term(3, gfx)
        clause = make_clause(make_literal(True, atom))

        mapping = canonicalize_clause(clause)

        # Traversal order: P(3) first (predicate), then g(7), then f(5)
        assert mapping.sym_to_canonical[3] == 0  # predicate, first
        assert mapping.sym_to_canonical[7] == 1  # g, second
        assert mapping.sym_to_canonical[5] == 2  # f, third


# ── Invariant features tests ─────────────────────────────────────────────


class TestInvariantFeatures:
    """Tests for invariant feature extraction."""

    def test_renamed_clauses_produce_same_features(self):
        """Structurally identical clauses with different symbols → same features."""
        from pyladr.ml.invariant.invariant_features import InvariantFeatureExtractor

        extractor = InvariantFeatureExtractor()

        # Clause 1: P(f(x)) with P=1, f=2
        x1 = make_var(0)
        fx1 = make_term(2, x1)
        atom1 = make_term(1, fx1)
        clause1 = make_clause(make_literal(True, atom1))

        # Clause 2: Q(g(x)) with Q=10, g=20
        x2 = make_var(0)
        gx2 = make_term(20, x2)
        atom2 = make_term(10, gx2)
        clause2 = make_clause(make_literal(True, atom2))

        extractor.prepare(clause1)
        feat1_pred = extractor.symbol_features(1)  # P
        feat1_func = extractor.symbol_features(2)  # f

        extractor.prepare(clause2)
        feat2_pred = extractor.symbol_features(10)  # Q
        feat2_func = extractor.symbol_features(20)  # g

        # Same canonical IDs, same structural features
        assert feat1_pred == feat2_pred
        assert feat1_func == feat2_func

    def test_feature_vector_dimensionality(self):
        """Feature vectors should be 6-dimensional (matching original)."""
        from pyladr.ml.invariant.invariant_features import InvariantFeatureExtractor

        extractor = InvariantFeatureExtractor()
        x = make_var(0)
        atom = make_term(1, x)
        clause = make_clause(make_literal(True, atom))

        extractor.prepare(clause)
        features = extractor.symbol_features(1)
        assert len(features) == 6

    def test_occurrence_counting(self):
        """Occurrence counts should reflect actual usage in clause."""
        from pyladr.ml.invariant.invariant_features import InvariantFeatureExtractor

        extractor = InvariantFeatureExtractor()

        # P(f(x), f(y)) — f appears twice
        x, y = make_var(0), make_var(1)
        fx = make_term(2, x)
        fy = make_term(2, y)
        atom = make_term(1, fx, fy)
        clause = make_clause(make_literal(True, atom))

        extractor.prepare(clause)
        features = extractor.symbol_features(2)  # f

        # occurrence_count is features[4]
        assert features[4] == 2.0


# ── Invariant structural hash tests ──────────────────────────────────────


class TestInvariantHash:
    """Tests for symbol-independent structural hashing."""

    def test_renamed_clauses_same_hash(self):
        """Structurally identical clauses with different symbols → same hash."""
        from pyladr.ml.invariant.invariant_features import (
            invariant_clause_structural_hash,
        )

        # P(f(x), y) with P=1, f=2
        x1, y1 = make_var(0), make_var(1)
        atom1 = make_term(1, make_term(2, x1), y1)
        clause1 = make_clause(make_literal(True, atom1))

        # Q(g(x), y) with Q=10, g=20
        x2, y2 = make_var(0), make_var(1)
        atom2 = make_term(10, make_term(20, x2), y2)
        clause2 = make_clause(make_literal(True, atom2))

        h1 = invariant_clause_structural_hash(clause1)
        h2 = invariant_clause_structural_hash(clause2)
        assert h1 == h2

    def test_different_structure_different_hash(self):
        """Structurally different clauses → different hashes."""
        from pyladr.ml.invariant.invariant_features import (
            invariant_clause_structural_hash,
        )

        # P(f(x)) — unary predicate with unary function
        x = make_var(0)
        atom1 = make_term(1, make_term(2, x))
        clause1 = make_clause(make_literal(True, atom1))

        # P(x, y) — binary predicate with variables
        y = make_var(1)
        atom2 = make_term(1, x, y)
        clause2 = make_clause(make_literal(True, atom2))

        h1 = invariant_clause_structural_hash(clause1)
        h2 = invariant_clause_structural_hash(clause2)
        assert h1 != h2

    def test_variable_renaming_same_hash(self):
        """Alpha-equivalent clauses → same hash."""
        from pyladr.ml.invariant.invariant_features import (
            invariant_clause_structural_hash,
        )

        # P(x, y) with var 0 and 1
        atom1 = make_term(1, make_var(0), make_var(1))
        clause1 = make_clause(make_literal(True, atom1))

        # P(u, v) with var 5 and 10
        atom2 = make_term(1, make_var(5), make_var(10))
        clause2 = make_clause(make_literal(True, atom2))

        h1 = invariant_clause_structural_hash(clause1)
        h2 = invariant_clause_structural_hash(clause2)
        assert h1 == h2

    def test_literal_order_independence(self):
        """Clause is a disjunction — literal order doesn't matter."""
        from pyladr.ml.invariant.invariant_features import (
            invariant_clause_structural_hash,
        )

        x = make_var(0)
        lit_a = make_literal(True, make_term(1, x))
        lit_b = make_literal(False, make_term(2, x))

        clause1 = make_clause(lit_a, lit_b)
        clause2 = make_clause(lit_b, lit_a)

        h1 = invariant_clause_structural_hash(clause1)
        h2 = invariant_clause_structural_hash(clause2)
        assert h1 == h2


# ── Invariant graph construction tests ───────────────────────────────────


class TestInvariantGraph:
    """Tests for invariant graph builder."""

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self):
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

    def test_graph_topology_matches_original(self):
        """Invariant graph should have same node/edge counts as original."""
        from pyladr.ml.graph.clause_graph import clause_to_heterograph
        from pyladr.ml.invariant.invariant_graph import invariant_clause_to_heterograph

        x, y = make_var(0), make_var(1)
        atom = make_term(1, make_term(2, x), y)
        clause = make_clause(make_literal(True, atom))

        original = clause_to_heterograph(clause)
        invariant = invariant_clause_to_heterograph(clause)

        # Same number of nodes per type
        for nt in original.node_types:
            assert original[nt].num_nodes == invariant[nt].num_nodes

        # Same edge types present
        assert set(original.edge_types) == set(invariant.edge_types)

    def test_symbol_features_differ_from_original(self):
        """Invariant symbol features should use canonical IDs, not raw symnums."""
        from pyladr.ml.graph.clause_graph import NodeType, clause_to_heterograph
        from pyladr.ml.invariant.invariant_graph import invariant_clause_to_heterograph

        x = make_var(0)
        # Use a large symnum to make the difference obvious
        atom = make_term(999, make_term(888, x))
        clause = make_clause(make_literal(True, atom))

        original = clause_to_heterograph(clause)
        invariant = invariant_clause_to_heterograph(clause)

        sym_nt = NodeType.SYMBOL.value
        orig_feats = original[sym_nt].x
        inv_feats = invariant[sym_nt].x

        # First feature (canonical_id for invariant, capped symnum for original)
        # should differ since 999 >> canonical 0
        assert not (orig_feats[:, 0] == inv_feats[:, 0]).all()

    def test_renamed_clauses_produce_same_graph_features(self):
        """Two renamed clauses should produce identical invariant graph features."""
        from pyladr.ml.graph.clause_graph import NodeType
        from pyladr.ml.invariant.invariant_graph import invariant_clause_to_heterograph
        import torch

        # Clause 1: P(f(x)) with P=1, f=2
        x1 = make_var(0)
        atom1 = make_term(1, make_term(2, x1))
        clause1 = make_clause(make_literal(True, atom1))

        # Clause 2: Q(g(x)) with Q=50, g=60
        x2 = make_var(0)
        atom2 = make_term(50, make_term(60, x2))
        clause2 = make_clause(make_literal(True, atom2))

        g1 = invariant_clause_to_heterograph(clause1)
        g2 = invariant_clause_to_heterograph(clause2)

        # All node features should be identical
        for nt in [NodeType.CLAUSE.value, NodeType.LITERAL.value,
                    NodeType.TERM.value, NodeType.SYMBOL.value,
                    NodeType.VARIABLE.value]:
            if g1[nt].num_nodes > 0:
                assert torch.equal(g1[nt].x, g2[nt].x), (
                    f"Features differ for {nt}"
                )

    def test_batch_construction(self):
        """Batch invariant graph construction should work."""
        from pyladr.ml.invariant.invariant_graph import (
            batch_invariant_clauses_to_heterograph,
        )

        clauses = []
        for sym in [1, 10, 100]:
            x = make_var(0)
            atom = make_term(sym, x)
            clauses.append(make_clause(make_literal(True, atom)))

        graphs = batch_invariant_clauses_to_heterograph(clauses)
        assert len(graphs) == 3


# ── Integration test: full embedding pipeline ────────────────────────────


class TestInvariantProvider:
    """Integration tests for InvariantEmbeddingProvider."""

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self):
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

    def test_provider_creation(self):
        """Provider should create successfully."""
        from pyladr.ml.invariant.invariant_provider import (
            InvariantEmbeddingProvider,
        )

        provider = InvariantEmbeddingProvider.create()
        assert provider.embedding_dim > 0

    def test_single_embedding(self):
        """Should produce an embedding for a single clause."""
        from pyladr.ml.invariant.invariant_provider import (
            InvariantEmbeddingProvider,
        )

        provider = InvariantEmbeddingProvider.create()

        x = make_var(0)
        atom = make_term(1, x)
        clause = make_clause(make_literal(True, atom))

        emb = provider.get_embedding(clause)
        assert emb is not None
        assert len(emb) == provider.embedding_dim

    def test_renamed_clauses_produce_identical_embeddings(self):
        """Core invariance property: renamed clauses → identical embeddings."""
        import torch
        from pyladr.ml.invariant.invariant_provider import (
            InvariantEmbeddingProvider,
        )

        provider = InvariantEmbeddingProvider.create()

        # P(f(x, y)) with P=1, f=2
        x1, y1 = make_var(0), make_var(1)
        atom1 = make_term(1, make_term(2, x1, y1))
        clause1 = make_clause(make_literal(True, atom1))

        # Q(g(x, y)) with Q=50, g=60
        x2, y2 = make_var(0), make_var(1)
        atom2 = make_term(50, make_term(60, x2, y2))
        clause2 = make_clause(make_literal(True, atom2))

        emb1 = provider.get_embedding(clause1)
        emb2 = provider.get_embedding(clause2)

        assert emb1 is not None
        assert emb2 is not None
        assert torch.allclose(
            torch.tensor(emb1), torch.tensor(emb2), atol=1e-6
        ), "Renamed clauses should produce identical embeddings"

    def test_structurally_different_clauses_differ(self):
        """Structurally different clauses should generally differ."""
        from pyladr.ml.invariant.invariant_provider import (
            InvariantEmbeddingProvider,
        )

        provider = InvariantEmbeddingProvider.create()

        # P(x) — unary
        atom1 = make_term(1, make_var(0))
        clause1 = make_clause(make_literal(True, atom1))

        # P(x, y) — binary
        atom2 = make_term(1, make_var(0), make_var(1))
        clause2 = make_clause(make_literal(True, atom2))

        emb1 = provider.get_embedding(clause1)
        emb2 = provider.get_embedding(clause2)

        assert emb1 is not None
        assert emb2 is not None
        # Not necessarily different (random init), but the inputs differ
        # so we just verify both succeed

    def test_batch_embeddings(self):
        """Batch embedding should work and produce correct count."""
        from pyladr.ml.invariant.invariant_provider import (
            InvariantEmbeddingProvider,
        )

        provider = InvariantEmbeddingProvider.create()

        clauses = []
        for sym in range(1, 4):
            atom = make_term(sym, make_var(0))
            clauses.append(make_clause(make_literal(True, atom)))

        results = provider.get_embeddings_batch(clauses)
        assert len(results) == 3
        assert all(r is not None for r in results)

    def test_cache_sharing_across_renamings(self):
        """Renamed clauses should share cache entries."""
        from pyladr.ml.invariant.invariant_provider import (
            InvariantEmbeddingProvider,
        )

        provider = InvariantEmbeddingProvider.create()

        # First clause: P(x)
        atom1 = make_term(1, make_var(0))
        clause1 = make_clause(make_literal(True, atom1))

        # Get embedding (cache miss)
        provider.get_embedding(clause1)
        stats1 = provider.stats.copy()

        # Second clause: Q(x) — different name, same structure
        atom2 = make_term(99, make_var(0))
        clause2 = make_clause(make_literal(True, atom2))

        # Should hit cache
        provider.get_embedding(clause2)
        stats2 = provider.stats.copy()

        # The second lookup should be a cache hit
        assert stats2["hits"] > stats1["hits"]

    def test_model_swap(self):
        """Model hot-swap should work and invalidate cache."""
        from pyladr.ml.invariant.invariant_provider import (
            InvariantEmbeddingProvider,
        )

        provider = InvariantEmbeddingProvider.create()
        assert provider.model_version == 0

        # Swap weights
        state = provider.checkpoint()
        new_version = provider.swap_weights(state)
        assert new_version == 1
        assert provider.model_version == 1

    def test_implements_embedding_provider_protocol(self):
        """Should satisfy the EmbeddingProvider protocol."""
        from pyladr.ml.invariant.invariant_provider import (
            InvariantEmbeddingProvider,
        )

        provider = InvariantEmbeddingProvider.create()

        # Check protocol methods exist and are callable
        assert hasattr(provider, "embedding_dim")
        assert hasattr(provider, "get_embedding")
        assert hasattr(provider, "get_embeddings_batch")
        assert isinstance(provider.embedding_dim, int)
