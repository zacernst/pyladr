"""Tests for pyladr.ml.embedding_provider — GNN-Cache bridge.

Tests cover:
- GNNEmbeddingProvider satisfies EmbeddingProvider protocol
- GNNEmbeddingProvider satisfies EmbeddingComputer protocol
- Single and batch embedding retrieval
- Cache hit/miss behavior
- SymbolTable integration
- Model update cache invalidation
- Factory methods (create, create_embedding_provider)
- NoOpEmbeddingProvider fallback behavior
- End-to-end pipeline with ml_selection and inference_guidance
- Error handling and graceful degradation
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

from torch_geometric.data import Batch

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.ml.embeddings.cache import CacheConfig, EmbeddingCache
from pyladr.ml.embedding_provider import (
    EmbeddingProviderConfig,
    GNNEmbeddingProvider,
    NoOpEmbeddingProvider,
    create_embedding_provider,
)
from pyladr.ml.graph.clause_encoder import GNNConfig, HeterogeneousClauseGNN
from pyladr.ml.graph.clause_graph import ClauseGraphConfig


# ── Helpers ────────────────────────────────────────────────────────────────


def _atom(symnum: int, *args):
    return get_rigid_term(symnum, len(args), tuple(args))


def _const(symnum: int):
    return get_rigid_term(symnum, 0)


def _var(n: int):
    return get_variable_term(n)


def _pos_lit(atom) -> Literal:
    return Literal(sign=True, atom=atom)


def _neg_lit(atom) -> Literal:
    return Literal(sign=False, atom=atom)


def _simple_clause(sym=4, const_sym=2) -> Clause:
    """P(a)"""
    return Clause(literals=(_pos_lit(_atom(sym, _const(const_sym))),))


def _complex_clause() -> Clause:
    """P(x) | -Q(f(x, a), b)"""
    x = _var(0)
    a = _const(2)
    b = _const(3)
    f_xa = _atom(1, x, a)
    lit1 = _pos_lit(_atom(4, x))
    lit2 = _neg_lit(_atom(5, f_xa, b))
    return Clause(literals=(lit1, lit2))


def _make_small_gnn() -> HeterogeneousClauseGNN:
    config = GNNConfig(hidden_dim=32, embedding_dim=64, num_layers=1)
    model = HeterogeneousClauseGNN(config)
    model.eval()
    return model


def _make_provider(
    symbol_table: SymbolTable | None = None,
) -> GNNEmbeddingProvider:
    model = _make_small_gnn()
    cache_config = CacheConfig(
        max_entries=1000, embedding_dim=64, device="cpu"
    )
    cache = EmbeddingCache(config=cache_config)
    return GNNEmbeddingProvider(
        model=model,
        cache=cache,
        symbol_table=symbol_table,
        device="cpu",
    )


# ── EmbeddingProvider protocol tests ──────────────────────────────────────


class TestEmbeddingProviderProtocol:
    """Test that GNNEmbeddingProvider satisfies EmbeddingProvider."""

    def test_embedding_dim(self):
        provider = _make_provider()
        assert provider.embedding_dim == 64

    def test_get_embedding_returns_list(self):
        provider = _make_provider()
        clause = _simple_clause()
        result = provider.get_embedding(clause)
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 64

    def test_get_embedding_returns_floats(self):
        provider = _make_provider()
        result = provider.get_embedding(_simple_clause())
        assert all(isinstance(x, float) for x in result)

    def test_get_embeddings_batch_returns_list(self):
        provider = _make_provider()
        clauses = [_simple_clause(), _complex_clause()]
        results = provider.get_embeddings_batch(clauses)
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert r is not None
            assert len(r) == 64

    def test_get_embeddings_batch_empty(self):
        provider = _make_provider()
        results = provider.get_embeddings_batch([])
        assert results == []


# ── EmbeddingComputer protocol tests ──────────────────────────────────────


class TestEmbeddingComputerProtocol:
    """Test that GNNEmbeddingProvider satisfies EmbeddingComputer."""

    def test_compute_embeddings_returns_tensor(self):
        provider = _make_provider()
        clauses = [_simple_clause(), _complex_clause()]
        result = provider.compute_embeddings(clauses)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 64)

    def test_compute_embeddings_single(self):
        provider = _make_provider()
        result = provider.compute_embeddings([_simple_clause()])
        assert result.shape == (1, 64)

    def test_compute_embeddings_empty(self):
        provider = _make_provider()
        result = provider.compute_embeddings([])
        assert result.shape == (0, 64)


# ── Cache integration tests ──────────────────────────────────────────────


class TestCacheIntegration:
    """Test cache hit/miss behavior through the provider."""

    def test_cache_populated_after_get(self):
        provider = _make_provider()
        clause = _simple_clause()
        assert len(provider.cache) == 0

        provider.get_embedding(clause)

        assert len(provider.cache) == 1

    def test_cache_hit_on_second_call(self):
        provider = _make_provider()
        clause = _simple_clause()

        # First call — cache miss
        emb1 = provider.get_embedding(clause)
        stats1 = provider.cache.stats.snapshot()
        assert stats1["misses"] == 1

        # Second call — cache hit
        emb2 = provider.get_embedding(clause)
        stats2 = provider.cache.stats.snapshot()
        assert stats2["hits"] == 1

        # Embeddings should be identical
        assert emb1 == emb2

    def test_batch_populates_cache(self):
        provider = _make_provider()
        clauses = [_simple_clause(), _complex_clause()]
        assert len(provider.cache) == 0

        provider.get_embeddings_batch(clauses)

        assert len(provider.cache) == 2

    def test_alpha_equivalent_clauses_share_cache(self):
        """Clauses with same structure but different var numbers share cache."""
        provider = _make_provider()

        # P(x0) and P(x1) are α-equivalent
        c1 = Clause(literals=(_pos_lit(_atom(4, _var(0))),))
        c2 = Clause(literals=(_pos_lit(_atom(4, _var(1))),))

        provider.get_embedding(c1)
        assert len(provider.cache) == 1

        provider.get_embedding(c2)
        # Should still be 1 — α-equivalent clauses share the same key
        assert len(provider.cache) == 1

    def test_compute_fn_registered(self):
        """Provider registers itself as the cache's compute_fn."""
        provider = _make_provider()
        assert provider.cache.compute_fn is provider


# ── Model lifecycle tests ────────────────────────────────────────────────


class TestModelLifecycle:
    """Test model update and cache invalidation."""

    def test_swap_weights_invalidates_cache(self):
        provider = _make_provider()

        # Populate cache
        provider.get_embedding(_simple_clause())
        assert len(provider.cache) == 1

        # Swap weights (no-op: reload current weights) to trigger invalidation
        provider.swap_weights(provider.model.state_dict())

        assert len(provider.cache) == 0

    def test_symbol_table_setter(self):
        provider = _make_provider()
        assert provider.symbol_table is None

        st = SymbolTable()
        provider.symbol_table = st
        assert provider.symbol_table is st


# ── Factory method tests ─────────────────────────────────────────────────


class TestFactory:
    """Test GNNEmbeddingProvider.create and create_embedding_provider."""

    def test_create_default(self):
        provider = GNNEmbeddingProvider.create()
        assert isinstance(provider, GNNEmbeddingProvider)
        # Default embedding_dim is 512
        assert provider.embedding_dim == 512

    def test_create_with_custom_gnn_config(self):
        gnn_config = GNNConfig(hidden_dim=32, embedding_dim=128, num_layers=1)
        provider = GNNEmbeddingProvider.create(gnn_config=gnn_config)
        assert provider.embedding_dim == 128

    def test_create_with_symbol_table(self):
        st = SymbolTable()
        provider = GNNEmbeddingProvider.create(symbol_table=st)
        assert provider.symbol_table is st

    def test_create_with_config(self):
        config = EmbeddingProviderConfig(
            cache_max_entries=500,
            graph_max_term_depth=5,
        )
        provider = GNNEmbeddingProvider.create(config=config)
        assert provider.cache.config.max_entries == 500

    def test_create_embedding_provider_returns_gnn(self):
        """When ML is available, returns GNNEmbeddingProvider."""
        provider = create_embedding_provider(
            gnn_config=GNNConfig(hidden_dim=32, embedding_dim=64, num_layers=1),
        )
        assert isinstance(provider, GNNEmbeddingProvider)

    def test_create_with_nonexistent_model_path(self):
        """Non-existent model path creates a fresh model."""
        config = EmbeddingProviderConfig(model_path="/nonexistent/model.pt")
        provider = GNNEmbeddingProvider.create(config=config)
        # Should still work — fresh model created
        assert isinstance(provider, GNNEmbeddingProvider)

    def test_create_with_saved_model(self, tmp_path):
        """Load from a saved model checkpoint."""
        from pyladr.ml.graph.clause_encoder import save_model

        gnn_config = GNNConfig(hidden_dim=32, embedding_dim=64, num_layers=1)
        model = HeterogeneousClauseGNN(gnn_config)
        model_path = tmp_path / "model.pt"
        save_model(model, model_path)

        config = EmbeddingProviderConfig(model_path=str(model_path))
        provider = GNNEmbeddingProvider.create(config=config)
        assert provider.embedding_dim == 64


# ── NoOp provider tests ─────────────────────────────────────────────────


class TestNoOpProvider:
    """Test NoOpEmbeddingProvider fallback behavior."""

    def test_embedding_dim(self):
        noop = NoOpEmbeddingProvider(embedding_dim=256)
        assert noop.embedding_dim == 256

    def test_get_embedding_returns_none(self):
        noop = NoOpEmbeddingProvider()
        assert noop.get_embedding(_simple_clause()) is None

    def test_get_embeddings_batch_returns_nones(self):
        noop = NoOpEmbeddingProvider()
        results = noop.get_embeddings_batch([_simple_clause(), _complex_clause()])
        assert results == [None, None]

    def test_get_embeddings_batch_empty(self):
        noop = NoOpEmbeddingProvider()
        assert noop.get_embeddings_batch([]) == []


# ── End-to-end integration tests ─────────────────────────────────────────


class TestEndToEndIntegration:
    """Full pipeline: provider → cache → GNN → embeddings → scoring."""

    def test_provider_with_selection(self):
        """Provider works with EmbeddingEnhancedSelection API."""
        provider = _make_provider()

        # Simulate what ml_selection does
        clauses = [_simple_clause(), _complex_clause()]
        embeddings = provider.get_embeddings_batch(clauses)

        assert len(embeddings) == 2
        assert all(e is not None for e in embeddings)
        assert all(len(e) == 64 for e in embeddings)

    def test_provider_with_inference_guidance(self):
        """Provider works with EmbeddingGuidedInference API."""
        provider = _make_provider()

        given = _simple_clause()
        usable = [_complex_clause(), _simple_clause(sym=5, const_sym=3)]

        # Simulate what inference_guidance does
        given_emb = provider.get_embedding(given)
        usable_embs = provider.get_embeddings_batch(usable)

        assert given_emb is not None
        assert len(given_emb) == 64
        assert all(e is not None for e in usable_embs)

    def test_multiple_problems_with_symbol_table_swap(self):
        """Provider handles symbol table changes between problems."""
        st1 = SymbolTable()
        st1.str_to_sn("f", 2)
        st1.str_to_sn("a", 0)

        provider = GNNEmbeddingProvider.create(
            symbol_table=st1,
            gnn_config=GNNConfig(hidden_dim=32, embedding_dim=64, num_layers=1),
        )

        # First problem
        emb1 = provider.get_embedding(_simple_clause())
        assert emb1 is not None

        # Switch to new problem with different symbols
        st2 = SymbolTable()
        st2.str_to_sn("g", 3)
        st2.str_to_sn("b", 0)
        provider.symbol_table = st2

        emb2 = provider.get_embedding(_simple_clause())
        assert emb2 is not None

    def test_deterministic_embeddings(self):
        """Same clause produces same embedding (eval mode, no dropout)."""
        provider = _make_provider()
        clause = _complex_clause()

        emb1 = provider.get_embedding(clause)
        # Clear cache to force recomputation
        provider.cache.invalidate_all()
        emb2 = provider.get_embedding(clause)

        assert emb1 is not None
        assert emb2 is not None
        for a, b in zip(emb1, emb2):
            assert abs(a - b) < 1e-5


# ── Stats reporting tests ────────────────────────────────────────────────


class TestStats:
    """Test statistics tracking through the provider."""

    def test_stats_after_operations(self):
        provider = _make_provider()

        # First call — miss
        provider.get_embedding(_simple_clause())
        stats = provider.stats
        assert stats["misses"] == 1
        assert stats["hits"] == 0

        # Second call — hit
        provider.get_embedding(_simple_clause())
        stats = provider.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_batch_stats(self):
        provider = _make_provider()
        clauses = [_simple_clause(), _complex_clause()]
        provider.get_embeddings_batch(clauses)

        stats = provider.stats
        assert stats["batch_requests"] == 1
        assert stats["batch_total_clauses"] == 2

    def test_stats_include_model_version(self):
        provider = _make_provider()
        stats = provider.stats
        assert "model_version" in stats
        assert "swap_count" in stats
        assert stats["model_version"] == 0
        assert stats["swap_count"] == 0


# ── Model hot-swapping tests ─────────────────────────────────────────────


class TestModelHotSwapping:
    """Test thread-safe model hot-swapping."""

    def test_swap_weights_invalidates_cache(self):
        provider = _make_provider()

        # Populate cache
        provider.get_embedding(_simple_clause())
        assert len(provider.cache) == 1

        # Swap weights
        new_state = provider.model.state_dict()
        provider.swap_weights(new_state)

        assert len(provider.cache) == 0

    def test_swap_weights_increments_version(self):
        provider = _make_provider()
        assert provider.model_version == 0
        assert provider.swap_count == 0

        new_state = provider.model.state_dict()
        version = provider.swap_weights(new_state)

        assert version == 1
        assert provider.model_version == 1
        assert provider.swap_count == 1

    def test_swap_weights_multiple(self):
        provider = _make_provider()

        for i in range(5):
            new_state = provider.model.state_dict()
            version = provider.swap_weights(new_state)
            assert version == i + 1

        assert provider.model_version == 5
        assert provider.swap_count == 5

    def test_swap_weights_produces_new_embeddings(self):
        provider = _make_provider()
        clause = _simple_clause()

        # Get embedding with original model
        emb1 = provider.get_embedding(clause)
        assert emb1 is not None

        # Perturb model weights
        new_state = provider.model.state_dict()
        for k in new_state:
            new_state[k] = new_state[k] + 0.5 * torch.randn_like(new_state[k])
        provider.swap_weights(new_state)

        # Get embedding with new model — should differ
        emb2 = provider.get_embedding(clause)
        assert emb2 is not None
        # With perturbed weights, embeddings should change
        assert emb1 != emb2

    def test_swap_model_replaces_instance(self):
        provider = _make_provider()
        original_model = provider.model

        new_model = _make_small_gnn()
        version = provider.swap_model(new_model)

        assert provider.model is new_model
        assert provider.model is not original_model
        assert version == 1

    def test_swap_model_invalidates_cache(self):
        provider = _make_provider()

        # Populate cache
        provider.get_embedding(_simple_clause())
        assert len(provider.cache) == 1

        new_model = _make_small_gnn()
        provider.swap_model(new_model)

        assert len(provider.cache) == 0

    def test_embeddings_work_after_swap(self):
        """Verify embedding generation works correctly after hot-swap."""
        provider = _make_provider()

        # Swap to a new model
        new_state = provider.model.state_dict()
        provider.swap_weights(new_state)

        # Should still be able to get embeddings
        emb = provider.get_embedding(_simple_clause())
        assert emb is not None
        assert len(emb) == 64

        batch = provider.get_embeddings_batch([_simple_clause(), _complex_clause()])
        assert len(batch) == 2
        assert all(e is not None for e in batch)

    def test_checkpoint_restore_round_trip(self):
        provider = _make_provider()
        clause = _simple_clause()

        # Get embedding with original model
        emb_original = provider.get_embedding(clause)

        # Take checkpoint
        ckpt = provider.checkpoint()

        # Perturb weights
        new_state = provider.model.state_dict()
        for k in new_state:
            new_state[k] = torch.randn_like(new_state[k])
        provider.swap_weights(new_state)

        emb_perturbed = provider.get_embedding(clause)
        assert emb_perturbed != emb_original

        # Restore checkpoint
        provider.restore_checkpoint(ckpt)

        emb_restored = provider.get_embedding(clause)
        assert emb_restored is not None
        for a, b in zip(emb_original, emb_restored):
            assert abs(a - b) < 1e-5

    def test_eval_mode_after_swap(self):
        """Model should be in eval mode after swap_weights."""
        provider = _make_provider()

        new_state = provider.model.state_dict()
        provider.swap_weights(new_state)

        assert not provider.model.training

    def test_eval_mode_after_swap_model(self):
        """Model should be in eval mode after swap_model."""
        provider = _make_provider()

        new_model = _make_small_gnn()
        new_model.train()  # explicitly set to training mode
        provider.swap_model(new_model)

        assert not provider.model.training


class TestModelHotSwapConcurrency:
    """Test thread-safety of model hot-swapping."""

    def test_concurrent_reads_during_swap(self):
        """Embedding reads should not crash during concurrent model swaps."""
        import concurrent.futures

        provider = _make_provider()
        clauses = [_simple_clause(), _complex_clause()]
        errors: list[Exception] = []

        def read_embeddings():
            try:
                for _ in range(20):
                    result = provider.get_embeddings_batch(clauses)
                    assert len(result) == 2
            except Exception as e:
                errors.append(e)

        def swap_weights():
            try:
                for _ in range(10):
                    state = provider.model.state_dict()
                    provider.swap_weights(state)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            # 3 reader threads
            for _ in range(3):
                futures.append(pool.submit(read_embeddings))
            # 1 writer thread
            futures.append(pool.submit(swap_weights))

            for f in concurrent.futures.as_completed(futures):
                f.result()  # Raise any exceptions

        assert errors == [], f"Concurrent errors: {errors}"

    def test_version_monotonically_increases(self):
        """Model version should only increase, even under concurrent swaps."""
        import concurrent.futures

        provider = _make_provider()
        versions: list[int] = []
        lock = __import__("threading").Lock()

        def swap_and_record():
            state = provider.model.state_dict()
            v = provider.swap_weights(state)
            with lock:
                versions.append(v)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(swap_and_record) for _ in range(20)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        # All versions should be unique and positive
        assert len(set(versions)) == len(versions)
        assert all(v > 0 for v in versions)
        assert provider.model_version == 20
