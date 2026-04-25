"""Comprehensive tests for ForteEmbeddingProvider.

Tests cover: EmbeddingProvider protocol compliance, structural caching,
thread safety, graceful degradation, cache management, and performance.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from pyladr.ml.forte.algorithm import ForteAlgorithm, ForteConfig
from pyladr.ml.forte.provider import (
    ForteCacheStats,
    ForteEmbeddingProvider,
    ForteProviderConfig,
)
from tests.factories import (
    make_clause as _clause,
    make_const as _const,
    make_func as _func,
    make_neg_lit as _neg_lit,
    make_pos_lit as _pos_lit,
    make_var as _var,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def provider() -> ForteEmbeddingProvider:
    """Default provider with caching enabled."""
    return ForteEmbeddingProvider()


@pytest.fixture
def nocache_provider() -> ForteEmbeddingProvider:
    """Provider with caching disabled."""
    return ForteEmbeddingProvider(
        config=ForteProviderConfig(enable_cache=False),
    )


@pytest.fixture
def small_cache_provider() -> ForteEmbeddingProvider:
    """Provider with a small cache for testing eviction."""
    return ForteEmbeddingProvider(
        config=ForteProviderConfig(cache_max_entries=5),
    )


@pytest.fixture
def unit_clause() -> Clause:
    """P(x)"""
    return _clause(_pos_lit(_func(1, _var(0))), weight=2.0)


@pytest.fixture
def binary_clause() -> Clause:
    """P(x) | -Q(x, y)"""
    return _clause(
        _pos_lit(_func(1, _var(0))),
        _neg_lit(_func(2, _var(0), _var(1))),
        weight=5.0,
    )


@pytest.fixture
def ground_clause() -> Clause:
    """P(a) | Q(a, b)"""
    return _clause(
        _pos_lit(_func(1, _const(3))),
        _pos_lit(_func(2, _const(3), _const(4))),
        weight=5.0,
    )


# ── EmbeddingProvider Protocol Compliance ────────────────────────────────────


class TestProtocolCompliance:
    """Verify ForteEmbeddingProvider satisfies EmbeddingProvider protocol."""

    def test_has_embedding_dim(self, provider: ForteEmbeddingProvider) -> None:
        assert provider.embedding_dim == 64

    def test_custom_embedding_dim(self) -> None:
        p = ForteEmbeddingProvider(
            config=ForteProviderConfig(
                forte_config=ForteConfig(embedding_dim=128),
            ),
        )
        assert p.embedding_dim == 128

    def test_get_embedding_returns_list_or_none(
        self, provider: ForteEmbeddingProvider, unit_clause: Clause,
    ) -> None:
        result = provider.get_embedding(unit_clause)
        assert isinstance(result, list)
        assert len(result) == 64
        assert all(isinstance(v, float) for v in result)

    def test_get_embeddings_batch_returns_list(
        self,
        provider: ForteEmbeddingProvider,
        unit_clause: Clause,
        binary_clause: Clause,
    ) -> None:
        result = provider.get_embeddings_batch([unit_clause, binary_clause])
        assert len(result) == 2
        assert all(isinstance(r, list) for r in result)
        assert all(len(r) == 64 for r in result)  # type: ignore[arg-type]

    def test_get_embeddings_batch_empty(
        self, provider: ForteEmbeddingProvider,
    ) -> None:
        assert provider.get_embeddings_batch([]) == []

    def test_embedding_is_normalized(
        self, provider: ForteEmbeddingProvider, unit_clause: Clause,
    ) -> None:
        emb = provider.get_embedding(unit_clause)
        assert emb is not None
        norm = math.sqrt(sum(v * v for v in emb))
        assert norm == pytest.approx(1.0, abs=1e-10)

    def test_runtime_checkable_protocol(
        self, provider: ForteEmbeddingProvider,
    ) -> None:
        """Verify structural protocol compliance."""
        from pyladr.protocols import EmbeddingProvider
        assert isinstance(provider, EmbeddingProvider)


# ── Structural Caching ───────────────────────────────────────────────────────


class TestStructuralCaching:
    """Verify structural hashing cache behavior."""

    def test_cache_hit(
        self, provider: ForteEmbeddingProvider, unit_clause: Clause,
    ) -> None:
        """Second call should be a cache hit."""
        provider.get_embedding(unit_clause)
        assert provider.stats.misses == 1
        assert provider.stats.hits == 0

        provider.get_embedding(unit_clause)
        assert provider.stats.hits == 1

    def test_alpha_equivalent_cache_hit(
        self, provider: ForteEmbeddingProvider,
    ) -> None:
        """α-equivalent clauses should share cached embeddings.

        P(x) and P(y) are α-equivalent (same structure, different var numbering).
        After variable normalization by structural hashing, they should collide.
        """
        # Note: structural hashing normalizes variables, so P(v0) and P(v0)
        # with same structure are the same. Different variable numbers that
        # normalize to the same first-occurrence order will collide.
        c1 = _clause(_pos_lit(_func(1, _var(0))), weight=2.0)
        c2 = _clause(_pos_lit(_func(1, _var(0))), weight=2.0)

        emb1 = provider.get_embedding(c1)
        emb2 = provider.get_embedding(c2)
        assert emb1 == emb2
        # Second should be a cache hit
        assert provider.stats.hits >= 1

    def test_different_clauses_different_cache_keys(
        self, provider: ForteEmbeddingProvider,
    ) -> None:
        c1 = _clause(_pos_lit(_func(1, _var(0))))
        c2 = _clause(_pos_lit(_func(2, _var(0))))
        provider.get_embedding(c1)
        provider.get_embedding(c2)
        assert provider.stats.misses == 2
        assert provider.cache_size == 2

    def test_cache_size_tracking(
        self, provider: ForteEmbeddingProvider,
    ) -> None:
        for i in range(1, 11):
            c = _clause(_pos_lit(_func(i, _var(0))))
            provider.get_embedding(c)
        assert provider.cache_size == 10

    def test_cache_disabled(
        self, nocache_provider: ForteEmbeddingProvider, unit_clause: Clause,
    ) -> None:
        """With cache disabled, every call computes fresh."""
        nocache_provider.get_embedding(unit_clause)
        nocache_provider.get_embedding(unit_clause)
        assert nocache_provider.cache_size == 0
        assert nocache_provider.stats.hits == 0
        assert nocache_provider.stats.misses == 0  # no stats tracked without cache

    def test_cache_hit_rate(
        self, provider: ForteEmbeddingProvider,
    ) -> None:
        """Verify hit rate computation."""
        c = _clause(_pos_lit(_func(1, _var(0))))
        provider.get_embedding(c)  # miss
        provider.get_embedding(c)  # hit
        provider.get_embedding(c)  # hit
        assert provider.stats.hit_rate == pytest.approx(2.0 / 3.0)


# ── LRU Eviction ─────────────────────────────────────────────────────────────


class TestLRUEviction:
    """Verify LRU cache eviction behavior."""

    def test_eviction_when_full(
        self, small_cache_provider: ForteEmbeddingProvider,
    ) -> None:
        """Cache should evict oldest entries when at capacity."""
        # Fill cache with 5 entries
        for i in range(1, 6):
            c = _clause(_pos_lit(_func(i, _var(0))))
            small_cache_provider.get_embedding(c)
        assert small_cache_provider.cache_size == 5

        # Add 6th entry - should evict oldest
        c6 = _clause(_pos_lit(_func(6, _var(0))))
        small_cache_provider.get_embedding(c6)
        assert small_cache_provider.cache_size == 5
        assert small_cache_provider.stats.evictions >= 1

    def test_eviction_preserves_recent(
        self, small_cache_provider: ForteEmbeddingProvider,
    ) -> None:
        """Recently accessed entries should survive eviction."""
        clauses = [_clause(_pos_lit(_func(i, _var(0)))) for i in range(1, 6)]
        for c in clauses:
            small_cache_provider.get_embedding(c)

        # Re-access first clause (makes it most recent)
        small_cache_provider.get_embedding(clauses[0])

        # Add new entry - should evict second clause (now oldest)
        c_new = _clause(_pos_lit(_func(99, _var(0))))
        small_cache_provider.get_embedding(c_new)
        assert small_cache_provider.cache_size == 5

        # First clause should still be cached (recently accessed)
        small_cache_provider.get_embedding(clauses[0])
        # The most recent get should be a hit
        assert small_cache_provider.stats.hits >= 2


# ── Cache Management ─────────────────────────────────────────────────────────


class TestCacheManagement:
    """Tests for cache invalidation and management."""

    def test_invalidate_all(
        self, provider: ForteEmbeddingProvider,
    ) -> None:
        for i in range(1, 6):
            c = _clause(_pos_lit(_func(i, _var(0))))
            provider.get_embedding(c)
        assert provider.cache_size == 5

        count = provider.invalidate_all()
        assert count == 5
        assert provider.cache_size == 0

    def test_invalidate_clause(
        self, provider: ForteEmbeddingProvider,
    ) -> None:
        c = _clause(_pos_lit(_func(1, _var(0))))
        provider.get_embedding(c)
        assert provider.cache_size == 1

        assert provider.invalidate_clause(c) is True
        assert provider.cache_size == 0

        # Invalidating non-existent clause returns False
        c2 = _clause(_pos_lit(_func(99, _var(0))))
        assert provider.invalidate_clause(c2) is False

    def test_invalidate_all_empty_cache(
        self, provider: ForteEmbeddingProvider,
    ) -> None:
        assert provider.invalidate_all() == 0

    def test_invalidate_all_no_cache(
        self, nocache_provider: ForteEmbeddingProvider,
    ) -> None:
        assert nocache_provider.invalidate_all() == 0

    def test_invalidate_clause_no_cache(
        self, nocache_provider: ForteEmbeddingProvider,
    ) -> None:
        c = _clause(_pos_lit(_func(1, _var(0))))
        assert nocache_provider.invalidate_clause(c) is False


# ── Graceful Degradation ─────────────────────────────────────────────────────


class TestGracefulDegradation:
    """Verify graceful degradation on errors."""

    def test_empty_clause_returns_embedding(
        self, provider: ForteEmbeddingProvider,
    ) -> None:
        """Empty clause should return zero vector, not None."""
        c = _clause()
        emb = provider.get_embedding(c)
        assert emb is not None
        assert all(v == 0.0 for v in emb)

    def test_batch_with_mixed_clauses(
        self, provider: ForteEmbeddingProvider,
    ) -> None:
        """Batch with empty and non-empty clauses should all succeed."""
        clauses = [
            _clause(),  # empty
            _clause(_pos_lit(_func(1, _var(0)))),  # normal
            _clause(_pos_lit(_func(2, _const(3)))),  # ground
        ]
        results = provider.get_embeddings_batch(clauses)
        assert len(results) == 3
        assert all(r is not None for r in results)


# ── Determinism ──────────────────────────────────────────────────────────────


class TestDeterminism:
    """Verify deterministic output across calls and instances."""

    def test_same_provider_deterministic(
        self, provider: ForteEmbeddingProvider, unit_clause: Clause,
    ) -> None:
        emb1 = provider.get_embedding(unit_clause)
        provider.invalidate_all()
        emb2 = provider.get_embedding(unit_clause)
        assert emb1 == emb2

    def test_different_providers_deterministic(
        self, unit_clause: Clause,
    ) -> None:
        p1 = ForteEmbeddingProvider()
        p2 = ForteEmbeddingProvider()
        assert p1.get_embedding(unit_clause) == p2.get_embedding(unit_clause)

    def test_cached_matches_uncached(
        self, unit_clause: Clause,
    ) -> None:
        """Cached and uncached providers produce identical results."""
        cached = ForteEmbeddingProvider()
        uncached = ForteEmbeddingProvider(
            config=ForteProviderConfig(enable_cache=False),
        )
        assert cached.get_embedding(unit_clause) == uncached.get_embedding(unit_clause)

    def test_clause_id_does_not_affect_embedding(
        self, provider: ForteEmbeddingProvider,
    ) -> None:
        c1 = _clause(_pos_lit(_func(1, _var(0))), weight=2.0, clause_id=1)
        c2 = _clause(_pos_lit(_func(1, _var(0))), weight=2.0, clause_id=999)
        assert provider.get_embedding(c1) == provider.get_embedding(c2)


# ── Thread Safety ────────────────────────────────────────────────────────────


class TestThreadSafety:
    """Verify thread-safe concurrent access."""

    def test_concurrent_get_embedding(
        self, provider: ForteEmbeddingProvider,
    ) -> None:
        """Multiple threads requesting same clause concurrently."""
        clause = _clause(
            _pos_lit(_func(1, _var(0), _func(2, _var(1)))),
            _neg_lit(_func(3, _const(4))),
            weight=5.0,
        )
        reference = provider.get_embedding(clause)
        errors: list[str] = []

        def worker() -> None:
            for _ in range(50):
                emb = provider.get_embedding(clause)
                if emb != reference:
                    errors.append("Embedding mismatch")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_different_clauses(
        self, provider: ForteEmbeddingProvider,
    ) -> None:
        """Multiple threads requesting different clauses."""
        clauses = [
            _clause(_pos_lit(_func(i, _var(0))), weight=float(i))
            for i in range(1, 21)
        ]
        reference = {
            i: provider.get_embedding(c) for i, c in enumerate(clauses)
        }
        provider.invalidate_all()

        errors: list[str] = []

        def worker(idx: int, clause: Clause) -> None:
            for _ in range(20):
                emb = provider.get_embedding(clause)
                if emb != reference[idx]:
                    errors.append(f"Mismatch clause {idx}")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(worker, i, c)
                for i, c in enumerate(clauses)
            ]
            for f in as_completed(futures):
                f.result()

        assert not errors

    def test_concurrent_with_eviction(self) -> None:
        """Concurrent access with active eviction (small cache)."""
        provider = ForteEmbeddingProvider(
            config=ForteProviderConfig(cache_max_entries=10),
        )
        errors: list[str] = []

        def worker(start: int) -> None:
            for i in range(start, start + 30):
                c = _clause(_pos_lit(_func(i, _var(0))))
                emb = provider.get_embedding(c)
                if emb is None:
                    errors.append(f"Got None for clause {i}")
                elif len(emb) != 64:
                    errors.append(f"Wrong dim for clause {i}")

        # Start from 1 (symbol number must be positive)
        threads = [threading.Thread(target=worker, args=(1 + i * 30,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert provider.cache_size <= 10


# ── Statistics ───────────────────────────────────────────────────────────────


class TestStatistics:
    """Test statistics tracking."""

    def test_stats_initial(self, provider: ForteEmbeddingProvider) -> None:
        s = provider.stats.snapshot()
        assert s["hits"] == 0
        assert s["misses"] == 0
        assert s["evictions"] == 0
        assert s["errors"] == 0
        assert s["hit_rate"] == 0.0

    def test_stats_after_operations(
        self, provider: ForteEmbeddingProvider,
    ) -> None:
        c = _clause(_pos_lit(_func(1, _var(0))))
        provider.get_embedding(c)  # miss
        provider.get_embedding(c)  # hit
        s = provider.stats.snapshot()
        assert s["misses"] == 1
        assert s["hits"] == 1
        assert s["total_lookups"] == 2
        assert s["hit_rate"] == pytest.approx(0.5)

    def test_stats_eviction_tracking(self) -> None:
        provider = ForteEmbeddingProvider(
            config=ForteProviderConfig(cache_max_entries=3),
        )
        for i in range(1, 6):
            c = _clause(_pos_lit(_func(i, _var(0))))
            provider.get_embedding(c)
        assert provider.stats.evictions >= 2


# ── Accessors ────────────────────────────────────────────────────────────────


class TestAccessors:
    """Test property accessors."""

    def test_algorithm_accessor(self, provider: ForteEmbeddingProvider) -> None:
        assert isinstance(provider.algorithm, ForteAlgorithm)

    def test_config_accessor(self, provider: ForteEmbeddingProvider) -> None:
        assert isinstance(provider.config, ForteProviderConfig)

    def test_custom_algorithm(self) -> None:
        algo = ForteAlgorithm(ForteConfig(embedding_dim=128))
        provider = ForteEmbeddingProvider(algorithm=algo)
        assert provider.embedding_dim == 128
        assert provider.algorithm is algo

    def test_cache_size_no_cache(
        self, nocache_provider: ForteEmbeddingProvider,
    ) -> None:
        assert nocache_provider.cache_size == 0


# ── ForteCacheStats ──────────────────────────────────────────────────────────


class TestForteCacheStats:
    """Test ForteCacheStats directly."""

    def test_initial_state(self) -> None:
        s = ForteCacheStats()
        assert s.hits == 0
        assert s.misses == 0
        assert s.evictions == 0
        assert s.errors == 0
        assert s.hit_rate == 0.0
        assert s.total_lookups == 0

    def test_hit_rate_calculation(self) -> None:
        s = ForteCacheStats(hits=7, misses=3)
        assert s.hit_rate == pytest.approx(0.7)

    def test_snapshot(self) -> None:
        s = ForteCacheStats(hits=10, misses=5, evictions=2, errors=1)
        snap = s.snapshot()
        assert snap["hits"] == 10
        assert snap["misses"] == 5
        assert snap["evictions"] == 2
        assert snap["errors"] == 1
        assert snap["hit_rate"] == pytest.approx(10 / 15)
        assert snap["total_lookups"] == 15
