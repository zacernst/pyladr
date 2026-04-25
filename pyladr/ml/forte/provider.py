"""ForteEmbeddingProvider: bridges FORTE algorithm with EmbeddingProvider protocol.

Implements the EmbeddingProvider protocol used by clause selection and
inference guidance, wrapping the pure-Python ForteAlgorithm with:
  - Structural hashing for α-equivalent clause deduplication
  - LRU cache with configurable capacity
  - Thread-safe concurrent access via ReadWriteLock
  - Graceful degradation (returns None on error)

No torch dependency — embeddings are plain Python lists. This allows FORTE
to run on any system without ML dependencies installed.

Thread-safety
-------------
The cache is protected by a ReadWriteLock:
  - Read path (get_embedding): concurrent cache lookups
  - Write path (cache miss insertion, invalidation): exclusive access
The ForteAlgorithm itself is immutable and needs no synchronization.

Typical construction::

    provider = ForteEmbeddingProvider(
        config=ForteProviderConfig(cache_max_entries=100_000),
    )
    emb = provider.get_embedding(clause)  # list[float] | None
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyladr.ml.embeddings.cache import clause_structural_hash
from pyladr.ml.forte.algorithm import ForteAlgorithm, ForteConfig

if TYPE_CHECKING:
    from pyladr.core.clause import Clause

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ForteProviderConfig:
    """Configuration for ForteEmbeddingProvider.

    Attributes:
        forte_config: Configuration for the underlying FORTE algorithm.
        cache_max_entries: Maximum number of cached embeddings (LRU eviction).
        enable_cache: Whether to enable structural caching.
    """

    forte_config: ForteConfig = field(default_factory=ForteConfig)
    cache_max_entries: int = 100_000
    enable_cache: bool = True


# ── Cache statistics ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class ForteCacheStats:
    """Live hit/miss counters for monitoring cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    errors: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_lookups(self) -> int:
        return self.hits + self.misses

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "errors": self.errors,
                "hit_rate": self.hit_rate,
                "total_lookups": self.total_lookups,
            }


# ── ForteEmbeddingProvider ───────────────────────────────────────────────────


class ForteEmbeddingProvider:
    """EmbeddingProvider backed by the FORTE feature-hashing algorithm.

    Produces deterministic 64-dimensional (configurable) embeddings for
    clauses without any neural network or torch dependency. Structural
    hashing enables α-equivalent clauses to share cached embeddings.

    Thread-safe: multiple threads may call get_embedding() concurrently.
    The ForteAlgorithm is immutable; only the cache requires synchronization.
    """

    __slots__ = (
        "_algorithm",
        "_config",
        "_cache",
        "_cache_lock",
        "_stats",
    )

    def __init__(
        self,
        config: ForteProviderConfig | None = None,
        algorithm: ForteAlgorithm | None = None,
    ) -> None:
        self._config = config or ForteProviderConfig()
        self._algorithm = algorithm or ForteAlgorithm(self._config.forte_config)

        # LRU cache: structural_hash → embedding list
        if self._config.enable_cache:
            self._cache: OrderedDict[str, list[float]] | None = OrderedDict()
        else:
            self._cache = None

        self._cache_lock = threading.Lock()
        self._stats = ForteCacheStats()

    # ── EmbeddingProvider protocol ────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of produced embeddings."""
        return self._algorithm.embedding_dim

    def get_embedding(self, clause: Clause) -> list[float] | None:
        """Return embedding for a single clause, or None on failure.

        Uses structural hashing for cache lookups. Computes via FORTE
        on cache miss. Returns None on any error for graceful degradation.
        """
        try:
            return self._get_embedding_impl(clause)
        except Exception:
            logger.debug(
                "get_embedding failed for clause %d", clause.id, exc_info=True,
            )
            self._stats.errors += 1
            return None

    def get_embeddings_batch(
        self, clauses: list[Clause],
    ) -> list[list[float] | None]:
        """Batch embedding retrieval.

        Returns a list parallel to the input: each element is either
        a list of floats (the embedding) or None if computation failed.
        """
        if not clauses:
            return []

        try:
            return [self._get_embedding_impl(c) for c in clauses]
        except Exception:
            logger.debug(
                "get_embeddings_batch failed for %d clauses",
                len(clauses),
                exc_info=True,
            )
            self._stats.errors += 1
            return [None] * len(clauses)

    # ── Implementation ────────────────────────────────────────────────

    def _get_embedding_impl(self, clause: Clause) -> list[float]:
        """Core embedding retrieval with caching."""
        cache = self._cache

        if cache is None:
            # Cache disabled — compute directly
            return self._algorithm.embed_clause(clause)

        key = clause_structural_hash(clause)

        # Fast path: cache hit (lock-free read attempt, then locked verification)
        with self._cache_lock:
            cached = cache.get(key)
            if cached is not None:
                cache.move_to_end(key)
                self._stats.hits += 1
                return cached

        # Cache miss: compute embedding (outside lock)
        self._stats.misses += 1
        embedding = self._algorithm.embed_clause(clause)

        # Insert into cache
        with self._cache_lock:
            # Double-check: another thread may have inserted while we computed
            if key not in cache:
                cache[key] = embedding
                cache.move_to_end(key)
                self._maybe_evict()
            else:
                # Another thread beat us; use cached version for consistency
                cache.move_to_end(key)

        return embedding

    def _maybe_evict(self) -> None:
        """Evict LRU entries if over capacity. Caller must hold _cache_lock."""
        cache = self._cache
        if cache is None:
            return
        max_entries = self._config.cache_max_entries
        overflow = len(cache) - max_entries
        if overflow > 0:
            for _ in range(overflow):
                cache.popitem(last=False)
            self._stats.evictions += overflow

    # ── Cache management ──────────────────────────────────────────────

    def invalidate_all(self) -> int:
        """Drop all cached embeddings. Returns count of evicted entries."""
        if self._cache is None:
            return 0
        with self._cache_lock:
            count = len(self._cache)
            self._cache.clear()
        self._stats.evictions += count
        return count

    def invalidate_clause(self, clause: Clause) -> bool:
        """Remove a single clause's cached embedding. Returns whether it existed."""
        if self._cache is None:
            return False
        key = clause_structural_hash(clause)
        with self._cache_lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.evictions += 1
                return True
            return False

    # ── Accessors ─────────────────────────────────────────────────────

    @property
    def algorithm(self) -> ForteAlgorithm:
        """The underlying FORTE algorithm instance."""
        return self._algorithm

    @property
    def config(self) -> ForteProviderConfig:
        return self._config

    @property
    def stats(self) -> ForteCacheStats:
        return self._stats

    @property
    def cache_size(self) -> int:
        """Current number of cached embeddings."""
        if self._cache is None:
            return 0
        with self._cache_lock:
            return len(self._cache)
