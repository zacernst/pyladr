"""RNN2VecEmbeddingProvider: bridges RNN2Vec with EmbeddingProvider protocol.

Implements the EmbeddingProvider protocol used by clause selection,
wrapping the RNN2Vec algorithm with:
  - Structural hashing for α-equivalent clause deduplication
  - LRU cache with configurable capacity
  - Thread-safe concurrent access via Lock
  - Graceful degradation (returns None on error or before training)
  - Optimized batch embedding via single RNN forward pass

No torch dependency at import time — embeddings are plain Python lists.

Thread-safety
-------------
The cache is protected by a Lock:
  - Read path (get_embedding): acquire, check, release
  - Write path (cache miss insertion): acquire, insert, release
The RNN2Vec algorithm itself needs no synchronization for embedding
generation (only reads trained weights). Training must complete before
any embedding requests.

Typical construction::

    provider = RNN2VecEmbeddingProvider.from_vampire_file(
        "vampire.in",
        config=RNN2VecProviderConfig(cache_max_entries=100_000),
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
from pyladr.ml.rnn2vec.algorithm import RNN2Vec, RNN2VecConfig
from pyladr.ml.rnn2vec.formula_processor import (
    RNN2VecProcessingResult,
    process_vampire_corpus,
    process_vampire_file,
)
from pyladr.ml.tree2vec.formula_processor import AugmentationConfig
from pyladr.ml.tree2vec.vampire_parser import VampireCorpus

if TYPE_CHECKING:
    from pyladr.core.clause import Clause

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RNN2VecProviderConfig:
    """Configuration for RNN2VecEmbeddingProvider.

    Attributes:
        rnn2vec_config: Configuration for the underlying RNN2Vec algorithm.
        augmentation_config: Data augmentation settings for training.
        cache_max_entries: Maximum number of cached embeddings (LRU eviction).
        enable_cache: Whether to enable structural caching.
    """

    rnn2vec_config: RNN2VecConfig = field(default_factory=RNN2VecConfig)
    augmentation_config: AugmentationConfig = field(default_factory=AugmentationConfig)
    cache_max_entries: int = 100_000
    enable_cache: bool = True


# ── Cache statistics ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class RNN2VecCacheStats:
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


# ── RNN2VecEmbeddingProvider ────────────────────────────────────────────────


class RNN2VecEmbeddingProvider:
    """EmbeddingProvider backed by RNN2Vec embeddings.

    Produces embeddings for clauses via RNN encoding of tree walk sequences.
    Structural hashing enables α-equivalent clauses to share cached embeddings.

    Thread-safe: multiple threads may call get_embedding() concurrently.
    The RNN2Vec model is read-only after training; only the cache requires
    synchronization.
    """

    __slots__ = (
        "_rnn2vec",
        "_config",
        "_cache",
        "_cache_lock",
        "_stats",
        "_model_version",
    )

    def __init__(
        self,
        rnn2vec: RNN2Vec,
        config: RNN2VecProviderConfig | None = None,
    ) -> None:
        self._rnn2vec = rnn2vec
        self._config = config or RNN2VecProviderConfig()

        if self._config.enable_cache:
            self._cache: OrderedDict[str, tuple[list[float], int]] | None = OrderedDict()
        else:
            self._cache = None

        self._cache_lock = threading.Lock()
        self._stats = RNN2VecCacheStats()
        self._model_version: int = 0

    # ── Factory methods ───────────────────────────────────────────────

    @classmethod
    def from_vampire_file(
        cls,
        filepath: str,
        config: RNN2VecProviderConfig | None = None,
    ) -> RNN2VecEmbeddingProvider:
        """Create a provider by training on a vampire.in file.

        Parses the file, augments the corpus, trains RNN2Vec, and
        wraps the result in a provider with caching.

        Args:
            filepath: Path to vampire.in file.
            config: Provider configuration.

        Returns:
            Trained RNN2VecEmbeddingProvider ready for embedding requests.
        """
        config = config or RNN2VecProviderConfig()
        result = process_vampire_file(
            filepath,
            rnn2vec_config=config.rnn2vec_config,
            augmentation_config=config.augmentation_config,
        )
        logger.info(
            "RNN2Vec trained from %s: vocab=%d, dim=%d, loss=%.4f",
            filepath,
            result.rnn2vec.vocab_size,
            result.rnn2vec.embedding_dim,
            result.training_stats.get("loss", 0.0),
        )
        return cls(rnn2vec=result.rnn2vec, config=config)

    @classmethod
    def from_saved_model(
        cls,
        model_path: str,
        config: RNN2VecProviderConfig | None = None,
    ) -> RNN2VecEmbeddingProvider:
        """Load a pre-trained RNN2Vec model from disk and wrap in a provider.

        The loaded model's configuration (embedding dim, RNN config, etc.) is
        authoritative; any rnn2vec_config in the supplied provider config is ignored.
        """
        rnn2vec = RNN2Vec.load(model_path)
        logger.info(
            "Loaded RNN2Vec model from %r: vocab=%d, dim=%d",
            model_path,
            rnn2vec.vocab_size,
            rnn2vec.embedding_dim,
        )
        return cls(rnn2vec=rnn2vec, config=config or RNN2VecProviderConfig())

    @classmethod
    def from_corpus(
        cls,
        corpus: VampireCorpus,
        config: RNN2VecProviderConfig | None = None,
    ) -> RNN2VecEmbeddingProvider:
        """Create a provider by training on a pre-parsed corpus.

        Args:
            corpus: Pre-parsed VampireCorpus.
            config: Provider configuration.

        Returns:
            Trained RNN2VecEmbeddingProvider.
        """
        config = config or RNN2VecProviderConfig()
        result = process_vampire_corpus(
            corpus,
            rnn2vec_config=config.rnn2vec_config,
            augmentation_config=config.augmentation_config,
        )
        return cls(rnn2vec=result.rnn2vec, config=config)

    # ── EmbeddingProvider protocol ────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of produced embeddings."""
        return self._rnn2vec.embedding_dim

    def get_embedding(self, clause: Clause) -> list[float] | None:
        """Return embedding for a single clause, or None on failure.

        Uses structural hashing for cache lookups. Computes via RNN2Vec
        on cache miss. Returns None on any error for graceful degradation.
        """
        if not self._rnn2vec.trained:
            return None
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
        """Batch embedding retrieval with optimized RNN forward pass.

        Returns a list parallel to the input: each element is either
        a list of floats (the embedding) or None if computation failed.

        Optimization: collects cache misses and computes them in a single
        batched RNN forward pass rather than one at a time.
        """
        if not clauses:
            return []

        if not self._rnn2vec.trained:
            return [None] * len(clauses)

        try:
            return self._get_embeddings_batch_impl(clauses)
        except Exception:
            logger.debug(
                "get_embeddings_batch failed for %d clauses",
                len(clauses),
                exc_info=True,
            )
            self._stats.errors += 1
            return [None] * len(clauses)

    # ── Implementation ────────────────────────────────────────────────

    def _get_embedding_impl(self, clause: Clause) -> list[float] | None:
        """Core embedding retrieval with version-aware caching."""
        cache = self._cache

        if cache is None:
            return self._rnn2vec.embed_clause(clause)

        key = clause_structural_hash(clause)

        # Check cache (version-aware: stale entries treated as misses)
        with self._cache_lock:
            cached = cache.get(key)
            if cached is not None:
                embedding, version = cached
                if version == self._model_version:
                    cache.move_to_end(key)
                    self._stats.hits += 1
                    return embedding
                # Stale entry — fall through to recompute

        # Cache miss or stale: compute embedding (outside lock)
        self._stats.misses += 1
        embedding = self._rnn2vec.embed_clause(clause)

        if embedding is None:
            return None

        # Insert into cache with current version
        with self._cache_lock:
            current_version = self._model_version
            cache[key] = (embedding, current_version)
            cache.move_to_end(key)
            self._maybe_evict()

        return embedding

    def _get_embeddings_batch_impl(
        self, clauses: list[Clause],
    ) -> list[list[float] | None]:
        """Batch implementation: check cache, batch-compute misses, insert."""
        cache = self._cache
        results: list[list[float] | None] = [None] * len(clauses)

        if cache is None:
            # No cache: use batched RNN forward pass directly
            return self._rnn2vec.embed_clauses_batch(clauses)

        # Phase 1: Check cache for all clauses
        keys: list[str] = [clause_structural_hash(c) for c in clauses]
        miss_indices: list[int] = []

        with self._cache_lock:
            for i, key in enumerate(keys):
                cached = cache.get(key)
                if cached is not None:
                    embedding, version = cached
                    if version == self._model_version:
                        cache.move_to_end(key)
                        self._stats.hits += 1
                        results[i] = embedding
                        continue
                # Miss or stale
                miss_indices.append(i)
                self._stats.misses += 1

        if not miss_indices:
            return results

        # Phase 2: Batch compute all misses in a single RNN forward pass
        miss_clauses = [clauses[i] for i in miss_indices]
        miss_embeddings = self._rnn2vec.embed_clauses_batch(miss_clauses)

        # Phase 3: Insert results into cache and assemble output
        with self._cache_lock:
            current_version = self._model_version
            for j, idx in enumerate(miss_indices):
                emb = miss_embeddings[j]
                results[idx] = emb
                if emb is not None:
                    cache[keys[idx]] = (emb, current_version)
                    cache.move_to_end(keys[idx])
            self._maybe_evict()

        return results

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

    def bump_model_version(self) -> int:
        """Increment model version, lazily invalidating all cache entries.

        Stale cache entries (version < current) will be treated as misses
        on the next lookup. This avoids the cost of a full cache clear.

        Returns:
            The new model version number.
        """
        with self._cache_lock:
            self._model_version += 1
            return self._model_version

    @property
    def model_version(self) -> int:
        """Current model version counter."""
        return self._model_version

    def invalidate_all(self) -> int:
        """Drop all cached embeddings. Returns count of evicted entries."""
        if self._cache is None:
            return 0
        with self._cache_lock:
            count = len(self._cache)
            self._cache.clear()
        self._stats.evictions += count
        return count

    # ── Accessors ─────────────────────────────────────────────────────

    @property
    def rnn2vec(self) -> RNN2Vec:
        """The underlying RNN2Vec model."""
        return self._rnn2vec

    @property
    def config(self) -> RNN2VecProviderConfig:
        return self._config

    @property
    def stats(self) -> RNN2VecCacheStats:
        return self._stats

    @property
    def cache_size(self) -> int:
        """Current number of cached embeddings."""
        if self._cache is None:
            return 0
        with self._cache_lock:
            return len(self._cache)
