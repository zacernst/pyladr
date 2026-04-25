"""Tree2VecEmbeddingProvider: bridges Tree2Vec with EmbeddingProvider protocol.

Implements the EmbeddingProvider protocol used by clause selection,
wrapping the Tree2Vec algorithm with:
  - Structural hashing for α-equivalent clause deduplication
  - LRU cache with configurable capacity
  - Thread-safe concurrent access via Lock
  - Graceful degradation (returns None on error or before training)

No torch dependency — embeddings are plain Python lists.

Thread-safety
-------------
The cache is protected by a Lock:
  - Read path (get_embedding): acquire, check, release
  - Write path (cache miss insertion): acquire, insert, release
The Tree2Vec algorithm itself needs no synchronization for embedding
generation (only reads trained weights). Training must complete before
any embedding requests.

Typical construction::

    provider = Tree2VecEmbeddingProvider.from_vampire_file(
        "vampire.in",
        config=Tree2VecProviderConfig(cache_max_entries=100_000),
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
from pyladr.ml.tree2vec.algorithm import Tree2Vec, Tree2VecConfig
from pyladr.ml.tree2vec.formula_processor import (
    AugmentationConfig,
    ProcessingResult,
    process_vampire_corpus,
    process_vampire_file,
)
from pyladr.ml.tree2vec.vampire_parser import VampireCorpus

if TYPE_CHECKING:
    from pyladr.core.clause import Clause

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Tree2VecProviderConfig:
    """Configuration for Tree2VecEmbeddingProvider.

    Attributes:
        tree2vec_config: Configuration for the underlying Tree2Vec algorithm.
        augmentation_config: Data augmentation settings for training.
        cache_max_entries: Maximum number of cached embeddings (LRU eviction).
        enable_cache: Whether to enable structural caching.
    """

    tree2vec_config: Tree2VecConfig = field(default_factory=Tree2VecConfig)
    augmentation_config: AugmentationConfig = field(default_factory=AugmentationConfig)
    cache_max_entries: int = 100_000
    enable_cache: bool = True


# ── Cache statistics ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class Tree2VecCacheStats:
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


# ── Tree2VecEmbeddingProvider ────────────────────────────────────────────────


class Tree2VecEmbeddingProvider:
    """EmbeddingProvider backed by Tree2Vec unsupervised embeddings.

    Produces embeddings for clauses via skip-gram training over tree walks.
    Structural hashing enables α-equivalent clauses to share cached embeddings.

    Thread-safe: multiple threads may call get_embedding() concurrently.
    The Tree2Vec model is read-only after training; only the cache requires
    synchronization.
    """

    __slots__ = (
        "_tree2vec",
        "_config",
        "_cache",
        "_cache_lock",
        "_stats",
        "_model_version",
    )

    def __init__(
        self,
        tree2vec: Tree2Vec,
        config: Tree2VecProviderConfig | None = None,
    ) -> None:
        self._tree2vec = tree2vec
        self._config = config or Tree2VecProviderConfig()

        if self._config.enable_cache:
            self._cache: OrderedDict[str, tuple[list[float], int]] | None = OrderedDict()
        else:
            self._cache = None

        self._cache_lock = threading.Lock()
        self._stats = Tree2VecCacheStats()
        self._model_version: int = 0

    # ── Factory methods ───────────────────────────────────────────────

    @classmethod
    def from_vampire_file(
        cls,
        filepath: str,
        config: Tree2VecProviderConfig | None = None,
    ) -> Tree2VecEmbeddingProvider:
        """Create a provider by training on a vampire.in file.

        Parses the file, augments the corpus, trains Tree2Vec, and
        wraps the result in a provider with caching.

        Args:
            filepath: Path to vampire.in file.
            config: Provider configuration.

        Returns:
            Trained Tree2VecEmbeddingProvider ready for embedding requests.
        """
        config = config or Tree2VecProviderConfig()
        result = process_vampire_file(
            filepath,
            tree2vec_config=config.tree2vec_config,
            augmentation_config=config.augmentation_config,
        )
        logger.info(
            "Tree2Vec trained from %s: vocab=%d, dim=%d, loss=%.4f",
            filepath,
            result.tree2vec.vocab_size,
            result.tree2vec.embedding_dim,
            result.training_stats.get("loss", 0.0),
        )
        return cls(tree2vec=result.tree2vec, config=config)

    @classmethod
    def from_saved_model(
        cls,
        model_path: str,
        config: Tree2VecProviderConfig | None = None,
    ) -> "Tree2VecEmbeddingProvider":
        """Load a pre-trained Tree2Vec model from disk and wrap in a provider.

        The loaded model's configuration (embedding dim, walk config, etc.) is
        authoritative; any tree2vec_config in the supplied provider config is ignored.
        """
        from pyladr.ml.tree2vec.algorithm import Tree2Vec

        tree2vec = Tree2Vec.load(model_path)
        logger.info(
            "Loaded Tree2Vec model from %r: vocab=%d, dim=%d",
            model_path,
            tree2vec.vocab_size,
            tree2vec.embedding_dim,
        )
        return cls(tree2vec=tree2vec, config=config or Tree2VecProviderConfig())

    @classmethod
    def from_corpus(
        cls,
        corpus: VampireCorpus,
        config: Tree2VecProviderConfig | None = None,
    ) -> "Tree2VecEmbeddingProvider":
        """Create a provider by training on a pre-parsed corpus.

        Args:
            corpus: Pre-parsed VampireCorpus.
            config: Provider configuration.

        Returns:
            Trained Tree2VecEmbeddingProvider.
        """
        config = config or Tree2VecProviderConfig()
        result = process_vampire_corpus(
            corpus,
            tree2vec_config=config.tree2vec_config,
            augmentation_config=config.augmentation_config,
        )
        return cls(tree2vec=result.tree2vec, config=config)

    # ── EmbeddingProvider protocol ────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of produced embeddings."""
        return self._tree2vec.embedding_dim

    def get_embedding(self, clause: Clause) -> list[float] | None:
        """Return embedding for a single clause, or None on failure.

        Uses structural hashing for cache lookups. Computes via Tree2Vec
        on cache miss. Returns None on any error for graceful degradation.
        """
        if not self._tree2vec.trained:
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
        """Batch embedding retrieval.

        Returns a list parallel to the input: each element is either
        a list of floats (the embedding) or None if computation failed.
        """
        if not clauses:
            return []

        if not self._tree2vec.trained:
            return [None] * len(clauses)

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

    def _get_embedding_impl(self, clause: Clause) -> list[float] | None:
        """Core embedding retrieval with version-aware caching."""
        cache = self._cache

        if cache is None:
            return self._tree2vec.embed_clause(clause)

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
        embedding = self._tree2vec.embed_clause(clause)

        if embedding is None:
            return None

        # Insert into cache with current version
        with self._cache_lock:
            current_version = self._model_version
            cache[key] = (embedding, current_version)
            cache.move_to_end(key)
            self._maybe_evict()

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
    def tree2vec(self) -> Tree2Vec:
        """The underlying Tree2Vec model."""
        return self._tree2vec

    @property
    def config(self) -> Tree2VecProviderConfig:
        return self._config

    @property
    def stats(self) -> Tree2VecCacheStats:
        return self._stats

    @property
    def cache_size(self) -> int:
        """Current number of cached embeddings."""
        if self._cache is None:
            return 0
        with self._cache_lock:
            return len(self._cache)
