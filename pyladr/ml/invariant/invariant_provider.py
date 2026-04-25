"""Invariant embedding provider — symbol-independent clause embeddings.

Implements the EmbeddingProvider protocol with property-invariant features.
This is a drop-in replacement for GNNEmbeddingProvider that produces
embeddings invariant to symbol renaming.

Architecture:
  - Uses _InvariantGraphBuilder for graph construction (canonical symbol IDs)
  - Reuses the same HeterogeneousClauseGNN architecture (no model changes)
  - Uses invariant structural hashing for cache keys (symbol-independent)
  - Thread-safe via the same RWLock pattern as GNNEmbeddingProvider

The key insight: by canonicalizing symbol features at the graph construction
level, the downstream GNN naturally produces invariant embeddings without
any architectural changes. The GNN's symbol embedding table learns to map
canonical roles rather than arbitrary symbol IDs.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pyladr.core.clause import Clause

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyladr.core.symbol import SymbolTable

logger = logging.getLogger(__name__)

try:
    import torch
    from torch_geometric.data import Batch, HeteroData

    from pyladr.ml.embeddings.cache import CacheConfig, EmbeddingCache
    from pyladr.ml.embedding_provider import _harmonize_graphs
    from pyladr.ml.graph.clause_encoder import (
        GNNConfig,
        HeterogeneousClauseGNN,
        load_model,
    )
    from pyladr.ml.graph.clause_graph import ClauseGraphConfig
    from pyladr.ml.invariant.invariant_features import (
        invariant_clause_structural_hash,
    )
    from pyladr.ml.invariant.invariant_graph import (
        batch_invariant_clauses_to_heterograph,
    )

    _ML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ML_AVAILABLE = False


@dataclass(frozen=True, slots=True)
class InvariantProviderConfig:
    """Configuration for the invariant embedding provider.

    Attributes:
        model_path: Path to a saved GNN checkpoint.
        device: PyTorch device string.
        cache_max_entries: Maximum cached embeddings.
        graph_max_term_depth: Depth limit for term graph traversal.
        graph_include_variable_sharing: Whether to add SHARED_VARIABLE edges.
    """

    model_path: str = ""
    device: str = "cpu"
    cache_max_entries: int = 100_000
    graph_max_term_depth: int = 0
    graph_include_variable_sharing: bool = True


class InvariantEmbeddingProvider:
    """Symbol-independent embedding provider implementing EmbeddingProvider.

    Produces embeddings that are invariant to symbol renaming by using
    canonical symbol features during graph construction. The GNN architecture
    is unchanged — invariance comes from the input representation.

    Thread-safe via a readers-writer lock for model access.
    """

    __slots__ = (
        "_model",
        "_cache",
        "_symbol_table",
        "_graph_config",
        "_device",
        "_model_lock",
        "_model_version",
        "_swap_count",
    )

    def __init__(
        self,
        model: HeterogeneousClauseGNN,
        cache: EmbeddingCache,
        symbol_table: SymbolTable | None = None,
        graph_config: ClauseGraphConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._cache = cache
        self._symbol_table = symbol_table
        self._graph_config = graph_config or ClauseGraphConfig()
        self._device = device
        self._model_lock = threading.RLock()
        self._model_version: int = 0
        self._swap_count: int = 0

        # Register ourselves as the cache's compute function
        self._cache.compute_fn = self

    # ── EmbeddingProvider protocol ─────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        return self._model.config.embedding_dim

    def get_embedding(self, clause: Clause) -> list[float] | None:
        try:
            result = self.get_or_compute_batch([clause])
            return result[0].tolist()
        except Exception:
            logger.debug(
                "get_embedding failed for clause %d", clause.id, exc_info=True
            )
            return None

    def get_embeddings_batch(
        self, clauses: list[Clause],
    ) -> list[list[float] | None]:
        if not clauses:
            return []
        try:
            result = self.get_or_compute_batch(clauses)
            return [row.tolist() for row in result.unbind(0)]
        except Exception:
            logger.debug(
                "get_embeddings_batch failed for %d clauses",
                len(clauses),
                exc_info=True,
            )
            return [None] * len(clauses)

    # ── EmbeddingComputer protocol (cache miss computation) ────────────

    def compute_embeddings(
        self, clauses: Sequence[Clause],
    ) -> torch.Tensor:
        """Compute invariant embeddings via the GNN.

        Uses invariant graph construction (canonical symbol features)
        so that symbol-renamed clauses produce identical embeddings.
        """
        graphs = batch_invariant_clauses_to_heterograph(
            list(clauses), self._symbol_table, self._graph_config
        )

        if not graphs:
            return torch.empty(0, self.embedding_dim, device=self._device)

        with self._model_lock:
            if len(graphs) == 1:
                data = graphs[0].to(self._device)
                return self._model.embed_clause(data)

            _harmonize_graphs(graphs)
            batch = Batch.from_data_list(graphs)
            batch = batch.to(self._device)
            return self._model.embed_clause(batch)

    # ── Batch computation with invariant cache keys ────────────────────

    def get_or_compute_batch(
        self, clauses: Sequence[Clause],
    ) -> torch.Tensor:
        """Get embeddings using invariant structural hashing for cache keys.

        Uses symbol-independent structural hashes so that clauses with
        renamed symbols share cached embeddings.
        """
        n = len(clauses)
        self._cache.stats.record_batch(n)

        # Phase 1: partition using invariant hashes
        keys = [invariant_clause_structural_hash(c) for c in clauses]

        hit_indices: list[int] = []
        miss_indices: list[int] = []
        hit_tensors: list[torch.Tensor] = []

        with self._cache._rw_lock.read_lock():
            for i, key in enumerate(keys):
                cached = self._cache._data.get(key)
                if cached is not None:
                    hit_indices.append(i)
                    hit_tensors.append(cached)
                else:
                    miss_indices.append(i)

        for _ in hit_indices:
            self._cache.stats.record_hit()
        for _ in miss_indices:
            self._cache.stats.record_miss()

        # Phase 2: compute misses
        miss_tensors: list[torch.Tensor] = []
        if miss_indices:
            miss_clauses = [clauses[i] for i in miss_indices]
            batch_result = self.compute_embeddings(miss_clauses)
            miss_tensors = list(batch_result.unbind(0))

        # Phase 3: insert misses
        if miss_tensors:
            with self._cache._rw_lock.write_lock():
                for idx, tensor in zip(miss_indices, miss_tensors):
                    key = keys[idx]
                    self._cache._data[key] = tensor
                    self._cache._data.move_to_end(key)
                self._cache._maybe_evict()

        # Touch hits for LRU
        if hit_indices:
            with self._cache._rw_lock.write_lock():
                for i in hit_indices:
                    key = keys[i]
                    if key in self._cache._data:
                        self._cache._data.move_to_end(key)

        # Phase 4: assemble output
        result = torch.empty(n, self.embedding_dim, device=self._device)
        hit_iter = iter(hit_tensors)
        miss_iter = iter(miss_tensors)
        hit_set = frozenset(hit_indices)
        for i in range(n):
            if i in hit_set:
                result[i] = next(hit_iter)
            else:
                result[i] = next(miss_iter)

        return result

    # ── Accessors ──────────────────────────────────────────────────────

    @property
    def model(self) -> HeterogeneousClauseGNN:
        return self._model

    @property
    def cache(self) -> EmbeddingCache:
        return self._cache

    @property
    def symbol_table(self) -> SymbolTable | None:
        return self._symbol_table

    @symbol_table.setter
    def symbol_table(self, st: SymbolTable | None) -> None:
        self._symbol_table = st

    @property
    def model_version(self) -> int:
        return self._model_version

    @property
    def swap_count(self) -> int:
        return self._swap_count

    @property
    def stats(self) -> dict:
        base = self._cache.stats.snapshot()
        base["model_version"] = self._model_version
        base["swap_count"] = self._swap_count
        return base

    # ── Model hot-swapping ─────────────────────────────────────────────

    def swap_weights(self, state_dict: dict) -> int:
        with self._model_lock:
            self._model.load_state_dict(state_dict)
            self._model.eval()
            self._model_version += 1
            self._swap_count += 1
        self._cache.on_model_update()
        logger.info(
            "Invariant model weights hot-swapped (version=%d)",
            self._model_version,
        )
        return self._model_version

    def swap_model(self, new_model: HeterogeneousClauseGNN) -> int:
        with self._model_lock:
            self._model = new_model
            self._model.eval()
            self._model_version += 1
            self._swap_count += 1
        self._cache.on_model_update()
        return self._model_version

    def checkpoint(self) -> dict:
        with self._model_lock:
            return {k: v.clone() for k, v in self._model.state_dict().items()}

    def restore_checkpoint(self, state: dict) -> None:
        with self._model_lock:
            self._model.load_state_dict(state)
            self._model.eval()
            self._model_version += 1
            self._swap_count += 1
        self._cache.on_model_update()

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        symbol_table: SymbolTable | None = None,
        config: InvariantProviderConfig | None = None,
        gnn_config: GNNConfig | None = None,
    ) -> InvariantEmbeddingProvider:
        """Create a fully wired InvariantEmbeddingProvider."""
        if not _ML_AVAILABLE:
            raise ImportError(
                "ML dependencies not available. Install with: pip install pyladr[ml]"
            )

        cfg = config or InvariantProviderConfig()

        if cfg.model_path and Path(cfg.model_path).exists():
            model, _ = load_model(cfg.model_path, device=cfg.device)
        else:
            model = HeterogeneousClauseGNN(gnn_config or GNNConfig())
            model.to(cfg.device)
            model.eval()

        graph_config = ClauseGraphConfig(
            max_term_depth=cfg.graph_max_term_depth,
            include_variable_sharing=cfg.graph_include_variable_sharing,
        )

        cache_config = CacheConfig(
            max_entries=cfg.cache_max_entries,
            embedding_dim=model.config.embedding_dim,
            device=cfg.device,
        )
        cache = EmbeddingCache(config=cache_config)

        return cls(
            model=model,
            cache=cache,
            symbol_table=symbol_table,
            graph_config=graph_config,
            device=cfg.device,
        )
