"""Bridge between the GNN architecture and the search component protocols.

Implements the EmbeddingProvider protocol (used by ml_selection and
inference_guidance) by wiring together:
  - HeterogeneousClauseGNN (graph → embedding)
  - EmbeddingCache (structural hashing + LRU caching)
  - SymbolTable (symbol metadata for graph construction)

Also implements EmbeddingComputer (used by the cache for batch miss
computation) as an internal detail.

All ML dependencies are optional — a NoOpEmbeddingProvider is available
as a graceful fallback when torch/torch_geometric are not installed.

Thread-safety
-------------
``GNNEmbeddingProvider`` supports concurrent embedding requests from
multiple search threads and model hot-swapping from the online learning
thread.  A readers–writer lock protects the model reference: concurrent
reads during forward passes, exclusive writes during model swaps.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pyladr.core.clause import Clause
from pyladr.core.symbol import SymbolTable

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Guard ML imports — everything must work without torch installed.
try:
    import torch
    from torch_geometric.data import Batch, HeteroData

    from pyladr.ml.embeddings.cache import CacheConfig, EmbeddingCache
    from pyladr.ml.graph.clause_encoder import (
        GNNConfig,
        HeterogeneousClauseGNN,
        load_model,
    )
    from pyladr.ml.graph.clause_graph import (
        ClauseGraphConfig,
        NodeType,
        batch_clauses_to_heterograph,
        clause_to_heterograph,
    )

    _ML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ML_AVAILABLE = False


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EmbeddingProviderConfig:
    """Configuration for the GNN embedding provider.

    Attributes:
        model_path: Path to a saved GNN checkpoint. If empty, a freshly
            initialised model is created (useful for online learning).
        device: PyTorch device string ("cpu", "cuda", "mps").
        cache_max_entries: Maximum cached embeddings.
        graph_max_term_depth: Depth limit for term graph traversal
            (0 = unlimited).
        graph_include_variable_sharing: Whether to add SHARED_VARIABLE edges.
    """

    model_path: str = ""
    device: str = "cpu"
    cache_max_entries: int = 100_000
    graph_max_term_depth: int = 0
    graph_include_variable_sharing: bool = True


# ── GNN Embedding Provider ─────────────────────────────────────────────────


class GNNEmbeddingProvider:
    """Bridges HeterogeneousClauseGNN with the EmbeddingProvider protocol.

    Satisfies both:
      - ``EmbeddingProvider`` (from ``ml_selection``) — used by
        ``EmbeddingEnhancedSelection`` and ``EmbeddingGuidedInference``.
      - ``EmbeddingComputer`` (from ``embeddings.cache``) — used by
        ``EmbeddingCache`` for computing misses.

    Thread-safety
    ~~~~~~~~~~~~~
    A readers–writer lock (``_model_lock``) protects the model reference:

    * **Read path** (``compute_embeddings``, ``get_embedding``,
      ``get_embeddings_batch``):  acquires the read lock so that
      multiple inference threads can run the GNN forward pass
      concurrently.
    * **Write path** (``swap_weights``, ``swap_model``,
      ``restore_checkpoint``):  acquires the write lock, guaranteeing
      that no forward pass is in flight while the model or its weights
      are being replaced.

    The cache has its own internal ``ReadWriteLock``; it is *never*
    held simultaneously with ``_model_lock`` in the same call chain,
    so deadlocks cannot occur.

    Typical construction::

        provider = GNNEmbeddingProvider.create(
            symbol_table=st,
            config=EmbeddingProviderConfig(model_path="model.pt"),
        )
        selection = EmbeddingEnhancedSelection(
            embedding_provider=provider,
            ml_config=MLSelectionConfig(enabled=True),
        )
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
        # Lock ordering: _model_lock is always acquired BEFORE any cache
        # lock.  The write path (swap_weights, swap_model, restore_checkpoint)
        # acquires _model_lock, releases it, then calls cache.on_model_update()
        # which acquires the cache's internal lock.  This ensures the two locks
        # are never held simultaneously, preventing deadlocks.
        self._model_lock = threading.RLock()
        self._model_version: int = 0
        self._swap_count: int = 0

        # Register ourselves as the cache's compute function
        self._cache.compute_fn = self

    # ── EmbeddingProvider protocol ─────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of produced embeddings."""
        return self._model.config.embedding_dim

    def get_embedding(self, clause: Clause) -> list[float] | None:
        """Return embedding for a single clause, or None on failure.

        Uses the cache for lookups. Computes via the GNN on cache miss.
        """
        try:
            result = self._cache.get_or_compute_batch([clause])
            return result[0].tolist()
        except Exception:
            logger.debug("get_embedding failed for clause %d", clause.id, exc_info=True)
            return None

    def get_embeddings_batch(
        self, clauses: list[Clause],
    ) -> list[list[float] | None]:
        """Batch embedding retrieval with cache integration.

        Returns a list parallel to the input: each element is either
        a list of floats (the embedding) or None if computation failed.
        """
        if not clauses:
            return []

        try:
            result = self._cache.get_or_compute_batch(clauses)
            return [row.tolist() for row in result.unbind(0)]
        except Exception:
            logger.debug(
                "get_embeddings_batch failed for %d clauses",
                len(clauses),
                exc_info=True,
            )
            return [None] * len(clauses)

    # ── EmbeddingComputer protocol (used by cache on misses) ───────────

    def compute_embeddings(
        self, clauses: Sequence[Clause],
    ) -> torch.Tensor:
        """Compute embeddings for a batch of clauses via the GNN.

        This is called by EmbeddingCache.get_or_compute_batch() for
        cache misses. Returns a tensor of shape (N, embedding_dim).

        Thread-safe: acquires the model lock to ensure the model is not
        being swapped while a forward pass is in progress.

        Security: validates output tensor for NaN/Inf values and correct
        dimensions to prevent corrupted embeddings from propagating into
        selection scoring.
        """
        graphs = batch_clauses_to_heterograph(
            list(clauses), self._symbol_table, self._graph_config
        )

        if not graphs:
            return torch.empty(0, self.embedding_dim, device=self._device)

        # Hold the model lock for the duration of the forward pass.
        with self._model_lock:
            if len(graphs) == 1:
                data = graphs[0].to(self._device)
                result = self._model.embed_clause(data)
            else:
                _harmonize_graphs(graphs)
                batch = Batch.from_data_list(graphs)
                batch = batch.to(self._device)
                result = self._model.embed_clause(batch)

        return _validate_embeddings(result, len(clauses), self.embedding_dim)

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
        """Update the symbol table (e.g., when starting a new problem)."""
        self._symbol_table = st

    @property
    def model_version(self) -> int:
        """Current model version (incremented on each hot-swap)."""
        return self._model_version

    @property
    def swap_count(self) -> int:
        """Total number of model swaps performed."""
        return self._swap_count

    @property
    def graph_config(self) -> ClauseGraphConfig:
        """Graph construction configuration."""
        return self._graph_config

    @property
    def device(self) -> str:
        """PyTorch device string."""
        return self._device

    @property
    def model_lock(self) -> threading.RLock:
        """Model readers-writer lock for thread-safe forward passes."""
        return self._model_lock

    @property
    def stats(self) -> dict:
        """Combined statistics from the cache and model versioning."""
        base = self._cache.stats.snapshot()
        base["model_version"] = self._model_version
        base["swap_count"] = self._swap_count
        return base

    # ── Model hot-swapping ─────────────────────────────────────────────

    def swap_weights(self, state_dict: dict) -> int:
        """Atomically replace the model weights and invalidate the cache.

        This is the primary API for ``OnlineLearningManager`` to push
        updated weights into the embedding provider during search.

        The operation is thread-safe: no forward pass can be in flight
        while the weights are being replaced.

        Args:
            state_dict: New model state dict (as from ``model.state_dict()``).

        Returns:
            The new model version number after the swap.
        """
        with self._model_lock:
            self._model.load_state_dict(state_dict)
            self._model.eval()
            self._model_version += 1
            self._swap_count += 1

        # Invalidate cache outside the model lock (cache has its own lock)
        self._cache.on_model_update()

        logger.info(
            "Model weights hot-swapped (version=%d, swap_count=%d)",
            self._model_version, self._swap_count,
        )
        return self._model_version

    def swap_model(self, new_model: HeterogeneousClauseGNN) -> int:
        """Replace the entire model object and invalidate the cache.

        Use this when the model architecture has changed (e.g., after
        loading a checkpoint with different hyperparameters).  For
        weight-only updates during online learning, prefer
        ``swap_weights()`` which is cheaper.

        Args:
            new_model: The new model instance.

        Returns:
            The new model version number after the swap.
        """
        with self._model_lock:
            self._model = new_model
            self._model.eval()
            self._model_version += 1
            self._swap_count += 1

        self._cache.on_model_update()

        logger.info(
            "Model instance hot-swapped (version=%d, swap_count=%d)",
            self._model_version, self._swap_count,
        )
        return self._model_version

    # ── Model lifecycle ────────────────────────────────────────────────

    def checkpoint(self) -> dict:
        """Save current model state for potential rollback.

        Returns:
            State dict that can be passed to ``restore_checkpoint()``.
        """
        with self._model_lock:
            return {k: v.clone() for k, v in self._model.state_dict().items()}

    def restore_checkpoint(self, state: dict) -> None:
        """Restore a previously saved model checkpoint.

        Thread-safe: acquires the model lock, then invalidates the cache.

        Args:
            state: State dict from a prior ``checkpoint()`` call.
        """
        with self._model_lock:
            self._model.load_state_dict(state)
            self._model.eval()
            self._model_version += 1
            self._swap_count += 1

        self._cache.on_model_update()

    def set_eval_mode(self) -> None:
        """Switch model to eval mode (disables dropout, etc.)."""
        with self._model_lock:
            self._model.eval()

    def set_train_mode(self) -> None:
        """Switch model to training mode (enables dropout, etc.)."""
        with self._model_lock:
            self._model.train()

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        symbol_table: SymbolTable | None = None,
        config: EmbeddingProviderConfig | None = None,
        gnn_config: GNNConfig | None = None,
    ) -> GNNEmbeddingProvider:
        """Create a fully wired GNNEmbeddingProvider.

        Args:
            symbol_table: SymbolTable for graph construction.
            config: Provider configuration.
            gnn_config: GNN hyperparameters (used when model_path is empty).

        Returns:
            A ready-to-use GNNEmbeddingProvider instance.
        """
        if not _ML_AVAILABLE:
            raise ImportError(
                "ML dependencies not available. Install with: "
                "pip install pyladr[ml]"
            )

        cfg = config or EmbeddingProviderConfig()

        # Load or create the GNN model
        if cfg.model_path and Path(cfg.model_path).exists():
            model, _metadata = load_model(cfg.model_path, device=cfg.device)
        else:
            model = HeterogeneousClauseGNN(gnn_config or GNNConfig())
            model.to(cfg.device)
            model.eval()

        # Create graph construction config
        graph_config = ClauseGraphConfig(
            max_term_depth=cfg.graph_max_term_depth,
            include_variable_sharing=cfg.graph_include_variable_sharing,
        )

        # Create cache
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


# ── Embedding validation ───────────────────────────────────────────────────


def _validate_embeddings(
    tensor: torch.Tensor,
    expected_rows: int,
    expected_dim: int,
) -> torch.Tensor:
    """Validate embedding tensor for shape correctness and finite values.

    Replaces any rows containing NaN or Inf with zero vectors rather than
    allowing corrupted values to propagate into selection scoring. This
    prevents a corrupted or adversarial model from crashing the search.

    Args:
        tensor: Raw embedding output from the GNN.
        expected_rows: Expected number of clause embeddings.
        expected_dim: Expected embedding dimensionality.

    Returns:
        Validated tensor with NaN/Inf rows zeroed out.
    """
    # Dimension validation
    if tensor.ndim != 2:
        logger.warning(
            "Embedding tensor has %d dims (expected 2), returning zeros",
            tensor.ndim,
        )
        return torch.zeros(expected_rows, expected_dim, device=tensor.device)

    if tensor.shape[0] != expected_rows:
        logger.warning(
            "Embedding batch size %d != expected %d, returning zeros",
            tensor.shape[0], expected_rows,
        )
        return torch.zeros(expected_rows, expected_dim, device=tensor.device)

    if tensor.shape[1] != expected_dim:
        logger.warning(
            "Embedding dim %d != expected %d, returning zeros",
            tensor.shape[1], expected_dim,
        )
        return torch.zeros(expected_rows, expected_dim, device=tensor.device)

    # Finite value validation — replace NaN/Inf rows with zeros
    finite_mask = torch.isfinite(tensor).all(dim=1)
    if not finite_mask.all():
        bad_count = (~finite_mask).sum().item()
        logger.warning(
            "Embedding tensor has %d/%d non-finite rows, replacing with zeros",
            bad_count, expected_rows,
        )
        tensor = tensor.clone()
        tensor[~finite_mask] = 0.0

    return tensor


# ── Graph harmonization helper ─────────────────────────────────────────────


def _harmonize_graphs(graphs: list[HeteroData]) -> None:
    """Ensure all graphs have consistent node type attributes for batching.

    PyG's Batch.from_data_list fails if some graphs have a node type with
    an ``x`` attribute and others don't. This adds empty (0-row) feature
    tensors for any missing node types so collation succeeds.

    Mutates the graphs in-place.
    """
    # Collect the union of (node_type, feature_dim) across all graphs
    type_dims: dict[str, int] = {}
    for g in graphs:
        for nt in NodeType:
            key = nt.value
            if key in g.node_types:
                store = g[key]
                if hasattr(store, "x") and store.x is not None:
                    type_dims[key] = store.x.shape[1]

    # Pad missing node types with empty tensors
    for g in graphs:
        for key, dim in type_dims.items():
            store = g[key]
            if not hasattr(store, "x") or store.x is None:
                store.x = torch.empty(0, dim, dtype=torch.float32)
                store.num_nodes = 0


# ── ClauseEncoder adapter ─────────────────────────────────────────────────


class GNNClauseEncoder:
    """Adapts GNNEmbeddingProvider to the ClauseEncoder protocol.

    ``OnlineLearningManager`` expects an encoder with ``encode_clauses()``,
    ``parameters()``, ``named_parameters()``, ``state_dict()``,
    ``load_state_dict()``, ``train()``, and ``eval()``.

    ``HeterogeneousClauseGNN`` only has ``embed_clause()`` (which takes PyG
    ``HeteroData``, not raw ``Clause`` objects, and detaches the output).

    This adapter bridges the gap by:
    - Implementing ``encode_clauses()`` via the graph-building pipeline +
      ``model.forward()`` (with gradients, for training).
    - Delegating ``parameters()`` et al. to the underlying GNN model.

    Thread-safety: delegates to the provider's model lock for forward passes.
    """

    __slots__ = ("_provider",)

    def __init__(self, provider: GNNEmbeddingProvider) -> None:
        self._provider = provider

    def encode_clauses(self, clauses: list[Clause]) -> torch.Tensor:
        """Encode clauses into embeddings WITH gradients for training.

        Unlike ``GNNEmbeddingProvider.compute_embeddings()`` (which uses
        ``embed_clause`` → ``no_grad`` → detach), this path calls
        ``model.forward()`` directly so that gradients flow back through
        the GNN for contrastive learning.

        Args:
            clauses: List of Clause objects.

        Returns:
            Tensor of shape ``(len(clauses), embedding_dim)`` with grad.
        """
        graphs = batch_clauses_to_heterograph(
            list(clauses),
            self._provider.symbol_table,
            self._provider.graph_config,
        )

        if not graphs:
            return torch.empty(
                0, self._provider.embedding_dim,
                device=self._provider.device,
            )

        with self._provider.model_lock:
            model = self._provider.model

            if len(graphs) == 1:
                data = graphs[0].to(self._provider.device)
                return model.forward(data)

            _harmonize_graphs(graphs)
            batch = Batch.from_data_list(graphs)
            batch = batch.to(self._provider.device)
            return model.forward(batch)

    # ── Delegate nn.Module-like interface to the underlying model ─────

    def parameters(self):
        return self._provider.model.parameters()

    def named_parameters(self):
        return self._provider.model.named_parameters()

    def state_dict(self):
        return self._provider.model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self._provider.model.load_state_dict(state_dict)

    def train(self, mode: bool = True):
        self._provider.model.train(mode)

    def eval(self):
        self._provider.model.eval()


# ── No-op fallback ────────────────────────────────────────────────────────


class NoOpEmbeddingProvider:
    """Fallback provider when ML dependencies are not available.

    Always returns None for embeddings, causing search components to
    fall back to traditional scoring. This ensures the system works
    without torch/torch_geometric installed.
    """

    __slots__ = ("_embedding_dim",)

    def __init__(self, embedding_dim: int = 512) -> None:
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def get_embedding(self, clause: Clause) -> list[float] | None:
        return None

    def get_embeddings_batch(
        self, clauses: list[Clause],
    ) -> list[list[float] | None]:
        return [None] * len(clauses)


# ── Convenience factory ──────────────────────────────────────────────────


def create_embedding_provider(
    symbol_table: SymbolTable | None = None,
    config: EmbeddingProviderConfig | None = None,
    gnn_config: GNNConfig | None = None,
) -> GNNEmbeddingProvider | NoOpEmbeddingProvider:
    """Create the best available embedding provider.

    Returns a GNNEmbeddingProvider if ML dependencies are installed,
    otherwise returns a NoOpEmbeddingProvider that gracefully degrades.

    Args:
        symbol_table: SymbolTable for graph construction.
        config: Provider configuration.
        gnn_config: GNN hyperparameters.

    Returns:
        An embedding provider satisfying the EmbeddingProvider protocol.
    """
    if not _ML_AVAILABLE:
        logger.info(
            "ML dependencies not available, using NoOpEmbeddingProvider. "
            "Install with: pip install pyladr[ml]"
        )
        dim = (gnn_config.embedding_dim if gnn_config else 512)
        return NoOpEmbeddingProvider(embedding_dim=dim)

    try:
        return GNNEmbeddingProvider.create(
            symbol_table=symbol_table,
            config=config,
            gnn_config=gnn_config,
        )
    except Exception:
        logger.warning(
            "Failed to create GNNEmbeddingProvider, falling back to NoOp",
            exc_info=True,
        )
        dim = (gnn_config.embedding_dim if gnn_config else 512)
        return NoOpEmbeddingProvider(embedding_dim=dim)


# ── Convenience alias ──────────────────────────────────────────────────────

# Provide EmbeddingProvider as an alias to the factory function for backwards
# compatibility with demo scripts
EmbeddingProvider = create_embedding_provider
