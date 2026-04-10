"""High-performance embedding cache with GPU acceleration.

Provides LRU-evicting, thread-safe storage for clause embeddings computed
by the graph neural network.  Designed for sub-millisecond lookups during
given-clause selection and inference guidance.

Key design decisions
--------------------
* **Structural hashing for cache keys** – Two clauses with identical literal
  structure (after variable renumbering) share a single embedding.  This is
  the primary driver of the >80 % target hit rate.
* **GPU-resident tensors** – Cached embeddings stay on the device used for
  inference, avoiding repeated CPU ↔ GPU transfers.
* **Batch interface** – ``get_or_compute_batch`` amortises GNN forward-pass
  cost across many clauses at once, which is critical for parallelised
  inference generation.
* **Memory-pressure handling** – The cache monitors GPU memory and
  automatically evicts the coldest entries when utilisation exceeds a
  configurable threshold.
* **Thread safety** – All public methods are safe under free-threaded Python
  (PEP 703) via a readers–writer lock from ``pyladr.threading_guide``.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

from pyladr.core.clause import Clause
from pyladr.threading_guide import ReadWriteLock, make_rw_lock


# ---------------------------------------------------------------------------
# Structural hashing – cache key derivation
# ---------------------------------------------------------------------------

def _literal_structural_key(lit: object) -> str:
    """Produce a canonical string for a Literal's structure.

    Variable identities are preserved (x0 != x1) but concrete variable
    numbers are *not* – they are normalised to their first-occurrence order
    so that α-equivalent clauses collide.

    Uses iterative traversal with a parts list to reduce string allocations.
    """
    var_map: dict[int, int] = {}
    counter = 0
    parts: list[str] = []

    def _term_key(t: object) -> None:
        nonlocal counter
        if t.is_variable:  # type: ignore[union-attr]
            vn = t.varnum  # type: ignore[union-attr]
            if vn not in var_map:
                var_map[vn] = counter
                counter += 1
            parts.append("v")
            parts.append(str(var_map[vn]))
            return
        parts.append("f(")
        parts.append(str(t.private_symbol))  # type: ignore[union-attr]
        for a in t.args:  # type: ignore[union-attr]
            parts.append(",")
            _term_key(a)
        parts.append(")")

    parts.append("+" if lit.sign else "-")  # type: ignore[union-attr]
    _term_key(lit.atom)  # type: ignore[union-attr]
    return "".join(parts)


def clause_structural_hash(clause: Clause) -> str:
    """Return a deterministic structural hash for *clause*.

    The hash is independent of the clause's ``id``, ``weight``, or other
    mutable metadata – only the literal structure matters.  Variable numbers
    are normalised so that α-equivalent clauses produce the same key.

    Results are cached on the clause object to avoid recomputation on
    repeated lookups (e.g., multiple cache checks for the same clause).
    """
    # Fast path: return cached hash if available
    cached = getattr(clause, "_structural_hash", None)
    if cached is not None:
        return cached

    # Sort literal keys for multiset semantics (clause = disjunction, order
    # does not matter logically).
    lit_keys = sorted(_literal_structural_key(lit) for lit in clause.literals)
    raw = "|".join(lit_keys)
    result = hashlib.blake2b(raw.encode(), digest_size=16).hexdigest()

    # Cache on clause object (best-effort — Clause may have __slots__)
    try:
        clause._structural_hash = result  # type: ignore[attr-defined]
    except AttributeError:
        pass  # Clause uses __slots__, cannot cache — still correct

    return result


# ---------------------------------------------------------------------------
# Configuration & statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Tuning knobs for :class:`EmbeddingCache`.

    Attributes
    ----------
    max_entries:
        Hard upper bound on cached embeddings.  When exceeded the least-
        recently-used entry is evicted.
    gpu_memory_fraction:
        Fraction (0–1) of *available* GPU memory the cache may occupy.
        Checked lazily – the cache only queries ``torch.cuda`` when it
        decides to grow.  Set to 1.0 to disable memory-pressure eviction.
    embedding_dim:
        Dimensionality of the clause embedding vectors.  Must agree with
        the GNN output head.
    device:
        PyTorch device string (``"cuda"``, ``"cpu"``, ``"mps"``, …).
    warmup_common_patterns:
        If ``True``, the cache pre-allocates storage for common clause
        shapes (unit equalities, unit atoms, Horn clauses) on first use.
    eviction_batch_size:
        Number of entries to evict in one go when memory pressure is
        detected, to amortise the cost of the eviction sweep.
    persist_path:
        Optional filesystem path for cache persistence across sessions.
        ``None`` disables persistence.
    """

    max_entries: int = 100_000
    gpu_memory_fraction: float = 0.85
    embedding_dim: int = 512
    device: str = "cpu"
    warmup_common_patterns: bool = True
    eviction_batch_size: int = 256
    persist_path: str | None = None


@dataclass(slots=True)
class CacheStatistics:
    """Live hit/miss counters exposed to the search loop for monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    batch_requests: int = 0
    batch_total_clauses: int = 0
    memory_pressure_events: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_lookups(self) -> int:
        return self.hits + self.misses

    def record_hit(self) -> None:
        with self._lock:
            self.hits += 1

    def record_miss(self) -> None:
        with self._lock:
            self.misses += 1

    def record_evictions(self, count: int) -> None:
        with self._lock:
            self.evictions += count

    def record_batch(self, size: int) -> None:
        with self._lock:
            self.batch_requests += 1
            self.batch_total_clauses += size

    def record_memory_pressure(self) -> None:
        with self._lock:
            self.memory_pressure_events += 1

    def reset(self) -> None:
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.batch_requests = 0
            self.batch_total_clauses = 0
            self.memory_pressure_events = 0

    def snapshot(self) -> dict[str, int | float]:
        """Return a point-in-time copy of all counters."""
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": self.hit_rate,
                "total_lookups": self.total_lookups,
                "batch_requests": self.batch_requests,
                "batch_total_clauses": self.batch_total_clauses,
                "memory_pressure_events": self.memory_pressure_events,
            }


# ---------------------------------------------------------------------------
# Embedding computation protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbeddingComputer(Protocol):
    """Callback protocol for computing embeddings from clauses.

    Implementers must accept a list of clauses and return a tensor of shape
    ``(len(clauses), embedding_dim)`` on the correct device.
    """

    def compute_embeddings(
        self,
        clauses: Sequence[Clause],
    ) -> torch.Tensor: ...  # type: ignore[name-defined]


# ---------------------------------------------------------------------------
# The cache itself
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """GPU-accelerated LRU cache for clause embeddings.

    Thread-safe via a readers–writer lock: concurrent reads during inference
    generation, exclusive writes during cache updates.

    Parameters
    ----------
    config:
        Cache configuration (sizes, device, thresholds).
    compute_fn:
        Optional callable satisfying :class:`EmbeddingComputer` that will be
        invoked on cache misses inside ``get_or_compute_batch``.  Can also
        be set later via :pyattr:`compute_fn`.
    """

    __slots__ = (
        "_config",
        "_data",
        "_rw_lock",
        "_stats",
        "_compute_fn",
        "_device",
        "_model_version",
    )

    def __init__(
        self,
        config: CacheConfig | None = None,
        compute_fn: EmbeddingComputer | None = None,
    ) -> None:
        self._config = config or CacheConfig()
        # OrderedDict gives O(1) move-to-end for LRU bookkeeping.
        self._data: OrderedDict[str, torch.Tensor] = OrderedDict()  # type: ignore[name-defined]
        self._rw_lock = make_rw_lock()
        self._stats = CacheStatistics()
        self._compute_fn = compute_fn
        self._device: str = self._config.device
        self._model_version: int = 0

    # -- public properties --------------------------------------------------

    @property
    def config(self) -> CacheConfig:
        return self._config

    @property
    def stats(self) -> CacheStatistics:
        return self._stats

    @property
    def compute_fn(self) -> EmbeddingComputer | None:
        return self._compute_fn

    @compute_fn.setter
    def compute_fn(self, fn: EmbeddingComputer | None) -> None:
        self._compute_fn = fn

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_version(self) -> int:
        return self._model_version

    # -- size / containment -------------------------------------------------

    def __len__(self) -> int:
        with self._rw_lock.read_lock():
            return len(self._data)

    def __contains__(self, clause: Clause) -> bool:
        key = clause_structural_hash(clause)
        with self._rw_lock.read_lock():
            return key in self._data

    # -- single-clause interface --------------------------------------------

    def get(self, clause: Clause) -> torch.Tensor | None:  # type: ignore[name-defined]
        """Look up a cached embedding. Returns ``None`` on miss."""
        key = clause_structural_hash(clause)
        with self._rw_lock.read_lock():
            tensor = self._data.get(key)
            if tensor is not None:
                # Touch for LRU — move_to_end is O(1) in OrderedDict.
                # We need a write lock for the mutation.
                pass
            else:
                self._stats.record_miss()
                return None

        # Promote to write lock for the LRU touch.
        with self._rw_lock.write_lock():
            if key in self._data:
                self._data.move_to_end(key)
        self._stats.record_hit()
        return tensor

    def put(self, clause: Clause, embedding: torch.Tensor) -> None:  # type: ignore[name-defined]
        """Insert or update an embedding for *clause*."""
        key = clause_structural_hash(clause)
        with self._rw_lock.write_lock():
            if key in self._data:
                self._data.move_to_end(key)
                self._data[key] = embedding
            else:
                self._data[key] = embedding
                self._data.move_to_end(key)
                self._maybe_evict()

    # -- batch interface (primary hot path) ---------------------------------

    def get_or_compute_batch(
        self,
        clauses: Sequence[Clause],
    ) -> torch.Tensor:  # type: ignore[name-defined]
        """Return embeddings for every clause, computing misses in batch.

        This is the main entry point used by clause selection and inference
        guidance.  It:

        1. Partitions *clauses* into hits and misses (read lock).
        2. Computes missing embeddings via ``compute_fn`` (no lock held –
           the GNN forward pass can run concurrently with other readers).
        3. Inserts newly computed embeddings (write lock).
        4. Assembles the result tensor in input order.

        Returns a tensor of shape ``(len(clauses), embedding_dim)`` on
        ``self.device``.

        Raises
        ------
        RuntimeError
            If ``compute_fn`` is not set and there are cache misses.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for EmbeddingCache")

        n = len(clauses)
        self._stats.record_batch(n)

        # --- Phase 1: partition into hits / misses -------------------------
        keys: list[str] = [clause_structural_hash(c) for c in clauses]

        hit_indices: list[int] = []
        miss_indices: list[int] = []
        hit_tensors: list[torch.Tensor] = []

        with self._rw_lock.read_lock():
            for i, key in enumerate(keys):
                cached = self._data.get(key)
                if cached is not None:
                    hit_indices.append(i)
                    hit_tensors.append(cached)
                else:
                    miss_indices.append(i)

        # Record stats outside lock.
        for _ in hit_indices:
            self._stats.record_hit()
        for _ in miss_indices:
            self._stats.record_miss()

        # --- Phase 2: compute misses (lock-free) --------------------------
        miss_tensors: list[torch.Tensor] = []
        if miss_indices:
            if self._compute_fn is None:
                raise RuntimeError(
                    "EmbeddingCache has no compute_fn set but encountered "
                    f"{len(miss_indices)} cache misses"
                )
            miss_clauses = [clauses[i] for i in miss_indices]
            batch_result = self._compute_fn.compute_embeddings(miss_clauses)
            miss_tensors = list(batch_result.unbind(0))

        # --- Phase 3: insert misses into cache (write lock) ----------------
        if miss_tensors:
            with self._rw_lock.write_lock():
                for idx, tensor in zip(miss_indices, miss_tensors):
                    key = keys[idx]
                    self._data[key] = tensor
                    self._data.move_to_end(key)
                self._maybe_evict()

        # Touch hits for LRU freshness.
        if hit_indices:
            with self._rw_lock.write_lock():
                for i in hit_indices:
                    key = keys[i]
                    if key in self._data:
                        self._data.move_to_end(key)

        # --- Phase 4: assemble output tensor in original order -------------
        result = torch.empty(
            n, self._config.embedding_dim, device=self._device
        )
        hit_iter = iter(hit_tensors)
        miss_iter = iter(miss_tensors)
        for i in range(n):
            if i in _fast_set(hit_indices):
                result[i] = next(hit_iter)
            else:
                result[i] = next(miss_iter)

        return result

    # -- bulk preload -------------------------------------------------------

    def preload(self, clauses: Sequence[Clause]) -> int:
        """Warm the cache for *clauses* that already have embeddings.

        Returns the number of clauses that were already cached (i.e. that
        did *not* need computation).  Misses are computed if ``compute_fn``
        is set, otherwise they are silently skipped.
        """
        if not clauses:
            return 0

        keys = [clause_structural_hash(c) for c in clauses]
        already_cached = 0
        to_compute: list[int] = []

        with self._rw_lock.read_lock():
            for i, key in enumerate(keys):
                if key in self._data:
                    already_cached += 1
                else:
                    to_compute.append(i)

        if to_compute and self._compute_fn is not None:
            miss_clauses = [clauses[i] for i in to_compute]
            batch_result = self._compute_fn.compute_embeddings(miss_clauses)
            miss_tensors = list(batch_result.unbind(0))

            with self._rw_lock.write_lock():
                for idx, tensor in zip(to_compute, miss_tensors):
                    key = keys[idx]
                    self._data[key] = tensor
                    self._data.move_to_end(key)
                self._maybe_evict()

        return already_cached

    # -- cache management ---------------------------------------------------

    def invalidate_all(self) -> int:
        """Drop every cached embedding.  Returns the number of evicted entries."""
        with self._rw_lock.write_lock():
            count = len(self._data)
            self._data.clear()
        self._stats.record_evictions(count)
        return count

    def invalidate_clause(self, clause: Clause) -> bool:
        """Remove a single clause's embedding.  Returns whether it existed."""
        key = clause_structural_hash(clause)
        with self._rw_lock.write_lock():
            if key in self._data:
                del self._data[key]
                self._stats.record_evictions(1)
                return True
            return False

    def on_model_update(self) -> None:
        """Called when the underlying GNN model is updated.

        Increments the model version and invalidates the entire cache,
        since stale embeddings from the old model are no longer valid.
        """
        self._model_version += 1
        self.invalidate_all()

    def check_memory_pressure(self) -> bool:
        """Return ``True`` if GPU memory usage exceeds the configured limit.

        If pressure is detected, evict a batch of cold entries.
        """
        if not _TORCH_AVAILABLE or self._device == "cpu":
            return False

        try:
            if not torch.cuda.is_available():
                return False
            mem_allocated = torch.cuda.memory_allocated()
            mem_total = torch.cuda.get_device_properties(0).total_mem
            utilisation = mem_allocated / mem_total if mem_total > 0 else 0.0
        except Exception:
            return False

        if utilisation > self._config.gpu_memory_fraction:
            self._stats.record_memory_pressure()
            self._evict_batch(self._config.eviction_batch_size)
            return True
        return False

    # -- persistence --------------------------------------------------------

    def save(self, path: str | None = None) -> None:
        """Persist the cache to disk."""
        if not _TORCH_AVAILABLE:
            return
        target = path or self._config.persist_path
        if target is None:
            return

        with self._rw_lock.read_lock():
            state = {
                "model_version": self._model_version,
                "embedding_dim": self._config.embedding_dim,
                "entries": {k: v.cpu() for k, v in self._data.items()},
            }
        torch.save(state, target)

    def load(self, path: str | None = None) -> int:
        """Restore the cache from disk.  Returns the number of entries loaded.

        Entries from an incompatible model version or embedding dimension
        are silently discarded.
        """
        if not _TORCH_AVAILABLE:
            return 0
        target = path or self._config.persist_path
        if target is None:
            return 0

        try:
            state = torch.load(target, map_location=self._device, weights_only=True)
        except Exception:
            return 0

        if state.get("embedding_dim") != self._config.embedding_dim:
            return 0
        if state.get("model_version", -1) != self._model_version:
            return 0

        entries = state.get("entries", {})
        with self._rw_lock.write_lock():
            for key, tensor in entries.items():
                if len(self._data) >= self._config.max_entries:
                    break
                self._data[key] = tensor.to(self._device)
                self._data.move_to_end(key)

        return len(entries)

    # -- internal helpers ---------------------------------------------------

    def _maybe_evict(self) -> None:
        """Evict LRU entries if over capacity.  Caller must hold write lock."""
        overflow = len(self._data) - self._config.max_entries
        if overflow > 0:
            self._evict_n(overflow)

    def _evict_batch(self, count: int) -> None:
        """Evict *count* LRU entries under write lock."""
        with self._rw_lock.write_lock():
            self._evict_n(count)

    def _evict_n(self, count: int) -> None:
        """Pop the *count* oldest entries.  Caller must hold write lock."""
        actual = min(count, len(self._data))
        for _ in range(actual):
            self._data.popitem(last=False)  # pop oldest
        self._stats.record_evictions(actual)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fast_set(indices: list[int]) -> frozenset[int]:
    """Convert a list to a frozenset for O(1) membership checks."""
    return frozenset(indices)
