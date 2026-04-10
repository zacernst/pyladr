"""Memory usage tracking and optimization for the ML pipeline.

Provides bounded-memory monitoring specifically for online learning components:
experience buffers, model version snapshots, embedding caches, and gradient
computation.  Designed for arbitrarily long searches where memory leaks in the
ML subsystem would otherwise be the dominant growth term.

Usage:
    tracker = MLMemoryTracker()
    tracker.snapshot(learning_manager, embedding_cache)
    print(tracker.report())
    if tracker.memory_growth_rate > threshold:
        tracker.recommend_actions()
"""

from __future__ import annotations

import gc
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyladr.ml.online_learning import OnlineLearningManager
    from pyladr.ml.embeddings.cache import EmbeddingCache

logger = logging.getLogger(__name__)


# ── Snapshot data ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class MLMemorySnapshot:
    """Point-in-time memory snapshot of ML pipeline components."""

    timestamp: float = 0.0
    iteration: int = 0

    # Experience buffer
    buffer_size: int = 0
    buffer_capacity: int = 0
    buffer_productive: int = 0
    buffer_unproductive: int = 0
    buffer_total_added: int = 0

    # Model versions
    num_versions: int = 0
    num_versions_with_state: int = 0
    version_memory_bytes: int = 0

    # Embedding cache
    cache_entries: int = 0
    cache_max_entries: int = 0
    cache_hit_rate: float = 0.0
    cache_memory_bytes: int = 0

    # Process level
    process_rss_bytes: int = 0

    # Gradient state
    optimizer_state_bytes: int = 0

    @property
    def total_ml_bytes(self) -> int:
        """Estimated total ML pipeline memory."""
        return (
            self.version_memory_bytes
            + self.cache_memory_bytes
            + self.optimizer_state_bytes
        )

    @property
    def total_ml_mb(self) -> float:
        return self.total_ml_bytes / (1024 * 1024)

    @property
    def buffer_utilization(self) -> float:
        if self.buffer_capacity == 0:
            return 0.0
        return self.buffer_size / self.buffer_capacity


@dataclass(slots=True)
class MemoryBudget:
    """Memory budget for the ML pipeline with soft and hard limits.

    Attributes:
        soft_limit_mb: Trigger cleanup when ML memory exceeds this.
        hard_limit_mb: Force aggressive cleanup / disable features above this.
        version_limit: Maximum model version snapshots with state_dicts.
        cache_limit: Maximum embedding cache entries.
    """

    soft_limit_mb: float = 256.0
    hard_limit_mb: float = 512.0
    version_limit: int = 10
    cache_limit: int = 100_000


# ── Size estimation helpers ────────────────────────────────────────────────


def _estimate_tensor_bytes(tensor: Any) -> int:
    """Estimate the memory usage of a PyTorch tensor."""
    try:
        return tensor.nelement() * tensor.element_size()
    except (AttributeError, RuntimeError):
        return 0


def _estimate_state_dict_bytes(state_dict: dict) -> int:
    """Estimate memory used by a model state_dict."""
    total = 0
    for v in state_dict.values():
        total += _estimate_tensor_bytes(v)
    return total


def _estimate_optimizer_bytes(optimizer: Any) -> int:
    """Estimate memory used by optimizer state (momentum buffers, etc.)."""
    if optimizer is None:
        return 0
    total = 0
    try:
        for group in optimizer.state.values():
            if isinstance(group, dict):
                for v in group.values():
                    total += _estimate_tensor_bytes(v)
    except (AttributeError, RuntimeError):
        pass
    return total


# ── Memory tracker ─────────────────────────────────────────────────────────


class MLMemoryTracker:
    """Tracks and bounds ML pipeline memory during long searches.

    Takes periodic snapshots of all ML components and computes growth
    rates.  Can recommend or trigger cleanup actions when memory exceeds
    configured budgets.

    The tracker itself uses bounded memory: snapshots are stored in a
    fixed-size deque.
    """

    __slots__ = (
        "_snapshots", "_max_snapshots", "_budget", "_start_time",
        "_cleanup_count", "_last_cleanup_time",
    )

    def __init__(
        self,
        budget: MemoryBudget | None = None,
        max_snapshots: int = 200,
    ) -> None:
        self._budget = budget or MemoryBudget()
        self._max_snapshots: int = max_snapshots
        self._snapshots: deque[MLMemorySnapshot] = deque(maxlen=max_snapshots)
        self._start_time = time.monotonic()
        self._cleanup_count = 0
        self._last_cleanup_time = 0.0

    @property
    def budget(self) -> MemoryBudget:
        return self._budget

    @property
    def snapshots(self) -> list[MLMemorySnapshot]:
        return list(self._snapshots)

    @property
    def cleanup_count(self) -> int:
        return self._cleanup_count

    def snapshot(
        self,
        manager: OnlineLearningManager | None = None,
        cache: EmbeddingCache | None = None,
        iteration: int = 0,
    ) -> MLMemorySnapshot:
        """Take a memory snapshot of ML pipeline components.

        Safe to call with None arguments — missing components are skipped.
        """
        snap = MLMemorySnapshot(
            timestamp=time.monotonic() - self._start_time,
            iteration=iteration,
        )

        if manager is not None:
            self._snapshot_manager(snap, manager)

        if cache is not None:
            self._snapshot_cache(snap, cache)

        # Process RSS
        snap.process_rss_bytes = _get_process_rss()

        self._snapshots.append(snap)
        return snap

    def _snapshot_manager(
        self, snap: MLMemorySnapshot, manager: OnlineLearningManager,
    ) -> None:
        """Fill snapshot fields from OnlineLearningManager."""
        # Buffer stats
        buf_snap = manager._buffer.snapshot()
        snap.buffer_size = buf_snap["size"]
        snap.buffer_capacity = buf_snap["capacity"]
        snap.buffer_productive = buf_snap["productive"]
        snap.buffer_unproductive = buf_snap["unproductive"]
        snap.buffer_total_added = buf_snap["total_added"]

        # Version stats
        snap.num_versions = len(manager._versions)
        version_bytes = 0
        versions_with_state = 0
        for v in manager._versions:
            if v.state_dict:
                versions_with_state += 1
                version_bytes += _estimate_state_dict_bytes(v.state_dict)
        snap.num_versions_with_state = versions_with_state
        snap.version_memory_bytes = version_bytes

        # Optimizer state
        snap.optimizer_state_bytes = _estimate_optimizer_bytes(manager._optimizer)

    def _snapshot_cache(
        self, snap: MLMemorySnapshot, cache: EmbeddingCache,
    ) -> None:
        """Fill snapshot fields from EmbeddingCache."""
        snap.cache_entries = len(cache)
        snap.cache_max_entries = cache.config.max_entries
        snap.cache_hit_rate = cache.stats.hit_rate
        # Estimate cache memory: entries * embedding_dim * 4 bytes (float32)
        snap.cache_memory_bytes = (
            len(cache) * cache.config.embedding_dim * 4
        )

    # ── Analysis ───────────────────────────────────────────────────────

    @property
    def memory_growth_rate_mb_per_hour(self) -> float:
        """Estimated MB/hour growth rate of ML memory."""
        if len(self._snapshots) < 2:
            return 0.0
        first = self._snapshots[0]
        last = self._snapshots[-1]
        dt = last.timestamp - first.timestamp
        if dt < 1.0:
            return 0.0
        delta_bytes = last.total_ml_bytes - first.total_ml_bytes
        return (delta_bytes / (1024 * 1024)) / (dt / 3600)

    @property
    def is_over_soft_limit(self) -> bool:
        if not self._snapshots:
            return False
        return self._snapshots[-1].total_ml_mb > self._budget.soft_limit_mb

    @property
    def is_over_hard_limit(self) -> bool:
        if not self._snapshots:
            return False
        return self._snapshots[-1].total_ml_mb > self._budget.hard_limit_mb

    def check_and_cleanup(
        self,
        manager: OnlineLearningManager | None = None,
        cache: EmbeddingCache | None = None,
    ) -> list[str]:
        """Check memory budget and perform cleanup if needed.

        Returns a list of actions taken.
        """
        if not self._snapshots:
            return []

        actions: list[str] = []
        current = self._snapshots[-1]

        # Soft limit: prune model versions aggressively
        if current.total_ml_mb > self._budget.soft_limit_mb:
            if manager is not None:
                pruned = _aggressive_version_prune(manager, self._budget.version_limit)
                if pruned > 0:
                    actions.append(f"Pruned {pruned} model version state_dicts")

        # Hard limit: also shrink embedding cache
        if current.total_ml_mb > self._budget.hard_limit_mb:
            if cache is not None:
                evicted = _shrink_cache(cache, fraction=0.25)
                if evicted > 0:
                    actions.append(f"Evicted {evicted} cache entries (hard limit)")

            # Force garbage collection
            gc.collect()
            actions.append("Forced garbage collection")

        if actions:
            self._cleanup_count += 1
            self._last_cleanup_time = time.monotonic()
            logger.info("ML memory cleanup: %s", "; ".join(actions))

        return actions

    def report(self) -> str:
        """Generate ML memory usage report."""
        if not self._snapshots:
            return "No ML memory snapshots recorded."

        lines = [
            "=" * 60,
            "ML PIPELINE MEMORY REPORT",
            "=" * 60,
        ]

        last = self._snapshots[-1]
        lines.append(f"Snapshots: {len(self._snapshots)}")
        lines.append(f"Cleanups triggered: {self._cleanup_count}")
        lines.append("")

        lines.append("Experience Buffer:")
        lines.append(
            f"  Size: {last.buffer_size}/{last.buffer_capacity} "
            f"({last.buffer_utilization:.0%})"
        )
        lines.append(
            f"  Productive/Unproductive: "
            f"{last.buffer_productive}/{last.buffer_unproductive}"
        )
        lines.append(f"  Total added: {last.buffer_total_added}")
        lines.append("")

        lines.append("Model Versions:")
        lines.append(
            f"  Total: {last.num_versions} "
            f"({last.num_versions_with_state} with state)"
        )
        lines.append(
            f"  Version memory: {last.version_memory_bytes / (1024*1024):.1f} MB"
        )
        lines.append("")

        lines.append("Embedding Cache:")
        lines.append(
            f"  Entries: {last.cache_entries}/{last.cache_max_entries}"
        )
        lines.append(f"  Hit rate: {last.cache_hit_rate:.1%}")
        lines.append(
            f"  Cache memory: {last.cache_memory_bytes / (1024*1024):.1f} MB"
        )
        lines.append("")

        lines.append("Optimizer State:")
        lines.append(
            f"  Memory: {last.optimizer_state_bytes / (1024*1024):.1f} MB"
        )
        lines.append("")

        lines.append(f"Total ML memory: {last.total_ml_mb:.1f} MB")
        lines.append(
            f"Budget: {self._budget.soft_limit_mb:.0f} MB soft / "
            f"{self._budget.hard_limit_mb:.0f} MB hard"
        )

        growth = self.memory_growth_rate_mb_per_hour
        if abs(growth) > 0.01:
            lines.append(f"Growth rate: {growth:.2f} MB/hour")

        lines.append("=" * 60)
        return "\n".join(lines)


# ── Cleanup helpers ────────────────────────────────────────────────────────


def _aggressive_version_prune(
    manager: OnlineLearningManager,
    keep_limit: int,
) -> int:
    """Aggressively prune model version state_dicts.

    In addition to clearing state_dicts (like _prune_versions), this
    also removes version objects from the list entirely when their
    state is cleared.  Keeps: v0, current, best, and the last
    `keep_limit` versions.
    """
    versions = manager._versions
    if len(versions) <= keep_limit:
        return 0

    # Build protected set
    protected_ids: set[int] = set()
    if versions:
        protected_ids.add(versions[0].version_id)  # v0
    if manager._current_version is not None:
        protected_ids.add(manager._current_version.version_id)
    # Best by productivity
    tested = [v for v in versions if v.selections_made > 0]
    if tested:
        best = max(tested, key=lambda v: v.productivity_rate)
        protected_ids.add(best.version_id)
    # Recent
    for v in versions[-keep_limit:]:
        protected_ids.add(v.version_id)

    # Clear state_dicts AND remove empty versions from list
    pruned = 0
    new_versions = []
    for v in versions:
        if v.version_id in protected_ids:
            new_versions.append(v)
        elif v.state_dict:
            v.state_dict = {}
            pruned += 1
            # Keep the version object but without state for metadata
            new_versions.append(v)
        else:
            # Already pruned — only keep if in recent window
            new_versions.append(v)

    manager._versions = new_versions
    return pruned


def _shrink_cache(cache: EmbeddingCache, fraction: float = 0.25) -> int:
    """Evict a fraction of the embedding cache entries."""
    current_size = len(cache)
    to_evict = int(current_size * fraction)
    if to_evict <= 0:
        return 0
    cache._evict_batch(to_evict)
    return to_evict


def _get_process_rss() -> int:
    """Get current process RSS in bytes."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = usage.ru_maxrss
        if sys.platform == "darwin":
            return rss
        return rss * 1024
    except (ImportError, OSError):
        return 0
