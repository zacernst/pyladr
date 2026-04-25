"""Background worker for online Tree2Vec skip-gram updates.

Runs model.update_online() on a daemon thread so the proof search loop
is not stalled while embeddings are being refined.  The main thread
submits clause batches via submit(); a completion callback signals when
weights + cache version have been updated.

Thread-safety contract
----------------------
SkipGramTrainer._update_lock must be held by update_online() and by
get_embedding()'s critical section.  BackgroundT2VUpdater respects this
by calling provider._tree2vec.update_online() directly (which acquires
the lock internally).

Queue semantics
---------------
queue.Queue(maxsize=1): if a batch is already queued (previous update
still in flight), new submissions are silently dropped.  This prevents
stale-embedding backlog — the next interval will submit a fresher batch.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from pyladr.core.clause import Clause
    from pyladr.ml.tree2vec.provider import Tree2VecEmbeddingProvider

logger = logging.getLogger(__name__)


class BackgroundT2VUpdater:
    """Runs Tree2Vec update_online() on a daemon thread.

    Args:
        provider: The Tree2VecEmbeddingProvider whose _tree2vec model is
            updated and whose bump_model_version() is called after each update.
        learning_rate: Learning rate passed to update_online().
        max_updates: Stop accepting submissions after this many updates (0 = unlimited).
        completion_callback: Called from the background thread immediately after
            each update completes.  Signature: (update_count: int, stats: dict).
            Must be thread-safe (post to a queue, set an event, etc.).
    """

    def __init__(
        self,
        provider: "Tree2VecEmbeddingProvider",
        learning_rate: float = 0.005,
        max_updates: int = 0,
        completion_callback: "Callable[[int, dict], None] | None" = None,
    ) -> None:
        self._provider = provider
        self._lr = learning_rate
        self._max_updates = max_updates
        self._completion_callback = completion_callback

        self._queue: queue.Queue[list["Clause"] | None] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()

        self._stats_lock = threading.Lock()
        self._update_count: int = 0
        self._last_stats: dict | None = None

        self._thread = threading.Thread(
            target=self._worker,
            name="t2v-bg-updater",
            daemon=True,
        )
        self._thread.start()

    # ── Public API ────────────────────────────────────────────────────

    def submit(self, batch: "list[Clause]") -> bool:
        """Submit a clause batch for background update.

        Non-blocking.  Returns True if the batch was queued, False if the
        queue was full (previous update still in flight) or max_updates
        reached — in either case the batch is silently dropped.
        """
        if self._stop_event.is_set():
            return False
        with self._stats_lock:
            if self._max_updates > 0 and self._update_count >= self._max_updates:
                return False
        try:
            self._queue.put_nowait(batch)
            return True
        except queue.Full:
            logger.debug("BackgroundT2VUpdater: queue full, batch dropped")
            return False

    def shutdown(self, drain: bool = True, timeout: float = 5.0) -> None:
        """Stop the background thread.

        If drain=True (default), waits for any in-flight batch to complete
        before returning.  Use drain=False for immediate stop (last update
        may be lost).
        """
        self._stop_event.set()
        # Wake the worker if it is blocked on queue.get()
        try:
            self._queue.put_nowait(None)  # sentinel
        except queue.Full:
            pass
        if drain:
            self._thread.join(timeout=timeout)

    @property
    def update_count(self) -> int:
        with self._stats_lock:
            return self._update_count

    @property
    def last_stats(self) -> "dict | None":
        with self._stats_lock:
            return dict(self._last_stats) if self._last_stats is not None else None

    @property
    def pending_count(self) -> int:
        """Number of batches waiting in the queue (0 or 1)."""
        return self._queue.qsize()

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    # ── Worker ────────────────────────────────────────────────────────

    def _worker(self) -> None:
        """Background thread: drain queue and run updates until stopped."""
        while not self._stop_event.is_set():
            try:
                batch = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if batch is None:  # shutdown sentinel
                self._queue.task_done()
                break

            try:
                self._run_update(batch)
            except Exception:
                logger.warning("BackgroundT2VUpdater: update failed", exc_info=True)
            finally:
                self._queue.task_done()

    def _run_update(self, batch: "list[Clause]") -> None:
        provider = self._provider
        # update_online acquires SkipGramTrainer._update_lock internally
        stats = provider._tree2vec.update_online(batch, learning_rate=self._lr)  # type: ignore[union-attr]
        # bump_model_version is already _cache_lock-protected in provider.py
        new_version = provider.bump_model_version()

        with self._stats_lock:
            self._update_count += 1
            update_num = self._update_count
            self._last_stats = dict(stats)

        logger.info(
            "BackgroundT2VUpdater: update #%d done — pairs=%d, loss=%.4f, model_v=%d",
            update_num,
            stats.get("pairs_trained", 0),
            stats.get("loss", 0.0),
            new_version,
        )

        if self._completion_callback is not None:
            try:
                self._completion_callback(update_num, stats)
            except Exception:
                logger.warning(
                    "BackgroundT2VUpdater: completion callback failed", exc_info=True
                )
