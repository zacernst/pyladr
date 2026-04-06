"""Free-threading design guide and thread-safe primitives for pyladr.

Python 3.14 free-threading (PEP 703) removes the GIL, enabling true parallel
execution. This module provides thread-safe building blocks and documents
the threading strategy for the theorem prover.

== Threading Strategy ==

The given-clause algorithm has natural parallelism opportunities:

1. INFERENCE GENERATION (embarrassingly parallel):
   - Given a selected clause, generate inferences (resolution, paramodulation)
     against all clauses in the usable set
   - Each inference is independent: different target clauses produce
     independent resolvents/paramodulants
   - Can split the usable set across N workers

2. CLAUSE PROCESSING (pipeline parallel):
   - Forward subsumption, demodulation, weight evaluation can pipeline
   - Each new clause goes through: demod -> subsume_check -> weight -> keep/discard
   - Multiple clauses can be in different pipeline stages

3. BACK OPERATIONS (batch parallel):
   - Back subsumption and back demodulation after adding a new clause
   - Can check multiple existing clauses in parallel against the new one

== What Must Be Sequential ==

- Given clause selection (inherently serial - picks best from SOS)
- Index updates (must be atomic - insert/delete into discrimination trees)
- Proof reconstruction (follows a single derivation chain)

== Thread-Safe Data Structure Requirements ==

- Clause lists (SOS, Usable): concurrent read, serialized write
- Discrimination trees: readers-writer lock (many readers, exclusive writer)
- Statistics counters: atomic increments
- Clause ID generation: atomic counter

== Performance Notes ==

- Free-threading overhead is ~5-10% for single-threaded code
- Target: 2-4x speedup with 4 threads on problems with >100 given clauses
- Small problems (< 50 given clauses) should use single-threaded path
- Use thread pools, not raw threads - amortize thread creation cost
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")

# Detect free-threading availability
FREE_THREADING_AVAILABLE = not getattr(sys, "_is_gil_enabled", lambda: True)()


class AtomicCounter:
    """Thread-safe counter using a lock.

    On free-threaded Python, this provides correct atomic increments.
    On GIL Python, the lock is essentially free.
    """

    __slots__ = ("_value", "_lock")

    def __init__(self, initial: int = 0) -> None:
        self._value = initial
        self._lock = threading.Lock()

    def increment(self, delta: int = 1) -> int:
        """Atomically increment and return the new value."""
        with self._lock:
            self._value += delta
            return self._value

    def get(self) -> int:
        """Read the current value."""
        with self._lock:
            return self._value

    def reset(self, value: int = 0) -> None:
        """Reset to a specific value."""
        with self._lock:
            self._value = value


class ReadWriteLock:
    """Readers-writer lock for protecting shared data structures.

    Allows multiple concurrent readers OR a single exclusive writer.
    Writers have priority to prevent starvation.

    Use this for discrimination trees and clause indices that are
    read frequently during inference but written to less often.
    """

    __slots__ = ("_readers", "_writers_waiting", "_writer_active", "_lock", "_can_read", "_can_write")

    def __init__(self) -> None:
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False
        self._lock = threading.Lock()
        self._can_read = threading.Condition(self._lock)
        self._can_write = threading.Condition(self._lock)

    @contextmanager
    def read_lock(self) -> Iterator[None]:
        """Acquire a read lock. Multiple readers can hold this simultaneously."""
        with self._lock:
            while self._writer_active or self._writers_waiting > 0:
                self._can_read.wait()
            self._readers += 1

        try:
            yield
        finally:
            with self._lock:
                self._readers -= 1
                if self._readers == 0:
                    self._can_write.notify()

    @contextmanager
    def write_lock(self) -> Iterator[None]:
        """Acquire an exclusive write lock."""
        with self._lock:
            self._writers_waiting += 1
            while self._writer_active or self._readers > 0:
                self._can_write.wait()
            self._writers_waiting -= 1
            self._writer_active = True

        try:
            yield
        finally:
            with self._lock:
                self._writer_active = False
                self._can_read.notify_all()
                self._can_write.notify()


class ThreadSafeList(list[T]):
    """Thread-safe list wrapper with fine-grained locking.

    For clause lists (SOS, Usable) that need concurrent access.
    Iteration is not thread-safe by design - take a snapshot first.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._lock = threading.Lock()

    def append(self, item: T) -> None:  # type: ignore[override]
        with self._lock:
            super().append(item)

    def remove(self, item: T) -> None:  # type: ignore[override]
        with self._lock:
            super().remove(item)

    def snapshot(self) -> list[T]:
        """Return a copy for safe iteration."""
        with self._lock:
            return list(self)


def parallel_map(
    func: Callable[[T], object],
    items: list[T],
    *,
    max_workers: int | None = None,
) -> list[object]:
    """Map a function over items in parallel using threads.

    Falls back to sequential execution for small inputs or when
    free-threading is not available.
    """
    # Don't bother with threads for small inputs
    if len(items) < 10 or not FREE_THREADING_AVAILABLE:
        return [func(item) for item in items]

    from concurrent.futures import ThreadPoolExecutor

    if max_workers is None:
        import os
        max_workers = min(os.cpu_count() or 4, len(items))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))
