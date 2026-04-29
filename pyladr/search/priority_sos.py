"""Priority-queue backed SOS for O(log n) weight-based clause selection.

Replaces the O(n) linear scan in GivenSelection._select_by_order() with
a heap-based priority queue, and replaces O(n) list.remove() with O(1)
set-based membership tracking.

The C implementation uses AVL trees; we use Python's heapq module with
lazy deletion for equivalent O(log n) insertion and extraction.

Design:
    - FIFO order preserved via a deque for age-based selection
    - Min-heap for weight-based selection (keyed by weight, then ID)
    - Set of active clause IDs for O(1) membership/removal
    - Lazy deletion in heap: removed IDs are skipped on extraction

Usage:
    from pyladr.search.priority_sos import PrioritySOS

    sos = PrioritySOS("sos")
    sos.append(clause)

    # O(log n) weight selection:
    lightest = sos.pop_lightest()

    # O(1) age selection:
    oldest = sos.pop_first()

    # O(1) arbitrary removal:
    sos.remove(some_clause)
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Iterator

from pyladr.core.clause import Clause


class PrioritySOS:
    """SOS with O(log n) weight selection and O(1) removal.

    Maintains three synchronized data structures:
    1. _fifo: deque for FIFO age-based selection (O(1) popleft)
    2. _heap: min-heap for weight-based selection (O(log n) pop)
    3. _active: set of clause IDs for O(1) membership and removal

    Lazy deletion: when a clause is removed, its ID is dropped from
    _active. Heap extraction skips IDs not in _active. FIFO popleft
    skips IDs not in _active.

    Implements the same interface as ClauseList for drop-in use.
    """

    __slots__ = (
        "name", "_fifo", "_heap", "_penalty_heap",
        "_active", "_by_id", "_penalty_initialized",
    )

    def __init__(self, name: str) -> None:
        self.name = name
        self._fifo: deque[Clause] = deque()
        self._heap: list[tuple[float, int, Clause]] = []
        # Penalty heap is lazily initialized on first use. Default
        # selection (weight+age) never uses it, so avoid the O(nodes)
        # computation per clause when not needed.
        self._penalty_heap: list[tuple[float, int, Clause]] = []
        self._active: set[int] = set()
        self._by_id: dict[int, Clause] = {}  # for iteration
        self._penalty_initialized: bool = False

    @property
    def length(self) -> int:
        return len(self._active)

    @property
    def is_empty(self) -> bool:
        return len(self._active) == 0

    @property
    def first(self) -> Clause | None:
        """Return the oldest active clause without removing it. O(1) amortized."""
        self._skip_stale_fifo()
        return self._fifo[0] if self._fifo else None

    def append(
        self,
        c: Clause,
        penalty_override: float | None = None,
    ) -> None:
        """Add clause to FIFO, heap, and active set. O(log n).

        Args:
            c: Clause to add.
            penalty_override: If provided, use this value for the penalty heap
                instead of computing _clause_generality_penalty(). Used by
                penalty propagation to inject combined (own + inherited) penalty.
        """
        self._fifo.append(c)
        heapq.heappush(self._heap, (c.weight, c.id, c))
        # Only compute and push to the penalty heap if it's been
        # activated (by a pop_lowest_penalty call).
        # Default weight+age selection never activates it.
        if self._penalty_initialized or penalty_override is not None:
            from pyladr.search.selection import _clause_generality_penalty
            if not self._penalty_initialized:
                self._init_penalty_heap()
            penalty = penalty_override if penalty_override is not None else _clause_generality_penalty(c)
            heapq.heappush(self._penalty_heap, (penalty, c.id, c))
        self._active.add(c.id)
        self._by_id[c.id] = c

    def remove(self, c: Clause) -> bool:
        """Remove clause by marking inactive. O(1)."""
        if c.id not in self._active:
            return False
        self._active.discard(c.id)
        self._by_id.pop(c.id, None)
        return True

    def contains(self, c: Clause) -> bool:
        """Check if clause is active. O(1)."""
        return c.id in self._active

    def pop_first(self) -> Clause | None:
        """Remove and return the oldest active clause. O(1) amortized."""
        self._skip_stale_fifo()
        if not self._fifo:
            return None
        c = self._fifo.popleft()
        self._active.discard(c.id)
        self._by_id.pop(c.id, None)
        return c

    def pop_lightest(self) -> Clause | None:
        """Remove and return the lightest active clause. O(log n) amortized."""
        while self._heap:
            weight, cid, clause = self._heap[0]
            if cid in self._active:
                heapq.heappop(self._heap)
                self._active.discard(cid)
                self._by_id.pop(cid, None)
                return clause
            heapq.heappop(self._heap)  # skip stale
        return None

    def pop_lowest_penalty(self) -> Clause | None:
        """Remove and return the lowest-penalty active clause. O(log n) amortized.

        Lazily initializes the penalty heap on first call.
        """
        if not self._penalty_initialized:
            self._init_penalty_heap()
        while self._penalty_heap:
            penalty, cid, clause = self._penalty_heap[0]
            if cid in self._active:
                heapq.heappop(self._penalty_heap)
                self._active.discard(cid)
                self._by_id.pop(cid, None)
                return clause
            heapq.heappop(self._penalty_heap)  # skip stale
        return None

    def _init_penalty_heap(self) -> None:
        """Build the penalty heap from all currently active clauses."""
        from pyladr.search.selection import _clause_generality_penalty
        self._penalty_initialized = True
        self._penalty_heap = [
            (_clause_generality_penalty(c), c.id, c)
            for c in self._by_id.values()
            if c.id in self._active
        ]
        heapq.heapify(self._penalty_heap)

    def peek_lightest(self) -> Clause | None:
        """Return the lightest active clause without removing it."""
        while self._heap:
            weight, cid, clause = self._heap[0]
            if cid in self._active:
                return clause
            heapq.heappop(self._heap)
        return None

    def _skip_stale_fifo(self) -> None:
        """Skip removed clauses at the front of the FIFO."""
        while self._fifo and self._fifo[0].id not in self._active:
            self._fifo.popleft()

    def compact(self) -> None:
        """Rebuild heap to reclaim memory from lazy deletions.

        Call when heap has accumulated many stale entries.
        Only compacts heaps that have been initialized.
        """
        self._heap = [
            (w, cid, c)
            for w, cid, c in self._heap
            if cid in self._active
        ]
        heapq.heapify(self._heap)
        if self._penalty_initialized:
            self._penalty_heap = [
                (p, cid, c)
                for p, cid, c in self._penalty_heap
                if cid in self._active
            ]
            heapq.heapify(self._penalty_heap)
        # Also trim FIFO
        self._fifo = deque(c for c in self._fifo if c.id in self._active)

    def __iter__(self) -> Iterator[Clause]:
        """Iterate over active clauses in insertion order."""
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._active)

    def __repr__(self) -> str:
        return f"PrioritySOS({self.name!r}, len={self.length})"
