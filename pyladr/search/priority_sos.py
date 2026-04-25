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
from collections.abc import Callable, Iterator

from pyladr.core.clause import Clause


def _forte_novelty_score(embedding: list[float]) -> float:
    """FORTE selection score: negative L1-norm (highest diversity selected first).

    For L2-normalized embeddings, L1-norm ranges from 1 (single active feature)
    to sqrt(dim) (uniform spread). Higher L1 = more diverse features.
    Negated because heapq is a min-heap.

    Uses map(abs, ...) instead of generator for faster C-level iteration.
    """
    return -sum(map(abs, embedding))


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
        "name", "_fifo", "_heap", "_entropy_heap", "_penalty_heap",
        "_forte_heap", "_forte_embeddings_ref",
        "_proof_guided_heap", "_proof_guided_scorer",
        "_active", "_by_id", "_entropy_initialized", "_penalty_initialized",
        "_forte_initialized", "_proof_guided_initialized",
    )

    def __init__(
        self,
        name: str,
        forte_embeddings: dict[int, list[float]] | None = None,
        proof_guided_scorer: Callable[[int], float] | None = None,
    ) -> None:
        self.name = name
        self._fifo: deque[Clause] = deque()
        self._heap: list[tuple[float, int, Clause]] = []
        # Entropy and penalty heaps are lazily initialized on first use.
        # Default selection (weight+age) never uses them, so avoid the
        # O(nodes) computation per clause when not needed.
        self._entropy_heap: list[tuple[float, int, Clause]] = []
        self._penalty_heap: list[tuple[float, int, Clause]] = []
        self._forte_heap: list[tuple[float, int, Clause]] = []
        self._forte_embeddings_ref = forte_embeddings
        self._proof_guided_heap: list[tuple[float, int, Clause]] = []
        self._proof_guided_scorer = proof_guided_scorer
        self._active: set[int] = set()
        self._by_id: dict[int, Clause] = {}  # for iteration
        self._entropy_initialized: bool = False
        self._penalty_initialized: bool = False
        self._forte_initialized: bool = False
        self._proof_guided_initialized: bool = False

    # ── ML-order support query ────────────────────────────────────────

    #: Orders that require heap-backed extraction only PrioritySOS provides.
    #: Values correspond to SelectionOrder.FORTE (5), PROOF_GUIDED (6),
    #: TREE2VEC (7), TREE2VEC_MAXIMIN (8).  Raw ints avoid circular import
    #: with selection.py which imports PrioritySOS at module level.
    _ML_ORDERS: frozenset[int] = frozenset({5, 6, 7, 8})

    @classmethod
    def supports_order(cls, order: int) -> bool:
        """Return True if *order* requires PrioritySOS (heap-backed extraction)."""
        return order in cls._ML_ORDERS

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
        forte_score: float | None = None,
    ) -> None:
        """Add clause to FIFO, heap, and active set. O(log n).

        Args:
            c: Clause to add.
            penalty_override: If provided, use this value for the penalty heap
                instead of computing _clause_generality_penalty(). Used by
                penalty propagation to inject combined (own + inherited) penalty.
            forte_score: If provided, push to the FORTE heap with this score.
        """
        self._fifo.append(c)
        heapq.heappush(self._heap, (c.weight, c.id, c))
        # Only compute and push to entropy/penalty heaps if they've been
        # activated (by a pop_highest_entropy/pop_lowest_penalty call).
        # Default weight+age selection never activates them.
        if self._entropy_initialized:
            from pyladr.search.selection import _clause_entropy
            heapq.heappush(self._entropy_heap, (-_clause_entropy(c), c.id, c))
        if self._penalty_initialized or penalty_override is not None:
            from pyladr.search.selection import _clause_generality_penalty
            if not self._penalty_initialized:
                self._init_penalty_heap()
            penalty = penalty_override if penalty_override is not None else _clause_generality_penalty(c)
            heapq.heappush(self._penalty_heap, (penalty, c.id, c))
        if forte_score is not None:
            if not self._forte_initialized:
                self._init_forte_heap()
            heapq.heappush(self._forte_heap, (forte_score, c.id, c))
        elif self._forte_initialized:
            # Already initialized but no explicit score — compute from embeddings
            ref = self._forte_embeddings_ref
            if ref is not None and c.id in ref:
                heapq.heappush(self._forte_heap, (_forte_novelty_score(ref[c.id]), c.id, c))
        if self._proof_guided_initialized and self._proof_guided_scorer is not None:
            heapq.heappush(
                self._proof_guided_heap,
                (-self._proof_guided_scorer(c.id), c.id, c),
            )
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

    def pop_highest_entropy(self) -> Clause | None:
        """Remove and return the highest-entropy active clause. O(log n) amortized.

        Lazily initializes the entropy heap on first call.
        """
        if not self._entropy_initialized:
            self._init_entropy_heap()
        while self._entropy_heap:
            neg_entropy, cid, clause = self._entropy_heap[0]
            if cid in self._active:
                heapq.heappop(self._entropy_heap)
                self._active.discard(cid)
                self._by_id.pop(cid, None)
                return clause
            heapq.heappop(self._entropy_heap)  # skip stale
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

    def pop_best_forte(self) -> Clause | None:
        """Remove and return the best FORTE-scored active clause. O(log n) amortized.

        Lazily initializes the FORTE heap on first call using the external
        embeddings reference. Score = negative L1-norm (highest diversity first).
        """
        if not self._forte_initialized:
            self._init_forte_heap()
        while self._forte_heap:
            score, cid, clause = self._forte_heap[0]
            if cid in self._active:
                heapq.heappop(self._forte_heap)
                self._active.discard(cid)
                self._by_id.pop(cid, None)
                return clause
            heapq.heappop(self._forte_heap)  # skip stale
        return None

    def pop_best_proof_guided(self) -> Clause | None:
        """Remove and return the best proof-guided scored active clause. O(log n) amortized.

        Lazily initializes the proof-guided heap on first call using the
        scorer function. Score = negated blended exploitation/exploration
        (highest blended score is selected first via min-heap negation).
        Falls back to FORTE selection if no scorer is set.
        """
        if self._proof_guided_scorer is None:
            return self.pop_best_forte()
        if not self._proof_guided_initialized:
            self._init_proof_guided_heap()
        while self._proof_guided_heap:
            score, cid, clause = self._proof_guided_heap[0]
            if cid in self._active:
                heapq.heappop(self._proof_guided_heap)
                self._active.discard(cid)
                self._by_id.pop(cid, None)
                return clause
            heapq.heappop(self._proof_guided_heap)  # skip stale
        return None

    def _init_entropy_heap(self) -> None:
        """Build the entropy heap from all currently active clauses."""
        from pyladr.search.selection import _clause_entropy
        self._entropy_initialized = True
        self._entropy_heap = [
            (-_clause_entropy(c), c.id, c)
            for c in self._by_id.values()
            if c.id in self._active
        ]
        heapq.heapify(self._entropy_heap)

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

    def _init_forte_heap(self) -> None:
        """Build the FORTE heap from external embeddings reference."""
        self._forte_initialized = True
        ref = self._forte_embeddings_ref
        if ref is None:
            return
        self._forte_heap = [
            (_forte_novelty_score(ref[c.id]), c.id, c)
            for c in self._by_id.values()
            if c.id in self._active and c.id in ref
        ]
        heapq.heapify(self._forte_heap)

    def _init_proof_guided_heap(self) -> None:
        """Build the proof-guided heap from all currently active clauses."""
        self._proof_guided_initialized = True
        scorer = self._proof_guided_scorer
        if scorer is None:
            return
        self._proof_guided_heap = [
            (-scorer(c.id), c.id, c)
            for c in self._by_id.values()
            if c.id in self._active
        ]
        heapq.heapify(self._proof_guided_heap)

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
        if self._entropy_initialized:
            self._entropy_heap = [
                (e, cid, c)
                for e, cid, c in self._entropy_heap
                if cid in self._active
            ]
            heapq.heapify(self._entropy_heap)
        if self._penalty_initialized:
            self._penalty_heap = [
                (p, cid, c)
                for p, cid, c in self._penalty_heap
                if cid in self._active
            ]
            heapq.heapify(self._penalty_heap)
        if self._forte_initialized:
            self._forte_heap = [
                (s, cid, c)
                for s, cid, c in self._forte_heap
                if cid in self._active
            ]
            heapq.heapify(self._forte_heap)
        if self._proof_guided_initialized:
            self._proof_guided_heap = [
                (s, cid, c)
                for s, cid, c in self._proof_guided_heap
                if cid in self._active
            ]
            heapq.heapify(self._proof_guided_heap)
        # Also trim FIFO
        self._fifo = deque(c for c in self._fifo if c.id in self._active)

    def __iter__(self) -> Iterator[Clause]:
        """Iterate over active clauses in insertion order."""
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._active)

    def __repr__(self) -> str:
        return f"PrioritySOS({self.name!r}, len={self.length})"
