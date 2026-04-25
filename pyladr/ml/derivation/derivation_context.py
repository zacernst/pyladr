"""Derivation DAG tracker — maintains clause ancestry during search.

DerivationContext builds and queries the derivation DAG by observing each
clause's justification as it enters the search.  It provides O(1) lookups
for derivation depth and O(depth) traversal for full ancestor chains.

Thread safety:  A threading.Lock protects the mutable mapping so that
concurrent inference threads can register new clauses safely.  Reads are
also serialised; for a read-heavy workload the lock could be upgraded to
a RWLock, but the critical section is tiny (dict lookup) so contention
should be negligible.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyladr.core.clause import Clause, JustType


@dataclass(slots=True)
class DerivationInfo:
    """Cached derivation metadata for a single clause.

    Attributes:
        clause_id: The clause's unique identifier.
        depth: Number of inference steps from the nearest input/goal ancestor.
            Input and goal clauses have depth 0.
        parent_ids: Direct parent clause IDs from the primary justification.
        primary_rule: The JustType of the primary (first) justification step.
        num_simplifications: Count of secondary justification steps
            (DEMOD, UNIT_DEL, FLIP, etc.) applied after the primary inference.
        inference_chain: Ordered sequence of JustType values from root to this
            clause.  Built lazily on first access via DerivationContext.
    """

    clause_id: int
    depth: int
    parent_ids: tuple[int, ...]
    primary_rule: int  # JustType value
    num_simplifications: int
    inference_chain: tuple[int, ...] = ()  # lazily populated


class DerivationContext:
    """Tracks the derivation DAG during search.

    Register each clause as it is created / kept.  The context computes
    derivation depth incrementally (max-parent-depth + 1) and caches
    parent links for later chain reconstruction.

    Usage::

        ctx = DerivationContext()

        # During search — call for every new clause
        ctx.register(clause)

        # Query
        info = ctx.get(clause.id)
        chain = ctx.get_inference_chain(clause.id, max_length=32)
    """

    def __init__(self, max_chain_length: int = 64) -> None:
        self._lock = threading.Lock()
        self._info: dict[int, DerivationInfo] = {}
        self._max_chain_length = max_chain_length

    # ── Registration ──────────────────────────────────────────────────

    def register(self, clause: Clause) -> DerivationInfo:
        """Register a clause and compute its derivation metadata.

        Safe to call multiple times for the same clause id; the first
        registration wins.

        Returns the (possibly pre-existing) DerivationInfo.
        """
        from pyladr.core.clause import JustType

        cid = clause.id
        with self._lock:
            if cid in self._info:
                return self._info[cid]

        # Compute outside the lock (read-only access to justification)
        parent_ids: tuple[int, ...]
        primary_rule: int
        num_simp: int

        if not clause.justification:
            parent_ids = ()
            primary_rule = int(JustType.INPUT)
            num_simp = 0
        else:
            prim = clause.justification[0]
            primary_rule = int(prim.just_type)
            parent_ids = _extract_parent_ids(prim)
            num_simp = len(clause.justification) - 1

        # Compute depth = max(parent depths) + 1, or 0 for axioms
        depth = self._compute_depth(parent_ids)

        info = DerivationInfo(
            clause_id=cid,
            depth=depth,
            parent_ids=parent_ids,
            primary_rule=primary_rule,
            num_simplifications=num_simp,
        )

        with self._lock:
            # Double-check (another thread may have inserted)
            if cid not in self._info:
                self._info[cid] = info
            return self._info[cid]

    # ── Queries ───────────────────────────────────────────────────────

    def get(self, clause_id: int) -> DerivationInfo | None:
        """Look up cached derivation info.  Returns None if not registered."""
        with self._lock:
            return self._info.get(clause_id)

    def get_inference_chain(
        self, clause_id: int, max_length: int | None = None
    ) -> tuple[int, ...]:
        """Return the inference rule chain from root ancestor to *clause_id*.

        The chain is a sequence of JustType int values ordered from the
        oldest ancestor to the clause itself.  If the derivation DAG is
        deeper than *max_length*, only the most recent *max_length* steps
        are returned (closest to *clause_id*).

        Returns an empty tuple if *clause_id* is not registered.
        """
        limit = max_length or self._max_chain_length
        chain: list[int] = []
        visited: set[int] = set()

        with self._lock:
            current = clause_id
            while len(chain) < limit:
                info = self._info.get(current)
                if info is None:
                    break
                if current in visited:
                    break  # cycle guard (should not happen)
                visited.add(current)
                chain.append(info.primary_rule)

                if not info.parent_ids:
                    break
                # Follow the first parent (main derivation path)
                current = info.parent_ids[0]

        chain.reverse()  # root → clause order
        return tuple(chain)

    def get_depth(self, clause_id: int) -> int:
        """Return derivation depth, or 0 if not registered."""
        with self._lock:
            info = self._info.get(clause_id)
            return info.depth if info is not None else 0

    @property
    def size(self) -> int:
        """Number of registered clauses."""
        with self._lock:
            return len(self._info)

    def clear(self) -> None:
        """Reset all tracked derivation information."""
        with self._lock:
            self._info.clear()

    # ── Internals ─────────────────────────────────────────────────────

    def _compute_depth(self, parent_ids: tuple[int, ...]) -> int:
        """Compute depth as max(parent depths) + 1.

        Input/goal clauses (no parents) have depth 0.
        """
        if not parent_ids:
            return 0
        max_parent_depth = 0
        with self._lock:
            for pid in parent_ids:
                info = self._info.get(pid)
                if info is not None and info.depth > max_parent_depth:
                    max_parent_depth = info.depth
        return max_parent_depth + 1


def _extract_parent_ids(just: object) -> tuple[int, ...]:
    """Extract parent clause IDs from a Justification object.

    Handles all justification types: clause_ids (resolution, hyper),
    clause_id (copy, deny), and para (paramodulation).
    """
    from pyladr.core.clause import JustType

    jt = just.just_type  # type: ignore[union-attr]

    # Multi-parent inferences
    if just.clause_ids:  # type: ignore[union-attr]
        return just.clause_ids  # type: ignore[union-attr]

    # Paramodulation
    if just.para is not None:  # type: ignore[union-attr]
        return (just.para.from_id, just.para.into_id)  # type: ignore[union-attr]

    # Single-parent
    if just.clause_id:  # type: ignore[union-attr]
        return (just.clause_id,)  # type: ignore[union-attr]

    return ()
