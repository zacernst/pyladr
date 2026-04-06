"""Search state management matching C search-structures.h Glob.

Manages the clause lists (usable, sos, limbo, disabled), clause ID
assignment, and index structures used during the given-clause search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from pyladr.core.clause import Clause
from pyladr.indexing import DiscrimWild, IndexType, Mindex
from pyladr.search.statistics import SearchStatistics


@dataclass(slots=True)
class ClauseList:
    """An ordered list of clauses matching C struct clist.

    Named clause lists (usable, sos, limbo, disabled) hold clauses
    at different stages of the search.
    """

    name: str
    _clauses: list[Clause] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self._clauses)

    @property
    def is_empty(self) -> bool:
        return len(self._clauses) == 0

    @property
    def first(self) -> Clause | None:
        return self._clauses[0] if self._clauses else None

    def append(self, c: Clause) -> None:
        """Append a clause to the end (C clist_append)."""
        self._clauses.append(c)

    def remove(self, c: Clause) -> bool:
        """Remove a clause by identity (C clist_remove)."""
        try:
            self._clauses.remove(c)
            return True
        except ValueError:
            return False

    def contains(self, c: Clause) -> bool:
        """Check if clause is in this list (C clist_member)."""
        return c in self._clauses

    def pop_first(self) -> Clause | None:
        """Remove and return the first clause."""
        if self._clauses:
            return self._clauses.pop(0)
        return None

    def __iter__(self):
        return iter(self._clauses)

    def __len__(self) -> int:
        return len(self._clauses)

    def __repr__(self) -> str:
        return f"ClauseList({self.name!r}, len={self.length})"


@dataclass(slots=True)
class SearchState:
    """Global search state matching C Glob struct in search-structures.h.

    Holds the clause lists, indexes, and configuration for the
    given-clause search loop.
    """

    # Clause lists (C: Glob.usable, Glob.sos, Glob.limbo, Glob.disabled)
    usable: ClauseList = field(default_factory=lambda: ClauseList("usable"))
    sos: ClauseList = field(default_factory=lambda: ClauseList("sos"))
    limbo: ClauseList = field(default_factory=lambda: ClauseList("limbo"))
    disabled: ClauseList = field(default_factory=lambda: ClauseList("disabled"))
    demods: ClauseList = field(default_factory=lambda: ClauseList("demodulators"))

    # Proof tracking (C: Glob.empties)
    empties: list[Clause] = field(default_factory=list)

    # Indexes for resolution (C: Glob.clashable_idx)
    clashable_idx: Mindex = field(
        default_factory=lambda: Mindex(IndexType.DISCRIM_WILD)
    )

    # Statistics
    stats: SearchStatistics = field(default_factory=SearchStatistics)

    # Clause ID counter (C: clause_id_count)
    _next_clause_id: int = field(default=1)
    _id_lock: Lock = field(default_factory=Lock)

    # Search control
    searching: bool = False
    return_code: int = 0

    def assign_clause_id(self, c: Clause) -> int:
        """Assign the next clause ID. Matches C assign_clause_id()."""
        with self._id_lock:
            c.id = self._next_clause_id
            self._next_clause_id += 1
        return c.id

    def clause_ids_assigned(self) -> int:
        """Number of clause IDs assigned so far."""
        return self._next_clause_id - 1

    def disable_clause(self, c: Clause) -> None:
        """Move a clause to the disabled list (C disable_clause).

        Removes from whichever list it's in, adds to disabled.
        """
        self.usable.remove(c)
        self.sos.remove(c)
        self.limbo.remove(c)
        self.disabled.append(c)

    def index_clashable(self, c: Clause, insert: bool = True) -> None:
        """Insert/remove clause literals into/from the clashable index.

        Matches C index_clashable() — indexes each literal's atom
        for resolution retrieval.
        """
        for lit in c.literals:
            if insert:
                self.clashable_idx.insert(lit.atom, (c, lit))
            else:
                self.clashable_idx.delete(lit.atom, (c, lit))
