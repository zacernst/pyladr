"""Literal index matching C lindex.c.

A literal index (Lindex) is a pair of Mindexes: one for positive literals,
one for negative literals. When a clause is inserted, each literal's atom
is indexed into the appropriate component based on sign.

This is used for subsumption and unit conflict detection where we need
to look up clauses by their literal signs.

== C Structure ==

    struct lindex {
        Mindex pos;   // index for positive literals
        Mindex neg;   // index for negative literals
    };

== Usage ==

    lindex = LiteralIndex(IndexType.DISCRIM_WILD)
    lindex.update(clause, insert=True)   # index all literals
    # For forward subsumption: retrieve generalizations from pos/neg
    # For backward subsumption: retrieve instances from pos/neg
"""

from __future__ import annotations

from pyladr.core.clause import Clause
from pyladr.indexing.discrimination_tree import IndexType, Mindex


class LiteralIndex:
    """Literal index matching C Lindex.

    Pairs a positive and negative Mindex. Each literal of a clause
    is indexed by its atom into the appropriate component based on sign.

    The data stored with each atom is (clause, literal) so retrieval
    can access both the clause and which literal matched.
    """

    __slots__ = ("pos", "neg", "_index_type", "_first_only")

    def __init__(
        self,
        index_type: IndexType = IndexType.DISCRIM_WILD,
        *,
        first_only: bool = False,
    ) -> None:
        """Initialize literal index.

        Args:
            index_type: Type of underlying Mindex.
            first_only: If True, only index the first literal of each clause
                        (used for back subsumption where we only need first
                        literal indexed). Matching C lindex_update_first().
        """
        self.pos = Mindex(index_type)
        self.neg = Mindex(index_type)
        self._index_type = index_type
        self._first_only = first_only

    def update(self, c: Clause, *, insert: bool = True) -> None:
        """Insert or remove a clause's literals from the index.

        Matching C lindex_update() / lindex_update_first().

        Args:
            c: Clause whose literals to index/deindex.
            insert: True to insert, False to delete.
        """
        lits = c.literals[:1] if self._first_only else c.literals
        for lit in lits:
            idx = self.pos if lit.sign else self.neg
            data = (c, lit)
            if insert:
                idx.insert(lit.atom, data)
            else:
                idx.delete(lit.atom, data)

    def is_empty(self) -> bool:
        """Check if both components are empty."""
        return self.pos.size == 0 and self.neg.size == 0
