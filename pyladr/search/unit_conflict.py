"""Unit conflict index for O(1) complementary unit clause detection.

Extracted from given_clause.py to reduce the god-object size.
Used by GivenClauseSearch._unit_conflict() for fast proof detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyladr.core.clause import Clause


def _term_key(t: object) -> tuple:
    """Compute a hashable structural key for a term (recursive tuple)."""
    if not t.args:  # type: ignore[union-attr]
        return (t.private_symbol,)  # type: ignore[union-attr]
    return (t.private_symbol, *((_term_key(a)) for a in t.args))  # type: ignore[union-attr]


class UnitConflictIndex:
    """O(1) lookup index for unit conflict detection.

    Stores unit clauses keyed by (sign, atom_key). When a new unit clause
    arrives, look up its complement in O(1) instead of scanning all usable.
    """

    __slots__ = ("_by_key",)

    def __init__(self) -> None:
        # Map: (sign, atom_key) -> Clause
        self._by_key: dict[tuple, Clause] = {}

    def insert(self, c: Clause) -> None:
        """Add a unit clause to the index."""
        if not c.is_unit:
            return
        lit = c.literals[0]
        key = (lit.sign, _term_key(lit.atom))
        # Keep the first (oldest) clause for each key
        if key not in self._by_key:
            self._by_key[key] = c

    def remove(self, c: Clause) -> None:
        """Remove a unit clause from the index."""
        if not c.is_unit:
            return
        lit = c.literals[0]
        key = (lit.sign, _term_key(lit.atom))
        stored = self._by_key.get(key)
        if stored is not None and stored.id == c.id:
            del self._by_key[key]

    def find_complement(self, c: Clause) -> Clause | None:
        """Find a unit clause complementary to c. O(1) lookup."""
        if not c.is_unit:
            return None
        lit = c.literals[0]
        comp_key = (not lit.sign, _term_key(lit.atom))
        return self._by_key.get(comp_key)
