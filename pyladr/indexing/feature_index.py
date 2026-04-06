"""Feature vector indexing for fast subsumption candidate filtering.

Feature vectors provide cheap prefiltering for subsumption: if C subsumes D,
then for each feature, C's count must be <= D's count (for positive features)
or >= D's count (for negative features). This eliminates many candidates
without the expensive full subsumption check.

Features include:
- Number of positive/negative literals
- Symbol occurrence counts (per function/predicate symbol)
- Depth statistics
- Variable counts

This is an optimization layer: the feature check is necessary but not
sufficient for subsumption. Candidates that pass the feature filter
still need full subsumption verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pyladr.core.clause import Clause
from pyladr.core.term import Term


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Feature vector for a clause, used for subsumption prefiltering.

    A clause C can only subsume D if:
    - C.pos_lit_count <= D.pos_lit_count
    - C.neg_lit_count <= D.neg_lit_count
    - For each symbol s: C.symbol_counts[s] <= D.symbol_counts[s]
    """

    pos_lit_count: int
    neg_lit_count: int
    symbol_counts: tuple[tuple[int, int], ...]  # sorted (symnum, count) pairs
    total_depth: int
    variable_count: int

    @staticmethod
    def from_clause(c: Clause) -> FeatureVector:
        """Compute feature vector for a clause."""
        pos = sum(1 for lit in c.literals if lit.sign)
        neg = sum(1 for lit in c.literals if not lit.sign)

        sym_counts: dict[int, int] = {}
        total_depth = 0
        var_set: set[int] = set()

        for lit in c.literals:
            _collect_features(lit.atom, sym_counts, var_set, 0)
            total_depth += lit.atom.depth

        return FeatureVector(
            pos_lit_count=pos,
            neg_lit_count=neg,
            symbol_counts=tuple(sorted(sym_counts.items())),
            total_depth=total_depth,
            variable_count=len(var_set),
        )

    def can_subsume(self, other: FeatureVector) -> bool:
        """Check if a clause with self's features could subsume one with other's.

        Necessary condition: subsumer must have <= counts for each feature.
        """
        if self.pos_lit_count > other.pos_lit_count:
            return False
        if self.neg_lit_count > other.neg_lit_count:
            return False

        # Check symbol counts: subsumer's symbols must be subset
        other_syms = dict(other.symbol_counts)
        for sym, count in self.symbol_counts:
            if count > other_syms.get(sym, 0):
                return False

        return True


def _collect_features(
    t: Term,
    sym_counts: dict[int, int],
    var_set: set[int],
    depth: int,
) -> None:
    """Collect symbol counts and variable set from a term."""
    if t.is_variable:
        var_set.add(t.varnum)
        return
    sym_counts[t.symnum] = sym_counts.get(t.symnum, 0) + 1
    for arg in t.args:
        _collect_features(arg, sym_counts, var_set, depth + 1)


class FeatureIndex:
    """Feature-based index for fast subsumption candidate filtering.

    Maintains feature vectors for all indexed clauses and provides
    fast candidate retrieval for forward and backward subsumption.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[int, tuple[Clause, FeatureVector]] = {}

    def insert(self, c: Clause) -> None:
        """Add a clause to the feature index."""
        fv = FeatureVector.from_clause(c)
        self._entries[c.id] = (c, fv)

    def delete(self, c: Clause) -> None:
        """Remove a clause from the feature index."""
        self._entries.pop(c.id, None)

    def forward_candidates(self, d: Clause) -> list[Clause]:
        """Return clauses that could potentially subsume d.

        Filters by feature vector: only returns clauses c where
        c's features are dominated by d's features.
        """
        d_fv = FeatureVector.from_clause(d)
        candidates = []
        for c, c_fv in self._entries.values():
            if c_fv.can_subsume(d_fv):
                candidates.append(c)
        return candidates

    def backward_candidates(self, c: Clause) -> list[Clause]:
        """Return clauses that could potentially be subsumed by c.

        Filters by feature vector: only returns clauses d where
        c's features are dominated by d's features.
        """
        c_fv = FeatureVector.from_clause(c)
        candidates = []
        for d, d_fv in self._entries.values():
            if d is not c and c_fv.can_subsume(d_fv):
                candidates.append(d)
        return candidates

    def __len__(self) -> int:
        return len(self._entries)
