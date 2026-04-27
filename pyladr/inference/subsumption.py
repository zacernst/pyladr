"""Subsumption checking matching C subsume.c.

Clause C subsumes clause D if there is a substitution theta such that
every literal in C*theta appears in D. This is NP-complete in general.

The implementation uses recursive literal mapping with backtracking,
matching the C subsume_literals() approach.

== Algorithm ==

1. subsumes(c, d): Check if c subsumes d using one-way matching.
   - For each literal clit of c, try to match clit into some literal dlit of d
     (same sign, pattern matches).
   - Recurse on the remaining literals of c.
   - Backtrack if a mapping fails.

2. forward_subsume(d, lindex): Find a clause in the index that subsumes d.
   - For each literal dlit of d, retrieve generalizations from lindex.
   - For each candidate c, check subsumes(c, d).
   - Unit subsumption is a fast path: if c is unit, one literal match suffices.

3. back_subsume(c, lindex): Find all clauses in the index subsumed by c.
   - Use first literal of c to retrieve instances from lindex.
   - For each candidate d, check subsumes(c, d).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyladr.core.clause import Clause, Literal
from pyladr.core.substitution import Context, Trail, match
from pyladr.core.term import Term

if TYPE_CHECKING:
    from pyladr.search.statistics import SearchStatistics


# ── Core subsumption ─────────────────────────────────────────────────────────


def _subsume_literals(
    clits: tuple[Literal, ...],
    clit_idx: int,
    subst: Context,
    d: Clause,
    trail: Trail,
) -> bool:
    """Recursive literal mapping matching C subsume_literals().

    Try to map each literal of the subsumer (starting at clit_idx) into
    some literal of d via one-way matching. Backtracks on failure.

    Args:
        clits: Literals of the potential subsumer clause.
        clit_idx: Current index into clits (how far we've matched).
        subst: Matching context for the subsumer's variables.
        d: Target clause being tested for subsumption.
        trail: Trail for backtracking bindings.

    Returns:
        True if all remaining literals of clits can be matched into d.
    """
    if clit_idx >= len(clits):
        return True  # All literals matched

    clit = clits[clit_idx]

    for dlit in d.literals:
        if clit.sign != dlit.sign:
            continue

        # Try to match clit.atom against dlit.atom
        mark = trail.position
        if match(clit.atom, subst, dlit.atom, trail):
            # Recurse on remaining literals
            if _subsume_literals(clits, clit_idx + 1, subst, d, trail):
                return True
            # Failed — undo bindings from this attempt
            trail.undo_to(mark)

    return False


def subsumes(c: Clause, d: Clause, stats: SearchStatistics | None = None) -> bool:
    """Check if clause c subsumes clause d.

    Matching C subsumes(). Uses one-way matching (not backtrack unification).
    A clause cannot subsume a shorter clause (prevents factor deletion).

    Args:
        c: Potential subsumer.
        d: Potential subsumee.
        stats: Optional SearchStatistics to increment nonunit_subsumption_tests.

    Returns:
        True if c subsumes d.
    """
    nc = c.num_literals
    nd = d.num_literals

    # Empty clause subsumes everything
    if nc == 0:
        return True

    # c cannot subsume shorter clause
    if nc > nd:
        return False

    # Unit subsumption: if c is unit and we can match its literal
    # into any literal of d, then c subsumes d
    if nc == 1:
        clit = c.literals[0]
        subst = Context()
        trail = Trail()
        for dlit in d.literals:
            if clit.sign == dlit.sign:
                if match(clit.atom, subst, dlit.atom, trail):
                    trail.undo()
                    return True
                trail.undo_to(0)
        return False

    # Non-unit subsumption: full recursive check
    if stats is not None:
        stats.nonunit_subsumption_tests += 1

    # Heuristic: reorder c's literals so the most restrictive (fewest
    # matching candidates in d by sign) come first.  This prunes the
    # backtracking search tree from the top.
    clits = c.literals
    if nc >= 3:
        d_lits = d.literals
        pos_count = sum(1 for dl in d_lits if dl.sign)
        neg_count = nd - pos_count
        # Sort by number of same-sign literals in d (ascending = most restrictive first)
        clits = tuple(sorted(clits, key=lambda l: pos_count if l.sign else neg_count))

    subst = Context()
    trail = Trail()
    result = _subsume_literals(clits, 0, subst, d, trail)
    if result:
        trail.undo()
    return result


# ── Forward subsumption ──────────────────────────────────────────────────────


def forward_subsume(
    d: Clause, pos_idx, neg_idx, stats: SearchStatistics | None = None,
) -> Clause | None:
    """Find a clause in the index that subsumes d.

    Matching C forward_subsume(). Checks all literals of d because when
    d is subsumed by c, not all literals of d need to match a literal in c.
    c is indexed on the first literal only.

    Args:
        d: Clause to check for subsumption.
        pos_idx: Mindex for positive literals (generalizations).
        neg_idx: Mindex for negative literals (generalizations).
        stats: Optional SearchStatistics for tracking nonunit subsumption tests.

    Returns:
        The subsumer clause, or None if d is not subsumed.
    """
    nd = d.num_literals
    subst = Context()
    seen: set[int] = set()

    for dlit in d.literals:
        # Retrieve generalizations of dlit from the appropriate index
        # Mindex.retrieve_generalizations returns stored objects directly.
        # DiscrimWild is an imperfect filter — verify with subsumes().
        idx = pos_idx if dlit.sign else neg_idx
        for data in idx.retrieve_generalizations(dlit.atom):
            c, c_first_lit = data
            # Only consider candidates indexed on their first literal
            if c.literals[0] is not c_first_lit:
                continue
            if c.id in seen:
                continue
            seen.add(c.id)
            nc = c.num_literals
            if nc <= nd and subsumes(c, d, stats):
                return c

    return None


def forward_subsume_from_lists(
    d: Clause,
    clause_lists: list,
    stats: SearchStatistics | None = None,
) -> Clause | None:
    """Forward subsumption by scanning clause lists directly.

    Simpler version that scans lists without index-based retrieval.
    Used when literal indexes are not yet built.

    Args:
        d: Clause to check for subsumption.
        clause_lists: Lists of clauses to check against.
        stats: Optional SearchStatistics for tracking nonunit subsumption tests.

    Returns:
        The subsumer clause, or None.
    """
    nd = d.num_literals

    for clist in clause_lists:
        for c in clist:
            nc = c.num_literals
            if nc == 0:
                return c  # Empty clause subsumes everything
            if nc <= nd and subsumes(c, d, stats):
                return c

    return None


# ── Backward subsumption ─────────────────────────────────────────────────────


def back_subsume(c: Clause, pos_idx, neg_idx, stats: SearchStatistics | None = None) -> list[Clause]:
    """Find all clauses in the index subsumed by c.

    Matching C back_subsume(). Uses the first literal of c to retrieve
    instances from the index, then does full subsumption check.

    Args:
        c: The new clause (potential subsumer).
        pos_idx: Mindex for positive literals (instances).
        neg_idx: Mindex for negative literals (instances).

    Returns:
        List of clauses subsumed by c.
    """
    nc = c.num_literals
    if nc == 0:
        return []

    subsumees: list[Clause] = []
    seen: set[int] = set()
    clit = c.literals[0]

    # Retrieve instances of the first literal of c
    idx = pos_idx if clit.sign else neg_idx
    subst = Context()

    for datom, data in idx.retrieve_instances(clit.atom):
        d, d_lit = data
        if d is c or d.id in seen:
            continue
        nd = d.num_literals
        # Unit subsumption: c is unit → it subsumes d
        if nc == 1:
            subsumees.append(d)
            seen.add(d.id)
        elif nc <= nd and subsumes(c, d, stats):
            subsumees.append(d)
            seen.add(d.id)

    return subsumees


def back_subsume_from_lists(
    c: Clause,
    clause_lists: list,
    stats: SearchStatistics | None = None,
) -> list[Clause]:
    """Backward subsumption by scanning clause lists directly.

    Simpler version without index retrieval.

    Args:
        c: The new clause (potential subsumer).
        clause_lists: Lists of clauses to check.
        stats: Optional SearchStatistics for tracking nonunit subsumption tests.

    Returns:
        List of clauses subsumed by c.
    """
    nc = c.num_literals
    if nc == 0:
        return []

    subsumees: list[Clause] = []

    for clist in clause_lists:
        for d in clist:
            if d is c:
                continue
            nd = d.num_literals
            if nc <= nd and subsumes(c, d, stats):
                subsumees.append(d)

    return subsumees


# ── Predicate-sign index for back subsumption ────────────────────────────────


_VAR_MARKER = 0  # Marker for variable positions in signature keys


def _clause_pred_signs(c: Clause) -> set[tuple[int, bool]]:
    """Extract (predicate_symnum, sign) set from a clause's literals."""
    result: set[tuple[int, bool]] = set()
    for lit in c.literals:
        atom = lit.atom
        # Use symnum for rigid atoms, -1 for variable atoms
        ps = -atom.private_symbol if atom.private_symbol < 0 else -1
        result.add((ps, lit.sign))
    return result


def _literal_signatures(c: Clause) -> list[tuple[int, bool, tuple[int, ...]]]:
    """Extract structural signatures for each literal of a clause.

    Each signature is (pred_symnum, sign, arg_roots) where arg_roots is a
    tuple of the root symbol IDs for each argument of the predicate atom.
    Variables use _VAR_MARKER (0) since private_symbol >= 0 means variable.

    Returns one signature per literal (in order).
    """
    sigs: list[tuple[int, bool, tuple[int, ...]]] = []
    for lit in c.literals:
        atom = lit.atom
        pred = -atom.private_symbol if atom.private_symbol < 0 else -1
        # For each argument of the predicate, record the root symbol
        # Variables get _VAR_MARKER; rigid symbols get their symnum
        arg_roots: tuple[int, ...] = tuple(
            (-a.private_symbol if a.private_symbol < 0 else _VAR_MARKER)
            for a in atom.args
        )
        sigs.append((pred, lit.sign, arg_roots))
    return sigs


class BackSubsumptionIndex:
    """Hash-based index for fast back subsumption candidate retrieval.

    Uses two levels of indexing:
    1. Coarse: (predicate_symnum, sign) buckets for fast intersection.
    2. Fine: (predicate_symnum, sign, arg_root_symbols) structural signature
       buckets that capture the top-level argument structure of each literal.

    For back subsumption, candidates are filtered by structural signature
    compatibility (a subsumer literal with rigid arg roots can only match
    literals with the same roots), plus weight bounds and literal count.
    """

    __slots__ = (
        "_by_pred_sign", "_by_signature", "_clauses",
        "_pred_signs_cache", "_sig_cache", "_weights",
    )

    def __init__(self) -> None:
        # Coarse: (symnum, sign) -> set of clause IDs
        self._by_pred_sign: dict[tuple[int, bool], set[int]] = {}
        # Fine: (symnum, sign, arg_roots) -> set of clause IDs
        self._by_signature: dict[tuple[int, bool, tuple[int, ...]], set[int]] = {}
        # Map: clause ID -> Clause
        self._clauses: dict[int, Clause] = {}
        # Cache: clause ID -> set of (symnum, sign) for deindex
        self._pred_signs_cache: dict[int, set[tuple[int, bool]]] = {}
        # Cache: clause ID -> list of full signatures for deindex
        self._sig_cache: dict[int, list[tuple[int, bool, tuple[int, ...]]]] = {}
        # Cache: clause ID -> weight (for fast weight-bound filtering)
        self._weights: dict[int, float] = {}

    def insert(self, c: Clause) -> None:
        """Add a clause to the back subsumption index."""
        cid = c.id
        self._clauses[cid] = c
        self._weights[cid] = c.weight

        ps_set = _clause_pred_signs(c)
        self._pred_signs_cache[cid] = ps_set
        for key in ps_set:
            bucket = self._by_pred_sign.get(key)
            if bucket is None:
                bucket = set()
                self._by_pred_sign[key] = bucket
            bucket.add(cid)

        sigs = _literal_signatures(c)
        self._sig_cache[cid] = sigs
        for sig in sigs:
            bucket = self._by_signature.get(sig)
            if bucket is None:
                bucket = set()
                self._by_signature[sig] = bucket
            bucket.add(cid)

    def remove(self, c: Clause) -> None:
        """Remove a clause from the back subsumption index."""
        cid = c.id
        self._clauses.pop(cid, None)
        self._weights.pop(cid, None)

        ps_set = self._pred_signs_cache.pop(cid, None)
        if ps_set is not None:
            for key in ps_set:
                bucket = self._by_pred_sign.get(key)
                if bucket is not None:
                    bucket.discard(cid)

        sigs = self._sig_cache.pop(cid, None)
        if sigs is not None:
            for sig in sigs:
                bucket = self._by_signature.get(sig)
                if bucket is not None:
                    bucket.discard(cid)

    def _sig_compatible_ids(
        self, pred: int, sign: bool, arg_roots: tuple[int, ...]
    ) -> set[int] | None:
        """Find clause IDs with a literal whose signature is compatible.

        A candidate literal d_lit is compatible with subsumer literal c_lit if
        for every position where c_lit has a rigid symbol, d_lit has the same
        symbol (variables in c_lit can match anything in d_lit).

        If all args in the subsumer are rigid, we can do a single hash lookup.
        If any arg is a variable (_VAR_MARKER), we must union over all matching
        signatures, falling back to the coarse (pred, sign) bucket.
        """
        has_var = _VAR_MARKER in arg_roots
        if not has_var:
            # All args rigid: exact signature match required
            bucket = self._by_signature.get((pred, sign, arg_roots))
            return bucket  # may be None

        # Variable args: fall back to coarse (pred, sign) bucket since the
        # subsumer variable can match any argument structure.
        return self._by_pred_sign.get((pred, sign))

    def candidates(self, c: Clause) -> list[Clause]:
        """Find clauses that could potentially be subsumed by c.

        Uses structural signature matching: for each literal of c, finds
        candidate clauses with a compatible literal (same pred, sign, and
        matching top-level argument roots). Intersects across all literals.

        Also filters by weight and literal count bounds.
        """
        nc = c.num_literals
        if nc == 0:
            return []

        sigs = _literal_signatures(c)

        # Collect candidate sets per (pred, sign) — use finest available filter.
        # Group signatures by (pred, sign) to handle multi-literal clauses.
        from collections import defaultdict
        by_ps: dict[tuple[int, bool], list[tuple[int, ...]]] = defaultdict(list)
        for pred, sign, arg_roots in sigs:
            by_ps[(pred, sign)].append(arg_roots)

        sets: list[set[int]] = []
        for (pred, sign), arg_root_list in by_ps.items():
            # For this (pred, sign), union all compatible signatures
            combined: set[int] | None = None
            for arg_roots in arg_root_list:
                bucket = self._sig_compatible_ids(pred, sign, arg_roots)
                if bucket is None:
                    # No candidates for this specific signature; if all sigs for
                    # this pred-sign fail, the intersection will be empty
                    continue
                if combined is None:
                    combined = set(bucket)
                else:
                    combined |= bucket

            if combined is None or not combined:
                return []  # No clause matches this (pred, sign) requirement
            sets.append(combined)

        if not sets:
            return []

        # Intersect: start with smallest set for efficiency
        sets.sort(key=len)
        result_ids = sets[0].copy()
        for s in sets[1:]:
            result_ids &= s
            if not result_ids:
                return []

        # Filter: c cannot subsume itself, nc <= nd required,
        # and c.weight <= d.weight required (subsumer cannot be heavier).
        cid = c.id
        cw = c.weight
        weights = self._weights
        candidates = []
        for did in result_ids:
            if did == cid:
                continue
            d = self._clauses.get(did)
            if d is not None and nc <= d.num_literals and cw <= weights[did]:
                candidates.append(d)
        return candidates


def back_subsume_indexed(
    c: Clause,
    back_idx: BackSubsumptionIndex,
    stats: SearchStatistics | None = None,
) -> list[Clause]:
    """Backward subsumption using predicate-sign index.

    Finds candidates via hash-based lookup, then verifies with subsumes().
    Much faster than linear scanning when there are many clauses.

    Args:
        c: The new clause (potential subsumer).
        back_idx: Index of existing clauses.
        stats: Optional SearchStatistics for tracking nonunit subsumption tests.

    Returns:
        List of clauses subsumed by c.
    """
    nc = c.num_literals
    if nc == 0:
        return []

    candidates = back_idx.candidates(c)
    subsumees: list[Clause] = []

    for d in candidates:
        if subsumes(c, d, stats):
            subsumees.append(d)

    return subsumees
