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

from pyladr.core.clause import Clause, Literal
from pyladr.core.substitution import Context, Trail, match
from pyladr.core.term import Term


# ── Statistics ────────────────────────────────────────────────────────────────

_nonunit_subsumption_tests = 0


def nonunit_subsumption_tests() -> int:
    """Return count of nonunit subsumption tests (C nonunit_subsumption_tests)."""
    return _nonunit_subsumption_tests


def reset_subsumption_stats() -> None:
    """Reset subsumption statistics (for testing)."""
    global _nonunit_subsumption_tests
    _nonunit_subsumption_tests = 0


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


def subsumes(c: Clause, d: Clause) -> bool:
    """Check if clause c subsumes clause d.

    Matching C subsumes(). Uses one-way matching (not backtrack unification).
    A clause cannot subsume a shorter clause (prevents factor deletion).

    Args:
        c: Potential subsumer.
        d: Potential subsumee.

    Returns:
        True if c subsumes d.
    """
    global _nonunit_subsumption_tests

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
    _nonunit_subsumption_tests += 1
    subst = Context()
    trail = Trail()
    result = _subsume_literals(c.literals, 0, subst, d, trail)
    if result:
        trail.undo()
    return result


# ── Forward subsumption ──────────────────────────────────────────────────────


def forward_subsume(d: Clause, pos_idx, neg_idx) -> Clause | None:
    """Find a clause in the index that subsumes d.

    Matching C forward_subsume(). Checks all literals of d because when
    d is subsumed by c, not all literals of d need to match a literal in c.
    c is indexed on the first literal only.

    Args:
        d: Clause to check for subsumption.
        pos_idx: Mindex for positive literals (generalizations).
        neg_idx: Mindex for negative literals (generalizations).

    Returns:
        The subsumer clause, or None if d is not subsumed.
    """
    nd = d.num_literals
    subst = Context()

    for dlit in d.literals:
        # Retrieve generalizations of dlit from the appropriate index
        idx = pos_idx if dlit.sign else neg_idx
        for catom, data in idx.retrieve_generalizations(dlit.atom):
            c, c_first_lit = data
            # Only consider candidates indexed on their first literal
            if c.literals[0] is not c_first_lit:
                continue
            nc = c.num_literals
            # Unit subsumption: first literal matched → c subsumes d
            if nc == 1:
                return c
            # Non-unit: full subsumption check (c cannot subsume shorter d)
            if nc <= nd and subsumes(c, d):
                return c

    return None


def forward_subsume_from_lists(
    d: Clause,
    clause_lists: list,
) -> Clause | None:
    """Forward subsumption by scanning clause lists directly.

    Simpler version that scans lists without index-based retrieval.
    Used when literal indexes are not yet built.

    Args:
        d: Clause to check for subsumption.
        clause_lists: Lists of clauses to check against.

    Returns:
        The subsumer clause, or None.
    """
    nd = d.num_literals

    for clist in clause_lists:
        for c in clist:
            nc = c.num_literals
            if nc == 0:
                return c  # Empty clause subsumes everything
            if nc <= nd and subsumes(c, d):
                return c

    return None


# ── Backward subsumption ─────────────────────────────────────────────────────


def back_subsume(c: Clause, pos_idx, neg_idx) -> list[Clause]:
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
        elif nc <= nd and subsumes(c, d):
            subsumees.append(d)
            seen.add(d.id)

    return subsumees


def back_subsume_from_lists(
    c: Clause,
    clause_lists: list,
) -> list[Clause]:
    """Backward subsumption by scanning clause lists directly.

    Simpler version without index retrieval.

    Args:
        c: The new clause (potential subsumer).
        clause_lists: Lists of clauses to check.

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
            if nc <= nd and subsumes(c, d):
                subsumees.append(d)

    return subsumees
