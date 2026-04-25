"""Positive hyper-resolution matching C LADR hyper_res.c.

Hyper-resolution resolves ALL negative literals of a nucleus clause
simultaneously with positive clauses (satellites). The result is
always an all-positive clause.

Algorithm (C hyper_resolution):
1. Given a nucleus clause with negative literals -A1, ..., -An
2. For each -Ai, find a satellite with a positive literal Ai' that unifies
3. Accumulate substitutions across all steps (shared trail)
4. Build resolvent from positive nucleus literals + remaining satellite literals
5. Result must be all-positive; if not, discard

The C implementation uses a backtracking search: try each candidate satellite
for the first unresolved negative literal, then recursively resolve the rest.
If any step fails, backtrack and try the next candidate.
"""

from __future__ import annotations

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.substitution import (
    Context,
    Trail,
    apply_substitution,
    unify,
)


def hyper_resolve(
    nucleus: Clause,
    satellites: list[Clause],
) -> list[Clause]:
    """Generate all positive hyper-resolvents of nucleus with satellites.

    Matches C hyper_resolution(): the nucleus must have at least one
    negative literal. Each negative literal is resolved with a positive
    literal from some satellite clause. The resolvent must be all-positive.

    Args:
        nucleus: The clause whose negative literals are resolved away.
        satellites: Available clauses to serve as satellites (must have
            positive literals).

    Returns:
        List of hyper-resolvents (all-positive clauses).
    """
    # Identify negative literal indices in the nucleus
    neg_indices = [i for i, lit in enumerate(nucleus.literals) if not lit.sign]
    if not neg_indices:
        return []  # No negative literals — nothing to hyper-resolve

    results: list[Clause] = []

    # Set up shared contexts and trail for the entire hyper-resolution step.
    # The nucleus gets its own context; each satellite gets a fresh context
    # allocated during the backtracking search.
    nuc_ctx = Context()
    trail = Trail()

    # Start recursive backtracking search
    _hyper_backtrack(
        nucleus=nucleus,
        nuc_ctx=nuc_ctx,
        neg_indices=neg_indices,
        depth=0,
        satellites=satellites,
        sat_bindings=[],  # (satellite_clause, satellite_context, resolved_lit_idx)
        trail=trail,
        results=results,
    )

    return results


def _hyper_backtrack(
    nucleus: Clause,
    nuc_ctx: Context,
    neg_indices: list[int],
    depth: int,
    satellites: list[Clause],
    sat_bindings: list[tuple[Clause, Context, int]],
    trail: Trail,
    results: list[Clause],
) -> None:
    """Iterative backtracking search for hyper-resolution satellites.

    Converts the recursive backtracking into an explicit stack to avoid
    hitting Python's recursion limit on problems with many negative literals.

    At each depth, try to resolve neg_indices[depth] with some positive
    literal from a satellite. On success, descend to the next depth.
    When all negative literals are resolved, build the resolvent.

    Each stack frame is a generator that yields candidates for one depth
    level. Generators maintain their iteration state across breaks,
    allowing the search to resume correctly after exploring deeper levels.
    """
    total = len(neg_indices)

    if depth >= total:
        resolvent = _build_hyper_resolvent(
            nucleus, nuc_ctx, neg_indices, sat_bindings,
        )
        if resolvent is not None:
            results.append(resolvent)
        return

    def _candidates(d: int):
        neg_lit = nucleus.literals[neg_indices[d]]
        for sat in satellites:
            for j, sat_lit in enumerate(sat.literals):
                if sat_lit.sign:
                    yield neg_lit, sat, j, sat_lit

    # Stack of generator iterators, one per depth level.
    # pending_undo tracks the trail position to restore when backing out
    # of each pushed level (1:1 with stack entries except the initial).
    stack = [_candidates(depth)]
    pending_undo: list[int] = []

    while stack:
        current_it = stack[-1]
        descended = False

        for neg_lit, sat, j, sat_lit in current_it:
            saved_pos = trail.position
            sat_ctx = Context()

            if unify(neg_lit.atom, nuc_ctx, sat_lit.atom, sat_ctx, trail):
                current_depth = depth + len(stack)

                if current_depth == total:
                    # All negative literals resolved — build resolvent
                    sat_bindings.append((sat, sat_ctx, j))
                    resolvent = _build_hyper_resolvent(
                        nucleus, nuc_ctx, neg_indices, sat_bindings,
                    )
                    if resolvent is not None:
                        results.append(resolvent)
                    sat_bindings.pop()
                    trail.undo_to(saved_pos)
                    # Continue iterating candidates at this level
                else:
                    # Descend to next depth level
                    sat_bindings.append((sat, sat_ctx, j))
                    pending_undo.append(saved_pos)
                    stack.append(_candidates(current_depth))
                    descended = True
                    break
            else:
                trail.undo_to(saved_pos)

        if not descended:
            # Current level exhausted — pop and backtrack
            stack.pop()
            if pending_undo:
                trail.undo_to(pending_undo.pop())
                sat_bindings.pop()


def _build_hyper_resolvent(
    nucleus: Clause,
    nuc_ctx: Context,
    neg_indices: list[int],
    sat_bindings: list[tuple[Clause, Context, int]],
) -> Clause | None:
    """Build the resolvent after all negative literals have been matched.

    The resolvent contains:
    1. Positive literals from the nucleus (instantiated)
    2. Remaining literals from each satellite (instantiated), excluding
       the literal that was resolved upon

    The resolvent must be all-positive; if any negative literal remains,
    return None.

    Args:
        nucleus: The nucleus clause.
        nuc_ctx: Context for nucleus variables.
        neg_indices: Indices of negative literals that were resolved.
        sat_bindings: (satellite, context, resolved_lit_idx) for each step.

    Returns:
        The resolvent Clause, or None if not all-positive.
    """
    neg_set = set(neg_indices)
    new_lits: list[Literal] = []

    # Add non-resolved literals from the nucleus (should be positive)
    for i, lit in enumerate(nucleus.literals):
        if i in neg_set:
            continue
        new_atom = apply_substitution(lit.atom, nuc_ctx)
        new_lits.append(Literal(sign=lit.sign, atom=new_atom))

    # Add remaining literals from each satellite
    for sat, sat_ctx, resolved_idx in sat_bindings:
        for j, lit in enumerate(sat.literals):
            if j == resolved_idx:
                continue  # Skip the literal that was resolved upon
            new_atom = apply_substitution(lit.atom, sat_ctx)
            new_lits.append(Literal(sign=lit.sign, atom=new_atom))

    # Hyper-resolution requires all-positive result
    if any(not lit.sign for lit in new_lits):
        return None

    # Build justification: nucleus ID + all satellite IDs
    clause_ids = (nucleus.id,) + tuple(sat.id for sat, _, _ in sat_bindings)

    just = Justification(
        just_type=JustType.HYPER_RES,
        clause_ids=clause_ids,
    )

    return Clause(
        literals=tuple(new_lits),
        justification=(just,),
    )


def hyper_resolve_with_satellite(
    nucleus: Clause,
    required_satellite: Clause,
    all_satellites: list[Clause],
) -> list[Clause]:
    """Generate hyper-resolvents where required_satellite is used for at least one negative literal.

    Efficient satellite-case implementation: instead of generating all
    hyper-resolvents and filtering, we try the required satellite at each
    negative literal position and recurse for the rest.

    Args:
        nucleus: The clause whose negative literals are resolved away.
        required_satellite: A specific satellite that MUST participate.
        all_satellites: All available satellites for the remaining positions.

    Returns:
        List of hyper-resolvents that use required_satellite.
    """
    neg_indices = [i for i, lit in enumerate(nucleus.literals) if not lit.sign]
    if not neg_indices:
        return []

    results: list[Clause] = []

    # Try required_satellite at each negative literal position
    for pos, neg_idx in enumerate(neg_indices):
        neg_lit = nucleus.literals[neg_idx]

        # Try each positive literal of required_satellite
        for j, sat_lit in enumerate(required_satellite.literals):
            if not sat_lit.sign:
                continue

            nuc_ctx = Context()
            sat_ctx = Context()
            trail = Trail()

            if not unify(neg_lit.atom, nuc_ctx, sat_lit.atom, sat_ctx, trail):
                trail.undo_to(0)
                continue

            # This position is fixed to required_satellite.
            # Now resolve remaining negative literals with all_satellites.
            remaining_neg = [ni for k, ni in enumerate(neg_indices) if k != pos]

            if not remaining_neg:
                # Only one negative literal, and we just resolved it
                resolvent = _build_hyper_resolvent(
                    nucleus, nuc_ctx, neg_indices,
                    [(required_satellite, sat_ctx, j)],
                )
                if resolvent is not None:
                    results.append(resolvent)
            else:
                # Recurse for remaining negative literals
                sat_bindings: list[tuple[Clause, Context, int]] = [
                    (required_satellite, sat_ctx, j),
                ]
                _hyper_backtrack_remaining(
                    nucleus=nucleus,
                    nuc_ctx=nuc_ctx,
                    neg_indices=neg_indices,
                    remaining_neg=remaining_neg,
                    depth=0,
                    satellites=all_satellites,
                    sat_bindings=sat_bindings,
                    fixed_pos=pos,
                    trail=trail,
                    results=results,
                )

            trail.undo_to(0)

    return results


def _hyper_backtrack_remaining(
    nucleus: Clause,
    nuc_ctx: Context,
    neg_indices: list[int],
    remaining_neg: list[int],
    depth: int,
    satellites: list[Clause],
    sat_bindings: list[tuple[Clause, Context, int]],
    fixed_pos: int,
    trail: Trail,
    results: list[Clause],
) -> None:
    """Iterative backtracking over remaining negative literals after fixing one position."""
    total = len(remaining_neg)

    def _build_ordered_resolvent():
        """Reorder sat_bindings to match neg_indices order and build resolvent."""
        ordered_bindings: list[tuple[Clause, Context, int]] = []
        remaining_iter = iter(
            b for b in sat_bindings[1:]  # skip the fixed one
        )
        for k in range(len(neg_indices)):
            if k == fixed_pos:
                ordered_bindings.append(sat_bindings[0])
            else:
                ordered_bindings.append(next(remaining_iter))

        resolvent = _build_hyper_resolvent(
            nucleus, nuc_ctx, neg_indices, ordered_bindings,
        )
        if resolvent is not None:
            results.append(resolvent)

    if depth >= total:
        _build_ordered_resolvent()
        return

    def _candidates(d: int):
        neg_lit = nucleus.literals[remaining_neg[d]]
        for sat in satellites:
            for j, sat_lit in enumerate(sat.literals):
                if sat_lit.sign:
                    yield neg_lit, sat, j, sat_lit

    stack = [_candidates(depth)]
    pending_undo: list[int] = []

    while stack:
        current_it = stack[-1]
        descended = False

        for neg_lit, sat, j, sat_lit in current_it:
            saved_pos = trail.position
            sat_ctx = Context()

            if unify(neg_lit.atom, nuc_ctx, sat_lit.atom, sat_ctx, trail):
                current_depth = depth + len(stack)

                if current_depth == total:
                    sat_bindings.append((sat, sat_ctx, j))
                    _build_ordered_resolvent()
                    sat_bindings.pop()
                    trail.undo_to(saved_pos)
                else:
                    sat_bindings.append((sat, sat_ctx, j))
                    pending_undo.append(saved_pos)
                    stack.append(_candidates(current_depth))
                    descended = True
                    break
            else:
                trail.undo_to(saved_pos)

        if not descended:
            stack.pop()
            if pending_undo:
                trail.undo_to(pending_undo.pop())
                sat_bindings.pop()


def all_hyper_resolvents(
    given: Clause,
    usable: list[Clause],
) -> list[Clause]:
    """Generate all hyper-resolvents involving the given clause.

    Matches C hyper_res_from_given(): the given clause can serve as
    either the nucleus or a satellite.

    Case 1: Given is nucleus — resolve its negative literals with
        positive literals from usable clauses.
    Case 2: Given is satellite — for each usable clause that has negative
        literals (potential nucleus), check if the given clause can help
        resolve one of the negative literals.

    Args:
        given: The newly selected given clause.
        usable: Clauses available as satellites (or nuclei).

    Returns:
        List of all hyper-resolvents found.
    """
    results: list[Clause] = []

    # Case 1: Given clause as nucleus
    if any(not lit.sign for lit in given.literals):
        resolvents = hyper_resolve(given, usable)
        results.extend(resolvents)

    # Case 2: Given clause as satellite
    # Only if the given clause has positive literals (can serve as satellite)
    # Use hyper_resolve_with_satellite to only generate resolvents where
    # the given clause participates, avoiding O(S²) full scan + filter.
    if any(lit.sign for lit in given.literals):
        for nuc in usable:
            if nuc is given:
                continue
            if not any(not lit.sign for lit in nuc.literals):
                continue  # Nucleus must have negative literals
            resolvents = hyper_resolve_with_satellite(nuc, given, usable)
            results.extend(resolvents)

    return results


def indexed_hyper_resolution(
    given: Clause,
    clashable_idx: object,
    usable: list[Clause],
) -> list[Clause]:
    """Generate hyper-resolvents using the clashable index for efficiency.

    Matches C hyper_resolution(): the given clause serves as either
    nucleus or satellite depending on its literal structure.

    For positive hyper-resolution:
    - If given is all-positive: it's a satellite. Find nuclei in usable
      that have negative literals, and use the index to find other satellites.
    - If given has negative literals: it's a nucleus. Use the index to
      find positive satellites.

    Args:
        given: The newly selected given clause.
        clashable_idx: The Mindex used for indexed literal retrieval.
        usable: Usable clause list (for nucleus scan in satellite case).

    Returns:
        List of all hyper-resolvents found.
    """
    results: list[Clause] = []

    has_neg = any(not lit.sign for lit in given.literals)
    has_pos = any(lit.sign for lit in given.literals)

    # Case 1: Given as nucleus (has negative literals to resolve)
    if has_neg:
        _hyper_nucleus_indexed(given, clashable_idx, results)

    # Case 2: Given as satellite (has positive literals)
    # For each literal of the given, look up nuclei whose negative literals
    # unify with it, then complete the hyper-resolution for those nuclei.
    if has_pos:
        _hyper_satellite_indexed(given, clashable_idx, usable, results)

    return results


def _hyper_nucleus_indexed(
    nucleus: Clause,
    clashable_idx: object,
    results: list[Clause],
) -> None:
    """Hyper-resolve with nucleus=given, using index to find satellites.

    For each negative literal in the nucleus, retrieve positive literals
    from the index that could unify. Then do backtracking search across
    all negative literals.
    """
    neg_indices = [i for i, lit in enumerate(nucleus.literals) if not lit.sign]
    if not neg_indices:
        return

    nuc_ctx = Context()
    trail = Trail()

    _hyper_backtrack_indexed(
        nucleus=nucleus,
        nuc_ctx=nuc_ctx,
        neg_indices=neg_indices,
        depth=0,
        clashable_idx=clashable_idx,
        sat_bindings=[],
        trail=trail,
        results=results,
    )


def _hyper_backtrack_indexed(
    nucleus: Clause,
    nuc_ctx: Context,
    neg_indices: list[int],
    depth: int,
    clashable_idx: object,
    sat_bindings: list[tuple[Clause, Context, int]],
    trail: Trail,
    results: list[Clause],
) -> None:
    """Iterative backtracking search using index to find satellite candidates.

    Like _hyper_backtrack but uses clashable_idx.retrieve_unifiables()
    to find candidate positive literals, matching C clash_recurse().
    Uses an explicit stack to avoid recursion limit issues.
    """
    total = len(neg_indices)

    if depth >= total:
        resolvent = _build_hyper_resolvent(
            nucleus, nuc_ctx, neg_indices, sat_bindings,
        )
        if resolvent is not None:
            results.append(resolvent)
        return

    def _candidates(d: int):
        neg_lit = nucleus.literals[neg_indices[d]]
        candidates = clashable_idx.retrieve_unifiables(neg_lit.atom)
        for candidate in candidates:
            sat_clause, sat_lit = candidate
            if not sat_lit.sign:
                continue
            sat_lit_idx = -1
            for j, lit in enumerate(sat_clause.literals):
                if lit is sat_lit:
                    sat_lit_idx = j
                    break
            if sat_lit_idx < 0:
                continue
            yield neg_lit, sat_clause, sat_lit_idx, sat_lit

    stack = [_candidates(depth)]
    pending_undo: list[int] = []

    while stack:
        current_it = stack[-1]
        descended = False

        for neg_lit, sat_clause, sat_lit_idx, sat_lit in current_it:
            saved_pos = trail.position
            sat_ctx = Context()

            if unify(neg_lit.atom, nuc_ctx, sat_lit.atom, sat_ctx, trail):
                current_depth = depth + len(stack)

                if current_depth == total:
                    sat_bindings.append((sat_clause, sat_ctx, sat_lit_idx))
                    resolvent = _build_hyper_resolvent(
                        nucleus, nuc_ctx, neg_indices, sat_bindings,
                    )
                    if resolvent is not None:
                        results.append(resolvent)
                    sat_bindings.pop()
                    trail.undo_to(saved_pos)
                else:
                    sat_bindings.append((sat_clause, sat_ctx, sat_lit_idx))
                    pending_undo.append(saved_pos)
                    stack.append(_candidates(current_depth))
                    descended = True
                    break
            else:
                trail.undo_to(saved_pos)

        if not descended:
            stack.pop()
            if pending_undo:
                trail.undo_to(pending_undo.pop())
                sat_bindings.pop()


def _hyper_satellite_indexed(
    given: Clause,
    clashable_idx: object,
    usable: list[Clause],
    results: list[Clause],
) -> None:
    """Hyper-resolve with given as satellite, finding nuclei from usable.

    For each positive literal in given, look up negative literals in the index
    to find potential nuclei. Then complete hyper-resolution for those nuclei
    using the index.

    Matches C hyper_satellite() which iterates the satellite's literals and
    uses hyper_sat_atom() to find matching nuclei via the index.
    """
    seen_nuclei: set[int] = set()

    for i, sat_lit in enumerate(given.literals):
        if not sat_lit.sign:
            continue  # Only positive literals can serve as satellite

        # Look up negative literals that unify with this positive literal
        candidates = clashable_idx.retrieve_unifiables(sat_lit.atom)

        for candidate in candidates:
            nuc_clause, nuc_lit = candidate
            if nuc_lit.sign:
                continue  # We need negative literals in the nucleus
            if nuc_clause is given:
                continue

            # Only process each nucleus once
            if nuc_clause.id in seen_nuclei:
                continue

            # Check this is a viable nucleus (has negative literals)
            if not any(not lit.sign for lit in nuc_clause.literals):
                continue

            seen_nuclei.add(nuc_clause.id)

            # Resolve only with given as a required satellite,
            # avoiding generation of resolvents that don't use given.
            nuc_results = hyper_resolve_with_satellite(
                nuc_clause, given, usable,
            )
            results.extend(nuc_results)
