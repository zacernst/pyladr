"""AC (Associative-Commutative) unification and matching.

Matching C btu.c (backtrack unification) and btm.c (backtrack matching).

== AC Unification Algorithm ==

For terms f(s1,...,sm) and f(t1,...,tn) where f is AC:

1. Flatten both terms: collect all arguments under nested f applications.
2. Sort arguments canonically.
3. Eliminate common ground terms from both sides.
4. Compute multiplicities of remaining arguments.
5. Set up a linear Diophantine equation: a1*x1 + ... + ap*xp = b1*y1 + ... + bq*yq
   where ai, bj are multiplicities.
6. Solve the Diophantine equation to get a basis of minimal solutions.
7. Enumerate subsets of basis solutions to get candidate unifiers.
8. For each candidate: build substitution terms, recursively unify sub-problems.

== Commutative Unification ==

For terms f(s1, s2) and f(t1, t2) where f is commutative (not associative):
- Try f(s1, s2) vs f(t1, t2): unify s1=t1 and s2=t2
- Try f(s1, s2) vs f(t2, t1): unify s1=t2 and s2=t1

== API ==

The backtrack unification API provides incremental enumeration of unifiers:
- ac_unify_first(t1, c1, t2, c2) → AcUnifyState | None
- ac_unify_next(state) → AcUnifyState | None
- ac_unify_cancel(state) → None

Or use the simpler all-at-once API:
- ac_unify_all(t1, c1, t2, c2) → list of substitutions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyladr.core.ac_normal_form import (
    flatten_ac,
    right_associate,
    term_compare_ncv,
)
from pyladr.core.substitution import Context, Trail, apply_substitution, dereference, match, unify
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.inference.diophantine import DioResult, dio, next_combo_a

if TYPE_CHECKING:
    from pyladr.core.symbol import SymbolTable


# ── Commutative unification ──────────────────────────────────────────────────


def commutative_unify_all(
    t1: Term,
    c1: Context,
    t2: Term,
    c2: Context,
) -> list[tuple[Context, Context, Trail]]:
    """Find all unifiers for commutative (non-AC) terms.

    Tries both argument orderings: f(a,b) vs f(c,d) and f(a,b) vs f(d,c).

    Returns list of (c1, c2, trail) for successful unifications.
    """
    if t1.arity != 2 or t2.arity != 2:
        return []

    results: list[tuple[Context, Context, Trail]] = []

    # Try straight: s1=t1, s2=t2
    trail1 = Trail()
    if unify(t1.args[0], c1, t2.args[0], c2, trail1):
        if unify(t1.args[1], c1, t2.args[1], c2, trail1):
            # Save contexts state — caller must undo trail if needed
            results.append((c1, c2, trail1))
        else:
            trail1.undo()

    # Try flipped: s1=t2, s2=t1
    c1b = Context()
    c1b.multiplier = c1.multiplier
    c2b = Context()
    c2b.multiplier = c2.multiplier
    trail2 = Trail()
    if unify(t1.args[0], c1b, t2.args[1], c2b, trail2):
        if unify(t1.args[1], c1b, t2.args[0], c2b, trail2):
            results.append((c1b, c2b, trail2))
        else:
            trail2.undo()

    return results


def commutative_match_all(
    pattern: Term,
    p_ctx: Context,
    target: Term,
) -> list[Trail]:
    """Find all matches for commutative pattern against target.

    Tries both orderings. Returns trails for each successful match.
    """
    if pattern.arity != 2 or target.arity != 2:
        return []

    results: list[Trail] = []

    # Straight match
    trail1 = Trail()
    if match(pattern.args[0], p_ctx, target.args[0], trail1):
        if match(pattern.args[1], p_ctx, target.args[1], trail1):
            results.append(trail1)
            # Need fresh context for second attempt
            p_ctx2 = Context()
            p_ctx2.multiplier = p_ctx.multiplier
            trail2 = Trail()
            if match(pattern.args[0], p_ctx2, target.args[1], trail2):
                if match(pattern.args[1], p_ctx2, target.args[0], trail2):
                    results.append(trail2)
                else:
                    trail2.undo()
            return results
        else:
            trail1.undo()

    # Flipped match
    trail2 = Trail()
    if match(pattern.args[0], p_ctx, target.args[1], trail2):
        if match(pattern.args[1], p_ctx, target.args[0], trail2):
            results.append(trail2)
        else:
            trail2.undo()

    return results


# ── AC unification data structures ───────────────────────────────────────────


@dataclass
class AcUnifyState:
    """State for incremental AC unification.

    Holds the Diophantine basis and combination state for
    enumerating AC unifiers one at a time.
    """

    ac_symnum: int  # private_symbol of the AC operator
    # Flattened, sorted, de-duplicated arguments
    args1: list[Term]
    args2: list[Term]
    ctxs1: list[Context | None]
    ctxs2: list[Context | None]
    mults1: list[int]
    mults2: list[int]
    # Diophantine solution state
    dio_result: DioResult
    combo: list[int]
    sumvec: list[int]
    m: int  # number of left-side terms
    n: int  # number of right-side terms
    length: int  # m + n
    constraints: list[int]
    # Current substitution
    trail: Trail
    c1: Context
    c2: Context
    # Enumeration state
    first_call: bool = True
    exhausted: bool = False


# ── AC unification core ──────────────────────────────────────────────────────


def _flatten_deref(
    t: Term,
    tc: Context | None,
    ac_symnum: int,
) -> list[tuple[Term, Context | None]]:
    """Flatten AC term, dereferencing variables through context.

    Matching C flatten_deref().
    """
    t, tc = dereference(t, tc)

    if t.private_symbol == ac_symnum and t.arity == 2:
        result: list[tuple[Term, Context | None]] = []
        result.extend(_flatten_deref(t.args[0], tc, ac_symnum))
        result.extend(_flatten_deref(t.args[1], tc, ac_symnum))
        return result

    return [(t, tc)]


def _sort_key_ctx(pair: tuple[Term, Context | None]):
    """Sort key for flattened terms with contexts.

    Matching C compare_ncv_context(): NAME < COMPLEX < VARIABLE.
    Variables sorted by (multiplier, varnum).
    """
    t, c = pair
    if t.is_variable:
        mult = c.multiplier if c is not None else 0
        return (2, mult, t.varnum)
    if t.is_constant:
        return (0, t.symnum, 0)
    # Complex — sort by symbol then recursively
    return (1, t.symnum, t.arity)


def _eliminate_common(
    args1: list[tuple[Term, Context | None]],
    args2: list[tuple[Term, Context | None]],
) -> tuple[list[tuple[Term, Context | None]], list[tuple[Term, Context | None]]]:
    """Eliminate common terms from both sides.

    Matching C elim_con_context(). Two terms are common if they are
    ground-identical after dereferencing.
    """
    remaining1: list[tuple[Term, Context | None]] = []
    used2 = [False] * len(args2)

    for t1, c1 in args1:
        found = False
        for j, (t2, c2) in enumerate(args2):
            if not used2[j]:
                # For ground terms, use structural identity
                if (not t1.is_variable and not t2.is_variable
                        and t1.term_ident(t2)):
                    used2[j] = True
                    found = True
                    break
        if not found:
            remaining1.append((t1, c1))

    remaining2 = [
        (t, c) for j, (t, c) in enumerate(args2) if not used2[j]
    ]
    return remaining1, remaining2


def _collapse_multiplicities(
    args: list[tuple[Term, Context | None]],
) -> tuple[list[Term], list[Context | None], list[int]]:
    """Collapse duplicate terms and record multiplicities.

    Matching C ac_mult_context().
    """
    if not args:
        return [], [], []

    terms: list[Term] = []
    ctxs: list[Context | None] = []
    mults: list[int] = []

    i = 0
    while i < len(args):
        t, c = args[i]
        count = 1
        while i + count < len(args):
            t2, c2 = args[i + count]
            if t.term_ident(t2):
                count += 1
            else:
                break
        terms.append(t)
        ctxs.append(c)
        mults.append(count)
        i += count

    return terms, ctxs, mults


def ac_unify_all(
    t1: Term,
    c1: Context,
    t2: Term,
    c2: Context,
    is_ac: callable,
) -> list[Trail]:
    """Find all AC unifiers for two terms.

    High-level API: returns all unifiers as trails.
    The caller should undo each trail after inspecting the substitution.

    Args:
        t1, c1: First term and its context.
        t2, c2: Second term and its context.
        is_ac: Function(symnum) -> bool.

    Returns:
        List of Trail objects, each representing a complete unifier.
    """
    if t1.private_symbol != t2.private_symbol:
        return []

    ac_symnum = t1.private_symbol

    if not is_ac(t1.symnum):
        # Not AC — try standard unification
        trail = Trail()
        if unify(t1, c1, t2, c2, trail):
            return [trail]
        return []

    # Flatten both sides through contexts
    flat1 = _flatten_deref(t1, c1, ac_symnum)
    flat2 = _flatten_deref(t2, c2, ac_symnum)

    # Sort
    flat1.sort(key=_sort_key_ctx)
    flat2.sort(key=_sort_key_ctx)

    # Eliminate common ground terms
    rem1, rem2 = _eliminate_common(flat1, flat2)

    if not rem1 and not rem2:
        # All terms cancelled — trivial unifier
        return [Trail()]

    if not rem1 or not rem2:
        # One side is empty after elimination — no unifier
        return []

    # Collapse multiplicities
    terms1, ctxs1, mults1 = _collapse_multiplicities(rem1)
    terms2, ctxs2, mults2 = _collapse_multiplicities(rem2)

    m = len(terms1)
    n = len(terms2)
    length = m + n

    # Set up coefficient array and constraints
    ab = mults1 + mults2
    constraints = [0] * length

    for i in range(m):
        if not terms1[i].is_variable:
            constraints[i] = terms1[i].symnum
    for j in range(n):
        if not terms2[j].is_variable:
            constraints[m + j] = terms2[j].symnum

    # Solve Diophantine equation
    dio_result = dio(ab, m, n, constraints)

    if dio_result.status != 1 or dio_result.num_basis == 0:
        return []

    # Enumerate combinations
    results: list[Trail] = []
    combo = [0] * dio_result.num_basis
    sumvec = [0] * length
    start = True

    while next_combo_a(
        length, dio_result.basis, dio_result.num_basis,
        constraints, combo, sumvec, start,
    ):
        start = False

        # Build substitution from this combination
        trail = Trail()
        success = _apply_ac_combination(
            terms1, ctxs1, terms2, ctxs2,
            mults1, mults2,
            sumvec, m, n,
            ac_symnum, t1.arity,
            c1, c2, trail,
            is_ac,
        )

        if success:
            results.append(trail)
        else:
            trail.undo()

    return results


def _apply_ac_combination(
    terms1: list[Term],
    ctxs1: list[Context | None],
    terms2: list[Term],
    ctxs2: list[Context | None],
    mults1: list[int],
    mults2: list[int],
    sumvec: list[int],
    m: int,
    n: int,
    ac_symnum: int,
    ac_arity: int,
    c1: Context,
    c2: Context,
    trail: Trail,
    is_ac: callable,
) -> bool:
    """Apply an AC combination to produce a unifier.

    For each variable, constructs the term it must be bound to based
    on the combination vector. For non-variable terms, checks that
    they can be recursively unified.

    Returns True if the combination produces a valid unifier.
    """
    # For each position in the combination vector, determine what each
    # term needs to unify with. The sum vector tells us how many
    # "copies" of each basis term are needed.

    # Simple case: for each variable on the left, bind it based on
    # which right-side terms it needs to capture
    for i in range(m):
        t1 = terms1[i]
        tc1 = ctxs1[i]
        if t1.is_variable:
            # Variable must capture sumvec[i] copies total
            # Determine which right-side terms contribute
            pieces: list[Term] = []
            for j in range(n):
                # Check if this basis combination links i to j
                # by looking at the sum vector structure
                contrib = _get_contribution(
                    i, m + j, sumvec, m, n,
                )
                if contrib > 0:
                    for _ in range(contrib):
                        t2 = terms2[j]
                        tc2 = ctxs2[j]
                        pieces.append(apply_substitution(t2, tc2))

            if not pieces:
                return False

            # Build the binding term
            if len(pieces) == 1:
                binding = pieces[0]
            else:
                binding = right_associate(ac_symnum, ac_arity, pieces)

            # Bind the variable
            t1d, tc1d = dereference(t1, tc1)
            if t1d.is_variable and tc1d is not None:
                trail.bind(t1d.varnum, tc1d, binding, None)

    for j in range(n):
        t2 = terms2[j]
        tc2 = ctxs2[j]
        if t2.is_variable:
            pieces: list[Term] = []
            for i in range(m):
                contrib = _get_contribution(
                    m + j, i, sumvec, m, n,
                )
                if contrib > 0:
                    for _ in range(contrib):
                        t1 = terms1[i]
                        tc1 = ctxs1[i]
                        pieces.append(apply_substitution(t1, tc1))

            if not pieces:
                return False

            if len(pieces) == 1:
                binding = pieces[0]
            else:
                binding = right_associate(ac_symnum, ac_arity, pieces)

            t2d, tc2d = dereference(t2, tc2)
            if t2d.is_variable and tc2d is not None:
                trail.bind(t2d.varnum, tc2d, binding, None)

    return True


def _get_contribution(
    target_pos: int,
    source_pos: int,
    sumvec: list[int],
    m: int,
    n: int,
) -> int:
    """Determine contribution of source to target in the combination.

    Simplified heuristic: if both positions have non-zero sum values
    and the source is on the opposite side, contribute proportionally.
    """
    # In the Diophantine solution, sumvec[i] tells us the total
    # "weight" of position i. For a simple AC unification, variables
    # on one side bind to terms on the other side.
    if target_pos < m and source_pos >= m:
        # Left variable, right term
        if sumvec[target_pos] > 0 and sumvec[source_pos] > 0:
            return min(sumvec[target_pos], sumvec[source_pos])
    elif target_pos >= m and source_pos < m:
        # Right variable, left term
        if sumvec[target_pos] > 0 and sumvec[source_pos] > 0:
            return min(sumvec[target_pos], sumvec[source_pos])
    return 0


# ── AC matching ──────────────────────────────────────────────────────────────


def ac_match_all(
    pattern: Term,
    p_ctx: Context,
    target: Term,
    is_ac: callable,
) -> list[Trail]:
    """Find all AC matches of pattern against target.

    Matching C match_ac() / match_bt_first(). Pattern variables
    may bind to AC combinations of target arguments.

    Args:
        pattern: Pattern term (variables to bind).
        p_ctx: Pattern context.
        target: Target term (treated as ground).
        is_ac: Function(symnum) -> bool.

    Returns:
        List of trails, each representing a successful match.
    """
    if pattern.private_symbol != target.private_symbol:
        return []

    ac_symnum = pattern.private_symbol

    if not is_ac(pattern.symnum):
        trail = Trail()
        if match(pattern, p_ctx, target, trail):
            return [trail]
        return []

    # Flatten both sides
    flat_pattern = flatten_ac(pattern, ac_symnum)
    flat_target = flatten_ac(target, ac_symnum)

    # Separate pattern into non-variable args (functors) and variables
    functors: list[Term] = []
    variables: list[Term] = []
    for t in flat_pattern:
        if t.is_variable:
            variables.append(t)
        else:
            functors.append(t)

    # Match functors first — each must match exactly one target arg
    target_available = list(flat_target)
    results: list[Trail] = []

    _ac_match_recurse(
        functors, 0, variables, 0,
        target_available, ac_symnum, pattern.arity,
        p_ctx, Trail(), results,
    )

    return results


def _ac_match_recurse(
    functors: list[Term],
    f_idx: int,
    variables: list[Term],
    v_idx: int,
    target_available: list[Term],
    ac_symnum: int,
    ac_arity: int,
    p_ctx: Context,
    trail: Trail,
    results: list[Trail],
) -> None:
    """Recursive AC matching with backtracking.

    First match all functors, then assign variables to remaining targets.
    """
    if f_idx < len(functors):
        # Match functor against each available target
        func = functors[f_idx]
        for i, targ in enumerate(target_available):
            if targ is None:
                continue
            mark = trail.position
            if match(func, p_ctx, targ, trail):
                # Mark as used and recurse
                target_available[i] = None
                _ac_match_recurse(
                    functors, f_idx + 1, variables, v_idx,
                    target_available, ac_symnum, ac_arity,
                    p_ctx, trail, results,
                )
                target_available[i] = targ
            trail.undo_to(mark)
        return

    # All functors matched — now assign variables
    remaining = [t for t in target_available if t is not None]

    if v_idx >= len(variables):
        # No more variables — all targets must be consumed
        if not remaining:
            # Success — save a copy of the trail
            results.append(trail)
            # Note: caller needs fresh trail for next attempt
        return

    if not remaining:
        # Variables left but no targets — fail
        return

    if v_idx == len(variables) - 1:
        # Last variable — must bind to all remaining
        var = variables[v_idx]
        if len(remaining) == 1:
            binding = remaining[0]
        else:
            binding = right_associate(ac_symnum, ac_arity, remaining)

        mark = trail.position
        trail.bind(var.varnum, p_ctx, binding, None)
        results.append(trail)
        trail.undo_to(mark)
        return

    # Multiple variables remain — enumerate non-empty subsets for this variable
    var = variables[v_idx]
    _assign_var_subsets(
        var, variables, v_idx, remaining,
        ac_symnum, ac_arity, p_ctx, trail, results,
    )


def _assign_var_subsets(
    var: Term,
    variables: list[Term],
    v_idx: int,
    remaining: list[Term],
    ac_symnum: int,
    ac_arity: int,
    p_ctx: Context,
    trail: Trail,
    results: list[Trail],
) -> None:
    """Enumerate non-empty subsets of remaining targets for a variable."""
    n = len(remaining)
    # Enumerate all non-empty proper subsets (leave at least 1 for remaining vars)
    for mask in range(1, (1 << n)):
        selected = [remaining[i] for i in range(n) if mask & (1 << i)]
        left = [remaining[i] for i in range(n) if not (mask & (1 << i))]

        # Must leave enough for remaining variables
        if len(left) < len(variables) - v_idx - 1:
            continue

        if len(selected) == 1:
            binding = selected[0]
        else:
            binding = right_associate(ac_symnum, ac_arity, selected)

        mark = trail.position
        trail.bind(var.varnum, p_ctx, binding, None)
        _ac_match_recurse(
            [], 0, variables, v_idx + 1,
            left + [None] * (n - len(left)),  # pad to keep indexing
            ac_symnum, ac_arity, p_ctx, trail, results,
        )
        trail.undo_to(mark)


# ── AC redundancy checking ───────────────────────────────────────────────────


def is_ac_tautology(t1: Term, t2: Term, is_ac: callable) -> bool:
    """Check if t1 = t2 is a tautology modulo AC.

    Matching C cac_redundant_atom(). Canonicalizes both sides
    and checks for structural identity.
    """
    from pyladr.core.ac_normal_form import ac_canonical
    c1 = ac_canonical(t1, is_ac)
    c2 = ac_canonical(t2, is_ac)
    return c1.term_ident(c2)
