"""Paramodulation inference rule matching C LADR paramod.c / parautil.c.

Implements:
- orient_equalities: orient equation literals based on term ordering
- paramodulate: build a paramodulant from two parent clauses
- para_from_into: generate all paramodulants between two clauses
- para_into_terms: recursive descent into term structure with position tracking

Paramodulation replaces equals-for-equals:
  Given from_clause containing alpha=beta and into_clause containing t[alpha'],
  where alpha unifies with alpha', produce the clause with t[beta'] substituted.

Position vectors use 1-indexed tuples matching C Ilist convention:
  (literal_number, argument_of_eq, subterm_path...)
"""

from __future__ import annotations

from pyladr.core.clause import Clause, Justification, JustType, Literal, ParaJust
from pyladr.core.substitution import (
    Context,
    Trail,
    apply_substitution,
    apply_substitute_at_pos,
    unify,
)
from pyladr.core.symbol import EQ_SYM, SymbolTable
from pyladr.core.term import Term
from pyladr.ordering.base import Ordertype
from pyladr.ordering.termorder import term_order


# ── Equality detection ────────────────────────────────────────────────────────


def is_eq_atom(atom: Term, symbol_table: SymbolTable) -> bool:
    """Check if a term is an equality atom (= with arity 2).

    Matches C eq_term(). Checks both the arity and that the
    symbol is actually the equality predicate.
    """
    if not atom.is_complex or atom.arity != 2:
        return False
    try:
        name = symbol_table.sn_to_str(atom.symnum)
        return name == EQ_SYM
    except KeyError:
        return False


def pos_eq(lit: Literal, symbol_table: SymbolTable) -> bool:
    """Positive equality literal. Matches C pos_eq()."""
    return lit.sign and is_eq_atom(lit.atom, symbol_table)


def neg_eq(lit: Literal, symbol_table: SymbolTable) -> bool:
    """Negative equality literal. Matches C neg_eq()."""
    return not lit.sign and is_eq_atom(lit.atom, symbol_table)


# ── Equation orientation ──────────────────────────────────────────────────────

# Orientation flags stored as module-level sets of Term objects.
# We store the atoms by structural value (not id) because:
# 1. Term is frozen with __hash__/__eq__ based on content — safe for set membership
# 2. Using id(atom) would cause spurious hits after GC recycles memory addresses
#    (Python reuses the same address for a new object, so id(new_atom) == id(old_atom))

_oriented_eqs: set[Term] = set()  # oriented equality atoms (LHS > RHS)
_renamable_flips: set[Term] = set()  # renamable-flip equality atoms


def reset_orientation_state() -> None:
    """Clear all orientation tracking state between proof searches.

    Module-level sets accumulate across searches if not cleared.
    Call this at the start of each GivenClauseSearch.run() to prevent
    stale orientation marks from affecting unrelated problems.
    """
    _oriented_eqs.clear()
    _renamable_flips.clear()


def mark_oriented_eq(atom: Term) -> None:
    """Mark an equality atom as oriented (left > right)."""
    _oriented_eqs.add(atom)


def is_oriented_eq(atom: Term) -> bool:
    """Check if an equality atom is oriented."""
    return atom in _oriented_eqs


def mark_renamable_flip(atom: Term) -> None:
    """Mark an equality as a renamable flip (symmetric under renaming)."""
    _renamable_flips.add(atom)


def is_renamable_flip(atom: Term) -> bool:
    """Check if an equality is a renamable flip."""
    return atom in _renamable_flips


def flip_eq(atom: Term) -> Term:
    """Swap left and right sides of an equality. Matches C flip_eq().

    Returns a new term with args swapped (terms are immutable).
    """
    assert atom.arity == 2
    return Term(
        private_symbol=atom.private_symbol,
        arity=2,
        args=(atom.args[1], atom.args[0]),
    )


def orient_equalities(clause: Clause, symbol_table: SymbolTable) -> Clause:
    """Orient equality literals based on term ordering.

    Matches C orient_equalities() from parautil.c.
    For each equality literal:
    - If left > right by term_order: mark oriented, keep as-is
    - If right > left: flip sides, mark oriented
    - If incomparable but symmetric under renaming: mark renamable_flip
    - Otherwise: leave unoriented (paramodulate from both sides)

    Returns a potentially modified clause with flipped equalities.
    """
    new_lits = list(clause.literals)
    changed = False

    for i, lit in enumerate(new_lits):
        if not is_eq_atom(lit.atom, symbol_table):
            continue

        alpha = lit.atom.args[0]  # left side
        beta = lit.atom.args[1]   # right side

        ord_result = term_order(alpha, beta, symbol_table)

        if ord_result == Ordertype.GREATER_THAN:
            # left > right: keep, mark oriented
            mark_oriented_eq(lit.atom)
        elif ord_result == Ordertype.LESS_THAN:
            # right > left: flip and mark oriented
            flipped_atom = flip_eq(lit.atom)
            new_lits[i] = Literal(sign=lit.sign, atom=flipped_atom)
            mark_oriented_eq(flipped_atom)
            changed = True
        elif ord_result == Ordertype.SAME_AS:
            # Identical sides — mark as renamable flip to avoid redundancy
            mark_renamable_flip(lit.atom)
        else:
            # NOT_COMPARABLE — check if renamable flip
            # Two terms are renamable-flip if they become identical
            # after independent variable renumbering. Simplified check:
            # if the terms are variants of each other.
            from pyladr.core.substitution import variant, Context as Ctx, Trail as Tr

            c_tmp = Ctx()
            tr_tmp = Tr()
            if variant(alpha, c_tmp, beta, tr_tmp):
                mark_renamable_flip(lit.atom)
            tr_tmp.undo()

    if changed:
        return Clause(
            literals=tuple(new_lits),
            id=clause.id,
            weight=clause.weight,
            justification=clause.justification,
            is_formula=clause.is_formula,
        )
    return clause


# ── Paramodulation control ────────────────────────────────────────────────────


def para_from_right(atom: Term) -> bool:
    """Check if we should try paramodulating from the right side of an equality.

    Matches C para_from_right():
    - If oriented: only from left (return False)
    - If renamable_flip: don't do both sides (return False)
    - Otherwise: allow both sides (return True)
    """
    if is_oriented_eq(atom):
        return False
    if is_renamable_flip(atom):
        return False
    return True


# ── Core paramodulation ──────────────────────────────────────────────────────


def paramodulate(
    from_clause: Clause,
    from_lit: Literal,
    from_side: int,
    from_subst: Context,
    into_clause: Clause,
    into_lit: Literal,
    into_pos: tuple[int, ...],
    into_subst: Context,
    symbol_table: SymbolTable,
) -> Clause:
    """Build a paramodulant from two parent clauses.

    Matches C paramodulate() from paramod.c.

    Given from_clause containing from_lit (an equality alpha=beta),
    and into_clause containing into_lit with a subterm at into_pos
    that unifies with alpha (from from_side of the equality):

    1. beta = the OTHER side of the equality from from_lit
    2. Apply from_subst to all from_clause literals except from_lit
    3. For into_clause literals except into_lit: apply into_subst
    4. For into_lit: apply substitution replacing the subterm at into_pos with beta

    Args:
        from_clause: Clause containing the equality
        from_lit: The equality literal (alpha = beta)
        from_side: 0 for left side, 1 for right side (which side alpha is on)
        from_subst: Context for from_clause variables
        into_clause: Clause being paramodulated into
        into_lit: The literal being modified
        into_pos: Position within into_lit's atom where substitution occurs
                  (1-indexed, relative to the atom, NOT including literal index)
        into_subst: Context for into_clause variables
        symbol_table: Symbol table for equality detection

    Returns:
        The paramodulant clause.
    """
    # beta is the OTHER side of the equality
    beta = from_lit.atom.args[1] if from_side == 0 else from_lit.atom.args[0]

    new_lits: list[Literal] = []

    # Add all from_clause literals except from_lit (applied with from_subst)
    for lit in from_clause.literals:
        if lit is not from_lit:
            new_atom = apply_substitution(lit.atom, from_subst)
            new_lits.append(Literal(sign=lit.sign, atom=new_atom))

    # Add all into_clause literals
    for lit in into_clause.literals:
        if lit is not into_lit:
            # Normal literals: apply into_subst
            new_atom = apply_substitution(lit.atom, into_subst)
            new_lits.append(Literal(sign=lit.sign, atom=new_atom))
        else:
            # The modified literal: replace subterm at into_pos with beta
            new_atom = apply_substitute_at_pos(
                lit.atom, beta, from_subst, into_pos, into_subst
            )
            new_lits.append(Literal(sign=lit.sign, atom=new_atom))

    # Build position vectors for justification (1-indexed, matching C)
    from_lit_idx = _literal_number(from_clause, from_lit)
    into_lit_idx = _literal_number(into_clause, into_lit)

    from_pos = (from_lit_idx, from_side + 1)  # C counts arg from 1
    full_into_pos = (into_lit_idx, *into_pos)

    para_just = ParaJust(
        from_id=from_clause.id,
        into_id=into_clause.id,
        from_pos=from_pos,
        into_pos=full_into_pos,
    )

    just = Justification(
        just_type=JustType.PARA,
        para=para_just,
    )

    return Clause(
        literals=tuple(new_lits),
        justification=(just,),
    )


def _literal_number(clause: Clause, lit: Literal) -> int:
    """Get 1-indexed position of a literal in a clause. Matches C literal_number()."""
    for i, l in enumerate(clause.literals):
        if l is lit:
            return i + 1
    raise ValueError("Literal not found in clause")


# ── Recursive descent into terms ─────────────────────────────────────────────


def _para_into_term(
    from_lit: Literal,
    from_side: int,
    from_subst: Context,
    from_clause: Clause,
    into_clause: Clause,
    into_lit: Literal,
    into_term: Term,
    into_subst: Context,
    into_pos: list[int],
    skip_top: bool,
    symbol_table: SymbolTable,
    para_into_vars: bool,
) -> list[Clause]:
    """Recursively traverse into_term, trying paramodulation at each subterm.

    Matches C para_into() from paramod.c.

    Builds position vectors as it descends, tries unifying at each node.

    Args:
        from_lit: The equality literal we're paramodulating from
        from_side: Which side of the equality (0=left, 1=right)
        from_subst: Context for from_clause
        from_clause: The clause containing the equality
        into_clause: The clause being paramodulated into
        into_lit: The specific literal being modified
        into_term: Current subterm being examined
        into_subst: Context for into_clause
        into_pos: Current position path (1-indexed, mutable for efficiency)
        skip_top: Whether to skip unification at the current node
        symbol_table: Symbol table
        para_into_vars: Whether to allow paramodulating into variable positions

    Returns:
        List of paramodulant clauses generated.
    """
    results: list[Clause] = []

    # Skip variables unless para_into_vars is set
    if into_term.is_variable and not para_into_vars:
        return results

    # Recurse into subterms first (for complex terms)
    if into_term.is_complex:
        for i in range(into_term.arity):
            into_pos.append(i + 1)  # 1-indexed
            results.extend(
                _para_into_term(
                    from_lit, from_side, from_subst, from_clause,
                    into_clause, into_lit, into_term.args[i], into_subst,
                    into_pos, False, symbol_table, para_into_vars,
                )
            )
            into_pos.pop()

    # Try unification at current position (unless skipped)
    if not skip_top:
        alpha = from_lit.atom.args[from_side]
        trail = Trail()
        if unify(alpha, from_subst, into_term, into_subst, trail):
            # Build the paramodulant
            pos_tuple = tuple(into_pos)
            p = paramodulate(
                from_clause, from_lit, from_side, from_subst,
                into_clause, into_lit, pos_tuple, into_subst,
                symbol_table,
            )
            results.append(p)
            trail.undo()
        else:
            trail.undo()

    return results


def _para_into_literal(
    from_lit: Literal,
    from_side: int,
    from_subst: Context,
    from_clause: Clause,
    into_clause: Clause,
    into_lit: Literal,
    into_subst: Context,
    check_top: bool,
    symbol_table: SymbolTable,
    para_into_vars: bool,
) -> list[Clause]:
    """Try paramodulating into all subterms of a literal.

    Matches C para_into_lit() from paramod.c.

    For each argument of the literal's atom, recursively descend
    and try unification. The check_top parameter controls whether
    we skip the top of equality arguments to avoid redundancy.
    """
    results: list[Clause] = []
    into_atom = into_lit.atom

    is_positive_eq = pos_eq(into_lit, symbol_table)

    for i in range(into_atom.arity):
        # Determine whether to skip the top of this argument
        # When check_top is True and into_lit is a positive equality:
        # - Skip left side (i==0) always
        # - Skip right side (i==1) if we wouldn't paramodulate from right
        skip_top = (
            check_top
            and is_positive_eq
            and (i == 0 or (i == 1 and not para_from_right(into_lit.atom)))
        )

        into_pos: list[int] = [i + 1]  # 1-indexed argument position
        results.extend(
            _para_into_term(
                from_lit, from_side, from_subst, from_clause,
                into_clause, into_lit, into_atom.args[i], into_subst,
                into_pos, skip_top, symbol_table, para_into_vars,
            )
        )

    return results


# ── Top-level paramodulation between two clauses ─────────────────────────────


def para_from_into(
    from_clause: Clause,
    into_clause: Clause,
    check_top: bool,
    symbol_table: SymbolTable,
    para_into_vars: bool = False,
) -> list[Clause]:
    """Generate all paramodulants between from_clause and into_clause.

    Matches C para_from_into() from paramod.c.

    Iterates through all positive equality literals in from_clause,
    all literals in into_clause, and tries paramodulating from each
    side of the equality into each subterm of each into_lit.

    Args:
        from_clause: Clause containing equality to paramodulate from
        into_clause: Clause to paramodulate into
        check_top: If True, skip paramodulating into top of equalities
                   (avoids redundant inferences when used with given=into)
        symbol_table: Symbol table for equality detection
        para_into_vars: Whether to allow paramodulating into variables

    Returns:
        List of all paramodulant clauses generated.
    """
    results: list[Clause] = []

    for from_lit in from_clause.literals:
        # Must be a positive equality
        if not pos_eq(from_lit, symbol_table):
            continue

        for into_lit in into_clause.literals:
            # Create fresh contexts for each (from_lit, into_lit) pair
            from_subst = Context()
            into_subst = Context()

            # Try paramodulating from left side (arg 0)
            results.extend(
                _para_into_literal(
                    from_lit, 0, from_subst, from_clause,
                    into_clause, into_lit, into_subst,
                    check_top, symbol_table, para_into_vars,
                )
            )

            # Try right side (arg 1) if allowed
            if para_from_right(from_lit.atom):
                # Need fresh contexts for independent unification
                from_subst2 = Context()
                into_subst2 = Context()
                results.extend(
                    _para_into_literal(
                        from_lit, 1, from_subst2, from_clause,
                        into_clause, into_lit, into_subst2,
                        check_top, symbol_table, para_into_vars,
                    )
                )

    return results
