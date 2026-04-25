"""Binary resolution and factoring matching C LADR resolve.c.

Implements:
- binary_resolve: resolve two clauses on complementary literals
- all_binary_resolvents: generate all resolvents of a clause against a set
- factor: generate all factors of a clause
- merge_literals: remove duplicate literals

The resolution algorithm:
1. Find complementary literals in two clauses (one positive, one negative)
2. Unify the atoms of the complementary literals
3. Apply the unifier to all remaining literals from both clauses
4. Build the resolvent as the disjunction of instantiated remaining literals
"""

from __future__ import annotations

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.substitution import (
    Context,
    Trail,
    apply_substitution,
    unify,
)
from pyladr.core.term import Term


def binary_resolve(
    c1: Clause,
    lit1_idx: int,
    c2: Clause,
    lit2_idx: int,
) -> Clause | None:
    """Resolve c1 and c2 on specified complementary literals.

    Matches C binary_resolvent() behavior:
    1. Check literals are complementary (opposite sign, unifiable atoms)
    2. Apply MGU to remaining literals from both clauses
    3. Build resolvent with proper justification

    Args:
        c1: First clause (nucleus).
        lit1_idx: Index of literal in c1 to resolve on.
        c2: Second clause (satellite).
        lit2_idx: Index of literal in c2 to resolve on.

    Returns:
        Resolvent clause, or None if unification fails.
    """
    from pyladr.cpp_backend import is_enabled
    if is_enabled():
        try:
            from pyladr._pyladr_core import binary_resolve_lits
            from pyladr.cpp_utils import py_term_to_cpp, cpp_term_to_py
            cpp_lits1 = [(lit.sign, py_term_to_cpp(lit.atom)) for lit in c1.literals]
            cpp_lits2 = [(lit.sign, py_term_to_cpp(lit.atom)) for lit in c2.literals]
            cpp_result = binary_resolve_lits(cpp_lits1, lit1_idx, cpp_lits2, lit2_idx)
            if cpp_result is None:
                return None
            new_lits = tuple(
                Literal(sign=sign, atom=cpp_term_to_py(atom))
                for sign, atom in cpp_result
            )
            just = Justification(
                just_type=JustType.BINARY_RES,
                clause_ids=(c1.id, c2.id),
            )
            return Clause(literals=new_lits, justification=(just,))
        except Exception:
            pass  # fall through to pure Python

    l1 = c1.literals[lit1_idx]
    l2 = c2.literals[lit2_idx]

    # Must be complementary (opposite signs)
    if l1.sign == l2.sign:
        return None

    # Create fresh contexts for standardization apart (C two-context approach)
    ctx1 = Context()
    ctx2 = Context()
    trail = Trail()

    # Unify the atoms
    if not unify(l1.atom, ctx1, l2.atom, ctx2, trail):
        trail.undo()
        return None

    # Build resolvent: apply substitution to remaining literals
    new_lits: list[Literal] = []

    # Add remaining literals from c1 (skip the resolved literal)
    for i, lit in enumerate(c1.literals):
        if i == lit1_idx:
            continue
        new_atom = apply_substitution(lit.atom, ctx1)
        new_lits.append(Literal(sign=lit.sign, atom=new_atom))

    # Add remaining literals from c2 (skip the resolved literal)
    for i, lit in enumerate(c2.literals):
        if i == lit2_idx:
            continue
        new_atom = apply_substitution(lit.atom, ctx2)
        new_lits.append(Literal(sign=lit.sign, atom=new_atom))

    trail.undo()

    # Build justification
    just = Justification(
        just_type=JustType.BINARY_RES,
        clause_ids=(c1.id, c2.id),
    )

    return Clause(
        literals=tuple(new_lits),
        justification=(just,),
    )


def all_binary_resolvents(
    c1: Clause,
    c2: Clause,
) -> list[Clause]:
    """Generate all binary resolvents between c1 and c2.

    Tries every pair of complementary literals, matching the C approach
    of iterating through all literal pairs.

    Args:
        c1: First clause.
        c2: Second clause.

    Returns:
        List of resolvent clauses (may be empty).
    """
    from pyladr.cpp_backend import is_enabled
    if is_enabled():
        try:
            from pyladr._pyladr_core import all_binary_resolvents_lits
            from pyladr.cpp_utils import py_term_to_cpp, cpp_term_to_py
            cpp_lits1 = [(lit.sign, py_term_to_cpp(lit.atom)) for lit in c1.literals]
            cpp_lits2 = [(lit.sign, py_term_to_cpp(lit.atom)) for lit in c2.literals]
            cpp_results = all_binary_resolvents_lits(cpp_lits1, cpp_lits2)
            results = []
            for cpp_result in cpp_results:
                new_lits = tuple(
                    Literal(sign=sign, atom=cpp_term_to_py(atom))
                    for sign, atom in cpp_result
                )
                just = Justification(
                    just_type=JustType.BINARY_RES,
                    clause_ids=(c1.id, c2.id),
                )
                results.append(Clause(literals=new_lits, justification=(just,)))
            return results
        except Exception:
            pass  # fall through to pure Python

    resolvents: list[Clause] = []
    for i, l1 in enumerate(c1.literals):
        for j, l2 in enumerate(c2.literals):
            if l1.sign != l2.sign:
                # Fast symbol pre-check: rigid atoms with different symbols cannot unify
                atom1, atom2 = l1.atom, l2.atom
                if (atom1.private_symbol < 0 and atom2.private_symbol < 0
                        and atom1.private_symbol != atom2.private_symbol):
                    continue
                resolvent = binary_resolve(c1, i, c2, j)
                if resolvent is not None:
                    resolvents.append(resolvent)
    return resolvents


def factor(clause: Clause) -> list[Clause]:
    """Generate all factors of a clause.

    Factoring resolves a clause with itself: if two literals with the
    same sign have unifiable atoms, merge them into one.

    Matches C binary_factors() behavior.

    Args:
        clause: The clause to factor.

    Returns:
        List of factor clauses (may be empty).
    """
    from pyladr.cpp_backend import is_enabled
    if is_enabled():
        try:
            from pyladr._pyladr_core import factor_lits
            from pyladr.cpp_utils import py_term_to_cpp, cpp_term_to_py
            cpp_lits = [(lit.sign, py_term_to_cpp(lit.atom)) for lit in clause.literals]
            cpp_results = factor_lits(cpp_lits)
            results = []
            for cpp_result in cpp_results:
                new_lits = tuple(
                    Literal(sign=sign, atom=cpp_term_to_py(atom))
                    for sign, atom in cpp_result
                )
                just = Justification(
                    just_type=JustType.FACTOR,
                    clause_ids=(clause.id,),
                )
                results.append(Clause(literals=new_lits, justification=(just,)))
            return results
        except Exception:
            pass  # fall through to pure Python

    factors: list[Clause] = []

    for i in range(len(clause.literals)):
        for j in range(i + 1, len(clause.literals)):
            li = clause.literals[i]
            lj = clause.literals[j]

            # Must have same sign
            if li.sign != lj.sign:
                continue

            ctx = Context()
            trail = Trail()

            # Unify the atoms of the two same-sign literals
            if unify(li.atom, ctx, lj.atom, ctx, trail):
                # Build factor: apply substitution, remove literal j
                new_lits: list[Literal] = []
                for k, lit in enumerate(clause.literals):
                    if k == j:
                        continue  # remove the second of the unified pair
                    new_atom = apply_substitution(lit.atom, ctx)
                    new_lits.append(Literal(sign=lit.sign, atom=new_atom))

                just = Justification(
                    just_type=JustType.FACTOR,
                    clause_ids=(clause.id,),
                )

                factors.append(
                    Clause(
                        literals=tuple(new_lits),
                        justification=(just,),
                    )
                )

            trail.undo()

    return factors


def merge_literals(clause: Clause) -> Clause:
    """Remove duplicate literals from a clause.

    Matches C merge_literals(): if two literals have the same sign
    and structurally identical atoms, keep only one.

    Args:
        clause: Input clause.

    Returns:
        Clause with duplicates removed (may be same object if no change).
    """
    if len(clause.literals) <= 1:
        return clause

    kept: list[Literal] = []
    for lit in clause.literals:
        is_dup = False
        for existing in kept:
            if lit.sign == existing.sign and lit.atom.term_ident(existing.atom):
                is_dup = True
                break
        if not is_dup:
            kept.append(lit)

    if len(kept) == len(clause.literals):
        return clause

    return Clause(
        literals=tuple(kept),
        id=clause.id,
        weight=clause.weight,
        justification=clause.justification,
        is_formula=clause.is_formula,
    )


def is_tautology(clause: Clause) -> bool:
    """Check if a clause is a tautology (contains complementary literals).

    A clause is a tautology if it contains both P and ~P for some atom P.

    Matches C tautology() check.
    """
    for i, li in enumerate(clause.literals):
        for j in range(i + 1, len(clause.literals)):
            lj = clause.literals[j]
            if li.complementary(lj):
                return True
    return False


def renumber_variables(clause: Clause) -> Clause:
    """Renumber clause variables to start from 0 consecutively.

    Matches C renumber_variables() / clause_set_variables().
    After resolution, variable numbers may have gaps or be large
    (due to multiplier-based standardization apart). This renumbers
    them to v0, v1, v2, ... in order of first occurrence.
    """
    var_map: dict[int, int] = {}
    next_var = 0

    def _remap_term(t: Term) -> Term:
        nonlocal next_var
        if t.is_variable:
            from pyladr.core.term import get_variable_term

            vn = t.varnum
            if vn not in var_map:
                var_map[vn] = next_var
                next_var += 1
            return get_variable_term(var_map[vn])
        if t.is_constant:
            return t
        new_args = tuple(_remap_term(a) for a in t.args)
        if new_args == t.args:
            return t
        return Term(private_symbol=t.private_symbol, arity=t.arity, args=new_args)

    new_lits = []
    for lit in clause.literals:
        new_atom = _remap_term(lit.atom)
        new_lits.append(Literal(sign=lit.sign, atom=new_atom))

    return Clause(
        literals=tuple(new_lits),
        id=clause.id,
        weight=clause.weight,
        justification=clause.justification,
    )
