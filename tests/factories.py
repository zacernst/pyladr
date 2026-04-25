"""Shared test factory functions for constructing Term, Literal, and Clause objects.

Eliminates duplicated helper definitions across 30+ test files.
All functions use the public pyladr.core API (get_rigid_term, get_variable_term).
"""

from __future__ import annotations

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term, get_rigid_term, get_variable_term


# ── Term factories ────────────────────────────────────────────────────────────


def make_var(n: int) -> Term:
    """Create a variable term with variable number *n*."""
    return get_variable_term(n)


def make_const(symnum: int) -> Term:
    """Create a constant (0-arity rigid symbol) with symbol number *symnum*."""
    return get_rigid_term(symnum, 0)


def make_func(symnum: int, *args: Term) -> Term:
    """Create a function application with symbol number *symnum* and given args."""
    return get_rigid_term(symnum, len(args), args)


def make_term(symnum: int, *args: Term) -> Term:
    """Create a rigid term — constant if no args, function application otherwise."""
    if args:
        return get_rigid_term(symnum, len(args), args)
    return get_rigid_term(symnum, 0)


# ── Literal factories ────────────────────────────────────────────────────────


def make_literal(sign: bool, atom: Term) -> Literal:
    """Create a literal with the given sign and atom."""
    return Literal(sign=sign, atom=atom)


def make_pos_lit(atom: Term) -> Literal:
    """Create a positive literal."""
    return Literal(sign=True, atom=atom)


def make_neg_lit(atom: Term) -> Literal:
    """Create a negative literal."""
    return Literal(sign=False, atom=atom)


# ── Clause factories ─────────────────────────────────────────────────────────


def make_clause(
    *literals: Literal,
    weight: float = 0.0,
    clause_id: int = 0,
) -> Clause:
    """Create a clause from pre-built Literal objects."""
    c = Clause(literals=literals, weight=weight, id=clause_id)
    return c


def make_clause_from_atoms(
    *atoms: Term,
    signs: tuple[bool, ...] | None = None,
    clause_id: int = 0,
) -> Clause:
    """Create a clause from atom terms with optional sign tuple.

    If *signs* is None, all literals are positive.
    """
    if signs is None:
        signs = (True,) * len(atoms)
    lits = tuple(Literal(sign=s, atom=a) for s, a in zip(signs, atoms))
    return Clause(literals=lits, id=clause_id)


def make_unit_eq(a: Term, b: Term, clause_id: int = 0) -> Clause:
    """Create a positive unit equality clause: a = b.

    Uses symnum=1 for the equality predicate (convention).
    """
    eq = make_func(1, a, b)
    return make_clause(make_pos_lit(eq), clause_id=clause_id)
