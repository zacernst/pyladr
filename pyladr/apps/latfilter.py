"""Latfilter: filter lattice identities using Whitman's algorithm.

Modernized Python 3.13+ version of apps.src/latfilter.c.

Reads meet/join equations from stdin and outputs those that are
lattice identities. Uses Whitman's decision procedure for free lattices.

Usage:
    pylatfilter [x]

The optional 'x' argument inverts output (non-identities only).
"""

from __future__ import annotations

import sys

from pyladr.apps.cli_common import read_clause_stream
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term


def _is_meet(t: Term, meet_syms: set[int]) -> bool:
    """Check if term is a meet operation."""
    return t.is_complex and t.symnum in meet_syms


def _is_join(t: Term, join_syms: set[int]) -> bool:
    """Check if term is a join operation."""
    return t.is_complex and t.symnum in join_syms


def lattice_leq(
    s: Term, t: Term, meet_syms: set[int], join_syms: set[int]
) -> bool:
    """Check if s <= t in free lattice theory using Whitman's algorithm.

    Implements the algorithm from "Free Lattices" by Freese, Jezek, and Nation.
    """
    # (1) Both atomic (variable or constant): must be identical
    if (s.is_variable or s.is_constant) and (t.is_variable or t.is_constant):
        if s.is_variable and t.is_variable:
            return s.varnum == t.varnum
        if s.is_constant and t.is_constant:
            return s.symnum == t.symnum
        return False

    # (2) s = s1 v s2: need s1 <= t AND s2 <= t
    if _is_join(s, join_syms):
        return (lattice_leq(s.args[0], t, meet_syms, join_syms) and
                lattice_leq(s.args[1], t, meet_syms, join_syms))

    # (3) t = t1 ^ t2: need s <= t1 AND s <= t2
    if _is_meet(t, meet_syms):
        return (lattice_leq(s, t.args[0], meet_syms, join_syms) and
                lattice_leq(s, t.args[1], meet_syms, join_syms))

    # (4) s atomic, t = t1 v t2: need s <= t1 OR s <= t2
    if (s.is_variable or s.is_constant) and _is_join(t, join_syms):
        return (lattice_leq(s, t.args[0], meet_syms, join_syms) or
                lattice_leq(s, t.args[1], meet_syms, join_syms))

    # (5) s = s1 ^ s2, t atomic: need s1 <= t OR s2 <= t
    if _is_meet(s, meet_syms) and (t.is_variable or t.is_constant):
        return (lattice_leq(s.args[0], t, meet_syms, join_syms) or
                lattice_leq(s.args[1], t, meet_syms, join_syms))

    # (6) s = s1 ^ s2, t = t1 v t2: need any of 4 subcases
    if _is_meet(s, meet_syms) and _is_join(t, join_syms):
        return (lattice_leq(s, t.args[0], meet_syms, join_syms) or
                lattice_leq(s, t.args[1], meet_syms, join_syms) or
                lattice_leq(s.args[0], t, meet_syms, join_syms) or
                lattice_leq(s.args[1], t, meet_syms, join_syms))

    return False


def lattice_identity(
    atom: Term, eq_sym: int, meet_syms: set[int], join_syms: set[int]
) -> bool:
    """Check if an equation is a lattice identity (s = t iff s <= t and t <= s)."""
    if not atom.is_complex or atom.symnum != eq_sym or atom.arity != 2:
        return False
    return (lattice_leq(atom.args[0], atom.args[1], meet_syms, join_syms) and
            lattice_leq(atom.args[1], atom.args[0], meet_syms, join_syms))


def main(argv: list[str] | None = None) -> int:
    """Entry point for pylatfilter command."""
    args = argv if argv is not None else sys.argv[1:]

    output_non_identities = "x" in args

    symbol_table = SymbolTable()
    eq_sym = symbol_table.str_to_sn("=", 2)
    meet_syms = {
        symbol_table.str_to_sn("^", 2),
        symbol_table.str_to_sn("m", 2),
        symbol_table.str_to_sn("meet", 2),
    }
    join_syms = {
        symbol_table.str_to_sn("v", 2),
        symbol_table.str_to_sn("j", 2),
        symbol_table.str_to_sn("join", 2),
    }

    clauses = read_clause_stream(sys.stdin, symbol_table)

    checked = 0
    passed = 0

    for clause in clauses:
        if len(clause.literals) != 1 or not clause.literals[0].sign:
            continue

        atom = clause.literals[0].atom
        is_ident = lattice_identity(atom, eq_sym, meet_syms, join_syms)
        checked += 1

        if (not output_non_identities and is_ident) or \
           (output_non_identities and not is_ident):
            passed += 1
            # Output the clause
            from pyladr.apps.cli_common import format_clause_bare
            print(format_clause_bare(clause))

    suffix = " x" if output_non_identities else ""
    sys.stderr.write(
        f"latfilter{suffix}: checked {checked}, passed {passed}.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
