"""Mirror-flip: filter clauses modulo mirror and flip equivalence.

Matches C apps.src/mirror-flip.c — reads a stream of equations and filters
out those that are equivalent under mirror (reverse all binary/ternary args)
and flip (swap equation sides) operations.

For binary ops: mirror(f(a,b)) = f(mirror(b), mirror(a))
For ternary ops: mirror(f(a,b,c)) = f(mirror(c), mirror(b), mirror(a))
Flip swaps the two sides of an equation.

Usage:
    pymirror-flip < input > output
    pymirror-flip -f input_file
"""

from __future__ import annotations

import sys
import time

from pyladr.apps.cli_common import (
    copy_clause,
    format_clause_bare,
    make_base_parser,
    open_input,
    read_clause_stream,
    report_stats,
)
from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term
from pyladr.inference.resolution import renumber_variables


def mirror_term(t: Term) -> Term:
    """Recursively reverse arguments of binary and ternary functions.

    Matches C mirror_term():
    - arity 2: swap args and recurse
    - arity 3: reverse all three args and recurse
    """
    if t.arity == 2:
        alpha = t.args[0]
        beta = t.args[1]
        return Term(
            private_symbol=t.private_symbol,
            arity=2,
            args=(mirror_term(beta), mirror_term(alpha)),
        )
    elif t.arity == 3:
        alpha = t.args[0]
        beta = t.args[1]
        gamma = t.args[2]
        return Term(
            private_symbol=t.private_symbol,
            arity=3,
            args=(mirror_term(gamma), mirror_term(beta), mirror_term(alpha)),
        )
    return t


def mirror(c: Clause) -> Clause:
    """Mirror a unit equation clause: apply mirror_term to both sides.

    Matches C mirror() — assumes unit equality literal.
    """
    m = copy_clause(c)
    atom = m.literals[0].atom
    new_atom = Term(
        private_symbol=atom.private_symbol,
        arity=atom.arity,
        args=(mirror_term(atom.args[0]), mirror_term(atom.args[1])),
    )
    new_lit = Literal(sign=m.literals[0].sign, atom=new_atom)
    result = Clause(literals=(new_lit,), id=m.id, justification=m.justification)
    return renumber_variables(result)


def flip(c: Clause) -> Clause:
    """Flip a unit equation clause: swap both sides of the equation.

    Matches C flip() — assumes unit equality literal.
    """
    f = copy_clause(c)
    atom = f.literals[0].atom
    new_atom = Term(
        private_symbol=atom.private_symbol,
        arity=atom.arity,
        args=(atom.args[1], atom.args[0]),
    )
    new_lit = Literal(sign=f.literals[0].sign, atom=new_atom)
    result = Clause(literals=(new_lit,), id=f.id, justification=f.justification)
    return renumber_variables(result)


def clause_ident(lits1: tuple[Literal, ...], lits2: tuple[Literal, ...]) -> bool:
    """Check if two literal tuples are structurally identical.

    Matches C clause_ident().
    """
    if len(lits1) != len(lits2):
        return False
    return all(
        l1.sign == l2.sign and l1.atom.term_ident(l2.atom)
        for l1, l2 in zip(lits1, lits2, strict=True)
    )


def contains_mirror_flip(c: Clause, kept: list[Clause]) -> bool:
    """Check if clause is equivalent to any kept clause under mirror/flip.

    Matches C contains_mirror_flip(): generates identity, flip, mirror,
    and flip(mirror) variants and checks against kept list.
    """
    f = flip(c)
    m = mirror(c)
    fm = flip(m)

    for k in kept:
        if (
            clause_ident(k.literals, c.literals)
            or clause_ident(k.literals, f.literals)
            or clause_ident(k.literals, m.literals)
            or clause_ident(k.literals, fm.literals)
        ):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    """Entry point for pymirror-flip command."""
    parser = make_base_parser(
        "pymirror-flip",
        "Filter equations modulo mirror and flip equivalence.",
    )
    parser.add_argument(
        "flags",
        nargs="*",
        help="Optional flags: help",
    )

    args = parser.parse_args(argv)
    if "help" in (args.flags or []):
        parser.print_help()
        return 1

    start_time = time.time()
    fin = open_input(args)
    try:
        clauses = read_clause_stream(fin)
    finally:
        if fin is not sys.stdin:
            fin.close()

    kept: list[Clause] = []
    number_read = 0
    number_kept = 0

    for c in clauses:
        number_read += 1
        if not contains_mirror_flip(c, kept):
            number_kept += 1
            kept.append(c)
            print(format_clause_bare(c))

    report_stats("mirror-flip", start_time, read=number_read, kept=number_kept)
    return 0


if __name__ == "__main__":
    sys.exit(main())
