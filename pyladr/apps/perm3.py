"""Perm3: filter clauses modulo ternary argument permutations.

Matches C apps.src/perm3.c — reads a stream of equations and filters
out those that are equivalent under all 6 permutations of ternary
function arguments.

Usage:
    pyperm3 < input > output
    pyperm3 -f input_file
"""

from __future__ import annotations

import itertools
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
from pyladr.apps.mirror_flip import clause_ident
from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term
from pyladr.inference.resolution import renumber_variables


def perm3_term(t: Term, p: tuple[int, int, int]) -> Term:
    """Permute arguments of ternary functions in a term.

    Matches C perm3_term(): for arity-3 functions, rearrange
    arguments according to permutation p.
    """
    if t.arity == 3:
        alpha = t.args[0]
        beta = t.args[1]
        gamma = t.args[2]
        orig = (alpha, beta, gamma)
        new_args_list = [Term(private_symbol=0, arity=0)] * 3  # placeholders
        new_args_list[p[0]] = perm3_term(orig[0], p)
        new_args_list[p[1]] = perm3_term(orig[1], p)
        new_args_list[p[2]] = perm3_term(orig[2], p)
        return Term(
            private_symbol=t.private_symbol,
            arity=3,
            args=tuple(new_args_list),
        )
    return t


def perm3(c: Clause, p: tuple[int, int, int]) -> Clause:
    """Apply ternary permutation to a unit equation clause.

    Matches C perm3() — applies permutation to both sides of equation.
    """
    m = copy_clause(c)
    atom = m.literals[0].atom
    new_atom = Term(
        private_symbol=atom.private_symbol,
        arity=atom.arity,
        args=(perm3_term(atom.args[0], p), perm3_term(atom.args[1], p)),
    )
    new_lit = Literal(sign=m.literals[0].sign, atom=new_atom)
    result = Clause(literals=(new_lit,), id=m.id, justification=m.justification)
    return renumber_variables(result)


# All 6 permutations of 3 elements, matching C perm3.c
ALL_PERMS: list[tuple[int, int, int]] = [
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]


def contains_perm3(c: Clause, kept: list[Clause]) -> bool:
    """Check if clause is equivalent to any kept clause under ternary permutations.

    Matches C contains_perm3(): generates all 6 permutation variants
    and checks against the kept list.
    """
    variants = [perm3(c, p) for p in ALL_PERMS]

    for k in kept:
        for v in variants:
            if clause_ident(k.literals, v.literals):
                return True
    return False


def main(argv: list[str] | None = None) -> int:
    """Entry point for pyperm3 command."""
    parser = make_base_parser(
        "pyperm3",
        "Filter equations modulo ternary argument permutations.",
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
        if not contains_perm3(c, kept):
            number_kept += 1
            kept.append(c)
            print(format_clause_bare(c))

    report_stats("perm3", start_time, read=number_read, kept=number_kept)
    return 0


if __name__ == "__main__":
    sys.exit(main())
