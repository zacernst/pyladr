"""Complex: evaluate syntactic complexity of terms.

Modernized Python 3.13+ version of apps.src/complex.c.

Reads terms/equations from stdin and computes a quadratic complexity
measure based on subterm overlap analysis.

Usage:
    pycomplex
"""

from __future__ import annotations

import sys

from pyladr.apps.cli_common import read_clause_stream
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term


def _term_size(t: Term) -> int:
    """Count total nodes in term."""
    if t.is_variable or t.is_constant:
        return 1
    return 1 + sum(_term_size(a) for a in t.args)


def _collect_subterms(t: Term) -> list[Term]:
    """Collect all subterms (including t itself)."""
    result = [t]
    if t.is_complex:
        for a in t.args:
            result.extend(_collect_subterms(a))
    return result


def _term_match_score(s: Term, t: Term) -> float:
    """Score how well two terms match structurally (0.0 to 1.0)."""
    if s.is_variable and t.is_variable:
        return 1.0 if s.varnum == t.varnum else 0.5
    if s.is_variable or t.is_variable:
        return 0.0
    if s.symnum != t.symnum or s.arity != t.arity:
        return 0.0
    if s.is_constant:
        return 1.0
    sub_scores = [_term_match_score(s.args[i], t.args[i]) for i in range(s.arity)]
    return sum(sub_scores) / len(sub_scores) if sub_scores else 1.0


def complex4(t: Term) -> float:
    """Compute quadratic complexity measure (matching C complex4).

    Counts structural overlaps between all pairs of subterms,
    normalized by term size squared.
    """
    subs = _collect_subterms(t)
    n = len(subs)
    if n <= 1:
        return 0.0

    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += _term_match_score(subs[i], subs[j])

    size = _term_size(t)
    return total / (size * size) if size > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    """Entry point for pycomplex command."""
    symbol_table = SymbolTable()
    clauses = read_clause_stream(sys.stdin, symbol_table)

    for clause in clauses:
        for lit in clause.literals:
            atom = lit.atom
            # For equations, compute complexity of each side
            if atom.arity == 2:
                for side in atom.args:
                    score = complex4(side)
                    print(f"complexity = {score:.6f}")
            else:
                score = complex4(atom)
                print(f"complexity = {score:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
