"""Clause formatting and structural analysis utilities.

Extracted from GivenClauseSearch — these are pure functions with no
search-loop coupling. They operate on Clause/Term structures and an
optional SymbolTable.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyladr.core.clause import Clause


def format_clause_std(symbol_table: object | None, clause: Clause) -> str:
    """Format clause matching C CL_FORM_STD (ID, literals, justification).

    Args:
        symbol_table: SymbolTable for resolving symbol names (or None).
        clause: The clause to format.
    """
    parts: list[str] = []
    if clause.id > 0:
        parts.append(f"{clause.id} ")
    if clause.is_empty:
        parts.append("$F")
    else:
        lit_strs = []
        for lit in clause.literals:
            atom_str = lit.atom.to_str(symbol_table)
            lit_strs.append(atom_str if lit.sign else f"-{atom_str}")
        parts.append(" | ".join(lit_strs))
    if clause.justification:
        just = clause.justification[0]
        parts.append(f".  [{just.just_type.name.lower()}].")
    else:
        parts.append(".")
    return "".join(parts)


def calculate_structural_entropy(clause: Clause) -> float:
    """Calculate Shannon entropy of clause interpreted as tree structure.

    Node types: Clause, Literal, Predicate, Function, Variable, Constant
    Formula: H = -sum p(v) log2 p(v) where p(v) is probability of node type v
    """
    node_counts = {
        'clause': 0,
        'literal': 0,
        'predicate': 0,
        'function': 0,
        'variable': 0,
        'constant': 0
    }

    # Count clause node
    node_counts['clause'] = 1

    # Count literals
    node_counts['literal'] = len(clause.literals)

    # Count predicate, function, variable, constant nodes in all terms
    for literal in clause.literals:
        _count_term_nodes(literal.atom, node_counts, is_predicate=True)

    # Calculate entropy
    total_nodes = sum(node_counts.values())
    if total_nodes <= 1:
        return 0.0

    entropy = 0.0
    for count in node_counts.values():
        if count > 0:
            p = count / total_nodes
            entropy -= p * math.log2(p)

    return entropy


def _count_term_nodes(term: object, node_counts: dict, is_predicate: bool = False) -> None:
    """Recursively count nodes in a term tree."""
    if term.is_variable:  # type: ignore[union-attr]
        node_counts['variable'] += 1
    elif term.is_constant:  # type: ignore[union-attr]
        node_counts['constant'] += 1
    else:
        # Complex term - distinguish between predicate and function
        if is_predicate:
            node_counts['predicate'] += 1
        else:
            node_counts['function'] += 1

        # Recursively count argument nodes
        for arg in term.args:  # type: ignore[union-attr]
            _count_term_nodes(arg, node_counts, is_predicate=False)
