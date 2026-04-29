"""Clause formatting utilities.

Extracted from GivenClauseSearch — these are pure functions with no
search-loop coupling. They operate on Clause/Term structures and an
optional SymbolTable.
"""

from __future__ import annotations

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
