"""Core data structures: terms, symbols, clauses, substitutions."""

from pyladr.core.clause import Clause, Justification, JustType, Literal, ParaJust
from pyladr.core.substitution import (
    Context,
    Trail,
    apply_demod,
    apply_substitute,
    apply_substitute_at_pos,
    apply_substitution,
    context_to_pairs,
    dereference,
    empty_substitution,
    match,
    occur_check,
    subst_changes_term,
    unify,
    variable_substitution,
    variant,
)
from pyladr.core.symbol import (
    ParseType,
    Symbol,
    SymbolTable,
    SymbolType,
    UnifTheory,
    VariableStyle,
)
from pyladr.core.term import (
    MAX_ARITY,
    MAX_VARS,
    MAX_VNUM,
    Term,
    TermType,
    build_binary_term,
    build_unary_term,
    copy_term,
    get_rigid_term,
    get_variable_term,
)

__all__ = [
    # Term
    "Term",
    "TermType",
    "MAX_VARS",
    "MAX_VNUM",
    "MAX_ARITY",
    "get_variable_term",
    "get_rigid_term",
    "build_binary_term",
    "build_unary_term",
    "copy_term",
    # Symbol
    "Symbol",
    "SymbolTable",
    "SymbolType",
    "ParseType",
    "UnifTheory",
    "VariableStyle",
    # Clause
    "Clause",
    "Literal",
    "Justification",
    "JustType",
    "ParaJust",
    # Substitution & Unification
    "Context",
    "Trail",
    "dereference",
    "apply_substitution",
    "apply_substitute",
    "apply_substitute_at_pos",
    "apply_demod",
    "occur_check",
    "unify",
    "match",
    "variant",
    "empty_substitution",
    "variable_substitution",
    "subst_changes_term",
    "context_to_pairs",
]
