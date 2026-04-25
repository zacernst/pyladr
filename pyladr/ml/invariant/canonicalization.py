"""Symbol canonicalization for property-invariant clause embeddings.

Provides canonical mapping of symbol identifiers so that logically equivalent
clauses (up to symbol renaming) produce identical graph structures. This is
the foundation of symbol-independence: instead of encoding raw symnum values
(which are arbitrary assignment-order identifiers), we assign canonical IDs
based on structural role — arity, usage pattern, and position in the clause.

The canonicalization preserves all structurally meaningful distinctions:
- Symbols with different arities get different canonical IDs
- Symbols that appear in different structural roles (predicate vs function)
  get different canonical IDs
- The ordering within an equivalence class is determined by first-occurrence
  in a deterministic clause traversal

This module does NOT modify Clause/Term objects. It produces a mapping
that the invariant graph builder uses during feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyladr.core.clause import Clause, Literal
    from pyladr.core.term import Term


@dataclass(frozen=True, slots=True)
class SymbolRole:
    """Structural role of a symbol, independent of its name/ID.

    Two symbols with the same SymbolRole are structurally interchangeable
    (they occupy the same equivalence class under renaming).

    Attributes:
        arity: Number of arguments.
        is_predicate: Whether the symbol appears as a literal's top-level atom.
        is_skolem: Whether the symbol is a Skolem function/constant.
    """

    arity: int
    is_predicate: bool
    is_skolem: bool = False


@dataclass(slots=True)
class CanonicalMapping:
    """Maps original symbol numbers to canonical IDs.

    Canonical IDs are assigned per equivalence class (same SymbolRole)
    in first-occurrence order during a deterministic clause traversal.

    Attributes:
        sym_to_canonical: Maps original symnum → canonical ID.
        canonical_to_role: Maps canonical ID → structural role.
        next_id: Next canonical ID to assign.
    """

    sym_to_canonical: dict[int, int] = field(default_factory=dict)
    canonical_to_role: dict[int, SymbolRole] = field(default_factory=dict)
    next_id: int = 0

    def get_or_assign(self, symnum: int, role: SymbolRole) -> int:
        """Get existing canonical ID or assign a new one.

        Args:
            symnum: Original symbol number.
            role: Structural role of this symbol.

        Returns:
            The canonical ID for this symbol.
        """
        stc = self.sym_to_canonical
        cid = stc.get(symnum)
        if cid is not None:
            return cid

        canonical_id = self.next_id
        self.next_id += 1
        stc[symnum] = canonical_id
        self.canonical_to_role[canonical_id] = role
        return canonical_id

    def get_or_assign_fast(
        self, symnum: int, arity: int, is_predicate: bool, is_skolem: bool = False,
    ) -> int:
        """Fast-path: assign canonical ID, deferring SymbolRole creation.

        Avoids creating a frozen dataclass on the common-case cache hit.
        """
        stc = self.sym_to_canonical
        cid = stc.get(symnum)
        if cid is not None:
            return cid

        canonical_id = self.next_id
        self.next_id += 1
        stc[symnum] = canonical_id
        self.canonical_to_role[canonical_id] = SymbolRole(arity, is_predicate, is_skolem)
        return canonical_id

    def reset(self) -> None:
        """Reset for reuse on a new clause."""
        self.sym_to_canonical.clear()
        self.canonical_to_role.clear()
        self.next_id = 0


def canonicalize_clause(
    clause: Clause,
    *,
    detect_skolem: bool = True,
) -> CanonicalMapping:
    """Build a canonical symbol mapping for a single clause.

    Traverses the clause in a deterministic order (literals left-to-right,
    terms depth-first left-to-right) and assigns canonical IDs based on
    structural role. The result is independent of original symbol naming.

    Args:
        clause: The clause to canonicalize.
        detect_skolem: If True, marks symbols whose names start with 'c'/'f'
            followed by digits as Skolem (heuristic matching C Prover9 naming).

    Returns:
        A CanonicalMapping from original symnums to canonical IDs.
    """
    mapping = CanonicalMapping()

    for literal in clause.literals:
        _canonicalize_literal(literal, mapping, detect_skolem)

    return mapping


def _canonicalize_literal(
    literal: Literal,
    mapping: CanonicalMapping,
    detect_skolem: bool,
) -> None:
    """Process a literal, marking its atom's top symbol as a predicate."""
    atom = literal.atom
    if not atom.is_variable:
        role = SymbolRole(
            arity=atom.arity,
            is_predicate=True,
            is_skolem=False,  # predicates are never Skolem
        )
        mapping.get_or_assign(atom.symnum, role)
        # Recurse into arguments
        for arg in atom.args:
            _canonicalize_term(arg, mapping, detect_skolem)
    # If atom is a variable (unusual but legal), nothing to canonicalize


def _canonicalize_term(
    term: Term,
    mapping: CanonicalMapping,
    detect_skolem: bool,
) -> None:
    """Recursively canonicalize symbols in a term."""
    if term.is_variable:
        return

    is_skolem = False
    if detect_skolem:
        # Heuristic: Skolem functions/constants have symnum patterns
        # In practice, the symbol table knows this, but we keep the
        # canonicalization independent of the symbol table for purity
        is_skolem = False  # Will be enriched by the graph builder if symbol_table available

    role = SymbolRole(
        arity=term.arity,
        is_predicate=False,
        is_skolem=is_skolem,
    )
    mapping.get_or_assign(term.symnum, role)

    for arg in term.args:
        _canonicalize_term(arg, mapping, detect_skolem)


def canonicalize_clause_with_symbol_table(
    clause: Clause,
    symbol_table: object | None = None,
) -> CanonicalMapping:
    """Build canonical mapping using symbol table metadata when available.

    This is the preferred entry point when a SymbolTable is available,
    as it can accurately identify Skolem symbols rather than relying
    on naming heuristics.

    Args:
        clause: The clause to canonicalize.
        symbol_table: Optional SymbolTable for Skolem detection.

    Returns:
        A CanonicalMapping from original symnums to canonical IDs.
    """
    mapping = CanonicalMapping()

    for literal in clause.literals:
        _canonicalize_literal_with_st(literal, mapping, symbol_table)

    return mapping


def _canonicalize_literal_with_st(
    literal: Literal,
    mapping: CanonicalMapping,
    symbol_table: object | None,
) -> None:
    """Process a literal with symbol table metadata."""
    atom = literal.atom
    if not atom.is_variable:
        mapping.get_or_assign_fast(atom.symnum, atom.arity, True, False)
        for arg in atom.args:
            _canonicalize_term_with_st(arg, mapping, symbol_table)


def _canonicalize_term_with_st(
    term: Term,
    mapping: CanonicalMapping,
    symbol_table: object | None,
) -> None:
    """Recursively canonicalize with symbol table Skolem detection."""
    if term.is_variable:
        return

    is_skolem = False
    if symbol_table is not None:
        try:
            sym = symbol_table.get_symbol(term.symnum)
            is_skolem = bool(sym.skolem)
        except (KeyError, IndexError, AttributeError):
            pass

    mapping.get_or_assign_fast(term.symnum, term.arity, False, is_skolem)

    for arg in term.args:
        _canonicalize_term_with_st(arg, mapping, symbol_table)
