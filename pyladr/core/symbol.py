"""Symbol table management matching C LADR symbols.h/symbols.c.

The symbol table maps (name, arity) pairs to unique integer IDs.
Each symbol has rich metadata: parse type, precedence, ordering weight,
unification theory, etc.

C uses two hash tables (By_id and By_sym) with 50000 buckets each.
We use Python dicts for equivalent O(1) lookup, thread-safe via lock.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import IntEnum, auto


# ── Enumerations matching C symbols.h ─────────────────────────────────────────


class SymbolType(IntEnum):
    """C Symbol_type enum."""

    UNSPECIFIED = 0
    FUNCTION = 1
    PREDICATE = 2


class ParseType(IntEnum):
    """C Parsetype enum for operator notation."""

    ORDINARY = 0  # standard prefix f(x, y)
    INFIX = auto()
    INFIX_LEFT = auto()
    INFIX_RIGHT = auto()
    PREFIX_PAREN = auto()  # prefix with parens: -( x )
    PREFIX = auto()  # prefix without parens: - x
    POSTFIX_PAREN = auto()
    POSTFIX = auto()


class UnifTheory(IntEnum):
    """C Unif_theory: unification theory for AC matching."""

    EMPTY_THEORY = 0
    COMMUTE = 1
    ASSOC_COMMUTE = 2


class LrpoStatus(IntEnum):
    """C Lrpo_status: for RPO term ordering."""

    LR_STATUS = 0
    MULTISET_STATUS = 1


class VariableStyle(IntEnum):
    """C Variable_style: how variables are printed."""

    STANDARD = 0  # x, y, z, u, v, w, v6, v7, ...
    PROLOG = 1  # A, B, C, ...
    INTEGER = 2  # 0, 1, 2, ...


# ── Symbol data ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Symbol:
    """Single symbol entry matching C struct symbol.

    Mutable because properties like occurrences, lex_val, kb_weight
    change during the search.
    """

    symnum: int  # unique ID
    name: str  # print string
    arity: int  # 0 for constants
    sym_type: SymbolType = SymbolType.UNSPECIFIED
    parse_type: ParseType = ParseType.ORDINARY
    parse_prec: int = 0  # precedence 1-999, 0 = unset
    unif_theory: UnifTheory = UnifTheory.EMPTY_THEORY
    occurrences: int = 0
    lex_val: int = 0  # precedence for term orderings
    kb_weight: int = 1  # Knuth-Bendix weight (default 1)
    lrpo_status: LrpoStatus = LrpoStatus.LR_STATUS
    skolem: bool = False
    unfold: bool = False
    auxiliary: bool = False  # not part of the theory


# ── Magic symbols (C static char* constants) ─────────────────────────────────


TRUE_SYM = "$T"
FALSE_SYM = "$F"
AND_SYM = "&"
OR_SYM = "|"
NOT_SYM = "-"
IFF_SYM = "<->"
IMP_SYM = "->"
IMPBY_SYM = "<-"
ALL_SYM = "all"
EXISTS_SYM = "exists"
EQ_SYM = "="
NEQ_SYM = "!="
ATTRIB_SYM = "#"


# ── Symbol Table ──────────────────────────────────────────────────────────────


class SymbolTable:
    """Thread-safe symbol table matching C By_id/By_sym hash tables.

    Maps (name, arity) → symnum and symnum → Symbol.
    Symbol IDs start at 1 and increment.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_id = 1
        # Primary indices
        self._by_id: dict[int, Symbol] = {}
        self._by_name_arity: dict[tuple[str, int], int] = {}
        # Variable style for printing
        self.variable_style: VariableStyle = VariableStyle.STANDARD

    def str_to_sn(self, name: str, arity: int) -> int:
        """Get or create symbol ID for (name, arity) pair.

        Matches C str_to_sn(). Thread-safe.
        """
        key = (name, arity)
        sn = self._by_name_arity.get(key)
        if sn is not None:
            return sn
        with self._lock:
            # Double-check after lock
            sn = self._by_name_arity.get(key)
            if sn is not None:
                return sn
            sn = self._next_id
            self._next_id += 1
            sym = Symbol(symnum=sn, name=name, arity=arity)
            self._by_id[sn] = sym
            self._by_name_arity[key] = sn
            return sn

    def sn_to_str(self, symnum: int) -> str:
        """Get symbol name from ID. Matches C sn_to_str()."""
        sym = self._by_id.get(symnum)
        if sym is None:
            raise KeyError(f"Unknown symbol ID: {symnum}")
        return sym.name

    def id_to_name(self, symnum: int) -> str:
        """Alias for sn_to_str for readability."""
        return self.sn_to_str(symnum)

    def sn_to_arity(self, symnum: int) -> int:
        """Get arity from symbol ID. Matches C sn_to_arity()."""
        sym = self._by_id.get(symnum)
        if sym is None:
            raise KeyError(f"Unknown symbol ID: {symnum}")
        return sym.arity

    def get_symbol(self, symnum: int) -> Symbol:
        """Get the full Symbol object by ID."""
        sym = self._by_id.get(symnum)
        if sym is None:
            raise KeyError(f"Unknown symbol ID: {symnum}")
        return sym

    def is_symbol(self, symnum: int, name: str, arity: int) -> bool:
        """Check if symnum matches (name, arity). Matches C is_symbol()."""
        sym = self._by_id.get(symnum)
        if sym is None:
            return False
        return sym.name == name and sym.arity == arity

    def set_parse_type(self, symnum: int, parse_type: ParseType, prec: int) -> None:
        """Set parsing notation for a symbol."""
        sym = self.get_symbol(symnum)
        sym.parse_type = parse_type
        sym.parse_prec = prec

    def set_kb_weight(self, symnum: int, weight: int) -> None:
        """Set Knuth-Bendix weight for a symbol."""
        sym = self.get_symbol(symnum)
        sym.kb_weight = weight

    def set_lrpo_status(self, symnum: int, status: LrpoStatus) -> None:
        """Set LRPO status for a symbol."""
        sym = self.get_symbol(symnum)
        sym.lrpo_status = status

    def mark_skolem(self, symnum: int) -> None:
        """Mark a symbol as Skolem."""
        sym = self.get_symbol(symnum)
        sym.skolem = True

    def increment_occurrences(self, symnum: int) -> None:
        """Increment occurrence count for a symbol."""
        sym = self.get_symbol(symnum)
        sym.occurrences += 1

    @property
    def all_symbols(self) -> list[Symbol]:
        """Return all symbols ordered by ID."""
        return sorted(self._by_id.values(), key=lambda s: s.symnum)

    def symbols_of_type(self, sym_type: SymbolType) -> list[Symbol]:
        """Return all symbols of a given type, ordered by ID."""
        return [s for s in self.all_symbols if s.sym_type == sym_type]

    def format_variable(self, varnum: int) -> str:
        """Format a variable number according to current variable style.

        Standard: x, y, z, u, v, w, v6, v7, ...
        Prolog: A, B, C, ..., Z, V26, V27, ...
        Integer: 0, 1, 2, ...
        """
        if self.variable_style == VariableStyle.INTEGER:
            return str(varnum)
        if self.variable_style == VariableStyle.PROLOG:
            if varnum < 26:
                return chr(ord("A") + varnum)
            return f"V{varnum}"
        # STANDARD style
        std_names = ["x", "y", "z", "u", "v", "w"]
        if varnum < len(std_names):
            return std_names[varnum]
        return f"v{varnum}"

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, symnum: int) -> bool:
        return symnum in self._by_id
