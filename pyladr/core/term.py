"""Term representation matching C LADR term.h/term.c.

Terms are the fundamental data structure in LADR. There are three types:
- VARIABLE: private_symbol >= 0 (value is variable number)
- CONSTANT: private_symbol < 0 and arity == 0
- COMPLEX: private_symbol < 0 and arity > 0

The C encoding stores symbol IDs as negative numbers in private_symbol,
accessed via SYMNUM(t) = -(t->private_symbol). Variable numbers are stored
directly as non-negative values in private_symbol.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

# ── Constants matching C term.h ───────────────────────────────────────────────

MAX_VARS = 100  # max distinct variables per term (for array indexing)
MAX_VNUM = 5000  # maximum variable ID for shared variable array
MAX_ARITY = 255  # max arity (unsigned char in C)


class TermType(IntEnum):
    """Term classification matching C macros VARIABLE/CONSTANT/COMPLEX."""

    VARIABLE = 0
    CONSTANT = 1
    COMPLEX = 2


# ── Term ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Term:
    """Immutable first-order term, matching C struct term.

    Internal encoding mirrors C:
      private_symbol >= 0  → variable (value = variable number)
      private_symbol < 0   → rigid symbol (SYMNUM = -private_symbol)
      arity == 0 with rigid symbol → constant
      arity > 0  with rigid symbol → complex term

    Args are stored as a tuple for immutability and hashability.
    """

    private_symbol: int
    arity: int = 0
    args: tuple[Term, ...] = ()
    # Container reference (clause, etc.) — not included in hash/eq
    container: object = field(default=None, compare=False, hash=False)
    # Unique ID for FPA indexing — not included in hash/eq
    term_id: int = field(default=0, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.arity != len(self.args):
            raise ValueError(
                f"Arity {self.arity} does not match number of args {len(self.args)}"
            )
        if self.arity > MAX_ARITY:
            raise ValueError(f"Arity {self.arity} exceeds MAX_ARITY ({MAX_ARITY})")

    # ── Type classification (C macros) ────────────────────────────────────

    @property
    def is_variable(self) -> bool:
        """VARIABLE(t): private_symbol >= 0."""
        return self.private_symbol >= 0

    @property
    def is_constant(self) -> bool:
        """CONSTANT(t): private_symbol < 0 and arity == 0."""
        return self.private_symbol < 0 and self.arity == 0

    @property
    def is_complex(self) -> bool:
        """COMPLEX(t): private_symbol < 0 and arity > 0."""
        return self.private_symbol < 0 and self.arity > 0

    @property
    def term_type(self) -> TermType:
        if self.private_symbol >= 0:
            return TermType.VARIABLE
        if self.arity == 0:
            return TermType.CONSTANT
        return TermType.COMPLEX

    # ── Access macros ─────────────────────────────────────────────────────

    @property
    def symnum(self) -> int:
        """SYMNUM(t): symbol ID for CONSTANT/COMPLEX terms.

        Raises ValueError if called on a variable.
        """
        if self.private_symbol >= 0:
            raise ValueError("SYMNUM called on variable term")
        return -self.private_symbol

    @property
    def varnum(self) -> int:
        """VARNUM(t): variable number for VARIABLE terms.

        Raises ValueError if called on non-variable.
        """
        if self.private_symbol < 0:
            raise ValueError("VARNUM called on non-variable term")
        return self.private_symbol

    def arg(self, i: int) -> Term:
        """ARG(t, i): get i-th argument (0-indexed)."""
        return self.args[i]

    # ── Hash matching C implementation ────────────────────────────────────

    def c_hash(self) -> int:
        """Hash matching C hash_term() for behavioral equivalence.

        C implementation:
            if VARIABLE: return VARNUM(t)
            else: x = SYMNUM(t); for each arg: x = (x << 3) ^ hash(arg); return x
        """
        if self.is_variable:
            return self.varnum & 0xFFFFFFFF
        x = self.symnum & 0xFFFFFFFF
        for a in self.args:
            x = ((x << 3) & 0xFFFFFFFF) ^ a.c_hash()
        return x

    # ── Structural identity (C term_ident) ────────────────────────────────

    def term_ident(self, other: Term) -> bool:
        """Check structural identity matching C term_ident().

        Two terms are identical iff they have the same private_symbol,
        same arity, and all arguments are recursively identical.
        """
        if self.private_symbol != other.private_symbol:
            return False
        if self.arity != other.arity:
            return False
        return all(a.term_ident(b) for a, b in zip(self.args, other.args, strict=True))

    # ── Tree traversal ────────────────────────────────────────────────────

    @property
    def is_ground(self) -> bool:
        """ground_term(t): no variables in term."""
        if self.is_variable:
            return False
        return all(a.is_ground for a in self.args)

    @property
    def depth(self) -> int:
        """term_depth(t): depth of term tree."""
        if self.arity == 0:
            return 0
        return 1 + max(a.depth for a in self.args)

    @property
    def symbol_count(self) -> int:
        """symbol_count(t): number of nodes in term tree."""
        return 1 + sum(a.symbol_count for a in self.args)

    def biggest_variable(self) -> int:
        """biggest_variable(t): largest variable number, or -1 if ground."""
        if self.is_variable:
            return self.varnum
        if self.arity == 0:
            return -1
        return max(a.biggest_variable() for a in self.args)

    def occurs_in(self, t2: Term) -> bool:
        """occurs_in(t1, t2): does self occur as subterm of t2?"""
        if self.term_ident(t2):
            return True
        return any(self.occurs_in(a) for a in t2.args)

    def subterms(self) -> Iterator[Term]:
        """Iterate all subterms (pre-order traversal)."""
        yield self
        for a in self.args:
            yield from a.subterms()

    def variables(self) -> set[int]:
        """Set of variable numbers occurring in this term."""
        result: set[int] = set()
        for t in self.subterms():
            if t.is_variable:
                result.add(t.varnum)
        return result

    # ── String representation (C fprint_term) ─────────────────────────────

    def to_str(self, symbol_table: object | None = None) -> str:
        """Format matching C fprint_term() output.

        Without a symbol table, uses raw symbol IDs: s1, s2, etc.
        With a symbol table, uses proper symbol names.
        """
        if self.is_variable:
            return f"v{self.varnum}"
        if symbol_table is not None:
            from pyladr.core.symbol import SymbolTable

            assert isinstance(symbol_table, SymbolTable)
            name = symbol_table.id_to_name(self.symnum)
        else:
            name = f"s{self.symnum}"
        if self.is_constant:
            return name
        arg_strs = ",".join(a.to_str(symbol_table) for a in self.args)
        return f"{name}({arg_strs})"

    def __repr__(self) -> str:
        if self.is_variable:
            return f"Var({self.varnum})"
        if self.is_constant:
            return f"Const(sn={self.symnum})"
        args_r = ", ".join(repr(a) for a in self.args)
        return f"Term(sn={self.symnum}, [{args_r}])"


# ── Term constructors ─────────────────────────────────────────────────────────

# Shared variable cache (matches C Shared_variables array)
_variable_cache: dict[int, Term] = {}
_variable_cache_lock = threading.Lock()


def get_variable_term(varnum: int) -> Term:
    """Get a shared variable term (C get_variable_term).

    Variables with the same number share the same object.
    Thread-safe via lock.
    """
    if varnum < 0:
        raise ValueError(f"Variable number must be non-negative, got {varnum}")
    t = _variable_cache.get(varnum)
    if t is not None:
        return t
    with _variable_cache_lock:
        # Double-check after acquiring lock
        t = _variable_cache.get(varnum)
        if t is not None:
            return t
        t = Term(private_symbol=varnum)
        _variable_cache[varnum] = t
        return t


def get_rigid_term(symnum: int, arity: int, args: tuple[Term, ...] = ()) -> Term:
    """Create a rigid (non-variable) term (C get_rigid_term_dangerously).

    Args:
        symnum: Symbol ID (positive integer).
        arity: Number of arguments.
        args: Argument terms.
    """
    if symnum <= 0:
        raise ValueError(f"Symbol number must be positive, got {symnum}")
    return Term(private_symbol=-symnum, arity=arity, args=args)


def build_binary_term(symnum: int, left: Term, right: Term) -> Term:
    """C build_binary_term: create f(left, right)."""
    return Term(private_symbol=-symnum, arity=2, args=(left, right))


def build_unary_term(symnum: int, arg: Term) -> Term:
    """C build_unary_term: create f(arg)."""
    return Term(private_symbol=-symnum, arity=1, args=(arg,))


def copy_term(t: Term) -> Term:
    """Deep copy a term tree (C copy_term).

    Since Terms are frozen, this returns the same object for variables
    (shared) and creates new Term instances for rigid terms.
    """
    if t.is_variable:
        return get_variable_term(t.varnum)
    if t.is_constant:
        return Term(private_symbol=t.private_symbol)
    new_args = tuple(copy_term(a) for a in t.args)
    return Term(private_symbol=t.private_symbol, arity=t.arity, args=new_args)
