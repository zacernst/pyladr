"""Clause and Literal structures matching C LADR literals.h/topform.h.

A Literal is a sign (positive/negative) plus an atomic formula (Term).
A Clause (Topform in C) is a disjunction of Literals with metadata.

C uses linked lists for literals; we use tuples for immutability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from pyladr.core.term import Term


# ── Justification types matching C just.h ─────────────────────────────────────


class JustType(IntEnum):
    """Primary justification types from C Just_type enum."""

    INPUT = 0
    GOAL = auto()
    DENY = auto()
    CLAUSIFY = auto()
    COPY = auto()
    BINARY_RES = auto()
    HYPER_RES = auto()
    UR_RES = auto()
    FACTOR = auto()
    PARA = auto()
    # Secondary (simplification) justifications
    DEMOD = auto()
    UNIT_DEL = auto()
    FLIP = auto()
    BACK_DEMOD = auto()
    BACK_UNIT_DEL = auto()
    NEW_SYMBOL = auto()
    EXPAND_DEF = auto()
    FOLD_DEF = auto()
    RENUMBER = auto()
    PROPOSITIONAL = auto()
    INSTANTIATE = auto()
    IVY = auto()


# ── Justification ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ParaJust:
    """Paramodulation justification details (C struct parajust)."""

    from_id: int
    into_id: int
    from_pos: tuple[int, ...]
    into_pos: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Justification:
    """Single justification step (C struct just).

    A clause's justification is a sequence of these steps:
    the first is the primary (origin) justification,
    followed by zero or more secondary (simplification) steps.
    """

    just_type: JustType
    # Depending on type, one of these is populated:
    clause_id: int = 0  # for COPY, DENY, etc.
    clause_ids: tuple[int, ...] = ()  # for BINARY_RES, HYPER_RES, etc.
    para: ParaJust | None = None  # for PARA
    demod_steps: tuple[tuple[int, int, int], ...] = ()  # for DEMOD (id, direction, position)


# ── Literal ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Literal:
    """A signed atomic formula matching C struct literals.

    sign=True for positive, sign=False for negative (C BOOL).
    atom is the atomic formula as a Term.
    """

    sign: bool
    atom: Term

    @property
    def is_positive(self) -> bool:
        return self.sign

    @property
    def is_negative(self) -> bool:
        return not self.sign

    @property
    def is_eq_literal(self) -> bool:
        """Check if this literal's atom is an equality (= predicate).

        WARNING: This checks arity == 2 only, NOT the symbol name.  It will
        return True for any binary predicate, not just equality.  For accurate
        equality detection when a SymbolTable is available, use
        ``pyladr.inference.paramodulation.is_eq_atom(lit.atom, symbol_table)``
        or ``pyladr.inference.paramodulation.pos_eq/neg_eq`` instead.
        """
        return self.atom.is_complex and self.atom.arity == 2

    def complementary(self, other: Literal) -> bool:
        """Check if self and other are complementary (opposite sign, same atom)."""
        return self.sign != other.sign and self.atom.term_ident(other.atom)

    def to_str(self, symbol_table: object | None = None) -> str:
        """Format matching C clause output.

        Negative literals are prefixed with '-' (C not_sym).
        """
        atom_str = self.atom.to_str(symbol_table)
        if self.sign:
            return atom_str
        return f"-{atom_str}"

    def __repr__(self) -> str:
        sign_str = "" if self.sign else "-"
        return f"Lit({sign_str}{self.atom!r})"


# ── Clause (Topform) ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class Clause:
    """A clause (disjunction of literals) matching C struct topform.

    Mutable for fields that change during search (weight, flags, etc.)
    but literals are stored as an immutable tuple.
    """

    # Core data
    literals: tuple[Literal, ...] = ()

    # Metadata
    id: int = 0
    weight: float = 0.0
    justification: tuple[Justification, ...] = ()

    # Flags matching C topform fields
    is_formula: bool = False
    normal_vars: bool = False
    used: bool = False
    official_id: bool = False
    initial: bool = False
    subsumer: bool = False

    # Selection annotation: rule name used when this clause was selected as given
    # (e.g. "W", "A", "T2V", "F", "E"). Empty string if never selected as given.
    given_selection: str = ""

    # Cosine distance to nearest goal at the time this clause was selected as given.
    # 0.0 means not set (clause was never selected, or goal-distance disabled).
    given_distance: float = 0.0

    # Cached property — set in __post_init__, avoids repeated len() calls
    _num_literals: int = 0

    def __lt__(self, other: object) -> bool:
        """Comparison for heapq tie-breaking: lighter weight wins, then lower id."""
        if not isinstance(other, Clause):
            return NotImplemented
        if self.weight != other.weight:
            return self.weight < other.weight
        return self.id < other.id

    def __post_init__(self) -> None:
        # Ensure literals is a tuple
        if not isinstance(self.literals, tuple):
            object.__setattr__(self, "literals", tuple(self.literals))
        object.__setattr__(self, "_num_literals", len(self.literals))

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        """Empty clause = FALSE = contradiction found."""
        return self._num_literals == 0

    @property
    def is_unit(self) -> bool:
        """Unit clause has exactly one literal."""
        return self._num_literals == 1

    @property
    def is_positive(self) -> bool:
        """All literals are positive."""
        return all(lit.sign for lit in self.literals)

    @property
    def is_negative(self) -> bool:
        """All literals are negative."""
        return all(not lit.sign for lit in self.literals)

    @property
    def is_horn(self) -> bool:
        """At most one positive literal."""
        return sum(1 for lit in self.literals if lit.sign) <= 1

    @property
    def is_definite(self) -> bool:
        """Exactly one positive literal (definite Horn clause)."""
        return sum(1 for lit in self.literals if lit.sign) == 1

    @property
    def is_ground(self) -> bool:
        """No variables in any literal."""
        return all(lit.atom.is_ground for lit in self.literals)

    @property
    def num_literals(self) -> int:
        return self._num_literals

    def variables(self) -> set[int]:
        """Set of all variable numbers in this clause."""
        result: set[int] = set()
        for lit in self.literals:
            result.update(lit.atom.variables())
        return result

    def all_terms(self) -> Iterator[Term]:
        """Iterate all terms in all literals (atoms and their subterms)."""
        for lit in self.literals:
            yield from lit.atom.subterms()

    # ── String representation (C fprint_clause) ───────────────────────────

    def to_str(self, symbol_table: object | None = None) -> str:
        """Format matching C fprint_clause() output.

        Format: "ID: lit1 | lit2 | ... ."
        Empty clause prints as "$F".
        """
        parts: list[str] = []
        if self.id > 0:
            parts.append(f"{self.id}: ")

        if self.is_empty:
            parts.append("$F")
        else:
            lit_strs = [lit.to_str(symbol_table) for lit in self.literals]
            parts.append(" | ".join(lit_strs))

        parts.append(".")
        return "".join(parts)

    def __repr__(self) -> str:
        lits = ", ".join(repr(lit) for lit in self.literals)
        return f"Clause(id={self.id}, [{lits}])"
