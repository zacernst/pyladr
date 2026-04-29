"""Vampire.in domain parser for RNN2Vec training.

Parses vampire.in files into PyLADR Clause/Term structures using the
existing LADR parser infrastructure. Extracts formulas from both SOS
and goals sections, along with all subterms for comprehensive training.

The vampire.in domain uses a constrained vocabulary:
- P: unary predicate (propositional wrapper)
- i: binary function (implication)
- n: unary function (negation)
- Variables: x, y, z, u, v, w

This constrained vocabulary makes RNN2Vec training especially effective,
as only ~6-8 unique tokens need to be learned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyladr.core.clause import Clause
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term
from pyladr.parsing.ladr_parser import LADRParser, ParseError

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VampireCorpus:
    """Parsed vampire.in corpus with categorized formulas.

    Attributes:
        sos_clauses: Clauses from the SOS (set of support) section.
        goal_clauses: Clauses from the goals section.
        all_terms: All unique top-level atom terms from all clauses.
        all_subterms: All unique subterms extracted from all clauses.
        symbol_table: The symbol table built during parsing.
    """

    sos_clauses: tuple[Clause, ...]
    goal_clauses: tuple[Clause, ...]
    all_terms: tuple[Term, ...]
    all_subterms: tuple[Term, ...]
    symbol_table: SymbolTable

    @property
    def all_clauses(self) -> tuple[Clause, ...]:
        return self.sos_clauses + self.goal_clauses

    @property
    def num_clauses(self) -> int:
        return len(self.sos_clauses) + len(self.goal_clauses)

    @property
    def num_unique_subterms(self) -> int:
        return len(self.all_subterms)


def parse_vampire_file(filepath: str) -> VampireCorpus:
    """Parse a vampire.in file into a VampireCorpus.

    Uses the existing LADRParser to handle the full LADR syntax,
    then extracts and categorizes all formulas and subterms.

    Args:
        filepath: Path to the vampire.in file.

    Returns:
        VampireCorpus with all parsed structures.

    Raises:
        FileNotFoundError: If the file does not exist.
        ParseError: If the file contains unparseable formulas.
    """
    with open(filepath) as f:
        text = f.read()
    return parse_vampire_text(text)


def parse_vampire_text(text: str) -> VampireCorpus:
    """Parse vampire.in format text into a VampireCorpus.

    Args:
        text: LADR-format text content.

    Returns:
        VampireCorpus with all parsed structures.
    """
    symbol_table = SymbolTable()
    parser = LADRParser(symbol_table)

    try:
        parsed = parser.parse_input(text)
    except ParseError as e:
        logger.error("Failed to parse vampire.in text: %s", e)
        raise

    sos_clauses = tuple(parsed.sos + parsed.usable)
    goal_clauses = tuple(parsed.goals)

    # Extract all top-level atom terms and all subterms.
    # Use Term as dict key (frozen dataclass with structural __hash__/__eq__)
    # rather than id(subterm): id() returns a memory address that CPython
    # reuses after GC, causing different subterms to falsely collide.
    all_terms: list[Term] = []
    seen_subterms: dict[Term, Term] = {}  # Term -> Term for structural dedup

    for clause in sos_clauses + goal_clauses:
        for lit in clause.literals:
            all_terms.append(lit.atom)
            for subterm in lit.atom.subterms():
                if subterm not in seen_subterms:
                    seen_subterms[subterm] = subterm

    logger.info(
        "Parsed vampire corpus: %d SOS clauses, %d goal clauses, "
        "%d terms, %d unique subterms",
        len(sos_clauses),
        len(goal_clauses),
        len(all_terms),
        len(seen_subterms),
    )

    return VampireCorpus(
        sos_clauses=sos_clauses,
        goal_clauses=goal_clauses,
        all_terms=tuple(all_terms),
        all_subterms=tuple(seen_subterms.values()),
        symbol_table=symbol_table,
    )


def parse_clauses_from_text(text: str) -> list[Clause]:
    """Parse bare clauses from text (convenience for augmented data).

    Handles both formula-wrapped and bare clause formats.
    """
    symbol_table = SymbolTable()
    parser = LADRParser(symbol_table)

    if "formulas(" in text or "clauses(" in text:
        parsed = parser.parse_input(text)
        return list(parsed.all_clauses)

    clauses: list[Clause] = []
    for stmt in text.replace("\n", " ").split("."):
        stmt = stmt.strip()
        if not stmt or stmt.startswith("%") or stmt == "end_of_list":
            continue
        try:
            clauses.append(parser.parse_clause_from_string(stmt))
        except ParseError:
            continue
    return clauses
