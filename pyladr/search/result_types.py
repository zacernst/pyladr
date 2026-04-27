"""Search result types: exit codes, proof traces, and search results.

Extracted from given_clause.py to reduce the god-object size.
These are pure data types with no search-loop coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyladr.core.clause import Clause
    from pyladr.search.statistics import SearchStatistics


# ── Exit codes matching C search.h ──────────────────────────────────────────


class ExitCode(IntEnum):
    """Search termination codes matching C exit codes."""

    MAX_PROOFS_EXIT = 1
    SOS_EMPTY_EXIT = 2
    MAX_GIVEN_EXIT = 3
    MAX_KEPT_EXIT = 4
    MAX_SECONDS_EXIT = 5
    MAX_GENERATED_EXIT = 6
    FATAL_EXIT = 7


# ── Proof result ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Proof:
    """A proof found during search.

    Contains the empty clause and a trace of clauses used.
    """

    empty_clause: Clause
    clauses: tuple[Clause, ...]

    def __repr__(self) -> str:
        return (
            f"Proof(empty_clause=id:{self.empty_clause.id}, "
            f"length={len(self.clauses)})"
        )


# ── Search result ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Result of a search invocation."""

    exit_code: ExitCode
    proofs: tuple[Proof, ...]
    stats: SearchStatistics
