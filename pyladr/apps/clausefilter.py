"""Clause filter: filter clauses/formulas through interpretations.

Modernized Python 3.13+ version of apps.src/clausefilter.c.

Reads interpretations from a file and clauses from stdin, outputs
clauses that pass the specified logical test.

Usage:
    pyclausefilter <interp_file> {true_in_all|true_in_some|false_in_all|false_in_some} [commands]
"""

from __future__ import annotations

import sys
from enum import Enum, auto

from pyladr.apps.cli_common import (
    format_clause_standard,
    read_clause_stream,
)
from pyladr.core.interpretation import (
    Interpretation,
    compile_interp_from_text,
    eval_clause,
)
from pyladr.core.symbol import SymbolTable


class FilterMode(Enum):
    TRUE_IN_ALL = auto()
    TRUE_IN_SOME = auto()
    FALSE_IN_ALL = auto()
    FALSE_IN_SOME = auto()


def _read_interps_from_file(filename: str) -> list[Interpretation]:
    """Read all interpretations from a file."""
    with open(filename) as f:
        text = f.read()

    interps: list[Interpretation] = []
    # Split on "interpretation(" boundaries
    parts = text.split("interpretation(")
    for part in parts[1:]:  # skip preamble
        chunk = "interpretation(" + part
        # Find the closing ")."
        end = chunk.find(").")
        if end >= 0:
            chunk = chunk[: end + 2]
        try:
            interp = compile_interp_from_text(chunk)
            interps.append(interp)
        except ValueError:
            continue

    return interps


def _sort_interps(interps: list[Interpretation]) -> list[Interpretation]:
    """Sort interpretations by size (smallest first)."""
    return sorted(interps, key=lambda i: i.size)


def filter_clause(
    clause: "Clause",  # noqa: F821
    interps: list[Interpretation],
    mode: FilterMode,
    symbol_table: SymbolTable | None = None,
    eq_symnum: int | None = None,
) -> bool:
    """Test whether a clause passes the filter."""
    for interp in interps:
        try:
            result = eval_clause(clause, interp, symbol_table, eq_symnum)
        except ValueError:
            continue  # skip non-evaluable

        if mode == FilterMode.TRUE_IN_SOME and result:
            return True
        if mode == FilterMode.FALSE_IN_SOME and not result:
            return True
        if mode == FilterMode.TRUE_IN_ALL and not result:
            return False
        if mode == FilterMode.FALSE_IN_ALL and result:
            return False

    # Reached end without early return
    if mode in (FilterMode.TRUE_IN_ALL, FilterMode.FALSE_IN_ALL):
        return True  # all passed
    return False  # none matched


def main(argv: list[str] | None = None) -> int:
    """Entry point for pyclausefilter command."""
    args = argv if argv is not None else sys.argv[1:]

    if len(args) < 2:
        sys.stderr.write(
            "Usage: pyclausefilter <interp_file> "
            "{true_in_all|true_in_some|false_in_all|false_in_some}\n"
        )
        return 1

    interp_file = args[0]
    mode_str = args[1]

    mode_map = {
        "true_in_all": FilterMode.TRUE_IN_ALL,
        "true_in_some": FilterMode.TRUE_IN_SOME,
        "false_in_all": FilterMode.FALSE_IN_ALL,
        "false_in_some": FilterMode.FALSE_IN_SOME,
    }

    if mode_str not in mode_map:
        sys.stderr.write(f"Unknown mode: {mode_str}\n")
        return 1

    mode = mode_map[mode_str]

    try:
        interps = _read_interps_from_file(interp_file)
    except FileNotFoundError:
        sys.stderr.write(f"Cannot open {interp_file}\n")
        return 1

    if not interps:
        sys.stderr.write(f"No interpretations found in {interp_file}\n")
        return 1

    interps = _sort_interps(interps)
    sys.stderr.write(f"clausefilter: {len(interps)} interpretation(s) loaded.\n")

    symbol_table = SymbolTable()
    eq_sym = symbol_table.str_to_sn("=", 2)

    clauses = read_clause_stream(sys.stdin, symbol_table)

    passed = 0
    for clause in clauses:
        if filter_clause(clause, interps, mode, symbol_table, eq_sym):
            print(format_clause_standard(clause))
            passed += 1

    sys.stderr.write(f"clausefilter: {len(clauses)} read, {passed} passed.\n")
    return 0 if passed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
