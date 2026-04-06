"""Interpretation filter: filter interpretations through clauses.

Modernized Python 3.13+ version of apps.src/interpfilter.c.

The inverse of clausefilter: reads clauses from a file and interpretations
from stdin, outputs interpretations that satisfy the specified test.

Usage:
    pyinterpfilter <clause_file> {all_true|some_true|all_false|some_false}
"""

from __future__ import annotations

import sys
from enum import Enum, auto

from pyladr.apps.cli_common import read_clause_stream
from pyladr.core.interpretation import (
    compile_interp_from_text,
    eval_clause,
    format_interp_standard,
)
from pyladr.core.symbol import SymbolTable


class InterpFilterMode(Enum):
    ALL_TRUE = auto()
    SOME_TRUE = auto()
    ALL_FALSE = auto()
    SOME_FALSE = auto()


def _read_interps_from_stream(text: str) -> list:
    """Parse all interpretations from text."""
    interps = []
    parts = text.split("interpretation(")
    for part in parts[1:]:
        chunk = "interpretation(" + part
        end = chunk.find(").")
        if end >= 0:
            chunk = chunk[: end + 2]
        try:
            interp = compile_interp_from_text(chunk)
            interps.append(interp)
        except ValueError:
            continue
    return interps


def main(argv: list[str] | None = None) -> int:
    """Entry point for pyinterpfilter command."""
    args = argv if argv is not None else sys.argv[1:]

    if len(args) < 2:
        sys.stderr.write(
            "Usage: pyinterpfilter <clause_file> "
            "{all_true|some_true|all_false|some_false}\n"
        )
        return 1

    clause_file = args[0]
    mode_str = args[1]

    mode_map = {
        "all_true": InterpFilterMode.ALL_TRUE,
        "some_true": InterpFilterMode.SOME_TRUE,
        "all_false": InterpFilterMode.ALL_FALSE,
        "some_false": InterpFilterMode.SOME_FALSE,
    }

    if mode_str not in mode_map:
        sys.stderr.write(f"Unknown mode: {mode_str}\n")
        return 1

    mode = mode_map[mode_str]

    symbol_table = SymbolTable()
    eq_sym = symbol_table.str_to_sn("=", 2)

    try:
        with open(clause_file) as f:
            clauses = read_clause_stream(f, symbol_table)
    except FileNotFoundError:
        sys.stderr.write(f"Cannot open {clause_file}\n")
        return 1

    if not clauses:
        sys.stderr.write(f"No clauses found in {clause_file}\n")
        return 1

    sys.stderr.write(f"interpfilter: {len(clauses)} clause(s) loaded.\n")

    text = sys.stdin.read()
    interps = _read_interps_from_stream(text)

    passed = 0
    for interp in interps:
        keep = True
        for clause in clauses:
            try:
                result = eval_clause(clause, interp, symbol_table, eq_sym)
            except ValueError:
                continue

            if mode == InterpFilterMode.SOME_TRUE and result:
                break
            if mode == InterpFilterMode.SOME_FALSE and not result:
                break
            if mode == InterpFilterMode.ALL_TRUE and not result:
                keep = False
                break
            if mode == InterpFilterMode.ALL_FALSE and result:
                keep = False
                break
        else:
            # Reached end of clauses without breaking
            if mode in (InterpFilterMode.SOME_TRUE, InterpFilterMode.SOME_FALSE):
                keep = False

        if keep:
            print(format_interp_standard(interp))
            print()
            passed += 1

    sys.stderr.write(f"interpfilter: {len(interps)} read, {passed} passed.\n")
    return 0 if passed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
