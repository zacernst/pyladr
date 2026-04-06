"""Rewriter: apply demodulation to terms/clauses.

Modernized Python 3.13+ version of apps.src/rewriter.c.

Reads demodulators from a file and applies term rewriting
to clauses/equations from stdin.

Usage:
    pyrewriter <demod_file>
"""

from __future__ import annotations

import sys

from pyladr.apps.cli_common import format_clause_bare, read_clause_stream
from pyladr.core.symbol import SymbolTable
from pyladr.inference.demodulation import (
    DemodulatorIndex,
    demodulate_clause,
    demodulator_type,
)


def main(argv: list[str] | None = None) -> int:
    """Entry point for pyrewriter command."""
    args = argv if argv is not None else sys.argv[1:]

    if len(args) < 1:
        sys.stderr.write("Usage: pyrewriter <demod_file>\n")
        return 1

    demod_file = args[0]

    symbol_table = SymbolTable()

    # Load demodulators
    try:
        with open(demod_file) as f:
            demod_clauses = read_clause_stream(f, symbol_table)
    except FileNotFoundError:
        sys.stderr.write(f"Cannot open {demod_file}\n")
        return 1

    # Build demodulator index
    demod_index = DemodulatorIndex()
    for clause in demod_clauses:
        dtype = demodulator_type(clause, symbol_table)
        if dtype:
            demod_index.insert(clause, dtype)

    sys.stderr.write(f"rewriter: {len(demod_index)} demodulator(s) loaded.\n")

    # Read clauses from stdin and rewrite
    clauses = read_clause_stream(sys.stdin, symbol_table)

    for clause in clauses:
        rewritten = demodulate_clause(clause, demod_index, symbol_table)
        print(format_clause_bare(rewritten))

    sys.stderr.write(f"rewriter: {len(clauses)} clause(s) rewritten.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
