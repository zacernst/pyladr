"""Sigtest: compute clause signatures across interpretations.

Modernized Python 3.13+ version of apps.src/sigtest.c.

For each interpretation, evaluates all clauses and reports the number
of false instances per clause in tabular format.

Usage:
    pysigtest <clause_file>
"""

from __future__ import annotations

import sys

from pyladr.apps.cli_common import read_clause_stream
from pyladr.core.interpretation import (
    compile_interp_from_text,
    eval_clause_false_instances,
)
from pyladr.core.symbol import SymbolTable


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
    """Entry point for pysigtest command."""
    args = argv if argv is not None else sys.argv[1:]

    if len(args) < 1:
        sys.stderr.write("Usage: pysigtest <clause_file>\n")
        return 1

    clause_file = args[0]

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

    sys.stderr.write(f"sigtest: {len(clauses)} clause(s) loaded.\n")

    text = sys.stdin.read()
    interps = _read_interps_from_stream(text)

    # Header
    print(f"% {len(clauses)} clause(s), {len(interps)} interpretation(s)")
    print(f"% Columns: interp_index, size, false_instances_per_clause...")
    print()

    for idx, interp in enumerate(interps):
        counts = []
        for clause in clauses:
            try:
                fc = eval_clause_false_instances(clause, interp, symbol_table, eq_sym)
                counts.append(fc)
            except ValueError:
                counts.append(-1)

        counts_str = " ".join(f"{c:4d}" for c in counts)
        print(f"{idx:3d} (size {interp.size:2d}): {counts_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
