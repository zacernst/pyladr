"""Clause tester: test clauses against interpretations.

Modernized Python 3.13+ version of apps.src/clausetester.c.

Reads interpretations from a file and clauses from stdin, reports
which interpretations satisfy each clause.

Usage:
    pyclausetester <interp_file> [commands]
"""

from __future__ import annotations

import sys

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


def _read_interps_from_file(filename: str) -> list[Interpretation]:
    """Read all interpretations from a file."""
    with open(filename) as f:
        text = f.read()

    interps: list[Interpretation] = []
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
    """Entry point for pyclausetester command."""
    args = argv if argv is not None else sys.argv[1:]

    if len(args) < 1:
        sys.stderr.write("Usage: pyclausetester <interp_file>\n")
        return 1

    interp_file = args[0]

    try:
        interps = _read_interps_from_file(interp_file)
    except FileNotFoundError:
        sys.stderr.write(f"Cannot open {interp_file}\n")
        return 1

    if not interps:
        sys.stderr.write(f"No interpretations found in {interp_file}\n")
        return 1

    sys.stderr.write(f"clausetester: {len(interps)} interpretation(s) loaded.\n")

    symbol_table = SymbolTable()
    eq_sym = symbol_table.str_to_sn("=", 2)

    clauses = read_clause_stream(sys.stdin, symbol_table)

    # Track how many clauses each interpretation models
    counts = [0] * len(interps)

    for clause in clauses:
        true_in: list[int] = []
        for idx, interp in enumerate(interps):
            try:
                if eval_clause(clause, interp, symbol_table, eq_sym):
                    true_in.append(idx)
                    counts[idx] += 1
            except ValueError:
                pass

        indices_str = ",".join(str(i) for i in true_in)
        print(f"{format_clause_standard(clause)} [{indices_str}]")

    # Summary
    print()
    print(f"% {len(clauses)} clause(s) tested against {len(interps)} interpretation(s).")
    for idx, count in enumerate(counts):
        print(f"% Interp {idx}: {count}/{len(clauses)} clauses true.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
