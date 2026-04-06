"""IDfilter: filter equations through demodulation.

Modernized Python 3.13+ version of apps.src/idfilter.c.

Reads demodulators from a file and equations from stdin, applies
rewriting, outputs equations that remain non-trivial (or trivial
with 'x' flag).

Usage:
    pyidfilter <demod_file> [x]
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


def _is_trivial_eq(clause, symbol_table: SymbolTable, eq_sym: int) -> bool:
    """Check if clause is a trivial identity x = x."""
    if len(clause.literals) != 1 or not clause.literals[0].sign:
        return False
    atom = clause.literals[0].atom
    if not atom.is_complex or atom.symnum != eq_sym or atom.arity != 2:
        return False
    return atom.args[0].term_ident(atom.args[1])


def main(argv: list[str] | None = None) -> int:
    """Entry point for pyidfilter command."""
    args = argv if argv is not None else sys.argv[1:]

    if len(args) < 1:
        sys.stderr.write("Usage: pyidfilter <demod_file> [x]\n")
        return 1

    demod_file = args[0]
    output_trivial = "x" in args[1:]

    symbol_table = SymbolTable()
    eq_sym = symbol_table.str_to_sn("=", 2)

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

    sys.stderr.write(f"idfilter: {len(demod_index)} demodulator(s) loaded.\n")

    # Read equations from stdin
    clauses = read_clause_stream(sys.stdin, symbol_table)

    checked = 0
    passed = 0

    for clause in clauses:
        checked += 1
        # Apply demodulation
        rewritten = demodulate_clause(clause, demod_index, symbol_table)
        is_trivial = _is_trivial_eq(rewritten, symbol_table, eq_sym)

        if (not output_trivial and not is_trivial) or \
           (output_trivial and is_trivial):
            passed += 1
            print(format_clause_bare(clause))

    suffix = " x" if output_trivial else ""
    sys.stderr.write(f"idfilter{suffix}: checked {checked}, passed {passed}.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
