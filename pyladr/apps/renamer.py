"""Renamer: renumber variables in clauses.

Matches C apps.src/renamer.c — reads a stream of clauses and renumbers
variables to consecutive v0, v1, v2, ... in order of first occurrence.

Usage:
    pyrenamer [commands] < input > output
    pyrenamer -f input_file

Options:
    commands    Read LADR commands before clause list
    -f FILE     Read from file instead of stdin
    help        Show this help message
"""

from __future__ import annotations

import sys

from pyladr.apps.cli_common import (
    format_clause_bare,
    make_base_parser,
    open_input,
    read_clause_stream,
)
from pyladr.inference.resolution import renumber_variables


def main(argv: list[str] | None = None) -> int:
    """Entry point for pyrenamer command."""
    parser = make_base_parser(
        "pyrenamer",
        "Renumber variables in clauses to consecutive v0, v1, v2, ...",
    )
    parser.add_argument(
        "flags",
        nargs="*",
        help="Optional flags: commands",
    )

    args = parser.parse_args(argv)

    if "help" in (args.flags or []):
        parser.print_help()
        return 1

    fin = open_input(args)
    try:
        clauses = read_clause_stream(fin)
    finally:
        if fin is not sys.stdin:
            fin.close()

    for clause in clauses:
        renamed = renumber_variables(clause)
        print(format_clause_bare(renamed))
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
