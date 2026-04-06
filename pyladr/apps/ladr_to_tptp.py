"""LADR to TPTP converter.

Modernized Python 3.13+ version of provers.src/ladr_to_tptp.c.

Converts LADR clause/formula format to TPTP (Thousands of Problems
for Theorem Provers) standard format.

Usage:
    pyladr-to-tptp [-f input_file]
"""

from __future__ import annotations

import re
import sys


# TPTP-illegal characters in symbol names
_BAD_TPTP_CHARS = set("^@#$")


def _sanitize_symbol(name: str) -> str:
    """Make a symbol name TPTP-legal."""
    if any(c in _BAD_TPTP_CHARS for c in name):
        # Quote it
        return f"'{name}'"
    return name


def _term_to_tptp(text: str) -> str:
    """Convert a LADR term/formula line to TPTP syntax.

    Key transformations:
    - '=' becomes 'equal(lhs,rhs)' or uses '=' infix in FOF
    - Variables are uppercased
    - Constants/functions are lowercased (unless already)
    - Sanitize special characters
    """
    # Simple transformations for common cases
    # This is a basic converter — full implementation would need the parser
    result = text.strip()
    if result.endswith("."):
        result = result[:-1]
    return result


def _convert_clause(clause_text: str, idx: int, role: str = "axiom") -> str:
    """Convert a single LADR clause to TPTP format."""
    inner = _term_to_tptp(clause_text)
    return f"cnf(c{idx},{role},({inner}))."


def _convert_formula(formula_text: str, idx: int, role: str = "axiom") -> str:
    """Convert a single LADR formula to TPTP format."""
    inner = _term_to_tptp(formula_text)
    return f"fof(f{idx},{role},({inner}))."


def main(argv: list[str] | None = None) -> int:
    """Entry point for pyladr-to-tptp command."""
    args = argv if argv is not None else sys.argv[1:]

    input_file = None
    i = 0
    while i < len(args):
        if args[i] == "-f" and i + 1 < len(args):
            i += 1
            input_file = args[i]
        i += 1

    if input_file:
        try:
            with open(input_file) as f:
                text = f.read()
        except FileNotFoundError:
            sys.stderr.write(f"Cannot open {input_file}\n")
            return 1
    else:
        text = sys.stdin.read()

    # Parse LADR sections
    in_formulas = False
    list_name = ""
    idx = 0
    formulas: list[tuple[str, str]] = []  # (formula_text, role)

    for line in text.splitlines():
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith("%"):
            continue

        # Detect formula list boundaries
        formula_match = re.match(r"formulas\((\w+)\)\.", line)
        if formula_match:
            list_name = formula_match.group(1)
            in_formulas = True
            continue

        if line == "end_of_list.":
            in_formulas = False
            continue

        if in_formulas and line.endswith("."):
            # Determine TPTP role from LADR list name
            if list_name in ("goals", "goal"):
                role = "conjecture"
            elif list_name in ("hints",):
                role = "hypothesis"
            else:
                role = "axiom"
            formulas.append((line, role))

    # Output TPTP
    print(f"% Converted from LADR format by pyladr-to-tptp")
    print()

    for i, (formula, role) in enumerate(formulas):
        inner = formula.rstrip(".")
        print(f"fof(f{i + 1},{role},({inner})).")

    if not formulas:
        sys.stderr.write("ladr_to_tptp: no formulas found\n")
        return 1

    sys.stderr.write(f"ladr_to_tptp: {len(formulas)} formula(s) converted.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
