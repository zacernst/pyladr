"""Get_givens: extract given clauses from Prover9 output.

Matches C utilities/get_givens shell script — reads Prover9 output from
stdin and extracts given clause lines.

Usage:
    pyget-givens < prover9_output
"""

from __future__ import annotations

import re
import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point for pyget-givens command."""
    for line in sys.stdin:
        if re.search(r"given\s*#?\d+", line, re.IGNORECASE):
            sys.stdout.write(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
