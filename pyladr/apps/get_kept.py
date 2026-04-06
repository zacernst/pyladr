"""Get_kept: extract kept clauses from Prover9 output.

Matches C utilities/get_kept shell script — reads Prover9 output from
stdin and extracts kept clause lines.

Usage:
    pyget-kept < prover9_output
"""

from __future__ import annotations

import re
import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point for pyget-kept command."""
    for line in sys.stdin:
        if re.search(r"kept\s*:", line, re.IGNORECASE):
            sys.stdout.write(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
