"""Get_interps: extract interpretations from Mace4 output.

Matches C utilities/get_interps shell script — reads Mace4 output from
stdin and extracts interpretations between MODEL markers.

Usage:
    pyget-interps < mace4_output
"""

from __future__ import annotations

import re
import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point for pyget-interps command."""
    collecting = False

    for line in sys.stdin:
        if re.search(r"= MODEL =", line):
            collecting = True
            continue
        if re.search(r"= end of model =", line):
            collecting = False
            print()
            continue
        if collecting:
            sys.stdout.write(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
