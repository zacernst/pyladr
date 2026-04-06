"""Rewriter2: programmed rewriting with dollar-expression rules.

Modernized Python 3.13+ version of apps.src/rewriter2.c.

Usage:
    pyrewriter2 <rule_file>

Note: Requires programmed_rewrite() infrastructure. Stub.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point for pyrewriter2 command."""
    sys.stderr.write(
        "rewriter2: requires programmed rewrite infrastructure "
        "(not yet implemented)\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
