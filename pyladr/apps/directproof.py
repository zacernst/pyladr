"""Directproof: transform proofs into direct (forward) form.

Modernized Python 3.13+ version of apps.src/directproof.c.

Reads proofs from stdin and reformats bidirectional inference
patterns into direct forward-chaining proofs.

Usage:
    pydirectproof

Note: Full implementation requires the proof expansion and
clause linking infrastructure. Currently provides a stub.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point for pydirectproof command."""
    sys.stderr.write(
        "directproof: requires proof expansion infrastructure "
        "(not yet implemented)\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
