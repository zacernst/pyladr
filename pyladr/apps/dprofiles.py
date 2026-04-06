"""Dprofiles: compute and display interpretation profiles.

Modernized Python 3.13+ version of apps.src/dprofiles.c.

Reads interpretations from stdin, normalizes them, and outputs
canonical profile information.

Usage:
    pydprofiles
"""

from __future__ import annotations

import sys

from pyladr.core.interpretation import (
    compile_interp_from_text,
    format_interp_standard,
    normal_interp,
)


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
    """Entry point for pydprofiles command."""
    text = sys.stdin.read()
    interps = _read_interps_from_stream(text)

    if not interps:
        sys.stderr.write("No interpretations found.\n")
        return 1

    for idx, interp in enumerate(interps):
        normed = normal_interp(interp)
        print(f"% Interpretation {idx + 1}:")
        print(f"% Occurrences: {normed.occurrences}")
        print(format_interp_standard(normed))
        print()

    sys.stderr.write(f"dprofiles: {len(interps)} interpretation(s) processed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
