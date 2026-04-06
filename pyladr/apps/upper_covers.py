"""Upper-covers: compute covering relation from lattice interpretations.

Modernized Python 3.13+ version of apps.src/upper-covers.c.

Reads lattice interpretations from stdin, extracts the meet operation,
builds the ordering, and outputs the upper-covers (Hasse diagram edges).

Usage:
    pyupper-covers
"""

from __future__ import annotations

import sys

from pyladr.core.interpretation import (
    Interpretation,
    compile_interp_from_text,
)


def _extract_meet_table(interp: Interpretation) -> list[int] | None:
    """Extract the meet operation table (usually named 'meet' or '^')."""
    for name in ("meet", "^", "m"):
        op = interp.get_table(name, 2)
        if op is not None:
            return op.values
    # Fallback: first binary function
    for op in interp.operations.values():
        if op.arity == 2:
            return op.values
    return None


def compute_upper_covers(interp: Interpretation) -> list[list[int]]:
    """Compute the upper-covers (covering relation) of a lattice.

    Given a meet operation, x <= y iff meet(x,y) = x.
    An upper cover of x is y where x < y and no z with x < z < y.
    """
    n = interp.size
    meet_table = _extract_meet_table(interp)
    if meet_table is None:
        return [[] for _ in range(n)]

    # Build less-than relation
    leq = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            # i <= j iff meet(i,j) = i
            leq[i][j] = meet_table[i * n + j] == i

    # Build strict less-than
    lt = [[leq[i][j] and i != j for j in range(n)] for i in range(n)]

    # Remove transitive edges to get covering relation
    covers: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if lt[i][j]:
                # Check if there's a z with i < z < j
                is_cover = True
                for z in range(n):
                    if lt[i][z] and lt[z][j]:
                        is_cover = False
                        break
                if is_cover:
                    covers[i].append(j)

    return covers


def _read_interps_from_stream(text: str) -> list:
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
    """Entry point for pyupper-covers command."""
    text = sys.stdin.read()
    interps = _read_interps_from_stream(text)

    if not interps:
        sys.stderr.write("No interpretations found.\n")
        return 1

    for idx, interp in enumerate(interps):
        covers = compute_upper_covers(interp)
        print(f"% Interpretation {idx + 1} (size {interp.size}):")
        for i, upper in enumerate(covers):
            if upper:
                print(f"  {i}: {upper}")
            else:
                print(f"  {i}: []")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
