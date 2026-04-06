"""Isofilter0: remove isomorphic interpretations (basic normalization).

Modernized Python 3.13+ version of apps.src/isofilter0.c.

Like isofilter but uses simpler normalization (no discriminators).

Usage:
    pyisofilter0 [ignore_constants] [wrap] [check '<operations>'] [output '<operations>']
"""

from __future__ import annotations

import sys

from pyladr.apps.isofilter import (
    _filter_operations,
    _iso_member,
    _read_interps_from_stream,
    _remove_constants,
)
from pyladr.core.interpretation import (
    format_interp_standard,
    normal_interp,
)


def main(argv: list[str] | None = None) -> int:
    """Entry point for pyisofilter0 command.

    Identical to isofilter but uses basic normalization only.
    """
    args = argv if argv is not None else sys.argv[1:]

    ignore_constants = False
    wrap = False
    check_ops: set[str] | None = None
    output_ops: set[str] | None = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "ignore_constants":
            ignore_constants = True
        elif arg == "wrap":
            wrap = True
        elif arg == "check" and i + 1 < len(args):
            i += 1
            check_ops = set(args[i].replace("'", "").replace('"', "").split())
        elif arg == "output" and i + 1 < len(args):
            i += 1
            output_ops = set(args[i].replace("'", "").replace('"', "").split())
        i += 1

    text = sys.stdin.read()
    interps = _read_interps_from_stream(text)

    if wrap:
        print("list(interpretations).")
        print()

    kept = []
    num_kept = 0

    for interp in interps:
        check_interp = interp
        if ignore_constants:
            check_interp = _remove_constants(check_interp)
        if check_ops is not None:
            check_interp = _filter_operations(check_interp, check_ops)

        check_interp = normal_interp(check_interp)

        if not _iso_member(check_interp, kept):
            kept.append(check_interp)
            num_kept += 1

            out_interp = interp
            if output_ops is not None:
                out_interp = _filter_operations(out_interp, output_ops)

            print(format_interp_standard(out_interp))
            print()

    if wrap:
        print("end_of_list.")

    sys.stderr.write(f"isofilter0: input={len(interps)}, kept={num_kept}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
