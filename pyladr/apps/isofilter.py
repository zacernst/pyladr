"""Isofilter: remove isomorphic interpretations from a stream.

Modernized Python 3.13+ version of apps.src/isofilter.c.

Reads interpretations from stdin and outputs only non-isomorphic ones.

Usage:
    pyisofilter [ignore_constants] [wrap] [check '<operations>'] [output '<operations>']
"""

from __future__ import annotations

import sys
import time

from pyladr.core.interpretation import (
    Interpretation,
    OperationTable,
    compile_interp_from_text,
    format_interp_standard,
    isomorphic_interps,
    normal_interp,
)


def _remove_constants(interp: Interpretation) -> Interpretation:
    """Remove constant (arity 0) operations from interpretation."""
    new = Interpretation(size=interp.size)
    new.comments = interp.comments
    for op in interp.operations.values():
        if op.arity > 0:
            new.add_operation(
                OperationTable(
                    name=op.name,
                    arity=op.arity,
                    table_type=op.table_type,
                    values=list(op.values),
                )
            )
    return new


def _filter_operations(
    interp: Interpretation, names: set[str], keep: bool = True
) -> Interpretation:
    """Filter operations by name. If keep=True, keep only listed; if False, remove listed."""
    new = Interpretation(size=interp.size)
    new.comments = interp.comments
    for op in interp.operations.values():
        in_set = op.name in names
        if (keep and in_set) or (not keep and not in_set):
            new.add_operation(
                OperationTable(
                    name=op.name,
                    arity=op.arity,
                    table_type=op.table_type,
                    values=list(op.values),
                )
            )
    return new


def _read_interps_from_stream(text: str) -> list[Interpretation]:
    """Parse all interpretations from text."""
    interps: list[Interpretation] = []
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


def _iso_member(
    candidate: Interpretation, kept: list[Interpretation]
) -> bool:
    """Check if candidate is isomorphic to any in kept list."""
    for k in kept:
        if isomorphic_interps(candidate, k):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    """Entry point for pyisofilter command."""
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
        else:
            sys.stderr.write(f"Unknown argument: {arg}\n")
        i += 1

    text = sys.stdin.read()
    interps = _read_interps_from_stream(text)

    start_time = time.time()

    if wrap:
        print("list(interpretations).")
        print()

    kept: list[Interpretation] = []
    num_input = 0
    num_kept = 0

    for interp in interps:
        num_input += 1

        # Prepare check version (for isomorphism testing)
        check_interp = interp
        if ignore_constants:
            check_interp = _remove_constants(check_interp)
        if check_ops is not None:
            check_interp = _filter_operations(check_interp, check_ops)

        # Normalize for faster iso checking
        check_interp = normal_interp(check_interp)

        if not _iso_member(check_interp, kept):
            kept.append(check_interp)
            num_kept += 1

            # Output version may differ from check version
            out_interp = interp
            if output_ops is not None:
                out_interp = _filter_operations(out_interp, output_ops)

            print(format_interp_standard(out_interp))
            print()

    if wrap:
        print("end_of_list.")

    elapsed = time.time() - start_time
    sys.stderr.write(
        f"isofilter: input={num_input}, kept={num_kept}, "
        f"checks={num_input * num_kept}, seconds={elapsed:.2f}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
