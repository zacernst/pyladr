"""Interpretation formatter (modelformat): reformat interpretations.

Modernized Python 3.13+ version of apps.src/interpformat.c.

Reads interpretations from stdin or file and outputs in a specified format.

Usage:
    pyinterpformat [format] [-f file]

Formats: standard, standard2, portable, tabular, raw, cooked, tex, xml
Default: standard2
"""

from __future__ import annotations

import sys

from pyladr.core.interpretation import (
    compile_interp_from_text,
    format_interp_cooked,
    format_interp_portable,
    format_interp_raw,
    format_interp_standard,
    format_interp_standard2,
    format_interp_tabular,
    format_interp_tex,
    format_interp_xml,
)

FORMATTERS = {
    "standard": format_interp_standard,
    "standard2": format_interp_standard2,
    "portable": format_interp_portable,
    "tabular": format_interp_tabular,
    "raw": format_interp_raw,
    "cooked": format_interp_cooked,
    "tex": format_interp_tex,
    "xml": format_interp_xml,
}


def _read_interps_from_stream(text: str) -> list:
    """Parse all interpretations from text."""
    interps = []

    # Try to find interpretation(...). blocks
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

    # If nothing found, try Mace4 output format (between MODEL markers)
    if not interps:
        import re

        model_blocks = re.findall(
            r"={10,}\s*MODEL\s*={10,}(.*?)={10,}\s*end of model\s*={10,}",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        for block in model_blocks:
            # Try to find interpretation inside the model block
            if "interpretation(" in block:
                try:
                    interp = compile_interp_from_text(block.strip())
                    interps.append(interp)
                except ValueError:
                    continue

    return interps


def main(argv: list[str] | None = None) -> int:
    """Entry point for pyinterpformat command."""
    args = argv if argv is not None else sys.argv[1:]

    fmt_name = "standard2"
    input_file = None
    wrap = False
    i = 0

    while i < len(args):
        arg = args[i]
        if arg in FORMATTERS:
            fmt_name = arg
        elif arg == "-f" and i + 1 < len(args):
            i += 1
            input_file = args[i]
        elif arg == "wrap":
            wrap = True
        else:
            sys.stderr.write(f"Unknown argument: {arg}\n")
            return 1
        i += 1

    formatter = FORMATTERS[fmt_name]

    if input_file:
        try:
            with open(input_file) as f:
                text = f.read()
        except FileNotFoundError:
            sys.stderr.write(f"Cannot open {input_file}\n")
            return 1
    else:
        text = sys.stdin.read()

    interps = _read_interps_from_stream(text)

    if not interps:
        sys.stderr.write("No interpretations found.\n")
        return 1

    if wrap:
        print("list(interpretations).")
        print()

    for idx, interp in enumerate(interps):
        if idx > 0:
            print()
        print(formatter(interp))

    if wrap:
        print()
        print("end_of_list.")

    sys.stderr.write(f"interpformat: {len(interps)} interpretation(s) formatted as {fmt_name}.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
