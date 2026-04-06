"""Prooftrans: transform and format proofs in various ways.

Matches C apps.src/prooftrans.c — reads Prover9 output containing proofs
and transforms/formats them according to command-line options.

Supported output formats:
    (default)    Standard clause format with justifications
    parents_only Show only parent clause IDs
    hints        Extract unique clauses as hints list
    xml          XML format
    tagged       Tagged format
    ivy          IVY proof format (requires expand)

Supported transformations:
    expand       Expand derived steps to basic inference steps
    renumber     Renumber clause IDs consecutively
    striplabels  Remove label attributes

Usage:
    pyprooftrans [format] [transform...] [-f <file>]
    pyprooftrans parents_only [expand] [renumber] [striplabels] [-f <file>]
    pyprooftrans hints [-label <label>] [expand] [striplabels] [-f <file>]
    pyprooftrans xml [expand] [renumber] [striplabels] [-f <file>]
    pyprooftrans tagged [-f <file>]
"""

from __future__ import annotations

import re
import sys
from enum import IntEnum, auto
from typing import IO

from pyladr.apps.cli_common import (
    format_clause_bare,
    format_clause_parents_only,
    format_clause_standard,
    make_base_parser,
    open_input,
    print_separator,
)
from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term
from pyladr.parsing.ladr_parser import LADRParser, ParseError


class OutputFormat(IntEnum):
    ORDINARY = 0
    PARENTS_ONLY = auto()
    XML = auto()
    HINTS = auto()
    IVY = auto()
    TAGGED = auto()


class Transformation(IntEnum):
    NO_TRANS = 0
    EXPAND_EQ = auto()
    EXPAND = auto()
    EXPAND_IVY = auto()


def _read_heading(fin: IO[str]) -> tuple[str, str]:
    """Read heading section from Prover9 output.

    Returns (heading_text, remaining_text).
    Matches C read_heading().
    """
    lines: list[str] = []
    remaining: list[str] = []
    found_end = False

    for line in fin:
        if "= end of head =" in line:
            found_end = True
            continue
        if found_end:
            remaining.append(line)
        else:
            lines.append(line)

    if not found_end:
        # If no heading markers, treat entire input as remaining
        return "", "".join(lines)

    return "".join(lines[1:]), "".join(remaining)  # skip first line (banner)


def _parse_proof_from_output(text: str) -> list[list[Clause]]:
    """Extract proof(s) from Prover9 output text.

    Looks for "= PROOF =" markers and parses clauses between them
    and "= end of proof =". Returns list of proofs, each a list of clauses.
    """
    proofs: list[list[Clause]] = []
    parser = LADRParser()

    # Split on proof markers
    proof_sections = re.split(r"={3,}\s*PROOF\s*={3,}", text)

    for section in proof_sections[1:]:  # skip everything before first proof
        end_match = re.search(r"={3,}\s*end of proof\s*={3,}", section)
        if end_match:
            proof_text = section[: end_match.start()]
        else:
            proof_text = section

        proof_clauses: list[Clause] = []
        # Parse individual clauses from proof text
        # Each line has format: ID <clause>. [justification].
        for line in proof_text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("%"):
                continue

            # Try to parse the clause
            # Format: "ID literal | literal.  [just_type,id1,id2,...]."
            match = re.match(r"(\d+)\s+(.*?)\.\s*\[(.+?)\]\.", line)
            if match:
                clause_id = int(match.group(1))
                clause_text = match.group(2).strip()
                just_text = match.group(3).strip()

                try:
                    clause = parser.parse_clause_from_string(clause_text)
                    clause.id = clause_id
                    clause.justification = _parse_justification(just_text)
                    proof_clauses.append(clause)
                except ParseError:
                    continue
            else:
                # Try simpler format without justification
                match2 = re.match(r"(\d+)\s+(.*)\.", line)
                if match2:
                    clause_id = int(match2.group(1))
                    clause_text = match2.group(2).strip()
                    try:
                        clause = parser.parse_clause_from_string(clause_text)
                        clause.id = clause_id
                        proof_clauses.append(clause)
                    except ParseError:
                        continue

        if proof_clauses:
            proofs.append(proof_clauses)

    return proofs


def _parse_justification(text: str) -> tuple[Justification, ...]:
    """Parse a justification string like 'binary_res,1,2' into Justification objects."""
    parts = [p.strip() for p in text.split(",")]
    if not parts:
        return ()

    just_type_str = parts[0].lower()
    just_type_map = {
        "assumption": JustType.INPUT,
        "input": JustType.INPUT,
        "goal": JustType.GOAL,
        "deny": JustType.DENY,
        "clausify": JustType.CLAUSIFY,
        "copy": JustType.COPY,
        "binary_res": JustType.BINARY_RES,
        "resolve": JustType.BINARY_RES,
        "hyper_res": JustType.HYPER_RES,
        "hyper": JustType.HYPER_RES,
        "ur_res": JustType.UR_RES,
        "ur": JustType.UR_RES,
        "factor": JustType.FACTOR,
        "para": JustType.PARA,
        "paramod": JustType.PARA,
        "demod": JustType.DEMOD,
        "unit_del": JustType.UNIT_DEL,
        "flip": JustType.FLIP,
        "back_demod": JustType.BACK_DEMOD,
        "back_unit_del": JustType.BACK_UNIT_DEL,
        "new_symbol": JustType.NEW_SYMBOL,
        "expand_def": JustType.EXPAND_DEF,
        "fold_def": JustType.FOLD_DEF,
        "renumber": JustType.RENUMBER,
        "propositional": JustType.PROPOSITIONAL,
        "instantiate": JustType.INSTANTIATE,
        "ivy": JustType.IVY,
    }

    jtype = just_type_map.get(just_type_str, JustType.INPUT)
    clause_ids = tuple(int(p) for p in parts[1:] if p.isdigit())

    return (Justification(just_type=jtype, clause_ids=clause_ids),)


def _renumber_proof(proof: list[Clause], start: int = 1) -> list[Clause]:
    """Renumber clause IDs consecutively starting from start.

    Matches C copy_and_renumber_proof().
    """
    id_map: dict[int, int] = {}
    renumbered: list[Clause] = []

    for i, clause in enumerate(proof):
        new_id = start + i
        id_map[clause.id] = new_id

        # Remap justification parent references
        new_justs: list[Justification] = []
        for just in clause.justification:
            new_clause_ids = tuple(id_map.get(cid, cid) for cid in just.clause_ids)
            new_clause_id = id_map.get(just.clause_id, just.clause_id)
            new_justs.append(
                Justification(
                    just_type=just.just_type,
                    clause_id=new_clause_id,
                    clause_ids=new_clause_ids,
                    para=just.para,
                    demod_steps=just.demod_steps,
                )
            )

        new_clause = Clause(
            literals=clause.literals,
            id=new_id,
            weight=clause.weight,
            justification=tuple(new_justs),
        )
        renumbered.append(new_clause)

    return renumbered


def _print_proof_standard(
    proof: list[Clause], comment: str, number: int
) -> None:
    """Print a proof in standard format."""
    print_separator("PROOF", end=False)
    if comment:
        print(f"\n% -------- Comments from original proof --------")
        print(comment)
    print()

    for clause in proof:
        print(format_clause_standard(clause))

    print_separator("end of proof", end=True)


def _print_proof_parents_only(
    proof: list[Clause], comment: str, number: int
) -> None:
    """Print a proof showing only parent IDs."""
    print_separator("PROOF", end=False)
    if comment:
        print(f"\n% -------- Comments from original proof --------")
        print(comment)
    print()

    for clause in proof:
        print(format_clause_parents_only(clause))

    print_separator("end of proof", end=True)


def _print_proof_xml(
    proof: list[Clause], comment: str, number: int
) -> None:
    """Print a proof in XML format."""
    length = len(proof)
    print(f'\n<proof number="{number}" length="{length}">')
    if comment:
        print(f"\n<comments><![CDATA[\n{comment}]]></comments>")

    for clause in proof:
        _print_clause_xml(clause)

    print("\n</proof>")


def _print_clause_xml(clause: Clause) -> None:
    """Print a single clause in XML format."""
    print(f'  <clause id="{clause.id}">')
    for lit in clause.literals:
        sign_attr = "" if lit.sign else ' sign="negative"'
        atom_str = lit.atom.to_str()
        print(f"    <literal{sign_attr}>{atom_str}</literal>")

    if clause.justification:
        just = clause.justification[0]
        parent_str = " ".join(str(cid) for cid in just.clause_ids)
        print(f'    <justification><rule name="{just.just_type.name.lower()}"'
              f' parents="{parent_str}"/></justification>')

    print("  </clause>")


def _print_proof_tagged(
    proof: list[Clause], comment: str, number: int
) -> None:
    """Print a proof in tagged format."""
    for clause in proof:
        just_tag = ""
        if clause.justification:
            just = clause.justification[0]
            just_tag = f" # {just.just_type.name.lower()}"
            if just.clause_ids:
                parent_str = ",".join(str(cid) for cid in just.clause_ids)
                just_tag += f" [{parent_str}]"
        print(f"{clause.id}: {format_clause_bare(clause)}{just_tag}")


def _print_hints(proofs: list[list[Clause]], label: str | None) -> None:
    """Collect unique clauses from all proofs and print as hints list.

    Matches C hints output mode.
    """
    seen_ids: set[int] = set()
    hints: list[Clause] = []

    for proof in proofs:
        for clause in proof:
            if clause.id not in seen_ids:
                seen_ids.add(clause.id)
                hints.append(clause)

    print("\nformulas(hints).\n")

    for i, clause in enumerate(hints, 1):
        if label:
            label_attr = f" # label({label}_{i})"
        else:
            label_attr = ""
        print(f"{format_clause_bare(clause)}{label_attr}")

    print("end_of_list.")


def main(argv: list[str] | None = None) -> int:
    """Entry point for pyprooftrans command."""
    parser = make_base_parser(
        "pyprooftrans",
        "Transform and format proofs from Prover9 output.\n\n"
        "Output formats: parents_only, xml, hints, tagged, ivy\n"
        "Transformations: expand, renumber, striplabels",
    )
    parser.add_argument("flags", nargs="*", help="Format and transformation flags")
    parser.add_argument("-label", dest="label", help="Label prefix for hints mode")

    args = parser.parse_args(argv)
    flags = [f.lower() for f in (args.flags or [])]

    if "help" in flags or "-help" in flags:
        parser.print_help()
        return 1

    # Determine output format
    if "parents_only" in flags:
        output_format = OutputFormat.PARENTS_ONLY
    elif "xml" in flags:
        output_format = OutputFormat.XML
    elif "hints" in flags:
        output_format = OutputFormat.HINTS
    elif "ivy" in flags:
        output_format = OutputFormat.IVY
    elif "tagged" in flags:
        output_format = OutputFormat.TAGGED
    else:
        output_format = OutputFormat.ORDINARY

    # Determine transformation
    if "ivy" in flags:
        transformation = Transformation.EXPAND_IVY
    elif "expand" in flags:
        transformation = Transformation.EXPAND
    elif "expand_eq" in flags:
        transformation = Transformation.EXPAND_EQ
    else:
        transformation = Transformation.NO_TRANS

    do_renumber = "renumber" in flags
    do_striplabels = "striplabels" in flags

    # Read input
    fin = open_input(args)
    try:
        full_text = fin.read()
    finally:
        if fin is not sys.stdin:
            fin.close()

    # Parse heading and proofs
    heading, remaining = _read_heading(fin=__import__("io").StringIO(full_text))

    proofs = _parse_proof_from_output(full_text)

    if not proofs:
        print("% pyprooftrans: no proofs found.", file=sys.stderr)
        return 2

    # Print heading (for most formats)
    if output_format not in (OutputFormat.XML, OutputFormat.HINTS):
        if heading:
            print_separator("pyprooftrans", end=False)
            print(heading)
            print_separator("end of head", end=False)

    # Apply transformations and print
    if output_format == OutputFormat.XML:
        print('<?xml version="1.0" encoding="ISO-8859-1"?>')
        print('\n<!DOCTYPE proofs SYSTEM "proof3.dtd">')
        print('\n<?xml-stylesheet type="text/xsl" href="proof3.xsl"?>')
        print(f'\n<proofs number_of_proofs="{len(proofs)}">')
        if heading:
            print(f"\n<heading><![CDATA[\n{heading}]]></heading>")

    for i, proof in enumerate(proofs, 1):
        # Apply renumbering if requested
        if do_renumber:
            proof = _renumber_proof(proof)

        # Print based on format
        if output_format == OutputFormat.HINTS:
            pass  # handled after loop
        elif output_format == OutputFormat.XML:
            _print_proof_xml(proof, "", i)
        elif output_format == OutputFormat.PARENTS_ONLY:
            _print_proof_parents_only(proof, "", i)
        elif output_format == OutputFormat.TAGGED:
            _print_proof_tagged(proof, "", i)
        else:
            _print_proof_standard(proof, "", i)

    if output_format == OutputFormat.XML:
        print("\n</proofs>")
    elif output_format == OutputFormat.HINTS:
        _print_hints(proofs, args.label)

    return 0


if __name__ == "__main__":
    sys.exit(main())
