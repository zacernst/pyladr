"""Shared CLI infrastructure for PyLADR auxiliary applications.

Provides common patterns matching the C apps' argument handling:
- Help string display with version info
- stdin/file input dispatch
- Clause reading/writing loops
- Statistics reporting
- Exit code conventions (0=success with output, 1=error/help, 2=no output)
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import IO, TextIO

from pyladr import __version__
from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term
from pyladr.parsing.ladr_parser import LADRParser, ParseError
import logging

_logger = logging.getLogger(__name__)

VERSION_DATE = f"PyLADR {__version__}"


def _configure_runtime() -> None:
    """Configure runtime settings. Call from CLI entry points."""
    sys.setrecursionlimit(5000)


def make_base_parser(program_name: str, description: str) -> argparse.ArgumentParser:
    """Create a base argument parser matching C app conventions.

    C apps use string_member() for flag detection. We use argparse
    but maintain the same interface: positional flags without dashes
    plus optional -f <file> input.
    """
    parser = argparse.ArgumentParser(
        prog=program_name,
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-f",
        dest="input_file",
        metavar="FILE",
        help="Input file (default: stdin)",
    )
    return parser


def open_input(args: argparse.Namespace) -> IO[str]:
    """Open the input file or return stdin.

    Note: caller is responsible for closing the returned file handle
    when it is not stdin. Consider using the returned handle in a
    ``with`` block or wrapping with contextlib.closing().
    """
    if hasattr(args, "input_file") and args.input_file:
        try:
            fh = open(args.input_file)  # noqa: SIM115
            return fh
        except FileNotFoundError:
            fatal_error(f"file {args.input_file} not found")
    return sys.stdin


def fatal_error(message: str) -> None:
    """Print error and exit, matching C fatal_error()."""
    print(f"Fatal error: {message}", file=sys.stderr)
    sys.exit(1)


def read_clause_stream(
    input_stream: IO[str],
    symbol_table: SymbolTable | None = None,
) -> list[Clause]:
    """Read clauses from a stream until end-of-list or EOF.

    Matches C read_clause() loop pattern used by most apps:
    reads clauses separated by periods, stops at end_of_list or EOF.
    """
    _configure_runtime()
    parser = LADRParser(symbol_table)
    text = input_stream.read()
    clauses: list[Clause] = []

    # Check for list wrapper
    if "formulas(" in text or "clauses(" in text:
        try:
            parsed = parser.parse_input(text)
            clauses.extend(parsed.all_clauses)
        except ParseError as e:
            fatal_error(f"parse error: {e}")
    else:
        # Bare clause stream: split on periods
        statements = text.replace("\n", " ").split(".")
        for line_num, stmt in enumerate(statements, 1):
            stmt = stmt.strip()
            if not stmt or stmt.startswith("%") or stmt == "end_of_list":
                continue
            try:
                clause = parser.parse_clause_from_string(stmt)
                clauses.append(clause)
            except ParseError as e:
                _logger.warning("Skipping unparseable statement at position %d: %r (%s)", line_num, stmt, e)
                continue

    return clauses


def format_clause_bare(clause: Clause, symbol_table: SymbolTable | None = None) -> str:
    """Format a clause in CL_FORM_BARE style (no ID, no justification).

    Matches C fwrite_clause(stdout, c, CL_FORM_BARE).
    """
    if clause.is_empty:
        return "$F."
    lit_strs = [_format_literal(lit, symbol_table) for lit in clause.literals]
    return " | ".join(lit_strs) + "."


def _format_literal(lit: Literal, symbol_table: SymbolTable | None = None) -> str:
    """Format a literal."""
    atom_str = lit.atom.to_str(symbol_table)
    if lit.sign:
        return atom_str
    return f"-{atom_str}"


def format_clause_standard(
    clause: Clause, symbol_table: SymbolTable | None = None
) -> str:
    """Format a clause in CL_FORM_STD style (with ID and justification).

    Matches C fwrite_clause(stdout, c, CL_FORM_STD).
    """
    parts: list[str] = []
    if clause.id > 0:
        parts.append(f"{clause.id} ")
    if clause.is_empty:
        parts.append("$F")
    else:
        lit_strs = [_format_literal(lit, symbol_table) for lit in clause.literals]
        parts.append(" | ".join(lit_strs))

    # Add justification
    if clause.justification:
        just = clause.justification[0]
        parts.append(f".  [{just.just_type.name.lower()}]")
    else:
        parts.append(".")

    return "".join(parts)


def format_clause_parents_only(
    clause: Clause, symbol_table: SymbolTable | None = None
) -> str:
    """Format a clause showing only parents (CL_FORM_PARENTS)."""
    parts: list[str] = []
    if clause.id > 0:
        parts.append(f"{clause.id} ")
    if clause.is_empty:
        parts.append("$F")
    else:
        lit_strs = [_format_literal(lit, symbol_table) for lit in clause.literals]
        parts.append(" | ".join(lit_strs))

    # Add parent IDs from justification
    if clause.justification:
        just = clause.justification[0]
        parent_ids = []
        if just.clause_id:
            parent_ids.append(str(just.clause_id))
        parent_ids.extend(str(cid) for cid in just.clause_ids)
        if just.para:
            parent_ids.extend([str(just.para.from_id), str(just.para.into_id)])
        if parent_ids:
            parts.append(f"  [{','.join(parent_ids)}]")

    parts.append(".")
    return "".join(parts)


def print_separator(label: str, end: bool = False) -> None:
    """Print separator line matching C print_separator()."""
    prefix = "============================== " if not end else "============================== end of "
    suffix = " =============================="
    print(f"{prefix}{label}{suffix}")


def report_stats(
    program_name: str,
    start_time: float,
    **kwargs: int,
) -> None:
    """Print statistics line matching C app output conventions."""
    elapsed = time.time() - start_time
    stats = ", ".join(f"{k} {v}" for k, v in kwargs.items())
    print(f"% {program_name}: {stats}, {elapsed:.2f} seconds.")


def copy_clause(clause: Clause) -> Clause:
    """Create a deep copy of a clause with new term instances.

    Matches C copy_clause(). Since Terms are frozen dataclasses,
    we only need to rebuild the clause structure.
    """
    return Clause(
        literals=tuple(
            Literal(sign=lit.sign, atom=_copy_term(lit.atom))
            for lit in clause.literals
        ),
        id=clause.id,
        weight=clause.weight,
        justification=clause.justification,
    )


def _copy_term(t: Term) -> Term:
    """Deep copy a term tree."""
    if t.is_variable or t.is_constant:
        return t  # immutable, sharing is fine
    new_args = tuple(_copy_term(a) for a in t.args)
    return Term(private_symbol=t.private_symbol, arity=t.arity, args=new_args)
