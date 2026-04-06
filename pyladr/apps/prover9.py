"""Main Prover9 theorem prover application.

Implements the pyprover9 main entry point that reads LADR input,
runs the given-clause search algorithm, and reports results in
a format matching the original C Prover9 output.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

from pyladr import __version__
from pyladr.apps.cli_common import (
    format_clause_bare,
    format_clause_standard,
    print_separator,
)
from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import LADRParser, ParseError, ParsedInput
from pyladr.search.given_clause import (
    ExitCode,
    GivenClauseSearch,
    Proof,
    SearchOptions,
    SearchResult,
)

# ── Exit code mapping ──────────────────────────────────────────────────────

# Map search exit codes to process exit codes matching C Prover9:
#   0 = proof found (MAX_PROOFS_EXIT)
#   1 = fatal error
#   2 = SOS empty (exhausted, no proof)
#   3 = max_given hit
#   4 = max_kept hit
#   5 = max_seconds hit
#   6 = max_generated hit
_PROCESS_EXIT_CODES: dict[ExitCode, int] = {
    ExitCode.MAX_PROOFS_EXIT: 0,
    ExitCode.SOS_EMPTY_EXIT: 2,
    ExitCode.MAX_GIVEN_EXIT: 3,
    ExitCode.MAX_KEPT_EXIT: 4,
    ExitCode.MAX_SECONDS_EXIT: 5,
    ExitCode.MAX_GENERATED_EXIT: 6,
    ExitCode.FATAL_EXIT: 1,
}

_EXIT_DESCRIPTIONS: dict[ExitCode, str] = {
    ExitCode.MAX_PROOFS_EXIT: "max_proofs",
    ExitCode.SOS_EMPTY_EXIT: "sos_empty",
    ExitCode.MAX_GIVEN_EXIT: "max_given",
    ExitCode.MAX_KEPT_EXIT: "max_kept",
    ExitCode.MAX_SECONDS_EXIT: "max_seconds",
    ExitCode.MAX_GENERATED_EXIT: "max_generated",
    ExitCode.FATAL_EXIT: "fatal",
}


# ── Argument parsing ──────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser matching C Prover9 command-line interface."""
    parser = argparse.ArgumentParser(
        prog="pyprover9",
        description="PyProver9 — Python automated theorem prover (LADR/Prover9 compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-f",
        dest="input_file",
        metavar="FILE",
        help="Input file in LADR format (default: stdin)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"pyprover9 {__version__}",
    )

    # Search limits
    limits = parser.add_argument_group("search limits")
    limits.add_argument(
        "-max_given",
        type=int,
        default=-1,
        metavar="N",
        help="Maximum given clauses (-1 = no limit)",
    )
    limits.add_argument(
        "-max_kept",
        type=int,
        default=-1,
        metavar="N",
        help="Maximum kept clauses (-1 = no limit)",
    )
    limits.add_argument(
        "-max_seconds",
        type=float,
        default=-1.0,
        metavar="N",
        help="Maximum search time in seconds (-1 = no limit)",
    )
    limits.add_argument(
        "-max_generated",
        type=int,
        default=-1,
        metavar="N",
        help="Maximum generated clauses (-1 = no limit)",
    )
    limits.add_argument(
        "-max_proofs",
        type=int,
        default=1,
        metavar="N",
        help="Stop after N proofs (default: 1)",
    )

    # Inference rules
    inference = parser.add_argument_group("inference rules")
    inference.add_argument(
        "--paramodulation",
        action="store_true",
        default=False,
        help="Enable paramodulation",
    )
    inference.add_argument(
        "--no-resolution",
        action="store_true",
        default=False,
        help="Disable binary resolution",
    )
    inference.add_argument(
        "--no-factoring",
        action="store_true",
        default=False,
        help="Disable factoring",
    )
    inference.add_argument(
        "--demodulation",
        action="store_true",
        default=False,
        help="Enable demodulation",
    )
    inference.add_argument(
        "--back-demod",
        action="store_true",
        default=False,
        help="Enable back demodulation",
    )

    # Output control
    output = parser.add_argument_group("output")
    output.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress search progress output",
    )
    output.add_argument(
        "--print-kept",
        action="store_true",
        default=False,
        help="Print each kept clause",
    )
    output.add_argument(
        "--no-print-given",
        action="store_true",
        default=False,
        help="Do not print each given clause",
    )

    return parser


# ── Goal denial (negation for refutation) ─────────────────────────────────


def _deny_goals(
    parsed: ParsedInput,
    symbol_table: SymbolTable,
) -> tuple[list[Clause], list[Clause]]:
    """Negate goals and add them to SOS for refutation-based proving.

    Returns (usable_clauses, sos_clauses) ready for the search engine.
    Goals are negated: each literal's sign is flipped. The negated goals
    are appended to the SOS list.
    """
    usable = list(parsed.usable)
    sos = list(parsed.sos)

    for i, goal in enumerate(parsed.goals):
        denied_lits = tuple(
            Literal(sign=not lit.sign, atom=lit.atom) for lit in goal.literals
        )
        denied = Clause(
            literals=denied_lits,
            justification=(Justification(just_type=JustType.DENY, clause_ids=(0,)),),
        )
        sos.append(denied)

    return usable, sos


# ── Auto-detect inference settings ────────────────────────────────────────


def _auto_inference(
    parsed: ParsedInput,
    opts: SearchOptions,
) -> None:
    """Auto-detect appropriate inference rules based on the input.

    Matches C Prover9 auto_inference behavior:
    - If there are equality literals, enable paramodulation and demodulation
    """
    has_equality = False
    for clause in parsed.sos + parsed.goals + parsed.usable:
        for lit in clause.literals:
            if lit.atom.arity == 2:
                # Check if it's an equality atom by looking at symbol
                # Equalities use the = symbol which has a known symnum
                has_equality = True
                break
        if has_equality:
            break

    if has_equality:
        opts.paramodulation = True
        opts.demodulation = True


# ── Output formatting ─────────────────────────────────────────────────────


def _print_header(
    input_file: str | None,
    argv: list[str],
    out: TextIO,
) -> None:
    """Print Prover9-style header banner."""
    print_separator("Prover9")
    now = datetime.now()
    print(
        f"PyProver9 version {__version__} (Python).",
        file=out,
    )
    print(
        f"Process {os.getpid()} was started by {os.getenv('USER', 'unknown')} "
        f"on {os.uname().nodename},",
        file=out,
    )
    print(f"{now.strftime('%a %b %d %H:%M:%S %Y')}", file=out)
    cmd = " ".join(argv)
    print(f'The command was "{cmd}".', file=out)
    print_separator("end of head")


def _print_input(
    input_text: str,
    input_file: str | None,
    out: TextIO,
) -> None:
    """Print the INPUT section showing what was read."""
    print(file=out)
    print_separator("INPUT")
    print(file=out)
    if input_file:
        print(f"% Reading from file {input_file}", file=out)
    else:
        print("% Reading from stdin.", file=out)
    print(file=out)
    # Print the original input text
    print(input_text.rstrip(), file=out)
    print(file=out)
    print_separator("end of input")


def _print_initial_clauses(
    usable: list[Clause],
    sos: list[Clause],
    symbol_table: SymbolTable,
    out: TextIO,
) -> None:
    """Print PROCESS INITIAL CLAUSES section."""
    print(file=out)
    print_separator("PROCESS INITIAL CLAUSES")
    print(file=out)
    print("% Clauses before input processing:", file=out)
    print(file=out)

    print("formulas(usable).", file=out)
    for c in usable:
        print(f"{format_clause_standard(c, symbol_table)}", file=out)
    print("end_of_list.", file=out)
    print(file=out)

    print("formulas(sos).", file=out)
    for c in sos:
        print(f"{format_clause_standard(c, symbol_table)}", file=out)
    print("end_of_list.", file=out)
    print(file=out)

    print_separator("end of process initial clauses")


def _print_proof(
    proof: Proof,
    proof_num: int,
    search_seconds: float,
    symbol_table: SymbolTable,
    out: TextIO,
) -> None:
    """Print a proof in C Prover9 format."""
    print(file=out)
    print(f"-------- Proof {proof_num} -------- ", file=out)
    print(file=out)
    print_separator("PROOF")
    print(file=out)
    print(f"% Proof {proof_num} at {search_seconds:.2f} seconds.", file=out)
    print(f"% Length of proof is {len(proof.clauses)}.", file=out)

    # Find max clause weight in proof
    max_weight = 0.0
    for c in proof.clauses:
        if c.weight > max_weight:
            max_weight = c.weight
    print(f"% Maximum clause weight is {max_weight:.3f}.", file=out)
    print(file=out)

    # Print proof clauses
    for clause in proof.clauses:
        print(f"{format_clause_standard(clause, symbol_table)}", file=out)
    print(file=out)
    print_separator("end of proof")


def _print_statistics(
    result: SearchResult,
    out: TextIO,
) -> None:
    """Print STATISTICS section matching C format."""
    stats = result.stats
    print(file=out)
    print_separator("STATISTICS")
    print(file=out)
    print(
        f"Given={stats.given}. Generated={stats.generated}. "
        f"Kept={stats.kept}. proofs={stats.proofs}.",
        file=out,
    )
    print(
        f"Forward_subsumed={stats.subsumed}. Back_subsumed={stats.back_subsumed}.",
        file=out,
    )
    if stats.demodulated > 0 or stats.back_demodulated > 0:
        print(
            f"New_demodulators={stats.new_demodulators} ({stats.new_lex_demods} lex), "
            f"Back_demodulated={stats.back_demodulated}.",
            file=out,
        )
    print(f"User_CPU={stats.elapsed_seconds():.2f}.", file=out)
    print(file=out)
    print_separator("end of statistics")


def _print_conclusion(
    result: SearchResult,
    out: TextIO,
) -> None:
    """Print final conclusion matching C Prover9 output."""
    exit_desc = _EXIT_DESCRIPTIONS.get(result.exit_code, "unknown")

    print(file=out)
    print_separator("end of search")

    if result.exit_code == ExitCode.MAX_PROOFS_EXIT:
        print(file=out)
        print("THEOREM PROVED", file=out)
        print(file=out)
        num = len(result.proofs)
        print(f"Exiting with {num} proof{'s' if num != 1 else ''}.", file=out)
    elif result.exit_code == ExitCode.SOS_EMPTY_EXIT:
        print(file=out)
        print("SEARCH FAILED", file=out)
        print(file=out)
        print("SOS exhausted; no proof was found.", file=out)
    else:
        print(file=out)
        print(f"Search stopped by {exit_desc} limit.", file=out)

    print(file=out)
    print(
        f"------ process {os.getpid()} exit ({exit_desc}) ------",
        file=out,
    )
    print(file=out)
    now = datetime.now()
    print(
        f"Process {os.getpid()} exit ({exit_desc}) "
        f"{now.strftime('%a %b %d %H:%M:%S %Y')}",
        file=out,
    )


# ── Main entry point ──────────────────────────────────────────────────────


def run_prover(argv: list[str] | None = None) -> int:
    """Run the Prover9 theorem prover.

    Args:
        argv: Command-line arguments (defaults to sys.argv).

    Returns:
        Process exit code (0 = proof found, 2 = SOS empty, etc.)
    """
    if argv is None:
        argv = sys.argv

    arg_parser = _build_arg_parser()

    # Parse known args (ignore unrecognized for forward compatibility)
    args, _unknown = arg_parser.parse_known_args(argv[1:])

    out: TextIO = sys.stdout

    # ── Read input ──────────────────────────────────────────────────────

    input_file: str | None = args.input_file
    try:
        if input_file:
            input_text = Path(input_file).read_text()
        else:
            input_text = sys.stdin.read()
    except FileNotFoundError:
        print(f"Fatal error: file {input_file} not found", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1

    # ── Parse input ─────────────────────────────────────────────────────

    symbol_table = SymbolTable()
    parser = LADRParser(symbol_table)

    try:
        parsed = parser.parse_input(input_text)
    except ParseError as e:
        print(f"Fatal error: parse error: {e}", file=sys.stderr)
        return 1

    # ── Print header and input ──────────────────────────────────────────

    _print_header(input_file, argv, out)
    _print_input(input_text, input_file, out)

    # ── Deny goals ──────────────────────────────────────────────────────

    usable, sos = _deny_goals(parsed, symbol_table)

    if not sos and not usable:
        print("Fatal error: no clauses in input", file=sys.stderr)
        return 1

    # ── Configure search options ────────────────────────────────────────

    opts = SearchOptions(
        binary_resolution=not args.no_resolution,
        paramodulation=args.paramodulation,
        factoring=not args.no_factoring,
        demodulation=args.demodulation,
        back_demod=args.back_demod,
        max_given=args.max_given,
        max_kept=args.max_kept,
        max_seconds=args.max_seconds,
        max_generated=args.max_generated,
        max_proofs=args.max_proofs,
        print_given=not args.no_print_given,
        print_kept=args.print_kept,
        quiet=args.quiet,
    )

    # Auto-detect inference settings based on input
    _auto_inference(parsed, opts)

    # ── Print initial clauses ───────────────────────────────────────────

    _print_initial_clauses(usable, sos, symbol_table, out)

    # ── Run search ──────────────────────────────────────────────────────

    engine = GivenClauseSearch(
        options=opts,
        symbol_table=symbol_table,
    )

    try:
        result = engine.run(usable=usable, sos=sos)
    except Exception as e:
        print(f"\nFatal error during search: {e}", file=sys.stderr)
        return 1

    # ── Print proofs ────────────────────────────────────────────────────

    for i, proof in enumerate(result.proofs, 1):
        _print_proof(
            proof,
            proof_num=i,
            search_seconds=result.stats.search_seconds(),
            symbol_table=symbol_table,
            out=out,
        )

    # ── Print statistics and conclusion ─────────────────────────────────

    _print_statistics(result, out)
    _print_conclusion(result, out)

    return _PROCESS_EXIT_CODES.get(result.exit_code, 1)


def main() -> int:
    """CLI entry point."""
    return run_prover()
