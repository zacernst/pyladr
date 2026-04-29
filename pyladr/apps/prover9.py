"""Main Prover9 theorem prover application.

Implements the pyprover9 main entry point that reads LADR input,
runs the given-clause search algorithm, and reports results in
a format matching the original C Prover9 output.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

from pyladr import __version__
from pyladr.apps.cli_common import (
    _configure_runtime,
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

# ── C++ backend activation ────────────────────────────────────────────────

def _configure_cpp_backend(use_cpp: bool) -> None:
    """Activate the C++ backend when --cpp is passed."""
    if not use_cpp:
        return
    from pyladr.cpp_backend import enable
    if enable():
        print("% C++ backend enabled (_pyladr_core)")


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
    limits.add_argument(
        "-max_weight",
        type=float,
        default=-1.0,
        metavar="W",
        help="Discard clauses heavier than W (-1 = no limit). "
             "Also settable via assign(max_weight, W). in the input file.",
    )
    limits.add_argument(
        "--max-weight-tighten-after",
        type=int,
        default=0,
        metavar="N",
        dest="max_weight_tighten_after",
        help="After N given clauses, tighten max_weight to --max-weight-tighten-to. "
             "0 = disabled (default).",
    )
    limits.add_argument(
        "--max-weight-tighten-to",
        type=float,
        default=-1.0,
        metavar="W",
        dest="max_weight_tighten_to",
        help="New max_weight cap applied after --max-weight-tighten-after given clauses.",
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

    # Selection strategy
    selection = parser.add_argument_group("selection strategy")
    selection.add_argument(
        "--weight-ratio",
        type=int,
        default=4,
        metavar="N",
        dest="weight_ratio",
        help="Number of weight-based selections per age-based selection (default: 4). "
             "Combined with 1 age slot gives a cycle of 1+N. "
             "Equivalent to C Prover9's age_factor=N+1.",
    )
    selection.add_argument(
        "--penalty-propagation",
        action="store_true",
        default=False,
        help="Enable parent-to-child penalty propagation for overly general clauses.",
    )
    selection.add_argument(
        "--penalty-propagation-mode",
        type=str,
        default="additive",
        choices=["additive", "multiplicative", "max"],
        metavar="MODE",
        help="Penalty combination mode: additive (default), multiplicative, or max.",
    )
    selection.add_argument(
        "--penalty-propagation-decay",
        type=float,
        default=0.5,
        metavar="D",
        help="Decay factor per generation for inherited penalties (0.0-1.0, default: 0.5).",
    )
    selection.add_argument(
        "--penalty-propagation-threshold",
        type=float,
        default=5.0,
        metavar="T",
        help="Minimum parent penalty to trigger propagation (default: 5.0).",
    )
    selection.add_argument(
        "--unification-weight",
        type=int,
        default=0,
        metavar="N",
        help="Ratio weight for generality-penalty clause selection (0 = disabled, default: 0). "
             "Adds N penalty selections per cycle, preferring specific clauses to reduce "
             "unification explosion.",
    )
    selection.add_argument(
        "--repetition-penalty",
        action="store_true",
        default=False,
        help="Enable subformula repetition penalty (deprioritize clauses with repeated patterns).",
    )
    selection.add_argument(
        "--repetition-penalty-weight",
        type=float,
        default=2.0,
        metavar="W",
        help="Penalty per extra occurrence of a repeated subformula (default: 2.0).",
    )
    selection.add_argument(
        "--repetition-penalty-min-size",
        type=int,
        default=2,
        metavar="S",
        help="Minimum subterm size (symbol count) to consider for repetition (default: 2).",
    )
    selection.add_argument(
        "--repetition-penalty-max",
        type=float,
        default=15.0,
        metavar="M",
        help="Maximum total repetition penalty per clause (default: 15.0).",
    )
    selection.add_argument(
        "--repetition-penalty-normalize",
        action="store_true",
        default=False,
        help="Enable variable-agnostic matching (i(x,x) ≡ i(y,y)).",
    )
    selection.add_argument(
        "--nucleus-unification-penalty",
        action="store_true",
        default=False,
        help="Enable nucleus unification penalty (deprioritize overly permissive hyperresolution nuclei).",
    )
    selection.add_argument(
        "--nucleus-penalty-threshold",
        type=float,
        default=3.0,
        metavar="T",
        help="Minimum nucleus penalty to apply (default: 3.0).",
    )
    selection.add_argument(
        "--nucleus-penalty-weight",
        type=float,
        default=1.5,
        metavar="W",
        help="Base penalty per overly general nucleus literal (default: 1.5).",
    )
    selection.add_argument(
        "--nucleus-penalty-max",
        type=float,
        default=15.0,
        metavar="M",
        help="Maximum total nucleus penalty per clause (default: 15.0).",
    )
    selection.add_argument(
        "--nucleus-penalty-cache-size",
        type=int,
        default=10000,
        metavar="N",
        help="Maximum cached nucleus patterns for LRU eviction (default: 10000).",
    )
    selection.add_argument(
        "--penalty-weight",
        action="store_true",
        default=False,
        help="Enable penalty-based clause weight adjustment (deprioritize high-penalty clauses).",
    )
    selection.add_argument(
        "--penalty-weight-threshold",
        type=float,
        default=5.0,
        metavar="T",
        help="Minimum combined penalty to trigger weight adjustment (default: 5.0).",
    )
    selection.add_argument(
        "--penalty-weight-multiplier",
        type=float,
        default=2.0,
        metavar="M",
        help="Weight increase factor (>= 1.0, default: 2.0).",
    )
    selection.add_argument(
        "--penalty-weight-max",
        type=float,
        default=1000.0,
        metavar="W",
        help="Maximum adjusted clause weight cap (default: 1000.0).",
    )
    selection.add_argument(
        "--penalty-weight-mode",
        type=str,
        default="exponential",
        choices=["linear", "exponential", "step"],
        metavar="MODE",
        help="Weight adjustment mode: linear, exponential (default), or step.",
    )

    # Machine Learning
    ml_group = parser.add_argument_group("machine learning")
    ml_group.add_argument(
        "--online-learning",
        action="store_true",
        default=False,
        help="Enable online learning from proof search feedback.",
    )
    ml_group.add_argument(
        "--ml-weight",
        type=float,
        metavar="W",
        help="ML selection weight (0.0-1.0, default: auto-determine).",
    )
    ml_group.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
        metavar="D",
        help="Embedding dimension for neural networks (default: 32).",
    )
    ml_group.add_argument(
        "--goal-directed",
        action="store_true",
        default=False,
        help="Enable goal-directed clause selection using goal proximity scoring.",
    )
    ml_group.add_argument(
        "--goal-proximity-weight",
        type=float,
        default=0.3,
        metavar="W",
        help="Goal proximity influence weight (0.0-1.0, default: 0.3). "
             "Higher values focus search more strongly on goal-relevant clauses.",
    )
    ml_group.add_argument(
        "--embedding-evolution-rate",
        type=float,
        default=0.01,
        metavar="R",
        help="Embedding evolution rate for online learning (default: 0.01).",
    )
    ml_group.add_argument(
        "--learn-from-back-subsumption",
        action="store_true",
        default=False,
        help="Enable ML learning from back-subsumption events.",
    )
    ml_group.add_argument(
        "--learn-from-forward-subsumption",
        action="store_true",
        default=False,
        help="Enable ML learning from forward-subsumption events.",
    )
    # RNN2Vec embedding options
    ml_group.add_argument(
        "--rnn2vec-embeddings",
        action="store_true",
        default=False,
        help="Enable RNN2Vec structural embeddings. "
             "Trains an RNN encoder on tree walks via contrastive learning.",
    )
    ml_group.add_argument(
        "--rnn2vec-weight",
        type=float,
        default=0.0,
        metavar="W",
        help="RNN2Vec selection ratio weight (0 = disabled, default: 0).",
    )
    ml_group.add_argument(
        "--rnn2vec-dim",
        type=int,
        default=64,
        metavar="D",
        dest="rnn2vec_dim",
        help="RNN2Vec output embedding dimension (default: 64).",
    )
    ml_group.add_argument(
        "--rnn2vec-hidden-dim",
        type=int,
        default=64,
        metavar="D",
        dest="rnn2vec_hidden_dim",
        help="RNN2Vec hidden state dimension (default: 64).",
    )
    ml_group.add_argument(
        "--rnn2vec-input-dim",
        type=int,
        default=32,
        metavar="D",
        dest="rnn2vec_input_dim",
        help="RNN2Vec token embedding dimension (default: 32).",
    )
    ml_group.add_argument(
        "--rnn2vec-max-walk-length",
        type=int,
        default=0,
        metavar="N",
        dest="rnn2vec_max_walk_length",
        help="RNN2Vec max tokens per walk (0 = unlimited, default: 0).",
    )
    ml_group.add_argument(
        "--rnn2vec-cache",
        type=int,
        default=10000,
        metavar="N",
        dest="rnn2vec_cache",
        help="RNN2Vec embedding cache size (default: 10000).",
    )
    ml_group.add_argument(
        "--rnn2vec-rnn-type",
        type=str,
        choices=["gru", "lstm", "elman"],
        default="gru",
        dest="rnn2vec_rnn_type",
        help="RNN variant for RNN2Vec encoder (default: gru).",
    )
    ml_group.add_argument(
        "--rnn2vec-composition",
        type=str,
        choices=["last_hidden", "mean_pool", "attention_pool"],
        default="mean",
        dest="rnn2vec_composition",
        help="RNN2Vec embedding composition strategy (default: mean).",
    )
    ml_group.add_argument(
        "--rnn2vec-online-learning",
        action="store_true",
        default=False,
        help="Enable online RNN2Vec updates during search.",
    )
    ml_group.add_argument(
        "--rnn2vec-online-interval",
        type=int,
        default=20,
        metavar="N",
        dest="rnn2vec_online_interval",
        help="Clauses kept between RNN2Vec online updates (default: 20).",
    )
    ml_group.add_argument(
        "--rnn2vec-online-batch-size",
        type=int,
        default=10,
        metavar="N",
        dest="rnn2vec_online_batch_size",
        help="Max clauses per RNN2Vec online update batch (default: 10).",
    )
    ml_group.add_argument(
        "--rnn2vec-online-lr",
        type=float,
        default=0.001,
        metavar="LR",
        dest="rnn2vec_online_lr",
        help="Learning rate for online RNN2Vec updates (default: 0.001).",
    )
    ml_group.add_argument(
        "--rnn2vec-online-max-updates",
        type=int,
        default=0,
        metavar="N",
        dest="rnn2vec_online_max_updates",
        help="Stop RNN2Vec online updates after N (default: 0 = unlimited).",
    )
    ml_group.add_argument(
        "--rnn2vec-training-epochs",
        type=int,
        default=5,
        metavar="N",
        dest="rnn2vec_training_epochs",
        help="RNN2Vec initial training epochs (default: 5).",
    )
    ml_group.add_argument(
        "--rnn2vec-training-lr",
        type=float,
        default=0.001,
        metavar="LR",
        dest="rnn2vec_training_lr",
        help="RNN2Vec initial training learning rate (default: 0.001).",
    )
    ml_group.add_argument(
        "--rnn2vec-load-model",
        metavar="DIR",
        default=None,
        dest="rnn2vec_load_model",
        help="Load pre-trained RNN2Vec model from DIR. Implies --rnn2vec-embeddings.",
    )
    ml_group.add_argument(
        "--rnn2vec-save-model",
        metavar="DIR",
        default="",
        dest="rnn2vec_save_model",
        help="Save the trained RNN2Vec model to DIR after training completes. "
             "The directory is created if it does not exist. "
             "Can be reloaded later with --rnn2vec-load-model.",
    )
    ml_group.add_argument(
        "--rnn2vec-dump-embeddings",
        metavar="FILE",
        default="",
        dest="rnn2vec_dump_embeddings",
        help="Write SOS clause embeddings to FILE (JSON) after each RNN2Vec training. "
             "Each entry contains the clause text, embedding vector, clause weight, "
             "and whether the clause appeared in a proof. "
             "The file is overwritten after every update.",
    )
    ml_group.add_argument(
        "--rnn2vec-var-identity",
        action="store_true",
        default=False,
        dest="rnn2vec_var_identity",
        help="Encode De Bruijn-style variable identity in RNN2Vec walk tokens. "
             "Variables are emitted as VAR_1, VAR_2, ... in order of first appearance "
             "within each walk. Repeated occurrences of the same variable receive the "
             "same index, capturing variable sharing across subtrees.",
    )
    ml_group.add_argument(
        "--rnn2vec-goal-proximity",
        action="store_true",
        default=False,
        dest="rnn2vec_goal_proximity",
        help="Enable goal-proximity scoring for RNN2Vec clause selection.",
    )
    ml_group.add_argument(
        "--rnn2vec-goal-proximity-weight",
        type=float,
        default=0.3,
        metavar="W",
        dest="rnn2vec_goal_proximity_weight",
        help="Weight of goal proximity in RNN2Vec scoring (default: 0.3).",
    )
    ml_group.add_argument(
        "--rnn2vec-random-goal-weight",
        type=float,
        default=0.0,
        metavar="W",
        dest="rnn2vec_random_goal_weight",
        help=(
            "Selection ratio weight for random-goal proximity mode. "
            "Each turn this rule fires, a goal is chosen at random and the "
            "SOS clause nearest to it (by RNN2Vec cosine similarity) is selected. "
            "0 disables (default: 0)."
        ),
    )
    ml_group.add_argument(
        "--rnn2vec-ancestor-tracking",
        action="store_true",
        default=True,
        dest="rnn2vec_ancestor_tracking",
        help="Track productive ancestors for goal-directed RNN2Vec selection (default: enabled).",
    )
    ml_group.add_argument(
        "--no-rnn2vec-ancestor-tracking",
        action="store_false",
        dest="rnn2vec_ancestor_tracking",
    )
    ml_group.add_argument(
        "--rnn2vec-ancestor-threshold",
        type=float,
        default=0.3,
        dest="rnn2vec_ancestor_proximity_threshold",
        help="Goal distance threshold below which a clause's parents are recorded as productive ancestors (default: 0.3).",
    )
    ml_group.add_argument(
        "--rnn2vec-ancestor-max",
        type=int,
        default=500,
        dest="rnn2vec_ancestor_max_count",
        help="Maximum number of productive ancestor embeddings to track (default: 500).",
    )
    ml_group.add_argument(
        "--rnn2vec-ancestor-decay",
        type=float,
        default=0.8,
        dest="rnn2vec_ancestor_decay",
        help="Weight decay per depth level for recursive ancestor scoring (default: 0.8).",
    )
    ml_group.add_argument(
        "--rnn2vec-ancestor-min-weight",
        type=float,
        default=0.1,
        dest="rnn2vec_ancestor_min_weight",
        help="Stop recursion when decay^depth falls below this value (default: 0.1).",
    )
    ml_group.add_argument(
        "--rnn2vec-ancestor-max-depth",
        type=int,
        default=5,
        dest="rnn2vec_ancestor_max_depth",
        help="Hard cap on ancestor recursion depth (default: 5).",
    )

    # Performance / backend
    perf = parser.add_argument_group("performance")
    perf.add_argument(
        "--cpp",
        action="store_true",
        default=False,
        dest="use_cpp",
        help=(
            "Enable C++ extension for unification and term operations. "
            "Requires the _pyladr_core extension to be compiled "
            "(run build_cpp.sh). Falls back to pure Python if unavailable."
        ),
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
    output.add_argument(
        "--print-given-stats",
        action="store_true",
        default=False,
        help="Print per-given-clause inference count statistics",
    )

    return parser


# ── Goal denial (negation for refutation) ─────────────────────────────────


def _collect_variables(t) -> set[int]:
    """Collect all variable numbers in a term.

    Recursively traverses the term tree and returns the set of
    variable numbers (varnum values) found.
    """
    if t.is_variable:
        return {t.varnum}
    result: set[int] = set()
    for arg in t.args:
        result |= _collect_variables(arg)
    return result


def _skolemize_term(t, var_to_skolem: dict[int, 'Term']) -> 'Term':
    """Replace variables in a term with Skolem constants.

    Args:
        t: The term to Skolemize.
        var_to_skolem: Mapping from variable number to Skolem constant Term.

    Returns:
        New term with variables replaced by Skolem constants.
    """
    from pyladr.core.term import Term

    if t.is_variable:
        return var_to_skolem.get(t.varnum, t)
    if t.arity == 0:
        return t
    new_args = tuple(_skolemize_term(arg, var_to_skolem) for arg in t.args)
    if new_args == t.args:
        return t
    return Term(private_symbol=t.private_symbol, arity=t.arity, args=new_args)


def _deny_goals(
    parsed: ParsedInput,
    symbol_table: SymbolTable,
) -> tuple[list[Clause], list[Clause], list[Clause]]:
    """Negate goals and add them to SOS for refutation-based proving.

    Returns (usable_clauses, sos_clauses, denied_clauses) where denied_clauses
    is parallel to parsed.goals (denied_clauses[i] is the negation of parsed.goals[i]).

    Matches C Prover9 semantics:
    - Goal: ∀x₁...∀xₙ φ(x₁,...,xₙ)
    - Negate: ∃x₁...∃xₙ ¬φ(x₁,...,xₙ)
    - Skolemize: ¬φ(c₁,...,cₙ) where cᵢ are fresh Skolem constants

    Each literal's sign is flipped, and all universally-quantified
    variables are replaced with fresh Skolem constants.
    """
    from pyladr.core.term import Term, get_rigid_term

    usable = list(parsed.usable)
    sos = list(parsed.sos)
    denied_clauses: list[Clause] = []

    skolem_counter = 1

    for i, goal in enumerate(parsed.goals):
        # Collect all variables across all literals of this goal
        all_vars: set[int] = set()
        for lit in goal.literals:
            all_vars |= _collect_variables(lit.atom)

        # Create Skolem constants for each variable
        var_to_skolem: dict[int, Term] = {}
        for varnum in sorted(all_vars):
            name = f"_sk{skolem_counter}"
            skolem_counter += 1
            sn = symbol_table.str_to_sn(name, 0)
            symbol_table.mark_skolem(sn)
            var_to_skolem[varnum] = get_rigid_term(sn, 0)

        # Negate literals and Skolemize
        denied_lits = tuple(
            Literal(
                sign=not lit.sign,
                atom=_skolemize_term(lit.atom, var_to_skolem),
            )
            for lit in goal.literals
        )
        denied = Clause(
            literals=denied_lits,
            justification=(Justification(just_type=JustType.DENY, clause_ids=(0,)),),
        )
        sos.append(denied)
        denied_clauses.append(denied)

    return usable, sos, denied_clauses


# ── Auto-detect inference settings ────────────────────────────────────────


def _auto_inference(
    parsed: ParsedInput,
    opts: SearchOptions,
) -> None:
    """Auto-detect appropriate inference rules based on the input.

    Matches C Prover9 auto_inference behavior:
    - If there are equality literals, enable paramodulation and demodulation
    - If set(auto) and clauses have negative literals, enable hyper-resolution
      (C: auto_inference() in auto.c enables hyper_res for Horn-like problems)
    """
    has_equality = False
    has_neg_lits = False
    for clause in parsed.sos + parsed.goals + parsed.usable:
        for lit in clause.literals:
            if lit.atom.arity == 2:
                has_equality = True
            if not lit.sign:
                has_neg_lits = True
        if has_equality and has_neg_lits:
            break

    if has_equality:
        opts.paramodulation = True
        opts.demodulation = True

    # C Prover9 auto mode
    if parsed.flags.get("auto", False):
        # Enable hyper-resolution when input has clauses with negative
        # literals (suitable nuclei for hyper-resolution).
        # C auto_inference() (auto.c) sets hyper_resolution AND clears
        # binary_resolution when the HNE pattern is detected.
        if has_neg_lits:
            opts.hyper_resolution = True
            opts.binary_resolution = False


def _auto_limits(
    parsed: ParsedInput,
    opts: SearchOptions,
) -> None:
    """Apply default auto limits (C: set(auto) -> set(auto_limits)).

    Only sets defaults when not explicitly overridden by assign() directives.
    """
    if opts.max_weight < 0:
        opts.max_weight = 100.0
    if opts.sos_limit < 0:
        opts.sos_limit = 20000


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
    unification_weight: int = 0,
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

    for clause in proof.clauses:
        clause_str = format_clause_standard(clause, symbol_table)
        extras: list[str] = []
        if clause.given_selection:
            extras.append(f"given: {clause.given_selection}")
        if unification_weight > 0:
            from pyladr.search.selection import _clause_generality_penalty
            penalty = _clause_generality_penalty(clause)
            extras.append(f"penalty: {penalty:.2f}")
        if clause.given_distance > 0.0:
            extras.append(f"gd: {clause.given_distance:.4f}")
        if extras:
            print(f"{clause_str}  [{', '.join(extras)}]", file=out)
        else:
            print(f"{clause_str}", file=out)
    print(file=out)
    print_separator("end of proof")


def _print_statistics(
    result: SearchResult,
    out: TextIO,
    print_given_stats: bool = False,
    ml_selection_stats: object | None = None,
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
    if print_given_stats and stats.given_inference_counts:
        print(file=out)
        top = stats.top_given_clauses(10)
        print("Given_inference_counts (top 10 most productive):", file=out)
        for clause_id, count in top:
            print(f"  given_clause_{clause_id}={count}.", file=out)
    # ML selection statistics (only when ML was actually used)
    if ml_selection_stats is not None:
        ml_report = ml_selection_stats.report()
        if ml_selection_stats.ml_selections > 0 or ml_selection_stats.traditional_selections > 0:
            print(file=out)
            print(f"ML_selection: {ml_report}", file=out)
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


# ── Apply parsed LADR settings to SearchOptions ───────────────────────────


# ── Declarative LADR→SearchOptions mappings ──────────────────────────────

# assign() directives: LADR name → (SearchOptions field, type converter)
_ASSIGN_MAP: dict[str, tuple[str, type]] = {
    "max_proofs": ("max_proofs", int),
    "max_given": ("max_given", int),
    "max_kept": ("max_kept", int),
    "max_seconds": ("max_seconds", float),
    "max_generated": ("max_generated", int),
    "max_weight": ("max_weight", float),
    "sos_limit": ("sos_limit", int),
    "weight_ratio": ("weight_ratio", int),
    "unification_weight": ("unification_weight", int),
    "penalty_propagation_decay": ("penalty_propagation_decay", float),
    "penalty_propagation_threshold": ("penalty_propagation_threshold", float),
    "penalty_propagation_mode": ("penalty_propagation_mode", str),
    "penalty_propagation_max_depth": ("penalty_propagation_max_depth", int),
    "penalty_propagation_max": ("penalty_propagation_max", float),
    "repetition_penalty_weight": ("repetition_penalty_weight", float),
    "repetition_penalty_min_size": ("repetition_penalty_min_size", int),
    "repetition_penalty_max": ("repetition_penalty_max", float),
    "nucleus_penalty_threshold": ("nucleus_penalty_threshold", float),
    "nucleus_penalty_weight": ("nucleus_penalty_weight", float),
    "nucleus_penalty_max": ("nucleus_penalty_max", float),
    "nucleus_penalty_cache_size": ("nucleus_penalty_cache_size", int),
    "hint_wt": ("hint_wt", float),
    "penalty_weight_threshold": ("penalty_weight_threshold", float),
    "penalty_weight_multiplier": ("penalty_weight_multiplier", float),
    "penalty_weight_max": ("penalty_weight_max", float),
    "penalty_weight_mode": ("penalty_weight_mode", str),
    "ml_weight": ("ml_weight", float),
    "goal_proximity_weight": ("goal_proximity_weight", float),
    "embedding_dim": ("embedding_dim", int),
    "embedding_evolution_rate": ("embedding_evolution_rate", float),
    "rnn2vec_weight": ("rnn2vec_weight", float),
    "rnn2vec_hidden_dim": ("rnn2vec_hidden_dim", int),
    "rnn2vec_embedding_dim": ("rnn2vec_embedding_dim", int),
    "rnn2vec_input_dim": ("rnn2vec_input_dim", int),
    "rnn2vec_max_walk_length": ("rnn2vec_max_walk_length", int),
    "rnn2vec_num_layers": ("rnn2vec_num_layers", int),
    "rnn2vec_cache_max_entries": ("rnn2vec_cache_max_entries", int),
    "rnn2vec_online_update_interval": ("rnn2vec_online_update_interval", int),
    "rnn2vec_online_batch_size": ("rnn2vec_online_batch_size", int),
    "rnn2vec_online_lr": ("rnn2vec_online_lr", float),
    "rnn2vec_online_max_updates": ("rnn2vec_online_max_updates", int),
    "rnn2vec_training_epochs": ("rnn2vec_training_epochs", int),
    "rnn2vec_training_lr": ("rnn2vec_training_lr", float),
    "rnn2vec_composition": ("rnn2vec_composition", str),
    "rnn2vec_rnn_type": ("rnn2vec_rnn_type", str),
    "rnn2vec_goal_proximity_weight": ("rnn2vec_goal_proximity_weight", float),
    "rnn2vec_random_goal_weight": ("rnn2vec_random_goal_weight", float),
    "lex_dep_demod_lim": ("lex_dep_demod_lim", int),
    "demod_step_limit": ("demod_step_limit", int),
    "backsub_check": ("backsub_check", int),
    "max_weight_tighten_after": ("max_weight_tighten_after", int),
    "max_weight_tighten_to": ("max_weight_tighten_to", float),
}

# Two-way flags: set()/clear() both honored. LADR name → SearchOptions field.
_FLAG_MAP: dict[str, str] = {
    "binary_resolution": "binary_resolution",
    "hyper_resolution": "hyper_resolution",
    "factoring": "factoring",
    "paramodulation": "paramodulation",
    "demodulation": "demodulation",
    "back_demod": "back_demod",
    "lex_order_vars": "lex_order_vars",
    "check_tautology": "check_tautology",
    "merge_lits": "merge_lits",
    "priority_sos": "priority_sos",
    "lazy_demod": "lazy_demod",
    "print_given": "print_given",
    "print_kept": "print_kept",
    "print_gen": "print_gen",
    "print_given_stats": "print_given_stats",
}

# One-way flags: only set(flag) is honored (set to True); clear() is ignored.
# LADR name → SearchOptions field.
_SET_ONLY_FLAG_MAP: dict[str, str] = {
    "penalty_propagation": "penalty_propagation",
    "repetition_penalty": "repetition_penalty",
    "repetition_penalty_normalize": "repetition_penalty_normalize",
    "nucleus_unification_penalty": "nucleus_unification_penalty",
    "penalty_weight": "penalty_weight_enabled",
    "online_learning": "online_learning",
    "goal_directed": "goal_directed",
    "rnn2vec_embeddings": "rnn2vec_embeddings",
    "rnn2vec_online_learning": "rnn2vec_online_learning",
    "rnn2vec_bidirectional": "rnn2vec_bidirectional",
    "rnn2vec_include_var_identity": "rnn2vec_include_var_identity",
    "rnn2vec_goal_proximity": "rnn2vec_goal_proximity",
}

# Extra recognized names not in the maps above (handled by auto-inference etc.)
_EXTRA_KNOWN_FLAGS = {
    "auto", "auto_inference", "auto_setup", "auto_limits",
    "auto_denials", "auto_process", "predicate_elim",
}
_EXTRA_KNOWN_ASSIGNS = {"eq_defs"}

# Derived known-name sets for unrecognized-name warnings
_KNOWN_FLAGS = set(_FLAG_MAP) | set(_SET_ONLY_FLAG_MAP) | _EXTRA_KNOWN_FLAGS
_KNOWN_ASSIGNS = set(_ASSIGN_MAP) | _EXTRA_KNOWN_ASSIGNS


def _apply_settings(parsed, opts: SearchOptions, st=None) -> None:
    """Apply parsed assign()/set()/clear() directives from LADR input to SearchOptions."""
    # assign() directives (LADR input overrides CLI defaults)
    for assign_key, (field_name, converter) in _ASSIGN_MAP.items():
        if assign_key in parsed.assigns:
            setattr(opts, field_name, converter(parsed.assigns[assign_key]))

    # Two-way flags: both set() and clear() are honored
    for flag_key, field_name in _FLAG_MAP.items():
        if flag_key in parsed.flags:
            setattr(opts, field_name, parsed.flags[flag_key])

    # One-way flags: only set() is honored (enables the feature)
    for flag_key, field_name in _SET_ONLY_FLAG_MAP.items():
        if parsed.flags.get(flag_key, False):
            setattr(opts, field_name, True)

    # Warn about unrecognized set()/clear() flags and assign() directives
    for flag_name in parsed.flags:
        if flag_name not in _KNOWN_FLAGS:
            print(f"WARNING: unrecognized flag '{flag_name}' (ignored).",
                  file=sys.stderr)
    for assign_name in parsed.assigns:
        if assign_name not in _KNOWN_ASSIGNS:
            print(f"WARNING: unrecognized parameter '{assign_name}' (ignored).",
                  file=sys.stderr)


# ── ML Selection Factory ────────────────────────────────────────────────────


def _should_use_ml_selection(opts: SearchOptions) -> bool:
    """Determine if ML selection should be used."""
    return (opts.online_learning or
            opts.ml_weight is not None or
            opts.goal_directed)


def _create_ml_selection(opts: SearchOptions, symbol_table: SymbolTable):
    """Create ML-enhanced selection with goal-directed features."""
    try:
        from pyladr.search.ml_selection import EmbeddingEnhancedSelection, MLSelectionConfig
        from pyladr.search.goal_directed import GoalDirectedEmbeddingProvider, GoalDirectedConfig

        # Create base embedding provider (with error handling for missing torch)
        try:
            from pyladr.ml.embedding_provider import create_embedding_provider
            base_provider = create_embedding_provider(symbol_table=symbol_table)
            # Check if we got a real provider or NoOp fallback
            from pyladr.ml.embedding_provider import NoOpEmbeddingProvider
            if isinstance(base_provider, NoOpEmbeddingProvider):
                print("% ML: using NoOp embedding provider (torch not available)")
                if not opts.goal_directed:
                    return None
            else:
                print("% ML: GNN embedding provider initialized")
        except ImportError:
            from pyladr.ml.embedding_provider import NoOpEmbeddingProvider
            base_provider = NoOpEmbeddingProvider()
            print("% ML: using NoOp embedding provider (ML modules not available)")
            if not opts.goal_directed:  # Only warn if ML actually requested
                return None

        provider = base_provider

        # Wrap with goal-directed features if enabled
        if opts.goal_directed:
            gd_config = GoalDirectedConfig(
                enabled=True,
                goal_proximity_weight=opts.goal_proximity_weight,
                proximity_method="max",
                online_learning=opts.online_learning,
            )
            provider = GoalDirectedEmbeddingProvider(
                base_provider=provider,
                config=gd_config,
            )

        # Create ML selection config
        ml_config = MLSelectionConfig(
            enabled=True,
            ml_weight=opts.ml_weight or 0.3,
        )

        return EmbeddingEnhancedSelection(
            embedding_provider=provider,
            ml_config=ml_config,
        )

    except Exception as e:
        import logging
        logging.warning(f"ML selection setup failed: {e}, using traditional selection")
        return None


def _extract_goals_from_sos(sos: list[Clause]) -> list[Clause]:
    """Extract goal clauses from SOS (identify DENY justifications)."""
    goals = []
    for clause in sos:
        if (clause.justification and
            len(clause.justification) > 0 and
            clause.justification[0].just_type == JustType.DENY):
            goals.append(clause)
    return goals


# ── Main entry point ──────────────────────────────────────────────────────


def run_prover(argv: list[str] | None = None) -> int:
    """Run the Prover9 theorem prover.

    Args:
        argv: Command-line arguments (defaults to sys.argv).

    Returns:
        Process exit code (0 = proof found, 2 = SOS empty, etc.)
    """
    _configure_runtime()
    if argv is None:
        argv = sys.argv

    arg_parser = _build_arg_parser()

    args = arg_parser.parse_args(argv[1:])

    _configure_cpp_backend(args.use_cpp)

    out: TextIO = sys.stdout

    # ── Configure logging for given clause output ───────────────────────

    # Create a custom handler that writes to stdout
    class StdoutHandler(logging.StreamHandler):
        def __init__(self):
            super().__init__(sys.stdout)

        def format(self, record):
            # Format given clause messages to match C Prover9 format
            if record.name == 'pyladr.search.given_clause':
                return record.getMessage()
            return super().format(record)

    # Configure logger for given clause search
    logger = logging.getLogger('pyladr.search.given_clause')
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add our custom handler
    handler = StdoutHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False

    # Configure logger for ML selection (visible when ML is active)
    ml_logger = logging.getLogger('pyladr.search.ml_selection')
    ml_logger.setLevel(logging.INFO)
    for h in ml_logger.handlers[:]:
        ml_logger.removeHandler(h)
    ml_handler = StdoutHandler()
    ml_handler.setLevel(logging.INFO)
    ml_logger.addHandler(ml_handler)
    ml_logger.propagate = False

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

    usable, sos, denied_clauses = _deny_goals(parsed, symbol_table)

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
        max_weight=args.max_weight,
        max_weight_tighten_after=args.max_weight_tighten_after,
        max_weight_tighten_to=args.max_weight_tighten_to,
        weight_ratio=args.weight_ratio,
        unification_weight=args.unification_weight,
        penalty_propagation=args.penalty_propagation,
        penalty_propagation_mode=args.penalty_propagation_mode,
        penalty_propagation_decay=args.penalty_propagation_decay,
        penalty_propagation_threshold=args.penalty_propagation_threshold,
        repetition_penalty=args.repetition_penalty,
        repetition_penalty_weight=args.repetition_penalty_weight,
        repetition_penalty_min_size=args.repetition_penalty_min_size,
        repetition_penalty_max=args.repetition_penalty_max,
        repetition_penalty_normalize=args.repetition_penalty_normalize,
        nucleus_unification_penalty=args.nucleus_unification_penalty,
        nucleus_penalty_threshold=args.nucleus_penalty_threshold,
        nucleus_penalty_weight=args.nucleus_penalty_weight,
        nucleus_penalty_max=args.nucleus_penalty_max,
        nucleus_penalty_cache_size=args.nucleus_penalty_cache_size,
        penalty_weight_enabled=args.penalty_weight,
        penalty_weight_threshold=args.penalty_weight_threshold,
        penalty_weight_multiplier=args.penalty_weight_multiplier,
        penalty_weight_max=args.penalty_weight_max,
        penalty_weight_mode=args.penalty_weight_mode,
        print_given=not args.no_print_given,
        print_kept=args.print_kept,
        print_given_stats=args.print_given_stats,
        quiet=args.quiet,
        online_learning=args.online_learning,
        ml_weight=args.ml_weight,
        embedding_dim=args.embedding_dim,
        goal_directed=args.goal_directed,
        goal_proximity_weight=args.goal_proximity_weight,
        embedding_evolution_rate=args.embedding_evolution_rate,
        learn_from_back_subsumption=args.learn_from_back_subsumption,
        learn_from_forward_subsumption=args.learn_from_forward_subsumption,
        rnn2vec_embeddings=args.rnn2vec_embeddings or (args.rnn2vec_load_model is not None),
        rnn2vec_weight=args.rnn2vec_weight if args.rnn2vec_weight > 0 else (1.0 if (args.rnn2vec_embeddings or args.rnn2vec_load_model) else 0.0),
        rnn2vec_rnn_type=args.rnn2vec_rnn_type,
        rnn2vec_hidden_dim=args.rnn2vec_hidden_dim,
        rnn2vec_embedding_dim=args.rnn2vec_dim,
        rnn2vec_input_dim=args.rnn2vec_input_dim,
        rnn2vec_max_walk_length=args.rnn2vec_max_walk_length,
        rnn2vec_cache_max_entries=args.rnn2vec_cache,
        rnn2vec_composition=args.rnn2vec_composition,
        rnn2vec_online_learning=args.rnn2vec_online_learning,
        rnn2vec_online_update_interval=args.rnn2vec_online_interval,
        rnn2vec_online_batch_size=args.rnn2vec_online_batch_size,
        rnn2vec_online_lr=args.rnn2vec_online_lr,
        rnn2vec_online_max_updates=args.rnn2vec_online_max_updates,
        rnn2vec_training_epochs=args.rnn2vec_training_epochs,
        rnn2vec_training_lr=args.rnn2vec_training_lr,
        rnn2vec_model_path=args.rnn2vec_load_model or "",
        rnn2vec_save_model=args.rnn2vec_save_model,
        rnn2vec_include_var_identity=args.rnn2vec_var_identity,
        rnn2vec_goal_proximity=args.rnn2vec_goal_proximity,
        rnn2vec_goal_proximity_weight=args.rnn2vec_goal_proximity_weight,
        rnn2vec_ancestor_tracking=args.rnn2vec_ancestor_tracking,
        rnn2vec_ancestor_proximity_threshold=args.rnn2vec_ancestor_proximity_threshold,
        rnn2vec_ancestor_max_count=args.rnn2vec_ancestor_max_count,
        rnn2vec_ancestor_decay=args.rnn2vec_ancestor_decay,
        rnn2vec_ancestor_min_weight=args.rnn2vec_ancestor_min_weight,
        rnn2vec_ancestor_max_depth=args.rnn2vec_ancestor_max_depth,
        rnn2vec_random_goal_weight=args.rnn2vec_random_goal_weight,
        rnn2vec_dump_embeddings=args.rnn2vec_dump_embeddings,
    )

    _apply_settings(parsed, opts)

    # Validate penalty weight parameters
    if opts.penalty_weight_enabled:
        if opts.penalty_weight_multiplier < 1.0:
            print(
                "Fatal error: penalty_weight_multiplier must be >= 1.0"
                f" (got {opts.penalty_weight_multiplier})",
                file=sys.stderr,
            )
            return 1
        if opts.penalty_weight_threshold <= 0.0:
            print(
                "Fatal error: penalty_weight_threshold must be > 0.0"
                f" (got {opts.penalty_weight_threshold})",
                file=sys.stderr,
            )
            return 1
        if opts.penalty_weight_max <= 0.0:
            print(
                "Fatal error: penalty_weight_max must be > 0.0"
                f" (got {opts.penalty_weight_max})",
                file=sys.stderr,
            )
            return 1
        if opts.penalty_weight_mode not in ("linear", "exponential", "step"):
            print(
                "Fatal error: penalty_weight_mode must be linear, exponential, or step"
                f" (got '{opts.penalty_weight_mode}')",
                file=sys.stderr,
            )
            return 1

    # Auto-detect inference settings based on input.
    # C Prover9: set(auto) implies set(auto_inference), set(auto_setup),
    # set(auto_limits), set(auto_denials), set(auto_process).
    # Individual sub-flags can be explicitly cleared to disable them,
    # e.g. set(auto). clear(auto_inference). keeps limits but skips inference.
    auto_on = parsed.flags.get("auto", False)
    auto_inference_on = parsed.flags.get("auto_inference", auto_on)
    auto_limits_on = parsed.flags.get("auto_limits", auto_on)
    if auto_inference_on:
        _auto_inference(parsed, opts)
    if auto_limits_on:
        _auto_limits(parsed, opts)

    # ── Print initial clauses ───────────────────────────────────────────

    _print_initial_clauses(usable, sos, symbol_table, out)

    # ── ML Selection Setup ──────────────────────────────────────────────

    selection = None
    if _should_use_ml_selection(opts):
        selection = _create_ml_selection(opts, symbol_table)
        if selection:
            print("% ML-enhanced clause selection enabled"
                  f" (ml_weight={selection.ml_config.ml_weight:.2f})")
        else:
            print("% ML selection requested but could not be initialized;"
                  " using traditional selection")

    # ── Run search ──────────────────────────────────────────────────────

    engine = GivenClauseSearch(
        options=opts,
        selection=selection,  # Pass custom selection (None = default)
        symbol_table=symbol_table,
        hints=parsed.hints if parsed.hints else None,
    )

    # ── Reserve clause IDs for goal formulas (C Prover9 compatibility) ──
    # C Prover9 assigns IDs 1..N to goal formulas before clausification,
    # so the first kept clause gets ID (N+1). Match this behavior.
    num_goals = len(parsed.goals)
    if num_goals > 0:
        engine._state.reserve_clause_ids(num_goals)

    # ── Register Goals for Goal-Directed Selection ──────────────────────

    if opts.goal_directed and selection:
        goals = _extract_goals_from_sos(sos)  # Extract DENY justifications
        if goals and hasattr(selection, 'embedding_provider'):
            provider = selection.embedding_provider
            if hasattr(provider, 'register_goals'):
                provider.register_goals(goals)
                print(f"% ML: registered {len(goals)} goal(s) for goal-directed selection")

    proven_goal_indices: set[int] = set()

    def _on_proof(proof: Proof, proof_num: int) -> None:
        _print_proof(
            proof,
            proof_num=proof_num,
            search_seconds=engine.stats.search_seconds(),
            symbol_table=symbol_table,
            out=out,
            unification_weight=opts.unification_weight,
        )
        if parsed.goals:
            proof_clause_ids = {c.id for c in proof.clauses}
            for i, denied in enumerate(denied_clauses):
                if denied.id in proof_clause_ids:
                    proven_goal_indices.add(i)
            unproven = [
                parsed.goals[i]
                for i in range(len(parsed.goals))
                if i not in proven_goal_indices
            ]
            if unproven:
                print(f"% Unproven goal formulas ({len(unproven)} of {len(parsed.goals)}):", file=out)
                for g in unproven:
                    print(f"%   {format_clause_standard(g, symbol_table)}", file=out)

    engine.set_proof_callback(_on_proof)

    try:
        result = engine.run(usable=usable, sos=sos)
    except Exception as e:
        print(f"\nFatal error during search: {e}", file=sys.stderr)
        return 1

    # ── Print statistics and conclusion ─────────────────────────────────

    ml_stats = selection.ml_stats if selection and hasattr(selection, 'ml_stats') else None
    _print_statistics(result, out, print_given_stats=opts.print_given_stats,
                      ml_selection_stats=ml_stats)
    _print_conclusion(result, out)

    return _PROCESS_EXIT_CODES.get(result.exit_code, 1)


def main() -> int:
    """CLI entry point."""
    return run_prover()


if __name__ == "__main__":
    sys.exit(main())
