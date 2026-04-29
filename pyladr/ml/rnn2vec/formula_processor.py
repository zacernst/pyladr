"""Formula processing and data augmentation for RNN2Vec training.

Applies data augmentation (variable renamings, reversed literals, subterm
extraction) before training an RNN2Vec model.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term
from pyladr.ml.rnn2vec.algorithm import RNN2Vec, RNN2VecConfig
from pyladr.ml.rnn2vec.vampire_parser import VampireCorpus, parse_vampire_file

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Configuration for training data augmentation.

    Attributes:
        num_variable_renamings: Number of α-equivalent variants per clause.
        include_subterm_trees: Whether to add individual subterm trees.
        include_reversed_literals: Whether to include clauses with flipped literal order.
        seed: Random seed for reproducibility.
    """

    num_variable_renamings: int = 5
    include_subterm_trees: bool = True
    include_reversed_literals: bool = True
    seed: int = 42


@dataclass(frozen=True, slots=True)
class RNN2VecProcessingResult:
    """Result of formula processing and RNN2Vec training.

    Attributes:
        rnn2vec: Trained RNN2Vec model.
        training_stats: Statistics from RNN training.
        corpus_stats: Statistics about the processed corpus.
    """

    rnn2vec: RNN2Vec
    training_stats: dict[str, float]
    corpus_stats: dict[str, int]


def process_vampire_corpus(
    corpus: VampireCorpus,
    rnn2vec_config: RNN2VecConfig | None = None,
    augmentation_config: AugmentationConfig | None = None,
    progress_fn=None,
) -> RNN2VecProcessingResult:
    """Process a vampire.in corpus and train RNN2Vec embeddings.

    Pipeline:
    1. Collect all original clauses
    2. Generate augmented variants (variable renamings, reversed literals)
    3. Train RNN2Vec on the full augmented corpus

    Args:
        corpus: Parsed VampireCorpus from vampire_parser.
        rnn2vec_config: RNN2Vec configuration (uses defaults if None).
        augmentation_config: Augmentation settings (uses defaults if None).
        progress_fn: Optional per-epoch progress callback.

    Returns:
        RNN2VecProcessingResult with trained model and statistics.
    """
    aug_config = augmentation_config or AugmentationConfig()
    rnn2vec = RNN2Vec(rnn2vec_config)

    # Collect training clauses
    training_clauses: list[Clause] = list(corpus.all_clauses)
    original_count = len(training_clauses)

    # Augmentation: variable renamings
    rng = random.Random(aug_config.seed)
    renamed_count = 0
    for clause in list(corpus.all_clauses):
        for _ in range(aug_config.num_variable_renamings):
            renamed = _rename_variables(clause, rng)
            if renamed is not None:
                training_clauses.append(renamed)
                renamed_count += 1

    # Augmentation: reversed literal order
    reversed_count = 0
    if aug_config.include_reversed_literals:
        for clause in list(corpus.all_clauses):
            if clause.num_literals > 1:
                reversed_clause = Clause(
                    literals=tuple(reversed(clause.literals)),
                    weight=clause.weight,
                )
                training_clauses.append(reversed_clause)
                reversed_count += 1

    logger.info(
        "RNN2Vec training corpus: %d original + %d renamed + %d reversed = %d total clauses",
        original_count,
        renamed_count,
        reversed_count,
        len(training_clauses),
    )

    # Train on clauses
    training_stats = rnn2vec.train(training_clauses, progress_fn=progress_fn)

    # Optionally also train on individual subterms for richer signal
    subterm_count = 0
    if aug_config.include_subterm_trees and corpus.all_subterms:
        subterm_terms = [t for t in corpus.all_subterms if t.arity > 0]
        subterm_count = len(subterm_terms)
        if subterm_terms:
            sub_stats = rnn2vec.train_from_terms(subterm_terms)
            training_stats["subterm_loss"] = sub_stats.get("loss", 0.0)
            training_stats["subterm_pairs"] = sub_stats.get("training_pairs", 0)

    corpus_stats = {
        "original_clauses": original_count,
        "renamed_variants": renamed_count,
        "reversed_variants": reversed_count,
        "total_training_clauses": len(training_clauses),
        "subterm_trees": subterm_count,
        "sos_clauses": len(corpus.sos_clauses),
        "goal_clauses": len(corpus.goal_clauses),
    }

    return RNN2VecProcessingResult(
        rnn2vec=rnn2vec,
        training_stats=training_stats,
        corpus_stats=corpus_stats,
    )


def process_vampire_file(
    filepath: str,
    rnn2vec_config: RNN2VecConfig | None = None,
    augmentation_config: AugmentationConfig | None = None,
    progress_fn=None,
) -> RNN2VecProcessingResult:
    """Convenience: parse a file and process in one call.

    Args:
        filepath: Path to vampire.in file.
        rnn2vec_config: RNN2Vec configuration.
        augmentation_config: Augmentation settings.
        progress_fn: Optional per-epoch progress callback.

    Returns:
        RNN2VecProcessingResult with trained model and statistics.
    """
    corpus = parse_vampire_file(filepath)
    return process_vampire_corpus(
        corpus, rnn2vec_config, augmentation_config, progress_fn=progress_fn
    )


# ── Variable renaming (α-equivalence augmentation) ───────────────────────────


def _rename_variables(clause: Clause, rng: random.Random) -> Clause | None:
    """Create an α-equivalent clause with permuted variable numbers.

    Collects all variable numbers in the clause, generates a random
    permutation, and applies it consistently across all literals.

    Returns None if the clause has no variables (nothing to rename).
    """
    # Collect all variable numbers
    var_nums: set[int] = set()
    for lit in clause.literals:
        var_nums.update(lit.atom.variables())

    if not var_nums:
        return None

    # Create a random permutation mapping
    var_list = sorted(var_nums)
    shuffled = list(var_list)
    rng.shuffle(shuffled)
    var_map = dict(zip(var_list, shuffled))

    # Apply renaming to all literals
    new_literals = tuple(
        Literal(sign=lit.sign, atom=_rename_term_vars(lit.atom, var_map))
        for lit in clause.literals
    )

    return Clause(literals=new_literals, weight=clause.weight)


def _rename_term_vars(term: Term, var_map: dict[int, int]) -> Term:
    """Recursively rename variables in a term according to var_map."""
    if term.is_variable:
        new_varnum = var_map.get(term.varnum, term.varnum)
        return Term(private_symbol=new_varnum, arity=0, args=())

    if term.is_constant:
        return term

    new_args = tuple(_rename_term_vars(arg, var_map) for arg in term.args)
    return Term(
        private_symbol=term.private_symbol,
        arity=term.arity,
        args=new_args,
    )
