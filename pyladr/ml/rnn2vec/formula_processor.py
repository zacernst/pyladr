"""Formula processing and data augmentation for RNN2Vec training.

Reuses the augmentation pipeline from Tree2Vec (variable renamings,
reversed literals, subterm extraction) but trains an RNN2Vec model
instead of a skip-gram.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term
from pyladr.ml.rnn2vec.algorithm import RNN2Vec, RNN2VecConfig
from pyladr.ml.tree2vec.formula_processor import (
    AugmentationConfig,
    _rename_variables,
)
from pyladr.ml.tree2vec.vampire_parser import VampireCorpus, parse_vampire_file

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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
