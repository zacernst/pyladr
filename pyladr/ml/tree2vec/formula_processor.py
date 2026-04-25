"""Formula processing and data augmentation for Tree2Vec training.

Generates diverse training data from a vampire.in corpus by:
1. Extracting tree walks from all formulas (clauses, literals, terms)
2. Augmenting data via variable renaming (α-equivalence variations)
3. Extracting structural subterm patterns for richer training signal
4. Training Tree2Vec embeddings on the combined walk corpus

The augmentation strategy exploits the fact that α-equivalent formulas
(differing only in variable names) should produce identical embeddings,
while providing diverse training contexts for the skip-gram learner.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term
from pyladr.ml.tree2vec.algorithm import Tree2Vec, Tree2VecConfig
from pyladr.ml.tree2vec.vampire_parser import VampireCorpus, parse_vampire_file

if TYPE_CHECKING:
    from collections.abc import Sequence

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
class ProcessingResult:
    """Result of formula processing and Tree2Vec training.

    Attributes:
        tree2vec: Trained Tree2Vec model.
        training_stats: Statistics from skip-gram training.
        corpus_stats: Statistics about the processed corpus.
    """

    tree2vec: Tree2Vec
    training_stats: dict[str, float]
    corpus_stats: dict[str, int]


def process_vampire_corpus(
    corpus: VampireCorpus,
    tree2vec_config: Tree2VecConfig | None = None,
    augmentation_config: AugmentationConfig | None = None,
    progress_fn=None,
) -> ProcessingResult:
    """Process a vampire.in corpus and train Tree2Vec embeddings.

    Pipeline:
    1. Collect all original clauses
    2. Generate augmented variants (variable renamings, reversed literals)
    3. Optionally extract subterm trees as additional training terms
    4. Train Tree2Vec on the full augmented corpus

    Args:
        corpus: Parsed VampireCorpus from vampire_parser.
        tree2vec_config: Tree2Vec configuration (uses defaults if None).
        augmentation_config: Augmentation settings (uses defaults if None).

    Returns:
        ProcessingResult with trained model and statistics.
    """
    aug_config = augmentation_config or AugmentationConfig()
    tree2vec = Tree2Vec(tree2vec_config)

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
        "Training corpus: %d original + %d renamed + %d reversed = %d total clauses",
        original_count,
        renamed_count,
        reversed_count,
        len(training_clauses),
    )

    # Train on clauses
    training_stats = tree2vec.train(training_clauses, progress_fn=progress_fn)

    # Optionally also train on individual subterms for richer signal
    subterm_count = 0
    if aug_config.include_subterm_trees and corpus.all_subterms:
        # Use train_from_terms to add subterm walk data
        # Note: this extends the existing training, it doesn't retrain
        subterm_terms = [t for t in corpus.all_subterms if t.arity > 0]
        subterm_count = len(subterm_terms)
        if subterm_terms:
            sub_stats = tree2vec.train_from_terms(subterm_terms)
            # Merge stats
            training_stats["subterm_loss"] = sub_stats.get("loss", 0.0)
            training_stats["subterm_pairs"] = sub_stats.get("total_pairs", 0)

    corpus_stats = {
        "original_clauses": original_count,
        "renamed_variants": renamed_count,
        "reversed_variants": reversed_count,
        "total_training_clauses": len(training_clauses),
        "subterm_trees": subterm_count,
        "sos_clauses": len(corpus.sos_clauses),
        "goal_clauses": len(corpus.goal_clauses),
    }

    return ProcessingResult(
        tree2vec=tree2vec,
        training_stats=training_stats,
        corpus_stats=corpus_stats,
    )


def process_vampire_file(
    filepath: str,
    tree2vec_config: Tree2VecConfig | None = None,
    augmentation_config: AugmentationConfig | None = None,
    progress_fn=None,
) -> ProcessingResult:
    """Convenience: parse a file and process in one call.

    Args:
        filepath: Path to vampire.in file.
        tree2vec_config: Tree2Vec configuration.
        augmentation_config: Augmentation settings.
        progress_fn: Optional per-epoch progress callback passed to the trainer.

    Returns:
        ProcessingResult with trained model and statistics.
    """
    corpus = parse_vampire_file(filepath)
    return process_vampire_corpus(corpus, tree2vec_config, augmentation_config,
                                  progress_fn=progress_fn)


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
