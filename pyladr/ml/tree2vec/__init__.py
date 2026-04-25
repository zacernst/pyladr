"""Tree2Vec: Unsupervised embedding generation from logical formula trees.

Opt-in embedding system: enable with ``--tree2vec-embeddings`` CLI flag or
``set(tree2vec_embeddings).`` in LADR input. Pure Python, no torch required.

Implements skip-gram training over tree walks to learn structural embeddings
for terms and clauses in first-order logic. Optimized for constrained
vocabularies like the vampire.in domain (P, i, n, variables).
"""

from pyladr.ml.tree2vec.algorithm import Tree2Vec, Tree2VecConfig
from pyladr.ml.tree2vec.formula_processor import (
    AugmentationConfig,
    ProcessingResult,
    process_vampire_corpus,
    process_vampire_file,
)
from pyladr.ml.tree2vec.provider import (
    Tree2VecCacheStats,
    Tree2VecEmbeddingProvider,
    Tree2VecProviderConfig,
)
from pyladr.ml.tree2vec.skipgram import SkipGramConfig, SkipGramTrainer
from pyladr.ml.tree2vec.vampire_parser import VampireCorpus, parse_vampire_file as parse_vampire
from pyladr.ml.tree2vec.walks import TreeWalker, WalkConfig, WalkType

__all__ = [
    "AugmentationConfig",
    "ProcessingResult",
    "SkipGramConfig",
    "SkipGramTrainer",
    "Tree2Vec",
    "Tree2VecCacheStats",
    "Tree2VecConfig",
    "Tree2VecEmbeddingProvider",
    "Tree2VecProviderConfig",
    "TreeWalker",
    "VampireCorpus",
    "WalkConfig",
    "WalkType",
    "parse_vampire",
    "process_vampire_corpus",
    "process_vampire_file",
]
