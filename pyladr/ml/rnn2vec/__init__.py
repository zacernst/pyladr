"""RNN2Vec: RNN-based structural embeddings for logical formulas.

Opt-in embedding system: enable with ``--rnn2vec-embeddings`` CLI flag or
``set(rnn2vec_embeddings).`` in LADR input. Requires torch.

Implements RNN encoding over tree walks to learn structural embeddings
for terms and clauses in first-order logic. Uses contrastive training
instead of skip-gram.
"""

from pyladr.ml.rnn2vec.algorithm import RNN2Vec, RNN2VecConfig
from pyladr.ml.rnn2vec.background_updater import BackgroundRNN2VecUpdater
from pyladr.ml.rnn2vec.formula_processor import (
    RNN2VecProcessingResult,
    process_vampire_corpus,
    process_vampire_file,
)
from pyladr.ml.rnn2vec.provider import (
    RNN2VecCacheStats,
    RNN2VecEmbeddingProvider,
    RNN2VecProviderConfig,
)
from pyladr.ml.rnn2vec.tokenizer import TokenVocab

__all__ = [
    "BackgroundRNN2VecUpdater",
    "RNN2Vec",
    "RNN2VecCacheStats",
    "RNN2VecConfig",
    "RNN2VecEmbeddingProvider",
    "RNN2VecProcessingResult",
    "RNN2VecProviderConfig",
    "TokenVocab",
    "process_vampire_corpus",
    "process_vampire_file",
]
