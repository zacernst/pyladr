"""FORTE: Feature-Oriented Representation for Theorem-proving Embeddings.

Opt-in embedding system: enable with ``--forte-embeddings`` CLI flag or
``set(forte_embeddings).`` in LADR input. Pure Python, no torch required.

Ultra-fast deterministic clause embedding via feature hashing.
Target: 15-25 us per clause, 64-dimensional output vectors.
"""

from pyladr.ml.forte.algorithm import ForteAlgorithm, ForteConfig
from pyladr.ml.forte.provider import ForteEmbeddingProvider, ForteProviderConfig
from pyladr.search.proof_pattern_memory import (
    ProofGuidedConfig,
    ProofPatternMemory,
    proof_guided_score,
)

__all__ = [
    "ForteAlgorithm",
    "ForteConfig",
    "ForteEmbeddingProvider",
    "ForteProviderConfig",
    "ProofGuidedConfig",
    "ProofPatternMemory",
    "proof_guided_score",
]
