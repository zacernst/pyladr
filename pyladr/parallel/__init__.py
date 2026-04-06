"""Free-threading parallelization for pyladr.

Provides parallel execution strategies for the given-clause search loop,
optimized for Python 3.14 free-threading (PEP 703).

Key components:
- ParallelInferenceEngine: Parallel inference generation
- ParallelSearchConfig: Configuration for parallel execution
"""

from pyladr.parallel.inference_engine import (
    ParallelInferenceEngine,
    ParallelSearchConfig,
)

__all__ = [
    "ParallelInferenceEngine",
    "ParallelSearchConfig",
]
