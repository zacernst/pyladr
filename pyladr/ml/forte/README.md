# FORTE: Feature-Oriented Representation for Theorem-proving Embeddings

High-performance deterministic clause embedding algorithm for PyLADR.

## Overview

FORTE is an opt-in embedding system that produces 64-dimensional clause
embeddings via feature hashing. It is **pure Python** with no torch or
ML framework dependency.

### Enabling

- **CLI flag**: `--forte-embeddings`
- **LADR input**: `set(forte_embeddings).`
- **SearchOptions**: `forte_embeddings=True`

Additional options:
- `--forte-weight N` (selection ratio weight, default 1.0 when enabled)
- `--forte-dim N` (embedding dimension, default 64)
- `--forte-cache N` (cache size, default 100000)
- `--proof-guided` (enable proof-guided selection on top of FORTE)

## Quick Start

```python
from pyladr.ml.forte import ForteAlgorithm, ForteEmbeddingProvider

# Basic usage: algorithm only
algorithm = ForteAlgorithm()
embedding = algorithm.embed_clause(clause)  # list[float], 64-dimensional

# Full PyLADR integration: provider with caching
provider = ForteEmbeddingProvider()
embedding = provider.get_embedding(clause)  # list[float] | None
embeddings = provider.get_embeddings_batch(clauses)  # batch processing
```

## Files

- `algorithm.py`: Core FORTE algorithm implementation
- `provider.py`: EmbeddingProvider protocol integration + caching
- `proof_patterns.py`: Proof-guided selection using FORTE embeddings
- `__init__.py`: Public API exports

## Performance

- **Speed**: ~25 us per clause (pure Python)
- **Memory**: ~4KB per cached embedding
- **Dependencies**: None (pure Python + PyLADR core)

## Known Limitations

- Does not integrate with ProofGuidedSelection's goal proximity features
- Proof-guided selection (`--proof-guided`) requires FORTE to be enabled

## Testing

```bash
# Run FORTE-specific tests
python3 -m pytest tests/unit/test_forte_*.py -v
```
