# Tree2Vec: Unsupervised Formula Embeddings via Tree Walks

Unsupervised embedding generation from logical formula trees for PyLADR.

## Overview

Tree2Vec learns structural embeddings for terms and clauses by running
skip-gram training over tree-structured random walks. It is **pure Python**
with no torch or ML framework dependency.

### Enabling

- **CLI flag**: `--tree2vec-embeddings`
- **LADR input**: `set(tree2vec_embeddings).`
- **SearchOptions**: `tree2vec_embeddings=True`

Additional options:
- `--tree2vec-weight N` (selection ratio weight, default 1.0 when enabled)
- `--tree2vec-dim N` (embedding dimension, default 64)
- `--tree2vec-cache N` (cache size, default 100000)
- `--tree2vec-online-learning` (enable online re-training during search)
- `--tree2vec-goal-proximity` (enable goal proximity scoring)
- Various walk configuration flags: `--tree2vec-position`, `--tree2vec-depth`,
  `--tree2vec-var-identity`, `--tree2vec-skip-predicate`, `--tree2vec-path-length`

## How It Works

1. Parses the input problem's formula trees
2. Generates random walks over the tree structure (depth-first, breadth-first,
   or random walks)
3. Trains a skip-gram model on walk sequences to learn node embeddings
4. Composes node embeddings into clause-level embeddings
5. Uses embeddings for diversity-based clause selection during search

## Files

- `algorithm.py`: Core Tree2Vec skip-gram training and embedding lookup
- `walks.py`: Tree walk generation (multiple walk types)
- `formula_processor.py`: Corpus processing and data augmentation
- `vampire_parser.py`: Parser for vampire.in format input files
- `skipgram.py`: Skip-gram training implementation
- `provider.py`: EmbeddingProvider protocol integration + caching
- `__init__.py`: Public API exports

## Performance

- **Dependencies**: None (pure Python + PyLADR core)
- Training happens once at search start; embeddings are cached thereafter

## Known Limitations

- Does not integrate with ProofGuidedSelection or goal proximity features
  from the FORTE subsystem
- Tree2Vec selection reuses the FORTE diversity heap in PrioritySOS;
  falls back to age-based selection if no heap entries are available

## Testing

```bash
# Run Tree2Vec-specific tests
python3 -m pytest tests/unit/test_tree2vec*.py tests/unit/test_t2v_*.py -v
```
