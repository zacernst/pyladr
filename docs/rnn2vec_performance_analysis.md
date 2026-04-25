# RNN2Vec Performance Analysis & Integration Plan

## 1. Computational Complexity Analysis

### Skip-gram (Tree2Vec current)

Per training pair: O(K × D) where K = negative samples (5), D = embedding_dim (64).
- 1 positive + K negative samples × D-dimensional dot product + gradient update
- Per pair: ~6 × 64 = 384 multiply-adds
- Per token (window=3): ~6 context positions × 384 = 2,304 multiply-adds
- Per walk (L=20 tokens): ~20 × 2,304 ≈ 46,080 multiply-adds

### RNN forward pass (proposed)

Per sequence timestep: O(H × (I + H)) for GRU (3 gates), O(H × (I + H)) × 4/3 for LSTM (4 gates).
- GRU with H=128, I=32: 3 × 128 × (32 + 128) = 61,440 multiply-adds per timestep
- Per walk (L=20): 20 × 61,440 = 1,228,800 multiply-adds
- Plus token embedding lookup: negligible (table lookup)
- Plus output projection (H→D): 128 × 64 = 8,192 multiply-adds

### RNN backward pass (BPTT)

- ~2-3× forward pass cost for backpropagation through time
- Per walk: ~2.5 × 1,228,800 ≈ 3,072,000 multiply-adds

### Comparison

| Metric | Skip-gram | RNN (GRU, H=128) | Ratio |
|--------|-----------|-------------------|-------|
| Multiply-adds per walk (forward) | 46,080 | 1,237,000 | **27×** |
| Multiply-adds per walk (train) | 46,080 | 3,072,000 | **67×** |
| Multiply-adds per walk (embed only) | ~1,000 (lookup + avg) | 1,237,000 | **~1200×** |

**Key insight:** Embedding generation (inference) shows the largest gap. Skip-gram just looks up and averages pre-computed vectors; RNN must run a full forward pass per walk.

### Smaller config (H=64, I=16)

| Metric | Skip-gram | RNN (GRU, H=64) | Ratio |
|--------|-----------|------------------|-------|
| Multiply-adds per walk (forward) | 46,080 | 245,760 | **5.3×** |
| Multiply-adds per walk (train) | 46,080 | 614,400 | **13×** |

With H=64, the gap narrows substantially to ~5-13× — manageable.

---

## 2. Pure Python Feasibility

### The problem

Tree2Vec skip-gram is pure Python (no numpy/torch) and trains in <1 second on vampire.in (~50 tokens, ~200 clauses). The RNN requires matrix multiplications that are dramatically more expensive in pure Python.

### Empirical estimate

A GRU timestep with H=64 requires:
- 3 matrix-vector products of size (64, 80) × (80,) = 15,360 multiply-adds each
- 3 element-wise operations of size 64
- Total: ~46,080 multiply-adds per timestep

In CPython, each float multiply-add costs ~50-100ns (list access + arithmetic overhead). Per timestep: ~2.3-4.6ms. Per walk (L=20): ~46-92ms. Per epoch over 200 clauses × 13 walks each = 2,600 walks: **120-240 seconds**.

Compare: skip-gram trains in ~0.5-1.0 seconds on the same corpus.

**Pure Python RNN training is 100-500× slower than skip-gram. This is not viable for interactive use.**

### Recommendation: Require torch

Unlike Tree2Vec which deliberately avoids torch, RNN2Vec should **require torch** as a dependency:
- `nn.GRU` / `nn.LSTM` use optimized C++/CUDA kernels
- With torch on CPU: GRU forward pass for 2,600 walks batched = ~5-50ms (cuDNN/MKL)
- With torch on GPU: even faster via parallelism
- Total training time estimate: **0.5-5 seconds** — comparable to skip-gram

**No pure Python fallback.** The architecture plan should state that `pyladr.ml.rnn2vec` requires `torch` (import-guarded as in `embedding_provider.py`). When torch is not installed, the module gracefully fails with a clear error.

### Alternative: numpy fallback

A numpy-only path is theoretically possible but:
- Implementing GRU/LSTM backward pass in numpy is error-prone (~200 lines)
- No autograd — manual gradient computation
- Performance: ~10-50× faster than pure Python, still ~5-10× slower than torch
- **Not worth the complexity.** Torch is already an optional dependency in the project.

---

## 3. Memory Footprint

### Skip-gram (Tree2Vec)

```
2 matrices: input_embeddings + output_embeddings
= 2 × vocab_size × embedding_dim × 8 bytes (float64 in Python lists)

Typical: vocab=50, dim=64
= 2 × 50 × 64 × 8 = 51,200 bytes ≈ 50 KB
```

Plus vocabulary structures: ~5 KB. **Total: ~55 KB.**

### RNN (GRU, H=128, I=32, D=64)

```
Token embeddings: vocab_size × input_dim × 4 bytes (float32)
= 50 × 32 × 4 = 6,400 bytes

GRU weights (1 layer):
  weight_ih: 3 × H × I = 3 × 128 × 32 = 12,288 params
  weight_hh: 3 × H × H = 3 × 128 × 128 = 49,152 params
  bias_ih + bias_hh: 2 × 3 × H = 768 params
  Total: 62,208 params × 4 bytes = 248,832 bytes

Output projection: H × D = 128 × 64 = 8,192 params × 4 = 32,768 bytes

Total: 6,400 + 248,832 + 32,768 ≈ 288 KB
```

### Comparison

| Component | Skip-gram | RNN (GRU H=128) | RNN (GRU H=64) |
|-----------|-----------|------------------|-----------------|
| Embeddings/weights | 50 KB | 288 KB | 80 KB |
| Cache (100K entries, D=64) | 51.2 MB | 51.2 MB | 51.2 MB |
| **Total (with cache)** | **51.3 MB** | **51.5 MB** | **51.3 MB** |

**Memory is not a concern.** The LRU cache dominates; the model itself is negligible. Even LSTM (4/3× GRU weights) would add only ~100 KB.

---

## 4. Embedding Quality Comparison

### Tree2Vec (skip-gram) strengths
- Captures **co-occurrence patterns**: tokens that appear in similar walk contexts get similar embeddings
- Excellent for **symbol similarity**: learns that FUNC:3/2 and FUNC:4/2 share contexts
- Works well with **sparse vocabularies**: negative sampling handles rare tokens well
- **No sequential information**: treats walks as bags of (center, context) pairs

### RNN2Vec strengths
- Captures **sequential structure**: walk order matters (parent→child→grandchild)
- Better for **argument ordering**: P(a,b) vs P(b,a) produce different hidden states
- **Compositional**: the hidden state accumulates structural information along the walk
- **Variable-length generalization**: trained on short walks, can embed longer sequences

### Expected quality trade-offs

| Scenario | Skip-gram | RNN | Why |
|----------|-----------|-----|-----|
| Symbol similarity (P ≈ Q) | Better | Worse | Co-occurrence captures functional similarity |
| Argument order sensitivity | Poor | Better | RNN hidden state is order-sensitive |
| Deep term structure | Poor | Better | RNN accumulates depth information sequentially |
| Small vocabulary (6-8 tokens) | Better | Comparable | Skip-gram excels with constrained vocabularies |
| Large vocabulary (50+ tokens) | Comparable | Better | RNN generalizes from sequential patterns |
| Online learning stability | Better | Worse | Skip-gram updates are local; RNN gradients affect all weights |

### Recommendation
RNN2Vec is most valuable as a **complement to Tree2Vec**, not a replacement. The two capture different structural aspects. A future `ensemble` mode could blend both.

---

## 5. Integration Plan: `search/options.py`

### New bounds in `_NUMERIC_BOUNDS`:

```python
# RNN2Vec
("rnn2vec_weight", 0.0, None, "RNN2Vec embedding selection ratio weight"),
("rnn2vec_embedding_dim", 1, 4096, "RNN2Vec embedding dimension"),
("rnn2vec_hidden_dim", 1, 4096, "RNN2Vec hidden dimension"),
("rnn2vec_input_dim", 1, 4096, "RNN2Vec input embedding dimension"),
("rnn2vec_num_layers", 1, 10, "RNN2Vec number of RNN layers"),
("rnn2vec_cache_max_entries", 1, 10_000_000, "RNN2Vec cache size"),
("rnn2vec_online_lr", 0.0001, 1.0, "RNN2Vec online learning rate"),
("rnn2vec_online_update_interval", 1, 10_000, "RNN2Vec online update interval"),
("rnn2vec_online_batch_size", 1, 10_000, "RNN2Vec online batch size"),
("rnn2vec_online_max_updates", 0, None, "RNN2Vec max online updates (0=unlimited)"),
("rnn2vec_goal_proximity_weight", 0.0, 1.0, "RNN2Vec goal proximity weight"),
("rnn2vec_training_epochs", 1, 1000, "RNN2Vec training epochs"),
("rnn2vec_training_lr", 0.00001, 1.0, "RNN2Vec training learning rate"),
("rnn2vec_training_batch_size", 1, 10_000, "RNN2Vec training batch size"),
```

### New semantic validations in `validate_search_options_semantic`:

```python
# RNN2Vec requires torch
if getattr(opts, "rnn2vec_embeddings", False):
    try:
        import torch
    except ImportError:
        warnings.append("rnn2vec_embeddings=True requires torch to be installed")

# RNN2Vec dropout only effective with >1 layer
if getattr(opts, "rnn2vec_dropout", 0.0) > 0 and getattr(opts, "rnn2vec_num_layers", 1) == 1:
    warnings.append("rnn2vec_dropout has no effect with rnn2vec_num_layers=1")
```

### No `embedding_provider_type` enum field

Considered adding `embedding_provider_type: str = "none"` but rejected:
- **Breaking change**: existing `tree2vec_embeddings` / `forte_embeddings` boolean flags control provider creation
- **Multiple providers**: users may want tree2vec AND rnn2vec simultaneously (one for T2V selection, one for R2V selection)
- **Pattern consistency**: follow the existing `{name}_embeddings: bool` pattern

Instead, keep the established pattern:
```python
rnn2vec_embeddings: bool = False   # master switch
rnn2vec_weight: float = 0.0       # selection ratio (0 = disabled)
```

---

## 6. Integration Plan: `search/ml_selection.py`

### Current state
`ml_selection.py` contains `EmbeddingEnhancedSelection` which uses a single `EmbeddingProvider`. This is used for the GNN-based ML selection pathway.

RNN2Vec does **not** integrate through `ml_selection.py`. Instead, it follows the **Tree2Vec/FORTE pattern**: a `SelectionOrder` enum value + provider used directly in `GivenClauseSearch`.

### Integration path (in `given_clause.py`)

1. **SelectionOrder**: Add `RNN2VEC = 9` to `selection.py:SelectionOrder`

2. **GivenClauseSearch.__init__**: Add `rnn2vec_weight` to selection rules:
   ```python
   if opts.rnn2vec_weight > 0:
       rules.append(SelectionRule("R2V", SelectionOrder.RNN2VEC, part=opts.rnn2vec_weight))
   ```

3. **Provider initialization** (in `_maybe_init_ml_providers` or equivalent):
   ```python
   if opts.rnn2vec_embeddings:
       from pyladr.ml.rnn2vec.provider import RNN2VecEmbeddingProvider, RNN2VecProviderConfig
       # Build config from opts, create provider
   ```

4. **Selection dispatch** (in `_make_inferences` or equivalent):
   When `current_order == SelectionOrder.RNN2VEC`, score SOS clauses by RNN2Vec embedding diversity (same pattern as TREE2VEC).

5. **PrioritySOS**: Add `RNN2VEC` to the heap-backed extraction dispatch in `selection.py:_pop_from_priority_sos`.

6. **Online learning**: Follow the tree2vec background updater pattern — `BackgroundR2VUpdater` daemon thread that calls `rnn2vec.update_online()` on batches of kept clauses.

### No factory function needed
A `create_embedding_provider(options)` factory is unnecessary complexity. The initialization is ~20 lines in `GivenClauseSearch.__init__` (same as tree2vec), and each provider has distinct configuration needs.

---

## 7. Benchmark Plan

### File: `tests/benchmarks/bench_rnn2vec_performance.py`

```python
"""Benchmarks for RNN2Vec vs Tree2Vec performance.

Measures:
1. Training time (corpus → trained model)
2. Single embedding time (clause → vector)
3. Batch embedding time (N clauses → N vectors)
4. Online update time (batch of kept clauses → updated model)
5. Memory usage (peak RSS during training and embedding)
"""
```

### Benchmark scenarios

| Scenario | Clauses | Tokens | Expected T2V time | Expected R2V time |
|----------|---------|--------|--------------------|--------------------|
| Small (vampire.in) | ~30 | ~8 | <0.5s | <2s |
| Medium (group theory) | ~200 | ~30 | <2s | <5s |
| Large (ring theory) | ~1000 | ~60 | <5s | <15s |

### Metrics to collect
- Wall clock time (training, single embed, batch embed, online update)
- Memory (peak RSS via `resource.getrusage`)
- Embedding quality: cosine similarity between structurally similar clauses
- Cache hit rates after 1000 embed calls

### Running
```bash
pytest tests/benchmarks/bench_rnn2vec_performance.py -v --benchmark-json=rnn2vec_bench.json
```

---

## 8. Phased Implementation Roadmap

### Phase 1: Core RNN Algorithm + Provider (no online learning)
**Files:** `rnn2vec/__init__.py`, `algorithm.py`, `encoder.py`, `tokenizer.py`, `provider.py`, `formula_processor.py`

Tasks:
1. Implement `TokenVocab` (tokenizer.py) — vocabulary builder with PAD/UNK
2. Implement `RNNEncoder` (encoder.py) — nn.Embedding + nn.GRU/LSTM + projection + 3 composition strategies
3. Implement `RNN2Vec` (algorithm.py) — train, embed_term, embed_clause, embed_clauses_batch, save, load
4. Implement contrastive training loop (algorithm.py) — InfoNCE loss over walk pairs
5. Implement `RNN2VecEmbeddingProvider` (provider.py) — factory methods, cache, thread safety
6. Implement `process_rnn2vec_corpus` (formula_processor.py) — reuse augmentation from tree2vec
7. Write `__init__.py` with public API exports
8. Unit tests for each component

**Estimated tasks:** 8-10
**Estimated effort:** Core implementation sprint

### Phase 2: Online Learning Support
**Files:** `algorithm.py` (extend), `provider.py` (extend), new `background_updater.py`

Tasks:
1. Implement `RNN2Vec.update_online()` — incremental gradient steps on new clauses
2. Implement vocab extension for OOV tokens during online updates
3. Implement `BackgroundR2VUpdater` (background_updater.py) — daemon thread pattern from tree2vec
4. Add `bump_model_version()` + version-aware cache invalidation
5. Unit tests for online learning

**Estimated tasks:** 5-6

### Phase 3: Search Integration
**Files:** `selection.py`, `given_clause.py`, `options.py`, `cli_common.py`

Tasks:
1. Add `SelectionOrder.RNN2VEC` enum value
2. Add `rnn2vec_*` fields to `SearchOptions`
3. Add `rnn2vec_*` bounds to `options.py`
4. Add provider initialization in `GivenClauseSearch.__init__`
5. Add selection dispatch for `RNN2VEC` order
6. Add PrioritySOS support for RNN2VEC
7. Add CLI directive parsing for `set(rnn2vec_embeddings)` etc.
8. Add goal proximity support (optional, follow tree2vec pattern)
9. Integration tests

**Estimated tasks:** 8-10

### Phase 4: Performance Optimization + Benchmarks
**Files:** `bench_rnn2vec_performance.py`, possible encoder optimizations

Tasks:
1. Write benchmark suite
2. Profile training bottlenecks
3. Optimize batch embedding (pad + batch RNN forward in single call)
4. Optimize cache miss batching in provider
5. Consider torch.compile() for encoder if torch >= 2.0
6. Write comparison report vs Tree2Vec

**Estimated tasks:** 4-6

### Total estimated tasks: 25-32

### Dependencies between phases

```
Phase 1 ──→ Phase 2 ──→ Phase 3
                  └──→ Phase 4 (can start after Phase 1)
```

Phase 4 benchmarks can begin after Phase 1 (basic training/embedding works). Phase 3 integration requires Phase 2 for online learning hooks.
