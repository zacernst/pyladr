# FORTE CLI Integration Guide

## Current Status

FORTE algorithm implementation is **complete and tested**, but **CLI integration is pending**. The core components are ready:

- ✅ `ForteAlgorithm`: Core feature extraction and embedding generation
- ✅ `ForteEmbeddingProvider`: EmbeddingProvider protocol implementation with caching
- ✅ Thread safety and performance optimization
- ✅ Comprehensive test suite
- ❌ Command-line interface integration

## Required Integration Steps

### 1. Add CLI Arguments (`pyladr/apps/prover9.py`)

Add FORTE-specific arguments to the ML argument group:

```python
# Around line 200, in the ML arguments section
ml_group.add_argument(
    "--forte-embeddings",
    action="store_true",
    help="Enable FORTE clause embeddings for enhanced selection"
)

ml_group.add_argument(
    "--forte-weight",
    type=float,
    default=0.3,
    metavar="W",
    help="Weight for FORTE embeddings in selection (0.0-1.0, default: 0.3)"
)

ml_group.add_argument(
    "--forte-dim",
    type=int,
    default=64,
    metavar="DIM",
    help="FORTE embedding dimension (default: 64)"
)

ml_group.add_argument(
    "--forte-cache",
    type=int,
    default=100000,
    metavar="SIZE",
    help="FORTE embedding cache size (default: 100000)"
)
```

### 2. Update SearchOptions (`pyladr/search/options.py`)

Add FORTE configuration fields:

```python
@dataclass(frozen=True)
class SearchOptions:
    # ... existing fields ...

    # FORTE embedding configuration
    use_forte: bool = False
    forte_weight: float = 0.3
    forte_embedding_dim: int = 64
    forte_cache_size: int = 100_000
```

### 3. Map CLI Args to SearchOptions (`pyladr/apps/prover9.py`)

In the `main()` function, around line 450:

```python
# Map FORTE arguments to SearchOptions
search_options = SearchOptions(
    # ... existing mappings ...

    # FORTE configuration
    use_forte=args.forte_embeddings,
    forte_weight=args.forte_weight,
    forte_embedding_dim=args.forte_dim,
    forte_cache_size=args.forte_cache,
)
```

### 4. Integrate with GivenClauseSearch (`pyladr/search/given_clause.py`)

Add FORTE provider initialization in `GivenClauseSearch.__init__()`:

```python
def __init__(self, options: SearchOptions, ...):
    # ... existing initialization ...

    # Initialize FORTE provider if enabled
    self._forte_provider: ForteEmbeddingProvider | None = None
    if options.use_forte:
        try:
            from pyladr.ml.forte import ForteEmbeddingProvider, ForteProviderConfig, ForteConfig

            forte_config = ForteConfig(
                embedding_dim=options.forte_embedding_dim,
                seed=42  # For reproducible embeddings
            )

            provider_config = ForteProviderConfig(
                forte_config=forte_config,
                cache_max_entries=options.forte_cache_size,
                enable_cache=True
            )

            self._forte_provider = ForteEmbeddingProvider(provider_config)
            self._logger.info(f"FORTE embeddings enabled (dim={options.forte_embedding_dim})")

        except ImportError:
            self._logger.warning("FORTE embeddings requested but not available")
            self._forte_provider = None
```

### 5. Connect to Clause Selection (`pyladr/search/ml_selection.py`)

Extend `EmbeddingEnhancedSelection` to optionally use FORTE:

```python
class EmbeddingEnhancedSelection(ClauseSelection):
    def __init__(self, ..., forte_provider: ForteEmbeddingProvider | None = None):
        # ... existing initialization ...
        self._forte_provider = forte_provider

    def select_given(self, candidates: list[Clause]) -> tuple[Clause, SelectionType]:
        if not candidates:
            return self._fallback_selection.select_given(candidates)

        # Get embeddings from both sources
        gnn_embeddings = None
        forte_embeddings = None

        if self._embedding_provider:
            gnn_embeddings = self._embedding_provider.get_embeddings_batch(candidates)

        if self._forte_provider:
            forte_embeddings = self._forte_provider.get_embeddings_batch(candidates)

        # Combine embeddings or use available source
        if gnn_embeddings and forte_embeddings:
            # Blend both embedding types
            scores = self._compute_blended_scores(candidates, gnn_embeddings, forte_embeddings)
        elif forte_embeddings:
            # Use FORTE only
            scores = self._compute_forte_scores(candidates, forte_embeddings)
        elif gnn_embeddings:
            # Use GNN only (existing behavior)
            scores = self._compute_gnn_scores(candidates, gnn_embeddings)
        else:
            # Fallback to traditional selection
            return self._fallback_selection.select_given(candidates)

        # Select best scoring clause
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return candidates[best_idx], SelectionType.ML_ENHANCED
```

### 6. Wire Everything Together

In `GivenClauseSearch.__init__()`, pass FORTE provider to selection:

```python
if options.ml_selection and options.ml_weight > 0:
    # Create embedding-enhanced selection with FORTE support
    enhanced_selection = EmbeddingEnhancedSelection(
        fallback_selection=traditional_selection,
        embedding_provider=self._embedding_provider,  # GNN provider
        forte_provider=self._forte_provider,           # FORTE provider
        ml_weight=options.ml_weight,
        forte_weight=options.forte_weight
    )
    self._clause_selection = enhanced_selection
```

## Testing the Integration

Once integrated, test with:

```bash
# Basic FORTE usage
python3 -m pyladr.apps.prover9 --forte-embeddings vampire.in

# FORTE with custom parameters
python3 -m pyladr.apps.prover9 \
    --forte-embeddings \
    --forte-weight=0.5 \
    --forte-dim=128 \
    --forte-cache=50000 \
    vampire.in

# FORTE combined with GNN embeddings
python3 -m pyladr.apps.prover9 \
    --forte-embeddings \
    --forte-weight=0.3 \
    --ml-weight=0.4 \
    --embedding-model=model.pt \
    vampire.in

# Verbose output to see FORTE statistics
python3 -m pyladr.apps.prover9 \
    --forte-embeddings \
    --verbose \
    vampire.in
```

## Expected Output

With FORTE enabled, you should see:

```
============================== Prover9 ==============================
PyProver9 version 0.1.0 (Python).
Process 12345 was started by user on machine,
Wed Apr 16 10:30:15 2026
The command was "prover9 --forte-embeddings vampire.in".
FORTE embeddings enabled (dim=64)
============================== end of head ==============================

# ... normal prover output ...

============================== STATISTICS ==============================
FORTE Statistics:
  Cache hits: 1,247
  Cache misses: 892
  Hit rate: 58.3%
  Avg embedding time: 0.022 ms
  Total embeddings generated: 2,139
============================== end of statistics ============================
```

## Performance Validation

After integration, validate:

1. **Compatibility**: Results match C Prover9 when `--forte-weight=0.0`
2. **Performance**: No significant slowdown in search loop
3. **Memory**: Cache size stays within limits
4. **Thread safety**: No race conditions during concurrent embedding requests

## Troubleshooting

**Import errors**: Ensure FORTE modules are in Python path
**Performance regression**: Check if cache hit rate is reasonable (>50%)
**Memory issues**: Reduce `--forte-cache` size
**Inconsistent results**: Verify deterministic seed is set in ForteConfig

The integration is designed to be non-breaking: FORTE is disabled by default and falls back gracefully when unavailable.