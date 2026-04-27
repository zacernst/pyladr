# PyLADR Performance Guide

## Performance Characteristics

### Typical Throughput

PyLADR throughput varies significantly by problem type. Measured on
representative benchmark problems (April 2026):

| Problem type | given/sec | generated/sec | Notes |
|-------------|-----------|---------------|-------|
| Equational (para+demod+back_demod) | 6–9 | 600–1,100 | Ring commutativity, lattice problems |
| Single-predicate equational | ~16 | — | vampire.in with back-subsumption weight filter + Context/Trail reuse |
| Resolution + hyperresolution | 4–6 | 1,000–1,400 | Condensed detachment (Luka20) |

The given/sec rate is the primary throughput metric — it reflects selection,
inference, simplification, and subsumption per iteration of the main loop.

### Bottleneck Breakdown

For a typical equational search (paramodulation + demodulation + back-demod):

| Phase | % Time | Notes |
|-------|--------|-------|
| Back-subsumption + back-demodulation | 55–70% | Dominates on medium/large problems |
| Forward demodulation | 15–20% | Term rewriting of new clauses |
| Inference generation | 6–10% | Paramodulation, resolution |
| Forward subsumption | 5–8% | Checking new clauses against existing |
| Selection + bookkeeping | 2–5% | Given clause selection, statistics |

Back-subsumption is the dominant cost because each kept clause must be
checked against all existing clauses for subsumption. On non-equational
problems (resolution only), forward+backward subsumption together account
for 80–90% of search time.

## Optimization Strategies

### 1. Search Limits

Always set search limits to avoid runaway searches:

```bash
uv run pyprover9 -f problem.in \
    -max_given 1000 \
    -max_seconds 60 \
    -max_generated 50000
```

### 2. Enable Demodulation for Equational Problems

Demodulation aggressively simplifies clauses, reducing the search space:

```bash
uv run pyprover9 -f problem.in --paramodulation --demodulation
```

Back demodulation (`--back-demod`) further reduces redundancy by re-simplifying existing clauses when new demodulators are discovered. This has overhead but pays off for heavily equational problems.

### 3. Choose the Right Inference Rules

- **Pure equational problems:** Use `--paramodulation --demodulation --no-resolution`
- **Mixed logic + equality:** Use default resolution + `--paramodulation --demodulation`
- **No equality:** Default resolution + factoring is usually sufficient

### 4. Weight-Based Filtering

Use `max_weight` to discard heavy (complex) clauses:

```
% In input file:
set(max_weight, 30).
```

Lighter clauses are generally more useful. Aggressive weight limits reduce memory and speed up subsumption, but too aggressive risks discarding needed clauses.

## Penalty-Based Search Optimization

The penalty weight system can significantly improve search efficiency by
deprioritizing overly general and repetitive clauses. See the
[Penalty Weight Guide](PENALTY_WEIGHT_GUIDE.md) for full details.

### Quick Start

```bash
# Conservative: mild deprioritization of general clauses
uv run pyprover9 -f problem.in --penalty-propagation --penalty-weight

# Moderate: add repetition detection
uv run pyprover9 -f problem.in \
    --penalty-propagation --repetition-penalty --penalty-weight

# Aggressive: strong suppression for bloated searches
uv run pyprover9 -f problem.in \
    --penalty-propagation --penalty-propagation-decay 0.8 \
    --repetition-penalty --repetition-penalty-normalize \
    --penalty-weight --penalty-weight-multiplier 3.0
```

### Performance Impact

| Component | Per-clause overhead | Memory |
|-----------|-------------------|--------|
| Penalty propagation | O(1) cache + O(k) parents | ~80 bytes/clause |
| Repetition penalty (exact) | O(n) subterms | Negligible |
| Repetition penalty (normalized) | O(n log n) | Per-clause cache |
| Penalty weight adjustment | O(1) arithmetic | None |

Total overhead is typically < 5% of search time. The search efficiency
improvement from reduced clause bloat generally far exceeds this cost.

## Parallel Execution

### Requirements

- **Python 3.14+** with free-threading enabled (`--disable-gil` build)
- Falls back to sequential on GIL Python (no overhead penalty)

### How It Works

The parallel engine splits the most expensive phase — inference generation — across multiple threads:

```
Sequential: select_given → move_to_usable → index
Parallel:   generate_inferences(given, usable_chunks)
Sequential: process each inference → limbo_process
```

Each thread processes a chunk of the usable set independently. Thread-local `Context`/`Trail` objects eliminate contention during unification.

### Configuration

```python
from pyladr.parallel.inference_engine import ParallelSearchConfig
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

config = ParallelSearchConfig(
    enabled=True,
    max_workers=4,               # Thread count (None = cpu_count)
    min_usable_for_parallel=50,  # Don't parallelize small sets
    chunk_size=25,               # Usable clauses per work unit
)

options = SearchOptions(parallel=config)
search = GivenClauseSearch(options=options)
```

### When Parallelism Helps

- **Large usable sets** (100+ clauses) — parallelism amortizes overhead
- **Complex inference rules** (paramodulation, AC unification) — more work per clause
- **Problems with many given clauses** — cumulative speedup

### When It Doesn't Help

- **Small problems** (< 50 usable clauses) — overhead exceeds benefit
- **GIL Python** — threads share the GIL, no true parallelism
- **I/O bound** — parsing/output phases are inherently sequential

### Expected Speedup

Target: **2–4x** with 4 threads on problems with 100+ given clauses.

Single-threaded overhead of the parallel infrastructure: ~5–10% (thread pool management, snapshot copies).

## Indexing Performance

### Discrimination Trees

PyLADR uses discrimination trees for efficient term retrieval:

- **`DiscrimWild` (WILD):** Fast imperfect filter. Returns supersets of matches — callers must verify. Best for resolution where false positives are cheap to filter.
- **`DiscrimBind` (BIND):** Slower but exact. Produces substitutions directly. Better when unification is expensive (AC).

The default (`WILD`) is appropriate for most problems.

### Feature Vector Indexing

`FeatureIndex` provides fast prefiltering for subsumption. It computes lightweight feature vectors (symbol counts, variable counts) and uses them to prune candidates before expensive subsumption checking.

## Memory Considerations

### Clause Lifecycle

1. **Generated** — Created by inference rules
2. **Kept** — Passes forward subsumption, tautology, weight checks
3. **Usable** — Available for inference (indexed)
4. **Deleted** — Removed by back subsumption or back demodulation

Aggressive forward subsumption and weight limits keep memory bounded.

### Term Sharing

Terms are immutable (frozen dataclasses) and can be safely shared. Variable terms are cached via `get_variable_term()` to avoid allocation. The `clear_term_caches()` function can release cached terms if memory is a concern.

## Benchmarking

### Running Benchmarks

```bash
# Run performance benchmarks
uv run pytest -m benchmark tests/benchmarks/

# Compare against C baselines
uv run pytest tests/benchmarks/test_performance.py -v
```

### Benchmark Infrastructure

`tests/benchmarks/bench_harness.py` provides timing and comparison utilities. `tests/benchmarks/c_baselines.py` stores reference timing data from the C implementation.

## Nucleus Unification Penalty

The nucleus penalty system prevents unification explosion in hyperresolution-heavy problems by deprioritizing overly general nucleus clauses.

### When to Enable

Enable when your problem:
- Uses hyperresolution (`set(hyper_resolution).`)
- Contains condensed detachment patterns (`-P(i(x,y)) | -P(x) | P(y)`)
- Generates excessive derived clauses from variable-heavy negative literals

### Configuration

```
set(nucleus_unification_penalty).
assign(nucleus_penalty_weight, 5.0).    % Base penalty (higher = more aggressive)
assign(nucleus_penalty_threshold, 0.3). % Min generality ratio to trigger
assign(nucleus_penalty_max, 20.0).      % Hard cap on penalty
assign(nucleus_penalty_cache_size, 10000). % LRU pattern cache entries
```

### Performance Impact

| Scenario | Overhead |
|----------|----------|
| Disabled (default) | 0% — no code paths executed |
| Enabled, typical problem | <1% — O(n) per clause, n = argument positions |
| Enabled, pattern cache full | <2% — LRU eviction adds minor bookkeeping |

### Tuning Recommendations

| Problem Type | Weight | Threshold | Max |
|-------------|--------|-----------|-----|
| Condensed detachment | 5.0–10.0 | 0.3 | 20.0 |
| General hyperresolution | 2.0–5.0 | 0.3 | 15.0 |
| Light touch (preserve order) | 1.0 | 0.5 | 8.0 |

## Profiling

### Python Profiling

```bash
# Profile a specific problem
python -m cProfile -s cumulative -m pyladr.apps.prover9 -f problem.in

# Line-level profiling (install line_profiler)
kernprof -l -v pyladr/search/given_clause.py
```

### Key Functions to Profile

- `GivenClauseSearch._make_inferences` — Main inference loop
- `all_binary_resolvents` — Resolution generation
- `para_from_into` — Paramodulation generation
- `subsumes` — Subsumption checking
- `demodulate_clause` — Demodulation
- `unify` — Core unification
