# Design: Enhanced SearchStatistics with Per-Given-Clause Inference Tracking

## Overview

Add per-given-clause inference counting to `SearchStatistics` so that after a search
completes, callers can query how many inferences (generated clauses) each given clause
produced. This enables analysis of which given clauses were most productive.

## Design Principles

1. **Zero impact on existing functionality** — all new fields use `default_factory`;
   existing code paths untouched.
2. **C Prover9 output format preserved** — `report()` output unchanged; new data
   available only through new accessor methods.
3. **No thread-safety concerns** — `_cl_process()` always runs sequentially on the
   main thread (even when inference generation is parallelized, processing is
   sequential per the `ParallelInferenceEngine` architecture).
4. **`slots=True` compatible** — new fields declared as dataclass fields with proper
   type annotations.

## SearchStatistics Changes

### New Fields

```python
@dataclass(slots=True)
class SearchStatistics:
    # ... existing fields unchanged ...

    # Per-given-clause inference tracking
    given_inference_counts: dict[int, int] = field(default_factory=dict)
    _current_given_id: int = 0  # ID of the given clause currently being processed
```

- `given_inference_counts`: Maps `clause.id → count` of clauses generated while
  that clause was the active given clause.
- `_current_given_id`: Tracks which given clause is currently active. Set when a
  given clause is selected, read when `_cl_process` increments `generated`.

### New Methods

```python
def begin_given(self, clause_id: int) -> None:
    """Mark the start of inference generation for a given clause.

    Called once per given clause, right after stats.given is incremented.
    Initializes the counter for this clause and sets it as current.
    """
    self._current_given_id = clause_id
    self.given_inference_counts[clause_id] = 0

def record_generated(self) -> None:
    """Record one generated clause, attributing it to the current given clause.

    Replaces direct `self.generated += 1` in _cl_process.
    Increments both the global `generated` counter and the per-given counter.
    """
    self.generated += 1
    if self._current_given_id != 0:
        self.given_inference_counts[self._current_given_id] = (
            self.given_inference_counts.get(self._current_given_id, 0) + 1
        )

def get_given_inference_count(self, clause_id: int) -> int:
    """Return how many clauses were generated from a specific given clause.

    Returns 0 for unknown clause IDs (safe default).
    """
    return self.given_inference_counts.get(clause_id, 0)

def top_given_clauses(self, n: int = 10) -> list[tuple[int, int]]:
    """Return the top-N most productive given clauses by inference count.

    Returns list of (clause_id, count) sorted descending by count.
    """
    return sorted(
        self.given_inference_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:n]
```

### report() — UNCHANGED

The `report()` method output is **not modified**. It continues to produce the exact
same C-compatible format. The new data is accessible only through the new methods.

## GivenClauseSearch Integration Points

### 1. `_make_inferences()` — Call `begin_given()` after incrementing `stats.given`

```python
# In _make_inferences(), after stats.given += 1:
self._state.stats.begin_given(given.id)
```

### 2. `_process_initial_clauses()` — Same pattern for initial (I) clauses

```python
# In _process_initial_clauses(), after stats.given += 1:
self._state.stats.begin_given(c.id)
```

### 3. `_cl_process()` — Replace direct increment with `record_generated()`

```python
# In _cl_process(), replace:
#   self._state.stats.generated += 1
# with:
self._state.stats.record_generated()
```

This is the **only line that changes** in `_cl_process`. The method signature,
return type, and all other behavior remain identical.

## Thread Safety Analysis

No additional synchronization is needed:

- **Parallel inference generation** (`_given_infer_parallel`): Workers produce
  clause lists but do NOT call `_cl_process`. The engine collects results and
  then processes them sequentially (line 705-708 in given_clause.py).
- **Sequential processing**: `_cl_process` is always called from the main thread
  in sequence, so `record_generated()` has no concurrent access.
- **`_current_given_id`**: Set once in `_make_inferences` before any `_cl_process`
  calls, read during those calls — single-threaded, no race.

## Backward Compatibility Guarantees

| Aspect | Impact |
|--------|--------|
| `SearchStatistics()` default construction | No change — new fields have defaults |
| `report()` output | Identical — no new fields in report |
| `stats.generated` value | Identical — `record_generated()` increments it the same way |
| `stats.given` value | Identical — `begin_given()` doesn't touch it |
| Exit codes | Unchanged — all limit checks use same counters |
| Clause processing order | Unchanged — no algorithmic changes |
| Memory | ~O(G) dict entries where G = number of given clauses (negligible) |

## File Changes Summary

| File | Change |
|------|--------|
| `pyladr/search/statistics.py` | Add 2 fields, 4 methods |
| `pyladr/search/given_clause.py` | 3 one-line changes (2 `begin_given` calls, 1 `record_generated` call) |

Total: ~30 lines of new code, 1 line modified.
