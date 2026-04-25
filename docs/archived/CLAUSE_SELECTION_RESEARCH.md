# PyLADR Clause Selection Mechanisms: Research Report

## Executive Summary

The PyLADR codebase implements a sophisticated two-layer clause selection system:
1. **Traditional Selection (selection.py)**: Ratio-based cycling between weight and age selectors, matching Prover9's given-clause algorithm
2. **ML-Enhanced Selection (ml_selection.py)**: Optional embedding-based scoring that blends traditional metrics with ML signals

The architecture is highly modular with clear integration points for adding new bias mechanisms like repetition detection.

---

## 1. Traditional Selection Architecture

### Core Classes

#### `GivenSelection` (selection.py:46-212)
The primary selection manager implementing the ratio-based cycling algorithm.

**Key Attributes:**
- `rules: list[SelectionRule]` - Ordered list of selection rules (default: weight 5:age 1)
- `_initial_queue: deque[Clause]` - Initial clauses processed first in ID order
- `_weight_heap: list[tuple[float, int, Clause]]` - Min-heap for O(log n) weight-based selection
- `_removed_ids: set[int]` - Lazy deletion tracker (prevents actual heap removal cost)
- `_current_idx, _count, _cycle_size` - Cycle state management

**Key Methods:**

1. **`select_given(sos: ClauseList, given_count: int) -> tuple[Clause | None, str]`** (line 113)
   - Main entry point for clause selection
   - Two-phase process:
     - **Phase 1 (Initial Pass)**: Process initial clauses in ID order (marked "I")
     - **Phase 2 (Ratio Cycle)**: Cycle through rules with specified ratios (marked "T"/"A")
   - Returns selected clause and selection type label

2. **`add_clause_to_selectors(c: Clause) -> None`** (line 96)
   - Registers a clause for weight-based selection
   - Pushes to min-heap: `(weight, id, clause)`
   - Called when clause added to SOS

3. **`remove_clause_from_selectors(c: Clause) -> None`** (line 104)
   - Marks clause as removed (lazy deletion)
   - Adds to `_removed_ids` set
   - Actual heap removal deferred to pop time (O(1) vs O(log n))

4. **`_get_current_rule() -> SelectionRule`** (line 157)
   - Determines which rule to use based on cycle position
   - Implements ratio-based cycling via cumulative sum

5. **`_select_by_order(sos: ClauseList, order: SelectionOrder) -> Clause | None`** (line 174)
   - **AGE**: Returns `sos.first` (FIFO, O(1))
   - **WEIGHT**: Calls `_pop_lightest_valid()` from heap (O(log n) amortized)
   - Falls back to linear scan if heap unavailable

#### `SelectionRule` (line 30)
Represents a single selection strategy.

**Attributes:**
- `name: str` - Label ("T" for weight, "A" for age, "I" for initial)
- `order: SelectionOrder` - WEIGHT, AGE, or RANDOM
- `part: int` - Ratio weight (e.g., 5 for weight means 5 selections per cycle)
- `selected: int` - Count of clauses selected by this rule

#### `SelectionOrder` (line 22)
Enum: `WEIGHT=0`, `AGE=1`, `RANDOM=2`

### Weight Calculation

#### `default_clause_weight(c: Clause) -> float`** (line 237)
```python
def default_clause_weight(c: Clause) -> float:
    """Default clause weight: number of symbols in all literals."""
    total = 0
    for lit in c.literals:
        total += lit.atom.symbol_count
    return float(total)
```
- Sums symbol count across all literals in the clause
- Simpler clauses have lower weight (preferred in traditional selection)
- Uses cached `symbol_count` on Term objects

### Selection Flow

```
select_given() called
├─ Phase 1: Initial clauses
│  ├─ Pop from _initial_queue (FIFO)
│  ├─ Check if still in SOS (not subsumed)
│  ├─ Return with label "I"
│  └─ Repeat until queue empty
│
└─ Phase 2: Ratio cycle
   ├─ Get current rule via _get_current_rule()
   ├─ Advance cycle counter
   ├─ Select by rule.order
   │  ├─ WEIGHT: pop from heap (O(log n))
   │  └─ AGE: take first from SOS (O(1))
   ├─ Remove from SOS
   ├─ Mark as removed in _removed_ids
   └─ Return with rule.name label ("T" or "A")
```

### Integration with Search Loop (given_clause.py)

**Initialization (lines 296-303):**
```python
for c in self._state.sos:
    c.weight = default_clause_weight(c)
    self._selection.add_clause_to_selectors(c)        # Register for weight heap
    if c.initial:
        self._selection.mark_initial(c)                # Queue for initial pass
    self._state.index_subsumption(c, insert=True)
```

**Main Loop Selection (line 329):**
```python
given, selection_type = self._selection.select_given(
    self._state.sos,
    self._state.stats.given,
)
```

**Removal on Subsumption (lines 711, 738):**
```python
self._selection.remove_clause_from_selectors(victim)   # Lazy deletion
self._state.sos.remove(victim)
```

**Addition on New Kept Clause (line 756):**
```python
self._state.sos.append(c)
self._selection.add_clause_to_selectors(c)             # Register for future selection
```

---

## 2. ML-Enhanced Selection Architecture

### Core Classes

#### `EmbeddingEnhancedSelection` (ml_selection.py:140-410)
Extends `GivenSelection` to blend traditional scoring with ML signals.

**Key Attributes:**
- `embedding_provider: EmbeddingProvider | None` - Source of clause embeddings
- `ml_config: MLSelectionConfig` - Configuration for ML-based scoring
- `ml_stats: MLSelectionStats` - Tracks ML selection effectiveness
- `_recent_embeddings: deque[list[float]]` - Window of recent given clause embeddings (for diversity)

**Key Methods:**

1. **`select_given(sos: ClauseList, given_count: int) -> tuple[Clause | None, str]`** (line 187)
   - Overrides parent `select_given()`
   - Decision tree:
     ```
     if (rule.order == WEIGHT and _should_use_ml(sos)):
         selected, score = _ml_select(sos)
         if selected:
             return selected, "W+ML"
     # Fallback or age-based: use traditional
     return _select_by_order(sos, rule.order)
     ```
   - Only applies ML to weight-based steps (age-based always uses FIFO for fairness)
   - Tracks statistics in `ml_stats`

2. **`_should_use_ml(sos: ClauseList) -> bool`** (line 241)
   - Gating function: returns True iff:
     - ML is enabled (`ml_config.enabled`)
     - Provider available (`embedding_provider is not None`)
     - SOS size sufficient (`sos.length >= ml_config.min_sos_for_ml`)

3. **`_ml_select(sos: ClauseList) -> tuple[Clause | None, float]`** (line 249)
   - Wrapper with exception handling
   - Calls `_ml_select_inner()` with fallback on error
   - Returns (best_clause, best_score) or (None, 0.0)

4. **`_ml_select_inner(sos: ClauseList) -> tuple[Clause | None, float]`** (line 264)
   - **Core ML scoring logic**
   - For each clause in SOS:
     - Get embedding from provider
     - Compute **traditional score**: adaptive weight preference
     - Compute **ML score** if embedding available
     - Blend scores: `(1 - ml_weight) * trad + ml_weight * ml_score`
   - Return clause with highest blended score

5. **`_compute_ml_score(embedding: list[float], clause_weight: float, ml_weight: float) -> float`** (line 327)
   - Blends three components:
     - **Diversity**: Distance from recent given clause embeddings (encourages exploration)
     - **Proof Potential**: Embedding norm inverse (smaller norm = more productive)
     - **Weight Exploration**: Log-scaled clause weight (progressive as ml_weight increases)
   - Weights configured via `MLSelectionConfig`

6. **`_diversity_score(embedding: list[float]) -> float`** (line 359)
   - Average cosine distance to recent given clause embeddings
   - Returns [0, 1]: 1 = maximally diverse
   - Uses window of recent embeddings from `_recent_embeddings`

7. **`_proof_potential_score(embedding: list[float]) -> float`** (line 381)
   - Inverse sigmoid of embedding norm
   - Learned correlation: productive clauses have smaller embeddings
   - Range [0, 1]: higher = better

8. **`_record_embedding(clause: Clause) -> None`** (line 400)
   - Records embedding of selected clause for future diversity scoring
   - Updates `_recent_embeddings` deque

#### `MLSelectionConfig` (line 68)
Configuration dataclass for ML-enhanced selection.

**Key Fields:**
```python
enabled: bool = False                          # Master switch
ml_weight: float = 0.3                         # 0=pure traditional, 1=pure ML
diversity_weight: float = 0.5                  # Weight of diversity in ML score
proof_potential_weight: float = 0.5            # Weight of proof-potential in ML score
diversity_window: int = 20                     # Recent givens tracked
min_sos_for_ml: int = 10                       # Minimum SOS size for ML activation
fallback_on_error: bool = True                 # Graceful degradation
log_selections: bool = False                   # Debug logging
```

#### `EmbeddingProvider` (Protocol, line 37)
Interface for embedding providers.

```python
@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def embedding_dim(self) -> int:
        """Dimensionality of produced embeddings."""

    def get_embedding(self, clause: Clause) -> list[float] | None:
        """Return embedding vector for a clause, or None if unavailable."""

    def get_embeddings_batch(self, clauses: list[Clause]) -> list[list[float] | None]:
        """Batch embedding retrieval."""
```

#### `MLSelectionStats` (line 98)
Tracks ML selection effectiveness.

**Key Fields:**
- `ml_selections: int` - Count of ML-based selections
- `traditional_selections: int` - Count of traditional selections
- `fallback_count: int` - Count of ML failures → fallback
- `embedding_miss_count: int` - Count of missing embeddings
- `avg_ml_score: float` - Average score of ML selections

### Scoring Algorithm Details

**In `_ml_select_inner()` (lines 264-325):**

```python
# Compute weight statistics for normalization
weights = [c.weight for c in clauses]
max_w, min_w = max(weights), min(weights)
w_range = max_w - min_w if max_w > min_w else 1.0

best_clause, best_score = None, -∞

for clause in clauses:
    # 1. Traditional score: weight preference strength depends on ml_weight
    base_weight_score = 1.0 - (clause.weight - min_w) / w_range
    weight_bias_strength = 1.0 - ml_weight  # Decreases as ML influence increases
    trad_score = weight_bias_strength * base_weight_score + (1.0 - weight_bias_strength) * 0.5

    # 2. Get embedding (if available)
    emb = embedding_provider.get_embedding(clause)

    if emb is None:
        score = trad_score  # Use traditional only
    else:
        # 3. ML score with three components
        raw_ml_score = _compute_ml_score(emb, clause.weight, ml_weight)
        ml_score = raw_ml_score * 10.0  # Scale to compete with traditional

        # 4. Blend traditional and ML
        score = (1.0 - ml_weight) * trad_score + ml_weight * ml_score

    if score > best_score:
        best_score, best_clause = score, clause

return best_clause, best_score
```

**Key Design Points:**
- Weight bias is **progressive**: as `ml_weight` increases, the traditional scorer becomes more neutral (0.5)
- ML score is **scaled 10x** to match traditional score range
- **Fallback to traditional**: if embedding unavailable or provider error
- **Fairness**: age-based selection never uses ML (always FIFO)

---

## 3. Clause Management and State

### `ClauseList` (state.py:19-86)
Manages ordered clause lists (usable, sos, limbo, disabled).

**Key Methods:**
- `append(c)` - Add clause (O(1))
- `remove(c)` - Lazy deletion via ID set (O(1))
- `contains(c)` - Check membership (O(1))
- `first` - Get first non-removed clause (O(1) amortized)
- `__iter__` - Iterate over active clauses only

**Lazy Deletion Strategy:**
- Maintains `deque[Clause]` and `set[int]` of active clause IDs
- Removal marks clause ID as inactive (doesn't modify deque)
- Iteration skips inactive clauses
- Cost: O(1) removal, O(1) amortized access

### `SearchState` (state.py:89+)
Global search state with clause lists and indexes.

**Key Attributes:**
- `usable: ClauseList` - Available clauses for inference
- `sos: ClauseList` - Candidates for given clause selection
- `limbo: ClauseList` - Clauses awaiting processing
- `disabled: ClauseList` - Subsumed/demodulated clauses

---

## 4. Related Systems

### Inference Guidance (inference_guidance.py)
**Purpose:** Prioritizes which usable clauses to attempt inference with (NOT clause selection itself).

**Key Config (InferenceGuidanceConfig):**
- `enabled` - Master switch
- `max_candidates` - Limit usable clauses per given
- `compatibility_threshold` - Minimum score threshold
- `structural_weight` - Structural compatibility weight
- `semantic_weight` - Embedding similarity weight

**Decoupled from Selection:** Guidance filters **inferences** (which clauses to try with given), selection chooses **which clause is given**.

### Online Integration (online_integration.py)
**Purpose:** Connects online learning with the search loop.

**Key Functions:**
- Collects inference outcomes as training data
- Triggers model updates during search
- Manages cache invalidation after learning
- Adapts `ml_weight` dynamically based on learning progress

**Decoupled from Selection:** Manages **learning pipeline**, not selection directly.

---

## 5. Integration Points for New Bias Mechanisms

### Direct Integration Points

#### 1. **Add New Selection Rule Type**
- Create new `SelectionOrder` enum value
- Implement selection logic in `_select_by_order()` method
- Add rule to default rules in `__post_init__()`
- Example: Repetition-based selection could check structural similarity

**Files:** `selection.py` lines 22-28, 174-198

#### 2. **Extend ML Scoring Components**
- Add new factor to `_compute_ml_score()` method
- Mix with diversity, proof_potential, weight_exploration via weighted sum
- Add corresponding configuration fields to `MLSelectionConfig`

**Example Code Pattern:**
```python
def _repetition_bias_score(self, clause: Clause) -> float:
    """Score clause based on structural repetition."""
    # Implementation here
    return score_value

def _compute_ml_score(self, embedding, clause_weight, ml_weight):
    # ... existing code ...
    repetition_score = self._repetition_bias_score(clause)
    # Add to blend
    return (...existing blend...) + rep_w * repetition_score
```

**Files:** `ml_selection.py` lines 327-357

#### 3. **Override Default Weight Calculation**
- Modify `default_clause_weight()` function
- Or provide custom weight calculator to `GivenSelection`
- Weight influences both traditional and ML scoring

**Files:** `selection.py` lines 237-246

#### 4. **Custom Selection Rule Cycler**
- Override `_get_current_rule()` to implement custom cycling logic
- Could weight rules based on search state, clause properties, etc.

**Files:** `selection.py` lines 157-168

#### 5. **Track and Use Clause Metadata**
- Add fields to `Clause` class (or use attributes dynamically)
- Compute repetition statistics during search
- Use in scoring functions

**Files:** `core/clause.py`

#### 6. **Extend EmbeddingProvider**
- Implement custom provider that includes repetition information in embedding
- Returns higher-dimensional vectors encoding structural patterns
- No changes to selection code needed

**Files:** Implement in separate module, configure in `MLSelectionConfig`

### Hook Points in Search Loop

**Clause Registration (given_clause.py:299):**
```python
# Called when clause added to SOS - good place to compute/cache repetition metrics
self._selection.add_clause_to_selectors(c)
```

**Clause Removal (given_clause.py:711, 738):**
```python
# Called when clause removed via subsumption/demodulation
self._selection.remove_clause_from_selectors(victim)
```

**Main Selection Call (given_clause.py:329):**
```python
# Returns selection type label - can be extended with custom labels
given, selection_type = self._selection.select_given(...)
```

---

## 6. Key Design Patterns

### Pattern 1: Graceful Degradation (ML-Enhanced Selection)
- All ML features are **opt-in** via `ml_config.enabled`
- When disabled or provider unavailable: behavior identical to `GivenSelection`
- Exception handling falls back to traditional selection

### Pattern 2: Lazy Deletion
- Remove operations O(1) via ID set marking
- Actual cleanup deferred to pop time
- Reduces contention in hot path

### Pattern 3: Ratio-Based Cycling
- Implements **fairness** by guaranteeing selection rule proportions
- Default 5:1 weight:age matches Prover9 behavior
- Extensible to arbitrary numbers of rules

### Pattern 4: Progressive ML Integration
- `ml_weight` parameter controls traditional vs ML influence
- As `ml_weight` increases, traditional scorer becomes less biased
- Allows smooth transition from traditional to ML-guided search

### Pattern 5: Protocol-Based Extensibility
- `EmbeddingProvider` is Protocol (structural typing)
- Any object with right methods works - no explicit inheritance needed
- Easy to compose with other systems

---

## 7. File Organization Summary

| File | Responsibility | Key Classes |
|------|-----------------|-------------|
| `selection.py` | Traditional clause selection | `GivenSelection`, `SelectionRule`, `SelectionOrder` |
| `ml_selection.py` | ML-enhanced selection | `EmbeddingEnhancedSelection`, `MLSelectionConfig`, `EmbeddingProvider` |
| `given_clause.py` | Main search loop integration | `GivenClauseSearch`, integrates selection |
| `state.py` | Clause list management | `ClauseList`, `SearchState` |
| `statistics.py` | Search metrics | `SearchStatistics` |
| `inference_guidance.py` | Inference targeting (not selection) | `InferenceGuidanceConfig` |
| `online_integration.py` | Online learning integration | `OnlineIntegrationConfig` |

---

## 8. Testing Approach

### Selection Tests (test_given_clause.py:234-268)
- `test_default_weight_calculation()` - Weight computation
- `test_selection_ratio_cycling()` - Ratio-based cycling behavior

### ML Selection Tests (test_ml_selection.py)
- Backward compatibility tests
- ML scoring component tests
- Fallback behavior tests
- Statistics tracking tests

**Test Pattern:**
```python
# Setup
selector = GivenSelection()  # or EmbeddingEnhancedSelection
sos = ClauseList("test")
for i in range(n):
    c = Clause(...)
    sos.append(c)

# Select
given, sel_type = selector.select_given(sos, given_count=0)

# Verify
assert given is not None
assert sel_type in ["I", "T", "A", "T+ML"]
```

---

## 9. Recommendations for Repetition Bias Implementation

### Approach 1: Extend ML Scoring (Recommended)
**Pros:**
- Minimal code changes
- Leverages existing blending architecture
- Can be configured via `MLSelectionConfig`
- Graceful degradation built-in

**Implementation:**
1. Add `repetition_weight` to `MLSelectionConfig`
2. Implement `_repetition_bias_score()` method
3. Integrate into `_compute_ml_score()` blend
4. Track repetition metrics during `_record_embedding()`

**Code Location:** `ml_selection.py` lines 327-357

### Approach 2: Custom Selection Rule
**Pros:**
- Can apply to age-based selection too
- Full control over ranking

**Implementation:**
1. Create `SelectionOrder.REPETITION` enum
2. Override `_select_by_order()` in custom class
3. Implement repetition-based ranking

**Code Location:** `selection.py` lines 174-198, 22-28

### Approach 3: Modify Weight Calculation
**Pros:**
- Simplest - affects all existing selection strategies
- No new components needed

**Implementation:**
1. Add repetition penalty to `default_clause_weight()`
2. Or provide custom weight calculator

**Code Location:** `selection.py` lines 237-246

**Recommendation:** Start with **Approach 1** (extend ML scoring), then optionally add **Approach 2** (custom rule) for flexibility.

---

## 10. Code Snippets for Reference

### Default ML Configuration
```python
from pyladr.search.ml_selection import MLSelectionConfig
config = MLSelectionConfig(
    enabled=True,
    ml_weight=0.3,
    diversity_weight=0.5,
    proof_potential_weight=0.5,
    min_sos_for_ml=10,
)
```

### Creating ML-Enhanced Selection
```python
from pyladr.search.ml_selection import EmbeddingEnhancedSelection
selection = EmbeddingEnhancedSelection(
    embedding_provider=my_provider,
    ml_config=config,
)
search = GivenClauseSearch(options, selection=selection)
```

### Accessing Selection Statistics
```python
result = search.run(usable, sos)
if hasattr(search._selection, 'ml_stats'):
    print(search._selection.ml_stats.report())
```

---

## Conclusion

PyLADR's selection architecture is well-designed for extensibility:
- **Clear separation**: traditional vs ML-enhanced
- **Modular design**: each component has single responsibility
- **Multiple integration points**: scoring, rule cycling, weight calculation
- **Graceful degradation**: ML features are opt-in
- **Extensible protocols**: can add new embedding providers without modifying core

For implementing repetition-based bias against structurally repetitious clauses, the recommended approach is to extend the ML scoring component, leveraging the existing blending framework and configuration system.

