# Hierarchical GNN System: Compatibility & Integration Strategy

**Document Status:** Foundational Analysis for Hierarchical GNN Implementation
**Date:** 2026-04-08
**Critical Constraint:** NO BREAKING CHANGES vs reference C Prover9

---

## Executive Summary

The PyLADR codebase has mature ML infrastructure (EmbeddingProvider, online learning, clause graph construction) integrated into a search algorithm that maintains binary compatibility with C Prover9. The hierarchical GNN enhancement must:

1. **Preserve C Prover9 compatibility** - Exit codes, search termination, proof detection
2. **Extend, not replace** - Hierarchical embeddings enhance existing ML selection, don't replace it
3. **Respect thread-safety** - Hot-swappable models with readers-writer locks
4. **Maintain performance characteristics** - No degradation in default (non-ML) mode
5. **Enable graceful degradation** - When hierarchical GNN unavailable, fall back to existing flat GNN

---

## 1. CORE COMPATIBILITY REQUIREMENTS

### 1.1 C Prover9 Algorithm Equivalence

The `GivenClauseSearch` class in `pyladr/search/given_clause.py` (876 lines) implements the reference Prover9 search algorithm with verified equivalence to C `search.c`.

**What Must NOT Change:**

1. **Exit Codes** (lines 61-74 of given_clause.py)
   - Must match C enums exactly:
     - `MAX_PROOFS_EXIT = 1` (found proof)
     - `SOS_EMPTY_EXIT = 2` (ran out of clauses)
     - `MAX_GIVEN_EXIT = 3` (hit given clause limit)
     - `MAX_KEPT_EXIT = 4` (hit kept clause limit)
     - `MAX_SECONDS_EXIT = 5` (timeout)
     - `MAX_GENERATED_EXIT = 6` (generation limit)
     - `FATAL_EXIT = 7` (error)

2. **Search Loop Structure** (lines 160-400 of given_clause.py)
   ```python
   while not finished:
       1. select_given(sos, given_count) → (clause, selection_type)
       2. Move clause to usable, index it
       3. _given_infer(given) → generate inferences
       4. Process new clauses: simplify, check tautology, index
       5. Back-subsumption, back-demodulation (limbo)
       6. Check proof, update statistics
   ```

   **Key invariants to preserve:**
   - Given clause always selected from SOS (Set Of Support)
   - Age-based and weight-based selection ratio maintained
   - Inference rules applied in correct order
   - Backward subsumption/demod applied to new clauses
   - Proof detected via empty clause
   - Search fails when SOS empty

3. **Clause Processing Order**
   - Initial clauses processed first (marks clauses with `initial=True`)
   - Units prioritized in given selection (existing C behavior)
   - Weight-based then age-based selection in ratio cycle
   - No reordering of inference results

4. **Justification Tracking**
   - Every clause has `justification: tuple[Justification, ...]`
   - Links to parent clauses, inference rule, positions
   - Used to reconstruct proofs (enabled/disabled by SearchOptions)
   - Compatible with C `just.c` structures (ParaJust has from_id, into_id, positions)

### 1.2 Clause/Term/Literal Data Structures

These structures are **intentionally C-compatible** for cross-validation testing.

**Clause** (`pyladr/core/clause.py`):
```python
@dataclass
class Clause:
    literals: tuple[Literal, ...]  # Immutable disjunction
    weight: float                   # default_clause_weight()
    justification: tuple[Justification, ...]
    id: int                         # Assigned during search
    initial: bool                   # Initial clause flag
    normal_vars: bool               # Normalization marker
```

**Term** (`pyladr/core/term.py`) - Internal encoding:
- `private_symbol >= 0` → Variable (index)
- `private_symbol < 0` → Rigid symbol (symnum = -private_symbol)
- Arity determines constant vs complex
- Compatible with C term encoding

**Literal** (`pyladr/core/clause.py`):
```python
@dataclass
class Literal:
    sign: bool      # True=positive, False=negative
    atom: Term      # Atomic formula
```

**What Must NOT Be Modified:**
- Term internal encoding (private_symbol)
- Clause ID assignment during search
- Literal sign representation
- Justification format

**What CAN Be Enhanced:**
- Add optional ML-related metadata (via attributes)
- Extend Graph construction without modifying core
- Cache computed features (ground, symbol_count)

### 1.3 Cross-Validation Against C Prover9

The test suite verifies PyLADR against reference C binary.

**Test Infrastructure** (`tests/cross_validation/`):
- `c_runner.py` - Invokes C binary at `reference-prover9/bin/prover9`
- `test_c_runner.py` - Tests C invocation
- `test_search_equivalence.py` - Compares Python vs C results
- `test_c_vs_python_comprehensive.py` - Comprehensive validation
- `comparator.py` - Result comparison utilities

**Comparison Checks:**
```python
ProverResult:
  - exit_code (must match)
  - theorem_proved (must match)
  - clauses_given (within tolerance)
  - clauses_kept (within tolerance)
  - clauses_generated (within tolerance)
  - proof_length (if proved, must match)
  - max_weight (statistics)
```

**Critical:** Any change to search algorithm must not cause exit code divergence on standard problems.

### 1.4 Selection Mechanism Hook Points

The hierarchical GNN must integrate at the **selection layer**, not the search core.

**Current Hook:** `GivenClauseSearch.select_given()` (lines 346-400)
```python
def _make_inferences(self, ...):
    # 1. Select given clause
    given_clause, selection_type = self.selection.select_given(
        self.sos, self.given_count
    )

    # 2. Rest of search continues unchanged
    self.usable.append(given_clause)
    self._index_clause(given_clause)
    self._given_infer(given_clause)
    # ...
```

**Guarantee:** Selection returns `(Clause, SelectionType)` tuple. Search doesn't care how the selection is computed (weight, ML embedding, hierarchical GNN, etc.).

**Implementation Pattern for Hierarchical GNN:**
```python
class HierarchicalGNNSelection(EmbeddingEnhancedSelection):
    """Extends existing ML selection with hierarchical embeddings"""

    def select_given(self, sos, given_count):
        # 1. Use hierarchical_embedding_provider for embeddings
        # 2. Compute hierarchical scores
        # 3. Fall back to parent class if hierarchical unavailable
        # 4. Return (clause, selection_type) as before

        if self.hierarchical_provider is None:
            return super().select_given(sos, given_count)

        # Use hierarchical
        embeddings = self.hierarchical_provider.get_embeddings_batch(sos)
        scores = self._compute_hierarchical_scores(embeddings, sos)
        # ... select and return
```

---

## 2. EXISTING ML INFRASTRUCTURE TO INTEGRATE WITH

### 2.1 EmbeddingProvider Protocol

The `EmbeddingProvider` is a Protocol (structural typing) that enables multiple implementations:

```python
@runtime_checkable
class EmbeddingProvider(Protocol):
    """Any object implementing this protocol works with ml_selection"""

    def get_embeddings_batch(self, clauses: list[Clause]) -> dict[int, np.ndarray]:
        """Returns ID → embedding dict, None values for failures"""
        ...

    def get_embedding(self, clause: Clause) -> Optional[np.ndarray]:
        """Single clause embedding"""
        ...
```

**Existing implementations:**
1. `GNNEmbeddingProvider` - Production GNN with caching
2. `NoOpEmbeddingProvider` - Graceful fallback when torch unavailable
3. `GNNClauseEncoder` - Adapter for training

**Hierarchical GNN Should Provide:**
1. `HierarchicalGNNEmbeddingProvider` - Heterogeneous graph with hierarchical pooling
   - Satisfies `EmbeddingProvider` protocol
   - Returns higher-dimensional embeddings (hierarchical concatenation)
   - Maintains thread-safety with RWLock for model updates
   - Supports same caching strategy as flat GNN

2. Backward compat: Can return same protocol as current provider
   - Or return extended embeddings (e.g., flat concat of hierarchy layers)
   - Selection layer only cares about dict[int, np.ndarray]

### 2.2 Current EmbeddingProvider Architecture

**Location:** `pyladr/ml/embedding_provider.py` (630 lines)

**Key Properties:**
- **Thread-safe hot-swap:** RWLock for concurrent embeddings + updates
- **LRU caching:** `EmbeddingCache(max_entries=100K)` with automatic eviction
- **Device-agnostic:** CPU/GPU/MPS support via torch
- **Graceful degradation:** Returns NoOpEmbeddingProvider if torch unavailable
- **Batch processing:** Efficient `get_embeddings_batch()` for many clauses

**Configuration:**
```python
@dataclass
class EmbeddingProviderConfig:
    model_path: str                    # Path to saved model
    device: str = "auto"               # "cpu", "cuda", "mps"
    cache_max_entries: int = 100_000   # LRU cache size
    graph_construction_config: GraphConstructionConfig = ...
```

**Hierarchical GNN Opportunity:**
- Reuse caching infrastructure (same EmbeddingCache)
- Reuse thread-safety patterns (RWLock for model swap)
- Reuse batch processing
- Provide hierarchical_provider.config with extended graph config

### 2.3 ML Selection Integration

**Location:** `pyladr/search/ml_selection.py` (441 lines)

The `EmbeddingEnhancedSelection` class blends traditional weight-based selection with ML scoring:

```python
class EmbeddingEnhancedSelection(GivenSelection):
    def __init__(self, embedding_provider, config: MLSelectionConfig):
        self.embedding_provider = embedding_provider
        self.ml_weight = config.ml_weight  # Blend ratio (0-1)
        self.diversity_weight = 0.5        # How much to weight diversity
        self.proof_potential_weight = 0.5  # How much to weight proof signal
        self.diversity_tracker = DiversityTracker(window=20)

    def select_given(self, sos, given_count):
        # 1. Decide: use age-based or weight-based?
        # 2. If weight-based: compute ml_score
        # 3. Blend: score = (1-ml_weight)×traditional + ml_weight×ml_score
        # 4. Select highest-scoring clause
```

**Selection Score Components:**
1. **Diversity:** Cosine distance to recent 20 given clauses
   - Pushes exploration away from recently selected clauses
   - Prevents local optima

2. **Proof Potential:** Inverse sigmoid on embedding norm
   - Clauses with larger embeddings (unusual structure) higher potential
   - Heuristic signal for interesting clauses

3. **Weight Exploration:** Progressive increase with cycle count
   - Gradually explore heavier clauses over time
   - Prevents weight-only starvation

**Hierarchical GNN Integration Point:**
- Hierarchical embeddings replace flat GNN embeddings
- All scoring components (diversity, proof_potential) use hierarchical embeddings
- Configuration: add `use_hierarchical: bool` to `MLSelectionConfig`

**Critical:** Must not break existing selection logic
```python
@dataclass
class MLSelectionConfig:
    enabled: bool = True
    ml_weight: float = 0.3                    # Blend ratio
    diversity_weight: float = 0.5
    proof_potential_weight: float = 0.5
    diversity_window: int = 20
    use_hierarchical: bool = False            # NEW: opt-in hierarchical
    hierarchical_config: Optional[HierarchicalConfig] = None
```

### 2.4 Online Learning System

**Location:** `pyladr/ml/online_learning.py` (870 lines)

Continuously updates embeddings during search based on inference outcomes.

**Key Components:**
1. **OnlineLearningManager** - Tracks outcomes, triggers updates
2. **ExperienceBuffer** - Stores inference outcomes (KEPT, SUBSUMED, PROOF, etc.)
3. **ModelVersion** - Version tracking with productivity rates
4. **ABTestTracker** - Sliding window comparison

**Update Trigger:**
```python
manager.should_update():
    # Returns True every 200 examples or after min_examples reached

manager.update():
    1. sample_contrastive_batch() from experience buffer
    2. Encode clauses with gradients
    3. Compute contrastive loss (positive/negative pairs from proof traces)
    4. Backward + optimizer.step()
    5. Apply EMA (exponential moving average)
    6. Check for degradation (rollback if needed)
    7. Hot-swap model weights via embedding_provider
    8. Invalidate cache
```

**Hierarchical GNN Integration:**
- Update can handle hierarchical model parameters
- Contrastive loss still works (any embedding dimension)
- EMA smoothing applies to all parameters
- Model versioning/rollback unchanged

**Key Pattern:** Online learning is **decoupled** from selection via interface:
- Selection gets embeddings from `embedding_provider`
- Online learning updates `embedding_provider.model`
- Cache automatically invalidated on update

### 2.5 Graph Construction

**Location:** `pyladr/ml/graph/clause_graph.py` (413 lines)

Builds heterogeneous graph from clauses for GNN input.

**Node Types:**
- CLAUSE, LITERAL, TERM, SYMBOL, VARIABLE

**Edge Types:**
- CONTAINS_LITERAL (clause→literal)
- HAS_ATOM (literal→term)
- HAS_ARG (term→term)
- SYMBOL_OF (term→symbol)
- VAR_OCCURRENCE (term→variable)
- SHARED_VARIABLE (variable↔variable)

**Features Extracted:**
- Clause nodes: 7 features (size, num_literals, num_variables, ...)
- Literal nodes: 3 features (sign, polarity, position)
- Term nodes: 8 features (type, arity, depth, is_ground, ...)
- Symbol nodes: 6 features (arity, weight, ...)
- Variable nodes: 1 feature (occurrence count)

**Hierarchical GNN Opportunity:**
- Current `ClauseGraphBuilder` creates single heterogeneous graph
- **Hierarchical Enhancement:** Create multi-level graph structure
  - Level 0: Current flat structure (term-level detail)
  - Level 1: Term → Literal aggregation
  - Level 2: Literal → Clause aggregation
  - Inter-level edges for skip connections

- Reuse existing node types and features
- Add hierarchical edge types (UP_TO_PARENT, DOWN_TO_CHILDREN)
- Message passing at each hierarchy level

**Backward Compat:** Flat GNN continues to work unchanged

---

## 3. DATA FLOW: How Hierarchical GNN Plugs In

### 3.1 Training Data Generation Flow

```
Search -> Outcomes -> Online Learning -> Model Update

1. During search:
   GivenClauseSearch.run()
     ├─ For each inference:
     │   └─ manager.record_outcome(InferenceOutcome(
     │       clause_id, inference_rule, outcome_type
     │     ))
     │       └─ experience_buffer.add(outcome)
     │
     └─ Every N outcomes:
         └─ manager.update()
             ├─ buffer.sample_contrastive_batch()
             │   └─ Extract positive pairs (proof outcomes)
             │   └─ Extract negative pairs (subsumed outcomes)
             │
             ├─ hierarchical_encoder.encode_clauses()
             │   └─ Produces hierarchical embeddings with gradients
             │
             ├─ loss = contrastive_loss(anchor, pos, neg)
             │   └─ Uses any embedding dimension
             │
             ├─ loss.backward()
             ├─ optimizer.step()
             ├─ _apply_ema()
             │
             └─ embedding_provider.swap_weights(state_dict)
                 └─ cache.on_model_update()
                     └─ Invalidate cached embeddings
```

**Key Invariant:** Online learning doesn't know if embeddings are flat or hierarchical - only cares about encoder interface.

### 3.2 Inference (Selection) Data Flow

```
Selection -> Hierarchy Levels -> Aggregation -> Score

1. During select_given():
   EmbeddingEnhancedSelection.select_given(sos)
     │
     ├─ batch = list(sos)[:batch_size]
     │
     ├─ embeddings = hierarchical_provider.get_embeddings_batch(batch)
     │   └─ For each clause:
     │       ├─ Build hierarchical graph
     │       ├─ Apply message passing at each level
     │       │   Level 0 (term-level): Compute term embeddings
     │       │   Level 1 (literal-level): Aggregate terms → literals
     │       │   Level 2 (clause-level): Aggregate literals → clause
     │       │
     │       ├─ Collect outputs from all levels
     │       └─ Concatenate into hierarchical embedding
     │           OR return clause-level only (backward compat)
     │
     ├─ For each embedding:
     │   ├─ diversity_score = cosine_dist_to_recent()
     │   ├─ proof_score = proof_potential(embedding)
     │   ├─ final_score = blend(diversity, proof)
     │
     └─ Select clause with max score
```

### 3.3 Backward Compatibility Guarantee

**Scenario 1: Hierarchical GNN not available**
```
hierarchical_provider = None
selection = EmbeddingEnhancedSelection(
    embedding_provider=flat_gnn_provider,  # Existing GNN
    hierarchical_provider=None,
    config=MLSelectionConfig(use_hierarchical=False)
)
# Result: Uses flat GNN, identical behavior to before
```

**Scenario 2: Hierarchical GNN available, disabled**
```
selection = EmbeddingEnhancedSelection(
    embedding_provider=flat_gnn_provider,
    hierarchical_provider=hierarchical_provider,
    config=MLSelectionConfig(use_hierarchical=False)
)
# Result: Loads flat GNN only, hierarchical provider unused
```

**Scenario 3: Hierarchical GNN available, enabled**
```
selection = EmbeddingEnhancedSelection(
    embedding_provider=None,  # Hierarchical provider replaces
    hierarchical_provider=hierarchical_provider,
    config=MLSelectionConfig(use_hierarchical=True)
)
# Result: Uses hierarchical embeddings for scoring
# All other selection logic unchanged
```

---

## 4. INTEGRATION POINTS AND EXTENSION STRATEGY

### 4.1 Clause Selection Integration

**Primary Hook:** `GivenClauseSearch._make_inferences()` → `selection.select_given()`

**What Changes:**
- `MLSelectionConfig` gets optional `use_hierarchical` flag
- `EmbeddingEnhancedSelection` constructor accepts optional `hierarchical_provider`
- Selection logic unchanged; only embedding source changes

**Implementation Pattern:**
```python
class EmbeddingEnhancedSelection(GivenSelection):
    def __init__(self, embedding_provider=None,
                 hierarchical_provider=None,
                 config=None):
        # NEW: Support hierarchical provider
        self.embedding_provider = embedding_provider
        self.hierarchical_provider = hierarchical_provider
        self.use_hierarchical = config.use_hierarchical if config else False

    def _get_embeddings_batch(self, clauses):
        if self.use_hierarchical and self.hierarchical_provider:
            return self.hierarchical_provider.get_embeddings_batch(clauses)
        elif self.embedding_provider:
            return self.embedding_provider.get_embeddings_batch(clauses)
        else:
            return {}  # Graceful degradation

    def select_given(self, sos, given_count):
        # Existing logic, but uses _get_embeddings_batch()
        ...
```

### 4.2 Graph Construction Extension

**Current:** `ClauseGraphBuilder` in `pyladr/ml/graph/clause_graph.py`

**Strategy:** Create `HierarchicalGraphBuilder` alongside existing builder

**Pattern:**
```python
# Existing code unchanged
flat_builder = ClauseGraphBuilder(config=GraphConstructionConfig(...))
flat_graph = flat_builder.build_graph(clause)

# NEW: Hierarchical builder
hierarchical_builder = HierarchicalClauseGraphBuilder(
    config=HierarchicalGraphConfig(
        base_config=GraphConstructionConfig(...),
        include_level_0=True,   # Term-level detail
        include_level_1=True,   # Literal-level aggregation
        include_level_2=True,   # Clause-level aggregation
        skip_connections=True   # Inter-level edges
    )
)
hierarchical_graph = hierarchical_builder.build_graph(clause)
```

**Data Structure:** MultiLevelHeteroGraph
```python
@dataclass
class MultiLevelHeteroGraph:
    graphs: list[HeteroData]  # One per level
    level_edges: dict[str, EdgeIndex]  # Inter-level connections
    metadata: dict  # Level names, feature dims, etc.
```

### 4.3 Hierarchical GNN Architecture

**New Component:** `pyladr/ml/graph/hierarchical_gnn.py`

**Key Design:**
1. Extend or compose with existing `ClauseEncoder` (373 lines)
2. Add hierarchical message passing layers
3. Maintain same input protocol (HeteroData → embedding)
4. Support both flat and hierarchical modes

**Architecture Pattern:**
```python
class HierarchicalClauseEncoder(nn.Module):
    def __init__(self, config: HierarchicalGNNConfig):
        super().__init__()
        # Level-specific encoders
        self.level_encoders = nn.ModuleList([
            level_encoder for level_encoder in create_level_encoders(config)
        ])
        # Aggregation across levels
        self.aggregator = HierarchicalAggregator(config)
        # Output projection
        self.output_projection = nn.Linear(config.output_dim, config.embedding_dim)

    def forward(self, hierarchical_graph: MultiLevelHeteroGraph):
        # Level 0 (term-level)
        z0 = self.level_encoders[0](hierarchical_graph.graphs[0])

        # Level 1 (with skip connection to level 0)
        z1 = self.level_encoders[1](
            hierarchical_graph.graphs[1],
            skip_connection=hierarchical_graph.level_edges['to_level_1']
        )

        # Level 2 (clause-level)
        z2 = self.level_encoders[2](
            hierarchical_graph.graphs[2],
            skip_connection=hierarchical_graph.level_edges['to_level_2']
        )

        # Aggregate across levels
        z_agg = self.aggregator([z0, z1, z2])

        # Output
        z_out = self.output_projection(z_agg)
        return z_out
```

### 4.4 Provider Interface

**New Component:** `pyladr/ml/embedding_provider_hierarchical.py`

```python
class HierarchicalGNNEmbeddingProvider(EmbeddingProvider):
    """Provides hierarchical embeddings while maintaining interface compatibility"""

    def __init__(self, config: EmbeddingProviderConfig):
        self.config = config
        self.hierarchical_builder = HierarchicalGraphBuilder(config.graph_config)
        self.encoder = HierarchicalClauseEncoder(config.hierarchical_gnn_config)
        self.cache = EmbeddingCache(max_entries=config.cache_max_entries)
        self.model_lock = RWLock()

    def get_embeddings_batch(self, clauses: list[Clause]) -> dict[int, np.ndarray]:
        """Same interface as GNNEmbeddingProvider"""
        result = {}
        for clause in clauses:
            if cache_hit := self.cache.get(clause.id):
                result[clause.id] = cache_hit
            else:
                hierarchical_graph = self.hierarchical_builder.build_graph(clause)
                with self.model_lock.read_lock():
                    embedding = self.encoder.forward(hierarchical_graph)
                embedding = embedding.detach().cpu().numpy()
                self.cache.put(clause.id, embedding)
                result[clause.id] = embedding
        return result

    def swap_weights(self, state_dict):
        """Hot-swap for online learning"""
        with self.model_lock.write_lock():
            self.encoder.load_state_dict(state_dict)
        self.cache.clear()  # Invalidate on update
```

### 4.5 Online Learning Compatibility

**No changes needed to `OnlineLearningManager`** - it's architecture-agnostic.

**Instead, create new encoder:**
```python
class HierarchicalClauseEncoderAdapter(ClauseEncoder):
    """Implements ClauseEncoder protocol for hierarchical GNN"""

    def __init__(self, provider: HierarchicalGNNEmbeddingProvider):
        self.provider = provider
        self.hierarchical_builder = provider.hierarchical_builder

    def encode_clause(self, clause: Clause) -> torch.Tensor:
        """Encode with gradients enabled (for training)"""
        hierarchical_graph = self.hierarchical_builder.build_graph(clause)
        # Convert to torch tensors with gradient tracking
        return self.provider.encoder.forward(hierarchical_graph)
```

**Online Learning Usage:**
```python
encoder = HierarchicalClauseEncoderAdapter(hierarchical_provider)
manager = OnlineLearningManager(
    encoder=encoder,
    embedding_provider=hierarchical_provider,
    config=OnlineLearningConfig(...)
)
# Result: OnlineLearningManager updates hierarchical GNN seamlessly
```

---

## 5. TESTING STRATEGY FOR COMPATIBILITY

### 5.1 Backward Compatibility Tests

**New Test File:** `tests/unit/test_hierarchical_gnn_compat.py`

```python
def test_hierarchical_disabled_matches_flat():
    """When use_hierarchical=False, results identical to flat GNN"""
    flat_config = MLSelectionConfig(use_hierarchical=False)
    hierarchical_config = MLSelectionConfig(use_hierarchical=False)

    flat_selection = EmbeddingEnhancedSelection(
        embedding_provider=flat_provider,
        config=flat_config
    )

    hierarchical_selection = EmbeddingEnhancedSelection(
        embedding_provider=flat_provider,
        hierarchical_provider=hierarchical_provider,  # Provided but disabled
        config=hierarchical_config
    )

    # Same SOS, same given_count
    result_flat = flat_selection.select_given(sos, given_count)
    result_hierarchical = hierarchical_selection.select_given(sos, given_count)

    assert result_flat[0].id == result_hierarchical[0].id
    assert result_flat[1] == result_hierarchical[1]  # Same selection type

def test_graceful_degradation_no_hierarchical():
    """When hierarchical provider missing, falls back to flat"""
    selection = EmbeddingEnhancedSelection(
        embedding_provider=flat_provider,
        hierarchical_provider=None,
        config=MLSelectionConfig(use_hierarchical=True)  # Enabled but unavailable
    )
    # Should use flat provider and still work
    result = selection.select_given(sos, given_count)
    assert result is not None
    assert isinstance(result, tuple)

def test_no_breaking_changes_to_search():
    """Adding hierarchical GNN doesn't change search behavior"""
    options = SearchOptions(
        max_given=100,
        max_kept=1000,
        ml_weight=0.0  # ML disabled
    )
    result_flat = search_with_flat_gnn(problem, options)
    result_no_ml = search_with_hierarchical_disabled(problem, options)

    # Same exit code
    assert result_flat.exit_code == result_no_ml.exit_code
    # Same or very close stats
    assert abs(result_flat.clauses_given - result_no_ml.clauses_given) <= 1
```

### 5.2 C Prover9 Equivalence Tests

**Extend existing:** `tests/cross_validation/test_search_equivalence.py`

```python
def test_hierarchical_c_equivalence_disabled():
    """Hierarchical GNN disabled = same results as C Prover9"""
    for problem in standard_problems:
        options = SearchOptions(
            ml_weight=0.0,  # ML disabled
            use_hierarchical=False
        )
        py_result = pyladr_search(problem, options)
        c_result = c_prover9_search(problem)

        assert py_result.exit_code == c_result.exit_code
        assert py_result.theorem_proved == c_result.theorem_proved

def test_hierarchical_doesnt_change_non_ml_search():
    """Without ML enabled, hierarchical presence is irrelevant"""
    for problem in standard_problems:
        options_flat = SearchOptions(
            ml_weight=0.0,
            embedding_provider=flat_provider
        )
        options_hierarchical = SearchOptions(
            ml_weight=0.0,
            hierarchical_provider=hierarchical_provider
        )
        result_flat = pyladr_search(problem, options_flat)
        result_hierarchical = pyladr_search(problem, options_hierarchical)

        # Identical search behavior
        assert result_flat.exit_code == result_hierarchical.exit_code
        assert result_flat.clauses_given == result_hierarchical.clauses_given
```

### 5.3 Integration Tests

**New Test File:** `tests/integration/test_hierarchical_integration.py`

```python
def test_hierarchical_with_online_learning():
    """Hierarchical GNN works with online learning"""
    options = SearchOptions(
        ml_weight=0.5,
        use_hierarchical=True,
        online_learning=True
    )
    result = search_with_options(problem, options)

    # Completes successfully
    assert result.exit_code in [1, 2, 3, 4, 5, 6]  # Valid codes
    # Online learning recorded updates
    assert len(manager.model_versions) > 1

def test_hierarchical_with_inference_guidance():
    """Hierarchical GNN compatible with inference guidance"""
    options = SearchOptions(
        use_hierarchical=True,
        inference_guidance_enabled=True,
        inference_guidance_config=InferenceGuidanceConfig(...)
    )
    result = search_with_options(problem, options)
    assert result.exit_code in [1, 2, 3, 4, 5, 6]

def test_hierarchical_performance():
    """Hierarchical GNN doesn't degrade performance"""
    times = {}
    for use_hier in [False, True]:
        start = time.time()
        search_with_options(problem, SearchOptions(
            use_hierarchical=use_hier,
            ml_weight=0.5
        ))
        times[use_hier] = time.time() - start

    # Hierarchical ~10-20% slower per embedding is acceptable
    # (more computation, but potentially better selection)
    # Should not timeout or hang
    assert times[True] < timeout_limit
```

### 5.4 Embedding Quality Tests

**New Test File:** `tests/unit/test_hierarchical_embeddings.py`

```python
def test_hierarchical_produces_valid_embeddings():
    """Hierarchical embeddings are valid tensors"""
    embeddings = hierarchical_provider.get_embeddings_batch(clauses)

    for clause_id, embedding in embeddings.items():
        assert embedding is not None
        assert embedding.shape == (embedding_dim,)
        assert not np.isnan(embedding).any()
        assert not np.isinf(embedding).any()

def test_hierarchical_diversity_scoring():
    """Hierarchical embeddings produce sensible diversity scores"""
    embeddings = hierarchical_provider.get_embeddings_batch(clauses)

    # Similar clauses should have high cosine similarity
    sim = cosine_similarity(embeddings[0], embeddings[1])
    assert 0 <= sim <= 1

def test_hierarchical_discriminative():
    """Different clauses have different embeddings"""
    flat_embeddings = flat_provider.get_embeddings_batch(clauses)
    hier_embeddings = hierarchical_provider.get_embeddings_batch(clauses)

    # Not all embeddings should be identical
    assert len(set(map(tuple, hier_embeddings.values()))) > 1
```

---

## 6. CRITICAL DO's AND DON'Ts

### 6.1 DO's

✅ **DO:**
- Create new `HierarchicalGNNEmbeddingProvider` alongside existing `GNNEmbeddingProvider`
- Add `use_hierarchical` configuration flag (disabled by default)
- Extend `ClauseGraphBuilder` with `HierarchicalGraphBuilder` (don't modify existing)
- Use same `EmbeddingProvider` protocol interface
- Maintain thread-safety with RWLock for model hot-swapping
- Reuse existing caching infrastructure (`EmbeddingCache`)
- Run full cross-validation test suite against C Prover9
- Document all new configuration parameters
- Provide graceful degradation when hierarchical unavailable
- Test both with online learning enabled and disabled

### 6.2 DON'Ts

❌ **DON'T:**
- Modify `GivenClauseSearch` core loop (lines 160-400)
- Change exit codes or search termination logic
- Alter clause ID assignment or justification format
- Modify `Term` internal encoding (private_symbol)
- Break `EmbeddingProvider` protocol compatibility
- Add required dependencies (torch_geometric) to core (optional only)
- Deprecate or remove existing `GNNEmbeddingProvider` (backward compat)
- Change default behavior when hierarchical disabled
- Modify online learning manager's core logic
- Assume hierarchical provider always available

### 6.3 Integration Pattern

✅ **Correct Pattern:**
```python
# User code
provider = HierarchicalGNNEmbeddingProvider(config=config)
selection = EmbeddingEnhancedSelection(
    hierarchical_provider=provider,
    config=MLSelectionConfig(use_hierarchical=True)
)
search = GivenClauseSearch(
    usable=[],
    sos=clauses,
    options=SearchOptions(ml_weight=0.5),
    selection=selection
)
result = search.run()
```

❌ **Wrong Pattern:**
```python
# DON'T modify search directly
search = GivenClauseSearch(usable=[], sos=clauses)
search.use_hierarchical = True  # DON'T do this
search.hierarchical_gnn = provider  # DON'T do this
```

---

## 7. IMPLEMENTATION ROADMAP

### Phase 1: Graph Construction (Foundational)
1. Create `HierarchicalGraphBuilder` alongside `ClauseGraphBuilder`
2. Build multi-level graph data structure
3. Extend node/edge types for hierarchy
4. **Test:** Verify hierarchical graphs valid, contain all needed info
5. **Compat:** Ensure flat graphs unchanged

### Phase 2: Hierarchical GNN Model
1. Implement `HierarchicalClauseEncoder`
2. Per-level message passing
3. Inter-level skip connections
4. Aggregation across levels
5. **Test:** Unit tests for architecture
6. **Compat:** Same input/output protocol as flat GNN

### Phase 3: Provider Integration
1. Create `HierarchicalGNNEmbeddingProvider`
2. Implement `get_embeddings_batch()` method
3. Integrate caching (reuse `EmbeddingCache`)
4. Thread-safety (reuse RWLock pattern)
5. **Test:** Provider tests, cache invalidation
6. **Compat:** Graceful degradation tests

### Phase 4: Selection Integration
1. Extend `MLSelectionConfig` with `use_hierarchical` flag
2. Modify `EmbeddingEnhancedSelection` to support provider choice
3. Implement fallback logic
4. **Test:** Selection unchanged when hierarchical disabled
5. **Compat:** C Prover9 equivalence tests

### Phase 5: Online Learning Support
1. Create `HierarchicalClauseEncoderAdapter`
2. Verify online learning updates work
3. Model versioning/rollback unchanged
4. **Test:** Online learning with hierarchical GNN
5. **Compat:** Cross-validation tests

### Phase 6: Comprehensive Testing
1. Run full test suite (C equivalence, integration, etc.)
2. Benchmark performance
3. Verify no degradation in default mode
4. **Compat:** All cross-validation tests pass

---

## 8. CONCLUSION

The hierarchical GNN system can be successfully integrated into PyLADR while preserving strict C Prover9 compatibility through:

1. **Non-invasive architecture:** New components extend existing infrastructure without modifying search core
2. **Protocol-based design:** `EmbeddingProvider` interface enables drop-in replacement
3. **Opt-in feature:** Disabled by default, enabled only when explicitly configured
4. **Graceful degradation:** Works when hierarchical unavailable, seamlessly falls back to flat GNN
5. **Thread-safe hot-swapping:** Model updates don't interrupt search
6. **Comprehensive testing:** Cross-validation against C Prover9 validates equivalence

**Key Guarantee:** `GivenClauseSearch` search loop, exit codes, and clause structures **remain unchanged**. The hierarchical GNN is purely an enhancement to clause selection scoring, not the search algorithm itself.
