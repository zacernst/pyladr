# Hierarchical GNN Integration Plan

## Overview

This document outlines the step-by-step integration plan for deploying the hierarchical GNN architecture in PyLADR. The plan ensures **zero breaking changes** to existing functionality while enabling gradual rollout of hierarchical features.

## Integration Phases

### Phase 1: Foundation (Week 1-2)
**Goal**: Establish hierarchical infrastructure with full backward compatibility

#### 1.1 Core Architecture Implementation
- [ ] Implement `HierarchyLevel` enum and basic structures
- [ ] Create `HierarchicalClauseGNN` with base GNN embedding
- [ ] Implement `HierarchicalEmbeddingProvider` with fallback mechanisms
- [ ] Add factory functions with automatic fallback

**Deliverables**:
- `pyladr/ml/hierarchical/architecture.py`
- `pyladr/ml/hierarchical/provider.py`
- `pyladr/ml/hierarchical/__init__.py`
- `tests/unit/test_hierarchical_architecture.py`

**Testing Requirements**:
- [ ] All existing tests pass without modification
- [ ] Backward compatibility test suite passes
- [ ] Factory functions correctly fall back to base provider
- [ ] No performance regression when hierarchical features disabled

#### 1.2 Configuration System
- [ ] Extend existing configuration classes
- [ ] Add feature flags for gradual rollout
- [ ] Implement configuration validation and defaults

**Configuration Example**:
```python
# Existing code continues to work unchanged
provider = create_embedding_provider(symbol_table, config)

# New hierarchical features are opt-in
hierarchical_config = HierarchicalEmbeddingProviderConfig(
    use_hierarchical_features=True,  # Explicit opt-in required
    enable_goal_directed_selection=False,  # Start with basic features
)
provider = create_hierarchical_embedding_provider(config=hierarchical_config)
```

### Phase 2: Basic Hierarchical Features (Week 3-4)
**Goal**: Implement and test basic hierarchical message passing

#### 2.1 Message Passing Implementation
- [ ] Implement `IntraLevelMP` for within-level message passing
- [ ] Implement `InterLevelMP` for adjacent-level communication
- [ ] Create hierarchical graph construction utilities
- [ ] Add level-specific feature extraction

**Implementation Files**:
- `pyladr/ml/hierarchical/message_passing.py`
- `pyladr/ml/hierarchical/graph_builder.py`

#### 2.2 Integration Testing
- [ ] Test hierarchical message passing correctness
- [ ] Validate embedding quality improvements
- [ ] Performance benchmarking vs baseline
- [ ] Memory usage validation

**Testing Approach**:
```python
# Test dual functionality - both modes should work
def test_hierarchical_vs_base_functionality():
    # Same input, different processing modes
    data = create_test_graph()

    # Base mode (backward compatible)
    base_embedding = model.forward(data, use_hierarchical=False)

    # Hierarchical mode (new features)
    hier_embedding = model.forward(data, use_hierarchical=True)

    # Both should produce valid embeddings
    assert base_embedding.shape == hier_embedding.shape
    assert torch.allclose(base_embedding, baseline_expected, atol=1e-6)
```

### Phase 3: Cross-Level Attention (Week 5-6)
**Goal**: Add cross-level attention for non-adjacent level communication

#### 3.1 Cross-Level Attention Implementation
- [ ] Implement `CrossLevelAttention` module
- [ ] Add attention weight visualization tools
- [ ] Create cross-level index mapping utilities
- [ ] Optimize attention computation for large graphs

#### 3.2 Performance Optimization
- [ ] Implement attention sparsity optimizations
- [ ] Add gradient checkpointing for memory efficiency
- [ ] Create attention head pruning mechanisms
- [ ] Benchmark cross-level attention overhead

**Performance Targets**:
- Cross-level attention adds <50% compute overhead
- Memory usage increases <2x vs base model
- Attention weights should be interpretable and sparse

### Phase 4: Goal-Directed Features (Week 7-8)
**Goal**: Implement goal-directed selection and distance computation

#### 4.1 Goal Encoder Implementation
- [ ] Implement `GoalEncoder` for conjecture/context encoding
- [ ] Create `GoalDirectedAttention` mechanism
- [ ] Add `DistanceComputer` with multiple metrics
- [ ] Implement goal context caching

#### 4.2 Selection Integration
- [ ] Extend `MLSelection` with goal-directed scoring
- [ ] Add goal context management to search state
- [ ] Implement distance-based clause filtering
- [ ] Create goal recomputation scheduling

**API Design**:
```python
# Goal-directed selection seamlessly integrates
selection = EmbeddingEnhancedSelection(
    embedding_provider=hierarchical_provider,
    ml_config=MLSelectionConfig(
        enabled=True,
        ml_weight=0.3,
        goal_directed=True,  # New feature flag
        goal_weight=0.2,     # Blend with traditional scoring
    )
)

# Set proof goals
selection.set_goals(conjectures, context_clauses)

# Selection automatically uses goal-directed scoring
given_clause = selection.select_given(sos, usable)
```

### Phase 5: Incremental Updates (Week 9-10)
**Goal**: Add real-time embedding updates during search

#### 5.1 Incremental Update System
- [ ] Implement `IncrementalContext` for state tracking
- [ ] Create `StructuralChangeDetector` for dependency analysis
- [ ] Add `UpdateScheduler` for efficient batch updates
- [ ] Implement staleness scoring and cache management

#### 5.2 Search Integration
- [ ] Hook incremental updates into search loops
- [ ] Add update triggers for new clauses/inferences
- [ ] Implement selective update scheduling
- [ ] Create update performance monitoring

**Integration Points**:
```python
# Automatic incremental updates during search
class GivenClauseSearch:
    def __init__(self, embedding_provider):
        self.embedding_provider = embedding_provider

    def add_new_clause(self, clause):
        # Standard clause addition
        self.sos.add(clause)

        # Trigger incremental embedding update
        if hasattr(self.embedding_provider, 'incremental_update'):
            self.embedding_provider.incremental_update([clause])
```

### Phase 6: Command-Line Interface (Week 11)
**Goal**: Expose hierarchical features through CLI

#### 6.1 CLI Extension
- [ ] Add hierarchical GNN command-line flags
- [ ] Implement configuration file support for hierarchical settings
- [ ] Add verbose logging and debugging options
- [ ] Create feature compatibility warnings

**CLI Interface**:
```bash
# Backward compatible - no changes needed
pyladr --ml-enhanced input.in

# New hierarchical features (opt-in)
pyladr --ml-enhanced --hierarchical-gnn input.in

# Advanced hierarchical options
pyladr --ml-enhanced --hierarchical-gnn \
       --hierarchy-levels 5 \
       --cross-level-attention \
       --goal-directed \
       --incremental-updates \
       input.in

# Configuration file support
pyladr --config hierarchical_config.json input.in
```

#### 6.2 Documentation and Examples
- [ ] Update user documentation with hierarchical features
- [ ] Create example configuration files
- [ ] Add performance tuning guides
- [ ] Create troubleshooting documentation

### Phase 7: Production Validation (Week 12)
**Goal**: Validate production readiness and performance

#### 7.1 Comprehensive Testing
- [ ] Run full TPTP benchmark suite
- [ ] Cross-validation against reference Prover9
- [ ] Performance regression testing
- [ ] Memory leak and stability testing

#### 7.2 Production Deployment
- [ ] Create deployment scripts
- [ ] Add monitoring and alerting
- [ ] Implement gradual rollout mechanisms
- [ ] Create rollback procedures

## Backward Compatibility Strategy

### 1. Protocol Preservation
All existing `EmbeddingProvider` methods remain unchanged:
```python
# These signatures NEVER change
def get_embedding(self, clause: Clause) -> List[float] | None
def get_embeddings_batch(self, clauses: List[Clause]) -> List[List[float] | None]
@property
def embedding_dim(self) -> int
```

### 2. Configuration Compatibility
Existing configuration code continues to work:
```python
# Old code - still works
config = EmbeddingProviderConfig(model_path="model.pt")
provider = create_embedding_provider(config=config)

# New code - opt-in to hierarchical features
hier_config = HierarchicalEmbeddingProviderConfig(
    base_config=config,
    use_hierarchical_features=True
)
provider = create_hierarchical_embedding_provider(config=hier_config)
```

### 3. Factory Pattern
Factory functions provide seamless fallback:
```python
def create_embedding_provider(config=None):
    # Try hierarchical first if requested
    if isinstance(config, HierarchicalEmbeddingProviderConfig):
        return create_hierarchical_embedding_provider(config)

    # Fall back to standard provider
    return GNNEmbeddingProvider.create(config)
```

### 4. Feature Flags
All hierarchical features are disabled by default:
```python
@dataclass
class HierarchicalEmbeddingProviderConfig:
    use_hierarchical_features: bool = False  # Default: disabled
    enable_goal_directed_selection: bool = False
    enable_incremental_updates: bool = False
    cross_level_attention: bool = False
```

## Testing Strategy

### 1. Compatibility Test Suite
```python
class TestBackwardCompatibility:
    def test_existing_code_unchanged(self):
        # All existing demo scripts should work without modification

    def test_prover9_equivalence(self):
        # Results should be identical when hierarchical features disabled

    def test_performance_no_regression(self):
        # Performance should not degrade when features disabled
```

### 2. Feature-Specific Testing
Each feature has dedicated test coverage:
- Message passing correctness tests
- Cross-level attention validation
- Goal-directed selection effectiveness
- Incremental update efficiency
- Thread safety and concurrency

### 3. Integration Testing
Full workflow testing with hierarchical features enabled/disabled:
- Proof search completeness
- Memory usage patterns
- Performance characteristics
- Error handling and recovery

## Deployment Architecture

### 1. Modular Design
```
pyladr/ml/
├── embedding_provider.py          # Existing (unchanged)
├── graph/                         # Existing (unchanged)
└── hierarchical/                  # New module
    ├── __init__.py               # Public API
    ├── architecture.py           # Core GNN
    ├── provider.py               # Enhanced provider
    ├── message_passing.py        # MP components
    ├── goals.py                  # Goal-directed features
    ├── incremental.py            # Incremental updates
    └── factory.py                # Factory functions
```

### 2. Feature Flag System
```python
HIERARCHICAL_FEATURES = {
    "message_passing": False,       # Basic hierarchical processing
    "cross_level_attention": False, # Cross-level attention
    "goal_directed": False,         # Goal-directed selection
    "incremental_updates": False,   # Incremental updates
    "advanced_attention": False,    # Advanced attention mechanisms
}
```

### 3. Gradual Rollout
- Week 1-2: Internal testing with basic features
- Week 3-4: Alpha testing with selected power users
- Week 5-6: Beta testing with broader community
- Week 7-8: Production rollout with monitoring
- Week 9+: Full feature availability

## Risk Mitigation

### 1. Automatic Fallback
Every hierarchical feature has automatic fallback to base functionality:
```python
def hierarchical_forward(self, data, use_hierarchical=True):
    try:
        if use_hierarchical and self.config.use_hierarchical_features:
            return self._hierarchical_forward(data)
    except Exception:
        logger.warning("Hierarchical processing failed, falling back to base")

    return self.base_gnn.forward(data)
```

### 2. Performance Monitoring
- Continuous performance benchmarking
- Memory usage monitoring
- Error rate tracking
- User feedback collection

### 3. Rollback Procedures
- Feature flags can disable problematic features immediately
- Configuration-based rollback to previous versions
- Database migrations for configuration changes
- Monitoring alerts for performance degradation

## Success Metrics

### 1. Compatibility Metrics
- [ ] 100% of existing tests pass without modification
- [ ] Zero breaking changes in public APIs
- [ ] No performance regression when features disabled
- [ ] Successful integration with all existing demo scripts

### 2. Performance Metrics
- [ ] <3x computational overhead with all features enabled
- [ ] <2x memory usage overhead
- [ ] >20% improvement in proof search effectiveness
- [ ] <1% error rate in production deployment

### 3. Adoption Metrics
- [ ] Successful deployment in development environment
- [ ] Positive feedback from alpha/beta users
- [ ] Measurable improvement in proof success rates
- [ ] Documentation completeness and clarity

## Conclusion

This integration plan ensures that the hierarchical GNN architecture can be deployed safely and incrementally without breaking existing PyLADR functionality. The modular design, comprehensive testing strategy, and gradual rollout approach minimize risk while maximizing the benefits of the enhanced architecture.

Key principles:
- **Backward Compatibility**: Existing code continues to work unchanged
- **Incremental Deployment**: Features can be enabled progressively
- **Automatic Fallback**: System gracefully handles failures
- **Comprehensive Testing**: TDD approach ensures reliability
- **Performance Monitoring**: Continuous validation of system health

The architecture is designed to enhance PyLADR's capabilities while preserving the reliability and compatibility that users depend on.