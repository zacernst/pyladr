# Hierarchical GNN Architecture Specification

## Executive Summary

This document specifies a complete hierarchical Graph Neural Network (GNN) architecture for goal-directed clause selection in PyLADR. The system extends the existing heterogeneous GNN with a 5-level hierarchical message passing system that enables both bottom-up feature aggregation and top-down goal-oriented guidance.

**Key Features:**
- 5-level hierarchy: Symbol → Term → Literal → Clause → Proof
- Bidirectional message passing within and between levels
- Cross-level attention for direct non-adjacent communication
- Incremental embedding updates for real-time search evolution
- Goal-directed distance measures and selection integration
- Full backward compatibility with existing EmbeddingProvider protocol
- Test-driven development with comprehensive coverage

## Current Architecture Analysis

### Existing Components
- **HeterogeneousClauseGNN**: Current flat heterogeneous GNN with clause-level pooling
- **ClauseGraph**: Heterogeneous graph construction (5 node types, 6 edge types)
- **EmbeddingProvider**: Protocol-based interface with caching and thread-safety
- **MLSelection**: Blended ML+traditional scoring with fallback mechanisms

### Node Types (Existing)
```python
NodeType.SYMBOL     # Function/predicate symbols with metadata
NodeType.VARIABLE   # Logic variables with sharing relationships
NodeType.TERM       # Complex terms with argument structure
NodeType.LITERAL    # Signed atoms within clauses
NodeType.CLAUSE     # Complete logical clauses
```

### Edge Types (Existing)
```python
EdgeType.CONTAINS_LITERAL    # clause → literal
EdgeType.HAS_ATOM           # literal → term
EdgeType.HAS_ARG            # term → term (function arguments)
EdgeType.SYMBOL_OF          # term → symbol
EdgeType.VAR_OCCURRENCE     # variable → term
EdgeType.SHARED_VARIABLE    # variable ↔ variable (cross-literal)
```

## Hierarchical Architecture Design

### 1. Hierarchy Levels

```python
class HierarchyLevel(Enum):
    SYMBOL = 0      # Foundation: symbols and variables
    TERM = 1        # Compositional: terms and subterms
    LITERAL = 2     # Logical: literals and equations
    CLAUSE = 3      # Unit: complete clauses
    PROOF = 4       # Global: proof context and goals
```

### 2. Hierarchical Message Passing

#### 2.1 Intra-Level Message Passing
Each level maintains its own message passing graph:

```python
class IntraLevelMP(nn.Module):
    """Message passing within a single hierarchy level."""

    def __init__(self, level: HierarchyLevel, hidden_dim: int):
        self.level = level
        self.attention = MultiHeadAttention(hidden_dim)
        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim)
            for _ in range(3)  # 3 layers per level
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Self-attention within level
        attn_out = self.attention(x, x, x)

        # GNN message passing
        for layer in self.gnn_layers:
            x = layer(x + attn_out, edge_index)
            x = F.relu(x)

        return x
```

#### 2.2 Inter-Level Message Passing
Hierarchical message passing between adjacent levels:

```python
class InterLevelMP(nn.Module):
    """Bidirectional message passing between hierarchy levels."""

    def __init__(self, lower_dim: int, upper_dim: int):
        # Bottom-up aggregation (many-to-one)
        self.bottom_up = nn.Sequential(
            nn.Linear(lower_dim, upper_dim),
            nn.ReLU(),
            nn.LayerNorm(upper_dim)
        )

        # Top-down broadcast (one-to-many)
        self.top_down = nn.Sequential(
            nn.Linear(upper_dim, lower_dim),
            nn.ReLU(),
            nn.LayerNorm(lower_dim)
        )

        # Attention mechanisms
        self.bottom_up_attn = nn.MultiheadAttention(upper_dim, num_heads=8)
        self.top_down_attn = nn.MultiheadAttention(lower_dim, num_heads=8)

    def forward(self, lower_x: torch.Tensor, upper_x: torch.Tensor,
                bottom_up_index: torch.Tensor, top_down_index: torch.Tensor):
        # Bottom-up: aggregate lower level into upper level
        lower_projected = self.bottom_up(lower_x)
        aggregated = scatter_mean(lower_projected, bottom_up_index, dim=0)
        upper_updated, _ = self.bottom_up_attn(upper_x, aggregated, aggregated)

        # Top-down: broadcast upper level to lower level
        upper_projected = self.top_down(upper_x)
        broadcasted = upper_projected[top_down_index]
        lower_updated, _ = self.top_down_attn(lower_x, broadcasted, broadcasted)

        return lower_updated, upper_updated
```

#### 2.3 Cross-Level Attention
Direct communication between non-adjacent levels:

```python
class CrossLevelAttention(nn.Module):
    """Attention mechanism for direct cross-level communication."""

    def __init__(self, levels: List[HierarchyLevel], hidden_dim: int):
        self.levels = levels
        self.cross_attention = nn.ModuleDict()

        # Create attention modules for all level pairs
        for i, level_i in enumerate(levels):
            for j, level_j in enumerate(levels):
                if abs(i - j) > 1:  # Non-adjacent levels only
                    key = f"{level_i.name}_to_{level_j.name}"
                    self.cross_attention[key] = nn.MultiheadAttention(
                        hidden_dim, num_heads=4
                    )

    def forward(self, level_embeddings: Dict[HierarchyLevel, torch.Tensor],
                cross_indices: Dict[str, torch.Tensor]) -> Dict[HierarchyLevel, torch.Tensor]:
        updated = {}

        for level in self.levels:
            x = level_embeddings[level]
            cross_updates = []

            # Collect cross-level attention updates
            for other_level in self.levels:
                if abs(level.value - other_level.value) > 1:
                    key = f"{other_level.name}_to_{level.name}"
                    if key in self.cross_attention:
                        other_x = level_embeddings[other_level]
                        index_key = f"{other_level.name}_to_{level.name}_index"
                        if index_key in cross_indices:
                            indices = cross_indices[index_key]
                            attended = other_x[indices]
                            update, _ = self.cross_attention[key](x, attended, attended)
                            cross_updates.append(update)

            # Combine updates
            if cross_updates:
                combined_update = sum(cross_updates) / len(cross_updates)
                updated[level] = x + combined_update
            else:
                updated[level] = x

        return updated
```

### 3. Core Architecture: HierarchicalClauseGNN

```python
@dataclass(frozen=True, slots=True)
class HierarchicalGNNConfig:
    """Configuration for the hierarchical GNN."""

    # Base configuration (compatible with existing GNNConfig)
    hidden_dim: int = 256
    embedding_dim: int = 512
    dropout: float = 0.1

    # Hierarchical configuration
    hierarchy_levels: int = 5
    intra_level_layers: int = 3
    inter_level_rounds: int = 2
    cross_level_enabled: bool = True
    cross_level_heads: int = 4

    # Goal-directed features
    goal_attention_enabled: bool = True
    goal_embedding_dim: int = 128
    distance_metric: str = "cosine"  # "cosine", "euclidean", "learned"

    # Incremental updates
    incremental_enabled: bool = True
    update_batch_size: int = 32
    staleness_threshold: float = 0.1


class HierarchicalClauseGNN(nn.Module):
    """Hierarchical GNN with 5-level message passing and goal guidance."""

    def __init__(self, config: HierarchicalGNNConfig):
        super().__init__()
        self.config = config

        # Input projections (same as existing)
        self.input_projections = self._create_input_projections()

        # Hierarchical message passing modules
        self.intra_level_mp = nn.ModuleDict({
            level.name: IntraLevelMP(level, config.hidden_dim)
            for level in HierarchyLevel
        })

        self.inter_level_mp = nn.ModuleDict()
        for i in range(len(HierarchyLevel) - 1):
            lower = HierarchyLevel(i)
            upper = HierarchyLevel(i + 1)
            self.inter_level_mp[f"{lower.name}_to_{upper.name}"] = InterLevelMP(
                config.hidden_dim, config.hidden_dim
            )

        # Cross-level attention
        if config.cross_level_enabled:
            self.cross_attention = CrossLevelAttention(
                list(HierarchyLevel), config.hidden_dim
            )

        # Goal-directed components
        if config.goal_attention_enabled:
            self.goal_encoder = GoalEncoder(config.goal_embedding_dim)
            self.goal_attention = GoalDirectedAttention(
                config.hidden_dim, config.goal_embedding_dim
            )

        # Distance computation
        self.distance_computer = DistanceComputer(
            config.hidden_dim, config.distance_metric
        )

        # Output projection (maintains compatibility)
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        # Incremental update components
        if config.incremental_enabled:
            self.incremental_updater = IncrementalUpdater(config)

    def forward(self, data: HeteroData, goal_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with hierarchical message passing."""

        # Step 1: Input projections and initial embeddings
        level_embeddings = self._project_inputs(data)

        # Step 2: Multiple rounds of hierarchical message passing
        for round_idx in range(self.config.inter_level_rounds):
            # Intra-level message passing
            for level in HierarchyLevel:
                if level.name in level_embeddings:
                    level_embeddings[level] = self.intra_level_mp[level.name](
                        level_embeddings[level],
                        self._get_intra_level_edges(data, level)
                    )

            # Inter-level message passing (bottom-up then top-down)
            level_embeddings = self._inter_level_propagation(level_embeddings, data)

            # Cross-level attention
            if self.config.cross_level_enabled:
                cross_indices = self._build_cross_level_indices(data)
                level_embeddings = self.cross_attention(level_embeddings, cross_indices)

        # Step 3: Goal-directed attention (if available)
        if goal_context is not None and self.config.goal_attention_enabled:
            level_embeddings = self.goal_attention(level_embeddings, goal_context)

        # Step 4: Final clause-level aggregation and projection
        clause_embeddings = level_embeddings[HierarchyLevel.CLAUSE]
        return self.output_projection(clause_embeddings)

    def compute_goal_distance(self, clause_emb: torch.Tensor,
                            goal_emb: torch.Tensor) -> torch.Tensor:
        """Compute goal-directed distance measure."""
        return self.distance_computer(clause_emb, goal_emb)

    def incremental_update(self, new_clauses: List[Clause],
                          context: IncrementalContext) -> torch.Tensor:
        """Incrementally update embeddings for new clauses."""
        if not self.config.incremental_enabled:
            return self.forward(self._build_graph(new_clauses))

        return self.incremental_updater.update(new_clauses, context)
```

### 4. Goal-Directed Components

#### 4.1 Goal Encoder
```python
class GoalEncoder(nn.Module):
    """Encodes proof goals into embedding space."""

    def __init__(self, goal_dim: int):
        super().__init__()
        self.goal_dim = goal_dim

        # Different encoders for different goal types
        self.conjecture_encoder = nn.GRU(goal_dim, goal_dim, batch_first=True)
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(goal_dim, nhead=8), num_layers=2
        )

    def forward(self, conjectures: List[Clause],
                context_clauses: List[Clause]) -> torch.Tensor:
        """Encode proof goals into dense representation."""
        # Encode conjectures
        conj_emb = self._encode_clauses(conjectures)
        conj_out, _ = self.conjecture_encoder(conj_emb.unsqueeze(0))

        # Encode context
        ctx_emb = self._encode_clauses(context_clauses)
        ctx_out = self.context_encoder(ctx_emb.unsqueeze(0))

        # Combine goal representations
        return torch.cat([conj_out.squeeze(0), ctx_out.squeeze(0)], dim=-1)


class GoalDirectedAttention(nn.Module):
    """Goal-directed attention mechanism."""

    def __init__(self, hidden_dim: int, goal_dim: int):
        super().__init__()
        self.goal_proj = nn.Linear(goal_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)

    def forward(self, level_embeddings: Dict[HierarchyLevel, torch.Tensor],
                goal_context: torch.Tensor) -> Dict[HierarchyLevel, torch.Tensor]:
        """Apply goal-directed attention to all levels."""
        goal_query = self.goal_proj(goal_context)

        updated = {}
        for level, x in level_embeddings.items():
            attended, _ = self.attention(goal_query.unsqueeze(0), x, x)
            updated[level] = x + attended.squeeze(0)

        return updated
```

#### 4.2 Distance Computation
```python
class DistanceComputer(nn.Module):
    """Computes goal-directed distances between embeddings."""

    def __init__(self, embedding_dim: int, metric: str = "cosine"):
        super().__init__()
        self.metric = metric

        if metric == "learned":
            self.distance_net = nn.Sequential(
                nn.Linear(embedding_dim * 3, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        if self.metric == "cosine":
            return 1 - F.cosine_similarity(emb_a, emb_b, dim=-1)
        elif self.metric == "euclidean":
            return torch.norm(emb_a - emb_b, dim=-1)
        elif self.metric == "learned":
            combined = torch.cat([emb_a, emb_b, emb_a * emb_b], dim=-1)
            return self.distance_net(combined).squeeze(-1)
        else:
            raise ValueError(f"Unknown distance metric: {self.metric}")
```

### 5. Incremental Updates

```python
class IncrementalContext:
    """Context for incremental embedding updates."""

    def __init__(self):
        self.cached_embeddings: Dict[int, torch.Tensor] = {}
        self.dependency_graph: Dict[int, Set[int]] = {}
        self.staleness_scores: Dict[int, float] = {}
        self.last_update: Dict[int, int] = {}

    def is_stale(self, clause_id: int, threshold: float = 0.1) -> bool:
        return self.staleness_scores.get(clause_id, 1.0) > threshold

    def mark_dependencies_stale(self, clause_id: int):
        """Mark dependent clauses as stale when a clause changes."""
        if clause_id in self.dependency_graph:
            for dep_id in self.dependency_graph[clause_id]:
                self.staleness_scores[dep_id] = 1.0


class IncrementalUpdater(nn.Module):
    """Handles incremental embedding updates during search."""

    def __init__(self, config: HierarchicalGNNConfig):
        super().__init__()
        self.config = config

        # Change detection
        self.change_detector = StructuralChangeDetector()

        # Selective update mechanisms
        self.update_scheduler = UpdateScheduler(config.update_batch_size)

    def update(self, new_clauses: List[Clause],
               context: IncrementalContext) -> torch.Tensor:
        """Incrementally update embeddings for new clauses."""

        # Detect what needs updating
        stale_ids = [
            c.id for c in new_clauses
            if context.is_stale(c.id, self.config.staleness_threshold)
        ]

        if not stale_ids:
            # Return cached embeddings
            return torch.stack([
                context.cached_embeddings[c.id] for c in new_clauses
            ])

        # Batch updates efficiently
        update_batches = self.update_scheduler.schedule(stale_ids)

        updated_embeddings = []
        for batch in update_batches:
            batch_clauses = [c for c in new_clauses if c.id in batch]
            batch_emb = self._compute_batch_embeddings(batch_clauses, context)

            # Cache results
            for i, clause in enumerate(batch_clauses):
                context.cached_embeddings[clause.id] = batch_emb[i]
                context.staleness_scores[clause.id] = 0.0

            updated_embeddings.append(batch_emb)

        return torch.cat(updated_embeddings, dim=0)
```

## API Design

### 6.1 Enhanced EmbeddingProvider Interface

```python
class HierarchicalEmbeddingProvider:
    """Enhanced provider supporting hierarchical GNN features."""

    def __init__(self, model: HierarchicalClauseGNN, config: EmbeddingProviderConfig):
        self.model = model
        self.config = config
        self.incremental_context = IncrementalContext()

        # Maintain backward compatibility
        self._base_provider = GNNEmbeddingProvider(...)

    # Backward compatible methods
    def get_embedding(self, clause: Clause) -> List[float] | None:
        """Backward compatible single clause embedding."""
        return self._base_provider.get_embedding(clause)

    def get_embeddings_batch(self, clauses: List[Clause]) -> List[List[float] | None]:
        """Backward compatible batch embedding."""
        return self._base_provider.get_embeddings_batch(clauses)

    @property
    def embedding_dim(self) -> int:
        return self.model.config.embedding_dim

    # Enhanced hierarchical methods
    def get_hierarchical_embedding(self, clause: Clause,
                                 level: HierarchyLevel) -> List[float] | None:
        """Get embedding at specific hierarchy level."""
        # Implementation details...

    def get_goal_directed_embedding(self, clause: Clause,
                                  goal_context: List[Clause]) -> List[float] | None:
        """Get goal-directed embedding."""
        # Implementation details...

    def compute_goal_distance(self, clause: Clause, goal: List[Clause]) -> float:
        """Compute goal-directed distance."""
        clause_emb = self.get_embedding(clause)
        goal_emb = self._encode_goal(goal)
        return self.model.compute_goal_distance(
            torch.tensor(clause_emb), torch.tensor(goal_emb)
        ).item()

    def incremental_update(self, new_clauses: List[Clause]) -> None:
        """Trigger incremental update for new clauses."""
        self.model.incremental_update(new_clauses, self.incremental_context)
```

### 6.2 Configuration Interface

```python
@dataclass(frozen=True)
class HierarchicalSelectionConfig:
    """Configuration for hierarchical selection features."""

    # Base ML config (backward compatible)
    base_config: MLSelectionConfig = field(default_factory=MLSelectionConfig)

    # Hierarchical features
    use_hierarchical_gnn: bool = False
    hierarchy_weight: float = 0.4
    goal_directed_weight: float = 0.3
    cross_level_attention: bool = True
    incremental_updates: bool = True

    # Goal-directed parameters
    goal_context_size: int = 50
    goal_distance_threshold: float = 0.7
    goal_recompute_interval: int = 100
```

## Integration Plan

### 7.1 Backward Compatibility Strategy

1. **Protocol Compatibility**: All existing EmbeddingProvider methods remain unchanged
2. **Configuration Compatibility**: HierarchicalGNNConfig extends GNNConfig
3. **Factory Pattern**: Automatic fallback to base implementation when hierarchical features are disabled
4. **No Breaking Changes**: Existing code continues to work without modification

### 7.2 Incremental Deployment

```python
def create_embedding_provider(
    symbol_table: SymbolTable | None = None,
    config: EmbeddingProviderConfig | None = None,
    hierarchical_config: HierarchicalGNNConfig | None = None,
) -> EmbeddingProvider:
    """Factory with hierarchical support."""

    if hierarchical_config is not None and hierarchical_config.enabled:
        # Use new hierarchical provider
        model = HierarchicalClauseGNN(hierarchical_config)
        return HierarchicalEmbeddingProvider(model, config or EmbeddingProviderConfig())
    else:
        # Fall back to existing implementation
        return GNNEmbeddingProvider.create(symbol_table, config)
```

### 7.3 Command-Line Interface

```bash
# Enable hierarchical GNN
pyladr --ml-enhanced --hierarchical-gnn \
       --hierarchy-levels 5 \
       --cross-level-attention \
       --goal-directed \
       input.in

# Backward compatible (no changes)
pyladr --ml-enhanced input.in
```

## Test-Driven Development Structure

### 8.1 Unit Tests Structure

```python
# tests/unit/test_hierarchical_gnn.py

class TestHierarchicalGNN:
    """Comprehensive test suite for hierarchical GNN."""

    def test_backward_compatibility(self):
        """Ensure existing EmbeddingProvider interface works unchanged."""

    def test_hierarchy_message_passing(self):
        """Test message passing at each hierarchy level."""

    def test_cross_level_attention(self):
        """Test cross-level attention mechanisms."""

    def test_goal_directed_attention(self):
        """Test goal-directed attention and distance computation."""

    def test_incremental_updates(self):
        """Test incremental embedding update correctness."""

    def test_performance_metrics(self):
        """Test performance compared to baseline GNN."""

    def test_thread_safety(self):
        """Test thread-safe model hot-swapping with hierarchical features."""

# tests/integration/test_hierarchical_selection.py

class TestHierarchicalSelection:
    """Integration tests for hierarchical selection."""

    def test_prover9_equivalence(self):
        """Ensure no breaking changes vs reference Prover9."""

    def test_ml_selection_integration(self):
        """Test integration with existing MLSelection."""

    def test_proof_workflows(self):
        """Test complete proof workflows with hierarchical features."""
```

### 8.2 Performance Benchmarks

```python
# tests/benchmarks/test_hierarchical_performance.py

class TestHierarchicalPerformance:
    """Performance validation for hierarchical GNN."""

    def test_embedding_computation_speed(self):
        """Benchmark embedding computation vs baseline."""

    def test_memory_usage(self):
        """Benchmark memory usage vs baseline."""

    def test_incremental_update_efficiency(self):
        """Benchmark incremental update performance."""

    def test_goal_directed_effectiveness(self):
        """Measure proof search effectiveness with goal guidance."""
```

## Performance and Memory Projections

### 9.1 Computational Complexity

| Component | Current O() | Hierarchical O() | Overhead |
|-----------|-------------|------------------|----------|
| Node Processing | O(N) | O(N) | 1x |
| Edge Processing | O(E) | O(E + H*E) | ~3x |
| Cross-level Attention | - | O(N²/L) | New |
| Goal Distance | - | O(N*G) | New |
| Total Forward Pass | O(N + E) | O(N + H*E + N²/L) | ~3-4x |

Where:
- N = total nodes, E = total edges, H = hierarchy levels (5), L = levels, G = goals

### 9.2 Memory Usage

| Component | Current Memory | Hierarchical Memory | Overhead |
|-----------|---------------|-------------------|----------|
| Node Embeddings | N * D | N * D | 1x |
| Level Projections | - | H * D² | +H*D² |
| Cross Attention | - | L² * D² | +L²*D² |
| Incremental Cache | - | C * D | +C*D |
| Total | ~N*D | ~N*D + H*D² + L²*D² + C*D | ~2-3x |

**Estimated Requirements:**
- Base case: 1000 clauses * 512 dim = 512KB embeddings
- Hierarchical: ~1.5-2MB total memory overhead
- Acceptable for modern hardware with significant performance benefits

## Conclusion

This hierarchical GNN architecture provides:

1. **Enhanced Representation**: 5-level hierarchy captures logical structure more effectively
2. **Goal-Directed Search**: Direct goal guidance improves proof search efficiency
3. **Incremental Processing**: Real-time updates during search
4. **Backward Compatibility**: No breaking changes to existing code
5. **Comprehensive Testing**: TDD approach ensures reliability

The architecture maintains the existing EmbeddingProvider protocol while adding powerful hierarchical features that can be enabled incrementally. The design supports both immediate deployment with basic features and future expansion with advanced goal-directed capabilities.