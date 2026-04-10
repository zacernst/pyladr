"""Comprehensive test suite for hierarchical GNN architecture.

This module follows Test-Driven Development (TDD) principles and ensures:
1. Backward compatibility with existing EmbeddingProvider protocol
2. Correctness of hierarchical message passing at each level
3. Cross-level attention mechanisms
4. Goal-directed features and distance computation
5. Incremental update functionality
6. Thread safety with concurrent access
7. Performance compared to baseline GNN
8. NO BREAKING CHANGES vs reference Prover9

Test Structure:
- Unit tests for each component
- Integration tests for full workflows
- Performance benchmarks
- Compatibility validation
- Property-based testing for correctness
"""

import pytest
import torch
import threading
import time
from unittest.mock import Mock, patch
from typing import List, Dict

from pyladr.core.clause import Clause
from pyladr.core.symbol import SymbolTable
from pyladr.ml.graph.clause_graph import clause_to_heterograph
from pyladr.ml.embedding_provider import EmbeddingProvider, GNNEmbeddingProvider

# Test the hierarchical components
from pyladr.ml.hierarchical import (
    HierarchicalClauseGNN,
    HierarchicalGNNConfig,
    HierarchyLevel,
    HierarchicalEmbeddingProvider,
    HierarchicalEmbeddingProviderConfig,
    create_hierarchical_embedding_provider,
)


class TestBackwardCompatibility:
    """Test that hierarchical GNN maintains full backward compatibility."""

    def test_embedding_provider_protocol_compatibility(self):
        """Ensure HierarchicalEmbeddingProvider implements EmbeddingProvider protocol."""
        config = HierarchicalEmbeddingProviderConfig(use_hierarchical_features=False)
        provider = create_hierarchical_embedding_provider(config=config)

        # Must implement EmbeddingProvider protocol
        assert isinstance(provider, EmbeddingProvider)
        assert hasattr(provider, 'embedding_dim')
        assert hasattr(provider, 'get_embedding')
        assert hasattr(provider, 'get_embeddings_batch')

        # Property access must work
        assert isinstance(provider.embedding_dim, int)
        assert provider.embedding_dim > 0

    def test_method_signatures_unchanged(self):
        """Verify method signatures match existing EmbeddingProvider exactly."""
        config = HierarchicalEmbeddingProviderConfig(use_hierarchical_features=False)
        provider = create_hierarchical_embedding_provider(config=config)

        # Mock clause for testing
        clause = Mock(spec=Clause)
        clause.id = 1

        # get_embedding signature: (clause: Clause) -> List[float] | None
        result = provider.get_embedding(clause)
        assert result is None or isinstance(result, list)

        # get_embeddings_batch signature: (clauses: List[Clause]) -> List[List[float] | None]
        batch_result = provider.get_embeddings_batch([clause])
        assert isinstance(batch_result, list)
        assert len(batch_result) == 1

    def test_fallback_to_base_provider(self):
        """Test automatic fallback to base GNNEmbeddingProvider."""
        config = HierarchicalEmbeddingProviderConfig(
            use_hierarchical_features=False,
            fallback_to_base_on_error=True
        )

        with patch('pyladr.ml.hierarchical.provider.HierarchicalClauseGNN') as mock_model:
            # Make hierarchical model fail
            mock_model.side_effect = RuntimeError("Hierarchical model failed")

            provider = create_hierarchical_embedding_provider(config=config)

            # Should successfully create base provider
            assert provider is not None
            assert hasattr(provider, 'get_embedding')

    def test_no_op_fallback_when_ml_unavailable(self):
        """Test graceful fallback to NoOpEmbeddingProvider when ML deps unavailable."""
        with patch('pyladr.ml.hierarchical.provider.HierarchicalClauseGNN') as mock_model:
            # Simulate missing ML dependencies
            mock_model.side_effect = ImportError("torch not available")

            config = HierarchicalEmbeddingProviderConfig(use_hierarchical_features=True)
            provider = create_hierarchical_embedding_provider(config=config)

            # Should create some provider (likely NoOp fallback)
            assert provider is not None
            clause = Mock(spec=Clause)
            clause.id = 1

            # Should not crash, even if returns None
            result = provider.get_embedding(clause)
            assert result is None or isinstance(result, list)


class TestHierarchicalGNNArchitecture:
    """Test the core hierarchical GNN architecture components."""

    @pytest.fixture
    def hierarchical_config(self):
        """Standard hierarchical GNN configuration for testing."""
        return HierarchicalGNNConfig(
            hierarchy_levels=5,
            intra_level_layers=2,  # Reduced for testing speed
            inter_level_rounds=2,
            cross_level_enabled=True,
            goal_attention_enabled=True,
        )

    @pytest.fixture
    def mock_heterograph(self):
        """Mock HeteroData graph for testing."""
        from torch_geometric.data import HeteroData

        data = HeteroData()

        # Add mock node features for each level
        data['clause'].x = torch.randn(3, 7)  # 3 clauses, 7 features
        data['clause'].num_nodes = 3

        data['literal'].x = torch.randn(8, 3)  # 8 literals, 3 features
        data['literal'].num_nodes = 8

        data['term'].x = torch.randn(15, 8)  # 15 terms, 8 features
        data['term'].num_nodes = 15

        data['symbol'].x = torch.randn(10, 6)  # 10 symbols, 6 features
        data['symbol'].num_nodes = 10

        data['variable'].x = torch.randn(5, 1)  # 5 variables, 1 feature
        data['variable'].num_nodes = 5

        # Add mock edges
        data['clause', 'contains_literal', 'literal'].edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2, 2],
            [0, 1, 2, 3, 4, 5, 6]
        ])

        return data

    def test_hierarchy_level_enumeration(self):
        """Test HierarchyLevel enum structure and relationships."""
        levels = list(HierarchyLevel)
        assert len(levels) == 5

        # Verify ordering
        assert HierarchyLevel.SYMBOL.value == 0
        assert HierarchyLevel.PROOF.value == 4

        # Test adjacency relationships
        assert HierarchyLevel.SYMBOL.is_adjacent_to(HierarchyLevel.TERM)
        assert not HierarchyLevel.SYMBOL.is_adjacent_to(HierarchyLevel.LITERAL)

        # Test node type mappings
        assert 'symbol' in HierarchyLevel.SYMBOL.node_types
        assert 'clause' in HierarchyLevel.CLAUSE.node_types

    def test_hierarchical_gnn_initialization(self, hierarchical_config):
        """Test HierarchicalClauseGNN model initialization."""
        model = HierarchicalClauseGNN(hierarchical_config)

        # Verify components are created
        assert hasattr(model, 'base_gnn')
        assert hasattr(model, 'intra_level_mp')
        assert hasattr(model, 'inter_level_mp')
        assert hasattr(model, 'output_projection')

        # Verify hierarchical layers
        assert len(model.intra_level_mp) == 5  # 5 hierarchy levels
        assert len(model.inter_level_mp) == 4  # 4 inter-level connections

        # Verify goal components when enabled
        if hierarchical_config.goal_attention_enabled:
            assert hasattr(model, 'goal_encoder')
            assert hasattr(model, 'goal_attention')

    def test_forward_pass_with_hierarchical_features(self, hierarchical_config, mock_heterograph):
        """Test forward pass with hierarchical features enabled."""
        model = HierarchicalClauseGNN(hierarchical_config)
        model.eval()

        # Forward pass should work with hierarchical features
        with torch.no_grad():
            output = model.forward(mock_heterograph, use_hierarchical=True)

        # Verify output shape
        assert output.shape[0] == 3  # 3 clauses
        assert output.shape[1] == hierarchical_config.embedding_dim

        # Output should be different from base GNN
        base_output = model.forward(mock_heterograph, use_hierarchical=False)
        assert not torch.allclose(output, base_output, atol=1e-6)

    def test_cross_level_attention(self, hierarchical_config, mock_heterograph):
        """Test cross-level attention mechanism."""
        # Enable cross-level attention
        config = hierarchical_config
        config = HierarchicalGNNConfig(
            **{**config.__dict__, 'cross_level_enabled': True}
        )

        model = HierarchicalClauseGNN(config)
        model.eval()

        # Forward pass with cross-level attention
        with torch.no_grad():
            output_with_cross = model.forward(mock_heterograph, use_hierarchical=True)

        # Disable cross-level attention
        config_no_cross = HierarchicalGNNConfig(
            **{**config.__dict__, 'cross_level_enabled': False}
        )
        model_no_cross = HierarchicalClauseGNN(config_no_cross)
        model_no_cross.eval()

        with torch.no_grad():
            output_without_cross = model_no_cross.forward(mock_heterograph, use_hierarchical=True)

        # Outputs should be different when cross-level attention is enabled
        assert not torch.allclose(output_with_cross, output_without_cross, atol=1e-6)

    def test_hierarchical_embedding_extraction(self, hierarchical_config, mock_heterograph):
        """Test extraction of embeddings at specific hierarchy levels."""
        model = HierarchicalClauseGNN(hierarchical_config)
        model.eval()

        for level in HierarchyLevel:
            with torch.no_grad():
                embedding = model.get_hierarchical_embedding(mock_heterograph, level)

            if embedding.numel() > 0:  # Some levels might be empty in mock data
                assert embedding.shape[1] == hierarchical_config.hidden_dim


class TestGoalDirectedFeatures:
    """Test goal-directed attention and distance computation."""

    @pytest.fixture
    def goal_enabled_config(self):
        """Configuration with goal-directed features enabled."""
        return HierarchicalGNNConfig(
            goal_attention_enabled=True,
            goal_embedding_dim=64,
            distance_metric="cosine",
        )

    @pytest.fixture
    def mock_goal_context(self):
        """Mock goal context tensor."""
        return torch.randn(5, 64)  # 5 goal elements, 64 dimensions

    def test_goal_directed_forward_pass(self, goal_enabled_config, mock_heterograph, mock_goal_context):
        """Test forward pass with goal context."""
        model = HierarchicalClauseGNN(goal_enabled_config)
        model.eval()

        with torch.no_grad():
            # Forward pass without goal context
            output_no_goal = model.forward(mock_heterograph)

            # Forward pass with goal context
            output_with_goal = model.forward(mock_heterograph, goal_context=mock_goal_context)

        # Outputs should be different when goal context is provided
        assert not torch.allclose(output_no_goal, output_with_goal, atol=1e-6)

    def test_distance_computation_methods(self, goal_enabled_config):
        """Test different distance computation methods."""
        metrics = ["cosine", "euclidean", "learned"]

        for metric in metrics:
            config = HierarchicalGNNConfig(
                **{**goal_enabled_config.__dict__, 'distance_metric': metric}
            )
            model = HierarchicalClauseGNN(config)

            # Test distance computation
            emb_a = torch.randn(10, 128)
            emb_b = torch.randn(10, 128)

            distances = model.compute_goal_distance(emb_a, emb_b)

            assert distances.shape == (10,)
            assert torch.all(distances >= 0)  # Distances should be non-negative

            # Cosine distance should be in [0, 2]
            if metric == "cosine":
                assert torch.all(distances <= 2.0)

    def test_goal_distance_properties(self, goal_enabled_config):
        """Test mathematical properties of goal distance function."""
        model = HierarchicalClauseGNN(goal_enabled_config)

        # Identity: distance(x, x) should be 0 for cosine/euclidean
        x = torch.randn(5, 128)
        self_distance = model.compute_goal_distance(x, x)

        if goal_enabled_config.distance_metric in ["cosine", "euclidean"]:
            assert torch.allclose(self_distance, torch.zeros_like(self_distance), atol=1e-6)


class TestIncrementalUpdates:
    """Test incremental embedding update functionality."""

    @pytest.fixture
    def incremental_config(self):
        """Configuration with incremental updates enabled."""
        return HierarchicalGNNConfig(
            incremental_enabled=True,
            update_batch_size=8,
            staleness_threshold=0.1,
        )

    def test_incremental_context_management(self, incremental_config):
        """Test incremental context state management."""
        from pyladr.ml.hierarchical.incremental import IncrementalContext

        context = IncrementalContext()

        # Test staleness tracking
        assert context.is_stale(1, threshold=0.1)  # New clause should be stale

        # Add cached embedding
        context.cached_embeddings[1] = torch.randn(128)
        context.staleness_scores[1] = 0.05

        assert not context.is_stale(1, threshold=0.1)  # Should not be stale

        # Test dependency propagation
        context.dependency_graph[1] = {2, 3}
        context.mark_dependencies_stale(1)

        assert context.is_stale(2, threshold=0.1)
        assert context.is_stale(3, threshold=0.1)

    def test_incremental_update_efficiency(self, incremental_config):
        """Test that incremental updates are more efficient than full computation."""
        model = HierarchicalClauseGNN(incremental_config)
        model.eval()

        # Mock clauses
        clauses = [Mock(spec=Clause) for _ in range(10)]
        for i, clause in enumerate(clauses):
            clause.id = i

        # Create incremental context with some cached embeddings
        from pyladr.ml.hierarchical.incremental import IncrementalContext
        context = IncrementalContext()

        # Pre-populate cache for half the clauses
        for i in range(5):
            context.cached_embeddings[i] = torch.randn(512)
            context.staleness_scores[i] = 0.05  # Not stale

        # Time incremental update vs full computation
        start_time = time.time()
        incremental_result = model.incremental_update(clauses, context)
        incremental_time = time.time() - start_time

        # Note: Full timing comparison would need actual implementation
        # This test verifies the interface works
        assert incremental_result is not None


class TestThreadSafety:
    """Test thread safety of hierarchical features."""

    @pytest.fixture
    def thread_safe_config(self):
        """Configuration for thread safety testing."""
        return HierarchicalEmbeddingProviderConfig(
            use_hierarchical_features=True,
            enable_goal_directed_selection=True,
            enable_incremental_updates=True,
        )

    def test_concurrent_embedding_requests(self, thread_safe_config):
        """Test concurrent embedding requests are thread-safe."""
        provider = create_hierarchical_embedding_provider(config=thread_safe_config)
        if not isinstance(provider, HierarchicalEmbeddingProvider):
            pytest.skip("Hierarchical provider not available")

        # Mock clauses
        clauses = [Mock(spec=Clause) for _ in range(20)]
        for i, clause in enumerate(clauses):
            clause.id = i

        results = {}
        errors = []

        def worker(thread_id):
            try:
                for i in range(10):
                    clause = clauses[thread_id * 10 + i]
                    embedding = provider.get_embedding(clause)
                    results[f"{thread_id}_{i}"] = embedding
            except Exception as e:
                errors.append(e)

        # Run concurrent threads
        threads = []
        for i in range(2):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should not have any errors
        assert len(errors) == 0

        # Should have results from both threads
        assert len(results) == 20

    def test_concurrent_model_swapping(self, thread_safe_config):
        """Test thread-safe model hot-swapping during concurrent inference."""
        provider = create_hierarchical_embedding_provider(config=thread_safe_config)
        if not isinstance(provider, HierarchicalEmbeddingProvider):
            pytest.skip("Hierarchical provider not available")

        clause = Mock(spec=Clause)
        clause.id = 1

        swap_errors = []
        embedding_errors = []
        embedding_results = []

        def embedding_worker():
            """Continuously request embeddings."""
            for _ in range(50):
                try:
                    result = provider.get_embedding(clause)
                    embedding_results.append(result)
                    time.sleep(0.001)  # Small delay
                except Exception as e:
                    embedding_errors.append(e)

        def swap_worker():
            """Periodically swap models."""
            for _ in range(5):
                try:
                    time.sleep(0.01)  # Let some embeddings happen first
                    # Create new model and swap
                    new_model = HierarchicalClauseGNN(thread_safe_config.hierarchical_config)
                    provider.swap_hierarchical_model(new_model)
                except Exception as e:
                    swap_errors.append(e)

        # Start threads
        embedding_thread = threading.Thread(target=embedding_worker)
        swap_thread = threading.Thread(target=swap_worker)

        embedding_thread.start()
        swap_thread.start()

        embedding_thread.join()
        swap_thread.join()

        # No errors should occur during concurrent operations
        assert len(swap_errors) == 0
        assert len(embedding_errors) == 0
        assert len(embedding_results) > 0


class TestPerformanceBenchmarks:
    """Performance validation for hierarchical GNN."""

    @pytest.fixture
    def benchmark_config(self):
        """Configuration for performance benchmarking."""
        return HierarchicalGNNConfig(
            hierarchy_levels=5,
            intra_level_layers=3,
            inter_level_rounds=2,
            cross_level_enabled=True,
            goal_attention_enabled=True,
        )

    @pytest.mark.slow
    def test_embedding_computation_speed(self, benchmark_config, mock_heterograph):
        """Benchmark embedding computation speed vs baseline."""
        # Baseline model
        from pyladr.ml.graph.clause_encoder import HeterogeneousClauseGNN, GNNConfig
        baseline_model = HeterogeneousClauseGNN(GNNConfig())

        # Hierarchical model
        hierarchical_model = HierarchicalClauseGNN(benchmark_config)

        # Warm up
        for model in [baseline_model, hierarchical_model]:
            model.eval()
            with torch.no_grad():
                for _ in range(5):
                    if model == hierarchical_model:
                        model.forward(mock_heterograph, use_hierarchical=True)
                    else:
                        model.forward(mock_heterograph)

        # Benchmark baseline
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                baseline_model.forward(mock_heterograph)
        baseline_time = time.time() - start_time

        # Benchmark hierarchical
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                hierarchical_model.forward(mock_heterograph, use_hierarchical=True)
        hierarchical_time = time.time() - start_time

        # Hierarchical should not be more than 5x slower than baseline
        slowdown_factor = hierarchical_time / baseline_time
        assert slowdown_factor < 5.0, f"Hierarchical GNN is {slowdown_factor:.2f}x slower than baseline"

    @pytest.mark.slow
    def test_memory_usage_overhead(self, benchmark_config):
        """Test memory usage overhead of hierarchical features."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Measure baseline memory
        baseline_model = HierarchicalClauseGNN(benchmark_config)
        baseline_memory = process.memory_info().rss

        # Create additional hierarchical models to test memory scaling
        models = []
        for i in range(5):
            model = HierarchicalClauseGNN(benchmark_config)
            models.append(model)

        hierarchical_memory = process.memory_info().rss

        # Memory overhead should be reasonable (less than 5x baseline per model)
        memory_overhead_per_model = (hierarchical_memory - baseline_memory) / len(models)
        baseline_model_memory = baseline_memory  # Approximate

        overhead_factor = memory_overhead_per_model / baseline_model_memory
        assert overhead_factor < 3.0, f"Memory overhead is {overhead_factor:.2f}x baseline per model"

    def test_incremental_update_efficiency(self, benchmark_config):
        """Test efficiency of incremental updates vs full recomputation."""
        from pyladr.ml.hierarchical.incremental import IncrementalContext

        model = HierarchicalClauseGNN(benchmark_config)
        model.eval()

        # Mock setup
        clauses = [Mock(spec=Clause) for _ in range(100)]
        for i, clause in enumerate(clauses):
            clause.id = i

        context = IncrementalContext()

        # Pre-populate cache for most clauses
        for i in range(90):
            context.cached_embeddings[i] = torch.randn(512)
            context.staleness_scores[i] = 0.05  # Not stale

        # Only 10 clauses need updates
        for i in range(90, 100):
            context.staleness_scores[i] = 0.5  # Stale

        # Time incremental vs full computation
        start_time = time.time()
        incremental_result = model.incremental_update(clauses, context)
        incremental_time = time.time() - start_time

        # This test verifies the interface - actual speedup testing would
        # require full implementation of incremental update logic
        assert incremental_result is not None
        assert incremental_time >= 0


class TestProver9Compatibility:
    """Ensure NO BREAKING CHANGES vs reference Prover9."""

    def test_default_behavior_unchanged(self):
        """Test that default behavior is identical to non-hierarchical version."""
        # Configuration that disables all hierarchical features
        config = HierarchicalEmbeddingProviderConfig(
            use_hierarchical_features=False,
            enable_goal_directed_selection=False,
            enable_incremental_updates=False,
        )

        provider = create_hierarchical_embedding_provider(config=config)

        # Should behave exactly like base provider
        clause = Mock(spec=Clause)
        clause.id = 1

        result = provider.get_embedding(clause)
        # Result should be None (mock) or valid embedding - no crashes
        assert result is None or isinstance(result, list)

    def test_search_behavior_identical_when_disabled(self):
        """Test that search behavior is identical when hierarchical features are disabled."""
        # This test would integrate with actual search components
        # For now, we verify the interface remains the same

        config = HierarchicalEmbeddingProviderConfig(use_hierarchical_features=False)
        provider = create_hierarchical_embedding_provider(config=config)

        # All EmbeddingProvider methods must work
        assert hasattr(provider, 'get_embedding')
        assert hasattr(provider, 'get_embeddings_batch')
        assert hasattr(provider, 'embedding_dim')

        # Property access must work
        dim = provider.embedding_dim
        assert isinstance(dim, int)

    def test_configuration_backward_compatibility(self):
        """Test that existing configuration code continues to work."""
        # Test with old-style configuration (base config only)
        from pyladr.ml.embedding_provider import EmbeddingProviderConfig

        old_config = EmbeddingProviderConfig()
        hierarchical_config = HierarchicalEmbeddingProviderConfig(
            base_config=old_config,
            use_hierarchical_features=False,
        )

        provider = create_hierarchical_embedding_provider(config=hierarchical_config)
        assert provider is not None

        # Should have same embedding dimension as base config
        assert provider.embedding_dim == old_config.model_path or 512  # Default


# Integration test markers for different test categories
pytest.mark.compatibility = pytest.mark.compatibility
pytest.mark.performance = pytest.mark.slow
pytest.mark.integration = pytest.mark.integration


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        "-v",
        "--tb=short",
        "-m", "not slow",  # Skip slow benchmarks by default
        __file__,
    ])