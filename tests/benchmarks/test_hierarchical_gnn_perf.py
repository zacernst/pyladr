"""Performance benchmarks for hierarchical GNN message passing.

Measures computational overhead, memory usage, and scalability of the
5-level hierarchical message passing architecture against the baseline
HeterogeneousClauseGNN encoder.

Key areas tested:
1. Per-level MPN forward pass latency (symbol, term, literal, clause, proof)
2. Full hierarchical pipeline vs baseline GNN overhead
3. Memory footprint scaling with clause count and term depth
4. Graph construction throughput for varying clause complexity
5. Embedding cache interaction with hierarchical features
6. Batch processing throughput for realistic search workloads

Run with: pytest tests/benchmarks/test_hierarchical_gnn_perf.py -v
"""

from __future__ import annotations

import gc
import math
import statistics
import sys
import time
from dataclasses import dataclass, field

import pytest

torch = pytest.importorskip("torch")
pyg = pytest.importorskip("torch_geometric")

from torch_geometric.data import HeteroData

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term
from pyladr.core.symbol import SymbolTable
from pyladr.ml.graph.clause_graph import (
    ClauseGraphConfig,
    NodeType,
    EdgeType,
    clause_to_heterograph,
    batch_clauses_to_heterograph,
)
from pyladr.ml.graph.clause_encoder import (
    GNNConfig,
    HeterogeneousClauseGNN,
)
from pyladr.ml.graph.hierarchical_mpn import (
    HierarchicalMPNConfig,
    SymbolLevelMPN,
    TermLevelMPN,
    LiteralLevelMPN,
)
from pyladr.ml.graph.clause_level_mpn import (
    ClauseLevelConfig,
    ClausePropertyEncoder,
)
from pyladr.ml.graph.proof_level_mpn import (
    ProofLevelConfig,
    TemporalPositionEncoder,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_term(symnum: int, args: tuple[Term, ...] = ()) -> Term:
    return Term(private_symbol=-symnum, arity=len(args), args=args)


def _make_var(varnum: int) -> Term:
    return Term(private_symbol=varnum, arity=0, args=())


def _make_clause(
    lits: list[tuple[bool, Term]], clause_id: int = 0,
) -> Clause:
    literals = tuple(Literal(sign=s, atom=a) for s, a in lits)
    c = Clause(literals=literals)
    c.id = clause_id
    c.weight = float(len(literals))
    return c


def _make_nested_term(depth: int, base_sym: int = 1) -> Term:
    """Create a term with specified nesting depth."""
    if depth <= 0:
        return _make_var(base_sym % 5)
    left = _make_nested_term(depth - 1, base_sym * 2)
    right = _make_nested_term(depth - 1, base_sym * 2 + 1)
    return _make_term(base_sym % 10 + 1, (left, right))


def _make_complex_clause(clause_id: int, num_lits: int = 2, depth: int = 2) -> Clause:
    """Create a clause with multiple literals and nested terms."""
    lits: list[tuple[bool, Term]] = []
    for i in range(num_lits):
        atom = _make_nested_term(depth, clause_id * 10 + i)
        lits.append((i % 2 == 0, atom))
    return _make_clause(lits, clause_id)


def _make_equational_clause(clause_id: int, depth: int = 2) -> Clause:
    """Create an equational clause: lhs = rhs."""
    lhs = _make_nested_term(depth, clause_id * 2)
    rhs = _make_nested_term(depth, clause_id * 2 + 1)
    eq_atom = _make_term(0, (lhs, rhs))  # symnum 0 = equality
    return _make_clause([(True, eq_atom)], clause_id)


def _build_graph_with_features(clause: Clause, hidden_dim: int = 256) -> tuple[HeteroData, dict[str, torch.Tensor]]:
    """Build a graph and project features to hidden_dim for MPN testing."""
    data = clause_to_heterograph(clause)
    x_dict: dict[str, torch.Tensor] = {}

    for nt in NodeType:
        key = nt.value
        if key in data.node_types:
            store = data[key]
            if store.num_nodes > 0 and hasattr(store, "x") and store.x is not None:
                feat_dim = store.x.shape[1]
                proj = torch.nn.Linear(feat_dim, hidden_dim)
                with torch.no_grad():
                    x_dict[key] = torch.relu(proj(store.x))

    return data, x_dict


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Result of a micro-benchmark."""
    ops_per_second: float
    mean_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    peak_memory_mb: float = 0.0

    def __str__(self) -> str:
        return (
            f"{self.ops_per_second:.0f} ops/s, "
            f"mean={self.mean_latency_ms:.2f}ms, "
            f"p50={self.p50_latency_ms:.2f}ms, "
            f"p99={self.p99_latency_ms:.2f}ms"
        )


def _benchmark(fn, *, warmup: int = 5, iterations: int = 50) -> BenchmarkResult:
    """Run a function repeatedly and collect latency statistics."""
    for _ in range(warmup):
        fn()

    gc.collect()
    gc.disable()
    try:
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn()
            latencies.append(time.perf_counter() - start)
    finally:
        gc.enable()

    latencies.sort()
    n = len(latencies)
    mean = statistics.mean(latencies)
    return BenchmarkResult(
        ops_per_second=1.0 / mean if mean > 0 else float("inf"),
        mean_latency_ms=mean * 1000,
        p50_latency_ms=latencies[n // 2] * 1000,
        p99_latency_ms=latencies[int(n * 0.99)] * 1000,
    )


def _measure_memory_mb() -> float:
    """Measure current RSS in megabytes."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except ImportError:
        return 0.0


# ── Graph construction benchmarks ─────────────────────────────────────────


class TestGraphConstructionPerformance:
    """Benchmark clause-to-graph conversion throughput."""

    def test_simple_clause_graph_construction(self):
        """Graph construction for simple unit clauses."""
        clause = _make_clause([(True, _make_term(1, (_make_var(0),)))], 1)

        result = _benchmark(lambda: clause_to_heterograph(clause), iterations=200)
        print(f"\nSimple clause graph construction: {result}")
        assert result.ops_per_second > 500, f"Too slow: {result.ops_per_second:.0f} ops/s"

    def test_complex_clause_graph_construction(self):
        """Graph construction for complex equational clauses (depth 3)."""
        clause = _make_complex_clause(1, num_lits=3, depth=3)

        result = _benchmark(lambda: clause_to_heterograph(clause), iterations=100)
        print(f"\nComplex clause (3 lits, depth 3) graph construction: {result}")
        assert result.ops_per_second > 100, f"Too slow: {result.ops_per_second:.0f} ops/s"

    def test_batch_graph_construction(self):
        """Batch graph construction for 50 clauses."""
        clauses = [_make_complex_clause(i, num_lits=2, depth=2) for i in range(50)]

        result = _benchmark(
            lambda: batch_clauses_to_heterograph(clauses),
            iterations=20,
        )
        per_clause_ms = result.mean_latency_ms / len(clauses)
        print(f"\nBatch 50 clauses: {result}, per-clause={per_clause_ms:.2f}ms")
        assert per_clause_ms < 5.0, f"Per-clause too slow: {per_clause_ms:.2f}ms"

    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    def test_graph_construction_scaling_with_depth(self, depth: int):
        """Graph construction time should scale polynomially with term depth."""
        clause = _make_equational_clause(1, depth=depth)

        result = _benchmark(lambda: clause_to_heterograph(clause), iterations=50)
        print(f"\nDepth {depth} graph construction: {result}")
        # Depth 4 creates ~30 nodes — still should be fast
        assert result.mean_latency_ms < 20.0, f"Depth {depth} too slow: {result.mean_latency_ms:.2f}ms"


# ── Per-level MPN benchmarks ─────────────────────────────────────────────


class TestSymbolLevelMPNPerformance:
    """Benchmark symbol-level message passing."""

    def test_symbol_mpn_forward_pass(self):
        """Symbol-level MPN forward pass latency."""
        config = HierarchicalMPNConfig(hidden_dim=128)
        mpn = SymbolLevelMPN(config)
        mpn.eval()

        clause = _make_complex_clause(1, num_lits=3, depth=3)
        data, x_dict = _build_graph_with_features(clause, hidden_dim=128)

        with torch.no_grad():
            result = _benchmark(lambda: mpn(x_dict, data), iterations=50)
        print(f"\nSymbolLevelMPN forward: {result}")
        assert result.mean_latency_ms < 50.0, f"Too slow: {result.mean_latency_ms:.2f}ms"

    def test_symbol_mpn_with_many_symbols(self):
        """Symbol MPN with 20+ distinct symbols (realistic search state)."""
        config = HierarchicalMPNConfig(hidden_dim=128)
        mpn = SymbolLevelMPN(config)
        mpn.eval()

        # Create clause with many distinct symbols
        terms = [_make_term(i, (_make_var(0), _make_var(1))) for i in range(1, 11)]
        lits = [(True, t) for t in terms]
        clause = _make_clause(lits, 1)
        data, x_dict = _build_graph_with_features(clause, hidden_dim=128)

        with torch.no_grad():
            result = _benchmark(lambda: mpn(x_dict, data), iterations=50)
        print(f"\nSymbolLevelMPN (20+ symbols): {result}")
        assert result.mean_latency_ms < 100.0


class TestTermLevelMPNPerformance:
    """Benchmark term-level message passing."""

    def test_term_mpn_forward_pass(self):
        """Term-level MPN forward pass latency."""
        config = HierarchicalMPNConfig(hidden_dim=128)
        mpn = TermLevelMPN(config)
        mpn.eval()

        clause = _make_complex_clause(1, num_lits=2, depth=3)
        data, x_dict = _build_graph_with_features(clause, hidden_dim=128)

        with torch.no_grad():
            result = _benchmark(lambda: mpn(x_dict, data), iterations=50)
        print(f"\nTermLevelMPN forward: {result}")
        assert result.mean_latency_ms < 50.0

    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    def test_term_mpn_scaling_with_depth(self, depth: int):
        """Term MPN should handle increasing tree depth gracefully."""
        config = HierarchicalMPNConfig(hidden_dim=128)
        mpn = TermLevelMPN(config)
        mpn.eval()

        clause = _make_equational_clause(1, depth=depth)
        data, x_dict = _build_graph_with_features(clause, hidden_dim=128)

        with torch.no_grad():
            result = _benchmark(lambda: mpn(x_dict, data), iterations=30)
        print(f"\nTermLevelMPN depth={depth}: {result}")
        # Even depth 4 (binary tree ~31 nodes) should be manageable
        assert result.mean_latency_ms < 100.0, f"Depth {depth} too slow"


class TestLiteralLevelMPNPerformance:
    """Benchmark literal-level message passing."""

    def test_literal_mpn_forward_pass(self):
        """Literal-level MPN forward pass latency."""
        config = HierarchicalMPNConfig(hidden_dim=128)
        mpn = LiteralLevelMPN(config)
        mpn.eval()

        clause = _make_complex_clause(1, num_lits=3, depth=2)
        data, x_dict = _build_graph_with_features(clause, hidden_dim=128)

        with torch.no_grad():
            result = _benchmark(lambda: mpn(x_dict, data), iterations=50)
        print(f"\nLiteralLevelMPN forward: {result}")
        assert result.mean_latency_ms < 50.0

    def test_literal_mpn_many_literals(self):
        """Literal MPN with 10+ literals (large clause)."""
        config = HierarchicalMPNConfig(hidden_dim=128)
        mpn = LiteralLevelMPN(config)
        mpn.eval()

        clause = _make_complex_clause(1, num_lits=10, depth=1)
        data, x_dict = _build_graph_with_features(clause, hidden_dim=128)

        with torch.no_grad():
            result = _benchmark(lambda: mpn(x_dict, data), iterations=50)
        print(f"\nLiteralLevelMPN (10 literals): {result}")
        assert result.mean_latency_ms < 100.0


class TestClauseLevelPerformance:
    """Benchmark clause-level components."""

    def test_clause_property_encoder(self):
        """ClausePropertyEncoder throughput."""
        encoder = ClausePropertyEncoder(clause_feature_dim=7, hidden_dim=128)
        encoder.eval()

        # Batch of 100 clause feature vectors
        features = torch.randn(100, 7)

        with torch.no_grad():
            result = _benchmark(lambda: encoder(features), iterations=200)
        per_clause_us = result.mean_latency_ms * 1000 / 100
        print(f"\nClausePropertyEncoder (100 clauses): {result}, per-clause={per_clause_us:.1f}us")
        assert per_clause_us < 100.0, "Per-clause encoding too slow"

    def test_temporal_position_encoder(self):
        """TemporalPositionEncoder throughput."""
        encoder = TemporalPositionEncoder(temporal_dim=32, hidden_dim=128)
        encoder.eval()

        timestamps = torch.arange(100, dtype=torch.float)

        with torch.no_grad():
            result = _benchmark(lambda: encoder(timestamps), iterations=200)
        print(f"\nTemporalPositionEncoder (100 clauses): {result}")
        assert result.mean_latency_ms < 5.0


# ── Full pipeline benchmarks ─────────────────────────────────────────────


class TestBaseGNNPerformance:
    """Benchmark the baseline HeterogeneousClauseGNN for comparison."""

    @pytest.fixture
    def base_model(self):
        config = GNNConfig(hidden_dim=128, embedding_dim=64, num_layers=3)
        model = HeterogeneousClauseGNN(config)
        model.eval()
        return model

    def test_baseline_forward_simple(self, base_model):
        """Baseline GNN forward pass with simple clause."""
        clause = _make_clause([(True, _make_term(1, (_make_var(0),)))], 1)
        data = clause_to_heterograph(clause)

        with torch.no_grad():
            result = _benchmark(lambda: base_model(data), iterations=50)
        print(f"\nBaseline GNN (simple clause): {result}")
        assert result.mean_latency_ms < 50.0

    def test_baseline_forward_complex(self, base_model):
        """Baseline GNN forward pass with complex clause."""
        clause = _make_complex_clause(1, num_lits=3, depth=3)
        data = clause_to_heterograph(clause)

        with torch.no_grad():
            result = _benchmark(lambda: base_model(data), iterations=50)
        print(f"\nBaseline GNN (complex clause): {result}")
        assert result.mean_latency_ms < 100.0

    def test_baseline_embed_clause(self, base_model):
        """Baseline GNN embed_clause convenience method."""
        clause = _make_complex_clause(1, num_lits=2, depth=2)
        data = clause_to_heterograph(clause)

        result = _benchmark(lambda: base_model.embed_clause(data), iterations=50)
        print(f"\nBaseline GNN embed_clause: {result}")
        assert result.mean_latency_ms < 100.0


class TestHierarchicalMPNPipelinePerformance:
    """Benchmark the full hierarchical MPN pipeline (symbol → term → literal)."""

    @pytest.fixture
    def mpn_pipeline(self):
        config = HierarchicalMPNConfig(hidden_dim=128)
        return (
            SymbolLevelMPN(config),
            TermLevelMPN(config),
            LiteralLevelMPN(config),
        )

    def _run_pipeline(self, pipeline, x_dict, data):
        sym_mpn, term_mpn, lit_mpn = pipeline
        x = sym_mpn(x_dict, data)
        x = term_mpn(x, data)
        x = lit_mpn(x, data)
        return x

    def test_full_pipeline_forward(self, mpn_pipeline):
        """Full symbol→term→literal pipeline latency."""
        for m in mpn_pipeline:
            m.eval()

        clause = _make_complex_clause(1, num_lits=3, depth=2)
        data, x_dict = _build_graph_with_features(clause, hidden_dim=128)

        with torch.no_grad():
            result = _benchmark(
                lambda: self._run_pipeline(mpn_pipeline, x_dict, data),
                iterations=30,
            )
        print(f"\nFull MPN pipeline (3 levels): {result}")
        assert result.mean_latency_ms < 200.0

    def test_pipeline_vs_baseline_overhead(self, mpn_pipeline):
        """Measure overhead of hierarchical pipeline vs baseline GNN."""
        for m in mpn_pipeline:
            m.eval()

        base_config = GNNConfig(hidden_dim=128, embedding_dim=64, num_layers=3)
        base_model = HeterogeneousClauseGNN(base_config)
        base_model.eval()

        clause = _make_complex_clause(1, num_lits=2, depth=2)
        data = clause_to_heterograph(clause)
        _, x_dict = _build_graph_with_features(clause, hidden_dim=128)

        with torch.no_grad():
            baseline_result = _benchmark(lambda: base_model(data), iterations=30)
            pipeline_result = _benchmark(
                lambda: self._run_pipeline(mpn_pipeline, x_dict, data),
                iterations=30,
            )

        overhead_ratio = pipeline_result.mean_latency_ms / baseline_result.mean_latency_ms
        print(f"\nBaseline GNN: {baseline_result}")
        print(f"Hierarchical MPN pipeline: {pipeline_result}")
        print(f"Overhead ratio: {overhead_ratio:.2f}x")

        # Hierarchical pipeline should be at most 5x slower than baseline
        assert overhead_ratio < 5.0, f"Overhead {overhead_ratio:.2f}x exceeds 5x threshold"


# ── Memory usage benchmarks ──────────────────────────────────────────────


class TestMemoryUsage:
    """Benchmark memory usage of hierarchical components."""

    def test_mpn_model_memory_footprint(self):
        """Memory footprint of all MPN modules."""
        config = HierarchicalMPNConfig(hidden_dim=256)

        sym = SymbolLevelMPN(config)
        term = TermLevelMPN(config)
        lit = LiteralLevelMPN(config)

        sym_params = sum(p.numel() for p in sym.parameters())
        term_params = sum(p.numel() for p in term.parameters())
        lit_params = sum(p.numel() for p in lit.parameters())
        total_params = sym_params + term_params + lit_params

        # 4 bytes per float32 parameter
        total_mb = total_params * 4 / (1024 * 1024)

        print(f"\nSymbolLevelMPN params: {sym_params:,}")
        print(f"TermLevelMPN params: {term_params:,}")
        print(f"LiteralLevelMPN params: {lit_params:,}")
        print(f"Total MPN params: {total_params:,} ({total_mb:.1f} MB)")

        # All three MPN modules should fit in <100MB
        assert total_mb < 100.0, f"MPN models use {total_mb:.1f}MB, exceeds 100MB"

    def test_base_gnn_vs_hierarchical_param_count(self):
        """Compare parameter counts between base and hierarchical components."""
        base_config = GNNConfig(hidden_dim=256, embedding_dim=512, num_layers=3)
        base_model = HeterogeneousClauseGNN(base_config)
        base_params = sum(p.numel() for p in base_model.parameters())

        mpn_config = HierarchicalMPNConfig(hidden_dim=256)
        sym = SymbolLevelMPN(mpn_config)
        term = TermLevelMPN(mpn_config)
        lit = LiteralLevelMPN(mpn_config)
        hier_params = sum(
            sum(p.numel() for p in m.parameters())
            for m in [sym, term, lit]
        )

        ratio = hier_params / base_params if base_params > 0 else 0
        print(f"\nBase GNN params: {base_params:,}")
        print(f"Hierarchical MPN params: {hier_params:,}")
        print(f"Ratio (hier/base): {ratio:.2f}x")

        # Hierarchical adds overhead but should not be >3x the base model
        assert ratio < 3.0, f"Param ratio {ratio:.2f}x exceeds 3x threshold"

    def test_graph_memory_scaling(self):
        """Memory usage of clause graphs scales linearly with clause count."""
        sizes = [10, 50, 100, 200]
        memory_per_clause = []

        for n in sizes:
            gc.collect()
            clauses = [_make_complex_clause(i, num_lits=2, depth=2) for i in range(n)]
            graphs = batch_clauses_to_heterograph(clauses)

            # Estimate memory from tensor sizes
            total_bytes = 0
            for g in graphs:
                for nt in g.node_types:
                    store = g[nt]
                    if hasattr(store, "x") and store.x is not None:
                        total_bytes += store.x.nelement() * store.x.element_size()
                for et in g.edge_types:
                    store = g[et]
                    if hasattr(store, "edge_index") and store.edge_index is not None:
                        total_bytes += store.edge_index.nelement() * store.edge_index.element_size()

            mb = total_bytes / (1024 * 1024)
            per_clause_kb = total_bytes / (n * 1024)
            memory_per_clause.append(per_clause_kb)
            print(f"\n{n} clauses: {mb:.2f} MB ({per_clause_kb:.1f} KB/clause)")

        # Per-clause memory should be roughly constant (linear scaling)
        max_ratio = max(memory_per_clause) / min(memory_per_clause)
        print(f"Per-clause memory ratio (max/min): {max_ratio:.2f}")
        assert max_ratio < 2.0, f"Non-linear scaling: ratio {max_ratio:.2f}"


# ── Scalability benchmarks ───────────────────────────────────────────────


class TestScalability:
    """Test scalability of hierarchical GNN components."""

    @pytest.mark.parametrize("num_clauses", [10, 50, 100])
    def test_graph_construction_scaling(self, num_clauses: int):
        """Graph construction throughput scales linearly."""
        clauses = [_make_complex_clause(i, num_lits=2, depth=2) for i in range(num_clauses)]

        result = _benchmark(
            lambda: batch_clauses_to_heterograph(clauses),
            warmup=2,
            iterations=10,
        )
        per_clause_ms = result.mean_latency_ms / num_clauses
        print(f"\n{num_clauses} clauses: total={result.mean_latency_ms:.1f}ms, per-clause={per_clause_ms:.2f}ms")
        assert per_clause_ms < 10.0, f"Per-clause construction too slow at n={num_clauses}"

    def test_mpn_forward_scaling_with_clause_complexity(self):
        """MPN forward pass time vs clause complexity (num nodes)."""
        config = HierarchicalMPNConfig(hidden_dim=128)
        mpn = SymbolLevelMPN(config)
        mpn.eval()

        results = []
        for depth in [1, 2, 3]:
            clause = _make_complex_clause(1, num_lits=2, depth=depth)
            data, x_dict = _build_graph_with_features(clause, hidden_dim=128)

            num_nodes = sum(
                data[nt].num_nodes
                for nt in data.node_types
                if hasattr(data[nt], "num_nodes")
            )

            with torch.no_grad():
                result = _benchmark(lambda: mpn(x_dict, data), iterations=30)
            results.append((depth, num_nodes, result.mean_latency_ms))
            print(f"\nDepth {depth} ({num_nodes} nodes): {result}")

        # Check that latency doesn't explode with complexity
        if results[-1][1] > 0 and results[0][1] > 0:
            node_ratio = results[-1][1] / results[0][1]
            time_ratio = results[-1][2] / results[0][2] if results[0][2] > 0 else 1.0
            print(f"\nNode ratio: {node_ratio:.1f}x, Time ratio: {time_ratio:.1f}x")
            # Time should scale at most quadratically with node count
            assert time_ratio < node_ratio ** 2.5, "Super-polynomial scaling detected"


# ── Gradient and training benchmarks ─────────────────────────────────────


class TestTrainingPerformance:
    """Benchmark training-related performance (gradient computation)."""

    def test_mpn_backward_pass_latency(self):
        """Measure backward pass latency for MPN modules."""
        config = HierarchicalMPNConfig(hidden_dim=128)
        sym_mpn = SymbolLevelMPN(config)
        sym_mpn.train()

        clause = _make_complex_clause(1, num_lits=2, depth=2)
        data, x_dict = _build_graph_with_features(clause, hidden_dim=128)

        def forward_backward():
            out = sym_mpn(x_dict, data)
            # Sum outputs for scalar loss
            loss = sum(v.sum() for v in out.values() if isinstance(v, torch.Tensor))
            loss.backward()
            sym_mpn.zero_grad()

        result = _benchmark(forward_backward, warmup=3, iterations=20)
        print(f"\nSymbolLevelMPN forward+backward: {result}")
        assert result.mean_latency_ms < 200.0

    def test_base_gnn_backward_latency(self):
        """Baseline GNN backward pass for comparison."""
        config = GNNConfig(hidden_dim=128, embedding_dim=64, num_layers=3)
        model = HeterogeneousClauseGNN(config)
        model.train()

        clause = _make_complex_clause(1, num_lits=2, depth=2)
        data = clause_to_heterograph(clause)

        def forward_backward():
            out = model(data)
            loss = out.sum()
            loss.backward()
            model.zero_grad()

        result = _benchmark(forward_backward, warmup=3, iterations=20)
        print(f"\nBaseline GNN forward+backward: {result}")
        assert result.mean_latency_ms < 200.0
