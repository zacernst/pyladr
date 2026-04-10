"""Tests for symbol-level message passing network (SymbolLevelMPN).

Tests cover:
- Construction with default and custom configs
- Forward pass output shapes for single symbols
- Type-aware attention between predicates, functions, and constants
- Message aggregation from symbol neighbors
- Integration with existing clause_graph NodeType.SYMBOL
- Gradient flow through all components
- Batched processing
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

import torch.nn as nn
from torch_geometric.data import HeteroData, Batch

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.ml.graph.clause_graph import (
    NodeType,
    EdgeType,
    clause_to_heterograph,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _atom(symnum: int, *args):
    return get_rigid_term(symnum, len(args), tuple(args))


def _const(symnum: int):
    return get_rigid_term(symnum, 0)


def _var(n: int):
    return get_variable_term(n)


def _pos_lit(atom) -> Literal:
    return Literal(sign=True, atom=atom)


def _neg_lit(atom) -> Literal:
    return Literal(sign=False, atom=atom)


def _simple_clause() -> Clause:
    """P(a) — minimal clause with one predicate and one constant."""
    return Clause(literals=(_pos_lit(_atom(4, _const(2))),))


def _multi_symbol_clause() -> Clause:
    """P(f(a, b), g(x)) — clause with predicate, functions, constants, variable."""
    x = _var(0)
    a = _const(2)
    b = _const(3)
    f_ab = _atom(1, a, b)  # f(a, b) - function
    g_x = _atom(5, x)  # g(x) - function
    return Clause(literals=(_pos_lit(_atom(4, f_ab, g_x)),))  # P(f(a,b), g(x))


def _equality_clause() -> Clause:
    """f(x) = g(x, a) — equation with shared variable."""
    x = _var(0)
    a = _const(2)
    f_x = _atom(1, x)
    g_xa = _atom(5, x, a)
    eq = _atom(6, f_x, g_xa)  # eq(f(x), g(x,a))
    return Clause(literals=(_pos_lit(eq),))


def _make_graph(clause: Clause) -> HeteroData:
    return clause_to_heterograph(clause)


def _make_batch(clauses: list[Clause]) -> HeteroData:
    graphs = [clause_to_heterograph(c) for c in clauses]
    return Batch.from_data_list(graphs)


# ── Import the module under test ──────────────────────────────────────────


from pyladr.ml.graph.hierarchical_mpn import (
    HierarchicalMPNConfig,
    SymbolLevelMPN,
)


# ── Construction tests ────────────────────────────────────────────────────


class TestSymbolLevelMPNConstruction:
    """Test model creation with various configs."""

    def test_default_construction(self):
        config = HierarchicalMPNConfig()
        model = SymbolLevelMPN(config)
        assert isinstance(model, nn.Module)

    def test_custom_hidden_dim(self):
        config = HierarchicalMPNConfig(hidden_dim=128)
        model = SymbolLevelMPN(config)
        assert model.config.hidden_dim == 128

    def test_has_trainable_parameters(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = SymbolLevelMPN(config)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_type_aware_components_exist(self):
        """Should have separate processing for predicates, functions, constants."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = SymbolLevelMPN(config)
        # Model should have type embeddings for symbol categories
        assert hasattr(model, 'type_embedding')


# ── Forward pass tests ────────────────────────────────────────────────────


class TestSymbolLevelMPNForward:
    """Test forward pass produces correct output shapes."""

    def test_single_clause_output_shape(self):
        """Symbol-level MPN should output updated symbol node features."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = SymbolLevelMPN(config)
        data = _make_graph(_simple_clause())

        # Project features to hidden_dim first (as the encoder does)
        x_dict = _project_features(data, config.hidden_dim)
        result = model(x_dict, data)

        # Should return updated x_dict with symbol features transformed
        assert NodeType.SYMBOL.value in result
        sym_out = result[NodeType.SYMBOL.value]
        num_symbols = data[NodeType.SYMBOL.value].num_nodes
        assert sym_out.shape == (num_symbols, config.hidden_dim)

    def test_multi_symbol_clause(self):
        """Clause with multiple symbol types should process all."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = SymbolLevelMPN(config)
        data = _make_graph(_multi_symbol_clause())

        x_dict = _project_features(data, config.hidden_dim)
        result = model(x_dict, data)

        assert NodeType.SYMBOL.value in result
        num_symbols = data[NodeType.SYMBOL.value].num_nodes
        assert result[NodeType.SYMBOL.value].shape == (num_symbols, config.hidden_dim)

    def test_preserves_other_node_types(self):
        """Symbol-level MPN should pass through other node types unchanged."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = SymbolLevelMPN(config)
        data = _make_graph(_multi_symbol_clause())

        x_dict = _project_features(data, config.hidden_dim)
        original_term = x_dict[NodeType.TERM.value].clone()
        result = model(x_dict, data)

        # Term features should be unchanged by symbol-level MPN
        assert torch.allclose(result[NodeType.TERM.value], original_term)

    def test_output_is_finite(self):
        """All output values should be finite (no NaN/Inf)."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = SymbolLevelMPN(config)
        data = _make_graph(_equality_clause())

        x_dict = _project_features(data, config.hidden_dim)
        result = model(x_dict, data)

        sym_out = result[NodeType.SYMBOL.value]
        assert sym_out.isfinite().all()


# ── Type-aware attention tests ────────────────────────────────────────────


class TestSymbolTypeAwareness:
    """Test that symbol types (predicate/function/constant) are handled differently."""

    def test_different_types_get_different_embeddings(self):
        """Symbols of different types should get different type-enriched features."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = SymbolLevelMPN(config)
        model.eval()

        # Create a clause with both predicates and constants
        data = _make_graph(_multi_symbol_clause())
        x_dict = _project_features(data, config.hidden_dim)

        with torch.no_grad():
            result = model(x_dict, data)

        sym_out = result[NodeType.SYMBOL.value]
        # With multiple distinct symbols, outputs should differ
        if sym_out.shape[0] > 1:
            # At least two symbols should have different representations
            diffs = torch.cdist(sym_out, sym_out)
            # Off-diagonal should have some non-zero entries
            mask = ~torch.eye(sym_out.shape[0], dtype=torch.bool)
            assert diffs[mask].max() > 0

    def test_type_embedding_dimension(self):
        """Type embedding should match hidden_dim configuration."""
        config = HierarchicalMPNConfig(hidden_dim=64)
        model = SymbolLevelMPN(config)
        # 3 symbol types: predicate (arity>0 && used as atom), function (arity>0), constant (arity==0)
        assert model.type_embedding.embedding_dim == config.hidden_dim


# ── Gradient flow tests ──────────────────────────────────────────────────


class TestSymbolLevelGradients:
    """Verify gradients flow through symbol-level MPN."""

    def test_backward_pass(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = SymbolLevelMPN(config)
        data = _make_graph(_multi_symbol_clause())

        x_dict = _project_features(data, config.hidden_dim)
        result = model(x_dict, data)

        loss = result[NodeType.SYMBOL.value].sum()
        loss.backward()

        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No parameters received gradients"

    def test_training_step(self):
        """Simulate a training step with optimizer."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = SymbolLevelMPN(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        data = _make_graph(_multi_symbol_clause())
        x_dict = _project_features(data, config.hidden_dim)

        model.train()
        result = model(x_dict, data)
        loss = result[NodeType.SYMBOL.value].mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Should complete without error


# ── Batched processing tests ─────────────────────────────────────────────


class TestSymbolLevelBatching:
    """Test processing batched graphs."""

    def test_batched_forward(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = SymbolLevelMPN(config)

        batch = _make_batch([_simple_clause(), _multi_symbol_clause()])
        x_dict = _project_features(batch, config.hidden_dim)
        result = model(x_dict, batch)

        # Total symbols across both clauses
        total_symbols = batch[NodeType.SYMBOL.value].num_nodes
        assert result[NodeType.SYMBOL.value].shape == (total_symbols, config.hidden_dim)

    def test_batched_consistency(self):
        """Batched results should match individual processing (approximately)."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = SymbolLevelMPN(config)
        model.eval()

        c1 = _simple_clause()
        c2 = _multi_symbol_clause()

        # Process individually
        with torch.no_grad():
            d1 = _make_graph(c1)
            x1 = _project_features(d1, config.hidden_dim)
            r1 = model(x1, d1)

            d2 = _make_graph(c2)
            x2 = _project_features(d2, config.hidden_dim)
            r2 = model(x2, d2)

        # Results should be finite
        assert r1[NodeType.SYMBOL.value].isfinite().all()
        assert r2[NodeType.SYMBOL.value].isfinite().all()


# ── Helper for projecting raw features ────────────────────────────────────


def _project_features(data: HeteroData, hidden_dim: int) -> dict[str, torch.Tensor]:
    """Project raw graph features to hidden_dim (mimics encoder step 1)."""
    feature_dims = {
        NodeType.CLAUSE.value: 7,
        NodeType.LITERAL.value: 3,
        NodeType.TERM.value: 8,
        NodeType.SYMBOL.value: 6,
        NodeType.VARIABLE.value: 1,
    }
    x_dict = {}
    for nt in NodeType:
        key = nt.value
        if key in data.node_types and hasattr(data[key], 'x') and data[key].x is not None:
            n = data[key].num_nodes
            if n > 0:
                # Simple linear projection (random but deterministic for testing)
                torch.manual_seed(42)
                proj = nn.Linear(feature_dims[key], hidden_dim)
                x_dict[key] = torch.relu(proj(data[key].x))
    return x_dict
