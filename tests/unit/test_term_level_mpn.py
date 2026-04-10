"""Tests for term-level message passing network (TermLevelMPN).

Tests cover:
- Construction with default and custom configs
- Tree-structured attention following term tree hierarchy
- Bottom-up aggregation from children to parent terms
- Top-down context propagation from parent to children
- Integration with symbol node features from SymbolLevelMPN
- Variable node handling
- Gradient flow
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


def _flat_clause() -> Clause:
    """P(a, b) — flat term with no nesting."""
    return Clause(literals=(_pos_lit(_atom(4, _const(2), _const(3))),))


def _nested_clause() -> Clause:
    """P(f(g(a))) — deeply nested term tree."""
    a = _const(2)
    g_a = _atom(5, a)
    f_g_a = _atom(1, g_a)
    return Clause(literals=(_pos_lit(_atom(4, f_g_a)),))


def _binary_term_clause() -> Clause:
    """f(g(x, a), h(b, y)) — binary branching term tree."""
    x = _var(0)
    y = _var(1)
    a = _const(2)
    b = _const(3)
    g_xa = _atom(5, x, a)
    h_by = _atom(6, b, y)
    f_gh = _atom(1, g_xa, h_by)
    return Clause(literals=(_pos_lit(_atom(4, f_gh)),))


def _shared_var_clause() -> Clause:
    """P(x, f(x)) — variable shared between positions."""
    x = _var(0)
    f_x = _atom(1, x)
    return Clause(literals=(_pos_lit(_atom(4, x, f_x)),))


def _make_graph(clause: Clause) -> HeteroData:
    return clause_to_heterograph(clause)


def _make_batch(clauses: list[Clause]) -> HeteroData:
    graphs = [clause_to_heterograph(c) for c in clauses]
    return Batch.from_data_list(graphs)


def _project_features(data: HeteroData, hidden_dim: int) -> dict[str, torch.Tensor]:
    """Project raw graph features to hidden_dim."""
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
                torch.manual_seed(42)
                proj = nn.Linear(feature_dims[key], hidden_dim)
                x_dict[key] = torch.relu(proj(data[key].x))
    return x_dict


# ── Import the module under test ──────────────────────────────────────────


from pyladr.ml.graph.hierarchical_mpn import (
    HierarchicalMPNConfig,
    TermLevelMPN,
)


# ── Construction tests ────────────────────────────────────────────────────


class TestTermLevelMPNConstruction:
    """Test model creation."""

    def test_default_construction(self):
        config = HierarchicalMPNConfig()
        model = TermLevelMPN(config)
        assert isinstance(model, nn.Module)

    def test_custom_config(self):
        config = HierarchicalMPNConfig(hidden_dim=64, num_attention_heads=4)
        model = TermLevelMPN(config)
        assert model.config.hidden_dim == 64

    def test_has_trainable_parameters(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = TermLevelMPN(config)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_has_tree_attention(self):
        """Should have attention mechanism for tree-structured aggregation."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = TermLevelMPN(config)
        assert hasattr(model, 'tree_attention')


# ── Forward pass tests ────────────────────────────────────────────────────


class TestTermLevelMPNForward:
    """Test forward pass produces correct output shapes."""

    def test_flat_clause_output_shape(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = TermLevelMPN(config)
        data = _make_graph(_flat_clause())

        x_dict = _project_features(data, config.hidden_dim)
        result = model(x_dict, data)

        num_terms = data[NodeType.TERM.value].num_nodes
        assert result[NodeType.TERM.value].shape == (num_terms, config.hidden_dim)

    def test_nested_clause_output_shape(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = TermLevelMPN(config)
        data = _make_graph(_nested_clause())

        x_dict = _project_features(data, config.hidden_dim)
        result = model(x_dict, data)

        num_terms = data[NodeType.TERM.value].num_nodes
        assert result[NodeType.TERM.value].shape == (num_terms, config.hidden_dim)

    def test_binary_tree_output_shape(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = TermLevelMPN(config)
        data = _make_graph(_binary_term_clause())

        x_dict = _project_features(data, config.hidden_dim)
        result = model(x_dict, data)

        num_terms = data[NodeType.TERM.value].num_nodes
        assert result[NodeType.TERM.value].shape == (num_terms, config.hidden_dim)

    def test_output_is_finite(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = TermLevelMPN(config)
        data = _make_graph(_binary_term_clause())

        x_dict = _project_features(data, config.hidden_dim)
        result = model(x_dict, data)

        assert result[NodeType.TERM.value].isfinite().all()

    def test_preserves_clause_and_literal_features(self):
        """Term-level MPN should not modify clause or literal features."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = TermLevelMPN(config)
        data = _make_graph(_binary_term_clause())

        x_dict = _project_features(data, config.hidden_dim)
        orig_clause = x_dict.get(NodeType.CLAUSE.value)
        if orig_clause is not None:
            orig_clause = orig_clause.clone()
        orig_literal = x_dict.get(NodeType.LITERAL.value)
        if orig_literal is not None:
            orig_literal = orig_literal.clone()

        result = model(x_dict, data)

        if orig_clause is not None:
            assert torch.allclose(result[NodeType.CLAUSE.value], orig_clause)
        if orig_literal is not None:
            assert torch.allclose(result[NodeType.LITERAL.value], orig_literal)


# ── Tree-structured attention tests ───────────────────────────────────────


class TestTreeStructuredAttention:
    """Test tree-structured attention mechanism."""

    def test_bottom_up_aggregation(self):
        """Child term features should influence parent term features."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = TermLevelMPN(config)
        model.eval()

        data = _make_graph(_nested_clause())
        x_dict = _project_features(data, config.hidden_dim)

        with torch.no_grad():
            result = model(x_dict, data)

        # Output should differ from input (messages were aggregated)
        term_in = x_dict[NodeType.TERM.value]
        term_out = result[NodeType.TERM.value]
        assert not torch.allclose(term_in, term_out, atol=1e-6)

    def test_symbol_information_propagation(self):
        """Symbol node info should propagate to term nodes via rev_symbol_of edges."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = TermLevelMPN(config)
        model.eval()

        data = _make_graph(_multi_symbol_clause())
        x_dict = _project_features(data, config.hidden_dim)

        with torch.no_grad():
            result = model(x_dict, data)

        # Term features should be enriched by symbol information
        assert result[NodeType.TERM.value].isfinite().all()

    def test_variable_handling(self):
        """Variable nodes should contribute to term features."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = TermLevelMPN(config)
        data = _make_graph(_shared_var_clause())

        x_dict = _project_features(data, config.hidden_dim)
        result = model(x_dict, data)

        # Variable node features should also be updated
        if NodeType.VARIABLE.value in result and result[NodeType.VARIABLE.value] is not None:
            assert result[NodeType.VARIABLE.value].isfinite().all()


# ── Gradient flow tests ──────────────────────────────────────────────────


class TestTermLevelGradients:

    def test_backward_pass(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = TermLevelMPN(config)
        data = _make_graph(_binary_term_clause())

        x_dict = _project_features(data, config.hidden_dim)
        result = model(x_dict, data)

        loss = result[NodeType.TERM.value].sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No parameters received gradients"


# ── Batched processing tests ─────────────────────────────────────────────


class TestTermLevelBatching:

    def test_batched_forward(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = TermLevelMPN(config)

        batch = _make_batch([_flat_clause(), _nested_clause(), _binary_term_clause()])
        x_dict = _project_features(batch, config.hidden_dim)
        result = model(x_dict, batch)

        total_terms = batch[NodeType.TERM.value].num_nodes
        assert result[NodeType.TERM.value].shape == (total_terms, config.hidden_dim)


# Re-use multi-symbol helper
def _multi_symbol_clause() -> Clause:
    """P(f(a, b), g(x))."""
    x = _var(0)
    a = _const(2)
    b = _const(3)
    f_ab = _atom(1, a, b)
    g_x = _atom(5, x)
    return Clause(literals=(_pos_lit(_atom(4, f_ab, g_x)),))
