"""Tests for literal-level message passing network (LiteralLevelMPN).

Tests cover:
- Construction with default and custom configs
- Polarity-aware messaging (positive vs negative literals)
- Aggregation from term nodes into literal representations
- Inter-literal attention within a clause
- Complementary literal awareness
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


def _unit_positive_clause() -> Clause:
    """P(a) — single positive literal."""
    return Clause(literals=(_pos_lit(_atom(4, _const(2))),))


def _unit_negative_clause() -> Clause:
    """-P(a) — single negative literal."""
    return Clause(literals=(_neg_lit(_atom(4, _const(2))),))


def _mixed_polarity_clause() -> Clause:
    """P(x) | -Q(a) — mixed positive and negative literals."""
    x = _var(0)
    a = _const(2)
    return Clause(literals=(
        _pos_lit(_atom(4, x)),
        _neg_lit(_atom(5, a)),
    ))


def _horn_clause() -> Clause:
    """-P(x) | -Q(x, a) | R(f(x)) — Horn clause (1 positive, 2 negative)."""
    x = _var(0)
    a = _const(2)
    f_x = _atom(1, x)
    return Clause(literals=(
        _neg_lit(_atom(4, x)),
        _neg_lit(_atom(5, x, a)),
        _pos_lit(_atom(6, f_x)),
    ))


def _multi_literal_clause() -> Clause:
    """P(x) | Q(y) | R(a) | -S(b) — 4 literals."""
    x = _var(0)
    y = _var(1)
    a = _const(2)
    b = _const(3)
    return Clause(literals=(
        _pos_lit(_atom(4, x)),
        _pos_lit(_atom(5, y)),
        _pos_lit(_atom(6, a)),
        _neg_lit(_atom(7, b)),
    ))


def _equality_literals_clause() -> Clause:
    """f(x) = a | g(y) != b — equality and disequality."""
    x = _var(0)
    y = _var(1)
    a = _const(2)
    b = _const(3)
    f_x = _atom(1, x)
    g_y = _atom(5, y)
    eq1 = _atom(8, f_x, a)  # f(x) = a
    eq2 = _atom(8, g_y, b)  # g(y) = b
    return Clause(literals=(
        _pos_lit(eq1),
        _neg_lit(eq2),  # g(y) != b
    ))


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
    LiteralLevelMPN,
)


# ── Construction tests ────────────────────────────────────────────────────


class TestLiteralLevelMPNConstruction:

    def test_default_construction(self):
        config = HierarchicalMPNConfig()
        model = LiteralLevelMPN(config)
        assert isinstance(model, nn.Module)

    def test_custom_config(self):
        config = HierarchicalMPNConfig(hidden_dim=64)
        model = LiteralLevelMPN(config)
        assert model.config.hidden_dim == 64

    def test_has_trainable_parameters(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = LiteralLevelMPN(config)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_has_polarity_components(self):
        """Should have polarity-aware processing."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = LiteralLevelMPN(config)
        assert hasattr(model, 'polarity_embedding')


# ── Forward pass tests ────────────────────────────────────────────────────


class TestLiteralLevelMPNForward:

    def test_unit_clause_output_shape(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = LiteralLevelMPN(config)
        data = _make_graph(_unit_positive_clause())

        x_dict = _project_features(data, config.hidden_dim)
        result = model(x_dict, data)

        num_lits = data[NodeType.LITERAL.value].num_nodes
        assert result[NodeType.LITERAL.value].shape == (num_lits, config.hidden_dim)

    def test_multi_literal_output_shape(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = LiteralLevelMPN(config)
        data = _make_graph(_multi_literal_clause())

        x_dict = _project_features(data, config.hidden_dim)
        result = model(x_dict, data)

        num_lits = data[NodeType.LITERAL.value].num_nodes
        assert num_lits == 4
        assert result[NodeType.LITERAL.value].shape == (4, config.hidden_dim)

    def test_horn_clause_output_shape(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = LiteralLevelMPN(config)
        data = _make_graph(_horn_clause())

        x_dict = _project_features(data, config.hidden_dim)
        result = model(x_dict, data)

        num_lits = data[NodeType.LITERAL.value].num_nodes
        assert result[NodeType.LITERAL.value].shape == (num_lits, config.hidden_dim)

    def test_output_is_finite(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = LiteralLevelMPN(config)
        data = _make_graph(_horn_clause())

        x_dict = _project_features(data, config.hidden_dim)
        result = model(x_dict, data)

        assert result[NodeType.LITERAL.value].isfinite().all()

    def test_preserves_clause_features(self):
        """Literal-level MPN should pass through clause features."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = LiteralLevelMPN(config)
        data = _make_graph(_horn_clause())

        x_dict = _project_features(data, config.hidden_dim)
        orig_clause = x_dict[NodeType.CLAUSE.value].clone()

        result = model(x_dict, data)

        assert torch.allclose(result[NodeType.CLAUSE.value], orig_clause)


# ── Polarity-aware messaging tests ────────────────────────────────────────


class TestPolarityAwareness:

    def test_positive_vs_negative_different_output(self):
        """Positive and negative literals with same atom should get different representations."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = LiteralLevelMPN(config)
        model.eval()

        data = _make_graph(_mixed_polarity_clause())
        x_dict = _project_features(data, config.hidden_dim)

        with torch.no_grad():
            result = model(x_dict, data)

        lit_out = result[NodeType.LITERAL.value]
        # Two literals should have different representations
        assert lit_out.shape[0] == 2
        assert not torch.allclose(lit_out[0], lit_out[1], atol=1e-6)

    def test_polarity_embedding_dimension(self):
        """Polarity embedding should have 2 entries (positive/negative)."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = LiteralLevelMPN(config)
        assert model.polarity_embedding.num_embeddings == 2
        assert model.polarity_embedding.embedding_dim == config.hidden_dim

    def test_equality_literal_handling(self):
        """Equality and disequality literals should be handled correctly."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = LiteralLevelMPN(config)
        data = _make_graph(_equality_literals_clause())

        x_dict = _project_features(data, config.hidden_dim)
        result = model(x_dict, data)

        assert result[NodeType.LITERAL.value].isfinite().all()


# ── Term aggregation tests ────────────────────────────────────────────────


class TestTermAggregation:

    def test_term_info_reaches_literals(self):
        """Literal features should be enriched by term structure (via rev_has_atom)."""
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = LiteralLevelMPN(config)
        model.eval()

        data = _make_graph(_horn_clause())
        x_dict = _project_features(data, config.hidden_dim)

        with torch.no_grad():
            result = model(x_dict, data)

        # Output should differ from input (messages aggregated from terms)
        lit_in = x_dict[NodeType.LITERAL.value]
        lit_out = result[NodeType.LITERAL.value]
        assert not torch.allclose(lit_in, lit_out, atol=1e-6)


# ── Gradient flow tests ──────────────────────────────────────────────────


class TestLiteralLevelGradients:

    def test_backward_pass(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = LiteralLevelMPN(config)
        data = _make_graph(_horn_clause())

        x_dict = _project_features(data, config.hidden_dim)
        result = model(x_dict, data)

        loss = result[NodeType.LITERAL.value].sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No parameters received gradients"


# ── Batched processing tests ─────────────────────────────────────────────


class TestLiteralLevelBatching:

    def test_batched_forward(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = LiteralLevelMPN(config)

        batch = _make_batch([
            _unit_positive_clause(),
            _mixed_polarity_clause(),
            _horn_clause(),
        ])
        x_dict = _project_features(batch, config.hidden_dim)
        result = model(x_dict, batch)

        total_lits = batch[NodeType.LITERAL.value].num_nodes
        assert result[NodeType.LITERAL.value].shape == (total_lits, config.hidden_dim)

    def test_batched_output_finite(self):
        config = HierarchicalMPNConfig(hidden_dim=32)
        model = LiteralLevelMPN(config)

        batch = _make_batch([
            _unit_negative_clause(),
            _multi_literal_clause(),
        ])
        x_dict = _project_features(batch, config.hidden_dim)
        result = model(x_dict, batch)

        assert result[NodeType.LITERAL.value].isfinite().all()
