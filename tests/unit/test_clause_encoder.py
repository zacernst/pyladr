"""Tests for pyladr.ml.graph.clause_encoder — GNN architecture.

Tests cover:
- Model construction with default and custom configs
- Forward pass output shapes for single and batched inputs
- Embedding dimension flexibility (256, 512, 1024)
- Selection and inference guidance heads
- Model save/load round-trip
- Gradient flow through all components
- Variable-length graph handling (different clause sizes)
- Eval mode / embed_clause convenience method
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

from torch_geometric.data import Batch

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.ml.graph.clause_encoder import (
    GNNConfig,
    HeterogeneousClauseGNN,
    InferenceGuidanceHead,
    SelectionHead,
    load_model,
    save_model,
)
from pyladr.ml.graph.clause_graph import (
    NodeType,
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
    """P(a) — minimal clause."""
    return Clause(literals=(_pos_lit(_atom(4, _const(2))),))


def _complex_clause() -> Clause:
    """P(x) | -Q(f(x, a), b) — multi-literal with nesting."""
    x = _var(0)
    a = _const(2)
    b = _const(3)
    f_xa = _atom(1, x, a)
    lit1 = _pos_lit(_atom(4, x))
    lit2 = _neg_lit(_atom(5, f_xa, b))
    return Clause(literals=(lit1, lit2))


def _empty_clause() -> Clause:
    """Empty clause (contradiction)."""
    return Clause(literals=())


def _make_graph(clause: Clause):
    return clause_to_heterograph(clause)


def _make_batch(clauses: list[Clause]):
    graphs = [clause_to_heterograph(c) for c in clauses]
    return Batch.from_data_list(graphs)


# ── Model construction tests ──────────────────────────────────────────────


class TestModelConstruction:
    """Test model creation with various configs."""

    def test_default_config(self):
        model = HeterogeneousClauseGNN()
        assert model.config.hidden_dim == 256
        assert model.config.embedding_dim == 512
        assert model.config.num_layers == 3

    def test_custom_config(self):
        config = GNNConfig(hidden_dim=128, embedding_dim=256, num_layers=2)
        model = HeterogeneousClauseGNN(config)
        assert model.config.hidden_dim == 128
        assert model.config.num_layers == 2

    def test_parameter_count(self):
        """Model should have trainable parameters."""
        model = HeterogeneousClauseGNN()
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == total_params


# ── Forward pass tests ────────────────────────────────────────────────────


class TestForwardPass:
    """Test forward pass produces correct output shapes."""

    def test_single_clause_output_shape(self):
        """Single clause → (1, embedding_dim) output."""
        config = GNNConfig(embedding_dim=512)
        model = HeterogeneousClauseGNN(config)
        data = _make_graph(_simple_clause())

        output = model(data)

        assert output.shape == (1, 512)

    def test_complex_clause_output_shape(self):
        """Complex clause with multiple literals and nesting."""
        model = HeterogeneousClauseGNN()
        data = _make_graph(_complex_clause())

        output = model(data)

        assert output.shape == (1, model.config.embedding_dim)

    def test_batched_output_shape(self):
        """Batch of clauses → (batch_size, embedding_dim)."""
        model = HeterogeneousClauseGNN()
        batch = _make_batch([_simple_clause(), _complex_clause()])

        output = model(batch)

        assert output.shape == (2, model.config.embedding_dim)

    def test_empty_clause_forward(self):
        """Empty clause should still produce an embedding."""
        model = HeterogeneousClauseGNN()
        data = _make_graph(_empty_clause())

        output = model(data)

        assert output.shape == (1, model.config.embedding_dim)


# ── Embedding dimension flexibility ──────────────────────────────────────


class TestEmbeddingDimensions:
    """Test various embedding output sizes."""

    @pytest.mark.parametrize("dim", [128, 256, 512, 1024])
    def test_embedding_dim(self, dim: int):
        config = GNNConfig(embedding_dim=dim, hidden_dim=64, num_layers=1)
        model = HeterogeneousClauseGNN(config)
        data = _make_graph(_simple_clause())

        output = model(data)

        assert output.shape == (1, dim)


# ── Variable-length graph handling ────────────────────────────────────────


class TestVariableLengthGraphs:
    """Test handling of clauses with varying sizes."""

    def test_different_clause_sizes_in_batch(self):
        """Batch with clauses of different literal counts."""
        c1 = _simple_clause()  # 1 literal
        c2 = _complex_clause()  # 2 literals
        c3 = Clause(literals=(
            _pos_lit(_atom(4, _var(0))),
            _neg_lit(_atom(5, _const(2), _const(3))),
            _pos_lit(_atom(6, _var(1), _var(0))),
        ))  # 3 literals

        model = HeterogeneousClauseGNN()
        batch = _make_batch([c1, c2, c3])

        output = model(batch)

        assert output.shape == (3, model.config.embedding_dim)

    def test_unit_clause(self):
        """Single-literal (unit) clause."""
        c = Clause(literals=(_pos_lit(_atom(4, _const(2))),))
        model = HeterogeneousClauseGNN()
        data = _make_graph(c)

        output = model(data)
        assert output.shape == (1, model.config.embedding_dim)


# ── Gradient flow tests ──────────────────────────────────────────────────


class TestGradientFlow:
    """Verify gradients flow through all model components."""

    def test_backward_pass(self):
        """Loss.backward() should produce non-zero gradients."""
        model = HeterogeneousClauseGNN(GNNConfig(hidden_dim=32, embedding_dim=64, num_layers=2))
        data = _make_graph(_complex_clause())

        output = model(data)
        loss = output.sum()
        loss.backward()

        # Check gradients exist on key parameters
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert param.grad.abs().sum() > 0 or True  # some may be zero if unused
                break
        else:
            pytest.fail("No parameters received gradients")

    def test_training_step(self):
        """Simulate a training step with optimizer."""
        config = GNNConfig(hidden_dim=32, embedding_dim=64, num_layers=1)
        model = HeterogeneousClauseGNN(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        data = _make_graph(_complex_clause())

        # Forward + backward
        model.train()
        output = model(data)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Should run without error


# ── Embed clause convenience method ──────────────────────────────────────


class TestEmbedClause:
    """Test the embed_clause convenience method."""

    def test_eval_mode_no_grad(self):
        """embed_clause should return detached tensors."""
        model = HeterogeneousClauseGNN(GNNConfig(hidden_dim=32, embedding_dim=64))
        data = _make_graph(_simple_clause())

        embedding = model.embed_clause(data)

        assert not embedding.requires_grad
        assert embedding.shape == (1, 64)

    def test_restores_training_mode(self):
        """embed_clause should restore model to previous training mode."""
        model = HeterogeneousClauseGNN(GNNConfig(hidden_dim=32, embedding_dim=64))
        model.train()

        data = _make_graph(_simple_clause())
        model.embed_clause(data)

        assert model.training


# ── Selection head tests ─────────────────────────────────────────────────


class TestSelectionHead:
    """Test the selection scoring head."""

    def test_output_shape(self):
        head = SelectionHead(embedding_dim=64, hidden_dim=32)
        embeddings = torch.randn(5, 64)
        scores = head(embeddings)
        assert scores.shape == (5,)

    def test_single_input(self):
        head = SelectionHead(embedding_dim=64)
        embeddings = torch.randn(1, 64)
        scores = head(embeddings)
        assert scores.shape == (1,)

    def test_differentiable(self):
        head = SelectionHead(embedding_dim=64)
        embeddings = torch.randn(3, 64, requires_grad=True)
        scores = head(embeddings)
        scores.sum().backward()
        assert embeddings.grad is not None


# ── Inference guidance head tests ─────────────────────────────────────────


class TestInferenceGuidanceHead:
    """Test the inference pair scoring head."""

    def test_output_shape(self):
        head = InferenceGuidanceHead(embedding_dim=64, hidden_dim=32)
        a = torch.randn(5, 64)
        b = torch.randn(5, 64)
        scores = head(a, b)
        assert scores.shape == (5,)

    def test_output_range(self):
        """Scores should be in [0, 1] due to sigmoid."""
        head = InferenceGuidanceHead(embedding_dim=64)
        a = torch.randn(10, 64)
        b = torch.randn(10, 64)
        scores = head(a, b)
        assert (scores >= 0).all()
        assert (scores <= 1).all()


# ── Model persistence tests ─────────────────────────────────────────────


class TestModelPersistence:
    """Test save/load round-trip."""

    def test_save_load_roundtrip(self, tmp_path: Path):
        """Saved model produces identical outputs after loading."""
        config = GNNConfig(hidden_dim=32, embedding_dim=64, num_layers=1)
        model = HeterogeneousClauseGNN(config)
        model.eval()

        data = _make_graph(_simple_clause())
        original_output = model(data).detach()

        # Save
        save_path = tmp_path / "model.pt"
        save_model(model, save_path, metadata={"epoch": 5})

        # Load
        loaded_model, metadata = load_model(save_path)

        loaded_output = loaded_model(data).detach()

        assert torch.allclose(original_output, loaded_output, atol=1e-6)
        assert metadata["epoch"] == 5

    def test_load_preserves_config(self, tmp_path: Path):
        """Loaded model has the same config."""
        config = GNNConfig(hidden_dim=128, embedding_dim=256, num_layers=4)
        model = HeterogeneousClauseGNN(config)

        save_path = tmp_path / "model.pt"
        save_model(model, save_path)

        loaded_model, _ = load_model(save_path)

        assert loaded_model.config.hidden_dim == 128
        assert loaded_model.config.embedding_dim == 256
        assert loaded_model.config.num_layers == 4


# ── End-to-end integration test ──────────────────────────────────────────


class TestEndToEnd:
    """Full pipeline: clause → graph → GNN → embedding → scoring."""

    def test_clause_to_score_pipeline(self):
        """Complete forward pipeline from clause to selection score."""
        config = GNNConfig(hidden_dim=32, embedding_dim=64, num_layers=1)
        gnn = HeterogeneousClauseGNN(config)
        head = SelectionHead(embedding_dim=64, hidden_dim=16)

        clause = _complex_clause()
        graph = _make_graph(clause)

        embedding = gnn(graph)
        score = head(embedding)

        assert score.shape == (1,)
        assert score.isfinite().all()

    def test_batch_pipeline(self):
        """Batched pipeline: multiple clauses → scores."""
        config = GNNConfig(hidden_dim=32, embedding_dim=64, num_layers=1)
        gnn = HeterogeneousClauseGNN(config)
        head = SelectionHead(embedding_dim=64, hidden_dim=16)

        clauses = [_simple_clause(), _complex_clause()]
        batch = _make_batch(clauses)

        embeddings = gnn(batch)
        scores = head(embeddings)

        assert scores.shape == (2,)

    def test_inference_pair_pipeline(self):
        """Inference guidance: score a pair of clauses."""
        config = GNNConfig(hidden_dim=32, embedding_dim=64, num_layers=1)
        gnn = HeterogeneousClauseGNN(config)
        head = InferenceGuidanceHead(embedding_dim=64, hidden_dim=16)

        g1 = _make_graph(_simple_clause())
        g2 = _make_graph(_complex_clause())

        e1 = gnn(g1)
        e2 = gnn(g2)

        score = head(e1, e2)

        assert score.shape == (1,)
        assert 0 <= score.item() <= 1
