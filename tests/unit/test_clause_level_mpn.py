"""Tests for pyladr.ml.graph.clause_level_mpn — Clause-level message passing.

Tests cover:
- ClauseLevelMPN construction with default and custom configs
- Clause structure attention (multi-head attention over literal representations)
- Inference rule potential messaging between clause pairs
- Literal-to-clause composition (aggregating literal embeddings)
- Hierarchical updates from literal level
- Clause property feature encoding
- Integration with existing PyLADR Clause data structures
- Batch processing of multiple clauses
- Gradient flow through all components
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.term import get_rigid_term, get_variable_term


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


def _unit_clause(id_: int = 1) -> Clause:
    """P(a) — single literal clause."""
    c = Clause(literals=(_pos_lit(_atom(4, _const(2))),), id=id_)
    return c


def _binary_clause(id_: int = 2) -> Clause:
    """P(x) | -Q(f(x, a), b) — two literals with nesting."""
    x = _var(0)
    a = _const(2)
    b = _const(3)
    f_xa = _atom(1, x, a)
    lit1 = _pos_lit(_atom(4, x))
    lit2 = _neg_lit(_atom(5, f_xa, b))
    return Clause(literals=(lit1, lit2), id=id_)


def _ternary_clause(id_: int = 3) -> Clause:
    """P(x) | -Q(a, b) | R(x, y) — three literals."""
    x, y = _var(0), _var(1)
    a, b = _const(2), _const(3)
    lit1 = _pos_lit(_atom(4, x))
    lit2 = _neg_lit(_atom(5, a, b))
    lit3 = _pos_lit(_atom(6, x, y))
    return Clause(literals=(lit1, lit2, lit3), id=id_)


def _goal_clause(id_: int = 10) -> Clause:
    """A clause derived from a goal (denial)."""
    x, y = _var(0), _var(1)
    c = Clause(
        literals=(_neg_lit(_atom(7, x, y)),),
        id=id_,
        justification=(Justification(just_type=JustType.DENY, clause_id=0),),
    )
    return c


def _derived_clause(parent_ids: tuple[int, ...], id_: int = 20) -> Clause:
    """A clause derived by binary resolution."""
    return Clause(
        literals=(_pos_lit(_atom(8, _const(2))),),
        id=id_,
        justification=(
            Justification(just_type=JustType.BINARY_RES, clause_ids=parent_ids),
        ),
    )


def _empty_clause() -> Clause:
    """Empty clause (contradiction)."""
    return Clause(literals=(), id=99)


# ── Import the module under test ───────────────────────────────────────────

from pyladr.ml.graph.clause_level_mpn import (
    ClauseLevelMPN,
    ClauseLevelConfig,
    ClausePropertyEncoder,
    InferenceRulePotential,
)


# ── Configuration tests ──────────────────────────────────────────────────


class TestClauseLevelConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        config = ClauseLevelConfig()
        assert config.hidden_dim == 256
        assert config.num_attention_heads == 4
        assert config.num_layers == 2
        assert config.dropout == 0.1
        assert config.clause_feature_dim == 7

    def test_custom_config(self):
        config = ClauseLevelConfig(hidden_dim=128, num_attention_heads=8, num_layers=3)
        assert config.hidden_dim == 128
        assert config.num_attention_heads == 8
        assert config.num_layers == 3


# ── ClausePropertyEncoder tests ──────────────────────────────────────────


class TestClausePropertyEncoder:
    """Test encoding of clause structural properties."""

    def test_output_shape(self):
        encoder = ClausePropertyEncoder(clause_feature_dim=7, hidden_dim=64)
        # 7 features: [num_literals, is_unit, is_horn, is_positive, is_negative, is_ground, weight]
        features = torch.randn(5, 7)
        output = encoder(features)
        assert output.shape == (5, 64)

    def test_single_clause(self):
        encoder = ClausePropertyEncoder(clause_feature_dim=7, hidden_dim=64)
        features = torch.randn(1, 7)
        output = encoder(features)
        assert output.shape == (1, 64)

    def test_from_clause_objects(self):
        """Extract and encode features directly from Clause objects."""
        encoder = ClausePropertyEncoder(clause_feature_dim=7, hidden_dim=64)
        clauses = [_unit_clause(), _binary_clause(), _ternary_clause()]
        features = ClausePropertyEncoder.extract_features(clauses)
        assert features.shape == (3, 7)
        output = encoder(features)
        assert output.shape == (3, 64)

    def test_feature_extraction_values(self):
        """Verify extracted feature values match clause properties."""
        c = _unit_clause()
        features = ClausePropertyEncoder.extract_features([c])
        # [num_literals=1, is_unit=1, is_horn=1, is_positive=1, is_negative=0, is_ground=1, weight]
        assert features[0, 0].item() == 1.0  # num_literals
        assert features[0, 1].item() == 1.0  # is_unit
        assert features[0, 3].item() == 1.0  # is_positive

    def test_differentiable(self):
        encoder = ClausePropertyEncoder(clause_feature_dim=7, hidden_dim=64)
        features = torch.randn(3, 7, requires_grad=True)
        output = encoder(features)
        output.sum().backward()
        assert features.grad is not None


# ── Clause Structure Attention tests ─────────────────────────────────────


class TestClauseStructureAttention:
    """Test multi-head attention over literal representations within a clause."""

    def test_single_literal_clause(self):
        """Unit clause: attention over a single literal."""
        config = ClauseLevelConfig(hidden_dim=64, num_attention_heads=4)
        mpn = ClauseLevelMPN(config)

        # 1 clause with 1 literal, hidden_dim=64
        literal_reprs = torch.randn(1, 1, 64)
        clause_features = torch.randn(1, 7)

        output = mpn.clause_structure_attention(literal_reprs, clause_features)
        assert output.shape == (1, 64)

    def test_multi_literal_clause(self):
        """Clause with multiple literals: attention combines them."""
        config = ClauseLevelConfig(hidden_dim=64, num_attention_heads=4)
        mpn = ClauseLevelMPN(config)

        # 1 clause with 3 literals
        literal_reprs = torch.randn(1, 3, 64)
        clause_features = torch.randn(1, 7)

        output = mpn.clause_structure_attention(literal_reprs, clause_features)
        assert output.shape == (1, 64)

    def test_batch_variable_literals(self):
        """Batch of clauses with different literal counts (padded)."""
        config = ClauseLevelConfig(hidden_dim=64, num_attention_heads=4)
        mpn = ClauseLevelMPN(config)

        # Batch of 3 clauses, padded to max 3 literals
        literal_reprs = torch.randn(3, 3, 64)
        clause_features = torch.randn(3, 7)
        # Mask: clause 0 has 1 lit, clause 1 has 2, clause 2 has 3
        mask = torch.tensor([
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ])

        output = mpn.clause_structure_attention(literal_reprs, clause_features, mask=mask)
        assert output.shape == (3, 64)

    def test_attention_respects_mask(self):
        """Masked positions should not influence the output."""
        config = ClauseLevelConfig(hidden_dim=64, num_attention_heads=4)
        mpn = ClauseLevelMPN(config)
        mpn.eval()

        literal_reprs = torch.randn(1, 3, 64)
        clause_features = torch.randn(1, 7)

        # With only first literal unmasked
        mask_one = torch.tensor([[True, False, False]])
        out_one = mpn.clause_structure_attention(literal_reprs, clause_features, mask=mask_one)

        # Changing masked positions should not change output
        literal_reprs2 = literal_reprs.clone()
        literal_reprs2[0, 1:] = torch.randn(2, 64)
        out_two = mpn.clause_structure_attention(literal_reprs2, clause_features, mask=mask_one)

        assert torch.allclose(out_one, out_two, atol=1e-5)


# ── Inference Rule Potential tests ───────────────────────────────────────


class TestInferenceRulePotential:
    """Test scoring of inference potential between clause pairs."""

    def test_output_shape(self):
        irp = InferenceRulePotential(hidden_dim=64)
        clause_a = torch.randn(5, 64)
        clause_b = torch.randn(5, 64)
        scores = irp(clause_a, clause_b)
        assert scores.shape == (5,)

    def test_output_range(self):
        """Scores should be in [0, 1] (sigmoid output)."""
        irp = InferenceRulePotential(hidden_dim=64)
        clause_a = torch.randn(10, 64)
        clause_b = torch.randn(10, 64)
        scores = irp(clause_a, clause_b)
        assert (scores >= 0).all()
        assert (scores <= 1).all()

    def test_symmetric_option(self):
        """When symmetric=True, score(A,B) == score(B,A)."""
        irp = InferenceRulePotential(hidden_dim=64, symmetric=True)
        a = torch.randn(3, 64)
        b = torch.randn(3, 64)
        score_ab = irp(a, b)
        score_ba = irp(b, a)
        assert torch.allclose(score_ab, score_ba, atol=1e-5)

    def test_differentiable(self):
        irp = InferenceRulePotential(hidden_dim=64)
        a = torch.randn(3, 64, requires_grad=True)
        b = torch.randn(3, 64, requires_grad=True)
        scores = irp(a, b)
        scores.sum().backward()
        assert a.grad is not None
        assert b.grad is not None


# ── ClauseLevelMPN full forward tests ────────────────────────────────────


class TestClauseLevelMPNForward:
    """Test the full clause-level message passing network."""

    def test_construction_default(self):
        mpn = ClauseLevelMPN()
        assert mpn.config.hidden_dim == 256

    def test_construction_custom(self):
        config = ClauseLevelConfig(hidden_dim=128, num_layers=1)
        mpn = ClauseLevelMPN(config)
        assert mpn.config.hidden_dim == 128
        assert mpn.config.num_layers == 1

    def test_parameter_count(self):
        """Model should have trainable parameters."""
        mpn = ClauseLevelMPN(ClauseLevelConfig(hidden_dim=64))
        total_params = sum(p.numel() for p in mpn.parameters())
        assert total_params > 0

    def test_forward_single_clause(self):
        """Forward pass with a single clause."""
        config = ClauseLevelConfig(hidden_dim=64, num_attention_heads=4)
        mpn = ClauseLevelMPN(config)

        # 1 clause, 2 literals, hidden_dim=64
        literal_reprs = torch.randn(1, 2, 64)
        clause_features = torch.randn(1, 7)

        output = mpn(literal_reprs, clause_features)
        assert output.shape == (1, 64)

    def test_forward_batch(self):
        """Forward pass with a batch of clauses."""
        config = ClauseLevelConfig(hidden_dim=64, num_attention_heads=4)
        mpn = ClauseLevelMPN(config)

        literal_reprs = torch.randn(4, 3, 64)
        clause_features = torch.randn(4, 7)
        mask = torch.ones(4, 3, dtype=torch.bool)

        output = mpn(literal_reprs, clause_features, mask=mask)
        assert output.shape == (4, 64)

    def test_forward_with_inter_clause_messaging(self):
        """Forward pass with inter-clause message passing (adjacency)."""
        config = ClauseLevelConfig(hidden_dim=64, num_attention_heads=4, num_layers=2)
        mpn = ClauseLevelMPN(config)

        literal_reprs = torch.randn(4, 3, 64)
        clause_features = torch.randn(4, 7)
        # Adjacency: clauses 0-1 and 2-3 are connected (potential inference partners)
        adjacency = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)

        output = mpn(literal_reprs, clause_features, adjacency=adjacency)
        assert output.shape == (4, 64)

    def test_forward_no_adjacency(self):
        """Without adjacency, only intra-clause attention is used."""
        config = ClauseLevelConfig(hidden_dim=64, num_attention_heads=4)
        mpn = ClauseLevelMPN(config)

        literal_reprs = torch.randn(3, 2, 64)
        clause_features = torch.randn(3, 7)

        output = mpn(literal_reprs, clause_features)
        assert output.shape == (3, 64)

    def test_gradient_flow(self):
        """Verify gradients flow through the full forward pass.

        Uses output sensitivity: changing literal_reprs should change output.
        """
        config = ClauseLevelConfig(hidden_dim=32, num_attention_heads=2, num_layers=1)
        mpn = ClauseLevelMPN(config)
        mpn.eval()

        clause_features = torch.randn(2, 7)
        lit_a = torch.randn(2, 2, 32)
        lit_b = lit_a + torch.randn_like(lit_a) * 0.1

        out_a = mpn(lit_a, clause_features)
        out_b = mpn(lit_b, clause_features)

        # Different inputs should produce different outputs
        assert not torch.allclose(out_a, out_b, atol=1e-5)


# ── Literal-to-Clause Composition tests ─────────────────────────────────


class TestLiteralToClauseComposition:
    """Test aggregation of literal representations into clause representations."""

    def test_compose_unit_clause(self):
        """Single literal → clause representation."""
        config = ClauseLevelConfig(hidden_dim=64)
        mpn = ClauseLevelMPN(config)

        literal_reprs = torch.randn(1, 1, 64)
        clause_features = torch.randn(1, 7)

        composed = mpn.compose_literals_to_clause(literal_reprs, clause_features)
        assert composed.shape == (1, 64)

    def test_compose_multi_literal(self):
        """Multiple literals → single clause representation."""
        config = ClauseLevelConfig(hidden_dim=64)
        mpn = ClauseLevelMPN(config)

        literal_reprs = torch.randn(1, 5, 64)
        clause_features = torch.randn(1, 7)

        composed = mpn.compose_literals_to_clause(literal_reprs, clause_features)
        assert composed.shape == (1, 64)

    def test_compose_batch(self):
        """Batch composition with mask."""
        config = ClauseLevelConfig(hidden_dim=64)
        mpn = ClauseLevelMPN(config)

        literal_reprs = torch.randn(3, 4, 64)
        clause_features = torch.randn(3, 7)
        mask = torch.tensor([
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, True],
        ])

        composed = mpn.compose_literals_to_clause(literal_reprs, clause_features, mask=mask)
        assert composed.shape == (3, 64)


# ── Hierarchical Update tests ────────────────────────────────────────────


class TestHierarchicalUpdate:
    """Test hierarchical updates from literal level."""

    def test_update_from_literal_level(self):
        """Clause representations should update when literal representations change."""
        config = ClauseLevelConfig(hidden_dim=64, num_attention_heads=4)
        mpn = ClauseLevelMPN(config)
        mpn.eval()

        clause_features = torch.randn(2, 7)

        # Initial literal representations
        lit_reprs_v1 = torch.randn(2, 2, 64)
        out_v1 = mpn(lit_reprs_v1, clause_features)

        # Updated literal representations (simulating update from below)
        lit_reprs_v2 = lit_reprs_v1 + torch.randn_like(lit_reprs_v1) * 0.5
        out_v2 = mpn(lit_reprs_v2, clause_features)

        # Outputs should differ since literal inputs changed
        assert not torch.allclose(out_v1, out_v2, atol=1e-3)


# ── Integration with Clause data structures ──────────────────────────────


class TestClauseIntegration:
    """Test integration with existing PyLADR Clause objects."""

    def test_feature_extraction_unit(self):
        """Extract features from a unit clause."""
        c = _unit_clause()
        features = ClausePropertyEncoder.extract_features([c])
        assert features.shape == (1, 7)
        assert features[0, 0].item() == 1.0  # num_literals=1

    def test_feature_extraction_binary(self):
        """Extract features from a two-literal clause."""
        c = _binary_clause()
        features = ClausePropertyEncoder.extract_features([c])
        assert features[0, 0].item() == 2.0  # num_literals=2
        assert features[0, 1].item() == 0.0  # is_unit=False

    def test_feature_extraction_empty(self):
        """Extract features from the empty clause."""
        c = _empty_clause()
        features = ClausePropertyEncoder.extract_features([c])
        assert features[0, 0].item() == 0.0  # num_literals=0

    def test_feature_extraction_batch(self):
        """Extract features from a batch of mixed clauses."""
        clauses = [_unit_clause(), _binary_clause(), _ternary_clause(), _empty_clause()]
        features = ClausePropertyEncoder.extract_features(clauses)
        assert features.shape == (4, 7)
        # Verify num_literals for each
        assert features[0, 0].item() == 1.0
        assert features[1, 0].item() == 2.0
        assert features[2, 0].item() == 3.0
        assert features[3, 0].item() == 0.0

    def test_goal_clause_features(self):
        """Goal-derived clauses should have correct features."""
        c = _goal_clause()
        features = ClausePropertyEncoder.extract_features([c])
        assert features[0, 4].item() == 1.0  # is_negative=True (denial of goal)

    def test_full_pipeline_from_clauses(self):
        """Full pipeline: Clause objects → features → ClauseLevelMPN → embeddings."""
        config = ClauseLevelConfig(hidden_dim=64, num_attention_heads=4)
        mpn = ClauseLevelMPN(config)

        clauses = [_unit_clause(), _binary_clause(), _ternary_clause()]
        clause_features = ClausePropertyEncoder.extract_features(clauses)

        # Simulate literal-level representations (would come from LiteralLevelMPN)
        max_lits = max(c.num_literals for c in clauses)
        literal_reprs = torch.randn(len(clauses), max_lits, 64)
        mask = torch.zeros(len(clauses), max_lits, dtype=torch.bool)
        for i, c in enumerate(clauses):
            mask[i, :c.num_literals] = True

        output = mpn(literal_reprs, clause_features, mask=mask)
        assert output.shape == (3, 64)
        assert output.isfinite().all()


# ── Edge case tests ──────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_empty_clause_handling(self):
        """Empty clause (no literals) should still produce output."""
        config = ClauseLevelConfig(hidden_dim=64, num_attention_heads=4)
        mpn = ClauseLevelMPN(config)

        # 1 clause with 0 valid literals (all masked out)
        literal_reprs = torch.randn(1, 1, 64)
        clause_features = torch.randn(1, 7)
        mask = torch.tensor([[False]])

        output = mpn(literal_reprs, clause_features, mask=mask)
        assert output.shape == (1, 64)
        assert output.isfinite().all()

    def test_large_clause(self):
        """Clause with many literals."""
        config = ClauseLevelConfig(hidden_dim=64, num_attention_heads=4)
        mpn = ClauseLevelMPN(config)

        literal_reprs = torch.randn(1, 20, 64)
        clause_features = torch.randn(1, 7)

        output = mpn(literal_reprs, clause_features)
        assert output.shape == (1, 64)
        assert output.isfinite().all()

    def test_deterministic_eval(self):
        """Same inputs should produce same outputs in eval mode."""
        config = ClauseLevelConfig(hidden_dim=64, num_attention_heads=4)
        mpn = ClauseLevelMPN(config)
        mpn.eval()

        literal_reprs = torch.randn(2, 3, 64)
        clause_features = torch.randn(2, 7)

        out1 = mpn(literal_reprs, clause_features)
        out2 = mpn(literal_reprs, clause_features)

        assert torch.allclose(out1, out2, atol=1e-6)
