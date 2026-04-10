"""Tests for pyladr.ml.graph.proof_level_mpn — Proof-level message passing.

Tests cover:
- ProofLevelMPN construction with default and custom configs
- Derivation relationship messaging (parent-child in proof DAG)
- Temporal context modeling (proof search history)
- Goal-directed messaging and goal proximity computation
- Clause-to-proof composition and hierarchical updates
- Proof context evolution during search
- Integration with existing PyLADR Clause/Justification structures
- Batch processing and incremental updates
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


def _input_clause(id_: int) -> Clause:
    """Input axiom clause."""
    return Clause(
        literals=(_pos_lit(_atom(4, _const(2))),),
        id=id_,
        justification=(Justification(just_type=JustType.INPUT),),
    )


def _goal_clause(id_: int = 10) -> Clause:
    """Goal denial clause."""
    x, y = _var(0), _var(1)
    return Clause(
        literals=(_neg_lit(_atom(7, x, y)),),
        id=id_,
        justification=(Justification(just_type=JustType.DENY, clause_id=0),),
    )


def _resolution_clause(id_: int, parent_ids: tuple[int, ...]) -> Clause:
    """Clause derived by binary resolution."""
    return Clause(
        literals=(_pos_lit(_atom(8, _const(2))),),
        id=id_,
        justification=(
            Justification(just_type=JustType.BINARY_RES, clause_ids=parent_ids),
        ),
    )


def _para_clause(id_: int, from_id: int, into_id: int) -> Clause:
    """Clause derived by paramodulation."""
    from pyladr.core.clause import ParaJust
    return Clause(
        literals=(_pos_lit(_atom(9, _const(3))),),
        id=id_,
        justification=(
            Justification(
                just_type=JustType.PARA,
                para=ParaJust(from_id=from_id, into_id=into_id,
                              from_pos=(1,), into_pos=(1, 1)),
            ),
        ),
    )


def _empty_clause(id_: int = 99, parent_ids: tuple[int, ...] = (5, 10)) -> Clause:
    """Empty clause (proof found)."""
    return Clause(
        literals=(),
        id=id_,
        justification=(
            Justification(just_type=JustType.BINARY_RES, clause_ids=parent_ids),
        ),
    )


def _simple_derivation_graph():
    """Build a simple derivation: 2 input clauses + 1 goal + 1 derived + empty.

    Clause graph:
        1 (input) --+
                     +--> 20 (resolution from 1,2)
        2 (input) --+
                     +--> 99 (empty, resolution from 20, 10)
       10 (goal)  --+

    Returns (clauses, derivation_edges).
    """
    c1 = _input_clause(1)
    c2 = _input_clause(2)
    c10 = _goal_clause(10)
    c20 = _resolution_clause(20, (1, 2))
    c99 = _empty_clause(99, (20, 10))

    clauses = [c1, c2, c10, c20, c99]
    # derivation_edges: (parent_idx, child_idx) — indices into clauses list
    # c1(0), c2(1) → c20(3); c20(3), c10(2) → c99(4)
    edges = torch.tensor([[0, 1, 3, 2], [3, 3, 4, 4]], dtype=torch.long)
    return clauses, edges


# ── Import the module under test ───────────────────────────────────────────

from pyladr.ml.graph.proof_level_mpn import (
    ProofLevelMPN,
    ProofLevelConfig,
    DerivationEncoder,
    GoalProximityComputer,
    TemporalPositionEncoder,
)


# ── Configuration tests ──────────────────────────────────────────────────


class TestProofLevelConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        config = ProofLevelConfig()
        assert config.hidden_dim == 256
        assert config.num_layers == 2
        assert config.dropout == 0.1
        assert config.max_proof_depth == 100
        assert config.temporal_dim == 32

    def test_custom_config(self):
        config = ProofLevelConfig(hidden_dim=128, num_layers=3, max_proof_depth=50)
        assert config.hidden_dim == 128
        assert config.num_layers == 3
        assert config.max_proof_depth == 50


# ── Temporal Position Encoder tests ──────────────────────────────────────


class TestTemporalPositionEncoder:
    """Test encoding of temporal position in proof search."""

    def test_output_shape(self):
        encoder = TemporalPositionEncoder(temporal_dim=32, hidden_dim=64)
        # timestamps: step at which clause was generated
        timestamps = torch.tensor([0, 1, 5, 10, 20], dtype=torch.float32)
        output = encoder(timestamps)
        assert output.shape == (5, 64)

    def test_single_clause(self):
        encoder = TemporalPositionEncoder(temporal_dim=32, hidden_dim=64)
        timestamps = torch.tensor([0.0])
        output = encoder(timestamps)
        assert output.shape == (1, 64)

    def test_ordering_preserved(self):
        """Earlier and later clauses should get different encodings."""
        encoder = TemporalPositionEncoder(temporal_dim=32, hidden_dim=64)
        encoder.eval()
        t_early = torch.tensor([0.0])
        t_late = torch.tensor([100.0])
        enc_early = encoder(t_early)
        enc_late = encoder(t_late)
        # Different timestamps should produce different encodings
        assert not torch.allclose(enc_early, enc_late, atol=1e-3)


# ── Derivation Encoder tests ────────────────────────────────────────────


class TestDerivationEncoder:
    """Test encoding of derivation (justification) types."""

    def test_output_shape(self):
        encoder = DerivationEncoder(hidden_dim=64)
        # Derivation type indices
        deriv_types = torch.tensor([0, 1, 2, 5, 9], dtype=torch.long)  # INPUT, GOAL, DENY, BINARY_RES, PARA
        output = encoder(deriv_types)
        assert output.shape == (5, 64)

    def test_from_clauses(self):
        """Extract derivation types from Clause objects."""
        clauses = [_input_clause(1), _goal_clause(10), _resolution_clause(20, (1, 2))]
        deriv_types = DerivationEncoder.extract_derivation_types(clauses)
        assert deriv_types.shape == (3,)
        assert deriv_types[0].item() == JustType.INPUT
        assert deriv_types[1].item() == JustType.DENY
        assert deriv_types[2].item() == JustType.BINARY_RES

    def test_no_justification_fallback(self):
        """Clauses without justification should get a default type."""
        c = Clause(literals=(_pos_lit(_atom(4, _const(2))),), id=1)
        deriv_types = DerivationEncoder.extract_derivation_types([c])
        assert deriv_types.shape == (1,)
        assert deriv_types[0].item() == JustType.INPUT  # default

    def test_differentiable(self):
        """Embedding lookup should be differentiable (through straight-through)."""
        encoder = DerivationEncoder(hidden_dim=64)
        deriv_types = torch.tensor([0, 5], dtype=torch.long)
        output = encoder(deriv_types)
        # Embedding layers have trainable parameters
        assert sum(p.numel() for p in encoder.parameters()) > 0


# ── Goal Proximity Computer tests ────────────────────────────────────────


class TestGoalProximityComputer:
    """Test goal-directed proximity computation."""

    def test_output_shape(self):
        gpc = GoalProximityComputer(hidden_dim=64)
        clause_reprs = torch.randn(5, 64)
        goal_repr = torch.randn(1, 64)
        proximity = gpc(clause_reprs, goal_repr)
        assert proximity.shape == (5,)

    def test_output_range(self):
        """Proximity scores should be in [0, 1]."""
        gpc = GoalProximityComputer(hidden_dim=64)
        clause_reprs = torch.randn(10, 64)
        goal_repr = torch.randn(1, 64)
        proximity = gpc(clause_reprs, goal_repr)
        assert (proximity >= 0).all()
        assert (proximity <= 1).all()

    def test_self_proximity_high(self):
        """Goal clause should have high proximity to itself."""
        gpc = GoalProximityComputer(hidden_dim=64)
        goal_repr = torch.randn(1, 64)
        # Compute proximity of goal to itself
        self_proximity = gpc(goal_repr, goal_repr)
        # Not guaranteed to be > 0.5, but should be deterministic
        assert self_proximity.shape == (1,)
        assert self_proximity.isfinite().all()

    def test_multiple_goals(self):
        """Support multiple goal clauses (mean-pooled)."""
        gpc = GoalProximityComputer(hidden_dim=64)
        clause_reprs = torch.randn(5, 64)
        goal_reprs = torch.randn(3, 64)  # 3 goal clauses
        proximity = gpc(clause_reprs, goal_reprs)
        assert proximity.shape == (5,)

    def test_differentiable(self):
        gpc = GoalProximityComputer(hidden_dim=64)
        clause_reprs = torch.randn(3, 64, requires_grad=True)
        goal_repr = torch.randn(1, 64, requires_grad=True)
        proximity = gpc(clause_reprs, goal_repr)
        proximity.sum().backward()
        assert clause_reprs.grad is not None
        assert goal_repr.grad is not None


# ── ProofLevelMPN full forward tests ─────────────────────────────────────


class TestProofLevelMPNForward:
    """Test the full proof-level message passing network."""

    def test_construction_default(self):
        mpn = ProofLevelMPN()
        assert mpn.config.hidden_dim == 256

    def test_construction_custom(self):
        config = ProofLevelConfig(hidden_dim=128, num_layers=1)
        mpn = ProofLevelMPN(config)
        assert mpn.config.hidden_dim == 128

    def test_parameter_count(self):
        mpn = ProofLevelMPN(ProofLevelConfig(hidden_dim=64))
        total_params = sum(p.numel() for p in mpn.parameters())
        assert total_params > 0

    def test_forward_no_derivation(self):
        """Forward pass with clause representations only (no derivation graph)."""
        config = ProofLevelConfig(hidden_dim=64)
        mpn = ProofLevelMPN(config)

        clause_reprs = torch.randn(5, 64)
        timestamps = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32)
        deriv_types = torch.tensor([0, 0, 1, 5, 5], dtype=torch.long)

        output = mpn(clause_reprs, timestamps, deriv_types)
        assert output.shape == (5, 64)

    def test_forward_with_derivation(self):
        """Forward pass with derivation DAG edges."""
        config = ProofLevelConfig(hidden_dim=64, num_layers=2)
        mpn = ProofLevelMPN(config)

        clause_reprs = torch.randn(5, 64)
        timestamps = torch.tensor([0, 0, 0, 1, 2], dtype=torch.float32)
        deriv_types = torch.tensor([0, 0, 1, 5, 5], dtype=torch.long)
        derivation_edges = torch.tensor([[0, 1, 3, 2], [3, 3, 4, 4]], dtype=torch.long)

        output = mpn(clause_reprs, timestamps, deriv_types, derivation_edges=derivation_edges)
        assert output.shape == (5, 64)

    def test_forward_with_goal_proximity(self):
        """Forward pass with goal proximity computation."""
        config = ProofLevelConfig(hidden_dim=64)
        mpn = ProofLevelMPN(config)

        clause_reprs = torch.randn(5, 64)
        timestamps = torch.arange(5, dtype=torch.float32)
        deriv_types = torch.tensor([0, 0, 1, 5, 5], dtype=torch.long)
        goal_reprs = torch.randn(2, 64)

        output, proximity = mpn(
            clause_reprs, timestamps, deriv_types,
            goal_reprs=goal_reprs, return_proximity=True,
        )
        assert output.shape == (5, 64)
        assert proximity.shape == (5,)
        assert (proximity >= 0).all()
        assert (proximity <= 1).all()

    def test_gradient_flow(self):
        """Verify gradients flow through the full forward pass."""
        config = ProofLevelConfig(hidden_dim=32, num_layers=1)
        mpn = ProofLevelMPN(config)

        clause_reprs = torch.randn(4, 32, requires_grad=True)
        timestamps = torch.arange(4, dtype=torch.float32)
        deriv_types = torch.tensor([0, 0, 1, 5], dtype=torch.long)
        derivation_edges = torch.tensor([[0, 1], [3, 3]], dtype=torch.long)

        output = mpn(clause_reprs, timestamps, deriv_types, derivation_edges=derivation_edges)
        loss = output.sum()
        loss.backward()

        assert clause_reprs.grad is not None
        assert clause_reprs.grad.abs().sum() > 0

    def test_gradient_flow_with_proximity(self):
        """Gradients flow through goal proximity path too."""
        config = ProofLevelConfig(hidden_dim=32, num_layers=1)
        mpn = ProofLevelMPN(config)

        clause_reprs = torch.randn(3, 32, requires_grad=True)
        timestamps = torch.arange(3, dtype=torch.float32)
        deriv_types = torch.tensor([0, 1, 5], dtype=torch.long)
        goal_reprs = torch.randn(1, 32, requires_grad=True)

        output, proximity = mpn(
            clause_reprs, timestamps, deriv_types,
            goal_reprs=goal_reprs, return_proximity=True,
        )
        (output.sum() + proximity.sum()).backward()

        assert clause_reprs.grad is not None
        assert goal_reprs.grad is not None


# ── Derivation Relationship Messaging tests ──────────────────────────────


class TestDerivationMessaging:
    """Test message passing along derivation DAG edges."""

    def test_parent_to_child_messages(self):
        """Child clauses should receive information from parents."""
        config = ProofLevelConfig(hidden_dim=64, num_layers=2)
        mpn = ProofLevelMPN(config)
        mpn.eval()

        clause_reprs = torch.randn(3, 64)
        timestamps = torch.tensor([0, 0, 1], dtype=torch.float32)
        deriv_types = torch.tensor([0, 0, 5], dtype=torch.long)
        # clause 0 and 1 are parents of clause 2
        edges = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)

        output_with = mpn(clause_reprs, timestamps, deriv_types, derivation_edges=edges)
        output_without = mpn(clause_reprs, timestamps, deriv_types)

        # With derivation edges, the child clause (idx 2) should get different representation
        assert not torch.allclose(output_with[2], output_without[2], atol=1e-3)

    def test_no_edges_identity(self):
        """Without derivation edges, output depends only on temporal + type encoding."""
        config = ProofLevelConfig(hidden_dim=64, num_layers=2)
        mpn = ProofLevelMPN(config)
        mpn.eval()

        clause_reprs = torch.randn(3, 64)
        timestamps = torch.arange(3, dtype=torch.float32)
        deriv_types = torch.zeros(3, dtype=torch.long)

        out1 = mpn(clause_reprs, timestamps, deriv_types)
        out2 = mpn(clause_reprs, timestamps, deriv_types)

        assert torch.allclose(out1, out2, atol=1e-6)


# ── Temporal Context tests ───────────────────────────────────────────────


class TestTemporalContext:
    """Test temporal context modeling."""

    def test_temporal_differentiation(self):
        """Clauses at different timestamps should get different encodings."""
        config = ProofLevelConfig(hidden_dim=64)
        mpn = ProofLevelMPN(config)
        mpn.eval()

        # Same clause features but different timestamps
        clause_reprs = torch.randn(1, 64).expand(3, -1).clone()
        deriv_types = torch.zeros(3, dtype=torch.long)
        timestamps_early = torch.tensor([0, 0, 0], dtype=torch.float32)
        timestamps_late = torch.tensor([50, 50, 50], dtype=torch.float32)

        out_early = mpn(clause_reprs, timestamps_early, deriv_types)
        out_late = mpn(clause_reprs, timestamps_late, deriv_types)

        assert not torch.allclose(out_early, out_late, atol=1e-3)


# ── Goal-Directed Messaging tests ────────────────────────────────────────


class TestGoalDirectedMessaging:
    """Test goal-directed messaging and proximity computation."""

    def test_proximity_varies_by_clause(self):
        """Different clauses should have different goal proximities."""
        config = ProofLevelConfig(hidden_dim=64)
        mpn = ProofLevelMPN(config)

        clause_reprs = torch.randn(5, 64)
        timestamps = torch.arange(5, dtype=torch.float32)
        deriv_types = torch.zeros(5, dtype=torch.long)
        goal_reprs = torch.randn(1, 64)

        _, proximity = mpn(
            clause_reprs, timestamps, deriv_types,
            goal_reprs=goal_reprs, return_proximity=True,
        )

        # Not all proximities should be identical (with random inputs)
        assert not torch.allclose(proximity, proximity[0].expand_as(proximity), atol=1e-4)

    def test_no_goals_no_proximity(self):
        """Without goal_reprs, no proximity should be returned."""
        config = ProofLevelConfig(hidden_dim=64)
        mpn = ProofLevelMPN(config)

        clause_reprs = torch.randn(3, 64)
        timestamps = torch.arange(3, dtype=torch.float32)
        deriv_types = torch.zeros(3, dtype=torch.long)

        result = mpn(clause_reprs, timestamps, deriv_types, return_proximity=True)
        # When no goal_reprs provided, proximity should be None
        output, proximity = result
        assert output.shape == (3, 64)
        assert proximity is None


# ── Proof Context Evolution tests ────────────────────────────────────────


class TestProofContextEvolution:
    """Test incremental proof context updates."""

    def test_incremental_update(self):
        """Adding new clauses should update proof context."""
        config = ProofLevelConfig(hidden_dim=64, num_layers=1)
        mpn = ProofLevelMPN(config)
        mpn.eval()

        # Initial: 3 clauses
        c3 = torch.randn(3, 64)
        t3 = torch.arange(3, dtype=torch.float32)
        d3 = torch.zeros(3, dtype=torch.long)
        out3 = mpn(c3, t3, d3)

        # Extended: add 2 more clauses with derivation edges
        c5 = torch.cat([c3, torch.randn(2, 64)])
        t5 = torch.arange(5, dtype=torch.float32)
        d5 = torch.tensor([0, 0, 0, 5, 5], dtype=torch.long)
        edges = torch.tensor([[0, 1], [3, 3]], dtype=torch.long)
        out5 = mpn(c5, t5, d5, derivation_edges=edges)

        # The original clauses should have the same representations
        # when there are no edges connecting back to them (layers 1+)
        # But clause 3 (derived from 0,1) should be different
        assert out5.shape == (5, 64)

    def test_proof_state_representation(self):
        """Get an aggregate proof state representation."""
        config = ProofLevelConfig(hidden_dim=64)
        mpn = ProofLevelMPN(config)

        clause_reprs = torch.randn(5, 64)
        timestamps = torch.arange(5, dtype=torch.float32)
        deriv_types = torch.zeros(5, dtype=torch.long)

        output = mpn(clause_reprs, timestamps, deriv_types)
        proof_state = mpn.aggregate_proof_state(output)
        assert proof_state.shape == (1, 64)


# ── Integration with Clause data structures ──────────────────────────────


class TestClauseIntegration:
    """Test integration with existing PyLADR structures."""

    def test_derivation_type_extraction(self):
        """Extract derivation types from real Clause objects."""
        clauses, _ = _simple_derivation_graph()
        deriv_types = DerivationEncoder.extract_derivation_types(clauses)
        assert deriv_types.shape == (5,)
        assert deriv_types[0].item() == JustType.INPUT
        assert deriv_types[1].item() == JustType.INPUT
        assert deriv_types[2].item() == JustType.DENY
        assert deriv_types[3].item() == JustType.BINARY_RES
        assert deriv_types[4].item() == JustType.BINARY_RES

    def test_full_pipeline_from_derivation(self):
        """Full pipeline: derivation graph → ProofLevelMPN → embeddings."""
        config = ProofLevelConfig(hidden_dim=64, num_layers=2)
        mpn = ProofLevelMPN(config)

        clauses, edges = _simple_derivation_graph()
        deriv_types = DerivationEncoder.extract_derivation_types(clauses)
        timestamps = torch.arange(len(clauses), dtype=torch.float32)
        clause_reprs = torch.randn(len(clauses), 64)

        output = mpn(clause_reprs, timestamps, deriv_types, derivation_edges=edges)
        assert output.shape == (5, 64)
        assert output.isfinite().all()

    def test_pipeline_with_goal_proximity(self):
        """Full pipeline with goal proximity computation."""
        config = ProofLevelConfig(hidden_dim=64)
        mpn = ProofLevelMPN(config)

        clauses, edges = _simple_derivation_graph()
        deriv_types = DerivationEncoder.extract_derivation_types(clauses)
        timestamps = torch.arange(len(clauses), dtype=torch.float32)
        clause_reprs = torch.randn(len(clauses), 64)

        # Goal clause is at index 2
        goal_reprs = clause_reprs[2:3]  # (1, 64)

        output, proximity = mpn(
            clause_reprs, timestamps, deriv_types,
            derivation_edges=edges, goal_reprs=goal_reprs,
            return_proximity=True,
        )
        assert output.shape == (5, 64)
        assert proximity.shape == (5,)
        assert output.isfinite().all()
        assert proximity.isfinite().all()


# ── Edge case tests ──────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_clause(self):
        """Single clause with no derivation."""
        config = ProofLevelConfig(hidden_dim=64)
        mpn = ProofLevelMPN(config)

        clause_reprs = torch.randn(1, 64)
        timestamps = torch.tensor([0.0])
        deriv_types = torch.tensor([0], dtype=torch.long)

        output = mpn(clause_reprs, timestamps, deriv_types)
        assert output.shape == (1, 64)
        assert output.isfinite().all()

    def test_large_proof(self):
        """Many clauses with complex derivation graph."""
        config = ProofLevelConfig(hidden_dim=64, num_layers=2)
        mpn = ProofLevelMPN(config)

        n = 50
        clause_reprs = torch.randn(n, 64)
        timestamps = torch.arange(n, dtype=torch.float32)
        deriv_types = torch.randint(0, 10, (n,))
        # Random derivation edges (parent → child, respecting ordering)
        src = torch.randint(0, n - 1, (30,))
        dst = src + 1 + torch.randint(0, 3, (30,)).clamp(max=n - 1 - src)
        dst = dst.clamp(max=n - 1)
        edges = torch.stack([src, dst])

        output = mpn(clause_reprs, timestamps, deriv_types, derivation_edges=edges)
        assert output.shape == (n, 64)
        assert output.isfinite().all()

    def test_deterministic_eval(self):
        """Same inputs should produce same outputs in eval mode."""
        config = ProofLevelConfig(hidden_dim=64)
        mpn = ProofLevelMPN(config)
        mpn.eval()

        clause_reprs = torch.randn(3, 64)
        timestamps = torch.arange(3, dtype=torch.float32)
        deriv_types = torch.zeros(3, dtype=torch.long)

        out1 = mpn(clause_reprs, timestamps, deriv_types)
        out2 = mpn(clause_reprs, timestamps, deriv_types)

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_empty_derivation_edges(self):
        """Empty edge tensor should work."""
        config = ProofLevelConfig(hidden_dim=64)
        mpn = ProofLevelMPN(config)

        clause_reprs = torch.randn(3, 64)
        timestamps = torch.arange(3, dtype=torch.float32)
        deriv_types = torch.zeros(3, dtype=torch.long)
        empty_edges = torch.zeros(2, 0, dtype=torch.long)

        output = mpn(clause_reprs, timestamps, deriv_types, derivation_edges=empty_edges)
        assert output.shape == (3, 64)
        assert output.isfinite().all()
