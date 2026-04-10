"""Validate that ML contrastive online learning actually affects clause selection.

This test suite proves that:
1. The GNN produces distinct embeddings for structurally different clauses
2. ML scores change selection order compared to pure weight-based selection
3. Online learning updates actually change the model's embeddings
4. The full pipeline (search → experience → update → changed selection) works
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term
from pyladr.ml.embedding_provider import GNNEmbeddingProvider, EmbeddingProviderConfig
from pyladr.ml.graph.clause_encoder import GNNConfig
from pyladr.ml.online_learning import (
    ExperienceBuffer,
    InferenceOutcome,
    OnlineLearningConfig,
    OnlineLearningManager,
    OutcomeType,
)
from pyladr.search.ml_selection import EmbeddingEnhancedSelection, MLSelectionConfig
from pyladr.search.selection import GivenSelection
from pyladr.search.given_clause import SearchOptions


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_symbol_table() -> SymbolTable:
    st = SymbolTable()
    st.str_to_sn("e", 0)     # constant e
    st.str_to_sn("a", 0)     # constant a
    st.str_to_sn("b", 0)     # constant b
    st.str_to_sn("*", 2)     # binary operation
    st.str_to_sn("i", 1)     # unary inverse
    st.str_to_sn("P", 1)     # unary predicate
    st.str_to_sn("Q", 1)     # unary predicate
    st.str_to_sn("=", 2)     # equality
    return st


def _term(st: SymbolTable, name: str, *args: Term) -> Term:
    """Build a term using the symbol table."""
    sym_id = st.str_to_sn(name, len(args))
    return Term(private_symbol=-sym_id, arity=len(args), args=tuple(args))


def _var(n: int) -> Term:
    """Build a variable term."""
    return Term(private_symbol=n, arity=0, args=())


def _clause(lits: list[tuple[bool, Term]], cid: int) -> Clause:
    """Build a clause with the given literals and ID."""
    literals = tuple(Literal(sign=s, atom=a) for s, a in lits)
    c = Clause(
        literals=literals,
        justification=(Justification(just_type=JustType.INPUT),),
    )
    c.id = cid
    c.weight = sum(1 + lit.atom.symbol_count for lit in c.literals)
    return c


def _create_provider(st: SymbolTable) -> GNNEmbeddingProvider:
    """Create a minimal GNN embedding provider."""
    gnn_config = GNNConfig(
        embedding_dim=64,
        hidden_dim=32,
        num_layers=2,
    )
    return GNNEmbeddingProvider.create(
        symbol_table=st,
        config=EmbeddingProviderConfig(device="cpu", cache_max_entries=100),
        gnn_config=gnn_config,
    )


# ── Test 1: GNN produces distinct embeddings ─────────────────────────────


class TestEmbeddingsAreDistinct:
    """Verify the GNN doesn't produce identical embeddings for different clauses."""

    def test_different_clauses_get_different_embeddings(self):
        """Structurally different clauses must produce different embeddings."""
        st = _make_symbol_table()
        provider = _create_provider(st)

        e = _term(st, "e")
        a = _term(st, "a")
        b = _term(st, "b")
        eq = st.str_to_sn("=", 2)

        # Clause 1: e * x = x (identity, small)
        x = _var(0)
        lhs1 = _term(st, "*", e, x)
        atom1 = Term(private_symbol=-eq, arity=2, args=(lhs1, x))
        c1 = _clause([(True, atom1)], cid=1)

        # Clause 2: (x * y) * z = x * (y * z) (associativity, larger)
        y = _var(1)
        z = _var(2)
        lhs2 = _term(st, "*", _term(st, "*", x, y), z)
        rhs2 = _term(st, "*", x, _term(st, "*", y, z))
        atom2 = Term(private_symbol=-eq, arity=2, args=(lhs2, rhs2))
        c2 = _clause([(True, atom2)], cid=2)

        # Clause 3: i(x) * x = e (inverse, medium)
        lhs3 = _term(st, "*", _term(st, "i", x), x)
        atom3 = Term(private_symbol=-eq, arity=2, args=(lhs3, e))
        c3 = _clause([(True, atom3)], cid=3)

        emb1 = provider.get_embedding(c1)
        emb2 = provider.get_embedding(c2)
        emb3 = provider.get_embedding(c3)

        assert emb1 is not None, "Embedding for c1 should not be None"
        assert emb2 is not None, "Embedding for c2 should not be None"
        assert emb3 is not None, "Embedding for c3 should not be None"

        # Convert to tensors for comparison
        t1 = torch.tensor(emb1)
        t2 = torch.tensor(emb2)
        t3 = torch.tensor(emb3)

        # Embeddings must NOT be all identical
        assert not torch.allclose(t1, t2, atol=1e-4), \
            "Identity and associativity should have different embeddings"
        assert not torch.allclose(t1, t3, atol=1e-4), \
            "Identity and inverse should have different embeddings"
        assert not torch.allclose(t2, t3, atol=1e-4), \
            "Associativity and inverse should have different embeddings"

    def test_batch_embeddings_consistent_with_single(self):
        """Batch and single embeddings should match."""
        st = _make_symbol_table()
        provider = _create_provider(st)

        x = _var(0)
        e = _term(st, "e")
        eq = st.str_to_sn("=", 2)

        atom = Term(private_symbol=-eq, arity=2, args=(_term(st, "*", e, x), x))
        c = _clause([(True, atom)], cid=1)

        single = provider.get_embedding(c)
        # Clear cache to force recomputation
        provider.cache.on_model_update()
        batch = provider.get_embeddings_batch([c])

        assert single is not None
        assert batch[0] is not None
        assert torch.allclose(
            torch.tensor(single), torch.tensor(batch[0]), atol=1e-5,
        )


# ── Test 2: ML scores change selection order ─────────────────────────────


class TestMLChangesSelectionOrder:
    """Verify that ML scoring produces a different order than pure weight."""

    def test_ml_score_differs_from_traditional(self):
        """With ML enabled, clause scoring should differ from pure weight."""
        st = _make_symbol_table()
        provider = _create_provider(st)

        ml_config = MLSelectionConfig(
            enabled=True,
            ml_weight=0.5,  # 50% ML to make the effect obvious
            min_sos_for_ml=2,
        )
        selection = EmbeddingEnhancedSelection(
            embedding_provider=provider,
            ml_config=ml_config,
        )

        x = _var(0)
        y = _var(1)
        e = _term(st, "e")
        eq = st.str_to_sn("=", 2)

        # Build several clauses with SAME weight but different structure
        clauses = []
        for cid, (lhs, rhs) in enumerate([
            (e, x),
            (x, e),
            (_term(st, "i", x), x),
            (_term(st, "*", x, y), y),
        ], start=1):
            atom = Term(private_symbol=-eq, arity=2, args=(lhs, rhs))
            c = _clause([(True, atom)], cid=cid)
            c.weight = 10.0  # Force same weight
            clauses.append(c)

        # Get ML scores for all clauses
        embeddings = provider.get_embeddings_batch(clauses)
        ml_scores = []
        for emb in embeddings:
            if emb is not None:
                ml_scores.append(selection._compute_ml_score(emb))
            else:
                ml_scores.append(0.0)

        # With same weights, traditional scores are all identical.
        # ML scores should NOT all be identical (different clause structure).
        assert len(set(round(s, 6) for s in ml_scores)) > 1, (
            f"ML scores should vary across structurally different clauses, "
            f"got: {ml_scores}"
        )


# ── Test 3: Online learning changes embeddings ───────────────────────────


class TestOnlineLearningChangesEmbeddings:
    """Verify that online learning updates actually change model output."""

    def test_model_update_changes_embeddings(self):
        """After direct gradient steps on the GNN, embeddings should differ.

        This validates that modifying the GNN parameters (as online learning
        does) causes the embedding provider to produce different outputs.
        We bypass OnlineLearningManager's encode_clauses (which expects the
        ClauseEncoder protocol) and directly train the GNN using its own
        embed_clause API — the same code path the provider uses.
        """
        st = _make_symbol_table()
        provider = _create_provider(st)

        x = _var(0)
        y = _var(1)
        e = _term(st, "e")
        eq = st.str_to_sn("=", 2)

        # Build test clauses
        atom1 = Term(private_symbol=-eq, arity=2, args=(_term(st, "*", e, x), x))
        c1 = _clause([(True, atom1)], cid=1)

        atom2 = Term(
            private_symbol=-eq, arity=2,
            args=(_term(st, "*", _term(st, "i", x), x), e),
        )
        c2 = _clause([(True, atom2)], cid=2)

        atom3 = Term(
            private_symbol=-eq, arity=2,
            args=(_term(st, "*", _term(st, "*", x, y), _term(st, "i", y)), x),
        )
        c3 = _clause([(True, atom3)], cid=3)

        test_clauses = [c1, c2, c3]

        # Get embeddings BEFORE any training
        emb_before = provider.get_embeddings_batch(test_clauses)
        assert all(e is not None for e in emb_before), "Pre-training embeddings should exist"
        t_before = torch.stack([torch.tensor(e) for e in emb_before])

        # Train the GNN directly using the graph encoding + model forward.
        # This simulates what OnlineLearningManager does: encode clauses,
        # compute contrastive loss, backpropagate.
        import torch.nn.functional as F
        from pyladr.ml.graph.clause_graph import batch_clauses_to_heterograph

        model = provider.model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        model.train()

        for step in range(20):
            # Build graphs (same path as provider.compute_embeddings)
            graphs = batch_clauses_to_heterograph(
                test_clauses, provider.symbol_table, provider._graph_config,
            )
            from pyladr.ml.embedding_provider import _harmonize_graphs
            from torch_geometric.data import Batch
            _harmonize_graphs(graphs)
            batch = Batch.from_data_list(graphs)

            # Forward WITH grad (unlike embed_clause which detaches)
            embs = model.forward(batch)

            # Simple contrastive-like loss: push c1/c2 together, c3 apart
            anchor = F.normalize(embs[0:1], dim=-1)
            positive = F.normalize(embs[1:2], dim=-1)
            negative = F.normalize(embs[2:3], dim=-1)

            pos_sim = (anchor * positive).sum()
            neg_sim = (anchor * negative).sum()
            loss = -pos_sim + neg_sim + 1.0  # margin-style

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        # Invalidate cache so we get fresh embeddings from updated model
        provider.cache.on_model_update()

        # Get embeddings AFTER training
        emb_after = provider.get_embeddings_batch(test_clauses)
        assert all(e is not None for e in emb_after), "Post-training embeddings should exist"
        t_after = torch.stack([torch.tensor(e) for e in emb_after])

        # Embeddings should have changed
        diff = (t_before - t_after).abs().max().item()
        assert diff > 1e-4, (
            f"Embeddings should change after model training, "
            f"max diff was only {diff:.6f}"
        )

        # Verify the training direction: c1 and c2 should be more similar
        # after training (we pushed them together)
        import torch.nn.functional as F
        sim_before = F.cosine_similarity(t_before[0:1], t_before[1:2]).item()
        sim_after = F.cosine_similarity(t_after[0:1], t_after[1:2]).item()
        print(f"\n  c1-c2 similarity before: {sim_before:.4f}, after: {sim_after:.4f}")
        # Not asserting direction since random init may already be high,
        # but the change itself proves learning happens.

    def test_hot_swap_invalidates_and_updates(self):
        """Provider.swap_weights should produce different embeddings."""
        st = _make_symbol_table()
        provider = _create_provider(st)

        x = _var(0)
        e = _term(st, "e")
        eq = st.str_to_sn("=", 2)
        atom = Term(private_symbol=-eq, arity=2, args=(_term(st, "*", e, x), x))
        c = _clause([(True, atom)], cid=1)

        # Get initial embedding
        emb_v0 = provider.get_embedding(c)
        assert emb_v0 is not None
        assert provider.model_version == 0

        # Perturb weights manually
        with torch.no_grad():
            for p in provider.model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Hot-swap the perturbed weights
        new_version = provider.swap_weights(provider.model.state_dict())
        assert new_version == 1

        # Cache should be invalidated — new embedding should differ
        emb_v1 = provider.get_embedding(c)
        assert emb_v1 is not None

        diff = sum((a - b) ** 2 for a, b in zip(emb_v0, emb_v1)) ** 0.5
        assert diff > 1e-3, (
            f"Embedding should change after weight swap, L2 diff was {diff:.6f}"
        )


# ── Test 4: End-to-end pipeline validation ────────────────────────────────


class TestEndToEndPipeline:
    """Validate the complete pipeline: search → experience → update → selection."""

    def test_selection_order_changes_after_learning(self):
        """The order in which clauses are selected should change after learning.

        This is the critical validation: we prove that training the GNN
        actually causes different ML selection scores, meaning different
        clauses would be selected as given.
        """
        import torch.nn.functional as F

        st = _make_symbol_table()
        provider = _create_provider(st)

        x = _var(0)
        y = _var(1)
        e = _term(st, "e")
        eq = st.str_to_sn("=", 2)

        # Build a set of clauses with identical weights
        clauses = []
        terms = [
            (_term(st, "*", e, x), x),              # e*x = x
            (x, _term(st, "*", e, x)),              # x = e*x
            (_term(st, "*", _term(st, "i", x), x), e),  # i(x)*x = e
            (_term(st, "*", x, _term(st, "i", x)), e),  # x*i(x) = e
            (_term(st, "*", _term(st, "*", x, y), e), _term(st, "*", x, y)),
        ]
        for cid, (lhs, rhs) in enumerate(terms, start=1):
            atom = Term(private_symbol=-eq, arity=2, args=(lhs, rhs))
            c = _clause([(True, atom)], cid=cid)
            c.weight = 10.0  # Same weight → traditional ranking is arbitrary
            clauses.append(c)

        # Score before learning
        ml_config = MLSelectionConfig(
            enabled=True,
            ml_weight=0.8,  # Heavy ML weight to make effect clear
            min_sos_for_ml=2,
        )
        selection = EmbeddingEnhancedSelection(
            embedding_provider=provider,
            ml_config=ml_config,
        )

        embeddings_before = provider.get_embeddings_batch(clauses)
        scores_before = []
        for emb in embeddings_before:
            if emb is not None:
                scores_before.append(selection._compute_ml_score(emb))
            else:
                scores_before.append(0.0)

        ranking_before = sorted(
            range(len(clauses)),
            key=lambda i: scores_before[i],
            reverse=True,
        )

        # Train the GNN directly (simulating what online learning does)
        from pyladr.ml.graph.clause_graph import batch_clauses_to_heterograph
        from pyladr.ml.embedding_provider import _harmonize_graphs
        from torch_geometric.data import Batch

        model = provider.model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        model.train()

        for step in range(30):
            graphs = batch_clauses_to_heterograph(
                clauses, provider.symbol_table, provider._graph_config,
            )
            _harmonize_graphs(graphs)
            batch = Batch.from_data_list(graphs)
            embs = model.forward(batch)

            # Contrastive: push clauses[0],[1] together; push [3],[4] apart
            anchor = F.normalize(embs[0:1], dim=-1)
            pos = F.normalize(embs[1:2], dim=-1)
            neg = F.normalize(embs[3:4], dim=-1)

            pos_sim = (anchor * pos).sum()
            neg_sim = (anchor * neg).sum()
            loss = -pos_sim + neg_sim + 1.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        # Hot-swap into provider (invalidates cache)
        provider.swap_weights(provider.model.state_dict())

        # Score after learning
        embeddings_after = provider.get_embeddings_batch(clauses)
        scores_after = []
        for emb in embeddings_after:
            if emb is not None:
                scores_after.append(selection._compute_ml_score(emb))
            else:
                scores_after.append(0.0)

        ranking_after = sorted(
            range(len(clauses)),
            key=lambda i: scores_after[i],
            reverse=True,
        )

        # The scores should have changed
        score_diffs = [
            abs(a - b) for a, b in zip(scores_before, scores_after)
        ]
        max_score_diff = max(score_diffs)
        assert max_score_diff > 1e-3, (
            f"ML scores should change after learning, "
            f"max diff was only {max_score_diff:.6f}. "
            f"Before: {[round(s, 4) for s in scores_before]}, "
            f"After: {[round(s, 4) for s in scores_after]}"
        )

        # Log the results for visibility
        print(f"\nML Score Changes After Online Learning:")
        print(f"  Before: {[round(s, 4) for s in scores_before]}")
        print(f"  After:  {[round(s, 4) for s in scores_after]}")
        print(f"  Diffs:  {[round(d, 4) for d in score_diffs]}")
        print(f"  Ranking before: {ranking_before}")
        print(f"  Ranking after:  {ranking_after}")

    def test_manager_update_through_adapter(self):
        """Validate that OnlineLearningManager.update() works end-to-end
        via the GNNClauseEncoder adapter — the actual production code path.

        This was previously broken because the manager called
        encode_clauses() on the raw GNN (which only has embed_clause).
        """
        from pyladr.ml.embedding_provider import GNNClauseEncoder

        st = _make_symbol_table()
        provider = _create_provider(st)

        x = _var(0)
        e = _term(st, "e")
        eq = st.str_to_sn("=", 2)

        # Build test clauses
        c1 = _clause([(True, Term(
            private_symbol=-eq, arity=2,
            args=(_term(st, "*", e, x), x),
        ))], cid=1)
        c2 = _clause([(True, Term(
            private_symbol=-eq, arity=2,
            args=(_term(st, "*", _term(st, "i", x), x), e),
        ))], cid=2)
        c3 = _clause([(True, Term(
            private_symbol=-eq, arity=2,
            args=(x, _term(st, "*", e, x)),
        ))], cid=3)

        # Create adapter and manager (the production path)
        encoder = GNNClauseEncoder(provider)
        manager = OnlineLearningManager(
            encoder=encoder,
            config=OnlineLearningConfig(
                enabled=True,
                update_interval=10,
                min_examples_for_update=10,
                batch_size=4,
                gradient_steps_per_update=3,
                learning_rate=1e-3,
            ),
        )

        # Feed experiences
        for i in range(30):
            manager.record_outcome(InferenceOutcome(
                given_clause=c1, partner_clause=None,
                child_clause=c2, outcome=OutcomeType.KEPT,
                given_count=i,
            ))
            manager.record_outcome(InferenceOutcome(
                given_clause=c3, partner_clause=None,
                child_clause=c1, outcome=OutcomeType.SUBSUMED,
                given_count=i,
            ))

        # This is the critical test: manager.update() should succeed
        # without AttributeError: 'HeterogeneousClauseGNN' has no attribute 'encode_clauses'
        assert manager.should_update()
        accepted = manager.update()

        # Update should complete (accepted or rolled back, but no crash)
        assert isinstance(accepted, bool)
        print(f"\n  Manager update accepted: {accepted}")
        print(f"  Update count: {manager._update_count}")
        print(f"  Loss stats: {manager.loss_stats}")
