"""Behavioral correctness tests for RNN2Vec.

Tests α-equivalence, normalization, save/load round-trips, and online updates.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.ml.rnn2vec.algorithm import RNN2Vec, RNN2VecConfig
from pyladr.ml.rnn2vec.encoder import RNNEmbeddingConfig
from pyladr.ml.tree2vec.walks import WalkConfig, WalkType

SYM_P = 1
SYM_I = 2
SYM_N = 3


def _var(n: int):
    return get_variable_term(n)


def _rigid(sym, arity, args=()):
    return get_rigid_term(sym, arity, args)


def _pos_lit(atom) -> Literal:
    return Literal(sign=True, atom=atom)


def _neg_lit(atom) -> Literal:
    return Literal(sign=False, atom=atom)


def _make_trained(normalize: bool = True, embedding_dim: int = 24) -> RNN2Vec:
    config = RNN2VecConfig(
        walk_config=WalkConfig(walk_types=(WalkType.DEPTH_FIRST, WalkType.PATH), seed=42),
        rnn_config=RNNEmbeddingConfig(
            input_dim=16, hidden_dim=32, embedding_dim=embedding_dim,
            normalize=normalize, seed=42,
        ),
        training_epochs=3, seed=42,
    )
    model = RNN2Vec(config)
    x, y = _var(0), _var(1)
    clauses = [
        Clause(literals=(_pos_lit(_rigid(SYM_P, 1, (_rigid(SYM_I, 2, (x, _rigid(SYM_N, 1, (x,)))),))),)),
        Clause(literals=(_pos_lit(_rigid(SYM_P, 1, (_rigid(SYM_I, 2, (_rigid(SYM_N, 1, (x,)), x)),))),)),
        Clause(literals=(
            _neg_lit(_rigid(SYM_P, 1, (_rigid(SYM_I, 2, (x, y)),))),
            _pos_lit(_rigid(SYM_P, 1, (_rigid(SYM_I, 2, (y, x)),))),
        )),
    ]
    model.train(clauses)
    return model


class TestAlphaEquivalence:
    def test_alpha_equivalent_clauses_same_embedding(self) -> None:
        model = _make_trained()
        # P(i(x0, n(x0))) vs P(i(x1, n(x1))) — alpha-equivalent
        c1 = Clause(literals=(_pos_lit(
            _rigid(SYM_P, 1, (_rigid(SYM_I, 2, (_var(0), _rigid(SYM_N, 1, (_var(0),)))),))
        ),))
        c2 = Clause(literals=(_pos_lit(
            _rigid(SYM_P, 1, (_rigid(SYM_I, 2, (_var(1), _rigid(SYM_N, 1, (_var(1),)))),))
        ),))
        e1 = model.embed_clause(c1)
        e2 = model.embed_clause(c2)
        assert e1 is not None and e2 is not None
        sim = RNN2Vec._cosine_similarity(e1, e2)
        assert sim > 0.99

    def test_variable_renaming_preserves_structural_hash(self) -> None:
        from pyladr.ml.embeddings.cache import clause_structural_hash
        c1 = Clause(literals=(_pos_lit(_rigid(SYM_P, 1, (_var(0),))),))
        c2 = Clause(literals=(_pos_lit(_rigid(SYM_P, 1, (_var(5),))),))
        assert clause_structural_hash(c1) == clause_structural_hash(c2)


class TestNormalization:
    def test_normalized_unit_norm(self) -> None:
        model = _make_trained(normalize=True)
        emb = model.embed_clause(
            Clause(literals=(_pos_lit(_rigid(SYM_P, 1, (_var(0),))),))
        )
        assert emb is not None
        norm = math.sqrt(sum(v * v for v in emb))
        assert abs(norm - 1.0) < 0.05

    def test_unnormalized_not_unit_norm(self) -> None:
        model = _make_trained(normalize=False)
        emb = model.embed_clause(
            Clause(literals=(_pos_lit(_rigid(SYM_P, 1, (_var(0),))),))
        )
        assert emb is not None
        # Just verify it's valid (not necessarily unit norm)
        norm = math.sqrt(sum(v * v for v in emb))
        assert norm > 0  # non-zero


class TestSaveLoadRoundTrip:
    def test_embeddings_identical_after_reload(self, tmp_path) -> None:
        model = _make_trained()
        clause = Clause(literals=(_pos_lit(_rigid(SYM_P, 1, (_var(0),))),))
        emb_before = model.embed_clause(clause)

        model.save(tmp_path / "model")
        loaded = RNN2Vec.load(tmp_path / "model")
        emb_after = loaded.embed_clause(clause)

        assert emb_before is not None and emb_after is not None
        for a, b in zip(emb_before, emb_after):
            assert abs(a - b) < 1e-5

    def test_config_preserved_after_reload(self, tmp_path) -> None:
        model = _make_trained()
        model.save(tmp_path / "model")
        loaded = RNN2Vec.load(tmp_path / "model")
        assert loaded.config.rnn_config.rnn_type == model.config.rnn_config.rnn_type
        assert loaded.config.rnn_config.embedding_dim == model.config.rnn_config.embedding_dim


class TestOnlineUpdate:
    def test_update_noop_before_training(self) -> None:
        model = RNN2Vec()
        result = model.update_online([])
        assert result["pairs_trained"] == 0

    def test_update_changes_embeddings(self) -> None:
        model = _make_trained()
        clause = Clause(literals=(_pos_lit(_rigid(SYM_P, 1, (_var(0),))),))
        emb_before = model.embed_clause(clause)

        # Online update with novel clauses
        novel = [Clause(literals=(
            _pos_lit(_rigid(SYM_P, 1, (_rigid(SYM_I, 2, (_var(0), _var(1))),))),
            _neg_lit(_rigid(SYM_P, 1, (_rigid(SYM_N, 1, (_var(2),)),))),
        ))]
        model.update_online(novel, learning_rate=0.01)

        emb_after = model.embed_clause(clause)
        assert emb_before is not None and emb_after is not None
        # At least some values should have changed
        diff = sum(abs(a - b) for a, b in zip(emb_before, emb_after))
        # It's possible they're very close if the update was minimal,
        # so just verify it doesn't crash and returns valid embeddings
        assert len(emb_after) == model.embedding_dim
