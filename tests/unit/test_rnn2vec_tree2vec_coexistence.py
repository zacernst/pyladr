"""Coexistence tests: RNN2Vec and Tree2Vec can operate side-by-side."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.ml.rnn2vec.algorithm import RNN2Vec, RNN2VecConfig
from pyladr.ml.rnn2vec.encoder import RNNEmbeddingConfig
from pyladr.ml.tree2vec.algorithm import Tree2Vec, Tree2VecConfig
from pyladr.ml.tree2vec.skipgram import SkipGramConfig
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


def _test_clauses() -> list[Clause]:
    x, y = _var(0), _var(1)
    return [
        Clause(literals=(_pos_lit(_rigid(SYM_P, 1, (_rigid(SYM_I, 2, (x, _rigid(SYM_N, 1, (x,)))),))),)),
        Clause(literals=(_pos_lit(_rigid(SYM_P, 1, (_rigid(SYM_I, 2, (_rigid(SYM_N, 1, (x,)), x)),))),)),
    ]


def _make_rnn2vec() -> RNN2Vec:
    config = RNN2VecConfig(
        walk_config=WalkConfig(walk_types=(WalkType.DEPTH_FIRST, WalkType.PATH), seed=42),
        rnn_config=RNNEmbeddingConfig(input_dim=16, hidden_dim=32, embedding_dim=24, seed=42),
        training_epochs=2, seed=42,
    )
    model = RNN2Vec(config)
    model.train(_test_clauses())
    return model


def _make_tree2vec() -> Tree2Vec:
    config = Tree2VecConfig(
        walk_config=WalkConfig(walk_types=(WalkType.DEPTH_FIRST, WalkType.PATH), seed=42),
        skipgram_config=SkipGramConfig(embedding_dim=16, num_epochs=2, seed=42),
    )
    model = Tree2Vec(config)
    model.train(_test_clauses())
    return model


class TestProviderCoexistence:
    def test_both_embed_same_clause(self) -> None:
        rnn = _make_rnn2vec()
        t2v = _make_tree2vec()
        clause = _test_clauses()[0]

        rnn_emb = rnn.embed_clause(clause)
        t2v_emb = t2v.embed_clause(clause)

        assert rnn_emb is not None
        assert t2v_emb is not None
        # Different algorithms, different dims
        assert len(rnn_emb) == 24
        assert len(t2v_emb) == 16

    def test_embedding_dims_independent(self) -> None:
        rnn = _make_rnn2vec()
        t2v = _make_tree2vec()
        assert rnn.embedding_dim == 24
        assert t2v.embedding_dim == 16

    def test_tree2vec_unaffected(self) -> None:
        """Tree2Vec still works correctly after rnn2vec is imported and used."""
        # First use rnn2vec
        rnn = _make_rnn2vec()
        clause = _test_clauses()[0]
        rnn.embed_clause(clause)

        # Now use tree2vec — should be completely independent
        t2v = _make_tree2vec()
        emb = t2v.embed_clause(clause)
        assert emb is not None
        assert len(emb) == 16
        assert t2v.trained
