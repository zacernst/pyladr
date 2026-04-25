"""Protocol compliance tests: RNN2VecEmbeddingProvider satisfies EmbeddingProvider."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.ml.rnn2vec.algorithm import RNN2Vec, RNN2VecConfig
from pyladr.ml.rnn2vec.encoder import RNNEmbeddingConfig
from pyladr.ml.rnn2vec.provider import RNN2VecEmbeddingProvider, RNN2VecProviderConfig
from pyladr.ml.tree2vec.walks import WalkConfig, WalkType
from pyladr.protocols import EmbeddingProvider

SYM_P = 1
SYM_I = 2
SYM_N = 3


def _var(n: int):
    return get_variable_term(n)


def _rigid(sym: int, arity: int, args=()):
    return get_rigid_term(sym, arity, args)


def _pos_lit(atom) -> Literal:
    return Literal(sign=True, atom=atom)


def _simple_clause() -> Clause:
    return Clause(literals=(_pos_lit(_rigid(SYM_P, 1, (_var(0),))),))


def _make_trained_provider() -> RNN2VecEmbeddingProvider:
    config = RNN2VecConfig(
        walk_config=WalkConfig(walk_types=(WalkType.DEPTH_FIRST, WalkType.PATH), seed=42),
        rnn_config=RNNEmbeddingConfig(input_dim=16, hidden_dim=32, embedding_dim=24, seed=42),
        training_epochs=3, seed=42,
    )
    model = RNN2Vec(config)
    model.train([_simple_clause()])
    return RNN2VecEmbeddingProvider(model, config=RNN2VecProviderConfig())


class TestRNN2VecProtocolCompliance:
    def test_has_embedding_dim_property(self) -> None:
        provider = _make_trained_provider()
        assert isinstance(provider.embedding_dim, int)
        assert provider.embedding_dim > 0

    def test_get_embedding_returns_list_float_or_none(self) -> None:
        provider = _make_trained_provider()
        result = provider.get_embedding(_simple_clause())
        assert result is None or (isinstance(result, list) and all(isinstance(v, float) for v in result))

    def test_embedding_length_matches_dim(self) -> None:
        provider = _make_trained_provider()
        result = provider.get_embedding(_simple_clause())
        if result is not None:
            assert len(result) == provider.embedding_dim

    def test_batch_length_matches_input(self) -> None:
        provider = _make_trained_provider()
        clauses = [_simple_clause(), _simple_clause()]
        results = provider.get_embeddings_batch(clauses)
        assert len(results) == len(clauses)

    def test_batch_element_types(self) -> None:
        provider = _make_trained_provider()
        results = provider.get_embeddings_batch([_simple_clause()])
        for r in results:
            assert r is None or (isinstance(r, list) and all(isinstance(v, float) for v in r))

    def test_graceful_degradation_untrained(self) -> None:
        model = RNN2Vec()
        provider = RNN2VecEmbeddingProvider(model)
        # Should NOT raise — returns None
        result = provider.get_embedding(_simple_clause())
        assert result is None

    def test_empty_batch_returns_empty(self) -> None:
        provider = _make_trained_provider()
        assert provider.get_embeddings_batch([]) == []

    def test_embedding_values_are_floats(self) -> None:
        provider = _make_trained_provider()
        result = provider.get_embedding(_simple_clause())
        if result is not None:
            for v in result:
                assert type(v) is float, f"Expected float, got {type(v)}"
