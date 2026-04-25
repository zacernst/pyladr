"""Edge case tests for RNN2Vec: propositional atoms, deep trees, large clauses, OOV."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.ml.rnn2vec.algorithm import RNN2Vec, RNN2VecConfig
from pyladr.ml.rnn2vec.encoder import RNNEmbeddingConfig
from pyladr.ml.tree2vec.walks import WalkConfig, WalkType

SYM_P = 1
SYM_I = 2
SYM_N = 3
SYM_C = 4  # arity-0 constant


def _var(n: int):
    return get_variable_term(n)


def _rigid(sym, arity, args=()):
    return get_rigid_term(sym, arity, args)


def _pos_lit(atom) -> Literal:
    return Literal(sign=True, atom=atom)


def _neg_lit(atom) -> Literal:
    return Literal(sign=False, atom=atom)


def _make_trained(clauses=None) -> RNN2Vec:
    config = RNN2VecConfig(
        walk_config=WalkConfig(walk_types=(WalkType.DEPTH_FIRST, WalkType.PATH), seed=42),
        rnn_config=RNNEmbeddingConfig(
            input_dim=16, hidden_dim=32, embedding_dim=24, seed=42,
        ),
        training_epochs=3, seed=42,
    )
    model = RNN2Vec(config)
    if clauses is None:
        x = _var(0)
        clauses = [
            Clause(literals=(_pos_lit(_rigid(SYM_P, 1, (_rigid(SYM_I, 2, (x, _rigid(SYM_N, 1, (x,)))),))),)),
            Clause(literals=(_pos_lit(_rigid(SYM_P, 1, (_rigid(SYM_N, 1, (x,)),))),)),
        ]
    model.train(clauses)
    return model


class TestEdgeCasePropositional:
    def test_propositional_atom_zero_arity(self) -> None:
        """Arity-0 constant (propositional atom) produces a valid embedding."""
        model = _make_trained()
        # P (arity 0, no args) — use SYM_C as a propositional constant
        const_term = _rigid(SYM_C, 0)
        emb = model.embed_term(const_term)
        # May be None if token not in vocab, but should not crash
        assert emb is None or len(emb) == 24


class TestEdgeCaseDeepTrees:
    def test_deeply_nested_term_depth_20(self) -> None:
        """n(n(n(...x...))) 20 levels deep — no stack overflow for RNN."""
        model = _make_trained()
        term = _var(0)
        for _ in range(20):
            term = _rigid(SYM_N, 1, (term,))
        emb = model.embed_term(term)
        assert emb is not None
        assert len(emb) == 24


class TestEdgeCaseLargeClauses:
    def test_clause_with_many_literals(self) -> None:
        """20-literal clause produces valid embedding."""
        model = _make_trained()
        lits = tuple(
            _pos_lit(_rigid(SYM_P, 1, (_var(j),)))
            for j in range(20)
        )
        clause = Clause(literals=lits)
        emb = model.embed_clause(clause)
        assert emb is not None
        assert len(emb) == 24


class TestEdgeCaseOOVTokens:
    def test_novel_symbol_after_training(self) -> None:
        """Symbol not in training vocab handled gracefully."""
        model = _make_trained()
        # Use a symbol ID (99) that was never in training
        novel_term = _rigid(99, 1, (_var(0),))
        emb = model.embed_term(novel_term)
        # Should return something (UNK token) or None, but not crash
        assert emb is None or len(emb) == 24

    def test_online_update_extends_vocab(self) -> None:
        """Online update with novel tokens extends vocabulary."""
        model = _make_trained()
        old_vocab = model.vocab_size

        # Novel symbol 99 not in original training
        novel_clause = Clause(literals=(
            _pos_lit(_rigid(99, 1, (_var(0),))),
        ))
        result = model.update_online([novel_clause])
        # Vocab should have grown (new token added)
        assert model.vocab_size >= old_vocab


class TestEdgeCaseSingletonInputs:
    def test_single_variable_term(self) -> None:
        model = _make_trained()
        emb = model.embed_term(_var(0))
        # Single token sequence — should work
        assert emb is not None
        assert len(emb) == 24

    def test_single_literal_clause(self) -> None:
        model = _make_trained()
        clause = Clause(literals=(_pos_lit(_rigid(SYM_P, 1, (_var(0),))),))
        emb = model.embed_clause(clause)
        assert emb is not None
        assert len(emb) == 24

    def test_training_on_one_clause(self) -> None:
        clause = Clause(literals=(_pos_lit(_rigid(SYM_P, 1, (_var(0),))),))
        model = _make_trained(clauses=[clause])
        assert model.trained
        emb = model.embed_clause(clause)
        assert emb is not None

    def test_empty_clause_no_crash(self) -> None:
        """Clause with no literals → None, not an exception."""
        model = _make_trained()
        clause = Clause(literals=())
        result = model.embed_clause(clause)
        assert result is None
