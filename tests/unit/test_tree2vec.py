"""Unit tests for Tree2Vec core algorithm.

Tests tree walk generation, skip-gram training, and embedding composition
using simple term trees representative of the vampire.in domain.
"""

from __future__ import annotations

import math

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.ml.tree2vec.algorithm import Tree2Vec, Tree2VecConfig
from pyladr.ml.tree2vec.skipgram import SkipGramConfig, SkipGramTrainer
from pyladr.ml.tree2vec.walks import TreeWalker, WalkConfig, WalkType, _node_token


# ── Helpers: vampire.in domain terms ──────────────────────────────────────

# Symbol IDs for vampire.in vocabulary
SYM_P = 1   # P: unary predicate
SYM_I = 2   # i: binary function (inverse/product)
SYM_N = 3   # n: unary function (negation/complement)


def var(n: int) -> Term:
    """Create variable term."""
    return get_variable_term(n)


def n(arg: Term) -> Term:
    """n(arg) - unary function."""
    return get_rigid_term(SYM_N, 1, (arg,))


def i(left: Term, right: Term) -> Term:
    """i(left, right) - binary function."""
    return get_rigid_term(SYM_I, 2, (left, right))


def P(arg: Term) -> Term:
    """P(arg) - unary predicate."""
    return get_rigid_term(SYM_P, 1, (arg,))


def make_literal(sign: bool, atom: Term) -> Literal:
    return Literal(sign=sign, atom=atom)


def make_clause(*lits: Literal, clause_id: int = 0) -> Clause:
    return Clause(literals=lits, id=clause_id)


# ── Walk tests ────────────────────────────────────────────────────────────


class TestNodeToken:
    def test_variable_token(self) -> None:
        assert _node_token(var(0)) == "VAR"
        assert _node_token(var(5)) == "VAR"  # All vars normalized

    def test_constant_token(self) -> None:
        c = get_rigid_term(7, 0)
        assert _node_token(c) == "CONST:7"

    def test_complex_token(self) -> None:
        t = i(var(0), var(1))
        assert _node_token(t) == "FUNC:2/2"

    def test_position_encoding(self) -> None:
        assert _node_token(var(0), position=1, include_position=True) == "VAR@1"

    def test_depth_encoding(self) -> None:
        assert _node_token(var(0), depth=3, include_depth=True) == "VAR#3"


class TestTreeWalker:
    def setup_method(self) -> None:
        self.walker = TreeWalker(WalkConfig(
            walk_types=(WalkType.DEPTH_FIRST,),
            seed=42,
        ))

    def test_depth_first_variable(self) -> None:
        walks = self.walker.walks_from_term(var(0))
        assert len(walks) == 1
        assert walks[0] == ["VAR"]

    def test_depth_first_unary(self) -> None:
        # n(x)
        t = n(var(0))
        walks = self.walker.walks_from_term(t)
        assert len(walks) == 1
        assert walks[0] == ["FUNC:3/1", "VAR"]

    def test_depth_first_binary(self) -> None:
        # i(x, y)
        t = i(var(0), var(1))
        walks = self.walker.walks_from_term(t)
        assert len(walks) == 1
        assert walks[0] == ["FUNC:2/2", "VAR", "VAR"]

    def test_depth_first_nested(self) -> None:
        # i(n(x), y)
        t = i(n(var(0)), var(1))
        walks = self.walker.walks_from_term(t)
        assert len(walks) == 1
        assert walks[0] == ["FUNC:2/2", "FUNC:3/1", "VAR", "VAR"]

    def test_breadth_first(self) -> None:
        walker = TreeWalker(WalkConfig(
            walk_types=(WalkType.BREADTH_FIRST,),
        ))
        # i(n(x), y) -> BFS: i, n, y, x
        t = i(n(var(0)), var(1))
        walks = walker.walks_from_term(t)
        assert len(walks) == 1
        assert walks[0] == ["FUNC:2/2", "FUNC:3/1", "VAR", "VAR"]

    def test_path_walks(self) -> None:
        walker = TreeWalker(WalkConfig(
            walk_types=(WalkType.PATH,),
        ))
        # i(n(x), y) has two paths: i->n->x and i->y
        t = i(n(var(0)), var(1))
        walks = walker.walks_from_term(t)
        assert len(walks) == 2
        # Path to left leaf: i -> n -> x
        assert walks[0] == ["FUNC:2/2", "FUNC:3/1", "VAR"]
        # Path to right leaf: i -> y
        assert walks[1] == ["FUNC:2/2", "VAR"]

    def test_random_walks_count(self) -> None:
        walker = TreeWalker(WalkConfig(
            walk_types=(WalkType.RANDOM,),
            num_random_walks=5,
            seed=42,
        ))
        t = i(n(var(0)), var(1))
        walks = walker.walks_from_term(t)
        assert len(walks) == 5

    def test_random_walks_deterministic(self) -> None:
        cfg = WalkConfig(walk_types=(WalkType.RANDOM,), num_random_walks=5, seed=42)
        walker1 = TreeWalker(cfg)
        walker2 = TreeWalker(cfg)
        t = i(n(var(0)), i(var(1), var(2)))
        assert walker1.walks_from_term(t) == walker2.walks_from_term(t)

    def test_max_walk_length(self) -> None:
        walker = TreeWalker(WalkConfig(
            walk_types=(WalkType.DEPTH_FIRST,),
            max_walk_length=2,
        ))
        t = i(n(var(0)), var(1))
        walks = walker.walks_from_term(t)
        assert all(len(w) <= 2 for w in walks)

    def test_walks_from_literal(self) -> None:
        lit = make_literal(True, P(var(0)))
        walks = self.walker.walks_from_literal(lit)
        assert len(walks) == 1
        assert walks[0][0] == "LIT:+FUNC:1/1"

    def test_walks_from_clause(self) -> None:
        clause = make_clause(
            make_literal(True, P(var(0))),
            make_literal(False, P(n(var(1)))),
        )
        walks = self.walker.walks_from_clause(clause)
        assert len(walks) == 2
        assert all(w[0] == "CLAUSE:2" for w in walks)

    def test_all_walk_types(self) -> None:
        walker = TreeWalker(WalkConfig(
            walk_types=(WalkType.DEPTH_FIRST, WalkType.BREADTH_FIRST, WalkType.PATH, WalkType.RANDOM),
            num_random_walks=3,
            seed=42,
        ))
        t = i(n(var(0)), var(1))
        walks = walker.walks_from_term(t)
        # 1 DFS + 1 BFS + 2 paths + 3 random = 7
        assert len(walks) == 7


# ── Skip-gram tests ───────────────────────────────────────────────────────


class TestSkipGramTrainer:
    def test_empty_training(self) -> None:
        trainer = SkipGramTrainer()
        stats = trainer.train([])
        assert stats["vocab_size"] == 0
        assert stats["total_pairs"] == 0

    def test_single_token(self) -> None:
        trainer = SkipGramTrainer(SkipGramConfig(
            embedding_dim=8, num_epochs=1, seed=42,
        ))
        stats = trainer.train([["A"]])
        assert stats["vocab_size"] == 1
        # Single token can't form pairs
        assert stats["total_pairs"] == 0

    def test_vocab_building(self) -> None:
        trainer = SkipGramTrainer(SkipGramConfig(
            embedding_dim=8, num_epochs=1, seed=42,
        ))
        walks = [["A", "B", "C"], ["B", "C", "D"]]
        trainer.train(walks)
        assert trainer.vocab_size == 4
        tok_ids = trainer.token_to_id
        assert "A" in tok_ids
        assert "B" in tok_ids
        assert "C" in tok_ids
        assert "D" in tok_ids

    def test_embedding_dimensions(self) -> None:
        dim = 16
        trainer = SkipGramTrainer(SkipGramConfig(
            embedding_dim=dim, num_epochs=2, seed=42,
        ))
        walks = [["A", "B", "C", "A", "B"]]
        trainer.train(walks)
        emb = trainer.get_embedding("A")
        assert emb is not None
        assert len(emb) == dim

    def test_unknown_token_returns_none(self) -> None:
        trainer = SkipGramTrainer(SkipGramConfig(
            embedding_dim=8, num_epochs=1, seed=42,
        ))
        trainer.train([["A", "B"]])
        assert trainer.get_embedding("UNKNOWN") is None

    def test_most_similar(self) -> None:
        trainer = SkipGramTrainer(SkipGramConfig(
            embedding_dim=16, num_epochs=5, seed=42,
            window_size=2,
        ))
        # A and B always appear together, C appears in different context
        walks = [
            ["A", "B", "A", "B", "A", "B"],
            ["C", "D", "C", "D", "C", "D"],
        ]
        trainer.train(walks)
        similar = trainer.most_similar("A", top_k=3)
        assert len(similar) > 0
        # Each result is (token, similarity)
        assert all(isinstance(s[0], str) and isinstance(s[1], float) for s in similar)

    def test_deterministic_training(self) -> None:
        config = SkipGramConfig(embedding_dim=8, num_epochs=2, seed=42)
        walks = [["A", "B", "C"], ["B", "C", "A"]]

        trainer1 = SkipGramTrainer(config)
        trainer1.train(walks)

        trainer2 = SkipGramTrainer(config)
        trainer2.train(walks)

        emb1 = trainer1.get_embedding("A")
        emb2 = trainer2.get_embedding("A")
        assert emb1 is not None and emb2 is not None
        assert emb1 == emb2  # Exact match with same seed


# ── Tree2Vec integration tests ────────────────────────────────────────────


class TestTree2Vec:
    def _make_vampire_clauses(self) -> list[Clause]:
        """Create a small set of clauses mimicking vampire.in patterns."""
        x, y, z = var(0), var(1), var(2)
        clauses = [
            # P(i(x,n(x)))  - group inverse axiom
            make_clause(make_literal(True, P(i(x, n(x)))), clause_id=1),
            # P(i(n(x),x))  - inverse other direction
            make_clause(make_literal(True, P(i(n(x), x))), clause_id=2),
            # -P(i(x,y)) | -P(i(y,z)) | P(i(x,z))  - transitivity-like
            make_clause(
                make_literal(False, P(i(x, y))),
                make_literal(False, P(i(y, z))),
                make_literal(True, P(i(x, z))),
                clause_id=3,
            ),
            # P(i(i(x,y),i(n(y),n(x))))  - nested structure
            make_clause(
                make_literal(True, P(i(i(x, y), i(n(y), n(x))))),
                clause_id=4,
            ),
        ]
        return clauses

    def test_train_and_embed(self) -> None:
        config = Tree2VecConfig(
            walk_config=WalkConfig(
                walk_types=(WalkType.DEPTH_FIRST, WalkType.PATH),
                seed=42,
            ),
            skipgram_config=SkipGramConfig(
                embedding_dim=32,
                num_epochs=3,
                seed=42,
            ),
        )
        t2v = Tree2Vec(config)
        clauses = self._make_vampire_clauses()
        stats = t2v.train(clauses)

        assert stats["vocab_size"] > 0
        assert stats["total_pairs"] > 0
        assert t2v.trained

    def test_embed_term(self) -> None:
        config = Tree2VecConfig(
            walk_config=WalkConfig(
                walk_types=(WalkType.DEPTH_FIRST, WalkType.PATH),
                seed=42,
            ),
            skipgram_config=SkipGramConfig(
                embedding_dim=16,
                num_epochs=3,
                seed=42,
            ),
        )
        t2v = Tree2Vec(config)
        t2v.train(self._make_vampire_clauses())

        # Embed a known term
        emb = t2v.embed_term(i(var(0), n(var(1))))
        assert emb is not None
        assert len(emb) == 16

        # Normalized embeddings should have unit norm
        norm = math.sqrt(sum(v * v for v in emb))
        assert abs(norm - 1.0) < 1e-6

    def test_embed_clause(self) -> None:
        config = Tree2VecConfig(
            walk_config=WalkConfig(
                walk_types=(WalkType.DEPTH_FIRST,),
                seed=42,
            ),
            skipgram_config=SkipGramConfig(
                embedding_dim=16,
                num_epochs=3,
                seed=42,
            ),
        )
        t2v = Tree2Vec(config)
        clauses = self._make_vampire_clauses()
        t2v.train(clauses)

        emb = t2v.embed_clause(clauses[0])
        assert emb is not None
        assert len(emb) == 16

    def test_similarity_structurally_similar(self) -> None:
        """Structurally similar terms should have higher similarity."""
        config = Tree2VecConfig(
            walk_config=WalkConfig(
                walk_types=(WalkType.DEPTH_FIRST, WalkType.PATH, WalkType.RANDOM),
                num_random_walks=10,
                seed=42,
            ),
            skipgram_config=SkipGramConfig(
                embedding_dim=32,
                num_epochs=10,
                window_size=3,
                seed=42,
            ),
        )
        t2v = Tree2Vec(config)
        t2v.train(self._make_vampire_clauses())

        x, y = var(0), var(1)
        # i(x, n(x)) vs i(y, n(y)) - same structure, different vars
        sim_same = t2v.similarity(i(x, n(x)), i(y, n(y)))
        assert sim_same is not None
        # Same structure should yield identical embeddings (vars normalized)
        assert sim_same > 0.99

    def test_untrained_returns_none(self) -> None:
        t2v = Tree2Vec()
        assert t2v.embed_term(var(0)) is None
        assert t2v.embed_clause(make_clause()) is None

    def test_composition_weighted_depth(self) -> None:
        config = Tree2VecConfig(
            walk_config=WalkConfig(
                walk_types=(WalkType.DEPTH_FIRST,),
                seed=42,
            ),
            skipgram_config=SkipGramConfig(
                embedding_dim=16,
                num_epochs=3,
                seed=42,
            ),
            composition="weighted_depth",
        )
        t2v = Tree2Vec(config)
        t2v.train(self._make_vampire_clauses())

        emb = t2v.embed_term(i(var(0), n(var(1))))
        assert emb is not None
        assert len(emb) == 16

    def test_composition_root_concat(self) -> None:
        config = Tree2VecConfig(
            walk_config=WalkConfig(
                walk_types=(WalkType.DEPTH_FIRST,),
                seed=42,
            ),
            skipgram_config=SkipGramConfig(
                embedding_dim=16,
                num_epochs=3,
                seed=42,
            ),
            composition="root_concat",
        )
        t2v = Tree2Vec(config)
        t2v.train(self._make_vampire_clauses())

        emb = t2v.embed_term(i(var(0), n(var(1))))
        assert emb is not None
        # root_concat doubles the dimension
        assert len(emb) == 32
        assert t2v.embedding_dim == 32

    def test_batch_embedding(self) -> None:
        config = Tree2VecConfig(
            walk_config=WalkConfig(walk_types=(WalkType.DEPTH_FIRST,), seed=42),
            skipgram_config=SkipGramConfig(embedding_dim=16, num_epochs=2, seed=42),
        )
        t2v = Tree2Vec(config)
        clauses = self._make_vampire_clauses()
        t2v.train(clauses)

        results = t2v.embed_clauses_batch(clauses)
        assert len(results) == len(clauses)
        assert all(r is not None for r in results)

    def test_train_from_terms(self) -> None:
        config = Tree2VecConfig(
            walk_config=WalkConfig(walk_types=(WalkType.DEPTH_FIRST,), seed=42),
            skipgram_config=SkipGramConfig(embedding_dim=16, num_epochs=2, seed=42),
        )
        t2v = Tree2Vec(config)
        terms = [i(var(0), n(var(1))), n(i(var(0), var(1)))]
        stats = t2v.train_from_terms(terms)
        assert stats["vocab_size"] > 0
        assert t2v.trained

    def test_most_similar_tokens(self) -> None:
        config = Tree2VecConfig(
            walk_config=WalkConfig(
                walk_types=(WalkType.DEPTH_FIRST, WalkType.PATH),
                seed=42,
            ),
            skipgram_config=SkipGramConfig(
                embedding_dim=16,
                num_epochs=3,
                seed=42,
            ),
        )
        t2v = Tree2Vec(config)
        t2v.train(self._make_vampire_clauses())

        similar = t2v.most_similar_tokens("FUNC:2/2", top_k=3)
        assert isinstance(similar, list)
        # Should find some similar tokens
        assert len(similar) > 0
