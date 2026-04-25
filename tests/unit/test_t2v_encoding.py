"""Unit tests for Tree2Vec encoding features.

Tests skip-predicate wrapper, path-length tokens, variable depth encoding,
weighted_depth default composition, and SearchOptions defaults.
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.ml.tree2vec.algorithm import Tree2Vec, Tree2VecConfig
from pyladr.ml.tree2vec.walks import TreeWalker, WalkConfig, WalkType, _node_token


# ── Helpers ──────────────────────────────────────────────────────────────

SYM_P = 1   # P: unary predicate
SYM_Q = 4   # Q: unary predicate (different from P)
SYM_I = 2   # i: binary function
SYM_N = 3   # n: unary function


def var(n: int) -> Term:
    return get_variable_term(n)


def n(arg: Term) -> Term:
    return get_rigid_term(SYM_N, 1, (arg,))


def i(left: Term, right: Term) -> Term:
    return get_rigid_term(SYM_I, 2, (left, right))


def P(arg: Term) -> Term:
    return get_rigid_term(SYM_P, 1, (arg,))


def Q(arg: Term) -> Term:
    return get_rigid_term(SYM_Q, 1, (arg,))


def prop_atom(symnum: int = 10) -> Term:
    """Propositional (arity-0) atom."""
    return get_rigid_term(symnum, 0)


def make_literal(sign: bool, atom: Term) -> Literal:
    return Literal(sign=sign, atom=atom)


def make_clause(*lits: Literal, clause_id: int = 0) -> Clause:
    return Clause(literals=lits, id=clause_id)


def _train_t2v(config: Tree2VecConfig) -> Tree2Vec:
    """Train a Tree2Vec on a small clause set and return it."""
    x, y, z = var(0), var(1), var(2)
    clauses = [
        make_clause(make_literal(True, P(i(x, n(x)))), clause_id=1),
        make_clause(make_literal(True, P(i(n(x), x))), clause_id=2),
        make_clause(
            make_literal(False, P(i(x, y))),
            make_literal(False, P(i(y, z))),
            make_literal(True, P(i(x, z))),
            clause_id=3,
        ),
    ]
    t2v = Tree2Vec(config)
    t2v.train(clauses)
    return t2v


# ── TestSkipPredicateWrapper ─────────────────────────────────────────────


class TestSkipPredicateWrapper:
    def test_walks_start_with_sign_token(self) -> None:
        walker = TreeWalker(WalkConfig(
            walk_types=(WalkType.DEPTH_FIRST,),
            skip_predicate_wrapper=True,
            seed=42,
        ))
        clause = make_clause(make_literal(True, P(var(0))))
        walks = walker.walks_from_clause(clause)
        assert len(walks) >= 1
        # Each walk: CLAUSE:N, SIGN:+/-, ...
        for walk in walks:
            assert walk[1] in ("SIGN:+", "SIGN:-")

    def test_no_predicate_token_in_walks(self) -> None:
        walker = TreeWalker(WalkConfig(
            walk_types=(WalkType.DEPTH_FIRST,),
            skip_predicate_wrapper=True,
            seed=42,
        ))
        clause = make_clause(
            make_literal(True, P(i(var(0), var(1)))),
        )
        walks = walker.walks_from_clause(clause)
        for walk in walks:
            for tok in walk:
                # No predicate FUNC token (P is SYM_P=1, arity 1)
                assert tok != f"FUNC:{SYM_P}/1"
                # No LIT: prefix tokens either
                assert not tok.startswith("LIT:")

    def test_negative_literal_sign(self) -> None:
        walker = TreeWalker(WalkConfig(
            walk_types=(WalkType.DEPTH_FIRST,),
            skip_predicate_wrapper=True,
            seed=42,
        ))
        clause = make_clause(make_literal(False, P(var(0))))
        walks = walker.walks_from_clause(clause)
        assert walks[0][1] == "SIGN:-"

    def test_propositional_fallback(self) -> None:
        """Propositional atoms (arity 0) fall back to normal literal walks."""
        walker = TreeWalker(WalkConfig(
            walk_types=(WalkType.DEPTH_FIRST,),
            skip_predicate_wrapper=True,
            seed=42,
        ))
        clause = make_clause(make_literal(True, prop_atom()))
        walks = walker.walks_from_clause(clause)
        assert len(walks) >= 1
        # Propositional fallback uses LIT: prefix, not SIGN:
        assert walks[0][1].startswith("LIT:")

    def test_embed_clause_with_skip_predicate(self) -> None:
        config = Tree2VecConfig(
            walk_config=WalkConfig(
                walk_types=(WalkType.DEPTH_FIRST, WalkType.PATH),
                skip_predicate_wrapper=True,
                seed=42,
            ),
            skipgram_config=_sgc(16),
        )
        t2v = _train_t2v(config)
        clause = make_clause(make_literal(True, P(i(var(0), n(var(1))))), clause_id=99)
        emb = t2v.embed_clause(clause)
        assert emb is not None
        assert len(emb) == 16

    def test_same_inner_structure_different_predicates(self) -> None:
        """Two clauses with same inner structure but different predicates produce same embedding."""
        config = Tree2VecConfig(
            walk_config=WalkConfig(
                walk_types=(WalkType.DEPTH_FIRST, WalkType.PATH),
                skip_predicate_wrapper=True,
                seed=42,
            ),
            skipgram_config=_sgc(32, epochs=5),
        )
        x = var(0)
        inner = i(x, n(x))
        clauses = [
            make_clause(make_literal(True, P(inner)), clause_id=1),
            make_clause(make_literal(True, Q(inner)), clause_id=2),
            # Need more training data
            make_clause(make_literal(True, P(i(n(x), x))), clause_id=3),
            make_clause(make_literal(True, Q(i(n(x), x))), clause_id=4),
        ]
        t2v = Tree2Vec(config)
        t2v.train(clauses)

        emb_p = t2v.embed_clause(clauses[0])
        emb_q = t2v.embed_clause(clauses[1])
        assert emb_p is not None and emb_q is not None
        # With skip_predicate, P and Q wrapper are removed — same inner structure
        sim = Tree2Vec._cosine_similarity(emb_p, emb_q)
        assert sim > 0.99, f"Expected high similarity, got {sim}"


# ── TestPathLengthTokens ────────────────────────────────────────────────


class TestPathLengthTokens:
    def test_path_walks_include_pathlen_token(self) -> None:
        walker = TreeWalker(WalkConfig(
            walk_types=(WalkType.PATH,),
            include_path_length=True,
            seed=42,
        ))
        t = i(n(var(0)), var(1))
        walks = walker.walks_from_term(t)
        assert len(walks) == 2
        for walk in walks:
            assert walk[0].startswith("PATHLEN:")

    def test_pathlen_values_match_walk_length(self) -> None:
        walker = TreeWalker(WalkConfig(
            walk_types=(WalkType.PATH,),
            include_path_length=True,
            seed=42,
        ))
        t = i(n(var(0)), var(1))
        walks = walker.walks_from_term(t)
        for walk in walks:
            # PATHLEN:N prepended, so walk content after PATHLEN is N tokens
            pathlen_tok = walk[0]
            n_val = int(pathlen_tok.split(":")[1])
            # The path length value = number of tokens in the path (excluding PATHLEN itself)
            assert n_val == len(walk) - 1

    def test_deep_term_higher_pathlen_than_shallow(self) -> None:
        walker = TreeWalker(WalkConfig(
            walk_types=(WalkType.PATH,),
            include_path_length=True,
            seed=42,
        ))
        # Deep: i(n(n(var(0))), var(1)) — left path has 4 nodes
        deep = i(n(n(var(0))), var(1))
        # Shallow: i(var(0), var(1)) — paths have 2 nodes
        shallow = i(var(0), var(1))

        deep_walks = walker.walks_from_term(deep)
        shallow_walks = walker.walks_from_term(shallow)

        deep_max = max(int(w[0].split(":")[1]) for w in deep_walks)
        shallow_max = max(int(w[0].split(":")[1]) for w in shallow_walks)
        assert deep_max > shallow_max

    def test_no_pathlen_when_disabled(self) -> None:
        walker = TreeWalker(WalkConfig(
            walk_types=(WalkType.PATH,),
            include_path_length=False,
            seed=42,
        ))
        t = i(var(0), var(1))
        walks = walker.walks_from_term(t)
        for walk in walks:
            assert not walk[0].startswith("PATHLEN:")

    def test_embed_clause_with_path_length(self) -> None:
        config = Tree2VecConfig(
            walk_config=WalkConfig(
                walk_types=(WalkType.DEPTH_FIRST, WalkType.PATH),
                include_path_length=True,
                seed=42,
            ),
            skipgram_config=_sgc(16),
        )
        t2v = _train_t2v(config)
        clause = make_clause(make_literal(True, P(i(var(0), n(var(1))))), clause_id=99)
        emb = t2v.embed_clause(clause)
        assert emb is not None


# ── TestVariableDepthEncoding ───────────────────────────────────────────


class TestVariableDepthEncoding:
    def test_var_depth_0(self) -> None:
        assert _node_token(var(0), depth=0, include_depth=True) == "VAR#0"

    def test_var_depth_2(self) -> None:
        assert _node_token(var(0), depth=2, include_depth=True) == "VAR#2"

    def test_func_depth_encoding(self) -> None:
        t = i(var(0), var(1))
        tok = _node_token(t, depth=1, include_depth=True)
        assert tok == "FUNC:2/2#1"

    def test_embed_clause_different_depths_different_embeddings(self) -> None:
        """Clauses with variables at different depths produce different embeddings."""
        config = Tree2VecConfig(
            walk_config=WalkConfig(
                walk_types=(WalkType.DEPTH_FIRST, WalkType.PATH),
                include_depth=True,
                seed=42,
            ),
            skipgram_config=_sgc(16, epochs=5),
        )
        # Shallow: P(x) — var at depth 1
        # Deep: P(n(n(x))) — var at depth 3
        x = var(0)
        shallow_clause = make_clause(make_literal(True, P(x)), clause_id=1)
        deep_clause = make_clause(make_literal(True, P(n(n(x)))), clause_id=2)
        extra = make_clause(make_literal(True, P(i(x, n(x)))), clause_id=3)

        t2v = Tree2Vec(config)
        t2v.train([shallow_clause, deep_clause, extra])

        emb_shallow = t2v.embed_clause(shallow_clause)
        emb_deep = t2v.embed_clause(deep_clause)
        assert emb_shallow is not None and emb_deep is not None
        # They should differ (different depth tokens)
        assert emb_shallow != emb_deep


# ── TestWeightedDepthDefault ────────────────────────────────────────────


class TestWeightedDepthDefault:
    def test_tree2vec_config_default_composition(self) -> None:
        config = Tree2VecConfig()
        assert config.composition == "weighted_depth"

    def test_search_options_default_composition(self) -> None:
        from pyladr.search.given_clause import SearchOptions
        opts = SearchOptions()
        assert opts.tree2vec_composition == "weighted_depth"

    def test_embed_uses_weighted_depth_by_default(self) -> None:
        config = Tree2VecConfig(
            walk_config=WalkConfig(
                walk_types=(WalkType.DEPTH_FIRST,),
                seed=42,
            ),
            skipgram_config=_sgc(16),
        )
        assert config.composition == "weighted_depth"
        t2v = _train_t2v(config)
        emb = t2v.embed_term(i(var(0), n(var(1))))
        assert emb is not None
        assert len(emb) == 16


# ── TestDefaultsEnabledByDefault ────────────────────────────────────────


class TestDefaultsEnabledByDefault:
    def test_skip_predicate_true_by_default(self) -> None:
        from pyladr.search.given_clause import SearchOptions
        opts = SearchOptions(tree2vec_embeddings=True)
        assert opts.tree2vec_skip_predicate is True

    def test_path_length_true_by_default(self) -> None:
        from pyladr.search.given_clause import SearchOptions
        opts = SearchOptions(tree2vec_embeddings=True)
        assert opts.tree2vec_include_path_length is True

    def test_can_disable_skip_predicate(self) -> None:
        from pyladr.search.given_clause import SearchOptions
        opts = SearchOptions(tree2vec_embeddings=True, tree2vec_skip_predicate=False)
        assert opts.tree2vec_skip_predicate is False

    def test_can_disable_path_length(self) -> None:
        from pyladr.search.given_clause import SearchOptions
        opts = SearchOptions(tree2vec_embeddings=True, tree2vec_include_path_length=False)
        assert opts.tree2vec_include_path_length is False

    def test_composition_default_weighted_depth(self) -> None:
        from pyladr.search.given_clause import SearchOptions
        opts = SearchOptions(tree2vec_embeddings=True)
        assert opts.tree2vec_composition == "weighted_depth"


# ── Helper for skip-gram configs ────────────────────────────────────────

def _sgc(dim: int = 16, epochs: int = 3) -> "SkipGramConfig":
    from pyladr.ml.tree2vec.skipgram import SkipGramConfig
    return SkipGramConfig(embedding_dim=dim, num_epochs=epochs, seed=42)
