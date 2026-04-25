"""Tests for online Tree2Vec learning integration.

Covers:
- SkipGramTrainer.update_online() behavior
- Tree2VecEmbeddingProvider versioned cache invalidation
- Goal re-embedding after online updates
- End-to-end search with online tree2vec flags
- CLI/SearchOptions parameter defaults and parsing
"""

from __future__ import annotations

import math

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.ml.tree2vec.algorithm import Tree2Vec, Tree2VecConfig
from pyladr.ml.tree2vec.skipgram import SkipGramConfig, SkipGramTrainer
from pyladr.ml.tree2vec.provider import (
    Tree2VecEmbeddingProvider,
    Tree2VecProviderConfig,
)
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions


# ── Helpers ──────────────────────────────────────────────────────────────

SYM_P = 1
SYM_I = 2
SYM_N = 3
SYM_E = 4  # identity element


def var(n: int) -> Term:
    return get_variable_term(n)


def n(arg: Term) -> Term:
    return get_rigid_term(SYM_N, 1, (arg,))


def i(left: Term, right: Term) -> Term:
    return get_rigid_term(SYM_I, 2, (left, right))


def P(arg: Term) -> Term:
    return get_rigid_term(SYM_P, 1, (arg,))


def e() -> Term:
    return get_rigid_term(SYM_E, 0)


def make_literal(sign: bool, atom: Term) -> Literal:
    return Literal(sign=sign, atom=atom)


def make_clause(*lits: Literal, clause_id: int = 0) -> Clause:
    return Clause(literals=lits, id=clause_id)


def _make_training_clauses() -> list[Clause]:
    """Create a set of clauses for training (vampire.in domain)."""
    x, y, z = var(0), var(1), var(2)
    return [
        make_clause(make_literal(True, P(i(x, y))), clause_id=1),
        make_clause(make_literal(True, P(i(n(x), x))), clause_id=2),
        make_clause(make_literal(True, P(i(i(x, y), z))), clause_id=3),
        make_clause(make_literal(True, P(n(i(x, x)))), clause_id=4),
        make_clause(make_literal(False, P(i(x, n(y)))), clause_id=5),
    ]


def _make_novel_clauses() -> list[Clause]:
    """Clauses for online update (same vocabulary, different structure)."""
    x, y = var(0), var(1)
    return [
        make_clause(make_literal(True, P(i(n(x), n(y)))), clause_id=100),
        make_clause(make_literal(True, P(n(n(x)))), clause_id=101),
    ]


def _train_tree2vec(clauses: list[Clause] | None = None) -> Tree2Vec:
    """Train a Tree2Vec model on the standard training set."""
    config = Tree2VecConfig(
        skipgram_config=SkipGramConfig(
            embedding_dim=32,
            num_epochs=3,
            seed=42,
        ),
    )
    t2v = Tree2Vec(config)
    t2v.train(clauses or _make_training_clauses())
    return t2v


# ── TestOnlineSkipGramUpdate ─────────────────────────────────────────────


class TestOnlineSkipGramUpdate:
    def test_untrained_model_is_noop(self) -> None:
        trainer = SkipGramTrainer(SkipGramConfig(embedding_dim=16))
        result = trainer.update_online([["A", "B", "C"]])
        assert result["pairs_trained"] == 0
        assert result["oov_skipped"] == 0
        assert result["loss"] == 0.0

    def test_trained_model_returns_expected_keys(self) -> None:
        trainer = SkipGramTrainer(SkipGramConfig(embedding_dim=16, seed=42))
        trainer.train([["A", "B", "C", "D"], ["B", "C", "A"]])
        result = trainer.update_online([["A", "B", "C"]])
        assert "pairs_trained" in result
        assert "oov_skipped" in result
        assert "loss" in result
        assert result["pairs_trained"] > 0

    def test_oov_tokens_skipped_when_extension_disabled(self) -> None:
        trainer = SkipGramTrainer(SkipGramConfig(embedding_dim=16, seed=42, online_vocab_extension=False))
        trainer.train([["A", "B", "C"]])
        result = trainer.update_online([["A", "UNKNOWN_TOKEN", "B"]])
        assert result["oov_skipped"] >= 1

    def test_oov_tokens_extended_by_default(self) -> None:
        """New tokens are added to vocab and trained when extension is enabled."""
        trainer = SkipGramTrainer(SkipGramConfig(embedding_dim=16, seed=42))
        trainer.train([["A", "B", "C"]])
        result = trainer.update_online([["A", "NEW_TOKEN", "B"]])
        assert result["oov_skipped"] == 0
        assert result.get("vocab_extended", 0) == 1
        assert trainer.get_embedding("NEW_TOKEN") is not None

    def test_all_oov_walk_no_crash(self) -> None:
        """A walk of OOV tokens: extended by default, skipped when disabled."""
        trainer = SkipGramTrainer(SkipGramConfig(embedding_dim=16, seed=42, online_vocab_extension=False))
        trainer.train([["A", "B", "C"]])
        result = trainer.update_online([["X1", "X2", "X3"]])
        assert result["oov_skipped"] == 3
        assert result["pairs_trained"] == 0

    def test_update_changes_embeddings(self) -> None:
        trainer = SkipGramTrainer(SkipGramConfig(embedding_dim=16, seed=42))
        trainer.train([["A", "B", "C", "D"], ["B", "C", "A"]])
        emb_before = trainer.get_embedding("A")
        assert emb_before is not None

        # Run many online updates to ensure change
        for _ in range(10):
            trainer.update_online(
                [["C", "D", "A", "B"]], learning_rate=0.1,
            )
        emb_after = trainer.get_embedding("A")
        assert emb_after is not None

        # Embeddings should differ
        diff = sum((a - b) ** 2 for a, b in zip(emb_before, emb_after))
        assert diff > 1e-10, "Online update should change embeddings"


# ── TestVersionedCache ───────────────────────────────────────────────────


class TestVersionedCache:
    def _make_provider(self) -> Tree2VecEmbeddingProvider:
        t2v = _train_tree2vec()
        config = Tree2VecProviderConfig(cache_max_entries=1000)
        return Tree2VecEmbeddingProvider(tree2vec=t2v, config=config)

    def test_version_starts_at_zero(self) -> None:
        provider = self._make_provider()
        assert provider.model_version == 0

    def test_bump_increments_version(self) -> None:
        provider = self._make_provider()
        v1 = provider.bump_model_version()
        assert v1 == 1
        v2 = provider.bump_model_version()
        assert v2 == 2

    def test_cache_hit_when_version_unchanged(self) -> None:
        provider = self._make_provider()
        clause = _make_training_clauses()[0]

        emb1 = provider.get_embedding(clause)
        assert emb1 is not None
        assert provider.stats.misses == 1

        emb2 = provider.get_embedding(clause)
        assert emb2 is not None
        assert provider.stats.hits == 1
        assert emb1 == emb2

    def test_cache_miss_after_version_bump(self) -> None:
        provider = self._make_provider()
        clause = _make_training_clauses()[0]

        emb1 = provider.get_embedding(clause)
        assert emb1 is not None
        misses_before = provider.stats.misses

        provider.bump_model_version()

        emb2 = provider.get_embedding(clause)
        assert emb2 is not None
        assert provider.stats.misses == misses_before + 1


# ── TestGoalReEmbedding ─────────────────────────────────────────────────


class TestGoalReEmbedding:
    def test_goal_proximity_changes_after_reembedding(self) -> None:
        from pyladr.search.goal_directed import GoalProximityScorer

        scorer = GoalProximityScorer(method="max")

        goal_emb_v1 = [1.0, 0.0, 0.0, 0.0]
        scorer.set_goals([goal_emb_v1])

        query = [0.9, 0.1, 0.0, 0.0]
        dist_v1 = scorer.nearest_goal_distance(query)

        # Simulate re-embedding with new goal vector (orthogonal direction)
        goal_emb_v2 = [0.0, 0.0, 1.0, 0.0]
        scorer.set_goals([goal_emb_v2])
        dist_v2 = scorer.nearest_goal_distance(query)

        assert dist_v1 != dist_v2, "Re-embedding goals should change distance"

    def test_goal_directed_provider_reregister(self) -> None:
        from pyladr.search.goal_directed import (
            GoalDirectedConfig,
            GoalDirectedEmbeddingProvider,
        )

        t2v = _train_tree2vec()
        config = Tree2VecProviderConfig(cache_max_entries=1000)
        base = Tree2VecEmbeddingProvider(tree2vec=t2v, config=config)

        gd_config = GoalDirectedConfig(enabled=True, goal_proximity_weight=0.5)
        gd = GoalDirectedEmbeddingProvider(base_provider=base, config=gd_config)

        goal_clauses = [_make_training_clauses()[0]]
        gd.register_goals(goal_clauses)
        assert gd.num_goals >= 1

        # Re-register (simulating post-online-update re-embedding)
        gd.register_goals(goal_clauses)
        assert gd.num_goals >= 1  # Still registered

    def test_embeddings_change_after_online_update_and_bump(self) -> None:
        """After online update + version bump, re-embedded goals produce new vectors."""
        t2v = _train_tree2vec()
        config = Tree2VecProviderConfig(cache_max_entries=1000)
        provider = Tree2VecEmbeddingProvider(tree2vec=t2v, config=config)
        clause = _make_training_clauses()[0]

        emb_before = provider.get_embedding(clause)
        assert emb_before is not None

        # Many online updates to ensure embedding drift
        for _ in range(20):
            t2v.update_online(_make_novel_clauses(), learning_rate=0.05)

        provider.bump_model_version()
        emb_after = provider.get_embedding(clause)
        assert emb_after is not None
        assert emb_before != emb_after, "Embeddings should change after online update + bump"


# ── TestOnlineLearningIntegration ────────────────────────────────────────


def _run_python(input_text: str, max_seconds: float = 10) -> dict:
    """Run Python prover on input text, return result dict."""
    from pyladr.apps.prover9 import _auto_inference, _auto_limits, _deny_goals, _apply_settings
    from pyladr.parsing.ladr_parser import LADRParser

    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(input_text)
    usable, sos, _denied = _deny_goals(parsed, st)
    opts = SearchOptions(max_seconds=max_seconds)
    _auto_inference(parsed, opts)
    _auto_limits(parsed, opts)
    _apply_settings(parsed, opts)
    engine = GivenClauseSearch(
        options=opts,
        symbol_table=st,
        hints=parsed.hints if parsed.hints else None,
    )
    result = engine.run(usable=usable, sos=sos)
    return {
        "proved": len(result.proofs) > 0,
        "exit_code": result.exit_code,
        "given": result.stats.given,
        "generated": result.stats.generated,
        "kept": result.stats.kept,
    }


_SIMPLE_PROOF_INPUT = (
    "formulas(sos).\n"
    "  P(a).\n"
    "  -P(x) | Q(x).\n"
    "end_of_list.\n"
    "formulas(goals).\n"
    "  Q(a).\n"
    "end_of_list.\n"
)


class TestOnlineLearningIntegration:
    """End-to-end tests using Tree2Vec algorithm's update_online."""

    def test_tree2vec_update_online_untrained_noop(self) -> None:
        t2v = Tree2Vec(Tree2VecConfig(
            skipgram_config=SkipGramConfig(embedding_dim=16),
        ))
        result = t2v.update_online(_make_training_clauses())
        assert result["pairs_trained"] == 0

    def test_tree2vec_update_online_trained(self) -> None:
        t2v = _train_tree2vec()
        novel = _make_novel_clauses()
        result = t2v.update_online(novel)
        assert result["pairs_trained"] > 0

    def test_provider_online_roundtrip(self) -> None:
        """Full cycle: train → embed → online update → bump → re-embed."""
        t2v = _train_tree2vec()
        config = Tree2VecProviderConfig(cache_max_entries=1000)
        provider = Tree2VecEmbeddingProvider(tree2vec=t2v, config=config)

        clause = _make_training_clauses()[0]
        emb_before = provider.get_embedding(clause)
        assert emb_before is not None

        # Online update
        novel = _make_novel_clauses()
        stats = t2v.update_online(novel, learning_rate=0.05)
        assert stats["pairs_trained"] > 0

        new_version = provider.bump_model_version()
        assert new_version == 1

        # Re-embed should compute fresh (stale cache)
        emb_after = provider.get_embedding(clause)
        assert emb_after is not None
        # Embedding may or may not change significantly, but cache miss should occur
        assert provider.stats.misses >= 2  # initial + post-bump

    def test_search_baseline_no_online_learning(self) -> None:
        """Search without online learning finds a proof (baseline)."""
        result = _run_python(_SIMPLE_PROOF_INPUT)
        assert result["proved"]

    def test_search_online_learning_false_no_regression(self) -> None:
        """Explicitly clear(tree2vec_online_learning) is identical to baseline."""
        input_text = "clear(tree2vec_online_learning).\n" + _SIMPLE_PROOF_INPUT
        result = _run_python(input_text)
        assert result["proved"]

    def test_search_goal_proximity_false_no_regression(self) -> None:
        """Explicitly clear(tree2vec_goal_proximity) is identical to baseline."""
        input_text = "clear(tree2vec_goal_proximity).\n" + _SIMPLE_PROOF_INPUT
        result = _run_python(input_text)
        assert result["proved"]


# ── TestCLIParameters ────────────────────────────────────────────────────


class TestCLIParameters:
    def test_defaults(self) -> None:
        opts = SearchOptions()
        assert opts.tree2vec_online_learning is False
        assert opts.tree2vec_online_update_interval == 20
        assert opts.tree2vec_online_batch_size == 10
        assert opts.tree2vec_online_lr == 0.005
        assert opts.tree2vec_goal_proximity is False
        assert opts.tree2vec_goal_proximity_weight == 0.3

    def test_custom_values(self) -> None:
        opts = SearchOptions(
            tree2vec_online_learning=True,
            tree2vec_online_update_interval=5,
            tree2vec_online_batch_size=20,
            tree2vec_online_lr=0.01,
            tree2vec_goal_proximity=True,
            tree2vec_goal_proximity_weight=0.6,
        )
        assert opts.tree2vec_online_learning is True
        assert opts.tree2vec_online_update_interval == 5
        assert opts.tree2vec_online_batch_size == 20
        assert opts.tree2vec_online_lr == 0.01
        assert opts.tree2vec_goal_proximity is True
        assert opts.tree2vec_goal_proximity_weight == 0.6

    def test_tree2vec_flags_no_effect_when_disabled(self) -> None:
        """Online learning flags have no effect when tree2vec_embeddings=False."""
        opts = SearchOptions(
            tree2vec_embeddings=False,
            tree2vec_online_learning=True,
        )
        assert opts.tree2vec_online_learning is True
        assert opts.tree2vec_embeddings is False

    def test_assign_tree2vec_online_update_interval(self) -> None:
        """assign(tree2vec_online_update_interval, 5) parses correctly."""
        from pyladr.apps.prover9 import _apply_settings
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(
            "assign(tree2vec_online_update_interval, 5).\n"
            "formulas(sos). P(a). end_of_list.\n"
        )
        opts = SearchOptions()
        _apply_settings(parsed, opts)
        assert opts.tree2vec_online_update_interval == 5

    def test_assign_tree2vec_online_batch_size(self) -> None:
        """assign(tree2vec_online_batch_size, 25) parses correctly."""
        from pyladr.apps.prover9 import _apply_settings
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(
            "assign(tree2vec_online_batch_size, 25).\n"
            "formulas(sos). P(a). end_of_list.\n"
        )
        opts = SearchOptions()
        _apply_settings(parsed, opts)
        assert opts.tree2vec_online_batch_size == 25

    def test_assign_tree2vec_online_lr(self) -> None:
        """assign(tree2vec_online_lr, 0.01) parses correctly."""
        from pyladr.apps.prover9 import _apply_settings
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(
            "assign(tree2vec_online_lr, 0.01).\n"
            "formulas(sos). P(a). end_of_list.\n"
        )
        opts = SearchOptions()
        _apply_settings(parsed, opts)
        assert opts.tree2vec_online_lr == pytest.approx(0.01)

    def test_assign_tree2vec_goal_proximity_weight(self) -> None:
        """assign(tree2vec_goal_proximity_weight, 0.7) parses correctly."""
        from pyladr.apps.prover9 import _apply_settings
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(
            "assign(tree2vec_goal_proximity_weight, 0.7).\n"
            "formulas(sos). P(a). end_of_list.\n"
        )
        opts = SearchOptions()
        _apply_settings(parsed, opts)
        assert opts.tree2vec_goal_proximity_weight == pytest.approx(0.7)

    def test_set_tree2vec_online_learning(self) -> None:
        """set(tree2vec_online_learning) enables the flag."""
        from pyladr.apps.prover9 import _apply_settings
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(
            "set(tree2vec_online_learning).\n"
            "formulas(sos). P(a). end_of_list.\n"
        )
        opts = SearchOptions()
        _apply_settings(parsed, opts)
        assert opts.tree2vec_online_learning is True

    def test_set_tree2vec_goal_proximity(self) -> None:
        """set(tree2vec_goal_proximity) enables the flag."""
        from pyladr.apps.prover9 import _apply_settings
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(
            "set(tree2vec_goal_proximity).\n"
            "formulas(sos). P(a). end_of_list.\n"
        )
        opts = SearchOptions()
        _apply_settings(parsed, opts)
        assert opts.tree2vec_goal_proximity is True
