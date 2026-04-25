"""Tests for RNN2VecEmbeddingProvider: caching, factories, thread safety."""

from __future__ import annotations

import concurrent.futures

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.ml.rnn2vec.algorithm import RNN2Vec, RNN2VecConfig
from pyladr.ml.rnn2vec.encoder import RNNEmbeddingConfig
from pyladr.ml.rnn2vec.provider import RNN2VecEmbeddingProvider, RNN2VecProviderConfig
from pyladr.ml.tree2vec.walks import WalkConfig, WalkType


# ── Helpers ──────────────────────────────────────────────────────────────

SYM_P = 1
SYM_I = 2
SYM_N = 3


def _var(n: int):
    return get_variable_term(n)


def _rigid(sym: int, arity: int, args=()):
    return get_rigid_term(sym, arity, args)


def _pos_lit(atom) -> Literal:
    return Literal(sign=True, atom=atom)


def _simple_clause(sym=SYM_P) -> Clause:
    return Clause(literals=(_pos_lit(_rigid(sym, 1, (_var(0),))),))


def _complex_clause() -> Clause:
    x, y = _var(0), _var(1)
    return Clause(literals=(
        _pos_lit(_rigid(SYM_P, 1, (_rigid(SYM_I, 2, (x, _rigid(SYM_N, 1, (y,)))),))),
    ))


def _make_trained_rnn2vec() -> RNN2Vec:
    config = RNN2VecConfig(
        walk_config=WalkConfig(walk_types=(WalkType.DEPTH_FIRST, WalkType.PATH), seed=42),
        rnn_config=RNNEmbeddingConfig(
            input_dim=16, hidden_dim=32, embedding_dim=24, seed=42,
        ),
        training_epochs=3,
        seed=42,
    )
    model = RNN2Vec(config)
    clauses = [_simple_clause(), _complex_clause()]
    model.train(clauses)
    return model


def _make_provider(
    cache_max_entries: int = 1000,
    enable_cache: bool = True,
) -> RNN2VecEmbeddingProvider:
    model = _make_trained_rnn2vec()
    config = RNN2VecProviderConfig(
        cache_max_entries=cache_max_entries,
        enable_cache=enable_cache,
    )
    return RNN2VecEmbeddingProvider(model, config=config)


# ── Basic tests ──────────────────────────────────────────────────────────


class TestRNN2VecEmbeddingProviderBasic:
    def test_embedding_dim_property(self) -> None:
        provider = _make_provider()
        assert provider.embedding_dim == 24

    def test_get_embedding_returns_list_or_none(self) -> None:
        provider = _make_provider()
        result = provider.get_embedding(_simple_clause())
        assert result is None or isinstance(result, list)

    def test_get_embedding_correct_length(self) -> None:
        provider = _make_provider()
        result = provider.get_embedding(_simple_clause())
        assert result is not None
        assert len(result) == 24

    def test_get_embeddings_batch_returns_list(self) -> None:
        provider = _make_provider()
        results = provider.get_embeddings_batch([_simple_clause(), _complex_clause()])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_get_embeddings_batch_empty_input(self) -> None:
        provider = _make_provider()
        assert provider.get_embeddings_batch([]) == []

    def test_untrained_returns_none(self) -> None:
        model = RNN2Vec()
        provider = RNN2VecEmbeddingProvider(model)
        assert provider.get_embedding(_simple_clause()) is None

    def test_untrained_batch_returns_all_none(self) -> None:
        model = RNN2Vec()
        provider = RNN2VecEmbeddingProvider(model)
        results = provider.get_embeddings_batch([_simple_clause(), _complex_clause()])
        assert results == [None, None]


# ── Cache tests ──────────────────────────────────────────────────────────


class TestRNN2VecProviderCache:
    def test_cache_populated_after_get(self) -> None:
        provider = _make_provider()
        assert provider.cache_size == 0
        provider.get_embedding(_simple_clause())
        assert provider.cache_size == 1

    def test_cache_hit_on_second_call(self) -> None:
        provider = _make_provider()
        provider.get_embedding(_simple_clause())
        snap1 = provider.stats.snapshot()
        assert snap1["misses"] == 1

        provider.get_embedding(_simple_clause())
        snap2 = provider.stats.snapshot()
        assert snap2["hits"] == 1

    def test_alpha_equivalent_clauses_share_cache(self) -> None:
        provider = _make_provider()
        c1 = Clause(literals=(_pos_lit(_rigid(SYM_P, 1, (_var(0),))),))
        c2 = Clause(literals=(_pos_lit(_rigid(SYM_P, 1, (_var(1),))),))

        provider.get_embedding(c1)
        assert provider.cache_size == 1

        provider.get_embedding(c2)
        # Alpha-equivalent should share the cache entry
        assert provider.cache_size == 1

    def test_cache_eviction_at_capacity(self) -> None:
        provider = _make_provider(cache_max_entries=2)
        c1 = _simple_clause(sym=SYM_P)
        c2 = _complex_clause()
        # Third clause with different structure
        c3 = Clause(literals=(
            _pos_lit(_rigid(SYM_P, 1, (_rigid(SYM_N, 1, (_var(0),)),))),
        ))

        provider.get_embedding(c1)
        provider.get_embedding(c2)
        assert provider.cache_size == 2

        provider.get_embedding(c3)
        assert provider.cache_size <= 2

        snap = provider.stats.snapshot()
        assert snap["evictions"] >= 1

    def test_cache_disabled(self) -> None:
        provider = _make_provider(enable_cache=False)
        result = provider.get_embedding(_simple_clause())
        assert result is not None
        assert provider.cache_size == 0

    def test_cache_stats_snapshot(self) -> None:
        provider = _make_provider()
        provider.get_embedding(_simple_clause())
        snap = provider.stats.snapshot()
        assert "hits" in snap
        assert "misses" in snap
        assert "evictions" in snap
        assert "hit_rate" in snap


# ── Versioning tests ─────────────────────────────────────────────────────


class TestRNN2VecProviderVersioning:
    def test_version_starts_at_zero(self) -> None:
        provider = _make_provider()
        assert provider.model_version == 0

    def test_bump_increments_version(self) -> None:
        provider = _make_provider()
        v1 = provider.bump_model_version()
        assert v1 == 1
        v2 = provider.bump_model_version()
        assert v2 == 2

    def test_stale_cache_after_bump(self) -> None:
        provider = _make_provider()
        provider.get_embedding(_simple_clause())
        assert provider.cache_size == 1
        snap1 = provider.stats.snapshot()

        # Bump version — existing cache entries become stale
        provider.bump_model_version()

        # Next lookup should miss (stale version)
        provider.get_embedding(_simple_clause())
        snap2 = provider.stats.snapshot()
        assert snap2["misses"] == snap1["misses"] + 1

    def test_invalidate_all_clears_cache(self) -> None:
        provider = _make_provider()
        provider.get_embedding(_simple_clause())
        provider.get_embedding(_complex_clause())
        assert provider.cache_size == 2

        evicted = provider.invalidate_all()
        assert evicted == 2
        assert provider.cache_size == 0


# ── Factory tests ────────────────────────────────────────────────────────


class TestRNN2VecProviderFactory:
    def test_from_vampire_file(self) -> None:
        import os
        filepath = os.path.join(
            os.path.dirname(__file__), "..", "fixtures", "inputs", "vampire.in"
        )
        if not os.path.exists(filepath):
            pytest.skip("vampire.in fixture not found")

        config = RNN2VecProviderConfig(
            rnn2vec_config=RNN2VecConfig(
                rnn_config=RNNEmbeddingConfig(
                    input_dim=16, hidden_dim=32, embedding_dim=24, seed=42,
                ),
                training_epochs=2,
                seed=42,
            ),
        )
        provider = RNN2VecEmbeddingProvider.from_vampire_file(filepath, config=config)
        assert provider.rnn2vec.trained
        assert provider.embedding_dim == 24

    def test_from_saved_model(self, tmp_path) -> None:
        model = _make_trained_rnn2vec()
        save_dir = tmp_path / "rnn2vec_model"
        model.save(save_dir)

        provider = RNN2VecEmbeddingProvider.from_saved_model(str(save_dir))
        assert provider.rnn2vec.trained
        assert provider.embedding_dim == 24

    def test_from_corpus(self) -> None:
        import os
        filepath = os.path.join(
            os.path.dirname(__file__), "..", "fixtures", "inputs", "vampire.in"
        )
        if not os.path.exists(filepath):
            pytest.skip("vampire.in fixture not found")

        from pyladr.ml.tree2vec.vampire_parser import parse_vampire_file
        corpus = parse_vampire_file(filepath)

        config = RNN2VecProviderConfig(
            rnn2vec_config=RNN2VecConfig(
                rnn_config=RNNEmbeddingConfig(
                    input_dim=16, hidden_dim=32, embedding_dim=24, seed=42,
                ),
                training_epochs=2,
                seed=42,
            ),
        )
        provider = RNN2VecEmbeddingProvider.from_corpus(corpus, config=config)
        assert provider.rnn2vec.trained


# ── Thread safety tests ──────────────────────────────────────────────────


class TestRNN2VecProviderThreadSafety:
    def test_concurrent_reads(self) -> None:
        provider = _make_provider()
        clauses = [_simple_clause(), _complex_clause()]
        errors: list[Exception] = []

        def read_embeddings():
            try:
                for _ in range(20):
                    results = provider.get_embeddings_batch(clauses)
                    assert len(results) == 2
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(read_embeddings) for _ in range(4)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        assert errors == [], f"Concurrent errors: {errors}"
