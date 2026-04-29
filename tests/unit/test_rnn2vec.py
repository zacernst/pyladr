"""Unit tests for RNN2Vec core algorithm.

Tests config, training, embedding, serialization, and similarity
using simple term trees representative of the vampire.in domain.
Mirrors tests/unit/test_tree2vec.py structure.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.ml.rnn2vec.algorithm import RNN2Vec, RNN2VecConfig
from pyladr.ml.rnn2vec.encoder import RNNEmbeddingConfig
from pyladr.ml.rnn2vec.walks import WalkConfig, WalkType


# ── Helpers: vampire.in domain terms ──────────────────────────────────────

SYM_P = 1   # P: unary predicate
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


def make_literal(sign: bool, atom: Term) -> Literal:
    return Literal(sign=sign, atom=atom)


def make_clause(*lits: Literal, clause_id: int = 0) -> Clause:
    return Clause(literals=lits, id=clause_id)


def _make_vampire_clauses() -> list[Clause]:
    """Small set of clauses mimicking vampire.in patterns."""
    x, y, z = var(0), var(1), var(2)
    return [
        make_clause(make_literal(True, P(i(x, n(x)))), clause_id=1),
        make_clause(make_literal(True, P(i(n(x), x))), clause_id=2),
        make_clause(
            make_literal(False, P(i(x, y))),
            make_literal(False, P(i(y, z))),
            make_literal(True, P(i(x, z))),
            clause_id=3,
        ),
        make_clause(
            make_literal(True, P(i(i(x, y), i(n(y), n(x))))),
            clause_id=4,
        ),
    ]


def _make_trained_rnn2vec(
    embedding_dim: int = 32,
    rnn_type: str = "gru",
    composition: str = "mean",
    normalize: bool = True,
) -> RNN2Vec:
    """Create and train a small RNN2Vec model."""
    config = RNN2VecConfig(
        walk_config=WalkConfig(
            walk_types=(WalkType.DEPTH_FIRST, WalkType.PATH),
            seed=42,
        ),
        rnn_config=RNNEmbeddingConfig(
            rnn_type=rnn_type,
            input_dim=16,
            hidden_dim=32,
            embedding_dim=embedding_dim,
            composition=composition,
            normalize=normalize,
            seed=42,
        ),
        training_epochs=3,
        seed=42,
    )
    model = RNN2Vec(config)
    model.train(_make_vampire_clauses())
    return model


# ── Config tests ──────────────────────────────────────────────────────────


class TestRNN2VecConfig:
    def test_default_config_values(self) -> None:
        cfg = RNN2VecConfig()
        assert cfg.training_epochs == 5
        assert cfg.learning_rate == 0.001
        assert cfg.batch_size == 32
        assert cfg.seed == 42

        rc = cfg.rnn_config
        assert rc.rnn_type == "gru"
        assert rc.hidden_dim == 64
        assert rc.embedding_dim == 64

    def test_frozen_config(self) -> None:
        cfg = RNN2VecConfig()
        with pytest.raises(AttributeError):
            cfg.seed = 99  # type: ignore[misc]

    def test_config_with_lstm(self) -> None:
        cfg = RNN2VecConfig(
            rnn_config=RNNEmbeddingConfig(rnn_type="lstm")
        )
        assert cfg.rnn_config.rnn_type == "lstm"

    def test_config_with_rnn(self) -> None:
        cfg = RNN2VecConfig(
            rnn_config=RNNEmbeddingConfig(rnn_type="rnn")
        )
        assert cfg.rnn_config.rnn_type == "rnn"


# ── Core RNN2Vec tests ────────────────────────────────────────────────────


class TestRNN2Vec:
    def test_train_and_embed(self) -> None:
        model = _make_trained_rnn2vec()
        assert model.trained
        assert model.vocab_size > 0

    def test_embed_term_dim(self) -> None:
        model = _make_trained_rnn2vec(embedding_dim=16)
        emb = model.embed_term(i(var(0), n(var(1))))
        assert emb is not None
        assert len(emb) == 16

    def test_embed_clause_dim(self) -> None:
        model = _make_trained_rnn2vec(embedding_dim=16)
        clause = make_clause(make_literal(True, P(i(var(0), var(1)))))
        emb = model.embed_clause(clause)
        assert emb is not None
        assert len(emb) == 16

    def test_embed_clauses_batch_length(self) -> None:
        model = _make_trained_rnn2vec()
        clauses = _make_vampire_clauses()
        results = model.embed_clauses_batch(clauses)
        assert len(results) == len(clauses)
        assert all(r is not None for r in results)

    def test_untrained_returns_none(self) -> None:
        model = RNN2Vec()
        assert model.embed_term(var(0)) is None
        assert model.embed_clause(make_clause()) is None

    def test_normalized_unit_norm(self) -> None:
        model = _make_trained_rnn2vec(normalize=True)
        emb = model.embed_term(i(var(0), n(var(1))))
        assert emb is not None
        norm = math.sqrt(sum(v * v for v in emb))
        assert abs(norm - 1.0) < 0.05

    def test_deterministic_with_seed(self) -> None:
        model1 = _make_trained_rnn2vec()
        model2 = _make_trained_rnn2vec()
        term = i(var(0), n(var(1)))
        emb1 = model1.embed_term(term)
        emb2 = model2.embed_term(term)
        assert emb1 is not None and emb2 is not None
        for a, b in zip(emb1, emb2):
            assert abs(a - b) < 1e-5

    def test_train_from_terms(self) -> None:
        config = RNN2VecConfig(
            walk_config=WalkConfig(walk_types=(WalkType.DEPTH_FIRST,), seed=42),
            rnn_config=RNNEmbeddingConfig(
                input_dim=16, hidden_dim=32, embedding_dim=16, seed=42
            ),
            training_epochs=2,
            seed=42,
        )
        model = RNN2Vec(config)
        terms = [i(var(0), n(var(1))), n(i(var(0), var(1)))]
        stats = model.train_from_terms(terms)
        assert stats["vocab_size"] > 0
        assert model.trained

    def test_similarity_range(self) -> None:
        model = _make_trained_rnn2vec()
        x, y = var(0), var(1)
        sim = model.similarity(i(x, n(x)), i(y, n(y)))
        assert sim is not None
        assert -1.0 <= sim <= 1.0

    def test_similarity_alpha_equivalent_high(self) -> None:
        """Alpha-equivalent terms (same structure, different vars) -> high similarity."""
        model = _make_trained_rnn2vec()
        x, y = var(0), var(1)
        sim = model.similarity(i(x, n(x)), i(y, n(y)))
        assert sim is not None
        assert sim > 0.99

    def test_get_token_embedding(self) -> None:
        model = _make_trained_rnn2vec()
        # VAR is a common token in the vocabulary
        emb = model.get_token_embedding("VAR")
        assert emb is not None
        assert isinstance(emb, list)
        assert all(isinstance(v, float) for v in emb)

    def test_get_token_embedding_unknown(self) -> None:
        model = _make_trained_rnn2vec()
        assert model.get_token_embedding("NONEXISTENT_TOKEN_XYZ") is None

    def test_embedding_dim_property(self) -> None:
        model = _make_trained_rnn2vec(embedding_dim=48)
        assert model.embedding_dim == 48

    def test_untrained_batch_returns_nones(self) -> None:
        model = RNN2Vec()
        results = model.embed_clauses_batch(_make_vampire_clauses())
        assert all(r is None for r in results)


# ── Serialization tests ──────────────────────────────────────────────────


class TestRNN2VecSerialization:
    def test_save_raises_untrained(self, tmp_path) -> None:
        model = RNN2Vec()
        with pytest.raises(RuntimeError, match="untrained"):
            model.save(tmp_path / "model")

    def test_save_load_round_trip(self, tmp_path) -> None:
        model = _make_trained_rnn2vec(embedding_dim=16)
        save_dir = tmp_path / "rnn2vec_model"
        model.save(save_dir)

        loaded = RNN2Vec.load(save_dir)
        assert loaded.trained
        assert loaded.embedding_dim == 16

        # Embeddings should match
        term = i(var(0), n(var(1)))
        emb_orig = model.embed_term(term)
        emb_loaded = loaded.embed_term(term)
        assert emb_orig is not None and emb_loaded is not None
        for a, b in zip(emb_orig, emb_loaded):
            assert abs(a - b) < 1e-5

    def test_load_wrong_version_raises(self, tmp_path) -> None:
        model = _make_trained_rnn2vec()
        save_dir = tmp_path / "rnn2vec_model"
        model.save(save_dir)

        # Corrupt the format_version
        import json
        config_path = save_dir / "config.json"
        data = json.loads(config_path.read_text())
        data["format_version"] = 999
        config_path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="format version"):
            RNN2Vec.load(save_dir)

    def test_save_load_preserves_config(self, tmp_path) -> None:
        model = _make_trained_rnn2vec(
            embedding_dim=24, rnn_type="gru", composition="mean"
        )
        save_dir = tmp_path / "rnn2vec_model"
        model.save(save_dir)

        loaded = RNN2Vec.load(save_dir)
        assert loaded.config.rnn_config.rnn_type == "gru"
        assert loaded.config.rnn_config.hidden_dim == 32
        assert loaded.config.rnn_config.embedding_dim == 24
        assert loaded.config.rnn_config.composition == "mean"

    def test_save_load_preserves_vocab(self, tmp_path) -> None:
        model = _make_trained_rnn2vec()
        save_dir = tmp_path / "rnn2vec_model"
        model.save(save_dir)

        loaded = RNN2Vec.load(save_dir)
        assert loaded.vocab_size == model.vocab_size


# ── Online update tests ──────────────────────────────────────────────────


class TestRNN2VecOnlineUpdate:
    def test_untrained_update_is_noop(self) -> None:
        model = RNN2Vec()
        result = model.update_online(_make_vampire_clauses())
        assert result["pairs_trained"] == 0

    def test_online_update_runs(self) -> None:
        model = _make_trained_rnn2vec()
        clauses = _make_vampire_clauses()
        result = model.update_online(clauses)
        assert result["pairs_trained"] >= 0
        assert "loss" in result

    def test_empty_clauses_update_noop(self) -> None:
        model = _make_trained_rnn2vec()
        result = model.update_online([])
        assert result["pairs_trained"] == 0


# ── Train/val/test split tests (data leakage prevention) ─────────────────


def _many_vampire_clauses(count: int = 20) -> list[Clause]:
    """Generate `count` distinct clauses for split-ratio testing."""
    clauses: list[Clause] = []
    for k in range(count):
        x = var(k % 3)
        y = var((k + 1) % 3)
        atom = P(i(x, n(y)))
        clauses.append(make_clause(make_literal(True, atom), clause_id=k + 1))
    return clauses


class TestRNN2VecConfigSplit:
    def test_default_ratios_sum_to_one(self) -> None:
        cfg = RNN2VecConfig()
        assert cfg.train_ratio == 0.70
        assert cfg.val_ratio == 0.15
        assert cfg.test_ratio == 0.15

    def test_ratios_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            RNN2VecConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_negative_ratio_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            RNN2VecConfig(train_ratio=1.1, val_ratio=-0.05, test_ratio=-0.05)

    def test_zero_train_ratio_rejected(self) -> None:
        with pytest.raises(ValueError, match="train_ratio must be > 0"):
            RNN2VecConfig(train_ratio=0.0, val_ratio=0.5, test_ratio=0.5)

    def test_custom_ratios_accepted(self) -> None:
        cfg = RNN2VecConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        assert cfg.train_ratio == 0.8


class TestRNN2VecSplit:
    def _make_split_config(self, **overrides) -> RNN2VecConfig:
        defaults = {
            "walk_config": WalkConfig(
                walk_types=(WalkType.DEPTH_FIRST, WalkType.PATH), seed=42,
            ),
            "rnn_config": RNNEmbeddingConfig(
                input_dim=16, hidden_dim=32, embedding_dim=16, seed=42,
            ),
            "training_epochs": 1,
            "seed": 42,
        }
        defaults.update(overrides)
        return RNN2VecConfig(**defaults)

    def test_stats_include_split_metrics(self) -> None:
        """Training stats must report val_loss, test_loss, and split sizes."""
        model = RNN2Vec(self._make_split_config())
        stats = model.train(_many_vampire_clauses(20))
        assert "val_loss" in stats
        assert "test_loss" in stats
        assert "train_clauses" in stats
        assert "val_clauses" in stats
        assert "test_clauses" in stats

    def test_default_split_sizes(self) -> None:
        """20 clauses with 70/15/15 → 14/3/3."""
        model = RNN2Vec(self._make_split_config())
        stats = model.train(_many_vampire_clauses(20))
        assert stats["train_clauses"] == 14
        assert stats["val_clauses"] == 3
        assert stats["test_clauses"] == 3

    def test_split_sizes_sum_to_total(self) -> None:
        """For any N, split sizes must partition the input exactly."""
        for n_clauses in (1, 2, 3, 4, 5, 7, 10, 25, 50):
            model = RNN2Vec(self._make_split_config())
            stats = model.train(_many_vampire_clauses(n_clauses))
            total = (
                stats["train_clauses"]
                + stats["val_clauses"]
                + stats["test_clauses"]
            )
            assert total == n_clauses, f"n={n_clauses}: {stats}"
            # Training must always receive at least one clause when n >= 1.
            assert stats["train_clauses"] >= 1

    def test_custom_ratios_80_10_10(self) -> None:
        """Ratios are respected: 20 clauses at 80/10/10 → 16/2/2."""
        cfg = self._make_split_config(
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        )
        model = RNN2Vec(cfg)
        stats = model.train(_many_vampire_clauses(20))
        assert stats["train_clauses"] == 16
        assert stats["val_clauses"] == 2
        assert stats["test_clauses"] == 2

    def test_split_is_deterministic_with_seed(self) -> None:
        """Same seed → same split assignment (and same val/test losses)."""
        clauses = _many_vampire_clauses(20)
        m1 = RNN2Vec(self._make_split_config(seed=7))
        m2 = RNN2Vec(self._make_split_config(seed=7))
        s1 = m1.train(clauses)
        s2 = m2.train(clauses)
        assert s1["train_clauses"] == s2["train_clauses"]
        assert s1["val_clauses"] == s2["val_clauses"]
        assert s1["test_clauses"] == s2["test_clauses"]
        assert abs(s1["val_loss"] - s2["val_loss"]) < 1e-6
        assert abs(s1["test_loss"] - s2["test_loss"]) < 1e-6

    def test_val_loss_uses_no_gradient_updates(self) -> None:
        """Parameters must be identical before and after held-out evaluation.

        Captures the encoder state right after training, re-invokes the
        evaluator on val+test data, and asserts weights did not change.
        """
        model = RNN2Vec(self._make_split_config())
        model.train(_many_vampire_clauses(20))

        snapshot = {
            name: p.detach().clone()
            for name, p in model._encoder.named_parameters()
        }
        val_walks = [
            model._walker.walks_from_clause(c)
            for c in _many_vampire_clauses(6)
        ]
        import random as _r
        model._evaluate_loss(val_walks, _r.Random(0))

        for name, p in model._encoder.named_parameters():
            assert torch.equal(snapshot[name], p), (
                f"evaluation mutated parameter {name}"
            )

    def test_vocab_built_from_training_only(self) -> None:
        """Vocabulary comes strictly from training walks (no val/test leakage).

        Constructs a corpus where one clause contains a uniquely identifiable
        token and verifies: if that clause is held out, its token is absent
        from the trained vocabulary.
        """
        # Tokens are structural, not symbol-number-based, so use a distinct
        # arity to produce a unique walk token.
        unique_sym = 99
        unique_atom = get_rigid_term(unique_sym, 3, (var(0), var(1), var(2)))
        unique_clause = make_clause(
            make_literal(True, unique_atom), clause_id=9999,
        )
        # Put the unique clause at index 0; shuffle w/ seed=42 may place it
        # in train or val/test. We assert the property by checking both cases.
        base_clauses = _many_vampire_clauses(19)
        all_clauses = [unique_clause] + base_clauses

        model = RNN2Vec(self._make_split_config(seed=42))
        model.train(all_clauses)

        # Reconstruct what the split produced to learn where unique_clause went.
        import random as _r
        rng_check = _r.Random(42)
        train_idx, val_idx, test_idx = model._split_indices(
            len(all_clauses), rng_check,
        )
        # unique_clause is at index 0 in the original list.
        unique_in_train = 0 in train_idx

        walk_tokens = set()
        for walk in model._walker.walks_from_clause(unique_clause):
            walk_tokens.update(walk)
        # Tokens that appear ONLY in the unique clause structure.
        # The unique arity-3 term produces a walk token referencing that arity.
        distinctive = [t for t in walk_tokens if "3" in t or "arity" in t.lower()]

        vocab_tokens = set(model._vocab._token_to_id.keys())
        if not unique_in_train and distinctive:
            # At least one distinctive token must be missing from vocab.
            assert any(t not in vocab_tokens for t in distinctive), (
                "held-out clause's distinctive tokens leaked into vocabulary"
            )

    def test_single_clause_all_to_training(self) -> None:
        """Edge case: 1 clause → all goes to training, val/test empty."""
        model = RNN2Vec(self._make_split_config())
        stats = model.train(_many_vampire_clauses(1))
        assert stats["train_clauses"] == 1
        assert stats["val_clauses"] == 0
        assert stats["test_clauses"] == 0
        assert stats["val_loss"] == 0.0
        assert stats["test_loss"] == 0.0

    def test_empty_clauses_returns_empty_split(self) -> None:
        """Edge case: 0 clauses → all three splits empty, no crash."""
        model = RNN2Vec(self._make_split_config())
        stats = model.train([])
        assert stats["train_clauses"] == 0
        assert stats["val_clauses"] == 0
        assert stats["test_clauses"] == 0

    def test_train_from_terms_also_splits(self) -> None:
        """train_from_terms applies the same split to term-level inputs."""
        model = RNN2Vec(self._make_split_config())
        terms = [i(var(k % 3), n(var((k + 1) % 3))) for k in range(20)]
        stats = model.train_from_terms(terms)
        assert "val_loss" in stats
        assert "test_loss" in stats
        assert stats["train_clauses"] == 14
        assert stats["val_clauses"] == 3
        assert stats["test_clauses"] == 3

    def test_save_load_preserves_split_ratios(self, tmp_path) -> None:
        """Split ratios round-trip through save/load."""
        cfg = self._make_split_config(
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        )
        model = RNN2Vec(cfg)
        model.train(_many_vampire_clauses(10))
        save_dir = tmp_path / "rnn2vec_split"
        model.save(save_dir)

        loaded = RNN2Vec.load(save_dir)
        assert loaded.config.train_ratio == 0.6
        assert loaded.config.val_ratio == 0.2
        assert loaded.config.test_ratio == 0.2

    def test_load_legacy_save_uses_default_ratios(self, tmp_path) -> None:
        """Loading a pre-split save uses default ratios (backward-compat)."""
        model = _make_trained_rnn2vec(embedding_dim=16)
        save_dir = tmp_path / "legacy"
        model.save(save_dir)

        import json
        cfg_path = save_dir / "config.json"
        cfg_data = json.loads(cfg_path.read_text())
        cfg_data.pop("train_ratio", None)
        cfg_data.pop("val_ratio", None)
        cfg_data.pop("test_ratio", None)
        cfg_path.write_text(json.dumps(cfg_data))

        loaded = RNN2Vec.load(save_dir)
        assert loaded.config.train_ratio == 0.70
        assert loaded.config.val_ratio == 0.15
        assert loaded.config.test_ratio == 0.15
