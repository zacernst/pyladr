"""Unit tests for RNN encoder: cells, forward pass, composition, vocab expansion.

Tests the RNNEncoder module from pyladr.ml.rnn2vec.encoder.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from pyladr.ml.rnn2vec.encoder import RNNEmbeddingConfig, RNNEncoder


# ── Config tests ──────────────────────────────────────────────────────────


class TestRNNEmbeddingConfig:
    def test_defaults(self) -> None:
        cfg = RNNEmbeddingConfig()
        assert cfg.rnn_type == "gru"
        assert cfg.input_dim == 32
        assert cfg.hidden_dim == 64
        assert cfg.embedding_dim == 64
        assert cfg.composition == "mean"
        assert cfg.normalize is True

    def test_frozen(self) -> None:
        cfg = RNNEmbeddingConfig()
        with pytest.raises(AttributeError):
            cfg.rnn_type = "lstm"  # type: ignore[misc]


# ── Forward pass tests ───────────────────────────────────────────────────


class TestRNNEncoderForwardPass:
    def _make_encoder(
        self,
        rnn_type: str = "gru",
        vocab_size: int = 20,
        composition: str = "mean",
        normalize: bool = True,
        bidirectional: bool = False,
        num_layers: int = 1,
    ) -> RNNEncoder:
        cfg = RNNEmbeddingConfig(
            rnn_type=rnn_type,
            input_dim=16,
            hidden_dim=32,
            embedding_dim=24,
            num_layers=num_layers,
            bidirectional=bidirectional,
            composition=composition,
            normalize=normalize,
            seed=42,
        )
        enc = RNNEncoder(vocab_size, cfg)
        enc.eval()
        return enc

    def test_gru_forward_single_sequence(self) -> None:
        enc = self._make_encoder(rnn_type="gru")
        ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        lens = torch.tensor([4], dtype=torch.long)
        with torch.no_grad():
            out = enc(ids, lens)
        assert out.shape == (1, 24)

    def test_lstm_forward_single_sequence(self) -> None:
        enc = self._make_encoder(rnn_type="lstm")
        ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        lens = torch.tensor([4], dtype=torch.long)
        with torch.no_grad():
            out = enc(ids, lens)
        assert out.shape == (1, 24)

    def test_rnn_forward_single_sequence(self) -> None:
        enc = self._make_encoder(rnn_type="rnn")
        ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        lens = torch.tensor([4], dtype=torch.long)
        with torch.no_grad():
            out = enc(ids, lens)
        assert out.shape == (1, 24)

    def test_batch_forward(self) -> None:
        enc = self._make_encoder()
        ids = torch.tensor([
            [2, 3, 4, 5],
            [6, 7, 0, 0],
            [2, 8, 9, 0],
            [3, 0, 0, 0],
        ], dtype=torch.long)
        lens = torch.tensor([4, 2, 3, 1], dtype=torch.long)
        with torch.no_grad():
            out = enc(ids, lens)
        assert out.shape == (4, 24)

    def test_output_embedding_dim(self) -> None:
        cfg = RNNEmbeddingConfig(
            input_dim=16, hidden_dim=32, embedding_dim=48, seed=42
        )
        enc = RNNEncoder(20, cfg)
        enc.eval()
        ids = torch.tensor([[2, 3, 4]], dtype=torch.long)
        lens = torch.tensor([3], dtype=torch.long)
        with torch.no_grad():
            out = enc(ids, lens)
        assert out.shape[1] == 48

    def test_encode_single_returns_list_float(self) -> None:
        enc = self._make_encoder()
        result = enc.encode_single([2, 3, 4, 5])
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 24
        assert all(isinstance(v, float) for v in result)

    def test_empty_sequence_graceful(self) -> None:
        enc = self._make_encoder()
        result = enc.encode_single([])
        assert result is None

    def test_normalization(self) -> None:
        enc = self._make_encoder(normalize=True)
        ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        lens = torch.tensor([4], dtype=torch.long)
        with torch.no_grad():
            out = enc(ids, lens)
        norm = torch.norm(out, dim=-1)
        assert abs(norm.item() - 1.0) < 1e-5

    def test_no_normalization(self) -> None:
        enc = self._make_encoder(normalize=False)
        ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        lens = torch.tensor([4], dtype=torch.long)
        with torch.no_grad():
            out = enc(ids, lens)
        # Not constrained to 1.0 (could be anything)
        norm = torch.norm(out, dim=-1).item()
        # Just verify it's finite
        assert not (norm != norm)  # not NaN

    def test_invalid_rnn_type_raises(self) -> None:
        cfg = RNNEmbeddingConfig(rnn_type="transformer")
        with pytest.raises(ValueError, match="Unknown rnn_type"):
            RNNEncoder(20, cfg)

    def test_bidirectional_forward(self) -> None:
        enc = self._make_encoder(bidirectional=True)
        ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        lens = torch.tensor([4], dtype=torch.long)
        with torch.no_grad():
            out = enc(ids, lens)
        # Output should still be embedding_dim (projection handles it)
        assert out.shape == (1, 24)

    def test_multi_layer(self) -> None:
        enc = self._make_encoder(num_layers=2)
        ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        lens = torch.tensor([4], dtype=torch.long)
        with torch.no_grad():
            out = enc(ids, lens)
        assert out.shape == (1, 24)


# ── Composition tests ────────────────────────────────────────────────────


class TestRNNEncoderComposition:
    def _encode(self, composition: str) -> torch.Tensor:
        cfg = RNNEmbeddingConfig(
            input_dim=16, hidden_dim=32, embedding_dim=24,
            composition=composition, seed=42,
        )
        enc = RNNEncoder(20, cfg)
        enc.eval()
        ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        lens = torch.tensor([4], dtype=torch.long)
        with torch.no_grad():
            return enc(ids, lens)

    def test_last_composition(self) -> None:
        out = self._encode("last")
        assert out.shape == (1, 24)

    def test_mean_composition(self) -> None:
        out = self._encode("mean")
        assert out.shape == (1, 24)

    def test_attention_composition(self) -> None:
        out = self._encode("attention")
        assert out.shape == (1, 24)

    def test_all_compositions_same_dim(self) -> None:
        for comp in ("last", "mean", "attention"):
            out = self._encode(comp)
            assert out.shape[1] == 24, f"composition={comp} produced wrong dim"


# ── Vocab expansion tests ────────────────────────────────────────────────


class TestRNNEncoderVocabExpansion:
    def test_expand_vocab(self) -> None:
        cfg = RNNEmbeddingConfig(input_dim=16, hidden_dim=32, embedding_dim=24, seed=42)
        enc = RNNEncoder(20, cfg)
        assert enc.vocab_size == 20
        enc.expand_vocab(30)
        assert enc.vocab_size == 30

    def test_expanded_vocab_valid_forward(self) -> None:
        cfg = RNNEmbeddingConfig(input_dim=16, hidden_dim=32, embedding_dim=24, seed=42)
        enc = RNNEncoder(20, cfg)
        enc.expand_vocab(30)
        enc.eval()
        # Use a token ID from the expanded range
        ids = torch.tensor([[2, 25, 4]], dtype=torch.long)
        lens = torch.tensor([3], dtype=torch.long)
        with torch.no_grad():
            out = enc(ids, lens)
        assert out.shape == (1, 24)

    def test_expand_noop_when_smaller(self) -> None:
        cfg = RNNEmbeddingConfig(input_dim=16, hidden_dim=32, embedding_dim=24, seed=42)
        enc = RNNEncoder(20, cfg)
        enc.expand_vocab(10)  # smaller — should be noop
        assert enc.vocab_size == 20

    def test_expand_preserves_existing_embeddings(self) -> None:
        cfg = RNNEmbeddingConfig(input_dim=16, hidden_dim=32, embedding_dim=24, seed=42)
        enc = RNNEncoder(20, cfg)
        old_weight = enc.token_embedding.weight.data[:20].clone()
        enc.expand_vocab(30)
        new_weight = enc.token_embedding.weight.data[:20]
        assert torch.allclose(old_weight, new_weight)


# ── Determinism tests ────────────────────────────────────────────────────


class TestRNNEncoderDeterminism:
    def test_deterministic_with_seed(self) -> None:
        cfg = RNNEmbeddingConfig(input_dim=16, hidden_dim=32, embedding_dim=24, seed=42)
        enc1 = RNNEncoder(20, cfg)
        enc2 = RNNEncoder(20, cfg)
        enc1.eval()
        enc2.eval()

        ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        lens = torch.tensor([4], dtype=torch.long)
        with torch.no_grad():
            out1 = enc1(ids, lens)
            out2 = enc2(ids, lens)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_dropout_no_effect_eval_mode(self) -> None:
        cfg = RNNEmbeddingConfig(
            input_dim=16, hidden_dim=32, embedding_dim=24,
            num_layers=2, dropout=0.5, seed=42,
        )
        enc = RNNEncoder(20, cfg)
        enc.eval()
        ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        lens = torch.tensor([4], dtype=torch.long)
        with torch.no_grad():
            out1 = enc(ids, lens)
            out2 = enc(ids, lens)
        assert torch.allclose(out1, out2, atol=1e-6)
