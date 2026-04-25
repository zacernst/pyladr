"""RNN encoder: token sequences -> fixed-size embeddings via GRU/LSTM.

All torch imports are guarded — this module can be imported safely when
torch is not installed, but instantiating RNNEncoder requires torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Guard ML imports — everything must work without torch installed.
try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


@dataclass(frozen=True, slots=True)
class RNNEmbeddingConfig:
    """Configuration for the RNN encoder.

    Attributes:
        rnn_type: RNN variant — "gru", "lstm", or "rnn".
        input_dim: Token embedding dimensionality (input to RNN).
        hidden_dim: RNN hidden state dimensionality.
        embedding_dim: Final output embedding dimensionality.
        num_layers: Number of stacked RNN layers.
        bidirectional: Use bidirectional RNN.
        dropout: Dropout between RNN layers (only when num_layers > 1).
        composition: How to derive fixed-size vector from RNN outputs.
            "last" — final hidden state.
            "mean" — mean of all output timesteps (masking PAD).
            "attention" — learned attention over timesteps.
        normalize: L2-normalize output embeddings.
        seed: Random seed for reproducibility.
    """

    rnn_type: str = "gru"
    input_dim: int = 32
    hidden_dim: int = 64
    embedding_dim: int = 64
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.0
    composition: str = "mean"
    normalize: bool = True
    seed: int = 42


if _TORCH_AVAILABLE:

    class RNNEncoder(nn.Module):
        """Token embedding lookup + RNN + composition + projection.

        Encodes batches of padded token ID sequences into fixed-dimensional
        embedding vectors.
        """

        def __init__(self, vocab_size: int, config: RNNEmbeddingConfig) -> None:
            super().__init__()
            self.config = config
            self.vocab_size = vocab_size

            torch.manual_seed(config.seed)

            # Token embedding lookup (padding_idx=0 keeps PAD at zero)
            self.token_embedding = nn.Embedding(
                vocab_size, config.input_dim, padding_idx=0
            )

            # RNN cell
            rnn_cls = {
                "gru": nn.GRU,
                "lstm": nn.LSTM,
                "rnn": nn.RNN,
            }
            if config.rnn_type not in rnn_cls:
                raise ValueError(
                    f"Unknown rnn_type={config.rnn_type!r}, "
                    f"expected one of {list(rnn_cls)}"
                )
            self.rnn = rnn_cls[config.rnn_type](
                input_size=config.input_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                batch_first=True,
                bidirectional=config.bidirectional,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
            )

            # Effective hidden dim (doubled for bidirectional)
            effective_hidden = config.hidden_dim * (2 if config.bidirectional else 1)

            # Attention layer (if needed)
            self._attn: nn.Linear | None = None
            if config.composition == "attention":
                self._attn = nn.Linear(effective_hidden, 1)

            # Projection to final embedding dim
            self.projection = nn.Linear(effective_hidden, config.embedding_dim)

        @property
        def effective_hidden_dim(self) -> int:
            return self.config.hidden_dim * (2 if self.config.bidirectional else 1)

        def forward(
            self,
            token_ids: torch.Tensor,
            lengths: torch.Tensor,
        ) -> torch.Tensor:
            """Encode batch of padded token sequences.

            Args:
                token_ids: (batch, seq_len) int64 padded token IDs.
                lengths: (batch,) int64 actual sequence lengths.

            Returns:
                (batch, embedding_dim) float32 embeddings.
            """
            # 1. Embed tokens
            embedded = self.token_embedding(token_ids)  # (batch, seq_len, input_dim)

            # 2. Pack for efficient RNN processing
            # Clamp lengths to at least 1 to avoid pack_padded_sequence errors
            lengths_clamped = lengths.clamp(min=1).cpu()
            packed = pack_padded_sequence(
                embedded, lengths_clamped, batch_first=True, enforce_sorted=False
            )

            # 3. Run RNN
            packed_output, hidden = self.rnn(packed)

            # 4. Apply composition strategy
            if self.config.composition == "last":
                composed = self._compose_last(hidden)
            elif self.config.composition == "mean":
                output, _ = pad_packed_sequence(
                    packed_output, batch_first=True
                )  # (batch, seq_len, effective_hidden)
                composed = self._compose_mean(output, lengths)
            elif self.config.composition == "attention":
                output, _ = pad_packed_sequence(packed_output, batch_first=True)
                composed = self._compose_attention(output, lengths)
            else:
                raise ValueError(
                    f"Unknown composition={self.config.composition!r}"
                )

            # 5. Project to embedding_dim
            result = self.projection(composed)  # (batch, embedding_dim)

            # 6. Optionally L2-normalize
            if self.config.normalize:
                result = torch.nn.functional.normalize(result, p=2, dim=-1)

            return result

        def _compose_last(self, hidden: torch.Tensor | tuple) -> torch.Tensor:
            """Use final hidden state."""
            # For LSTM, hidden is (h_n, c_n) — use h_n
            if isinstance(hidden, tuple):
                h_n = hidden[0]
            else:
                h_n = hidden
            # h_n shape: (num_layers * num_directions, batch, hidden_dim)
            if self.config.bidirectional:
                # Concatenate last layer forward and backward
                fwd = h_n[-2]  # (batch, hidden_dim)
                bwd = h_n[-1]  # (batch, hidden_dim)
                return torch.cat([fwd, bwd], dim=-1)  # (batch, 2*hidden_dim)
            else:
                return h_n[-1]  # (batch, hidden_dim)

        def _compose_mean(
            self, output: torch.Tensor, lengths: torch.Tensor
        ) -> torch.Tensor:
            """Mean of all output timesteps, masking PAD positions."""
            # output: (batch, seq_len, effective_hidden)
            batch_size, seq_len, _ = output.shape
            # Create mask: (batch, seq_len)
            arange = torch.arange(seq_len, device=output.device).unsqueeze(0)
            mask = arange < lengths.unsqueeze(1)  # (batch, seq_len)
            # Mask and average
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            summed = (output * mask_expanded).sum(dim=1)  # (batch, effective_hidden)
            counts = lengths.clamp(min=1).unsqueeze(-1).float()  # (batch, 1)
            return summed / counts

        def _compose_attention(
            self, output: torch.Tensor, lengths: torch.Tensor
        ) -> torch.Tensor:
            """Learned attention over output timesteps."""
            assert self._attn is not None
            # output: (batch, seq_len, effective_hidden)
            batch_size, seq_len, _ = output.shape

            # Compute attention scores
            scores = self._attn(output).squeeze(-1)  # (batch, seq_len)

            # Mask PAD positions with -inf before softmax
            arange = torch.arange(seq_len, device=output.device).unsqueeze(0)
            mask = arange >= lengths.unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))

            # Softmax over valid positions
            weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
            # Handle all-PAD edge case (softmax of all -inf = nan)
            weights = weights.nan_to_num(0.0)

            # Weighted sum
            return (weights.unsqueeze(-1) * output).sum(dim=1)

        def encode_single(self, token_ids: list[int]) -> list[float] | None:
            """Convenience: encode one sequence without batching.

            Args:
                token_ids: List of integer token IDs.

            Returns:
                List of floats (embedding), or None if input is empty.
            """
            if not token_ids:
                return None
            was_training = self.training
            self.eval()
            with torch.no_grad():
                ids_t = torch.tensor([token_ids], dtype=torch.long)
                lens_t = torch.tensor([len(token_ids)], dtype=torch.long)
                emb = self.forward(ids_t, lens_t)  # (1, embedding_dim)
            if was_training:
                self.train()
            return emb[0].tolist()

        def expand_vocab(self, new_vocab_size: int) -> None:
            """Extend token embedding to support a larger vocabulary.

            New rows are initialized to the mean of existing embeddings.
            Used for online learning OOV extension.

            Args:
                new_vocab_size: New total vocabulary size (must be >= current).
            """
            old_size = self.vocab_size
            if new_vocab_size <= old_size:
                return

            old_weight = self.token_embedding.weight.data  # (old_size, input_dim)
            # Mean of existing non-padding embeddings for initialization
            # Skip index 0 (PAD) which is always zero
            if old_size > 1:
                mean_vec = old_weight[1:].mean(dim=0, keepdim=True)
            else:
                mean_vec = torch.zeros(1, self.config.input_dim)

            new_embedding = nn.Embedding(
                new_vocab_size, self.config.input_dim, padding_idx=0
            )
            with torch.no_grad():
                new_embedding.weight[:old_size] = old_weight
                num_new = new_vocab_size - old_size
                new_embedding.weight[old_size:] = mean_vec.expand(num_new, -1)

            self.token_embedding = new_embedding
            self.vocab_size = new_vocab_size

else:  # pragma: no cover
    # Stub when torch is not available — allows import without torch
    class RNNEncoder:  # type: ignore[no-redef]
        """Stub RNNEncoder when torch is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "RNNEncoder requires torch. Install with: pip install torch"
            )
