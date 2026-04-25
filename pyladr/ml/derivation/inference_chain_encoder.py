"""Learned sequence encoder for inference chains.

Encodes variable-length inference rule sequences (JustType chains) into
fixed-size embeddings using:
  1. Learned JustType embeddings (one embedding per inference rule)
  2. Sinusoidal positional encoding (captures ordering and recency)
  3. Weighted mean pooling with exponential recency bias
     (recent steps matter more than ancient ancestors)

The output embedding can be concatenated with CLAUSE node features or
used as an auxiliary input to the hierarchical GNN.

Design:
- No RNN/transformer overhead: the encoder is a lightweight lookup +
  positional encoding + pooling pipeline suitable for online use during
  search.
- Differentiable: JustType embeddings are learned parameters that can
  be trained end-to-end via the contrastive learning pipeline.
- Graceful degradation: empty chains produce a learned "no-history"
  embedding rather than zeros.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class InferenceChainConfig:
    """Configuration for the inference chain encoder.

    Attributes:
        num_just_types: Number of distinct JustType values (including
            a padding/unknown token at index 0).
        chain_embed_dim: Dimension of the per-step embeddings.
        output_dim: Dimension of the final chain embedding.
        max_chain_length: Maximum sequence length to encode.
        recency_temperature: Controls the strength of recency bias in
            the weighted pooling.  Higher = more uniform weighting.
        dropout: Dropout on the output projection.
    """

    num_just_types: int = 24  # 22 JustType values + pad + unknown
    chain_embed_dim: int = 64
    output_dim: int = 128
    max_chain_length: int = 64
    recency_temperature: float = 10.0
    dropout: float = 0.1


_DEFAULT_CONFIG = InferenceChainConfig()


class InferenceChainEncoder(nn.Module):
    """Encodes inference rule chains into fixed-size embeddings.

    Usage::

        encoder = InferenceChainEncoder()

        # chain is a tuple[int, ...] of JustType values (root → clause)
        chain_tensor = torch.tensor([chain], dtype=torch.long)  # (1, L)
        embedding = encoder(chain_tensor)  # (1, output_dim)

    For batch encoding::

        # chains: list of variable-length tuples
        padded, lengths = encoder.pad_chains(chains)  # (B, max_len), (B,)
        embeddings = encoder(padded, lengths)  # (B, output_dim)
    """

    def __init__(self, config: InferenceChainConfig | None = None) -> None:
        super().__init__()
        self.config = config or _DEFAULT_CONFIG
        c = self.config

        # Learned embeddings for each JustType value
        # Index 0 = padding, index 1 = unknown
        self.rule_embedding = nn.Embedding(
            c.num_just_types, c.chain_embed_dim, padding_idx=0
        )

        # Sinusoidal positional encoding (pre-computed, not learned)
        pe = _sinusoidal_positional_encoding(c.max_chain_length, c.chain_embed_dim)
        self.register_buffer("positional_encoding", pe)

        # Learned "no history" embedding for empty chains
        self.no_history_embedding = nn.Parameter(
            torch.randn(c.output_dim) * 0.02
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(c.chain_embed_dim, c.output_dim),
            nn.ReLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.output_dim, c.output_dim),
        )

        # Layer norm on output
        self.output_norm = nn.LayerNorm(c.output_dim)

    def forward(
        self,
        chains: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode inference chains to fixed-size embeddings.

        Args:
            chains: (B, L) integer tensor of JustType values.
                Padded with 0 for shorter sequences.
            lengths: (B,) actual lengths.  If None, inferred from
                non-zero entries.

        Returns:
            (B, output_dim) chain embeddings.
        """
        B, L = chains.shape

        if lengths is None:
            lengths = (chains != 0).sum(dim=1)  # (B,)

        # Handle all-empty batch
        if lengths.max().item() == 0:
            return self.no_history_embedding.unsqueeze(0).expand(B, -1)

        # Clamp rule indices to valid range
        clamped = chains.clamp(0, self.config.num_just_types - 1)

        # Look up rule embeddings: (B, L, chain_embed_dim)
        rule_emb = self.rule_embedding(clamped)

        # Add positional encoding (truncate if L < max_chain_length)
        pos_enc = self.positional_encoding[:L, :]  # (L, chain_embed_dim)
        rule_emb = rule_emb + pos_enc.unsqueeze(0)

        # Compute recency-weighted mean pooling
        # Weight increases exponentially toward the end of the chain
        # (more recent steps = closer to the clause = higher weight)
        weights = self._recency_weights(L, lengths)  # (B, L)

        # Zero out padding positions
        padding_mask = torch.arange(L, device=chains.device).unsqueeze(0) < lengths.unsqueeze(1)
        weights = weights * padding_mask.float()

        # Normalise weights
        weight_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        weights = weights / weight_sum  # (B, L)

        # Weighted mean: (B, chain_embed_dim)
        pooled = (rule_emb * weights.unsqueeze(2)).sum(dim=1)

        # Project to output dimension
        output = self.output_projection(pooled)
        output = self.output_norm(output)

        # Replace empty-chain rows with the learned no-history embedding
        empty_mask = (lengths == 0).unsqueeze(1)  # (B, 1)
        output = torch.where(
            empty_mask, self.no_history_embedding.unsqueeze(0).expand_as(output), output
        )

        return output

    def _recency_weights(self, seq_len: int, lengths: torch.Tensor) -> torch.Tensor:
        """Compute exponential recency weights.

        Position 0 (root ancestor) gets the lowest weight; position
        length-1 (the clause itself) gets weight 1.0.

        Returns:
            (B, seq_len) weight tensor.
        """
        B = lengths.shape[0]
        positions = torch.arange(seq_len, device=lengths.device, dtype=torch.float32)
        # Normalise position to [0, 1] range per sequence
        # pos_norm[b, i] = i / max(length[b] - 1, 1)
        max_pos = (lengths.float() - 1).clamp(min=1.0).unsqueeze(1)  # (B, 1)
        pos_norm = positions.unsqueeze(0) / max_pos  # (B, L)

        # Exponential weighting: exp(pos_norm * temperature) / exp(temperature)
        # This gives weight ~exp(-temp) for root and ~1.0 for the clause
        t = self.config.recency_temperature
        weights = torch.exp(pos_norm * t) / math.exp(t)

        return weights

    def pad_chains(
        self, chains: list[tuple[int, ...]], device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad variable-length chains to a uniform tensor.

        Args:
            chains: List of variable-length JustType tuples.
            device: Target device.

        Returns:
            (padded, lengths) where padded is (B, max_len) and
            lengths is (B,).
        """
        if not chains:
            dev = device or torch.device("cpu")
            return (
                torch.zeros(1, 1, dtype=torch.long, device=dev),
                torch.zeros(1, dtype=torch.long, device=dev),
            )

        max_len = min(
            max(len(c) for c in chains),
            self.config.max_chain_length,
        )
        max_len = max(max_len, 1)  # at least 1

        B = len(chains)
        padded = torch.zeros(B, max_len, dtype=torch.long)
        lengths = torch.zeros(B, dtype=torch.long)

        for i, chain in enumerate(chains):
            # Truncate to max_chain_length, keeping the most recent steps
            if len(chain) > max_len:
                chain = chain[-max_len:]
            L = len(chain)
            if L > 0:
                padded[i, :L] = torch.tensor(chain, dtype=torch.long)
            lengths[i] = L

        if device is not None:
            padded = padded.to(device)
            lengths = lengths.to(device)

        return padded, lengths

    def encode_single(self, chain: tuple[int, ...]) -> torch.Tensor:
        """Convenience: encode a single chain without batching.

        Returns:
            (output_dim,) 1-D tensor.
        """
        if not chain:
            return self.no_history_embedding.detach()

        truncated = chain[-self.config.max_chain_length:]
        t = torch.tensor([truncated], dtype=torch.long)
        lengths = torch.tensor([len(truncated)], dtype=torch.long)

        if next(self.parameters()).is_cuda:
            t = t.cuda()
            lengths = lengths.cuda()

        with torch.no_grad():
            return self.forward(t, lengths).squeeze(0)


def _sinusoidal_positional_encoding(max_len: int, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Returns:
        (max_len, dim) tensor with sin/cos positional encodings.
    """
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
