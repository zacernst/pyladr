"""Skip-gram training with negative sampling for Tree2Vec.

Implements the core skip-gram objective over token sequences generated
from tree walks. Given a center token and its context tokens (within a
sliding window), learns embeddings that predict context from center.

Uses negative sampling for efficient training without computing the
full softmax over the vocabulary.

Mathematical foundation:
    For center word w and context word c, maximize:
        log σ(v_c · v_w) + Σ_{k=1}^{K} E[log σ(-v_k · v_w)]
    where K is the number of negative samples and v are embedding vectors.
"""

from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class SkipGramConfig:
    """Configuration for skip-gram training.

    Attributes:
        embedding_dim: Dimensionality of learned embeddings.
        window_size: Context window size (tokens on each side of center).
        num_negative_samples: Number of negative samples per positive pair.
        learning_rate: Initial learning rate.
        min_learning_rate: Minimum learning rate floor.
        num_epochs: Number of training epochs over all walks.
        subsample_threshold: Subsampling threshold for frequent tokens.
        seed: Random seed for reproducibility.
    """

    embedding_dim: int = 64
    window_size: int = 3
    num_negative_samples: int = 5
    learning_rate: float = 0.025
    min_learning_rate: float = 0.0001
    num_epochs: int = 5
    subsample_threshold: float = 1e-3
    seed: int = 42
    online_vocab_extension: bool = True  # extend vocab for OOV tokens during online updates


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


class SkipGramTrainer:
    """Skip-gram with negative sampling trainer for tree token sequences.

    Pure Python implementation optimized for small vocabularies like
    the vampire.in domain. Maintains two embedding matrices (input/output)
    and trains via stochastic gradient descent.
    """

    def __init__(self, config: SkipGramConfig | None = None) -> None:
        self.config = config or SkipGramConfig()
        self._rng = random.Random(self.config.seed)

        # Vocabulary mapping
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: list[str] = []
        self._vocab_size: int = 0

        # Token frequency counts (for subsampling and negative sampling)
        self._token_counts: list[int] = []
        self._total_tokens: int = 0

        # Negative sampling distribution (unigram^0.75)
        self._neg_sample_table: list[int] = []
        self._NEG_TABLE_SIZE: int = 100_000

        # Embedding matrices: input (center) and output (context)
        # Stored as list-of-lists for pure Python
        self._input_embeddings: list[list[float]] = []
        self._output_embeddings: list[list[float]] = []

        # Training state
        self._trained: bool = False
        self._neg_table_dirty: bool = False  # set True when vocab extended

        # Thread safety: held exclusively during update_online, shared during get_embedding
        self._update_lock = threading.Lock()

    # ── Vocabulary ─────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def token_to_id(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def _build_vocab(self, walks: Sequence[Sequence[str]]) -> None:
        """Build vocabulary and frequency counts from walks."""
        counts: dict[str, int] = {}
        for walk in walks:
            for token in walk:
                counts[token] = counts.get(token, 0) + 1

        self._token_to_id = {}
        self._id_to_token = []
        self._token_counts = []
        self._total_tokens = 0

        # Sort by frequency (descending) for deterministic ordering
        for token, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            tid = len(self._id_to_token)
            self._token_to_id[token] = tid
            self._id_to_token.append(token)
            self._token_counts.append(count)
            self._total_tokens += count

        self._vocab_size = len(self._id_to_token)

    def _build_neg_sample_table(self) -> None:
        """Build unigram^0.75 table for negative sampling.

        Following the original Word2Vec paper, the negative sampling
        distribution is the unigram distribution raised to the 3/4 power,
        which smooths the distribution (upweights rare tokens).
        """
        power = 0.75
        total_pow = sum(c ** power for c in self._token_counts)

        table: list[int] = []
        cumulative = 0.0
        idx = 0
        for i in range(self._vocab_size):
            cumulative += (self._token_counts[i] ** power) / total_pow
            target = int(cumulative * self._NEG_TABLE_SIZE)
            while idx < target and idx < self._NEG_TABLE_SIZE:
                table.append(i)
                idx += 1

        # Fill remainder
        while len(table) < self._NEG_TABLE_SIZE:
            table.append(self._vocab_size - 1)

        self._neg_sample_table = table

    def _sample_negative(self) -> int:
        """Sample a token ID from the negative sampling distribution."""
        idx = self._rng.randrange(self._NEG_TABLE_SIZE)
        return self._neg_sample_table[idx]

    def _extend_vocab(self, token: str) -> int:
        """Add a new token to the vocabulary with a mean-initialized embedding.

        The new input embedding is initialized to the mean of all existing
        input embeddings, placing the token at the semantic centroid until
        training pairs adjust it. The output embedding starts at zero.

        Returns:
            The new token's ID.
        """
        tid = self._vocab_size
        self._token_to_id[token] = tid
        self._id_to_token.append(token)
        self._token_counts.append(1)
        self._total_tokens += 1
        self._vocab_size += 1

        dim = self.config.embedding_dim
        if self._input_embeddings:
            # Mean of existing input embeddings
            mean = [
                sum(row[i] for row in self._input_embeddings) / len(self._input_embeddings)
                for i in range(dim)
            ]
        else:
            mean = [0.0] * dim
        self._input_embeddings.append(mean)
        self._output_embeddings.append([0.0] * dim)

        self._neg_table_dirty = True
        return tid

    # ── Embedding initialization ───────────────────────────────────────

    def _init_embeddings(self) -> None:
        """Initialize embedding matrices with small random values."""
        dim = self.config.embedding_dim
        scale = 0.5 / dim

        self._input_embeddings = [
            [self._rng.uniform(-scale, scale) for _ in range(dim)]
            for _ in range(self._vocab_size)
        ]
        self._output_embeddings = [
            [0.0] * dim
            for _ in range(self._vocab_size)
        ]

    # ── Training ───────────────────────────────────────────────────────

    def train(
        self,
        walks: Sequence[Sequence[str]],
        progress_fn=None,
    ) -> dict[str, float]:
        """Train skip-gram embeddings on token sequences from tree walks.

        Args:
            walks: Sequences of string tokens from tree walks.
            progress_fn: Optional callable invoked after each epoch with
                (epoch: int, num_epochs: int, epoch_loss: float, lr: float).
                Useful for progress reporting in CLI contexts.

        Returns:
            Training statistics dict with keys:
                - "loss": Final average loss
                - "vocab_size": Number of unique tokens
                - "total_pairs": Total training pairs processed
                - "epochs": Number of epochs completed
                - "epoch_losses": Per-epoch average losses (list)
        """
        self._build_vocab(walks)

        if self._vocab_size == 0:
            self._trained = True
            return {"loss": 0.0, "vocab_size": 0, "total_pairs": 0, "epochs": 0,
                    "epoch_losses": []}

        self._build_neg_sample_table()
        self._init_embeddings()

        # Convert walks to ID sequences
        id_walks: list[list[int]] = []
        for walk in walks:
            id_walk = []
            for token in walk:
                tid = self._token_to_id.get(token)
                if tid is not None:
                    # Subsampling of frequent tokens
                    if self._should_keep(tid):
                        id_walk.append(tid)
            if id_walk:
                id_walks.append(id_walk)

        total_pairs = 0
        total_loss = 0.0
        epoch_losses: list[float] = []
        total_words = sum(len(w) for w in id_walks) * self.config.num_epochs

        word_count = 0
        lr = self.config.learning_rate

        for epoch in range(self.config.num_epochs):
            # Shuffle walks each epoch for better convergence
            shuffled = list(range(len(id_walks)))
            self._rng.shuffle(shuffled)

            epoch_loss = 0.0
            epoch_pairs = 0

            for walk_idx in shuffled:
                id_walk = id_walks[walk_idx]
                for center_pos, center_id in enumerate(id_walk):
                    # Dynamic window size (like original Word2Vec)
                    actual_window = self._rng.randint(1, self.config.window_size)

                    # Context tokens within window
                    start = max(0, center_pos - actual_window)
                    end = min(len(id_walk), center_pos + actual_window + 1)

                    for ctx_pos in range(start, end):
                        if ctx_pos == center_pos:
                            continue

                        ctx_id = id_walk[ctx_pos]
                        loss = self._train_pair(center_id, ctx_id, lr)
                        total_loss += loss
                        epoch_loss += loss
                        total_pairs += 1
                        epoch_pairs += 1

                    word_count += 1

                    # Linear learning rate decay
                    lr = self.config.learning_rate * max(
                        self.config.min_learning_rate / self.config.learning_rate,
                        1.0 - word_count / total_words,
                    )

            avg_epoch_loss = epoch_loss / max(epoch_pairs, 1)
            epoch_losses.append(avg_epoch_loss)
            if progress_fn is not None:
                progress_fn(epoch + 1, self.config.num_epochs, avg_epoch_loss, lr)

        self._trained = True

        avg_loss = total_loss / max(total_pairs, 1)
        return {
            "loss": avg_loss,
            "vocab_size": self._vocab_size,
            "total_pairs": total_pairs,
            "epochs": self.config.num_epochs,
            "epoch_losses": epoch_losses,
        }

    def _should_keep(self, token_id: int) -> bool:
        """Subsampling of frequent tokens (Mikolov et al., 2013).

        Probability of keeping a token with frequency f:
            P(keep) = sqrt(t / f) + t / f
        where t is the subsampling threshold.
        """
        if self.config.subsample_threshold <= 0:
            return True
        freq = self._token_counts[token_id] / self._total_tokens
        if freq == 0:
            return True
        prob = (math.sqrt(freq / self.config.subsample_threshold) + 1) * (
            self.config.subsample_threshold / freq
        )
        if prob >= 1.0:
            return True
        return self._rng.random() < prob

    def _train_pair(self, center_id: int, context_id: int, lr: float) -> float:
        """Train on one (center, context) pair with negative sampling.

        Returns the approximate loss for this pair.
        """
        dim = self.config.embedding_dim
        center_vec = self._input_embeddings[center_id]
        loss = 0.0

        # Gradient accumulator for center vector
        grad_center = [0.0] * dim

        # Positive sample: context_id with label 1
        # Negative samples: random tokens with label 0
        samples = [(context_id, 1.0)]
        for _ in range(self.config.num_negative_samples):
            neg_id = self._sample_negative()
            if neg_id == context_id:
                continue
            samples.append((neg_id, 0.0))

        for sample_id, label in samples:
            out_vec = self._output_embeddings[sample_id]

            # Dot product
            dot = sum(center_vec[d] * out_vec[d] for d in range(dim))

            # Clamp for numerical stability
            dot = max(-6.0, min(6.0, dot))

            sig = _sigmoid(dot)
            gradient = (label - sig) * lr

            # Accumulate gradient for center vector
            for d in range(dim):
                grad_center[d] += gradient * out_vec[d]

            # Update output vector
            for d in range(dim):
                out_vec[d] += gradient * center_vec[d]

            # Loss contribution
            if label == 1.0:
                loss -= math.log(max(sig, 1e-10))
            else:
                loss -= math.log(max(1.0 - sig, 1e-10))

        # Update center vector
        for d in range(dim):
            center_vec[d] += grad_center[d]

        return loss

    # ── Online update ─────────────────────────────────────────────────

    def update_online(
        self,
        walks: Sequence[Sequence[str]],
        learning_rate: float | None = None,
    ) -> dict[str, float | int]:
        """Perform mini-batch SGD update on existing embeddings.

        Runs one pass over the provided walks. When ``online_vocab_extension``
        is True (default), new tokens are added to the vocabulary with
        mean-initialized embeddings and trained immediately. Otherwise OOV
        tokens are silently skipped.

        Args:
            walks: Token sequences from tree walks of recently kept clauses.
            learning_rate: Learning rate for this update. Defaults to
                ``config.min_learning_rate`` for conservative online updates.

        Returns:
            Statistics dict with keys:
                - "pairs_trained": Number of (center, context) pairs updated.
                - "loss": Average loss over trained pairs.
                - "oov_skipped": Number of OOV tokens skipped.
        """
        if not self._trained:
            return {"pairs_trained": 0, "loss": 0.0, "oov_skipped": 0}

        with self._update_lock:
            lr = learning_rate if learning_rate is not None else self.config.min_learning_rate

            total_pairs = 0
            total_loss = 0.0
            oov_skipped = 0
            vocab_extended = 0

            # Pass 1: convert walks to ID sequences, extending vocab for new tokens
            id_walks: list[list[int]] = []
            for walk in walks:
                id_walk: list[int] = []
                for token in walk:
                    tid = self._token_to_id.get(token)
                    if tid is not None:
                        id_walk.append(tid)
                    elif self.config.online_vocab_extension:
                        tid = self._extend_vocab(token)
                        id_walk.append(tid)
                        vocab_extended += 1
                    else:
                        oov_skipped += 1
                id_walks.append(id_walk)

            # Rebuild negative sampling table now if vocab was extended
            if self._neg_table_dirty:
                self._build_neg_sample_table()
                self._neg_table_dirty = False

            # Pass 2: train skip-gram pairs
            for id_walk in id_walks:
                if len(id_walk) < 2:
                    continue

                for center_pos, center_id in enumerate(id_walk):
                    actual_window = self._rng.randint(1, self.config.window_size)
                    start = max(0, center_pos - actual_window)
                    end = min(len(id_walk), center_pos + actual_window + 1)

                    for ctx_pos in range(start, end):
                        if ctx_pos == center_pos:
                            continue
                        ctx_id = id_walk[ctx_pos]
                        loss = self._train_pair(center_id, ctx_id, lr)
                        total_loss += loss
                        total_pairs += 1

            avg_loss = total_loss / max(total_pairs, 1)
            return {
                "pairs_trained": total_pairs,
                "loss": avg_loss,
                "oov_skipped": oov_skipped,
                "vocab_extended": vocab_extended,
            }

    # ── Embedding access ───────────────────────────────────────────────

    def get_embedding(self, token: str) -> list[float] | None:
        """Get the learned embedding for a token.

        Returns None if the token is not in the vocabulary.
        """
        if not self._trained:
            return None
        with self._update_lock:
            tid = self._token_to_id.get(token)
            if tid is None:
                return None
            return list(self._input_embeddings[tid])

    def get_all_embeddings(self) -> dict[str, list[float]]:
        """Get all learned embeddings as a token -> vector dict."""
        if not self._trained:
            return {}
        return {
            token: list(self._input_embeddings[tid])
            for token, tid in self._token_to_id.items()
        }

    def most_similar(self, token: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Find the most similar tokens by cosine similarity.

        Args:
            token: Query token.
            top_k: Number of results to return.

        Returns:
            List of (token, similarity) pairs, sorted by similarity descending.
        """
        if not self._trained:
            return []
        tid = self._token_to_id.get(token)
        if tid is None:
            return []

        query = self._input_embeddings[tid]
        query_norm = math.sqrt(sum(x * x for x in query))
        if query_norm == 0:
            return []

        similarities: list[tuple[str, float]] = []
        for other_tid in range(self._vocab_size):
            if other_tid == tid:
                continue
            other = self._input_embeddings[other_tid]
            other_norm = math.sqrt(sum(x * x for x in other))
            if other_norm == 0:
                continue
            dot = sum(query[d] * other[d] for d in range(self.config.embedding_dim))
            cos_sim = dot / (query_norm * other_norm)
            similarities.append((self._id_to_token[other_tid], cos_sim))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]
