"""RNN2Vec: RNN-based embedding generator for logical formula trees.

Encodes tree-walk token sequences with an RNN trained via contrastive +
next-token auxiliary objectives to produce clause/term embeddings.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from pyladr.core.clause import Clause
from pyladr.core.term import Term
from pyladr.ml.rnn2vec.tokenizer import TokenVocab
from pyladr.ml.rnn2vec.walks import TreeWalker, WalkConfig, WalkType

logger = logging.getLogger(__name__)

# Guard torch — module is importable without torch.
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "RNN2Vec requires torch. Install with: pip install torch"
        )


@dataclass(frozen=True, slots=True)
class RNN2VecConfig:
    """Top-level RNN2Vec configuration.

    Attributes:
        walk_config: Tree walk generation settings.
        rnn_config: RNN encoder configuration.
        training_epochs: Number of training epochs.
        learning_rate: Initial learning rate for Adam optimizer.
        batch_size: Mini-batch size for training.
        contrastive_temperature: Temperature for InfoNCE loss.
        next_token_weight: Auxiliary next-token loss weight (0 = disable).
        max_bptt_steps: Maximum BPTT truncation length (0 = unlimited).
        seed: Random seed for reproducibility.
        train_ratio: Fraction of clauses used for gradient updates.
        val_ratio: Fraction of clauses held out for validation loss.
        test_ratio: Fraction of clauses held out for test loss.
    """

    walk_config: WalkConfig = field(default_factory=WalkConfig)
    rnn_config: "RNNEmbeddingConfig" = field(default_factory=lambda: _default_rnn_config())
    training_epochs: int = 5
    learning_rate: float = 0.001
    batch_size: int = 32
    contrastive_temperature: float = 0.1
    next_token_weight: float = 0.2
    max_bptt_steps: int = 30
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self) -> None:
        for name, ratio in (
            ("train_ratio", self.train_ratio),
            ("val_ratio", self.val_ratio),
            ("test_ratio", self.test_ratio),
        ):
            if ratio < 0:
                raise ValueError(f"{name} must be non-negative, got {ratio}")
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"train/val/test ratios must sum to 1.0, got {total:.4f} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )
        if self.train_ratio <= 0:
            raise ValueError(f"train_ratio must be > 0, got {self.train_ratio}")


def _default_rnn_config():
    from pyladr.ml.rnn2vec.encoder import RNNEmbeddingConfig

    return RNNEmbeddingConfig()


class RNN2Vec:
    """Unsupervised RNN-based embedding generator for logical formula trees.

    An RNN encoder processes tree walk sequences to produce fixed-dimensional
    embeddings. Training uses InfoNCE contrastive loss with optional
    next-token prediction auxiliary.
    """

    def __init__(self, config: RNN2VecConfig | None = None) -> None:
        self.config = config or RNN2VecConfig()
        self._walker = TreeWalker(self.config.walk_config)
        self._vocab: TokenVocab | None = None
        self._encoder: object | None = None  # RNNEncoder when torch available
        self._next_token_head: object | None = None  # nn.Linear for aux loss
        self._trained: bool = False

    @property
    def embedding_dim(self) -> int:
        """Effective output embedding dimension."""
        return self.config.rnn_config.embedding_dim

    @property
    def trained(self) -> bool:
        return self._trained

    @property
    def vocab_size(self) -> int:
        if self._vocab is None:
            return 0
        return self._vocab.size

    # ── Training ───────────────────────────────────────────────────────

    def train(
        self,
        clauses: Sequence[Clause],
        progress_fn: Callable[[int, int, float, float], None] | None = None,
    ) -> dict[str, float]:
        """Train RNN2Vec embeddings from clauses.

        Pipeline:
        1. Split clauses into train/val/test (held-out — no data leakage).
        2. Generate walks per split; build TokenVocab from training walks only.
        3. Create RNNEncoder.
        4. Train with Adam on training walks: contrastive + optional next-token aux.
        5. Evaluate held-out val/test loss without gradient updates.

        The clause-level split prevents contrastive pair leakage (walks from a
        single clause cannot straddle the train/val/test boundary).

        Returns training statistics dict including `loss`, `val_loss`, `test_loss`,
        and per-split clause counts.
        """
        _require_torch()
        from pyladr.ml.rnn2vec.encoder import RNNEncoder

        rng = random.Random(self.config.seed)
        torch.manual_seed(self.config.seed)

        # 1. Split clauses into train/val/test (prevents data leakage)
        train_idx, val_idx, test_idx = self._split_indices(len(clauses), rng)

        train_walks_per_group: list[list[list[str]]] = []
        train_all_walks: list[list[str]] = []
        for i in train_idx:
            walks = self._walker.walks_from_clause(clauses[i])
            train_walks_per_group.append(walks)
            train_all_walks.extend(walks)

        val_walks_per_group = [
            self._walker.walks_from_clause(clauses[i]) for i in val_idx
        ]
        test_walks_per_group = [
            self._walker.walks_from_clause(clauses[i]) for i in test_idx
        ]

        if not train_all_walks:
            return {
                "loss": 0.0, "vocab_size": 0, "epochs": 0, "training_pairs": 0,
                "val_loss": 0.0, "test_loss": 0.0,
                "train_clauses": float(len(train_idx)),
                "val_clauses": float(len(val_idx)),
                "test_clauses": float(len(test_idx)),
            }

        # 2. Build vocabulary from training walks only
        self._vocab = TokenVocab.from_walks(train_all_walks)

        # 3. Create encoder
        self._encoder = RNNEncoder(self._vocab.size, self.config.rnn_config)
        self._encoder.train()

        # Next-token prediction head (if enabled)
        if self.config.next_token_weight > 0:
            effective_hidden = self._encoder.effective_hidden_dim
            self._next_token_head = nn.Linear(effective_hidden, self._vocab.size)
        else:
            self._next_token_head = None

        # 4. Encode training walks
        train_walk_id_seqs: list[list[int]] = []
        train_walk_group_idx: list[int] = []
        for ci, walks in enumerate(train_walks_per_group):
            for walk in walks:
                train_walk_id_seqs.append(self._vocab.encode_walk(walk))
                train_walk_group_idx.append(ci)

        # 5. Train
        params = list(self._encoder.parameters())
        if self._next_token_head is not None:
            params += list(self._next_token_head.parameters())
        optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

        total_loss = 0.0
        total_pairs = 0
        num_epochs = self.config.training_epochs

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_pairs = 0

            batches = self._build_contrastive_batches(
                train_walk_id_seqs, train_walk_group_idx, train_walks_per_group, rng
            )

            for batch in batches:
                optimizer.zero_grad()
                loss = self._compute_batch_loss(
                    batch, train_walk_id_seqs, train_walk_group_idx, rng
                )
                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    epoch_pairs += len(batch)

            avg_epoch_loss = epoch_loss / max(epoch_pairs, 1)
            total_loss += epoch_loss
            total_pairs += epoch_pairs

            lr = optimizer.param_groups[0]["lr"]
            if progress_fn is not None:
                progress_fn(epoch, num_epochs, avg_epoch_loss, lr)

            logger.debug(
                "RNN2Vec epoch %d/%d: loss=%.4f, pairs=%d",
                epoch + 1, num_epochs, avg_epoch_loss, epoch_pairs,
            )

        self._encoder.eval()
        self._trained = True

        # 6. Evaluate held-out val/test loss (no gradient updates)
        val_loss = self._evaluate_loss(val_walks_per_group, rng)
        test_loss = self._evaluate_loss(test_walks_per_group, rng)

        return {
            "loss": total_loss / max(total_pairs, 1),
            "val_loss": val_loss,
            "test_loss": test_loss,
            "vocab_size": float(self._vocab.size),
            "epochs": float(num_epochs),
            "training_pairs": float(total_pairs),
            "train_clauses": float(len(train_idx)),
            "val_clauses": float(len(val_idx)),
            "test_clauses": float(len(test_idx)),
        }

    def train_from_terms(self, terms: Sequence[Term]) -> dict[str, float]:
        """Train directly from terms without clause wrapper.

        Wraps each term as a single-literal clause for walk generation.
        Applies the same train/val/test split as `train()` at the term level.
        """
        _require_torch()

        rng = random.Random(self.config.seed)
        torch.manual_seed(self.config.seed)
        from pyladr.ml.rnn2vec.encoder import RNNEncoder

        train_idx, val_idx, test_idx = self._split_indices(len(terms), rng)

        train_walks_per_group: list[list[list[str]]] = []
        train_all_walks: list[list[str]] = []
        for i in train_idx:
            walks = self._walker.walks_from_term(terms[i])
            train_walks_per_group.append(walks)
            train_all_walks.extend(walks)

        val_walks_per_group = [
            self._walker.walks_from_term(terms[i]) for i in val_idx
        ]
        test_walks_per_group = [
            self._walker.walks_from_term(terms[i]) for i in test_idx
        ]

        if not train_all_walks:
            return {
                "loss": 0.0, "vocab_size": 0, "epochs": 0, "training_pairs": 0,
                "val_loss": 0.0, "test_loss": 0.0,
                "train_clauses": float(len(train_idx)),
                "val_clauses": float(len(val_idx)),
                "test_clauses": float(len(test_idx)),
            }

        self._vocab = TokenVocab.from_walks(train_all_walks)
        self._encoder = RNNEncoder(self._vocab.size, self.config.rnn_config)
        self._encoder.train()

        if self.config.next_token_weight > 0:
            effective_hidden = self._encoder.effective_hidden_dim
            self._next_token_head = nn.Linear(effective_hidden, self._vocab.size)
        else:
            self._next_token_head = None

        train_walk_id_seqs: list[list[int]] = []
        train_walk_group_idx: list[int] = []
        for ti, walks in enumerate(train_walks_per_group):
            for walk in walks:
                train_walk_id_seqs.append(self._vocab.encode_walk(walk))
                train_walk_group_idx.append(ti)

        params = list(self._encoder.parameters())
        if self._next_token_head is not None:
            params += list(self._next_token_head.parameters())
        optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

        total_loss = 0.0
        total_pairs = 0

        for epoch in range(self.config.training_epochs):
            batches = self._build_contrastive_batches(
                train_walk_id_seqs, train_walk_group_idx, train_walks_per_group, rng
            )
            for batch in batches:
                optimizer.zero_grad()
                loss = self._compute_batch_loss(
                    batch, train_walk_id_seqs, train_walk_group_idx, rng
                )
                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    total_pairs += len(batch)

        self._encoder.eval()
        self._trained = True

        val_loss = self._evaluate_loss(val_walks_per_group, rng)
        test_loss = self._evaluate_loss(test_walks_per_group, rng)

        return {
            "loss": total_loss / max(total_pairs, 1),
            "val_loss": val_loss,
            "test_loss": test_loss,
            "vocab_size": float(self._vocab.size),
            "epochs": float(self.config.training_epochs),
            "training_pairs": float(total_pairs),
            "train_clauses": float(len(train_idx)),
            "val_clauses": float(len(val_idx)),
            "test_clauses": float(len(test_idx)),
        }

    # ── Online update ──────────────────────────────────────────────────

    def update_online(
        self,
        clauses: Sequence[Clause],
        learning_rate: float | None = None,
    ) -> dict[str, float | int]:
        """Incremental update from recently kept clauses.

        Runs mini-batch gradient steps on walks from new clauses.
        New tokens get mean-initialized embeddings.
        No-op if not trained yet.
        """
        if not self._trained or self._encoder is None or self._vocab is None:
            return {"pairs_trained": 0, "loss": 0.0, "oov_skipped": 0}

        _require_torch()

        lr = learning_rate or self.config.learning_rate * 0.1
        rng = random.Random()

        # Generate walks grouped by clause
        clause_walks: list[list[list[str]]] = []
        all_walks: list[list[str]] = []
        for clause in clauses:
            walks = self._walker.walks_from_clause(clause)
            clause_walks.append(walks)
            all_walks.extend(walks)

        if not all_walks:
            return {"pairs_trained": 0, "loss": 0.0, "oov_skipped": 0}

        # Extend vocabulary for new tokens
        oov_count = 0
        for walk in all_walks:
            for token in walk:
                if self._vocab._token_to_id.get(token) is None:
                    self._vocab.extend(token)
                    oov_count += 1

        # Expand encoder embedding if vocab grew
        if self._vocab.size > self._encoder.vocab_size:
            self._encoder.expand_vocab(self._vocab.size)

        # Encode walks
        walk_id_seqs: list[list[int]] = []
        walk_clause_idx: list[int] = []
        for ci, walks in enumerate(clause_walks):
            for walk in walks:
                walk_id_seqs.append(self._vocab.encode_walk(walk))
                walk_clause_idx.append(ci)

        # Train one pass
        self._encoder.train()
        params = list(self._encoder.parameters())
        if self._next_token_head is not None:
            # Resize next-token head if vocab grew
            old_out = self._next_token_head.out_features
            if self._vocab.size > old_out:
                effective_hidden = self._encoder.effective_hidden_dim
                self._next_token_head = nn.Linear(effective_hidden, self._vocab.size)
            params += list(self._next_token_head.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        total_loss = 0.0
        total_pairs = 0

        batches = self._build_contrastive_batches(
            walk_id_seqs, walk_clause_idx, clause_walks, rng
        )
        for batch in batches:
            optimizer.zero_grad()
            loss = self._compute_batch_loss(batch, walk_id_seqs, walk_clause_idx, rng)
            if loss is not None:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_pairs += len(batch)

        self._encoder.eval()

        return {
            "pairs_trained": total_pairs,
            "loss": total_loss / max(total_pairs, 1),
            "oov_skipped": oov_count,
        }

    # ── Embedding generation ───────────────────────────────────────────

    def embed_term(self, term: Term) -> list[float] | None:
        """Generate walks from term, encode each via RNN, average."""
        if not self._trained or self._encoder is None or self._vocab is None:
            return None

        walks = self._walker.walks_from_term(term)
        if not walks:
            return None

        return self._encode_and_average_walks(walks)

    def embed_clause(self, clause: Clause) -> list[float] | None:
        """Embed a clause by averaging normalized per-literal embeddings.

        For each literal, generate walks, encode each, apply sign scaling.
        Average all literal embeddings, then normalize.
        """
        if not self._trained or self._encoder is None or self._vocab is None:
            return None

        lit_embeddings: list[list[float]] = []
        for lit in clause.literals:
            sign_scale = 1.0 if lit.sign else -1.0
            walks = self._walker.walks_from_term(lit.atom)
            if not walks:
                continue
            avg = self._encode_and_average_walks(walks)
            if avg is not None:
                lit_embeddings.append([sign_scale * v for v in avg])

        if not lit_embeddings:
            return None

        result = self._mean_vectors(lit_embeddings)
        return self._normalize_vec(result)

    def embed_clauses_batch(
        self, clauses: Sequence[Clause]
    ) -> list[list[float] | None]:
        """Efficient batched clause embedding.

        Collects all walks from all clauses, runs a single batched
        RNN forward pass, then groups results by clause.
        """
        if not self._trained or self._encoder is None or self._vocab is None:
            return [None] * len(clauses)

        _require_torch()

        # Collect walks for each literal in each clause
        # Track: (clause_idx, lit_idx, sign_scale, walk_indices)
        all_walk_ids: list[list[int]] = []
        clause_lit_info: list[list[tuple[float, list[int]]]] = []

        for ci, clause in enumerate(clauses):
            lit_info: list[tuple[float, list[int]]] = []
            for lit in clause.literals:
                sign_scale = 1.0 if lit.sign else -1.0
                walks = self._walker.walks_from_term(lit.atom)
                if not walks:
                    continue
                walk_indices: list[int] = []
                for walk in walks:
                    walk_indices.append(len(all_walk_ids))
                    all_walk_ids.append(self._vocab.encode_walk(walk))
                lit_info.append((sign_scale, walk_indices))
            clause_lit_info.append(lit_info)

        if not all_walk_ids:
            return [None] * len(clauses)

        # Single batched forward pass
        embeddings = self._batch_encode(all_walk_ids)

        # Group by clause
        results: list[list[float] | None] = []
        for ci, lit_info in enumerate(clause_lit_info):
            if not lit_info:
                results.append(None)
                continue
            lit_embeddings: list[list[float]] = []
            for sign_scale, walk_indices in lit_info:
                walk_embs = [embeddings[wi] for wi in walk_indices if embeddings[wi] is not None]
                if walk_embs:
                    avg = self._mean_vectors(walk_embs)
                    lit_embeddings.append([sign_scale * v for v in avg])
            if not lit_embeddings:
                results.append(None)
            else:
                results.append(self._normalize_vec(self._mean_vectors(lit_embeddings)))

        return results

    # ── Similarity ─────────────────────────────────────────────────────

    def similarity(self, term_a: Term, term_b: Term) -> float | None:
        """Cosine similarity between two term embeddings."""
        emb_a = self.embed_term(term_a)
        emb_b = self.embed_term(term_b)
        if emb_a is None or emb_b is None:
            return None
        return self._cosine_similarity(emb_a, emb_b)

    def clause_similarity(self, clause_a: Clause, clause_b: Clause) -> float | None:
        """Cosine similarity between two clause embeddings."""
        emb_a = self.embed_clause(clause_a)
        emb_b = self.embed_clause(clause_b)
        if emb_a is None or emb_b is None:
            return None
        return self._cosine_similarity(emb_a, emb_b)

    # ── Token-level access ─────────────────────────────────────────────

    def get_token_embedding(self, token: str) -> list[float] | None:
        """Get the learned lookup embedding for a specific token."""
        if not self._trained or self._encoder is None or self._vocab is None:
            return None
        tid = self._vocab._token_to_id.get(token)
        if tid is None:
            return None
        with torch.no_grad():
            return self._encoder.token_embedding.weight[tid].tolist()

    # ── Serialization ──────────────────────────────────────────────────

    SAVE_FORMAT_VERSION = 1

    def save(self, path: str | Path) -> None:
        """Save to directory: config.json, vocab.json, model.pt.

        Raises RuntimeError if not trained.
        """
        if not self._trained:
            raise RuntimeError("Cannot save an untrained RNN2Vec model.")

        _require_torch()

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        wc = self.config.walk_config
        rc = self.config.rnn_config

        config_data = {
            "format_version": self.SAVE_FORMAT_VERSION,
            "walk_config": {
                "walk_types": [wt.name for wt in wc.walk_types],
                "num_random_walks": wc.num_random_walks,
                "max_walk_length": wc.max_walk_length,
                "include_position": wc.include_position,
                "include_depth": wc.include_depth,
                "include_var_identity": wc.include_var_identity,
                "skip_predicate_wrapper": wc.skip_predicate_wrapper,
                "seed": wc.seed,
            },
            "rnn_config": {
                "rnn_type": rc.rnn_type,
                "input_dim": rc.input_dim,
                "hidden_dim": rc.hidden_dim,
                "embedding_dim": rc.embedding_dim,
                "num_layers": rc.num_layers,
                "bidirectional": rc.bidirectional,
                "dropout": rc.dropout,
                "composition": rc.composition,
                "normalize": rc.normalize,
                "seed": rc.seed,
            },
            "training_epochs": self.config.training_epochs,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "contrastive_temperature": self.config.contrastive_temperature,
            "next_token_weight": self.config.next_token_weight,
            "max_bptt_steps": self.config.max_bptt_steps,
            "seed": self.config.seed,
            "train_ratio": self.config.train_ratio,
            "val_ratio": self.config.val_ratio,
            "test_ratio": self.config.test_ratio,
        }

        (save_dir / "config.json").write_text(
            json.dumps(config_data, indent=2), encoding="utf-8"
        )

        # Save vocabulary
        (save_dir / "vocab.json").write_text(
            json.dumps(self._vocab.to_dict()), encoding="utf-8"
        )

        # Save model weights
        state = {"encoder": self._encoder.state_dict()}
        if self._next_token_head is not None:
            state["next_token_head"] = self._next_token_head.state_dict()
        torch.save(state, save_dir / "model.pt")

    @classmethod
    def load(cls, path: str | Path) -> RNN2Vec:
        """Load from directory saved by save()."""
        _require_torch()
        from pyladr.ml.rnn2vec.encoder import RNNEmbeddingConfig, RNNEncoder

        load_dir = Path(path)
        config_data = json.loads((load_dir / "config.json").read_text(encoding="utf-8"))

        if config_data.get("format_version") != cls.SAVE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported RNN2Vec format version {config_data.get('format_version')!r}; "
                f"expected {cls.SAVE_FORMAT_VERSION}"
            )

        wc_d = config_data["walk_config"]
        rc_d = config_data["rnn_config"]

        walk_config = WalkConfig(
            walk_types=tuple(WalkType[n] for n in wc_d["walk_types"]),
            num_random_walks=wc_d["num_random_walks"],
            max_walk_length=wc_d["max_walk_length"],
            include_position=wc_d["include_position"],
            include_depth=wc_d["include_depth"],
            include_var_identity=wc_d["include_var_identity"],
            skip_predicate_wrapper=wc_d["skip_predicate_wrapper"],
            seed=wc_d["seed"],
        )

        rnn_config = RNNEmbeddingConfig(
            rnn_type=rc_d["rnn_type"],
            input_dim=rc_d["input_dim"],
            hidden_dim=rc_d["hidden_dim"],
            embedding_dim=rc_d["embedding_dim"],
            num_layers=rc_d["num_layers"],
            bidirectional=rc_d["bidirectional"],
            dropout=rc_d["dropout"],
            composition=rc_d["composition"],
            normalize=rc_d["normalize"],
            seed=rc_d["seed"],
        )

        rnn2vec_config = RNN2VecConfig(
            walk_config=walk_config,
            rnn_config=rnn_config,
            training_epochs=config_data["training_epochs"],
            learning_rate=config_data["learning_rate"],
            batch_size=config_data["batch_size"],
            contrastive_temperature=config_data["contrastive_temperature"],
            next_token_weight=config_data["next_token_weight"],
            max_bptt_steps=config_data["max_bptt_steps"],
            seed=config_data["seed"],
            train_ratio=config_data.get("train_ratio", 0.70),
            val_ratio=config_data.get("val_ratio", 0.15),
            test_ratio=config_data.get("test_ratio", 0.15),
        )

        # Load vocabulary
        vocab_data = json.loads((load_dir / "vocab.json").read_text(encoding="utf-8"))
        vocab = TokenVocab.from_dict(vocab_data)

        # Create instance
        instance = cls(rnn2vec_config)
        instance._vocab = vocab
        instance._encoder = RNNEncoder(vocab.size, rnn_config)

        # Load model weights
        state = torch.load(load_dir / "model.pt", weights_only=True)
        instance._encoder.load_state_dict(state["encoder"])
        instance._encoder.eval()

        if "next_token_head" in state:
            effective_hidden = instance._encoder.effective_hidden_dim
            instance._next_token_head = nn.Linear(effective_hidden, vocab.size)
            instance._next_token_head.load_state_dict(state["next_token_head"])
        else:
            instance._next_token_head = None

        instance._trained = True
        return instance

    # ── Internal helpers ───────────────────────────────────────────────

    def _split_indices(
        self,
        n: int,
        rng: random.Random,
    ) -> tuple[list[int], list[int], list[int]]:
        """Partition n clause/term indices into train/val/test.

        Splits at the clause level (not the walk level) to ensure no
        contrastive pair straddles the train/val/test boundary. Deterministic
        given the caller's seeded `rng`.

        Small-N handling: training gets at least one item whenever n >= 1.
        Val/test may be empty for very small corpora — their loss is reported
        as 0.0 in that case.
        """
        if n == 0:
            return [], [], []

        indices = list(range(n))
        rng.shuffle(indices)

        n_train = int(n * self.config.train_ratio)
        n_val = int(n * (self.config.train_ratio + self.config.val_ratio)) - n_train

        if n_train == 0:
            n_train = 1
            if n_val > 0:
                n_val -= 1

        n_val = max(0, n_val)
        if n_train + n_val > n:
            n_val = n - n_train

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        return train_idx, val_idx, test_idx

    def _evaluate_loss(
        self,
        walks_per_group: list[list[list[str]]],
        rng: random.Random,
    ) -> float:
        """Compute average loss on held-out data without gradient updates.

        Uses the same contrastive + next-token objective as training but
        performs no backprop and no optimizer step. Returns 0.0 if the held-out
        set is empty or no contrastive pairs can be formed (e.g. only single-walk
        groups with next_token_weight=0).
        """
        if not walks_per_group or self._vocab is None or self._encoder is None:
            return 0.0

        walk_id_seqs: list[list[int]] = []
        walk_group_idx: list[int] = []
        for gi, walks in enumerate(walks_per_group):
            for walk in walks:
                walk_id_seqs.append(self._vocab.encode_walk(walk))
                walk_group_idx.append(gi)

        if not walk_id_seqs:
            return 0.0

        batches = self._build_contrastive_batches(
            walk_id_seqs, walk_group_idx, walks_per_group, rng
        )

        total_loss = 0.0
        total_pairs = 0
        with torch.no_grad():
            for batch in batches:
                loss = self._compute_batch_loss(
                    batch, walk_id_seqs, walk_group_idx, rng
                )
                if loss is not None:
                    total_loss += loss.item()
                    total_pairs += len(batch)

        return total_loss / max(total_pairs, 1)

    def _build_contrastive_batches(
        self,
        walk_id_seqs: list[list[int]],
        walk_clause_idx: list[int],
        group_walks: list[list[list[str]]],
        rng: random.Random,
    ) -> list[list[int]]:
        """Build batches of walk indices that have contrastive pairs.

        Returns list of batches, where each batch is a list of walk indices.
        Only includes walks from groups with >= 2 walks (needed for positive pairs).
        """
        # Group walk indices by clause/term
        group_to_walks: dict[int, list[int]] = {}
        for wi, ci in enumerate(walk_clause_idx):
            group_to_walks.setdefault(ci, []).append(wi)

        # Collect indices from groups with at least 2 walks
        eligible: list[int] = []
        for indices in group_to_walks.values():
            if len(indices) >= 2:
                eligible.extend(indices)

        # If next_token_weight > 0, also include single-walk groups
        if self.config.next_token_weight > 0:
            for indices in group_to_walks.values():
                if len(indices) == 1:
                    eligible.extend(indices)

        rng.shuffle(eligible)

        # Split into batches
        bs = self.config.batch_size
        return [eligible[i : i + bs] for i in range(0, len(eligible), bs)]

    def _compute_batch_loss(
        self,
        batch_indices: list[int],
        walk_id_seqs: list[list[int]],
        walk_clause_idx: list[int],
        rng: random.Random,
    ) -> torch.Tensor | None:
        """Compute combined contrastive + next-token loss for a batch."""
        if not batch_indices:
            return None

        # Encode all walks in the batch
        batch_seqs = [walk_id_seqs[i] for i in batch_indices]
        batch_groups = [walk_clause_idx[i] for i in batch_indices]

        # Pad sequences
        max_len = max(len(s) for s in batch_seqs)
        padded = [s + [0] * (max_len - len(s)) for s in batch_seqs]
        lengths = [len(s) for s in batch_seqs]

        ids_t = torch.tensor(padded, dtype=torch.long)
        lens_t = torch.tensor(lengths, dtype=torch.long)

        # Forward pass through encoder
        embeddings = self._encoder(ids_t, lens_t)  # (batch, embedding_dim)

        total_loss = torch.tensor(0.0)
        has_loss = False

        # ── Contrastive loss (InfoNCE) ─────────────────────────────
        contrastive_loss = self._infonce_loss(
            embeddings, batch_groups, batch_indices, walk_clause_idx
        )
        if contrastive_loss is not None:
            total_loss = total_loss + contrastive_loss
            has_loss = True

        # ── Next-token auxiliary loss ──────────────────────────────
        if self.config.next_token_weight > 0 and self._next_token_head is not None:
            nt_loss = self._next_token_loss(ids_t, lens_t)
            if nt_loss is not None:
                total_loss = total_loss + self.config.next_token_weight * nt_loss
                has_loss = True

        return total_loss if has_loss else None

    def _infonce_loss(
        self,
        embeddings: torch.Tensor,
        batch_groups: list[int],
        batch_indices: list[int],
        walk_clause_idx: list[int],
    ) -> torch.Tensor | None:
        """InfoNCE contrastive loss.

        For each anchor, one positive (same group) and all others as negatives.
        """
        n = embeddings.shape[0]
        if n < 2:
            return None

        # Group walks by their clause/term index
        group_to_batch_pos: dict[int, list[int]] = {}
        for bi, gi in enumerate(batch_groups):
            group_to_batch_pos.setdefault(gi, []).append(bi)

        # Only consider walks from groups with >= 2 in this batch
        anchors: list[int] = []
        positives: list[int] = []
        for gi, positions in group_to_batch_pos.items():
            if len(positions) < 2:
                continue
            for i, ai in enumerate(positions):
                # Pick a random positive from same group
                candidates = positions[:i] + positions[i + 1 :]
                pi = candidates[torch.randint(len(candidates), (1,)).item()]
                anchors.append(ai)
                positives.append(pi)

        if not anchors:
            return None

        tau = self.config.contrastive_temperature
        anchor_emb = embeddings[anchors]  # (num_pairs, dim)
        positive_emb = embeddings[positives]  # (num_pairs, dim)

        # Similarity of anchor with all embeddings in batch
        # (num_pairs, n)
        all_sim = torch.mm(anchor_emb, embeddings.t()) / tau
        # Positive similarity
        pos_sim = (anchor_emb * positive_emb).sum(dim=-1) / tau  # (num_pairs,)

        # InfoNCE: -log(exp(pos) / sum(exp(all)))
        # Subtract max for numerical stability
        log_sum_exp = torch.logsumexp(all_sim, dim=-1)  # (num_pairs,)
        loss = (-pos_sim + log_sum_exp).mean()

        return loss

    def _next_token_loss(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor | None:
        """Next-token prediction auxiliary loss.

        Uses the RNN's internal hidden states (before projection)
        to predict the next token at each timestep.
        """
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        # Get RNN hidden states (before projection/composition)
        embedded = self._encoder.token_embedding(token_ids)
        lengths_clamped = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(
            embedded, lengths_clamped, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self._encoder.rnn(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # output: (batch, seq_len, effective_hidden)

        batch_size, seq_len, hidden = output.shape
        if seq_len < 2:
            return None

        # Predict next token from each hidden state h_t -> token_{t+1}
        # Use h_0..h_{T-2} to predict tokens at positions 1..T-1
        h_input = output[:, :-1, :].reshape(-1, hidden)  # ((batch*(seq_len-1)), hidden)
        logits = self._next_token_head(h_input)  # ((batch*(seq_len-1)), vocab_size)

        targets = token_ids[:, 1:].reshape(-1)  # ((batch*(seq_len-1)),)

        # Create mask for non-padding positions
        # A target position is valid if position < length - 1 (both input and target are real)
        arange = torch.arange(seq_len - 1, device=token_ids.device).unsqueeze(0)
        mask = arange < (lengths.unsqueeze(1) - 1)
        mask_flat = mask.reshape(-1)

        if mask_flat.sum() == 0:
            return None

        loss = nn.functional.cross_entropy(
            logits[mask_flat], targets[mask_flat], ignore_index=0
        )
        return loss

    def _encode_and_average_walks(self, walks: list[list[str]]) -> list[float] | None:
        """Encode walks via RNN and average the embeddings."""
        if not walks or self._vocab is None:
            return None

        id_seqs = [self._vocab.encode_walk(w) for w in walks]
        embeddings = self._batch_encode(id_seqs)

        valid = [e for e in embeddings if e is not None]
        if not valid:
            return None

        return self._mean_vectors(valid)

    def _batch_encode(self, id_seqs: list[list[int]]) -> list[list[float] | None]:
        """Run batched RNN encoding on id sequences."""
        if not id_seqs or self._encoder is None:
            return [None] * len(id_seqs)

        # Filter empty sequences
        non_empty: list[tuple[int, list[int]]] = [
            (i, s) for i, s in enumerate(id_seqs) if s
        ]
        if not non_empty:
            return [None] * len(id_seqs)

        indices, seqs = zip(*non_empty)
        max_len = max(len(s) for s in seqs)
        padded = [list(s) + [0] * (max_len - len(s)) for s in seqs]
        lengths = [len(s) for s in seqs]

        with torch.no_grad():
            ids_t = torch.tensor(padded, dtype=torch.long)
            lens_t = torch.tensor(lengths, dtype=torch.long)
            emb = self._encoder(ids_t, lens_t)  # (n, embedding_dim)

        results: list[list[float] | None] = [None] * len(id_seqs)
        for j, orig_idx in enumerate(indices):
            results[orig_idx] = emb[j].tolist()
        return results

    @staticmethod
    def _mean_vectors(vectors: list[list[float]]) -> list[float]:
        """Element-wise mean of a list of vectors."""
        n = len(vectors)
        dim = len(vectors[0])
        result = [0.0] * dim
        for vec in vectors:
            for d in range(dim):
                result[d] += vec[d]
        return [v / n for v in result]

    def _normalize_vec(self, vec: list[float]) -> list[float]:
        """L2-normalize a vector."""
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
