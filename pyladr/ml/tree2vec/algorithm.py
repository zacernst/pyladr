"""Tree2Vec: Unsupervised structural embeddings for logical formulas.

Top-level algorithm that orchestrates tree walk generation and skip-gram
training to produce embeddings for terms, literals, and clauses.

The algorithm works in three phases:
1. Generate tree walks from formula structures (Term/Clause trees)
2. Train skip-gram embeddings on the walk token sequences
3. Compose token-level embeddings into term/clause embeddings

For the vampire.in domain (P, i, n, variables), the constrained vocabulary
makes training especially effective - only ~6-8 unique tokens to learn.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from pyladr.core.clause import Clause
from pyladr.core.term import Term
from pyladr.ml.tree2vec.skipgram import SkipGramConfig, SkipGramTrainer
from pyladr.ml.tree2vec.walks import TreeWalker, WalkConfig, WalkType, _node_token


@dataclass(frozen=True, slots=True)
class Tree2VecConfig:
    """Top-level Tree2Vec configuration.

    Attributes:
        walk_config: Tree walk generation settings.
        skipgram_config: Skip-gram training settings.
        composition: How to compose token embeddings into term embeddings.
            "mean" - Average all token embeddings in the tree.
            "weighted_depth" - Weight tokens by inverse depth.
            "root_concat" - Concatenate root embedding with mean of rest.
        normalize: Whether to L2-normalize final embeddings.
    """

    walk_config: WalkConfig = WalkConfig()
    skipgram_config: SkipGramConfig = SkipGramConfig()
    composition: str = "weighted_depth"
    normalize: bool = True


class Tree2Vec:
    """Unsupervised embedding generator for logical formula trees.

    Usage:
        tree2vec = Tree2Vec(config)
        stats = tree2vec.train(clauses)
        embedding = tree2vec.embed_term(term)
        clause_emb = tree2vec.embed_clause(clause)
    """

    def __init__(self, config: Tree2VecConfig | None = None) -> None:
        self.config = config or Tree2VecConfig()
        self._walker = TreeWalker(self.config.walk_config)
        self._trainer = SkipGramTrainer(self.config.skipgram_config)
        self._trained: bool = False

    @property
    def embedding_dim(self) -> int:
        """Effective embedding dimension after composition."""
        base = self.config.skipgram_config.embedding_dim
        if self.config.composition == "root_concat":
            return base * 2
        return base

    @property
    def trained(self) -> bool:
        return self._trained

    @property
    def vocab_size(self) -> int:
        return self._trainer.vocab_size

    # ── Training ───────────────────────────────────────────────────────

    def train(self, clauses: Sequence[Clause], progress_fn=None) -> dict[str, float]:
        """Train Tree2Vec embeddings from a set of clauses.

        Generates tree walks from all clauses, then trains skip-gram
        embeddings on the resulting token sequences.

        Args:
            clauses: Training clauses to learn embeddings from.
            progress_fn: Optional callable invoked after each epoch with
                (epoch, num_epochs, epoch_loss, lr). Passed through to
                the skip-gram trainer unchanged.

        Returns:
            Training statistics from the skip-gram trainer.
        """
        walks = self._walker.walks_from_clauses(clauses)
        stats = self._trainer.train(walks, progress_fn=progress_fn)
        self._trained = True
        return stats

    def train_from_terms(self, terms: Sequence[Term]) -> dict[str, float]:
        """Train directly from terms (without clause wrapper).

        Useful for training on raw term trees without clause structure.
        """
        walks: list[list[str]] = []
        for term in terms:
            walks.extend(self._walker.walks_from_term(term))
        stats = self._trainer.train(walks)
        self._trained = True
        return stats

    # ── Online update ─────────────────────────────────────────────────

    def update_online(
        self,
        clauses: Sequence[Clause],
        learning_rate: float | None = None,
    ) -> dict[str, float | int]:
        """Perform online update from recently kept clauses.

        Generates tree walks from the clauses and runs a mini-batch SGD
        update on the existing skip-gram embeddings. Does NOT extend the
        vocabulary — out-of-vocabulary tokens are silently skipped.

        No-op if the model has not been trained yet.

        Args:
            clauses: Recently kept clauses to learn from.
            learning_rate: Learning rate override. Defaults to the
                skip-gram trainer's min_learning_rate.

        Returns:
            Training statistics from the skip-gram online update.
        """
        if not self._trained:
            return {"pairs_trained": 0, "loss": 0.0, "oov_skipped": 0}

        walks = self._walker.walks_from_clauses(clauses)
        return self._trainer.update_online(walks, learning_rate)

    # ── Embedding generation ───────────────────────────────────────────

    def embed_term(self, term: Term) -> list[float] | None:
        """Generate an embedding for a single term.

        Composes token-level embeddings into a single term embedding
        using the configured composition strategy.

        Returns None if not trained or no tokens have embeddings.
        """
        if not self._trained:
            return None
        return self._compose_term(term)

    def embed_clause(self, clause: Clause) -> list[float] | None:
        """Generate an embedding for a clause.

        Averages the embeddings of all literal atoms in the clause.
        When skip_predicate_wrapper is enabled, composes from predicate
        arguments directly instead of the full atom. Falls back to normal
        behavior for propositional atoms (arity 0).

        Returns None if not trained or no literals have embeddings.
        """
        if not self._trained:
            return None

        skip_pred = self.config.walk_config.skip_predicate_wrapper
        lit_embeddings: list[list[float]] = []
        for lit in clause.literals:
            sign_scale = 1.0 if lit.sign else -1.0
            if skip_pred and lit.atom.arity > 0:
                # Compose from each predicate argument separately
                for arg in lit.atom.args:
                    emb = self._compose_term(arg)
                    if emb is not None:
                        lit_embeddings.append([sign_scale * v for v in emb])
            else:
                emb = self._compose_term(lit.atom)
                if emb is not None:
                    lit_embeddings.append([sign_scale * v for v in emb])

        if not lit_embeddings:
            return None

        return self._normalize(self._mean_vectors(lit_embeddings))

    def embed_clauses_batch(
        self, clauses: Sequence[Clause]
    ) -> list[list[float] | None]:
        """Batch embedding for multiple clauses."""
        return [self.embed_clause(c) for c in clauses]

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
        """Get the raw embedding for a specific token."""
        return self._trainer.get_embedding(token)

    def most_similar_tokens(
        self, token: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Find tokens most similar to the given token."""
        return self._trainer.most_similar(token, top_k)

    # ── Composition strategies ─────────────────────────────────────────

    def _compose_term(self, term: Term) -> list[float] | None:
        """Compose token embeddings into a term embedding."""
        if self.config.composition == "mean":
            return self._compose_mean(term)
        elif self.config.composition == "weighted_depth":
            return self._compose_weighted_depth(term)
        elif self.config.composition == "root_concat":
            return self._compose_root_concat(term)
        else:
            return self._compose_mean(term)

    def _compose_mean(self, term: Term) -> list[float] | None:
        """Average all token embeddings in the term tree."""
        inc_pos = self.config.walk_config.include_position
        inc_dep = self.config.walk_config.include_depth
        embeddings: list[list[float]] = []
        # Pre-order DFS matching subterms() order, tracking position/depth
        stack: list[tuple[Term, int, int]] = [(term, 0, 0)]
        while stack:
            node, depth, position = stack.pop()
            tok = _node_token(node, position, depth, inc_pos, inc_dep)
            emb = self._trainer.get_embedding(tok)
            if emb is not None:
                embeddings.append(emb)
            for i in range(node.arity - 1, -1, -1):
                stack.append((node.args[i], depth + 1, i))
        if not embeddings:
            return None
        return self._normalize(self._mean_vectors(embeddings))

    def _compose_weighted_depth(self, term: Term) -> list[float] | None:
        """Weight token embeddings by inverse depth (root has most weight)."""
        inc_pos = self.config.walk_config.include_position
        inc_dep = self.config.walk_config.include_depth
        dim = self.config.skipgram_config.embedding_dim
        result = [0.0] * dim
        total_weight = 0.0

        stack: list[tuple[Term, int, int]] = [(term, 0, 0)]
        while stack:
            node, depth, position = stack.pop()
            tok = _node_token(node, position, depth, inc_pos, inc_dep)
            emb = self._trainer.get_embedding(tok)
            if emb is not None:
                w = 1.0 / (1.0 + depth)
                total_weight += w
                for d in range(dim):
                    result[d] += w * emb[d]
            for i, arg in enumerate(node.args):
                stack.append((arg, depth + 1, i))

        if total_weight == 0:
            return None
        result = [v / total_weight for v in result]
        return self._normalize(result)

    def _compose_root_concat(self, term: Term) -> list[float] | None:
        """Concatenate root embedding with mean of all other embeddings."""
        inc_pos = self.config.walk_config.include_position
        inc_dep = self.config.walk_config.include_depth
        dim = self.config.skipgram_config.embedding_dim
        root_tok = _node_token(term, 0, 0, inc_pos, inc_dep)
        root_emb = self._trainer.get_embedding(root_tok)

        if root_emb is None:
            root_emb = [0.0] * dim

        # Collect non-root embeddings via pre-order DFS with position/depth
        other_embeddings: list[list[float]] = []
        stack: list[tuple[Term, int, int]] = []
        for i in range(term.arity - 1, -1, -1):
            stack.append((term.args[i], 1, i))
        while stack:
            node, depth, position = stack.pop()
            tok = _node_token(node, position, depth, inc_pos, inc_dep)
            emb = self._trainer.get_embedding(tok)
            if emb is not None:
                other_embeddings.append(emb)
            for i in range(node.arity - 1, -1, -1):
                stack.append((node.args[i], depth + 1, i))

        if other_embeddings:
            other_mean = self._mean_vectors(other_embeddings)
        else:
            other_mean = [0.0] * dim

        combined = root_emb + other_mean
        return self._normalize(combined)

    # ── Vector utilities ───────────────────────────────────────────────

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

    def _normalize(self, vec: list[float]) -> list[float]:
        """L2-normalize a vector if configured."""
        if not self.config.normalize:
            return vec
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    # ── Serialization ──────────────────────────────────────────────────

    SAVE_FORMAT_VERSION = 1

    def save(self, path: str | Path) -> None:
        """Serialize trained model to a JSON file.

        Raises RuntimeError if model is not yet trained.
        """
        if not self._trained:
            raise RuntimeError("Cannot save an untrained Tree2Vec model.")
        wc = self.config.walk_config
        sc = self.config.skipgram_config
        data = {
            "format_version": self.SAVE_FORMAT_VERSION,
            "config": {
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
                "skipgram_config": {
                    "embedding_dim": sc.embedding_dim,
                    "window_size": sc.window_size,
                    "num_negative_samples": sc.num_negative_samples,
                    "learning_rate": sc.learning_rate,
                    "min_learning_rate": sc.min_learning_rate,
                    "num_epochs": sc.num_epochs,
                    "subsample_threshold": sc.subsample_threshold,
                    "seed": sc.seed,
                    "online_vocab_extension": sc.online_vocab_extension,
                },
                "composition": self.config.composition,
                "normalize": self.config.normalize,
            },
            "trainer": {
                "token_to_id": self._trainer._token_to_id,
                "id_to_token": list(self._trainer._id_to_token),
                "input_embeddings": self._trainer._input_embeddings,
                "output_embeddings": self._trainer._output_embeddings,
                "token_counts": self._trainer._token_counts,
                "total_tokens": self._trainer._total_tokens,
            },
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tree2Vec":
        """Load a previously saved Tree2Vec model from a JSON file."""
        from pyladr.ml.tree2vec.walks import WalkConfig, WalkType
        from pyladr.ml.tree2vec.skipgram import SkipGramConfig

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format_version") != cls.SAVE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported Tree2Vec format version {data.get('format_version')!r}; "
                f"expected {cls.SAVE_FORMAT_VERSION}"
            )
        cfg = data["config"]
        wc_d = cfg["walk_config"]
        sc_d = cfg["skipgram_config"]
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
        skipgram_config = SkipGramConfig(
            embedding_dim=sc_d["embedding_dim"],
            window_size=sc_d["window_size"],
            num_negative_samples=sc_d["num_negative_samples"],
            learning_rate=sc_d["learning_rate"],
            min_learning_rate=sc_d["min_learning_rate"],
            num_epochs=sc_d["num_epochs"],
            subsample_threshold=sc_d["subsample_threshold"],
            seed=sc_d["seed"],
            online_vocab_extension=sc_d["online_vocab_extension"],
        )
        t2v_config = Tree2VecConfig(
            walk_config=walk_config,
            skipgram_config=skipgram_config,
            composition=cfg["composition"],
            normalize=cfg.get("normalize", True),
        )
        instance = cls(t2v_config)
        tr = data["trainer"]
        t = instance._trainer
        t._token_to_id = tr["token_to_id"]
        t._id_to_token = tr["id_to_token"]
        t._input_embeddings = tr["input_embeddings"]
        t._output_embeddings = tr["output_embeddings"]
        t._token_counts = tr["token_counts"]
        t._total_tokens = tr["total_tokens"]
        t._vocab_size = len(t._id_to_token)
        t._trained = True
        t._build_neg_sample_table()
        instance._trained = True
        return instance

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
