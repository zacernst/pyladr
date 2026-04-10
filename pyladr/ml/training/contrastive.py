"""Contrastive learning framework for proof pattern recognition.

Trains clause embedding models to capture productive inference patterns by
learning from successful and unsuccessful inference pairs during theorem
proving. The framework supports:

- Extraction of positive/negative pairs from proof derivation trees
- InfoNCE-based contrastive loss for embedding similarity optimization
- Data augmentation via variable renaming and literal permutation
- Evaluation metrics for embedding quality assessment
- Integration with TPTP problem library for training data
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

if TYPE_CHECKING:
    from pyladr.core.clause import Clause, Justification, JustType
    from pyladr.core.symbol import SymbolTable
    from pyladr.search.given_clause import Proof, SearchResult

logger = logging.getLogger(__name__)


# ── Protocols for model abstraction ────────────────────────────────────────


@runtime_checkable
class ClauseEncoder(Protocol):
    """Protocol for any model that encodes clauses into embedding vectors.

    The contrastive trainer works with any encoder satisfying this interface,
    decoupling training logic from the specific GNN architecture.
    """

    def encode_clauses(self, clauses: list[Clause]) -> torch.Tensor:
        """Encode a batch of clauses into embedding vectors.

        Args:
            clauses: List of PyLADR Clause objects.

        Returns:
            Tensor of shape (len(clauses), embedding_dim).
        """
        ...

    def parameters(self) -> ...:
        """Return model parameters for optimizer."""
        ...

    def train(self, mode: bool = True) -> ...:
        """Set training/eval mode."""
        ...

    def eval(self) -> ...:
        """Set eval mode."""
        ...


# ── Pair labeling ──────────────────────────────────────────────────────────


class PairLabel(IntEnum):
    """Labels for inference pairs used in contrastive learning."""

    PRODUCTIVE = 1       # Inference contributed to a proof
    UNPRODUCTIVE = 0     # Inference was generated but not used in proof
    SUBSUMPTION = 2      # One clause subsumes the other (structural similarity)


# ── Data structures ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class InferencePair:
    """A pair of clauses from an inference step with a productivity label.

    Attributes:
        parent1: First parent clause (e.g., given clause).
        parent2: Second parent clause (e.g., usable clause resolved against).
        child: The inferred clause (resolvent, paramodulant, etc.).
        label: Whether this inference was productive (used in proof).
        inference_type: Type of inference rule applied (e.g., BINARY_RES, PARA).
        proof_depth: Distance from the child to the empty clause in the proof
            tree, or -1 if not part of a proof.
    """

    parent1: Clause
    parent2: Clause | None
    child: Clause
    label: PairLabel
    inference_type: int = 0
    proof_depth: int = -1


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ContrastiveConfig:
    """Configuration for contrastive training.

    Attributes:
        temperature: Temperature for InfoNCE softmax scaling. Lower values
            produce sharper distributions (harder negatives matter more).
        margin: Margin for triplet-style loss variant.
        embedding_dim: Expected dimensionality of clause embeddings.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization strength.
        max_negatives: Maximum negative examples per positive pair.
        hard_negative_ratio: Fraction of negatives that are "hard" (high
            similarity to the anchor but unproductive).
        augmentation_prob: Probability of applying data augmentation to
            each training example.
        depth_weight: Whether to weight productive pairs by their proof
            depth (closer to empty clause = higher weight).
        warmup_steps: Number of linear warmup steps for learning rate.
        max_steps: Maximum total training steps (0 = unlimited).
    """

    temperature: float = 0.07
    margin: float = 0.5
    embedding_dim: int = 512
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_negatives: int = 15
    hard_negative_ratio: float = 0.3
    augmentation_prob: float = 0.2
    depth_weight: bool = True
    warmup_steps: int = 100
    max_steps: int = 0


_DEFAULT_CONFIG = ContrastiveConfig()


# ── Proof pattern extraction ──────────────────────────────────────────────


class ProofPatternExtractor:
    """Extracts training pairs from completed proof search results.

    Given a SearchResult containing proofs, traces the derivation tree to
    identify which inferences were productive (used in a proof) vs.
    unproductive (generated but discarded or unused).
    """

    def __init__(self, all_clauses: dict[int, Clause] | None = None):
        self._all_clauses = all_clauses or {}

    def extract_pairs(
        self,
        result: SearchResult,
        all_clauses: dict[int, Clause] | None = None,
    ) -> list[InferencePair]:
        """Extract inference pairs from a completed search.

        Args:
            result: The SearchResult from a given-clause search.
            all_clauses: Map of clause_id → Clause for the entire search.
                If not provided, uses the instance's clause map.

        Returns:
            List of InferencePair with productive/unproductive labels.
        """
        clauses = all_clauses or self._all_clauses
        if not clauses:
            return []

        pairs: list[InferencePair] = []

        for proof in result.proofs:
            proof_ids = {c.id for c in proof.clauses}
            depth_map = self._compute_proof_depths(proof, clauses)

            # Extract productive pairs from proof clauses
            for clause in proof.clauses:
                pair = self._extract_pair_from_justification(
                    clause, clauses, proof_ids, depth_map,
                )
                if pair is not None:
                    pairs.append(pair)

            # Sample unproductive pairs from non-proof clauses
            non_proof = [
                c for cid, c in clauses.items()
                if cid not in proof_ids and c.justification
            ]
            for clause in non_proof:
                pair = self._extract_pair_from_justification(
                    clause, clauses, proof_ids, depth_map,
                )
                if pair is not None:
                    pairs.append(pair)

        return pairs

    def _compute_proof_depths(
        self,
        proof: Proof,
        clauses: dict[int, Clause],
    ) -> dict[int, int]:
        """Compute depth of each clause in the proof tree.

        Depth is the shortest distance from the clause to the empty clause.
        Input/goal clauses are at the maximum depth.
        """
        from pyladr.core.clause import JustType

        depth: dict[int, int] = {}
        # BFS from empty clause backwards
        queue: list[tuple[int, int]] = [(proof.empty_clause.id, 0)]
        visited: set[int] = set()

        while queue:
            cid, d = queue.pop(0)
            if cid in visited:
                continue
            visited.add(cid)
            depth[cid] = d

            if cid not in clauses:
                continue
            c = clauses[cid]
            for just in c.justification:
                parent_ids = self._get_parent_ids(just)
                for pid in parent_ids:
                    if pid not in visited:
                        queue.append((pid, d + 1))

        return depth

    @staticmethod
    def _get_parent_ids(just: Justification) -> list[int]:
        """Extract parent clause IDs from a justification step."""
        ids: list[int] = []
        if just.clause_ids:
            ids.extend(just.clause_ids)
        if just.clause_id > 0:
            ids.append(just.clause_id)
        if just.para is not None:
            ids.append(just.para.from_id)
            ids.append(just.para.into_id)
        return ids

    def _extract_pair_from_justification(
        self,
        clause: Clause,
        clauses: dict[int, Clause],
        proof_ids: set[int],
        depth_map: dict[int, int],
    ) -> InferencePair | None:
        """Create an InferencePair from a clause's primary justification."""
        from pyladr.core.clause import JustType

        if not clause.justification:
            return None

        primary = clause.justification[0]
        parent_ids = self._get_parent_ids(primary)

        # Need at least one parent to form a pair
        if not parent_ids:
            return None

        parent1 = clauses.get(parent_ids[0])
        parent2 = clauses.get(parent_ids[1]) if len(parent_ids) > 1 else None

        if parent1 is None:
            return None

        is_productive = clause.id in proof_ids
        label = PairLabel.PRODUCTIVE if is_productive else PairLabel.UNPRODUCTIVE
        depth = depth_map.get(clause.id, -1)

        return InferencePair(
            parent1=parent1,
            parent2=parent2,
            child=clause,
            label=label,
            inference_type=int(primary.just_type),
            proof_depth=depth,
        )


# ── Data augmentation ─────────────────────────────────────────────────────


def augment_clause(clause: Clause, prob: float = 0.2) -> Clause:
    """Apply random augmentation to a clause for training robustness.

    Augmentation strategies (applied with probability `prob`):
    1. Literal permutation: shuffle the order of literals
    2. Variable renaming: consistently rename all variables

    These preserve logical equivalence while varying the syntactic form,
    teaching the model to focus on semantic structure.

    Args:
        clause: The clause to augment.
        prob: Probability of applying each augmentation.

    Returns:
        A new Clause (or the original if no augmentation applied).
    """
    from pyladr.core.clause import Clause as ClauseClass, Literal
    from pyladr.core.term import Term

    if random.random() > prob:
        return clause

    literals = list(clause.literals)

    # Strategy 1: Literal permutation
    if len(literals) > 1 and random.random() < 0.5:
        random.shuffle(literals)

    # Strategy 2: Variable renaming (consistent renumbering)
    if random.random() < 0.5:
        var_map: dict[int, int] = {}
        counter = 0

        def _remap_vars(term: Term) -> Term:
            nonlocal counter
            if term.is_variable:
                if term.varnum not in var_map:
                    var_map[term.varnum] = counter
                    counter += 1
                new_varnum = var_map[term.varnum]
                return Term(
                    private_symbol=new_varnum,
                    arity=0,
                    args=(),
                    container=term.container,
                    term_id=term.term_id,
                )
            if term.arity == 0:
                return term
            new_args = tuple(_remap_vars(a) for a in term.args)
            return Term(
                private_symbol=term.private_symbol,
                arity=term.arity,
                args=new_args,
                container=term.container,
                term_id=term.term_id,
            )

        # Randomize initial variable numbering
        counter = random.randint(0, 50)
        literals = [
            Literal(sign=lit.sign, atom=_remap_vars(lit.atom))
            for lit in literals
        ]

    return ClauseClass(
        literals=tuple(literals),
        id=clause.id,
        weight=clause.weight,
        justification=clause.justification,
    )


# ── Training dataset ──────────────────────────────────────────────────────


class TrainingDataset(Dataset):
    """PyTorch Dataset for contrastive training on inference pairs.

    Each item yields an (anchor, positive, negative) triplet of clauses:
    - anchor: a parent clause from a productive inference
    - positive: the other parent or child from the same productive inference
    - negative: a clause from an unproductive inference
    """

    def __init__(
        self,
        pairs: list[InferencePair],
        config: ContrastiveConfig | None = None,
    ):
        self._config = config or _DEFAULT_CONFIG
        self._productive = [p for p in pairs if p.label == PairLabel.PRODUCTIVE]
        self._unproductive = [p for p in pairs if p.label == PairLabel.UNPRODUCTIVE]

        if not self._productive:
            logger.warning("No productive pairs in training data")
        if not self._unproductive:
            logger.warning("No unproductive pairs in training data")

    def __len__(self) -> int:
        return len(self._productive)

    def __getitem__(self, idx: int) -> dict:
        """Return a training triplet.

        Returns:
            Dict with keys:
                'anchor': Clause (parent1 of productive inference)
                'positive': Clause (parent2 or child of productive inference)
                'negatives': list[Clause] (from unproductive inferences)
                'depth_weight': float (proof depth weighting)
        """
        pos_pair = self._productive[idx]

        # Anchor is always parent1
        anchor = pos_pair.parent1

        # Positive is parent2 if available, otherwise child
        positive = pos_pair.parent2 if pos_pair.parent2 is not None else pos_pair.child

        # Apply augmentation
        aug_prob = self._config.augmentation_prob
        anchor = augment_clause(anchor, aug_prob)
        positive = augment_clause(positive, aug_prob)

        # Sample negatives
        n_neg = min(self._config.max_negatives, len(self._unproductive))
        negatives = self._sample_negatives(anchor, n_neg)

        # Depth-based weighting: closer to empty clause = higher weight
        if self._config.depth_weight and pos_pair.proof_depth >= 0:
            depth_w = 1.0 / (1.0 + pos_pair.proof_depth)
        else:
            depth_w = 1.0

        return {
            "anchor": anchor,
            "positive": positive,
            "negatives": negatives,
            "depth_weight": depth_w,
        }

    def _sample_negatives(
        self, anchor: Clause, n: int,
    ) -> list[Clause]:
        """Sample negative examples, mixing random and hard negatives."""
        if not self._unproductive or n == 0:
            return []

        n_hard = max(1, int(n * self._config.hard_negative_ratio))
        n_random = n - n_hard

        # Random negatives
        random_negs = random.sample(
            self._unproductive,
            min(n_random, len(self._unproductive)),
        )
        random_clauses = [
            p.child for p in random_negs
        ]

        # Hard negatives: prefer unproductive inferences from the same
        # parent (they share structural similarity but weren't useful)
        anchor_id = anchor.id
        hard_candidates = [
            p for p in self._unproductive
            if p.parent1.id == anchor_id or (p.parent2 and p.parent2.id == anchor_id)
        ]
        if not hard_candidates:
            hard_candidates = self._unproductive

        hard_negs = random.sample(
            hard_candidates,
            min(n_hard, len(hard_candidates)),
        )
        hard_clauses = [p.child for p in hard_negs]

        return hard_clauses + random_clauses


# ── Contrastive loss functions ─────────────────────────────────────────────


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for clause embedding training.

    Given anchor, positive, and negative embeddings, maximizes similarity
    between anchor-positive pairs while minimizing similarity to negatives.
    Uses the NT-Xent (Normalized Temperature-scaled Cross Entropy) formulation.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            anchor: Anchor embeddings, shape (batch, dim).
            positive: Positive embeddings, shape (batch, dim).
            negatives: Negative embeddings, shape (batch, n_neg, dim).
            weights: Optional per-example weights, shape (batch,).
                Used for depth-based weighting.

        Returns:
            Scalar loss tensor.
        """
        # L2 normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        # Positive similarity: (batch,)
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature

        # Negative similarity: (batch, n_neg)
        neg_sim = torch.bmm(
            negatives, anchor.unsqueeze(-1)
        ).squeeze(-1) / self.temperature

        # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # Numerically stable: logits = [pos_sim, neg_sim] then cross-entropy
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels, reduction="none")

        if weights is not None:
            loss = loss * weights
            return loss.sum() / weights.sum().clamp(min=1e-8)

        return loss.mean()


class TripletMarginContrastiveLoss(nn.Module):
    """Margin-based triplet loss variant for clause embeddings.

    An alternative to InfoNCE that directly enforces a margin between
    positive and negative pair distances. Useful when the number of
    negatives per anchor is small.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute triplet margin loss.

        Args:
            anchor: shape (batch, dim)
            positive: shape (batch, dim)
            negatives: shape (batch, n_neg, dim)
            weights: optional per-example weights, shape (batch,)

        Returns:
            Scalar loss.
        """
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        # Positive distance
        pos_dist = 1.0 - torch.sum(anchor * positive, dim=-1)  # (batch,)

        # Negative distances — use the closest (hardest) negative
        neg_sims = torch.bmm(
            negatives, anchor.unsqueeze(-1)
        ).squeeze(-1)  # (batch, n_neg)
        neg_dist = 1.0 - neg_sims.max(dim=-1).values  # (batch,)

        loss = F.relu(pos_dist - neg_dist + self.margin)

        if weights is not None:
            loss = loss * weights
            return loss.sum() / weights.sum().clamp(min=1e-8)

        return loss.mean()


# ── Evaluation metrics ────────────────────────────────────────────────────


@dataclass(slots=True)
class EvalMetrics:
    """Evaluation metrics for embedding quality."""

    mean_positive_similarity: float = 0.0
    mean_negative_similarity: float = 0.0
    similarity_gap: float = 0.0
    retrieval_accuracy_at_1: float = 0.0
    retrieval_accuracy_at_5: float = 0.0
    mean_reciprocal_rank: float = 0.0
    num_pairs: int = 0


class EmbeddingEvaluator:
    """Evaluates the quality of learned clause embeddings.

    Metrics assess whether productive inference pairs have higher embedding
    similarity than unproductive ones, which is the core training objective.
    """

    def __init__(self, encoder: ClauseEncoder):
        self._encoder = encoder

    @torch.no_grad()
    def evaluate(self, pairs: list[InferencePair]) -> EvalMetrics:
        """Compute embedding quality metrics on a set of inference pairs.

        Args:
            pairs: List of labeled inference pairs.

        Returns:
            EvalMetrics with similarity statistics and retrieval accuracy.
        """
        self._encoder.eval()

        productive = [p for p in pairs if p.label == PairLabel.PRODUCTIVE]
        unproductive = [p for p in pairs if p.label == PairLabel.UNPRODUCTIVE]

        if not productive:
            return EvalMetrics()

        # Compute positive similarities
        pos_sims = self._compute_pair_similarities(productive)

        # Compute negative similarities
        neg_sims = self._compute_pair_similarities(unproductive) if unproductive else []

        # Basic statistics
        mean_pos = sum(pos_sims) / len(pos_sims) if pos_sims else 0.0
        mean_neg = sum(neg_sims) / len(neg_sims) if neg_sims else 0.0
        gap = mean_pos - mean_neg

        # Retrieval accuracy: for each productive pair, rank it among
        # a mix of productive + unproductive pairs
        acc_at_1, acc_at_5, mrr = self._compute_retrieval_metrics(
            productive, unproductive,
        )

        return EvalMetrics(
            mean_positive_similarity=mean_pos,
            mean_negative_similarity=mean_neg,
            similarity_gap=gap,
            retrieval_accuracy_at_1=acc_at_1,
            retrieval_accuracy_at_5=acc_at_5,
            mean_reciprocal_rank=mrr,
            num_pairs=len(pairs),
        )

    def _compute_pair_similarities(
        self, pairs: list[InferencePair],
    ) -> list[float]:
        """Compute cosine similarity for each pair's parent clauses."""
        if not pairs:
            return []

        parents1 = [p.parent1 for p in pairs]
        # Use parent2 if available, otherwise child
        parents2 = [
            p.parent2 if p.parent2 is not None else p.child
            for p in pairs
        ]

        emb1 = self._encoder.encode_clauses(parents1)
        emb2 = self._encoder.encode_clauses(parents2)

        emb1 = F.normalize(emb1, dim=-1)
        emb2 = F.normalize(emb2, dim=-1)

        sims = torch.sum(emb1 * emb2, dim=-1)
        return sims.cpu().tolist()

    def _compute_retrieval_metrics(
        self,
        productive: list[InferencePair],
        unproductive: list[InferencePair],
    ) -> tuple[float, float, float]:
        """Compute retrieval accuracy and MRR.

        For each productive pair's anchor (parent1), we rank the correct
        positive (parent2/child) against a pool of unproductive children.
        """
        if not productive or not unproductive:
            return 0.0, 0.0, 0.0

        # Build candidate pool from unproductive children
        pool_clauses = [p.child for p in unproductive[:100]]
        if not pool_clauses:
            return 0.0, 0.0, 0.0

        pool_embs = self._encoder.encode_clauses(pool_clauses)
        pool_embs = F.normalize(pool_embs, dim=-1)

        hits_at_1 = 0
        hits_at_5 = 0
        reciprocal_ranks: list[float] = []

        for pair in productive[:100]:
            anchor_emb = self._encoder.encode_clauses([pair.parent1])
            anchor_emb = F.normalize(anchor_emb, dim=-1)

            target = pair.parent2 if pair.parent2 is not None else pair.child
            target_emb = self._encoder.encode_clauses([target])
            target_emb = F.normalize(target_emb, dim=-1)

            # Similarity to target
            target_sim = torch.sum(anchor_emb * target_emb).item()

            # Similarities to pool
            pool_sims = torch.mv(pool_embs, anchor_emb.squeeze(0))

            # Rank: how many pool items have higher similarity than target
            rank = (pool_sims > target_sim).sum().item() + 1

            if rank <= 1:
                hits_at_1 += 1
            if rank <= 5:
                hits_at_5 += 1
            reciprocal_ranks.append(1.0 / rank)

        n = len(reciprocal_ranks)
        return (
            hits_at_1 / n if n else 0.0,
            hits_at_5 / n if n else 0.0,
            sum(reciprocal_ranks) / n if n else 0.0,
        )


# ── Main trainer ──────────────────────────────────────────────────────────


class ContrastiveTrainer:
    """Training pipeline for contrastive learning on clause embeddings.

    Manages the training loop, optimizer, learning rate schedule, and
    validation. Works with any model implementing the ClauseEncoder protocol.

    Usage:
        encoder = HeterogeneousClauseGNN(...)
        trainer = ContrastiveTrainer(encoder)
        trainer.train(training_pairs, validation_pairs)
    """

    def __init__(
        self,
        encoder: ClauseEncoder,
        config: ContrastiveConfig | None = None,
        loss_fn: nn.Module | None = None,
        device: torch.device | None = None,
    ):
        self._encoder = encoder
        self._config = config or _DEFAULT_CONFIG
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._loss_fn = loss_fn or ContrastiveLoss(
            temperature=self._config.temperature,
        )
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._step = 0
        self._best_gap = float("-inf")
        self._best_state: dict | None = None

    def train(
        self,
        train_pairs: list[InferencePair],
        val_pairs: list[InferencePair] | None = None,
        epochs: int = 10,
        eval_every: int = 100,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            train_pairs: Labeled inference pairs for training.
            val_pairs: Optional pairs for validation. If not provided,
                10% of train_pairs is held out.
            epochs: Number of training epochs.
            eval_every: Evaluate on validation set every N steps.

        Returns:
            Training history dict with loss and metric curves.
        """
        if val_pairs is None:
            split = max(1, len(train_pairs) // 10)
            random.shuffle(train_pairs)
            val_pairs = train_pairs[:split]
            train_pairs = train_pairs[split:]

        dataset = TrainingDataset(train_pairs, self._config)
        evaluator = EmbeddingEvaluator(self._encoder)

        self._optimizer = torch.optim.AdamW(
            self._encoder.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

        total_steps = epochs * max(1, len(dataset) // self._config.batch_size)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer, T_max=total_steps,
        )

        history: dict[str, list[float]] = defaultdict(list)
        self._step = 0
        self._best_gap = float("-inf")

        for epoch in range(epochs):
            epoch_loss = self._train_epoch(dataset, history)
            logger.info("Epoch %d/%d, loss=%.4f", epoch + 1, epochs, epoch_loss)

            # Validation
            if val_pairs:
                metrics = evaluator.evaluate(val_pairs)
                history["val_gap"].append(metrics.similarity_gap)
                history["val_mrr"].append(metrics.mean_reciprocal_rank)
                logger.info(
                    "  val: gap=%.4f, mrr=%.4f, acc@1=%.4f",
                    metrics.similarity_gap,
                    metrics.mean_reciprocal_rank,
                    metrics.retrieval_accuracy_at_1,
                )

                # Track best model
                if metrics.similarity_gap > self._best_gap:
                    self._best_gap = metrics.similarity_gap
                    self._best_state = {
                        k: v.clone()
                        for k, v in self._encoder.state_dict().items()
                    }

            if self._config.max_steps > 0 and self._step >= self._config.max_steps:
                break

        # Restore best model
        if self._best_state is not None:
            self._encoder.load_state_dict(self._best_state)

        return dict(history)

    def _train_epoch(
        self,
        dataset: TrainingDataset,
        history: dict[str, list[float]],
    ) -> float:
        """Train for one epoch."""
        self._encoder.train()

        indices = list(range(len(dataset)))
        random.shuffle(indices)

        total_loss = 0.0
        n_batches = 0
        batch_size = self._config.batch_size

        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start : batch_start + batch_size]
            batch_items = [dataset[i] for i in batch_indices]

            loss = self._train_step(batch_items)
            if loss is not None:
                total_loss += loss
                n_batches += 1
                history["train_loss"].append(loss)

            self._step += 1
            if self._config.max_steps > 0 and self._step >= self._config.max_steps:
                break

        return total_loss / max(1, n_batches)

    def _train_step(self, batch_items: list[dict]) -> float | None:
        """Execute a single training step on a batch.

        Args:
            batch_items: List of dicts from TrainingDataset.__getitem__.

        Returns:
            Loss value, or None if the batch couldn't be processed.
        """
        anchors = [item["anchor"] for item in batch_items]
        positives = [item["positive"] for item in batch_items]
        weights = torch.tensor(
            [item["depth_weight"] for item in batch_items],
            dtype=torch.float32,
            device=self._device,
        )

        # Gather negatives — pad to uniform count
        max_neg = max(len(item["negatives"]) for item in batch_items) if batch_items else 0
        if max_neg == 0:
            return None

        all_neg_clauses: list[Clause] = []
        neg_counts: list[int] = []
        for item in batch_items:
            negs = item["negatives"]
            neg_counts.append(len(negs))
            all_neg_clauses.extend(negs)

        if not all_neg_clauses:
            return None

        # Encode all clauses
        anchor_emb = self._encoder.encode_clauses(anchors)
        pos_emb = self._encoder.encode_clauses(positives)
        neg_emb = self._encoder.encode_clauses(all_neg_clauses)

        # Reshape negatives to (batch, max_neg, dim)
        batch_neg = torch.zeros(
            len(batch_items), max_neg, anchor_emb.size(-1),
            device=self._device,
        )
        offset = 0
        for i, count in enumerate(neg_counts):
            if count > 0:
                batch_neg[i, :count] = neg_emb[offset : offset + count]
            offset += count

        # Warmup learning rate
        if self._step < self._config.warmup_steps and self._optimizer is not None:
            warmup_factor = (self._step + 1) / self._config.warmup_steps
            for pg in self._optimizer.param_groups:
                pg["lr"] = self._config.learning_rate * warmup_factor

        # Compute loss and update
        loss = self._loss_fn(anchor_emb, pos_emb, batch_neg, weights)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._encoder.parameters(), 1.0)
        self._optimizer.step()

        if self._scheduler is not None and self._step >= self._config.warmup_steps:
            self._scheduler.step()

        return loss.item()

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        torch.save({
            "step": self._step,
            "encoder_state": self._encoder.state_dict(),
            "optimizer_state": self._optimizer.state_dict() if self._optimizer else None,
            "best_gap": self._best_gap,
            "config": self._config,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        ckpt = torch.load(path, weights_only=False)
        self._encoder.load_state_dict(ckpt["encoder_state"])
        self._step = ckpt["step"]
        self._best_gap = ckpt.get("best_gap", float("-inf"))
        if self._optimizer and ckpt.get("optimizer_state"):
            self._optimizer.load_state_dict(ckpt["optimizer_state"])
