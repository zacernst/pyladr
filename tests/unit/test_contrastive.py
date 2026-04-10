"""Tests for contrastive learning framework."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.term import Term
from pyladr.ml.training.contrastive import (
    ContrastiveConfig,
    ContrastiveLoss,
    EmbeddingEvaluator,
    EvalMetrics,
    InferencePair,
    PairLabel,
    ProofPatternExtractor,
    TrainingDataset,
    TripletMarginContrastiveLoss,
    augment_clause,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_term(symnum: int, args: tuple[Term, ...] = ()) -> Term:
    """Create a constant or complex term."""
    return Term(private_symbol=-symnum, arity=len(args), args=args)


def _make_var(varnum: int) -> Term:
    """Create a variable term."""
    return Term(private_symbol=varnum, arity=0, args=())


def _make_clause(
    lits: list[tuple[bool, Term]], clause_id: int = 0,
    justification: tuple[Justification, ...] = (),
) -> Clause:
    """Create a clause from (sign, atom) pairs."""
    literals = tuple(Literal(sign=s, atom=a) for s, a in lits)
    c = Clause(literals=literals, justification=justification)
    c.id = clause_id
    c.weight = float(len(literals))
    return c


def _make_proof_chain() -> tuple[dict[int, Clause], set[int]]:
    """Create a simple proof chain for testing pattern extraction.

    Chain: c1 + c2 -> c3, c3 + c4 -> c5 (empty)
    Non-proof clauses: c6, c7
    """
    f = _make_term(1)  # constant f
    g = _make_term(2)  # constant g
    x = _make_var(0)

    c1 = _make_clause([(True, f)], clause_id=1, justification=(
        Justification(just_type=JustType.INPUT),
    ))
    c2 = _make_clause([(False, f)], clause_id=2, justification=(
        Justification(just_type=JustType.INPUT),
    ))
    c3 = _make_clause([(True, g)], clause_id=3, justification=(
        Justification(just_type=JustType.BINARY_RES, clause_ids=(1, 2)),
    ))
    c4 = _make_clause([(False, g)], clause_id=4, justification=(
        Justification(just_type=JustType.INPUT),
    ))
    c5 = _make_clause([], clause_id=5, justification=(
        Justification(just_type=JustType.BINARY_RES, clause_ids=(3, 4)),
    ))

    # Non-proof clauses
    h = _make_term(3)
    c6 = _make_clause([(True, h)], clause_id=6, justification=(
        Justification(just_type=JustType.BINARY_RES, clause_ids=(1, 4)),
    ))
    c7 = _make_clause([(False, h)], clause_id=7, justification=(
        Justification(just_type=JustType.BINARY_RES, clause_ids=(2, 4)),
    ))

    clauses = {c.id: c for c in [c1, c2, c3, c4, c5, c6, c7]}
    proof_ids = {1, 2, 3, 4, 5}
    return clauses, proof_ids


class MockEncoder:
    """Mock encoder for testing that assigns random embeddings."""

    def __init__(self, dim: int = 64):
        self._dim = dim
        self._embedding_map: dict[int, torch.Tensor] = {}
        self._linear = torch.nn.Linear(dim, dim)

    def encode_clauses(self, clauses: list[Clause]) -> torch.Tensor:
        embeddings = []
        for c in clauses:
            if c.id not in self._embedding_map:
                self._embedding_map[c.id] = torch.randn(self._dim)
            embeddings.append(self._embedding_map[c.id])
        return torch.stack(embeddings)

    def parameters(self):
        return self._linear.parameters()

    def named_parameters(self):
        return self._linear.named_parameters()

    def state_dict(self):
        return self._linear.state_dict()

    def load_state_dict(self, state):
        self._linear.load_state_dict(state)

    def train(self, mode=True):
        self._linear.train(mode)

    def eval(self):
        self._linear.eval()


# ── InferencePair tests ────────────────────────────────────────────────────


class TestInferencePair:
    def test_creation(self):
        c1 = _make_clause([(True, _make_term(1))], clause_id=1)
        c2 = _make_clause([(False, _make_term(1))], clause_id=2)
        c3 = _make_clause([], clause_id=3)

        pair = InferencePair(
            parent1=c1, parent2=c2, child=c3,
            label=PairLabel.PRODUCTIVE,
            inference_type=int(JustType.BINARY_RES),
            proof_depth=1,
        )

        assert pair.label == PairLabel.PRODUCTIVE
        assert pair.parent1.id == 1
        assert pair.parent2.id == 2
        assert pair.child.id == 3
        assert pair.proof_depth == 1

    def test_unproductive_pair(self):
        c1 = _make_clause([(True, _make_term(1))], clause_id=1)
        c2 = _make_clause([(True, _make_term(2))], clause_id=6)

        pair = InferencePair(
            parent1=c1, parent2=None, child=c2,
            label=PairLabel.UNPRODUCTIVE,
        )

        assert pair.label == PairLabel.UNPRODUCTIVE
        assert pair.parent2 is None
        assert pair.proof_depth == -1


# ── ProofPatternExtractor tests ───────────────────────────────────────────


class TestProofPatternExtractor:
    def test_extract_pairs_from_proof(self):
        from pyladr.search.given_clause import Proof, SearchResult, ExitCode
        from pyladr.search.statistics import SearchStatistics

        clauses, proof_ids = _make_proof_chain()
        proof_clauses = tuple(clauses[i] for i in sorted(proof_ids))
        empty = clauses[5]

        proof = Proof(empty_clause=empty, clauses=proof_clauses)
        result = SearchResult(
            exit_code=ExitCode.MAX_PROOFS_EXIT,
            proofs=(proof,),
            stats=SearchStatistics(),
        )

        extractor = ProofPatternExtractor()
        pairs = extractor.extract_pairs(result, clauses)

        # Should have pairs from both proof and non-proof clauses
        assert len(pairs) > 0

        productive = [p for p in pairs if p.label == PairLabel.PRODUCTIVE]
        unproductive = [p for p in pairs if p.label == PairLabel.UNPRODUCTIVE]

        # Clauses 3 and 5 have binary_res justifications in the proof
        assert len(productive) >= 2

        # Clauses 6 and 7 are not in the proof
        assert len(unproductive) >= 2

    def test_proof_depths(self):
        from pyladr.search.given_clause import Proof, SearchResult, ExitCode
        from pyladr.search.statistics import SearchStatistics

        clauses, proof_ids = _make_proof_chain()
        proof_clauses = tuple(clauses[i] for i in sorted(proof_ids))
        empty = clauses[5]

        proof = Proof(empty_clause=empty, clauses=proof_clauses)
        result = SearchResult(
            exit_code=ExitCode.MAX_PROOFS_EXIT,
            proofs=(proof,),
            stats=SearchStatistics(),
        )

        extractor = ProofPatternExtractor()
        pairs = extractor.extract_pairs(result, clauses)

        # Find the pair for clause 5 (empty clause, depth 0)
        c5_pairs = [p for p in pairs if p.child.id == 5]
        if c5_pairs:
            assert c5_pairs[0].proof_depth == 0

        # Find the pair for clause 3 (depth 1 from empty)
        c3_pairs = [p for p in pairs if p.child.id == 3]
        if c3_pairs:
            assert c3_pairs[0].proof_depth == 1

    def test_empty_result(self):
        from pyladr.search.given_clause import SearchResult, ExitCode
        from pyladr.search.statistics import SearchStatistics

        result = SearchResult(
            exit_code=ExitCode.SOS_EMPTY_EXIT,
            proofs=(),
            stats=SearchStatistics(),
        )

        extractor = ProofPatternExtractor()
        pairs = extractor.extract_pairs(result)
        assert pairs == []


# ── Data augmentation tests ────────────────────────────────────────────────


class TestAugmentation:
    def test_augment_preserves_literals(self):
        c = _make_clause([
            (True, _make_term(1)),
            (False, _make_term(2)),
        ], clause_id=1)

        # With prob=1.0, augmentation always applies
        aug = augment_clause(c, prob=1.0)

        # Same number of literals
        assert len(aug.literals) == len(c.literals)
        # Same ID preserved
        assert aug.id == c.id

    def test_augment_prob_zero_no_change(self):
        c = _make_clause([(True, _make_term(1))], clause_id=1)
        aug = augment_clause(c, prob=0.0)
        assert aug is c  # exact same object

    def test_augment_with_variables(self):
        """Augmentation should handle clauses with variables."""
        x = _make_var(0)
        y = _make_var(1)
        f_xy = _make_term(1, args=(x, y))

        c = _make_clause([(True, f_xy)], clause_id=1)
        aug = augment_clause(c, prob=1.0)

        assert len(aug.literals) == 1


# ── TrainingDataset tests ──────────────────────────────────────────────────


class TestTrainingDataset:
    def _make_pairs(self) -> list[InferencePair]:
        """Create a mix of productive and unproductive pairs."""
        pairs = []
        for i in range(10):
            c1 = _make_clause([(True, _make_term(i))], clause_id=i * 3 + 1)
            c2 = _make_clause([(False, _make_term(i))], clause_id=i * 3 + 2)
            c3 = _make_clause([], clause_id=i * 3 + 3)
            pairs.append(InferencePair(
                parent1=c1, parent2=c2, child=c3,
                label=PairLabel.PRODUCTIVE if i < 5 else PairLabel.UNPRODUCTIVE,
                proof_depth=i if i < 5 else -1,
            ))
        return pairs

    def test_dataset_length(self):
        pairs = self._make_pairs()
        ds = TrainingDataset(pairs)
        # Length = number of productive pairs
        assert len(ds) == 5

    def test_getitem_structure(self):
        pairs = self._make_pairs()
        config = ContrastiveConfig(augmentation_prob=0.0, max_negatives=3)
        ds = TrainingDataset(pairs, config)

        item = ds[0]
        assert "anchor" in item
        assert "positive" in item
        assert "negatives" in item
        assert "depth_weight" in item
        assert isinstance(item["negatives"], list)
        assert len(item["negatives"]) <= 3

    def test_depth_weighting(self):
        pairs = self._make_pairs()
        config = ContrastiveConfig(augmentation_prob=0.0, depth_weight=True)
        ds = TrainingDataset(pairs, config)

        # Pair at depth 0 should have highest weight
        item0 = ds[0]
        item4 = ds[4]
        assert item0["depth_weight"] >= item4["depth_weight"]


# ── ContrastiveLoss tests ──────────────────────────────────────────────────


class TestContrastiveLoss:
    def test_basic_loss(self):
        loss_fn = ContrastiveLoss(temperature=0.07)
        batch = 4
        dim = 64
        n_neg = 8

        anchor = torch.randn(batch, dim)
        positive = anchor + 0.1 * torch.randn(batch, dim)  # similar
        negatives = torch.randn(batch, n_neg, dim)

        loss = loss_fn(anchor, positive, negatives)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_perfect_alignment_low_loss(self):
        loss_fn = ContrastiveLoss(temperature=0.5)

        anchor = torch.randn(4, 32)
        positive = anchor.clone()  # identical
        negatives = torch.randn(4, 8, 32)  # random

        loss = loss_fn(anchor, positive, negatives)
        # With identical positives and random negatives, loss should be low
        assert loss.item() < 2.0

    def test_weighted_loss(self):
        loss_fn = ContrastiveLoss(temperature=0.1)

        anchor = torch.randn(4, 32)
        positive = torch.randn(4, 32)
        negatives = torch.randn(4, 4, 32)
        weights = torch.tensor([2.0, 1.0, 1.0, 0.5])

        loss = loss_fn(anchor, positive, negatives, weights)
        assert loss.shape == ()
        assert loss.item() >= 0


class TestTripletMarginLoss:
    def test_basic_loss(self):
        loss_fn = TripletMarginContrastiveLoss(margin=0.5)
        batch = 4
        dim = 64

        anchor = torch.randn(batch, dim)
        positive = anchor + 0.1 * torch.randn(batch, dim)
        negatives = torch.randn(batch, 8, dim)

        loss = loss_fn(anchor, positive, negatives)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_zero_loss_when_margin_satisfied(self):
        loss_fn = TripletMarginContrastiveLoss(margin=0.1)

        anchor = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        positive = torch.tensor([[0.99, 0.1, 0.0, 0.0]])
        negatives = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])  # far away

        loss = loss_fn(anchor, positive, negatives)
        # Positive is close, negative is far, margin should be satisfied
        assert loss.item() < 0.5


# ── EmbeddingEvaluator tests ──────────────────────────────────────────────


class TestEmbeddingEvaluator:
    def test_evaluate_basic(self):
        encoder = MockEncoder(dim=64)
        evaluator = EmbeddingEvaluator(encoder)

        pairs = []
        for i in range(10):
            c1 = _make_clause([(True, _make_term(i))], clause_id=i * 3 + 1)
            c2 = _make_clause([(False, _make_term(i))], clause_id=i * 3 + 2)
            c3 = _make_clause([], clause_id=i * 3 + 3)
            pairs.append(InferencePair(
                parent1=c1, parent2=c2, child=c3,
                label=PairLabel.PRODUCTIVE if i < 5 else PairLabel.UNPRODUCTIVE,
            ))

        metrics = evaluator.evaluate(pairs)
        assert isinstance(metrics, EvalMetrics)
        assert metrics.num_pairs == 10
        # Similarities should be in [-1, 1] range
        assert -1.0 <= metrics.mean_positive_similarity <= 1.0
        assert -1.0 <= metrics.mean_negative_similarity <= 1.0

    def test_evaluate_empty_pairs(self):
        encoder = MockEncoder(dim=64)
        evaluator = EmbeddingEvaluator(encoder)
        metrics = evaluator.evaluate([])
        assert metrics.num_pairs == 0


# ── ContrastiveConfig tests ────────────────────────────────────────────────


class TestContrastiveConfig:
    def test_defaults(self):
        config = ContrastiveConfig()
        assert config.temperature == 0.07
        assert config.batch_size == 64
        assert config.learning_rate == 1e-4
        assert config.max_negatives == 15

    def test_custom_config(self):
        config = ContrastiveConfig(
            temperature=0.1,
            batch_size=32,
            max_negatives=10,
        )
        assert config.temperature == 0.1
        assert config.batch_size == 32
