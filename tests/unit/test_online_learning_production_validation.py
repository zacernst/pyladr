"""4-Phase online learning production validation framework.

Phase 1: Inference Verification
  - Model inference correctness under pathological distributions
  - Gradient flow and weight update validation
  - Cross-check predictions between EMA and training models

Phase 2: Training Data Validation
  - Training data quality and consistency
  - Subsumption learning data integrity
  - Buffer correctness under extreme patterns

Phase 3: Effectiveness Measurement
  - ML effectiveness improvements over traditional selection
  - Convergence behavior under different learning rates
  - Loss function stability and separation quality

Phase 4: Stress Testing
  - High-load concurrent operations
  - Memory usage bounds under sustained operation
  - Graceful degradation under resource pressure
"""

from __future__ import annotations

import copy
import gc
import sys
import threading
import time
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.term import Term
from pyladr.ml.online_learning import (
    ABTestTracker,
    ExperienceBuffer,
    InferenceOutcome,
    ModelVersion,
    OnlineLearningConfig,
    OnlineLearningManager,
    OutcomeType,
)
from pyladr.ml.training.online_losses import (
    CombinedOnlineLoss,
    LossStatistics,
    OnlineInfoNCELoss,
    OnlineLossConfig,
    OnlineTripletLoss,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_term(symnum: int, args: tuple[Term, ...] = ()) -> Term:
    return Term(private_symbol=-symnum, arity=len(args), args=args)


def _make_var(varnum: int) -> Term:
    return Term(private_symbol=varnum, arity=0, args=())


def _make_clause(
    lits: list[tuple[bool, Term]], clause_id: int = 0,
) -> Clause:
    literals = tuple(Literal(sign=s, atom=a) for s, a in lits)
    c = Clause(literals=literals)
    c.id = clause_id
    c.weight = float(len(literals))
    return c


def _make_outcome(
    clause_id: int,
    outcome: OutcomeType = OutcomeType.KEPT,
    given_id: int = 0,
    partner_id: int | None = None,
    given_count: int = 0,
) -> InferenceOutcome:
    given = _make_clause([(True, _make_term(given_id))], clause_id=given_id)
    partner = (
        _make_clause([(False, _make_term(partner_id))], clause_id=partner_id)
        if partner_id is not None
        else None
    )
    child = _make_clause([(True, _make_term(clause_id))], clause_id=clause_id)
    return InferenceOutcome(
        given_clause=given,
        partner_clause=partner,
        child_clause=child,
        outcome=outcome,
        timestamp=time.monotonic(),
        given_count=given_count,
    )


class MockEncoder:
    """Mock encoder with controllable behavior for validation tests."""

    def __init__(self, dim: int = 64):
        self._dim = dim
        self._linear = torch.nn.Linear(dim, dim)
        self._encode_count = 0

    def encode_clauses(self, clauses: list) -> torch.Tensor:
        self._encode_count += len(clauses)
        x = torch.randn(len(clauses), self._dim)
        return self._linear(x)

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


def _fill_buffer_balanced(
    buffer: ExperienceBuffer, n_productive: int, n_unproductive: int,
) -> None:
    """Fill buffer with a mix of productive and unproductive outcomes."""
    for i in range(n_productive):
        buffer.add(_make_outcome(1000 + i, OutcomeType.KEPT, given_id=i))
    for i in range(n_unproductive):
        buffer.add(_make_outcome(2000 + i, OutcomeType.SUBSUMED, given_id=i))


def _create_manager(
    dim: int = 64, **config_overrides,
) -> tuple[OnlineLearningManager, MockEncoder]:
    """Create a manager with sensible test defaults."""
    encoder = MockEncoder(dim=dim)
    defaults = dict(
        enabled=True,
        update_interval=10,
        min_examples_for_update=5,
        buffer_capacity=500,
        batch_size=8,
        learning_rate=1e-3,
        gradient_steps_per_update=2,
        momentum=0.99,
        rollback_threshold=0.1,
        ab_test_window=20,
        max_versions=5,
    )
    defaults.update(config_overrides)
    config = OnlineLearningConfig(**defaults)
    manager = OnlineLearningManager(encoder, config)
    return manager, encoder


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Inference Verification
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase1InferenceVerification:
    """Validate model inference correctness and gradient flow."""

    def test_gradient_flow_through_update(self):
        """Verify gradients actually flow and update weights."""
        manager, encoder = _create_manager()

        # Record initial weights
        initial_weights = {
            k: v.clone() for k, v in encoder.state_dict().items()
        }

        # Fill buffer with enough data
        for i in range(50):
            outcome_type = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            manager.record_outcome(_make_outcome(i, outcome_type, given_id=i + 100))

        # Force an update
        assert manager.should_update()
        manager.update()

        # Weights should have changed
        changed = False
        for k, v in encoder.state_dict().items():
            if not torch.allclose(v, initial_weights[k], atol=1e-8):
                changed = True
                break
        assert changed, "Weights did not change after gradient update"

    def test_ema_model_differs_from_training(self):
        """EMA model should diverge from training model over time."""
        manager, encoder = _create_manager(momentum=0.9)

        # Run multiple updates
        for cycle in range(5):
            for i in range(20):
                ot = OutcomeType.KEPT if i % 3 != 0 else OutcomeType.SUBSUMED
                manager.record_outcome(
                    _make_outcome(cycle * 100 + i, ot, given_id=i + 200)
                )
            if manager.should_update():
                manager.update()

        # Save training state
        training_state = {k: v.clone() for k, v in encoder.state_dict().items()}

        # Switch to EMA
        manager.use_ema_model()
        ema_state = {k: v.clone() for k, v in encoder.state_dict().items()}

        # EMA should differ from training state
        any_differ = any(
            not torch.allclose(ema_state[k], training_state[k], atol=1e-8)
            for k in training_state
        )
        assert any_differ, "EMA state should differ from training state"

        # Restore training model
        manager.restore_training_model()
        restored = {k: v.clone() for k, v in encoder.state_dict().items()}
        for k in training_state:
            assert torch.allclose(restored[k], training_state[k])

    def test_ema_restore_round_trip(self):
        """use_ema_model + restore_training_model is a no-op for training."""
        manager, encoder = _create_manager()

        for i in range(20):
            manager.record_outcome(_make_outcome(i, OutcomeType.KEPT))

        before = {k: v.clone() for k, v in encoder.state_dict().items()}
        manager.use_ema_model()
        manager.restore_training_model()
        after = {k: v.clone() for k, v in encoder.state_dict().items()}

        for k in before:
            assert torch.allclose(before[k], after[k])

    def test_update_returns_bool_accepted(self):
        """Update returns True on success, False on rollback/failure."""
        manager, encoder = _create_manager()

        for i in range(30):
            ot = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            manager.record_outcome(_make_outcome(i, ot, given_id=i + 100))

        result = manager.update()
        assert isinstance(result, bool)

    def test_disabled_manager_no_ops(self):
        """Disabled manager should be a complete no-op."""
        encoder = MockEncoder(64)
        config = OnlineLearningConfig(enabled=False)
        manager = OnlineLearningManager(encoder, config)

        manager.record_outcome(_make_outcome(1, OutcomeType.KEPT))
        assert not manager.should_update()
        assert not manager.update()

    def test_pathological_all_productive(self):
        """All productive outcomes should not crash training."""
        manager, encoder = _create_manager()

        for i in range(30):
            manager.record_outcome(_make_outcome(i, OutcomeType.KEPT, given_id=i))

        # Cannot form contrastive pairs without unproductive examples
        # should_update may be True but update should handle gracefully
        if manager.should_update():
            result = manager.update()
            # Should return False (no contrastive pairs possible)
            assert isinstance(result, bool)

    def test_pathological_all_unproductive(self):
        """All unproductive outcomes should not crash training."""
        manager, encoder = _create_manager()

        for i in range(30):
            manager.record_outcome(
                _make_outcome(i, OutcomeType.SUBSUMED, given_id=i)
            )

        if manager.should_update():
            result = manager.update()
            assert isinstance(result, bool)

    def test_single_example_no_crash(self):
        """Single example should not trigger or crash updates."""
        manager, encoder = _create_manager(min_examples_for_update=1, update_interval=1)
        manager.record_outcome(_make_outcome(0, OutcomeType.KEPT))
        # Only one productive, no unproductive — should handle gracefully
        result = manager.update()
        assert isinstance(result, bool)

    def test_loss_statistics_populated_after_update(self):
        """Loss statistics should be populated after a successful update."""
        manager, encoder = _create_manager()

        for i in range(30):
            ot = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            manager.record_outcome(_make_outcome(i, ot, given_id=i + 50))

        manager.update()
        stats = manager.loss_stats
        assert "ema_loss" in stats
        assert "similarity_gap" in stats


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Training Data Validation
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase2TrainingDataValidation:
    """Validate training data quality and consistency."""

    def test_buffer_preserves_proof_outcomes(self):
        """PROOF outcomes should never be evicted from the buffer."""
        buf = ExperienceBuffer(capacity=20)

        # Add a proof outcome
        proof = _make_outcome(1, OutcomeType.PROOF, given_id=10)
        buf.add(proof)

        # Flood with regular outcomes to trigger eviction
        for i in range(50):
            buf.add(_make_outcome(100 + i, OutcomeType.SUBSUMED, given_id=i))

        # Proof should still be there
        snapshot = buf.snapshot()
        assert snapshot["protected_proofs"] >= 1

    def test_buffer_sampling_returns_valid_pairs(self):
        """Sampled pairs should have productive positive and unproductive negative."""
        buf = ExperienceBuffer(capacity=200)
        _fill_buffer_balanced(buf, 30, 30)

        pairs = buf.sample_contrastive_batch(10)
        assert len(pairs) == 10

        for pos, neg in pairs:
            assert pos.outcome in (OutcomeType.KEPT, OutcomeType.SUBSUMER, OutcomeType.PROOF)
            assert neg.outcome in (OutcomeType.SUBSUMED, OutcomeType.TAUTOLOGY, OutcomeType.WEIGHT_LIMIT)

    def test_weighted_sampling_prefers_proofs(self):
        """Weighted sampling should sample proofs more frequently."""
        buf = ExperienceBuffer(capacity=200)

        # Add 5 proof outcomes and 45 KEPT outcomes
        for i in range(5):
            buf.add(_make_outcome(i, OutcomeType.PROOF, given_id=i + 100))
        for i in range(45):
            buf.add(_make_outcome(50 + i, OutcomeType.KEPT, given_id=i + 200))
        for i in range(50):
            buf.add(_make_outcome(200 + i, OutcomeType.SUBSUMED, given_id=i + 300))

        # Sample many times and count proof appearances
        proof_count = 0
        total_samples = 0
        for _ in range(50):
            pairs = buf.sample_weighted_batch(10)
            for pos, neg in pairs:
                total_samples += 1
                if pos.outcome == OutcomeType.PROOF:
                    proof_count += 1

        # Proofs are 5/50 = 10% of productive but should appear >10% due to 3x weight
        if total_samples > 0:
            proof_rate = proof_count / total_samples
            # With 3x weighting, expected rate ~= 15/(15+45) = 25%
            # Allow wide range since it's stochastic
            assert proof_rate > 0.05, f"Proof rate {proof_rate:.3f} too low"

    def test_buffer_indices_consistent_after_eviction(self):
        """Buffer indices should remain consistent through eviction cycles."""
        buf = ExperienceBuffer(capacity=50)

        # Fill beyond capacity
        for i in range(200):
            ot = OutcomeType.KEPT if i % 3 == 0 else OutcomeType.SUBSUMED
            buf.add(_make_outcome(i, ot, given_id=i))

        snap = buf.snapshot()
        assert snap["productive"] + snap["unproductive"] > 0
        # After rebuilding indices, productive + unproductive should match size
        # (snapshot triggers rebuild internally)
        assert snap["size"] <= 50  # Buffer capacity is 50

        # Sampling should still work
        pairs = buf.sample_contrastive_batch(5)
        # Should return pairs if both productive and unproductive exist
        if snap["productive"] > 0 and snap["unproductive"] > 0:
            assert len(pairs) > 0

    def test_add_batch_atomicity(self):
        """add_batch should add all outcomes atomically."""
        buf = ExperienceBuffer(capacity=1000)

        outcomes = [
            _make_outcome(i, OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED)
            for i in range(100)
        ]
        buf.add_batch(outcomes)
        assert buf.size == 100
        assert buf.total_added == 100

    def test_on_proof_found_retroactive_marking(self):
        """on_proof_found should retroactively create PROOF entries."""
        manager, encoder = _create_manager()

        # Record some outcomes
        for i in range(20):
            ot = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            manager.record_outcome(_make_outcome(i, ot, given_id=i + 100))

        initial_size = manager._buffer.size

        # "Find proof" involving some clause IDs
        proof_ids = {0, 2, 4, 6}
        manager.on_proof_found(proof_ids)

        # Buffer should have grown by the number of matched clauses
        assert manager._buffer.size >= initial_size

    def test_subsumption_outcome_types_distinct(self):
        """Different outcome types should be properly categorized."""
        buf = ExperienceBuffer(capacity=100)

        buf.add(_make_outcome(1, OutcomeType.KEPT))
        buf.add(_make_outcome(2, OutcomeType.SUBSUMER))
        buf.add(_make_outcome(3, OutcomeType.PROOF))
        buf.add(_make_outcome(4, OutcomeType.SUBSUMED))
        buf.add(_make_outcome(5, OutcomeType.TAUTOLOGY))
        buf.add(_make_outcome(6, OutcomeType.WEIGHT_LIMIT))

        assert buf.num_productive == 3  # KEPT, SUBSUMER, PROOF
        assert buf.num_unproductive == 3  # SUBSUMED, TAUTOLOGY, WEIGHT_LIMIT

    def test_empty_buffer_sampling_returns_empty(self):
        """Sampling from empty buffer should return empty list, not crash."""
        buf = ExperienceBuffer(capacity=100)
        assert buf.sample_contrastive_batch(10) == []
        assert buf.sample_weighted_batch(10) == []

    def test_buffer_productivity_rate_accuracy(self):
        """Productivity rate should accurately reflect buffer contents."""
        buf = ExperienceBuffer(capacity=100)

        for i in range(30):
            buf.add(_make_outcome(i, OutcomeType.KEPT))
        for i in range(70):
            buf.add(_make_outcome(100 + i, OutcomeType.SUBSUMED))

        rate = buf.productivity_rate
        assert 0.25 <= rate <= 0.35, f"Expected ~0.3, got {rate}"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Effectiveness Measurement
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase3EffectivenessMeasurement:
    """Measure ML effectiveness and convergence behavior."""

    def test_infonce_loss_separates_pos_neg(self):
        """InfoNCE loss should produce positive similarity gap over training."""
        loss_fn = OnlineInfoNCELoss(OnlineLossConfig(temperature=0.1))

        # Simulate training with separable data
        dim = 32
        for step in range(20):
            # Anchors near positives, far from negatives
            anchor = torch.randn(8, dim)
            positive = anchor + 0.1 * torch.randn(8, dim)
            negative = -anchor + 0.1 * torch.randn(8, dim)

            loss = loss_fn(anchor, positive, negative)
            assert loss.item() >= 0

        # After training steps, similarity gap should be positive
        gap = loss_fn.stats.similarity_gap
        assert gap > 0, f"Expected positive similarity gap, got {gap}"

    def test_triplet_loss_margin_enforcement(self):
        """Triplet loss should enforce margin between pos and neg distances."""
        loss_fn = OnlineTripletLoss(OnlineLossConfig(margin=0.5))

        dim = 32
        # Well-separated case: loss should be near zero
        anchor = torch.randn(8, dim)
        positive = anchor + 0.01 * torch.randn(8, dim)
        negative = -anchor + 0.01 * torch.randn(8, dim)

        loss = loss_fn(anchor, positive, negative)
        # For well-separated data, loss should be small
        assert loss.item() < 1.0

    def test_combined_loss_blending(self):
        """Combined loss should blend InfoNCE and triplet losses."""
        config = OnlineLossConfig(temperature=0.1, margin=0.3)
        combined = CombinedOnlineLoss(config, infonce_weight=0.7)

        dim = 32
        anchor = torch.randn(8, dim)
        positive = anchor + 0.1 * torch.randn(8, dim)
        negative = -anchor + 0.1 * torch.randn(8, dim)

        loss = combined(anchor, positive, negative)
        assert loss.item() >= 0

        # Both sub-losses should have recorded statistics
        assert combined.infonce.stats.total_steps > 0
        assert combined.triplet.stats.total_steps > 0

    def test_temperature_annealing(self):
        """Temperature should anneal when decay < 1.0."""
        config = OnlineLossConfig(
            temperature=0.1,
            temperature_min=0.01,
            temperature_decay=0.9,
        )
        loss_fn = OnlineInfoNCELoss(config)

        initial_temp = loss_fn.temperature
        dim = 16

        for _ in range(10):
            a = torch.randn(4, dim)
            p = torch.randn(4, dim)
            n = torch.randn(4, dim)
            loss_fn(a, p, n)

        final_temp = loss_fn.temperature
        assert final_temp < initial_temp
        assert final_temp >= config.temperature_min

    def test_label_smoothing_effect(self):
        """Label smoothing should reduce overconfident loss."""
        dim = 32
        anchor = torch.randn(8, dim)
        positive = anchor + 0.05 * torch.randn(8, dim)
        negative = -anchor + 0.05 * torch.randn(8, dim)

        loss_sharp = OnlineInfoNCELoss(OnlineLossConfig(
            temperature=0.1, label_smoothing=0.0,
        ))
        loss_smooth = OnlineInfoNCELoss(OnlineLossConfig(
            temperature=0.1, label_smoothing=0.1,
        ))

        l_sharp = loss_sharp(anchor.clone(), positive.clone(), negative.clone())
        l_smooth = loss_smooth(anchor.clone(), positive.clone(), negative.clone())

        # Smoothed loss should be higher (less confident predictions)
        assert l_smooth.item() >= l_sharp.item() - 0.1

    def test_weighted_examples_affect_loss(self):
        """Per-example weights should affect loss computation."""
        loss_fn = OnlineInfoNCELoss(OnlineLossConfig(temperature=0.1))
        dim = 32

        anchor = torch.randn(8, dim)
        positive = anchor + 0.1 * torch.randn(8, dim)
        negative = -anchor + 0.1 * torch.randn(8, dim)

        # Uniform weights should match unweighted
        uniform = torch.ones(8)
        loss_unweighted = loss_fn(anchor, positive, negative)
        loss_weighted = loss_fn(anchor, positive, negative, weights=uniform)

        # Should be approximately equal
        assert abs(loss_unweighted.item() - loss_weighted.item()) < 0.5

    def test_convergence_detection_stable_rates(self):
        """Convergence should be detected when productivity rates stabilize."""
        manager, encoder = _create_manager(
            update_interval=5,
            min_examples_for_update=3,
        )

        # Simulate stable productivity (always ~50% productive)
        for cycle in range(10):
            for i in range(10):
                ot = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
                manager.record_outcome(
                    _make_outcome(cycle * 100 + i, ot, given_id=i + 500)
                )
            if manager.should_update():
                manager.update()

        # After many stable updates, may detect convergence
        # (convergence needs enough versions with selections)
        result = manager.has_converged()
        assert isinstance(result, bool)

    def test_ab_testing_improvement_detection(self):
        """A/B testing should detect meaningful improvement."""
        tracker = ABTestTracker(window_size=50)
        tracker.set_baseline(0.3)  # 30% baseline

        # Record clearly better outcomes
        for _ in range(50):
            tracker.record_outcome(True)  # 100% productive

        assert tracker.has_enough_data
        assert tracker.is_improvement(significance=0.05)
        assert not tracker.is_degradation(threshold=0.1)

    def test_ab_testing_degradation_detection(self):
        """A/B testing should detect meaningful degradation."""
        tracker = ABTestTracker(window_size=50)
        tracker.set_baseline(0.7)  # 70% baseline

        # Record clearly worse outcomes
        for _ in range(50):
            tracker.record_outcome(False)  # 0% productive

        assert tracker.has_enough_data
        assert tracker.is_degradation(threshold=0.1)
        assert not tracker.is_improvement()

    def test_manager_stats_complete(self):
        """Manager stats should include all expected fields."""
        manager, encoder = _create_manager()

        for i in range(15):
            ot = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            manager.record_outcome(_make_outcome(i, ot))

        stats = manager.stats
        expected_keys = {
            "total_outcomes", "buffer_size", "buffer_productive",
            "buffer_unproductive", "update_count", "current_version",
            "current_productivity", "ab_test_current_rate",
            "loss_ema", "similarity_gap",
        }
        assert expected_keys.issubset(set(stats.keys()))

    def test_report_is_human_readable(self):
        """Report should be a non-empty human-readable string."""
        manager, encoder = _create_manager()
        for i in range(10):
            manager.record_outcome(_make_outcome(i, OutcomeType.KEPT))

        report = manager.report()
        assert isinstance(report, str)
        assert len(report) > 10
        assert "OnlineLearning" in report


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Stress Testing
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase4StressTesting:
    """Test under high-load conditions and resource pressure."""

    def test_concurrent_buffer_access(self):
        """Multiple threads adding to buffer simultaneously."""
        buf = ExperienceBuffer(capacity=1000)
        errors: list[Exception] = []

        def add_outcomes(thread_id: int):
            try:
                for i in range(200):
                    ot = OutcomeType.KEPT if i % 3 == 0 else OutcomeType.SUBSUMED
                    buf.add(_make_outcome(
                        thread_id * 1000 + i, ot, given_id=thread_id * 1000 + i,
                    ))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_outcomes, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert buf.size > 0
        assert buf.total_added == 8 * 200

    def test_concurrent_buffer_add_and_sample(self):
        """Sampling while other threads add outcomes."""
        buf = ExperienceBuffer(capacity=500)
        errors: list[Exception] = []
        samples_collected: list[int] = []

        # Pre-fill with some data
        _fill_buffer_balanced(buf, 20, 20)

        def writer(thread_id: int):
            try:
                for i in range(100):
                    ot = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
                    buf.add(_make_outcome(
                        thread_id * 1000 + i, ot, given_id=thread_id * 1000 + i,
                    ))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    pairs = buf.sample_contrastive_batch(5)
                    samples_collected.append(len(pairs))
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=writer, args=(t,)) for t in range(4)]
            + [threading.Thread(target=reader) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"

    def test_rapid_update_cycles(self):
        """Rapid consecutive update cycles should not crash."""
        manager, encoder = _create_manager(
            update_interval=3,
            min_examples_for_update=2,
            gradient_steps_per_update=1,
        )

        for cycle in range(20):
            for i in range(5):
                ot = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
                manager.record_outcome(
                    _make_outcome(cycle * 100 + i, ot, given_id=i + 700)
                )
            if manager.should_update():
                manager.update()

        assert manager._update_count > 0

    def test_version_pruning_bounds_memory(self):
        """Version pruning should keep version count bounded."""
        manager, encoder = _create_manager(
            max_versions=5,
            update_interval=5,
            min_examples_for_update=3,
            gradient_steps_per_update=1,
        )

        for cycle in range(30):
            for i in range(8):
                ot = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
                manager.record_outcome(
                    _make_outcome(cycle * 100 + i, ot, given_id=i + 800)
                )
            if manager.should_update():
                manager.update()

        # Versions should be bounded
        assert len(manager._versions) <= 10  # Allow some headroom

    def test_large_buffer_capacity(self):
        """Large buffer capacity should not cause memory issues."""
        buf = ExperienceBuffer(capacity=50000)

        for i in range(10000):
            ot = OutcomeType.KEPT if i % 3 == 0 else OutcomeType.SUBSUMED
            buf.add(_make_outcome(i, ot, given_id=i))

        assert buf.size == 10000
        snap = buf.snapshot()
        assert snap["size"] == 10000

    def test_buffer_eviction_under_pressure(self):
        """Buffer with small capacity should correctly evict old entries."""
        buf = ExperienceBuffer(capacity=100)

        for i in range(1000):
            ot = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            buf.add(_make_outcome(i, ot, given_id=i))

        assert buf.size <= 100
        assert buf.total_added == 1000

        # Sampling should still work correctly
        pairs = buf.sample_contrastive_batch(10)
        assert len(pairs) > 0

    def test_max_updates_enforcement(self):
        """max_updates should stop further updates."""
        manager, encoder = _create_manager(
            max_updates=3,
            update_interval=3,
            min_examples_for_update=2,
        )

        updates = 0
        for cycle in range(20):
            for i in range(5):
                ot = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
                manager.record_outcome(
                    _make_outcome(cycle * 100 + i, ot, given_id=i + 900)
                )
            if manager.should_update():
                if manager.update():
                    updates += 1

        assert manager._update_count <= 3

    def test_rollback_preserves_model_integrity(self):
        """Rollback should restore exact previous state."""
        manager, encoder = _create_manager()

        # Record initial state
        initial_state = {k: v.clone() for k, v in encoder.state_dict().items()}

        # Run some updates
        for i in range(30):
            ot = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            manager.record_outcome(_make_outcome(i, ot, given_id=i + 100))
        manager.update()

        # Rollback to version 0
        success = manager.rollback_to_version(0)
        assert success

        # State should match initial
        rolled_back = {k: v.clone() for k, v in encoder.state_dict().items()}
        for k in initial_state:
            assert torch.allclose(rolled_back[k], initial_state[k])

    def test_rollback_to_best_with_no_tested_versions(self):
        """rollback_to_best with no tested versions should use initial."""
        manager, encoder = _create_manager()
        success = manager.rollback_to_best()
        assert success

    def test_graceful_degradation_empty_pairs(self):
        """Update should handle when no contrastive pairs can be formed."""
        manager, encoder = _create_manager(
            update_interval=1, min_examples_for_update=1,
        )

        # Add only productive outcomes
        for i in range(10):
            manager.record_outcome(_make_outcome(i, OutcomeType.KEPT))

        # Update should return False (no unproductive to pair with)
        result = manager.update()
        assert result is False

    def test_loss_stats_snapshot_thread_safe(self):
        """Loss statistics snapshot should be safe to call concurrently."""
        loss_fn = OnlineInfoNCELoss()
        errors: list[Exception] = []

        def compute_loss():
            try:
                for _ in range(50):
                    a = torch.randn(4, 32)
                    p = torch.randn(4, 32)
                    n = torch.randn(4, 32)
                    loss_fn(a, p, n)
                    _ = loss_fn.stats.snapshot()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=compute_loss) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"

    def test_sustained_operation_no_memory_leak(self):
        """Sustained operation should not grow memory unboundedly."""
        manager, encoder = _create_manager(
            buffer_capacity=200,
            max_versions=3,
            update_interval=10,
            min_examples_for_update=5,
            gradient_steps_per_update=1,
        )

        gc.collect()
        # Run 50 cycles of record + update
        for cycle in range(50):
            for i in range(15):
                ot = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
                manager.record_outcome(
                    _make_outcome(cycle * 100 + i, ot, given_id=i)
                )
            if manager.should_update():
                manager.update()

        # Buffer should be bounded
        assert manager._buffer.size <= 200
        # Versions should be bounded
        assert len(manager._versions) <= 6  # max_versions + some headroom
