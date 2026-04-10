"""Online learning system for continuous model improvement during search.

Tracks inference outcomes (productive vs. unproductive) during live theorem
proving and periodically updates the clause embedding model. Designed to:

- Adapt to problem-specific patterns within ~1000 processed clauses
- Maintain stability without catastrophic forgetting
- Roll back poor updates that degrade performance
- Support A/B testing between model versions
"""

from __future__ import annotations

import copy
import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from pyladr.core.clause import Clause
    from pyladr.ml.training.contrastive import ClauseEncoder, InferencePair

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OnlineLearningConfig:
    """Configuration for online learning during theorem proving.

    Attributes:
        enabled: Master switch for online learning.
        update_interval: Number of new examples between model updates.
        min_examples_for_update: Minimum examples needed before first update.
        buffer_capacity: Maximum examples stored in the experience buffer.
        batch_size: Batch size for online gradient steps.
        learning_rate: Learning rate for online updates (typically lower
            than offline training to prevent catastrophic forgetting).
        gradient_steps_per_update: Number of gradient steps per update cycle.
        momentum: EMA momentum for model averaging (higher = more stable).
        rollback_threshold: If validation performance drops by more than this
            fraction, roll back to the previous model version.
        ab_test_window: Number of selections to track for A/B comparison.
        ab_test_significance: Minimum win-rate difference to accept new model.
        temperature: Temperature for contrastive loss in online updates.
        max_updates: Maximum number of online updates per search (0 = unlimited).
        max_versions: Maximum model version snapshots to retain. Older versions
            are pruned to bound memory growth during long searches. The initial
            version (v0), the best version, and the most recent versions are
            always kept. Set to 0 for unlimited (not recommended for long runs).
    """

    enabled: bool = True
    update_interval: int = 200
    min_examples_for_update: int = 50
    buffer_capacity: int = 5000
    batch_size: int = 32
    learning_rate: float = 5e-5
    gradient_steps_per_update: int = 5
    momentum: float = 0.995
    rollback_threshold: float = 0.1
    ab_test_window: int = 100
    ab_test_significance: float = 0.05
    temperature: float = 0.07
    max_updates: int = 0
    max_versions: int = 10


_DEFAULT_CONFIG = OnlineLearningConfig()


# ── Inference outcome tracking ─────────────────────────────────────────────


class OutcomeType(IntEnum):
    """Outcome of an inference step."""

    KEPT = auto()        # Clause was kept (passed all deletion checks)
    SUBSUMED = auto()    # Clause was forward subsumed
    TAUTOLOGY = auto()   # Clause was a tautology
    WEIGHT_LIMIT = auto()  # Exceeded weight/size limits
    PROOF = auto()       # Clause was part of a proof
    SUBSUMER = auto()    # Clause successfully subsumed another clause


@dataclass(frozen=True, slots=True)
class InferenceOutcome:
    """Records the outcome of a single inference step.

    Attributes:
        given_clause: The given clause that triggered the inference.
        partner_clause: The other clause involved (for binary inferences).
        child_clause: The resulting inferred clause.
        outcome: What happened to the child clause.
        timestamp: When this inference occurred (monotonic clock).
        given_count: The given clause count at the time of this inference.
    """

    given_clause: Clause
    partner_clause: Clause | None
    child_clause: Clause
    outcome: OutcomeType
    timestamp: float = 0.0
    given_count: int = 0


# ── Experience buffer ──────────────────────────────────────────────────────


class ExperienceBuffer:
    """Memory-efficient, thread-safe circular buffer for inference outcomes.

    Stores recent inference outcomes for training. Older examples are
    evicted when the buffer reaches capacity. Maintains separate indices
    for productive (kept/proof) and unproductive outcomes for balanced
    sampling.

    Thread safety is provided via a readers-writer lock, making this
    safe for concurrent access during parallel inference generation.
    """

    __slots__ = (
        "_capacity", "_protected_proofs", "_regular_buffer", "_productive_idx", "_unproductive_idx",
        "_lock", "_total_added", "_proof_outcomes",
    )

    def __init__(self, capacity: int = 5000):
        self._capacity = capacity
        # Split buffer: protected PROOF outcomes + regular circular buffer
        self._protected_proofs: list[InferenceOutcome] = []  # Never evicted
        self._regular_buffer: deque[InferenceOutcome] = deque()  # No maxlen - we'll manage manually
        self._productive_idx: list[int] = []
        self._unproductive_idx: list[int] = []
        self._lock = threading.Lock()
        self._total_added: int = 0
        self._proof_outcomes: list[int] = []  # indices of PROOF outcomes

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._protected_proofs) + len(self._regular_buffer)

    @property
    def _unified_buffer(self) -> list[InferenceOutcome]:
        """Unified view of both protected and regular buffers."""
        return self._protected_proofs + list(self._regular_buffer)

    @property
    def num_productive(self) -> int:
        with self._lock:
            return len(self._productive_idx)

    @property
    def num_unproductive(self) -> int:
        with self._lock:
            return len(self._unproductive_idx)

    @property
    def total_added(self) -> int:
        """Total outcomes ever added (including evicted)."""
        with self._lock:
            return self._total_added

    @property
    def productivity_rate(self) -> float:
        """Current fraction of productive outcomes in buffer."""
        with self._lock:
            total = len(self._productive_idx) + len(self._unproductive_idx)
            if total == 0:
                return 0.0
            return len(self._productive_idx) / total

    def add(self, outcome: InferenceOutcome) -> None:
        """Add an outcome to the buffer (thread-safe).

        PROOF outcomes are stored in protected buffer (never evicted).
        SUBSUMER/KEPT outcomes are productive; others are unproductive.
        All non-PROOF outcomes go to regular buffer with circular eviction.
        """
        with self._lock:
            self._total_added += 1

            if outcome.outcome == OutcomeType.PROOF:
                # PROOF outcomes go to protected buffer (never evicted)
                self._protected_proofs.append(outcome)
                # Index in unified buffer space
                idx = len(self._protected_proofs) - 1
                self._productive_idx.append(idx)
                self._proof_outcomes.append(idx)
            else:
                # Regular outcomes go to circular buffer
                self._regular_buffer.append(outcome)
                # Manage capacity manually for regular buffer
                regular_capacity = self._capacity - len(self._protected_proofs)
                while len(self._regular_buffer) > max(regular_capacity, 1):
                    self._regular_buffer.popleft()

                # Index in unified buffer space
                idx = len(self._protected_proofs) + len(self._regular_buffer) - 1
                if outcome.outcome in (OutcomeType.KEPT, OutcomeType.SUBSUMER):
                    self._productive_idx.append(idx)
                else:
                    self._unproductive_idx.append(idx)

            # Rebuild indices periodically to handle evictions/shifts
            if self._total_added % 500 == 0:
                self._rebuild_indices_unlocked()

    def add_batch(self, outcomes: list[InferenceOutcome]) -> None:
        """Add multiple outcomes atomically (thread-safe)."""
        with self._lock:
            for outcome in outcomes:
                self._total_added += 1

                if outcome.outcome == OutcomeType.PROOF:
                    # PROOF outcomes go to protected buffer (never evicted)
                    self._protected_proofs.append(outcome)
                    # Index in unified buffer space
                    idx = len(self._protected_proofs) - 1
                    self._productive_idx.append(idx)
                    self._proof_outcomes.append(idx)
                else:
                    # Regular outcomes go to circular buffer
                    self._regular_buffer.append(outcome)
                    # Manage capacity manually for regular buffer
                    regular_capacity = self._capacity - len(self._protected_proofs)
                    while len(self._regular_buffer) > max(regular_capacity, 1):
                        self._regular_buffer.popleft()

                    # Index in unified buffer space
                    idx = len(self._protected_proofs) + len(self._regular_buffer) - 1
                    if outcome.outcome in (OutcomeType.KEPT, OutcomeType.SUBSUMER):
                        self._productive_idx.append(idx)
                    else:
                        self._unproductive_idx.append(idx)

            # Rebuild indices at end to handle any evictions
            self._rebuild_indices_unlocked()

    def sample_contrastive_batch(
        self, batch_size: int,
    ) -> list[tuple[InferenceOutcome, InferenceOutcome]]:
        """Sample (productive, unproductive) pairs for contrastive training.

        Returns:
            List of (positive_outcome, negative_outcome) tuples.
        """
        with self._lock:
            self._rebuild_indices_unlocked()

            if not self._productive_idx or not self._unproductive_idx:
                return []

            n = min(batch_size, len(self._productive_idx), len(self._unproductive_idx))
            pos_samples = random.sample(self._productive_idx, n)
            neg_samples = random.sample(self._unproductive_idx, n)

            unified_buffer = self._unified_buffer
            return [
                (unified_buffer[pi], unified_buffer[ni])
                for pi, ni in zip(pos_samples, neg_samples)
            ]

    def sample_weighted_batch(
        self, batch_size: int,
    ) -> list[tuple[InferenceOutcome, InferenceOutcome]]:
        """Sample pairs with priority weighting for proof outcomes.

        Proof-participating outcomes are sampled with 3x weight relative
        to regular KEPT outcomes, providing stronger learning signal.
        """
        with self._lock:
            self._rebuild_indices_unlocked()

            if not self._productive_idx or not self._unproductive_idx:
                return []

            # Weight proof outcomes 3x higher
            proof_set = frozenset(self._proof_outcomes)
            weighted_productive: list[int] = []
            for idx in self._productive_idx:
                if idx in proof_set:
                    weighted_productive.extend([idx, idx, idx])
                else:
                    weighted_productive.append(idx)

            n = min(batch_size, len(weighted_productive), len(self._unproductive_idx))
            pos_samples = random.sample(weighted_productive, n)
            neg_samples = random.sample(self._unproductive_idx, n)

            unified_buffer = self._unified_buffer
            return [
                (unified_buffer[pi], unified_buffer[ni])
                for pi, ni in zip(pos_samples, neg_samples)
            ]

    def _rebuild_indices(self) -> None:
        """Rebuild productive/unproductive indices after evictions (thread-safe)."""
        with self._lock:
            self._rebuild_indices_unlocked()

    def _rebuild_indices_unlocked(self) -> None:
        """Rebuild indices — caller must hold the lock."""
        self._productive_idx = []
        self._unproductive_idx = []
        self._proof_outcomes = []

        # Index protected PROOF outcomes first
        for i, outcome in enumerate(self._protected_proofs):
            self._productive_idx.append(i)
            self._proof_outcomes.append(i)

        # Index regular buffer outcomes (offset by protected buffer size)
        protected_size = len(self._protected_proofs)
        for i, outcome in enumerate(self._regular_buffer):
            unified_idx = protected_size + i
            if outcome.outcome in (OutcomeType.KEPT, OutcomeType.SUBSUMER):
                self._productive_idx.append(unified_idx)
            else:
                self._unproductive_idx.append(unified_idx)

    def clear(self) -> None:
        """Clear the buffer (thread-safe)."""
        with self._lock:
            self._protected_proofs.clear()
            self._regular_buffer.clear()
            self._productive_idx.clear()
            self._unproductive_idx.clear()
            self._proof_outcomes.clear()

    def get_recent(self, n: int) -> list[InferenceOutcome]:
        """Get the N most recent outcomes (thread-safe)."""
        with self._lock:
            unified_buffer = self._unified_buffer
            if n >= len(unified_buffer):
                return list(unified_buffer)
            return list(unified_buffer)[-n:]

    def snapshot(self) -> dict[str, int | float]:
        """Return a point-in-time snapshot of buffer statistics."""
        with self._lock:
            total = len(self._productive_idx) + len(self._unproductive_idx)
            return {
                "size": len(self._protected_proofs) + len(self._regular_buffer),
                "capacity": self._capacity,
                "protected_proofs": len(self._protected_proofs),
                "productive": len(self._productive_idx),
                "unproductive": len(self._unproductive_idx),
                "proof_outcomes": len(self._proof_outcomes),
                "total_added": self._total_added,
                "productivity_rate": (
                    len(self._productive_idx) / total if total > 0 else 0.0
                ),
            }


# ── Model version tracking ─────────────────────────────────────────────────


@dataclass(slots=True)
class ModelVersion:
    """Tracks a model version with performance metadata.

    Attributes:
        version_id: Sequential version number.
        state_dict: Snapshot of model parameters.
        selections_made: Number of clause selections made with this version.
        productive_selections: Selections that led to kept/proof clauses.
        avg_loss: Average training loss when this version was created.
        created_at: Monotonic timestamp.
    """

    version_id: int
    state_dict: dict
    selections_made: int = 0
    productive_selections: int = 0
    avg_loss: float = float("inf")
    created_at: float = 0.0

    @property
    def productivity_rate(self) -> float:
        """Fraction of selections that were productive."""
        if self.selections_made == 0:
            return 0.0
        return self.productive_selections / self.selections_made


# ── A/B test tracker ───────────────────────────────────────────────────────


class ABTestTracker:
    """Tracks performance of two model versions for A/B comparison.

    Uses a sliding window to compare the productivity rate of the current
    model against the previous version.
    """

    __slots__ = ("_window_size", "_current_outcomes", "_baseline_rate")

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._current_outcomes: deque[bool] = deque(maxlen=window_size)
        self._baseline_rate: float = 0.0

    def set_baseline(self, rate: float) -> None:
        """Set the baseline productivity rate from the previous model."""
        self._baseline_rate = rate
        self._current_outcomes.clear()

    def record_outcome(self, productive: bool) -> None:
        """Record whether a selection was productive."""
        self._current_outcomes.append(productive)

    @property
    def current_rate(self) -> float:
        """Current model's productivity rate."""
        if not self._current_outcomes:
            return 0.0
        return sum(self._current_outcomes) / len(self._current_outcomes)

    @property
    def has_enough_data(self) -> bool:
        """Whether we have enough data to make a comparison."""
        return len(self._current_outcomes) >= self._window_size // 2

    def is_improvement(self, significance: float = 0.05) -> bool:
        """Whether the current model is significantly better."""
        if not self.has_enough_data:
            return False
        return self.current_rate > self._baseline_rate + significance

    def is_degradation(self, threshold: float = 0.1) -> bool:
        """Whether the current model is significantly worse."""
        if not self.has_enough_data:
            return False
        return self.current_rate < self._baseline_rate - threshold


# ── Online learning manager ────────────────────────────────────────────────


class OnlineLearningManager:
    """Manages continuous model improvement during theorem proving.

    Integrates with the search loop to:
    1. Track inference outcomes as they happen
    2. Periodically update the model with accumulated experience
    3. Maintain model stability via EMA and rollback mechanisms
    4. A/B test new model versions against the baseline

    Usage:
        manager = OnlineLearningManager(encoder, config)

        # During search, after each inference:
        manager.record_outcome(InferenceOutcome(...))

        # Periodically (e.g., after each given clause):
        if manager.should_update():
            manager.update()

        # After proof found:
        manager.on_proof_found(proof_clause_ids)
    """

    __slots__ = (
        "_encoder", "_config", "_buffer", "_optimizer",
        "_versions", "_current_version", "_ab_tracker",
        "_ema_state", "_pre_ema_state", "_update_count",
        "_examples_since_update", "_total_outcomes", "_device",
        "_loss_fn",
    )

    def __init__(
        self,
        encoder: ClauseEncoder,
        config: OnlineLearningConfig | None = None,
        device: torch.device | None = None,
    ):
        from pyladr.ml.training.online_losses import OnlineInfoNCELoss, OnlineLossConfig

        self._encoder = encoder
        self._config = config or _DEFAULT_CONFIG
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._buffer = ExperienceBuffer(self._config.buffer_capacity)
        self._optimizer: torch.optim.Optimizer | None = None
        self._loss_fn = OnlineInfoNCELoss(OnlineLossConfig(
            temperature=self._config.temperature,
        ))
        self._versions: list[ModelVersion] = []
        self._current_version: ModelVersion | None = None
        self._ab_tracker = ABTestTracker(self._config.ab_test_window)
        self._ema_state: dict[str, torch.Tensor] | None = None
        self._pre_ema_state: dict[str, torch.Tensor] | None = None
        self._update_count = 0
        self._examples_since_update = 0
        self._total_outcomes = 0

        if self._config.enabled:
            self._init_optimizer()
            self._snapshot_initial_model()

    def _init_optimizer(self) -> None:
        """Initialize optimizer for online updates."""
        self._optimizer = torch.optim.AdamW(
            self._encoder.parameters(),
            lr=self._config.learning_rate,
            weight_decay=1e-5,
        )

    def _snapshot_initial_model(self) -> None:
        """Take a snapshot of the initial model as version 0."""
        state = {k: v.clone() for k, v in self._encoder.state_dict().items()}
        version = ModelVersion(
            version_id=0,
            state_dict=state,
            created_at=time.monotonic(),
        )
        self._versions.append(version)
        self._current_version = version
        self._ema_state = {k: v.clone() for k, v in state.items()}

    # ── Outcome recording ──────────────────────────────────────────────

    def record_outcome(self, outcome: InferenceOutcome) -> None:
        """Record the outcome of an inference step.

        Call this for every inference generated during search. The manager
        accumulates these outcomes and uses them for periodic model updates.

        Args:
            outcome: The inference outcome to record.
        """
        if not self._config.enabled:
            return

        self._buffer.add(outcome)
        self._examples_since_update += 1
        self._total_outcomes += 1

        # Track A/B test outcomes
        productive = outcome.outcome in (OutcomeType.KEPT, OutcomeType.PROOF, OutcomeType.SUBSUMER)
        self._ab_tracker.record_outcome(productive)

    def on_proof_found(self, proof_clause_ids: set[int]) -> None:
        """Update outcomes in buffer when a proof is found.

        Retroactively marks clauses that participated in the proof as
        PROOF outcomes for stronger learning signal.

        Args:
            proof_clause_ids: Set of clause IDs in the proof derivation.
        """
        if not self._config.enabled:
            return

        # Collect proof outcomes first, then add them (avoid mutating during iteration)
        proof_outcomes: list[InferenceOutcome] = []
        for outcome in list(self._buffer._unified_buffer):
            if outcome.child_clause.id in proof_clause_ids:
                proof_outcomes.append(InferenceOutcome(
                    given_clause=outcome.given_clause,
                    partner_clause=outcome.partner_clause,
                    child_clause=outcome.child_clause,
                    outcome=OutcomeType.PROOF,
                    timestamp=outcome.timestamp,
                    given_count=outcome.given_count,
                ))
        for po in proof_outcomes:
            self._buffer.add(po)

    # ── Update control ─────────────────────────────────────────────────

    def should_update(self) -> bool:
        """Check if it's time for a model update.

        Returns True when enough new examples have accumulated since the
        last update, subject to minimum example and max update constraints.
        """
        if not self._config.enabled:
            return False

        if self._config.max_updates > 0 and self._update_count >= self._config.max_updates:
            return False

        if self._buffer.size < self._config.min_examples_for_update:
            return False

        return self._examples_since_update >= self._config.update_interval

    def update(self) -> bool:
        """Perform an online model update.

        Samples from the experience buffer and runs a few gradient steps
        of contrastive learning. Uses EMA for stability and checks for
        degradation via A/B testing.

        Returns:
            True if the update was accepted, False if rolled back.
        """
        if not self._config.enabled or self._optimizer is None:
            return False

        batch_pairs = self._buffer.sample_weighted_batch(
            self._config.batch_size,
        )
        if not batch_pairs:
            return False

        # Save pre-update state for potential rollback
        pre_state = {k: v.clone() for k, v in self._encoder.state_dict().items()}

        # Run gradient steps
        self._encoder.train()
        total_loss = 0.0
        n_steps = 0

        for step in range(self._config.gradient_steps_per_update):
            loss = self._gradient_step(batch_pairs)
            if loss is not None:
                total_loss += loss
                n_steps += 1

        avg_loss = total_loss / max(1, n_steps)

        # Apply EMA for stability
        self._apply_ema()

        # Check for degradation
        if self._should_rollback(avg_loss):
            logger.info(
                "Rolling back update %d (loss=%.4f, degradation detected)",
                self._update_count + 1, avg_loss,
            )
            self._encoder.load_state_dict(pre_state)
            return False

        # Accept update
        self._update_count += 1
        self._examples_since_update = 0

        # Snapshot new version
        state = {k: v.clone() for k, v in self._encoder.state_dict().items()}
        new_version = ModelVersion(
            version_id=self._update_count,
            state_dict=state,
            avg_loss=avg_loss,
            created_at=time.monotonic(),
        )
        self._versions.append(new_version)

        # Prune old versions to bound memory growth
        self._prune_versions()

        # Set up A/B tracking for new version
        if self._current_version is not None:
            self._ab_tracker.set_baseline(self._current_version.productivity_rate)
        self._current_version = new_version

        logger.info(
            "Online update %d accepted (loss=%.4f, buffer=%d/%d productive)",
            self._update_count, avg_loss,
            self._buffer.num_productive, self._buffer.size,
        )

        return True

    def _gradient_step(
        self,
        batch_pairs: list[tuple[InferenceOutcome, InferenceOutcome]],
    ) -> float | None:
        """Execute one gradient step on a batch of contrastive pairs.

        Args:
            batch_pairs: List of (productive, unproductive) outcome pairs.

        Returns:
            Loss value, or None if the step failed.
        """
        if not batch_pairs:
            return None

        # Build clause lists
        anchors = [p.given_clause for p, _ in batch_pairs]
        positives = [
            p.partner_clause if p.partner_clause is not None else p.child_clause
            for p, _ in batch_pairs
        ]
        negatives = [n.child_clause for _, n in batch_pairs]

        # Encode
        anchor_emb = self._encoder.encode_clauses(anchors)
        pos_emb = self._encoder.encode_clauses(positives)
        neg_emb = self._encoder.encode_clauses(negatives)

        # Compute loss via the online loss module (handles normalization,
        # temperature scaling, and optional in-batch negative mining)
        loss = self._loss_fn(anchor_emb, pos_emb, neg_emb)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._encoder.parameters(), 0.5)
        self._optimizer.step()

        return loss.item()

    def _apply_ema(self) -> None:
        """Apply exponential moving average to model parameters.

        EMA provides a smoothed version of the model that is more resistant
        to noisy online updates and catastrophic forgetting.
        """
        if self._ema_state is None:
            return

        momentum = self._config.momentum
        with torch.no_grad():
            for name, param in self._encoder.named_parameters():
                if name in self._ema_state:
                    self._ema_state[name].mul_(momentum).add_(
                        param.data, alpha=1.0 - momentum
                    )

    def _prune_versions(self) -> None:
        """Prune old model version snapshots to bound memory growth.

        Keeps: the initial version (v0), the best-performing version,
        the current version, and the most recent max_versions entries.
        All other versions are removed from the list entirely.
        """
        max_v = self._config.max_versions
        if max_v <= 0 or len(self._versions) <= max_v:
            return

        # Identify versions to protect
        protected_ids: set[int] = set()

        # Always keep initial (v0)
        if self._versions:
            protected_ids.add(self._versions[0].version_id)

        # Always keep current
        if self._current_version is not None:
            protected_ids.add(self._current_version.version_id)

        # Keep best by productivity rate
        tested = [v for v in self._versions if v.selections_made > 0]
        if tested:
            best = max(tested, key=lambda v: v.productivity_rate)
            protected_ids.add(best.version_id)

        # Keep most recent max_versions entries
        recent_ids = {
            v.version_id for v in self._versions[-max_v:]
        }
        protected_ids |= recent_ids

        # Remove unprotected versions from the list entirely
        before = len(self._versions)
        self._versions = [
            v for v in self._versions
            if v.version_id in protected_ids
        ]
        pruned = before - len(self._versions)

        if pruned > 0:
            logger.debug(
                "Pruned %d model version snapshots (%d remaining)",
                pruned, len(self._versions),
            )

    def _should_rollback(self, avg_loss: float) -> bool:
        """Check whether the update should be rolled back.

        Uses A/B test results if available, otherwise checks for
        loss divergence.
        """
        if self._current_version is None:
            return False

        # Check A/B test for degradation
        if self._ab_tracker.has_enough_data:
            if self._ab_tracker.is_degradation(self._config.rollback_threshold):
                return True

        # Check for loss explosion (more than 5x the previous best)
        if self._current_version.avg_loss < float("inf"):
            if avg_loss > 5.0 * self._current_version.avg_loss:
                return True

        return False

    def use_ema_model(self) -> None:
        """Switch the encoder to use EMA-averaged parameters.

        Call this before inference/selection to use the more stable
        EMA model. Call restore_training_model() before the next update.
        """
        if self._ema_state is None:
            return
        self._pre_ema_state = {
            k: v.clone() for k, v in self._encoder.state_dict().items()
        }
        self._encoder.load_state_dict(self._ema_state)

    def restore_training_model(self) -> None:
        """Restore the non-EMA model parameters for training."""
        if hasattr(self, "_pre_ema_state") and self._pre_ema_state is not None:
            self._encoder.load_state_dict(self._pre_ema_state)
            self._pre_ema_state = None

    # ── Rollback ───────────────────────────────────────────────────────

    def rollback_to_version(self, version_id: int) -> bool:
        """Roll back the model to a specific version.

        Args:
            version_id: The version to restore.

        Returns:
            True if rollback succeeded.
        """
        for v in self._versions:
            if v.version_id == version_id:
                self._encoder.load_state_dict(v.state_dict)
                self._current_version = v
                self._ema_state = {k: v2.clone() for k, v2 in v.state_dict.items()}
                logger.info("Rolled back to model version %d", version_id)
                return True
        logger.warning(
            "Cannot roll back to version %d: not found (may have been pruned)",
            version_id,
        )
        return False

    def rollback_to_best(self) -> bool:
        """Roll back to the version with the best productivity rate."""
        if not self._versions:
            return False

        best = max(
            self._versions,
            key=lambda v: v.productivity_rate if v.selections_made > 0 else -1,
        )
        if best.selections_made == 0:
            # No version has been tested yet, use initial
            best = self._versions[0]

        return self.rollback_to_version(best.version_id)

    # ── Statistics ─────────────────────────────────────────────────────

    @property
    def loss_stats(self) -> dict[str, float | int]:
        """Loss function statistics from the online contrastive loss."""
        return self._loss_fn.stats.snapshot()

    @property
    def stats(self) -> dict[str, float | int]:
        """Current online learning statistics."""
        base = {
            "total_outcomes": self._total_outcomes,
            "buffer_size": self._buffer.size,
            "buffer_productive": self._buffer.num_productive,
            "buffer_unproductive": self._buffer.num_unproductive,
            "update_count": self._update_count,
            "current_version": (
                self._current_version.version_id
                if self._current_version else -1
            ),
            "current_productivity": (
                self._current_version.productivity_rate
                if self._current_version else 0.0
            ),
            "ab_test_current_rate": self._ab_tracker.current_rate,
        }
        # Include loss EMA and similarity gap for monitoring
        loss_s = self._loss_fn.stats
        base["loss_ema"] = loss_s.ema_loss
        base["similarity_gap"] = loss_s.similarity_gap
        return base

    def report(self) -> str:
        """Human-readable summary of online learning state."""
        s = self.stats
        return (
            f"OnlineLearning: v{s['current_version']}, "
            f"updates={s['update_count']}, "
            f"buffer={s['buffer_size']} "
            f"({s['buffer_productive']}+/{s['buffer_unproductive']}-), "
            f"ab_rate={s['ab_test_current_rate']:.3f}"
        )

    # ── Convergence detection ──────────────────────────────────────────

    def has_converged(self, window: int = 5, threshold: float = 0.01) -> bool:
        """Detect if online learning has converged.

        Convergence is detected when the productivity rate has been stable
        (variance below threshold) over the last `window` model versions.
        """
        if len(self._versions) < window:
            return False

        recent = self._versions[-window:]
        rates = [v.productivity_rate for v in recent if v.selections_made > 0]
        if len(rates) < window // 2:
            return False

        mean_rate = sum(rates) / len(rates)
        variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
        return variance < threshold
