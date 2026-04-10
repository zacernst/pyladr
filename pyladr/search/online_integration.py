"""Integration layer connecting online learning with the search loop.

Provides the bridge between GivenClauseSearch and the ML online learning
infrastructure. This module handles:

- Experience collection from search events (clause kept/deleted/proof)
- Proof progress tracking with real-time feedback signals
- Learning trigger management (when to update the model)
- Cache invalidation after model updates
- Adaptive selection fallback control

The integration is designed to be non-intrusive: when disabled or when ML
dependencies are unavailable, search behavior is unchanged. All hooks are
no-ops unless explicitly activated.

Usage::

    from pyladr.search.online_integration import OnlineSearchIntegration

    integration = OnlineSearchIntegration.create(
        embedding_provider=provider,
        config=OnlineIntegrationConfig(enabled=True),
    )
    search = GivenClauseSearch(options, selection=selection)
    result = integration.run_with_learning(search, usable, sos)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyladr.core.clause import Clause

if TYPE_CHECKING:
    from pyladr.ml.embedding_provider import GNNEmbeddingProvider
    from pyladr.ml.online_learning import (
        OnlineLearningConfig,
        OnlineLearningManager,
    )
    from pyladr.search.given_clause import GivenClauseSearch, Proof, SearchResult
    from pyladr.search.ml_selection import EmbeddingEnhancedSelection

logger = logging.getLogger(__name__)


# Guard ML imports
try:
    from pyladr.ml.online_learning import (
        ExperienceBuffer,
        InferenceOutcome,
        OnlineLearningConfig as _OLConfig,
        OnlineLearningManager as _OLManager,
        OutcomeType,
    )
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OnlineIntegrationConfig:
    """Configuration for online learning integration with search.

    Attributes:
        enabled: Master switch for online learning during search.
        collect_experiences: Whether to record inference outcomes.
        trigger_updates: Whether to trigger model updates during search.
        track_proof_progress: Whether to track progress toward proof.
        invalidate_cache_on_update: Clear embedding cache after model update.
        adaptive_ml_weight: Dynamically adjust ML selection weight based
            on learning progress. When True, ml_weight increases as the
            model demonstrates improvement.
        initial_ml_weight: Starting ML weight (overrides MLSelectionConfig
            when adaptive_ml_weight is True).
        max_ml_weight: Maximum ML weight during adaptive adjustment.
        ml_weight_increase_rate: How quickly to increase ML weight when
            model improves (per successful update).
        ml_weight_decrease_rate: How quickly to decrease ML weight on
            rollback (per failed update).
        min_given_before_ml: Minimum given clauses before ML selection
            activates. Allows traditional selection to establish a baseline.
        log_integration_events: Log integration events for debugging.
    """

    enabled: bool = True
    collect_experiences: bool = True
    trigger_updates: bool = True
    track_proof_progress: bool = True
    invalidate_cache_on_update: bool = True
    adaptive_ml_weight: bool = True
    initial_ml_weight: float = 0.1
    max_ml_weight: float = 0.5
    ml_weight_increase_rate: float = 0.05
    ml_weight_decrease_rate: float = 0.1
    min_given_before_ml: int = 50
    log_integration_events: bool = False


# ── Learning Trigger Policy ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TriggerPolicyConfig:
    """Configuration for the adaptive learning trigger policy.

    Attributes:
        base_interval: Starting number of experiences between updates.
        min_interval: Minimum interval (fastest update rate).
        max_interval: Maximum interval (slowest update rate).
        productivity_target: Target productivity rate. When actual rate
            diverges from this, the trigger interval shortens.
        backoff_factor: Multiplicative increase to interval after a
            rollback (slows down when updates hurt).
        speedup_factor: Multiplicative decrease to interval after a
            successful update (speeds up when updates help).
        cooldown_after_rollback: Number of given clauses to skip after
            a rollback before allowing another update attempt.
        max_update_time_fraction: Maximum fraction of wall-clock time
            that model updates should consume (resource awareness).
        stagnation_threshold: Number of given clauses without progress
            before aggressively shortening the trigger interval.
    """

    base_interval: int = 200
    min_interval: int = 50
    max_interval: int = 1000
    productivity_target: float = 0.3
    backoff_factor: float = 1.5
    speedup_factor: float = 0.8
    cooldown_after_rollback: int = 30
    max_update_time_fraction: float = 0.15
    stagnation_threshold: int = 100


@dataclass(slots=True)
class TriggerStats:
    """Statistics for trigger policy decisions."""

    triggers_fired: int = 0
    triggers_suppressed_cooldown: int = 0
    triggers_suppressed_resource: int = 0
    current_interval: int = 200
    total_update_time: float = 0.0
    total_search_time: float = 0.0
    last_trigger_given: int = 0
    consecutive_rollbacks: int = 0
    consecutive_accepts: int = 0

    def update_time_fraction(self) -> float:
        """Fraction of wall time spent on model updates."""
        if self.total_search_time <= 0:
            return 0.0
        return self.total_update_time / self.total_search_time


class LearningTriggerPolicy:
    """Adaptive policy for deciding when to trigger model updates.

    The simple fixed-interval approach in OnlineLearningManager works but
    is suboptimal: it updates too often when the model is already good
    (wasting compute) and too rarely when the search is stagnating (missing
    learning opportunities).

    This policy adapts the trigger interval based on:

    1. **Learning effectiveness** — successful updates shorten the interval
       (the model is learning useful patterns), rollbacks lengthen it
       (the data distribution isn't ready yet).

    2. **Search stagnation** — when the search goes many given clauses
       without keeping new clauses, the interval shortens aggressively
       to try to break the deadlock.

    3. **Resource awareness** — if model updates are consuming too much
       wall-clock time relative to search, the interval lengthens to
       maintain search throughput.

    4. **Cooldown after rollback** — after a failed update, a cooldown
       period prevents thrashing.
    """

    __slots__ = (
        "_config", "_stats", "_current_interval",
        "_cooldown_remaining", "_search_start_time",
    )

    def __init__(self, config: TriggerPolicyConfig | None = None):
        self._config = config or TriggerPolicyConfig()
        self._current_interval = self._config.base_interval
        self._stats = TriggerStats(current_interval=self._current_interval)
        self._cooldown_remaining: int = 0
        self._search_start_time: float = time.monotonic()

    @property
    def stats(self) -> TriggerStats:
        return self._stats

    @property
    def current_interval(self) -> int:
        return self._current_interval

    def should_trigger(
        self,
        examples_since_update: int,
        given_count: int,
        progress_signals: ProofProgressSignals,
    ) -> bool:
        """Decide whether to trigger a model update now.

        Args:
            examples_since_update: Experiences accumulated since last update.
            given_count: Total given clauses processed so far.
            progress_signals: Current search progress signals.

        Returns:
            True if a model update should be triggered.
        """
        # Update elapsed search time
        self._stats.total_search_time = time.monotonic() - self._search_start_time

        # Cooldown: suppress triggers for N given clauses after rollback
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if examples_since_update >= self._current_interval:
                self._stats.triggers_suppressed_cooldown += 1
            return False

        # Resource awareness: if updates consume too much time, back off.
        # Only check when we have at least 1s of search time to avoid
        # division-by-near-zero at the start of search.
        if (
            self._stats.total_search_time >= 1.0
            and self._stats.update_time_fraction() > self._config.max_update_time_fraction
        ):
            if examples_since_update >= self._current_interval:
                self._stats.triggers_suppressed_resource += 1
                # Lengthen interval to reduce update frequency
                self._current_interval = min(
                    int(self._current_interval * 1.2),
                    self._config.max_interval,
                )
                self._stats.current_interval = self._current_interval
            return False

        # Stagnation override: if search is stuck, trigger more aggressively
        if progress_signals.given_since_last_progress >= self._config.stagnation_threshold:
            # Use half the current interval when stagnating
            effective_interval = max(
                self._config.min_interval,
                self._current_interval // 2,
            )
            return examples_since_update >= effective_interval

        # Normal interval check
        return examples_since_update >= self._current_interval

    def on_update_accepted(self, update_duration: float) -> None:
        """Called after a model update is accepted.

        Shortens the trigger interval since the model is learning.
        """
        self._stats.triggers_fired += 1
        self._stats.total_update_time += update_duration
        self._stats.consecutive_accepts += 1
        self._stats.consecutive_rollbacks = 0

        # Speed up: learning is working, try to capture more signal
        self._current_interval = max(
            self._config.min_interval,
            int(self._current_interval * self._config.speedup_factor),
        )
        self._stats.current_interval = self._current_interval

    def on_update_rolled_back(self, update_duration: float) -> None:
        """Called after a model update is rolled back.

        Lengthens the trigger interval and activates cooldown.
        """
        self._stats.triggers_fired += 1
        self._stats.total_update_time += update_duration
        self._stats.consecutive_rollbacks += 1
        self._stats.consecutive_accepts = 0

        # Back off: updates are hurting, wait longer
        self._current_interval = min(
            self._config.max_interval,
            int(self._current_interval * self._config.backoff_factor),
        )
        self._stats.current_interval = self._current_interval

        # Cooldown scales with consecutive rollbacks (max 3x base)
        self._cooldown_remaining = min(
            self._config.cooldown_after_rollback * self._stats.consecutive_rollbacks,
            self._config.cooldown_after_rollback * 3,
        )

    def on_search_start(self) -> None:
        """Reset timing at the start of a search run."""
        self._search_start_time = time.monotonic()


# ── Proof Progress Tracker ────────────────────────────────────────────────


@dataclass(slots=True)
class ProofProgressSignals:
    """Real-time signals indicating progress toward proof.

    These signals are derived from search behavior and used to
    provide feedback to the online learning system about which
    clause selections are productive.
    """

    unit_clauses_generated: int = 0
    empty_clause_attempts: int = 0
    subsumption_rate: float = 0.0
    avg_clause_weight_trend: float = 0.0
    productive_inference_rate: float = 0.0
    given_since_last_progress: int = 0
    proof_found: bool = False


class ProofProgressTracker:
    """Tracks search progress and identifies productive clause selections.

    Monitors the search loop for signals that indicate whether the prover
    is making progress toward finding a proof. These signals feed back into
    the online learning system to label clause selections as productive or
    unproductive.

    Tracked signals:
    - Unit clause generation rate (more units → closer to proof)
    - Clause weight trends (decreasing weights → simplification progress)
    - Inference productivity (kept/generated ratio)
    - Resolution participation (clauses used in multiple inferences)
    - Proof path contribution (retroactive labeling on proof discovery)
    """

    __slots__ = (
        "_window_size", "_recent_weights", "_recent_kept",
        "_recent_generated", "_unit_count", "_given_count",
        "_last_progress_given", "_given_clause_usage",
        "_clause_inference_count", "_max_tracked_clauses",
    )

    def __init__(self, window_size: int = 100, max_tracked_clauses: int = 10_000):
        self._window_size = window_size
        self._max_tracked_clauses = max_tracked_clauses
        self._recent_weights: list[float] = []
        self._recent_kept: int = 0
        self._recent_generated: int = 0
        self._unit_count: int = 0
        self._given_count: int = 0
        self._last_progress_given: int = 0
        self._given_clause_usage: dict[int, int] = {}
        self._clause_inference_count: dict[int, int] = {}

    def on_clause_generated(self, clause: Clause) -> None:
        """Record a newly generated clause."""
        self._recent_generated += 1
        if clause.is_unit:
            self._unit_count += 1

    def on_clause_kept(self, clause: Clause) -> None:
        """Record that a generated clause was kept."""
        self._recent_kept += 1
        self._recent_weights.append(clause.weight)
        if len(self._recent_weights) > self._window_size:
            self._recent_weights.pop(0)
        self._last_progress_given = self._given_count

    def on_clause_deleted(self, clause: Clause, reason: str) -> None:
        """Record that a clause was deleted (subsumption, weight, etc.)."""
        pass  # tracked implicitly via kept/generated ratio

    def on_given_selected(self, given: Clause) -> None:
        """Record that a clause was selected as given."""
        self._given_count += 1
        self._given_clause_usage[given.id] = 0
        # Prune oldest entries when dict exceeds bound
        if len(self._given_clause_usage) > self._max_tracked_clauses:
            self._prune_tracking_dicts()

    def on_inference_from_given(self, given_id: int, child: Clause) -> None:
        """Record that the given clause produced a child inference."""
        count = self._given_clause_usage.get(given_id, 0)
        self._given_clause_usage[given_id] = count + 1
        self._clause_inference_count[child.id] = (
            self._clause_inference_count.get(child.id, 0) + 1
        )

    def _prune_tracking_dicts(self) -> None:
        """Prune tracking dicts to bound memory in long searches.

        Keeps the most recent half of entries (by insertion order,
        which corresponds to clause ID order for given clauses).
        """
        keep = self._max_tracked_clauses // 2
        if len(self._given_clause_usage) > keep:
            # dict preserves insertion order in Python 3.7+
            keys = list(self._given_clause_usage.keys())
            for k in keys[:-keep]:
                del self._given_clause_usage[k]
        if len(self._clause_inference_count) > self._max_tracked_clauses:
            keys = list(self._clause_inference_count.keys())
            for k in keys[:-keep]:
                del self._clause_inference_count[k]

    def get_signals(self) -> ProofProgressSignals:
        """Compute current progress signals."""
        # Weight trend: negative slope = weights decreasing (good)
        weight_trend = 0.0
        if len(self._recent_weights) >= 10:
            first_half = self._recent_weights[: len(self._recent_weights) // 2]
            second_half = self._recent_weights[len(self._recent_weights) // 2 :]
            avg_first = sum(first_half) / len(first_half) if first_half else 0
            avg_second = sum(second_half) / len(second_half) if second_half else 0
            weight_trend = avg_first - avg_second  # positive = improving

        # Productivity rate
        prod_rate = 0.0
        if self._recent_generated > 0:
            prod_rate = self._recent_kept / self._recent_generated

        return ProofProgressSignals(
            unit_clauses_generated=self._unit_count,
            avg_clause_weight_trend=weight_trend,
            productive_inference_rate=prod_rate,
            given_since_last_progress=(
                self._given_count - self._last_progress_given
            ),
        )

    def given_clause_productivity(self, clause_id: int) -> float:
        """Return how productive a given clause was (0 to 1 scale).

        Based on how many kept inferences it participated in relative
        to average.
        """
        usage = self._given_clause_usage.get(clause_id, 0)
        if not self._given_clause_usage:
            return 0.5
        avg = sum(self._given_clause_usage.values()) / len(self._given_clause_usage)
        if avg == 0:
            return 0.5
        # Normalize to 0-1 with sigmoid-like mapping
        ratio = usage / avg
        return min(1.0, ratio / 2.0)

    def reset_window(self) -> None:
        """Reset the sliding window counters."""
        self._recent_kept = 0
        self._recent_generated = 0
        self._recent_weights.clear()


# ── Integration Statistics ────────────────────────────────────────────────


@dataclass(slots=True)
class OnlineIntegrationStats:
    """Statistics for the online learning integration."""

    experiences_collected: int = 0
    model_updates_triggered: int = 0
    model_updates_accepted: int = 0
    model_updates_rolled_back: int = 0
    cache_invalidations: int = 0
    ml_weight_adjustments: int = 0
    current_ml_weight: float = 0.0
    fallbacks_to_traditional: int = 0

    def report(self, trigger_stats: TriggerStats | None = None) -> str:
        parts = [
            f"OnlineIntegration: "
            f"experiences={self.experiences_collected}, "
            f"updates={self.model_updates_triggered} "
            f"(accepted={self.model_updates_accepted}, "
            f"rolled_back={self.model_updates_rolled_back}), "
            f"ml_weight={self.current_ml_weight:.3f}",
        ]
        if trigger_stats is not None:
            parts.append(
                f"trigger_interval={trigger_stats.current_interval}, "
                f"suppressed={trigger_stats.triggers_suppressed_cooldown}"
                f"+{trigger_stats.triggers_suppressed_resource}, "
                f"update_time_frac={trigger_stats.update_time_fraction():.2%}"
            )
        return " | ".join(parts)


# ── Main Integration Class ───────────────────────────────────────────────


class OnlineSearchIntegration:
    """Bridges the search loop with online learning components.

    This is the central coordinator that:
    1. Hooks into GivenClauseSearch events to collect experiences
    2. Triggers model updates at appropriate intervals
    3. Invalidates caches after model changes
    4. Adapts ML selection weight based on learning progress
    5. Provides proof progress tracking for feedback signals

    The integration wraps the search loop non-intrusively — the search
    engine itself is unmodified. Instead, this class intercepts events
    by wrapping the search's internal methods.
    """

    __slots__ = (
        "_config", "_manager", "_provider", "_progress_tracker",
        "_stats", "_current_given", "_enabled", "_trigger_policy",
    )

    def __init__(
        self,
        config: OnlineIntegrationConfig | None = None,
        manager: OnlineLearningManager | None = None,
        provider: GNNEmbeddingProvider | None = None,
        trigger_policy: LearningTriggerPolicy | None = None,
    ):
        self._config = config or OnlineIntegrationConfig()
        self._manager = manager
        self._provider = provider
        self._progress_tracker = ProofProgressTracker()

        # Show online learning startup configuration
        if self._config.enabled and self._manager is not None:
            buffer_cap = self._manager._buffer._capacity
            min_update = self._manager._config.min_examples_for_update
            adaptive_msg = ""
            if self._config.adaptive_ml_weight:
                adaptive_msg = f" (adaptive {self._config.initial_ml_weight:.2f}→{self._config.max_ml_weight:.2f})"

            print(f"📚 Online learning initialized: buffer capacity {buffer_cap}, "
                  f"min update threshold {min_update}{adaptive_msg}")
        elif self._config.enabled:
            print("⚠️  Online learning enabled but no ML manager available")
        self._stats = OnlineIntegrationStats()
        self._stats.current_ml_weight = self._config.initial_ml_weight
        self._current_given: Clause | None = None
        self._enabled = self._config.enabled and _ML_AVAILABLE
        self._trigger_policy = trigger_policy or LearningTriggerPolicy()

    @property
    def config(self) -> OnlineIntegrationConfig:
        return self._config

    @property
    def stats(self) -> OnlineIntegrationStats:
        return self._stats

    @property
    def progress_tracker(self) -> ProofProgressTracker:
        return self._progress_tracker

    @property
    def manager(self) -> OnlineLearningManager | None:
        return self._manager

    @property
    def trigger_policy(self) -> LearningTriggerPolicy:
        return self._trigger_policy

    # ── Search Event Hooks ─────────────────────────────────────────────

    def on_given_selected(self, given: Clause, selection_type: str) -> None:
        """Called when a given clause is selected from SOS."""
        if not self._enabled:
            return
        self._current_given = given
        self._progress_tracker.on_given_selected(given)

        # Log experience collection progress
        if self._manager is not None:
            buffer_size = self._manager._buffer.size
            max_size = self._manager._buffer._capacity
            if buffer_size > 0 and buffer_size % 25 == 0:  # Every 25 experiences
                print(f"📊 Learning progress: {buffer_size}/{max_size} experiences collected")

    def on_clause_generated(
        self,
        child: Clause,
        given: Clause,
        partner: Clause | None = None,
    ) -> None:
        """Called when an inference generates a new clause (before processing)."""
        if not self._enabled:
            return
        self._progress_tracker.on_clause_generated(child)

    def on_clause_kept(
        self,
        child: Clause,
        given: Clause | None = None,
        partner: Clause | None = None,
    ) -> None:
        """Called when a generated clause passes all deletion checks and is kept.

        Note: KEPT clauses are no longer added to the experience buffer.
        Only subsumption events (back/forward) provide positive training signal.
        """
        if not self._enabled or self._manager is None:
            return

        self._progress_tracker.on_clause_kept(child)

    def on_clause_deleted(
        self,
        child: Clause,
        reason: OutcomeType,
        given: Clause | None = None,
        partner: Clause | None = None,
    ) -> None:
        """Called when a generated clause is deleted (subsumed, tautology, etc.)."""
        if not self._enabled or self._manager is None:
            return

        self._progress_tracker.on_clause_deleted(child, reason.name)

        if self._config.collect_experiences:
            g = given or self._current_given
            if g is not None:
                outcome = InferenceOutcome(
                    given_clause=g,
                    partner_clause=partner,
                    child_clause=child,
                    outcome=reason,
                    timestamp=time.monotonic(),
                    given_count=self._progress_tracker._given_count,
                )
                self._manager.record_outcome(outcome)
                self._stats.experiences_collected += 1

    def on_proof_found(self, proof_clause_ids: set[int]) -> None:
        """Called when a proof is discovered."""
        if not self._enabled or self._manager is None:
            return

        self._progress_tracker._last_progress_given = (
            self._progress_tracker._given_count
        )
        self._manager.on_proof_found(proof_clause_ids)

        if self._config.log_integration_events:
            logger.info(
                "Proof found — %d clauses in derivation, "
                "%d experiences collected so far",
                len(proof_clause_ids),
                self._stats.experiences_collected,
            )

    def on_back_subsumption(self, subsuming_clause: Clause, subsumed_clause: Clause) -> None:
        """Called when a clause back-subsumes another clause.

        This indicates that the subsuming clause has a particularly useful
        structure - it's general enough to subsume an existing clause.
        We use this as positive feedback for online learning.
        """
        if not self._enabled or self._manager is None:
            return

        # Record the back-subsumption as a positive signal
        if self._config.collect_experiences:
            outcome = InferenceOutcome(
                given_clause=subsuming_clause,
                partner_clause=subsumed_clause,
                child_clause=subsuming_clause,
                outcome=OutcomeType.SUBSUMER,
                timestamp=time.monotonic(),
                given_count=self._progress_tracker._given_count,
            )

            # Record twice to give back-subsumption extra positive weight
            self._manager.record_outcome(outcome)
            self._manager.record_outcome(outcome)
            self._stats.experiences_collected += 2

            if self._config.log_integration_events:
                logger.info(
                    "Back-subsumption: clause %d subsumed clause %d → positive ML feedback",
                    subsuming_clause.id,
                    subsumed_clause.id,
                )

    def on_forward_subsumption(self, subsuming_clause: Clause, subsumed_clause: Clause) -> None:
        """Called when an existing clause forward-subsumes a new clause.

        The subsuming clause has demonstrated superior generality by
        subsuming a newly generated clause before it could be kept.
        We use this as positive feedback for online learning.
        """
        if not self._enabled or self._manager is None:
            return

        # Record the forward-subsumption as a positive signal
        if self._config.collect_experiences:
            outcome = InferenceOutcome(
                given_clause=subsuming_clause,
                partner_clause=subsumed_clause,
                child_clause=subsuming_clause,
                outcome=OutcomeType.SUBSUMER,
                timestamp=time.monotonic(),
                given_count=self._progress_tracker._given_count,
            )

            # Record twice to give forward-subsumption extra positive weight
            self._manager.record_outcome(outcome)
            self._manager.record_outcome(outcome)
            self._stats.experiences_collected += 2

            if self._config.log_integration_events:
                logger.info(
                    "Forward-subsumption: clause %d subsumed clause %d → positive ML feedback",
                    subsuming_clause.id,
                    subsumed_clause.id,
                )

    def on_inferences_complete(self) -> None:
        """Called after all inferences for a given clause are processed.

        This is the trigger point for model updates — after each batch
        of inferences is complete, we check if enough experience has
        accumulated and potentially update the model.

        Uses the adaptive LearningTriggerPolicy to decide whether to
        update, considering learning effectiveness, search stagnation,
        and resource consumption.
        """
        if not self._enabled or self._manager is None:
            return

        if not self._config.trigger_updates:
            return

        # Check minimum given threshold before allowing updates
        given_count = self._progress_tracker._given_count
        if given_count < self._config.min_given_before_ml:
            return

        # Minimum buffer size check (delegate interval decision to policy)
        if self._manager._buffer.size < self._manager._config.min_examples_for_update:
            return

        # Use the adaptive trigger policy instead of fixed interval
        signals = self._progress_tracker.get_signals()
        experiences_since_update = self._manager._examples_since_update
        buffer_size = self._manager._buffer.size

        should_update = self._trigger_policy.should_trigger(
            examples_since_update=experiences_since_update,
            given_count=given_count,
            progress_signals=signals,
        )

        # Show learning trigger evaluation every 10 given clauses
        if given_count % 10 == 0 and experiences_since_update > 0:
            status = "TRIGGERING" if should_update else "waiting"
            print(f"📈 Learning eval: {experiences_since_update} new experiences, "
                  f"{buffer_size} total → {status} (given #{given_count})")

        if should_update:
            self._trigger_model_update()

    # ── Model Update Logic ─────────────────────────────────────────────

    def _trigger_model_update(self) -> None:
        """Execute a model update and handle the consequences.

        Measures update duration and reports it to the trigger policy
        for resource-aware scheduling.
        """
        if self._manager is None:
            return

        self._stats.model_updates_triggered += 1

        t0 = time.monotonic()
        accepted = self._manager.update()
        update_duration = time.monotonic() - t0

        if accepted:
            self._stats.model_updates_accepted += 1
            self._trigger_policy.on_update_accepted(update_duration)

            # Hot-swap updated weights into the embedding provider
            if self._config.invalidate_cache_on_update and self._provider is not None:
                self._provider.swap_weights(self._manager._encoder.state_dict())
                self._stats.cache_invalidations += 1

            # Increase ML weight if adaptive
            old_weight = self._stats.current_ml_weight
            if self._config.adaptive_ml_weight:
                self._adjust_ml_weight(increase=True)

            # Always show successful model updates (not just in debug mode)
            buffer_size = self._manager._buffer.size if self._manager else 0
            print(f"🧠 Model update #{self._stats.model_updates_triggered} ✅ accepted "
                  f"({update_duration:.2f}s) | "
                  f"ML weight: {old_weight:.2f}→{self._stats.current_ml_weight:.2f} | "
                  f"Buffer: {buffer_size} experiences")

            if self._config.log_integration_events:
                logger.info(
                    "Model update #%d accepted (%.3fs), interval=%d, "
                    "ml_weight=%.3f",
                    self._stats.model_updates_triggered,
                    update_duration,
                    self._trigger_policy.current_interval,
                    self._stats.current_ml_weight,
                )
        else:
            self._stats.model_updates_rolled_back += 1
            self._trigger_policy.on_update_rolled_back(update_duration)

            # Decrease ML weight on rollback
            old_weight = self._stats.current_ml_weight
            if self._config.adaptive_ml_weight:
                self._adjust_ml_weight(increase=False)

            # Always show rollbacks (learning failures)
            print(f"🧠 Model update #{self._stats.model_updates_triggered} ❌ rolled back "
                  f"({update_duration:.2f}s) | "
                  f"ML weight: {old_weight:.2f}→{self._stats.current_ml_weight:.2f} | "
                  f"Cooldown: {self._trigger_policy._cooldown_remaining}")

            if self._config.log_integration_events:
                logger.info(
                    "Model update #%d rolled back (%.3fs), interval=%d, "
                    "cooldown=%d, ml_weight=%.3f",
                    self._stats.model_updates_triggered,
                    update_duration,
                    self._trigger_policy.current_interval,
                    self._trigger_policy._cooldown_remaining,
                    self._stats.current_ml_weight,
                )

    def _adjust_ml_weight(self, increase: bool) -> None:
        """Adjust the ML selection weight based on learning progress."""
        current = self._stats.current_ml_weight

        if increase:
            new_weight = min(
                current + self._config.ml_weight_increase_rate,
                self._config.max_ml_weight,
            )
        else:
            new_weight = max(
                current - self._config.ml_weight_decrease_rate,
                self._config.initial_ml_weight,
            )

        if new_weight != current:
            self._stats.current_ml_weight = new_weight
            self._stats.ml_weight_adjustments += 1

    def get_current_ml_weight(self) -> float:
        """Return the current adaptive ML selection weight."""
        if not self._enabled or not self._config.adaptive_ml_weight:
            return self._config.initial_ml_weight

        # Don't use ML until minimum given threshold
        if self._progress_tracker._given_count < self._config.min_given_before_ml:
            return 0.0

        return self._stats.current_ml_weight

    # ── Convenience: Run Search with Learning ─────────────────────────

    def create_search(
        self,
        options: object | None = None,
        selection: object | None = None,
        symbol_table: object | None = None,
    ) -> OnlineLearningSearch:
        """Create a GivenClauseSearch with online learning hooks built in.

        Returns an ``OnlineLearningSearch`` instance (a subclass of
        ``GivenClauseSearch``) with all hooks pre-installed.

        This is the primary integration point. The returned search object
        behaves identically to a normal GivenClauseSearch but additionally
        collects experiences, triggers model updates, and adapts ML weights.

        Args:
            options: SearchOptions (passed through to GivenClauseSearch).
            selection: GivenSelection or EmbeddingEnhancedSelection.
            symbol_table: SymbolTable for the problem.

        Returns:
            An OnlineLearningSearch ready for ``run()``.
        """
        from pyladr.search.given_clause import SearchOptions
        from pyladr.search.selection import GivenSelection
        from pyladr.core.symbol import SymbolTable as ST

        return OnlineLearningSearch(
            integration=self,
            options=options or SearchOptions(),
            selection=selection or GivenSelection(),
            symbol_table=symbol_table or ST(),
        )

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        embedding_provider: GNNEmbeddingProvider | None = None,
        config: OnlineIntegrationConfig | None = None,
        learning_config: OnlineLearningConfig | None = None,
        trigger_config: TriggerPolicyConfig | None = None,
    ) -> OnlineSearchIntegration:
        """Create a fully wired OnlineSearchIntegration.

        Args:
            embedding_provider: The GNN embedding provider (needed for
                cache invalidation and model access).
            config: Integration configuration.
            learning_config: Online learning configuration.
            trigger_config: Adaptive trigger policy configuration.

        Returns:
            Ready-to-use integration instance.
        """
        cfg = config or OnlineIntegrationConfig()

        if not _ML_AVAILABLE or not cfg.enabled:
            return cls(config=cfg)

        # Build trigger policy — use learning config's update_interval
        # as the base interval if no trigger config is provided
        if trigger_config is None:
            lcfg = learning_config or _OLConfig()
            trigger_config = TriggerPolicyConfig(
                base_interval=lcfg.update_interval,
            )
        trigger_policy = LearningTriggerPolicy(trigger_config)

        manager = None
        if embedding_provider is not None:
            try:
                from pyladr.ml.embedding_provider import GNNClauseEncoder

                # Wrap the provider in a ClauseEncoder adapter so that
                # OnlineLearningManager.encode_clauses() works correctly
                # (raw HeterogeneousClauseGNN only has embed_clause()).
                encoder = GNNClauseEncoder(embedding_provider)

                learning_cfg = learning_config or _OLConfig()
                manager = _OLManager(
                    encoder=encoder,
                    config=learning_cfg,
                )
            except Exception:
                logger.warning(
                    "Failed to create OnlineLearningManager",
                    exc_info=True,
                )

        return cls(
            config=cfg,
            manager=manager,
            provider=embedding_provider,
            trigger_policy=trigger_policy,
        )


# ── GivenClauseSearch subclass with online learning hooks ────────────────


class OnlineLearningSearch:
    """GivenClauseSearch with integrated online learning hooks.

    This is a composition wrapper (not a subclass) that delegates to a
    GivenClauseSearch internally while intercepting key events to feed
    the online learning system. Uses composition because GivenClauseSearch
    uses __slots__, preventing monkey-patching.

    The wrapper intercepts:
    - ``_keep_clause`` → records KEPT outcomes
    - ``_should_delete`` → records deletion outcomes
    - ``_handle_proof`` → notifies proof discovery
    - ``_make_inferences`` → triggers model updates after each batch

    All other behavior is unchanged from the base GivenClauseSearch.
    """

    __slots__ = ("_search", "_integration")

    def __init__(
        self,
        integration: OnlineSearchIntegration,
        options: object | None = None,
        selection: object | None = None,
        symbol_table: object | None = None,
    ):
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions
        from pyladr.search.selection import GivenSelection
        from pyladr.core.symbol import SymbolTable as ST

        self._integration = integration
        self._search = _OnlineLearningGivenClauseSearch(
            integration=integration,
            options=options or SearchOptions(),
            selection=selection or GivenSelection(),
            symbol_table=symbol_table or ST(),
        )

    def run(
        self,
        usable: list[Clause] | None = None,
        sos: list[Clause] | None = None,
    ) -> object:
        """Run search with online learning integration."""
        print("🚀 Starting proof search with online contrastive learning hooks active")
        return self._search.run(usable, sos)

    @property
    def state(self) -> object:
        return self._search.state

    @property
    def stats(self) -> object:
        return self._search.stats

    @property
    def integration(self) -> OnlineSearchIntegration:
        return self._integration

    def set_proof_callback(self, callback):
        """Delegate proof callback to the internal search engine."""
        if hasattr(self._search, 'set_proof_callback'):
            self._search.set_proof_callback(callback)


class _OnlineLearningGivenClauseSearch:
    """Internal subclass of GivenClauseSearch with hooks for online learning.

    Overrides key methods to intercept search events and feed them into
    the OnlineSearchIntegration.
    """

    def __new__(cls, *args, **kwargs):
        """Dynamically create a subclass of GivenClauseSearch."""
        from pyladr.search.given_clause import GivenClauseSearch

        # We can't use normal inheritance because of __slots__,
        # so we create an instance of GivenClauseSearch and bind
        # our hooked methods to it.
        integration = kwargs.pop("integration")

        instance = GivenClauseSearch.__new__(GivenClauseSearch)
        GivenClauseSearch.__init__(instance, **kwargs)

        # Store integration reference outside __slots__ via closure
        _original_make_inferences = GivenClauseSearch._make_inferences
        _original_keep_clause = GivenClauseSearch._keep_clause
        _original_should_delete = GivenClauseSearch._should_delete
        _original_handle_proof = GivenClauseSearch._handle_proof

        class HookedSearch(GivenClauseSearch):
            """Subclass with online learning hooks."""

            # No additional slots needed — we use the parent's slots
            # and capture `integration` via closure.

            def _make_inferences(self) -> object:
                # Capture given count before to detect new given selection
                given_before = self._state.stats.given
                result = _original_make_inferences(self)
                # If a new given was selected, notify integration
                if self._state.stats.given > given_before:
                    # Given clause was appended to usable's deque during selection
                    usable_deque = self._state.usable._clauses
                    if usable_deque:
                        last_given = usable_deque[-1]
                        integration.on_given_selected(last_given, "")
                integration.on_inferences_complete()
                return result

            def _keep_clause(self, c: Clause) -> object:
                result = _original_keep_clause(self, c)
                integration.on_clause_kept(c)
                return result

            def _should_delete(self, c: Clause) -> bool:
                deleted = _original_should_delete(self, c)
                if deleted:
                    integration.on_clause_deleted(
                        c, OutcomeType.SUBSUMED,
                    )
                return deleted

            def _handle_proof(self, empty: Clause) -> object:
                result = _original_handle_proof(self, empty)
                proof_ids = set()
                if self._proofs:
                    latest = self._proofs[-1]
                    proof_ids = {c.id for c in latest.clauses}
                integration.on_proof_found(proof_ids)
                return result

        # Create and return the hooked instance
        hooked = HookedSearch.__new__(HookedSearch)
        GivenClauseSearch.__init__(hooked, **kwargs)

        # Set up subsumption callbacks on the final hooked instance
        if hasattr(hooked, 'set_back_subsumption_callback'):
            hooked.set_back_subsumption_callback(integration.on_back_subsumption)
        if hasattr(hooked, 'set_forward_subsumption_callback'):
            hooked.set_forward_subsumption_callback(integration.on_forward_subsumption)

        return hooked
