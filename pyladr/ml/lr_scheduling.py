"""Adaptive learning rate scheduling for online contrastive learning.

Provides multiple scheduling strategies optimized for real-time model
updates during theorem proving search. Integrates with
OnlineLearningManager to adjust learning rates based on search
performance indicators.

Strategies:
- ExponentialDecay: conservative fixed decay (good default)
- PerformanceBased: higher rates when search struggling, lower when productive
- ConvergenceAware: reduces rate as productivity stabilizes
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto


# ── Configuration ─────────────────────────────────────────────────────────


class SchedulerType(Enum):
    """Available learning rate scheduling strategies."""

    EXPONENTIAL_DECAY = auto()
    PERFORMANCE_BASED = auto()
    CONVERGENCE_AWARE = auto()


@dataclass(frozen=True, slots=True)
class LRSchedulerConfig:
    """Configuration for adaptive learning rate scheduling.

    Attributes:
        strategy: Which scheduling strategy to use.
        initial_lr: Starting learning rate.
        min_lr: Floor for learning rate (never goes below this).
        max_lr: Ceiling for learning rate (never goes above this).
        decay_factor: Multiplicative decay per update (ExponentialDecay).
        warmup_updates: Number of updates before decay begins.
        performance_window: Window size for productivity rate tracking.
        boost_factor: Multiplier when search is struggling (PerformanceBased).
        reduction_factor: Multiplier when search is productive (PerformanceBased).
        struggle_threshold: Productivity rate below which we boost LR.
        success_threshold: Productivity rate above which we reduce LR.
        convergence_patience: Updates of stable productivity before reducing LR.
        convergence_reduction: Factor to reduce LR on convergence detection.
    """

    strategy: SchedulerType = SchedulerType.CONVERGENCE_AWARE
    initial_lr: float = 1e-4
    min_lr: float = 1e-6
    max_lr: float = 5e-4
    decay_factor: float = 0.95
    warmup_updates: int = 3
    performance_window: int = 50
    boost_factor: float = 1.5
    reduction_factor: float = 0.7
    struggle_threshold: float = 0.2
    success_threshold: float = 0.6
    convergence_patience: int = 5
    convergence_reduction: float = 0.5


_DEFAULT_CONFIG = LRSchedulerConfig()


# ── Scheduler state ──────────────────────────────────────────────────────


@dataclass(slots=True)
class LRSchedulerState:
    """Tracks scheduler state for monitoring and debugging."""

    current_lr: float = 0.0
    update_count: int = 0
    adjustments: int = 0
    last_productivity: float = 0.0
    strategy_name: str = ""
    converged: bool = False

    def summary(self) -> str:
        return (
            f"LR={self.current_lr:.2e}, updates={self.update_count}, "
            f"adjustments={self.adjustments}, "
            f"productivity={self.last_productivity:.3f}, "
            f"strategy={self.strategy_name}"
        )


# ── Base class ────────────────────────────────────────────────────────────


class LRScheduler(ABC):
    """Abstract base for learning rate schedulers.

    Schedulers observe search performance and adjust the optimizer's
    learning rate between online updates. They are called once per
    update cycle (every update_interval outcomes).
    """

    __slots__ = ("_config", "_state", "_optimizer")

    def __init__(self, config: LRSchedulerConfig | None = None):
        self._config = config or _DEFAULT_CONFIG
        self._state = LRSchedulerState(
            current_lr=self._config.initial_lr,
            strategy_name=self.__class__.__name__,
        )
        self._optimizer = None

    @property
    def current_lr(self) -> float:
        return self._state.current_lr

    @property
    def state(self) -> LRSchedulerState:
        return self._state

    def attach_optimizer(self, optimizer) -> None:
        """Attach a PyTorch optimizer to control its learning rate."""
        self._optimizer = optimizer
        self._apply_lr(self._config.initial_lr)

    @abstractmethod
    def step(self, productivity_rate: float, update_loss: float) -> float:
        """Compute the new learning rate after an online update.

        Args:
            productivity_rate: Fraction of recent selections that were
                productive (kept or proof clauses). Range [0, 1].
            update_loss: Average contrastive loss from the last update.

        Returns:
            The new learning rate.
        """
        ...

    def _apply_lr(self, lr: float) -> None:
        """Clamp and apply the learning rate to the optimizer."""
        lr = max(self._config.min_lr, min(self._config.max_lr, lr))
        self._state.current_lr = lr
        if self._optimizer is not None:
            for pg in self._optimizer.param_groups:
                pg["lr"] = lr


# ── Exponential decay ────────────────────────────────────────────────────


class ExponentialDecayScheduler(LRScheduler):
    """Simple exponential decay with optional warmup.

    Good default for stable training. Starts at initial_lr and decays
    by decay_factor after each update (post-warmup).
    """

    def step(self, productivity_rate: float, update_loss: float) -> float:
        self._state.update_count += 1
        self._state.last_productivity = productivity_rate

        if self._state.update_count <= self._config.warmup_updates:
            # During warmup, linearly ramp up
            frac = self._state.update_count / self._config.warmup_updates
            new_lr = self._config.initial_lr * frac
        else:
            # Post-warmup: exponential decay
            steps_past_warmup = (
                self._state.update_count - self._config.warmup_updates
            )
            new_lr = self._config.initial_lr * (
                self._config.decay_factor ** steps_past_warmup
            )

        self._state.adjustments += 1
        self._apply_lr(new_lr)
        return self._state.current_lr


# ── Performance-based ────────────────────────────────────────────────────


class PerformanceBasedScheduler(LRScheduler):
    """Adjusts learning rate based on search productivity.

    When the search is struggling (low productivity), boosts the learning
    rate to encourage faster adaptation. When the search is productive,
    reduces the rate to avoid destabilizing a working model.
    """

    __slots__ = ("_config", "_state", "_optimizer", "_recent_rates")

    def __init__(self, config: LRSchedulerConfig | None = None):
        super().__init__(config)
        self._recent_rates: deque[float] = deque(
            maxlen=self._config.performance_window,
        )

    def step(self, productivity_rate: float, update_loss: float) -> float:
        self._state.update_count += 1
        self._state.last_productivity = productivity_rate
        self._recent_rates.append(productivity_rate)

        if self._state.update_count <= self._config.warmup_updates:
            frac = self._state.update_count / self._config.warmup_updates
            new_lr = self._config.initial_lr * frac
        else:
            avg_rate = (
                sum(self._recent_rates) / len(self._recent_rates)
                if self._recent_rates else productivity_rate
            )

            current = self._state.current_lr

            if avg_rate < self._config.struggle_threshold:
                # Struggling: boost learning rate to adapt faster
                new_lr = current * self._config.boost_factor
            elif avg_rate > self._config.success_threshold:
                # Productive: reduce to preserve stability
                new_lr = current * self._config.reduction_factor
            else:
                # Neutral zone: gentle decay
                new_lr = current * 0.98

        self._state.adjustments += 1
        self._apply_lr(new_lr)
        return self._state.current_lr


# ── Convergence-aware ────────────────────────────────────────────────────


class ConvergenceAwareScheduler(LRScheduler):
    """Reduces learning rate when productivity rate plateaus.

    Monitors the variance of recent productivity rates. When variance
    drops below a threshold for `convergence_patience` consecutive
    updates, reduces the learning rate. This prevents over-training
    on a problem whose patterns have already been learned.

    Also incorporates loss-based feedback: if loss suddenly spikes,
    temporarily boosts the rate to recover.
    """

    __slots__ = (
        "_config", "_state", "_optimizer",
        "_recent_rates", "_recent_losses",
        "_plateau_count",
    )

    def __init__(self, config: LRSchedulerConfig | None = None):
        super().__init__(config)
        self._recent_rates: deque[float] = deque(
            maxlen=self._config.performance_window,
        )
        self._recent_losses: deque[float] = deque(maxlen=10)
        self._plateau_count = 0

    def step(self, productivity_rate: float, update_loss: float) -> float:
        self._state.update_count += 1
        self._state.last_productivity = productivity_rate
        self._recent_rates.append(productivity_rate)
        self._recent_losses.append(update_loss)

        if self._state.update_count <= self._config.warmup_updates:
            frac = self._state.update_count / self._config.warmup_updates
            new_lr = self._config.initial_lr * frac
            self._apply_lr(new_lr)
            return self._state.current_lr

        current = self._state.current_lr

        # Check for loss spike (recovery mode)
        if len(self._recent_losses) >= 3:
            recent_avg = sum(list(self._recent_losses)[-3:]) / 3
            older_avg = sum(list(self._recent_losses)[:-3]) / max(1, len(self._recent_losses) - 3)
            if older_avg > 0 and recent_avg > 2.0 * older_avg:
                # Loss spike: boost to recover
                new_lr = current * self._config.boost_factor
                self._plateau_count = 0
                self._state.adjustments += 1
                self._apply_lr(new_lr)
                return self._state.current_lr

        # Check for convergence (plateau detection)
        if len(self._recent_rates) >= self._config.convergence_patience:
            window = list(self._recent_rates)[
                -self._config.convergence_patience:
            ]
            mean_rate = sum(window) / len(window)
            variance = sum((r - mean_rate) ** 2 for r in window) / len(window)

            if variance < 0.001:  # Very stable
                self._plateau_count += 1
                if self._plateau_count >= self._config.convergence_patience:
                    # Plateau detected: reduce learning rate
                    new_lr = current * self._config.convergence_reduction
                    self._plateau_count = 0
                    self._state.converged = True
                    self._state.adjustments += 1
                    self._apply_lr(new_lr)
                    return self._state.current_lr
            else:
                self._plateau_count = 0
                self._state.converged = False

        # Default: gentle exponential decay
        new_lr = current * self._config.decay_factor
        self._state.adjustments += 1
        self._apply_lr(new_lr)
        return self._state.current_lr


# ── Factory ──────────────────────────────────────────────────────────────


def create_lr_scheduler(
    config: LRSchedulerConfig | None = None,
) -> LRScheduler:
    """Create a learning rate scheduler from configuration.

    Args:
        config: Scheduler configuration. Uses defaults if None.

    Returns:
        An LRScheduler instance of the configured type.
    """
    cfg = config or _DEFAULT_CONFIG

    if cfg.strategy == SchedulerType.EXPONENTIAL_DECAY:
        return ExponentialDecayScheduler(cfg)
    elif cfg.strategy == SchedulerType.PERFORMANCE_BASED:
        return PerformanceBasedScheduler(cfg)
    elif cfg.strategy == SchedulerType.CONVERGENCE_AWARE:
        return ConvergenceAwareScheduler(cfg)
    else:
        raise ValueError(f"Unknown scheduler strategy: {cfg.strategy}")
