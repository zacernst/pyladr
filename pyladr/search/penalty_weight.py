"""Penalty-based clause weight adjustment for search prioritization.

When a clause's combined penalty (from generality analysis and/or penalty
propagation) exceeds a threshold, its weight is increased to deprioritize
it during selection. This pushes overly general or structurally redundant
clauses further down the selection queue without discarding them entirely.

Three adjustment modes:
  - linear:      adjusted = base + multiplier * penalty
  - exponential: adjusted = base * multiplier^(penalty / threshold)
  - step:        adjusted = base * multiplier  (flat boost when over threshold)

The adjustment is applied after default_clause_weight() in _should_delete(),
replacing the bare weight with a penalty-aware weight. When disabled
(the default), weight calculation is unchanged, preserving C Prover9
compatibility.

Design:
    - Disabled by default (opt-in via SearchOptions.penalty_weight_enabled)
    - Composable with penalty_propagation and repetition_penalty
    - O(1) per clause (reads cached penalty, applies formula)
    - Max weight cap prevents unbounded inflation
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyladr.core.clause import Clause


class PenaltyWeightMode(Enum):
    """How penalty translates into weight increase."""

    LINEAR = auto()       # adjusted = base + multiplier * penalty
    EXPONENTIAL = auto()  # adjusted = base * multiplier^(penalty / threshold)
    STEP = auto()         # adjusted = base * multiplier (flat boost)


@dataclass(frozen=True, slots=True)
class PenaltyWeightConfig:
    """Configuration for penalty-based weight adjustment.

    All fields have safe defaults that disable or minimize the feature
    when not explicitly configured.
    """

    enabled: bool = False
    threshold: float = 5.0           # Only adjust if penalty >= threshold
    multiplier: float = 2.0          # Weight increase factor (>= 1.0)
    max_adjusted_weight: float = 1000.0  # Cap on adjusted weight
    mode: PenaltyWeightMode = PenaltyWeightMode.EXPONENTIAL


def penalty_adjusted_weight(
    base_weight: float,
    penalty: float,
    config: PenaltyWeightConfig,
) -> float:
    """Compute penalty-adjusted clause weight.

    If the penalty is below the threshold, returns the base weight unchanged.
    Otherwise, applies the configured adjustment mode.

    Args:
        base_weight: The clause's default symbol-count weight.
        penalty: Combined penalty score (from generality + propagation).
        config: Weight adjustment configuration.

    Returns:
        Adjusted weight, capped at config.max_adjusted_weight.
    """
    if not config.enabled or penalty < config.threshold:
        return base_weight

    mode = config.mode
    multiplier = config.multiplier

    if mode == PenaltyWeightMode.LINEAR:
        adjusted = base_weight + multiplier * penalty
    elif mode == PenaltyWeightMode.EXPONENTIAL:
        # Exponent scales with how far penalty exceeds threshold
        exponent = penalty / config.threshold
        adjusted = base_weight * (multiplier ** exponent)
    elif mode == PenaltyWeightMode.STEP:
        adjusted = base_weight * multiplier
    else:
        # Fallback to linear
        adjusted = base_weight + multiplier * penalty

    return min(adjusted, config.max_adjusted_weight)
