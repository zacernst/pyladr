"""Parent-to-child penalty propagation for overly general clause suppression.

When a clause has an overly general major premise (e.g., variables in
antecedents not appearing in consequents), its derived children inherit
a penalty. This discourages search explosion from "unifies with everything"
patterns like ¬P(x) ∨ Q resolved with P(t) → Q.

The penalty cache is a side structure on GivenClauseSearch, not a
modification to the frozen Clause dataclass. This preserves C Prover9
compatibility and immutable clause design.

Design:
    - Disabled by default (opt-in via SearchOptions.penalty_propagation)
    - Three combination modes: additive, multiplicative, max
    - Depth limiting to prevent unbounded accumulation
    - O(1) cache lookup per clause via dict[int, PenaltyRecord]
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyladr.core.clause import Clause, Justification

from pyladr.core.clause import JustType
from pyladr.search.selection import _clause_generality_penalty


class PenaltyCombineMode(Enum):
    """How to combine a clause's own penalty with inherited parent penalty."""

    ADDITIVE = auto()       # child = own + decay * parent_penalty
    MULTIPLICATIVE = auto() # child = own * (1 + decay * parent_penalty)
    MAX = auto()            # child = max(own, decay * parent_penalty)


@dataclass(frozen=True, slots=True)
class PenaltyPropagationConfig:
    """Configuration for penalty propagation.

    All fields have safe defaults that disable or minimize the feature
    when not explicitly configured.
    """

    enabled: bool = False
    mode: PenaltyCombineMode = PenaltyCombineMode.ADDITIVE
    decay: float = 0.5           # Decay factor per generation (0.0-1.0)
    threshold: float = 5.0       # Only propagate if parent penalty >= threshold
    max_depth: int = 3           # Max inheritance depth (0 = unlimited)
    max_penalty: float = 20.0    # Cap on accumulated penalty


@dataclass(slots=True)
class PenaltyRecord:
    """Cached penalty data for a single clause."""

    own_penalty: float        # Intrinsic generality penalty
    inherited_penalty: float  # Penalty inherited from parents
    combined_penalty: float   # Final combined penalty used for selection
    depth: int                # Inheritance depth (0 = no inheritance)


class PenaltyCache:
    """Side-structure cache mapping clause IDs to penalty records.

    Provides O(1) lookup for combined penalties. Populated during
    _cl_process() after simplification, before deletion checks.
    """

    __slots__ = ("_records", "_config")

    def __init__(self, config: PenaltyPropagationConfig) -> None:
        self._records: dict[int, PenaltyRecord] = {}
        self._config = config

    @property
    def config(self) -> PenaltyPropagationConfig:
        return self._config

    def get(self, clause_id: int) -> PenaltyRecord | None:
        """Get cached penalty record for a clause. O(1)."""
        return self._records.get(clause_id)

    def get_combined(self, clause_id: int) -> float:
        """Get combined penalty for a clause, or 0.0 if not cached."""
        rec = self._records.get(clause_id)
        return rec.combined_penalty if rec is not None else 0.0

    def put(self, clause_id: int, record: PenaltyRecord) -> None:
        """Store a penalty record. O(1)."""
        self._records[clause_id] = record

    def remove(self, clause_id: int) -> None:
        """Remove a clause from the cache (e.g., on back-subsumption)."""
        self._records.pop(clause_id, None)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, clause_id: int) -> bool:
        return clause_id in self._records


_EMPTY_IDS: tuple[int, ...] = ()


def extract_parent_ids(justification: tuple[Justification, ...]) -> tuple[int, ...]:
    """Extract parent clause IDs from a justification chain.

    Only examines the primary (first) justification step, which contains
    the inference rule that created the clause.

    Returns:
        Tuple of parent clause IDs (empty for INPUT/GOAL clauses).
    """
    if not justification:
        return _EMPTY_IDS

    # We only propagate from the primary parent (index 0); secondary parents
    # from simplification chains are excluded by design to avoid
    # double-penalizing via shared ancestors.
    just = justification[0]

    if just.just_type in (JustType.BINARY_RES, JustType.HYPER_RES, JustType.FACTOR):
        return just.clause_ids  # Already a tuple — zero-copy

    if just.just_type == JustType.PARA and just.para is not None:
        return (just.para.from_id, just.para.into_id)

    if just.just_type in (
        JustType.DEMOD, JustType.BACK_DEMOD, JustType.COPY, JustType.DENY,
    ) and just.clause_id > 0:
        return (just.clause_id,)

    # INPUT, GOAL, CLAUSIFY, etc. — no parents
    return _EMPTY_IDS


def compute_inherited_penalty(
    parent_ids: tuple[int, ...],
    cache: PenaltyCache,
    all_clauses: dict[int, Clause],
    config: PenaltyPropagationConfig,
) -> tuple[float, int]:
    """Compute the inherited penalty from parent clauses.

    Finds the maximum combined penalty among parents that exceed the
    threshold, applies decay, and respects depth limits.

    Args:
        parent_ids: IDs of parent clauses.
        cache: The penalty cache.
        all_clauses: Map of clause ID to Clause for parent lookup.
        config: Propagation configuration.

    Returns:
        (inherited_penalty, depth) tuple.
    """
    if not parent_ids:
        return 0.0, 0

    max_parent_penalty = 0.0
    max_parent_depth = 0

    for pid in parent_ids:
        rec = cache.get(pid)
        if rec is None:
            continue

        if rec.combined_penalty >= config.threshold:
            if rec.combined_penalty > max_parent_penalty:
                max_parent_penalty = rec.combined_penalty
                max_parent_depth = rec.depth

    if max_parent_penalty < config.threshold:
        return 0.0, 0

    # Depth check
    new_depth = max_parent_depth + 1
    if config.max_depth > 0 and new_depth > config.max_depth:
        return 0.0, 0

    inherited = config.decay * max_parent_penalty
    return inherited, new_depth


def combine_penalty(
    own_penalty: float,
    inherited_penalty: float,
    mode: PenaltyCombineMode,
    max_penalty: float,
) -> float:
    """Combine intrinsic and inherited penalties according to the mode.

    Args:
        own_penalty: Clause's intrinsic generality penalty.
        inherited_penalty: Penalty inherited from parents.
        mode: Combination strategy.
        max_penalty: Cap on accumulated penalty.

    Returns:
        Combined penalty, capped at max_penalty.
    """
    if inherited_penalty <= 0.0:
        return own_penalty

    if mode == PenaltyCombineMode.ADDITIVE:
        combined = own_penalty + inherited_penalty
    elif mode == PenaltyCombineMode.MULTIPLICATIVE:
        combined = own_penalty * (1.0 + inherited_penalty)
    elif mode == PenaltyCombineMode.MAX:
        combined = max(own_penalty, inherited_penalty)
    else:
        combined = own_penalty + inherited_penalty  # fallback to additive

    return min(combined, max_penalty)


def compute_and_cache_penalty(
    clause: Clause,
    cache: PenaltyCache,
    all_clauses: dict[int, Clause],
) -> float:
    """Compute combined penalty for a clause and store in cache.

    This is the main entry point called from _cl_process().
    Computes intrinsic penalty, looks up parent penalties,
    combines them, and caches the result.

    Optimized hot path: short-circuits when no parents have penalties
    above the threshold (the common case), avoiding unnecessary
    combine_penalty() call and reducing PenaltyRecord field writes.

    Args:
        clause: The clause to compute penalty for.
        cache: The penalty cache.
        all_clauses: Map of clause ID to Clause.

    Returns:
        The combined penalty value.
    """
    config = cache.config
    own_penalty = _clause_generality_penalty(clause)

    # Extract parent IDs from justification (tuple — zero-copy for most types)
    parent_ids = extract_parent_ids(clause.justification)

    # Fast path: no parents or no inheritance → skip inheritance machinery
    if not parent_ids:
        cache.put(clause.id, PenaltyRecord(own_penalty, 0.0, own_penalty, 0))
        return own_penalty

    # Compute inherited penalty
    inherited, depth = compute_inherited_penalty(
        parent_ids, cache, all_clauses, config,
    )

    # Fast path: no inheritance triggered (common case — threshold not met)
    if inherited <= 0.0:
        cache.put(clause.id, PenaltyRecord(own_penalty, 0.0, own_penalty, 0))
        return own_penalty

    # Slow path: combine own + inherited penalties
    combined = combine_penalty(
        own_penalty, inherited, config.mode, config.max_penalty,
    )

    cache.put(clause.id, PenaltyRecord(own_penalty, inherited, combined, depth))
    return combined
