"""Fixed-size derivation feature extraction from clause justifications.

Produces a compact numerical feature vector that summarises a clause's
derivation history without requiring the full inference chain.  These
features augment the existing 7-dimensional CLAUSE node features in the
graph builder.

Feature vector layout (13 dimensions):
  [0]  derivation_depth         — normalised depth in the derivation DAG
  [1]  num_parents              — direct parent count (0–N)
  [2]  num_simplifications      — secondary justification step count
  [3]  is_input                 — 1.0 if INPUT/GOAL/DENY, else 0.0
  [4]  is_resolution            — 1.0 if BINARY_RES or HYPER_RES
  [5]  is_paramodulation        — 1.0 if PARA
  [6]  is_factor                — 1.0 if FACTOR
  [7]  is_demodulated           — 1.0 if any DEMOD in secondary steps
  [8]  primary_rule_id          — integer JustType value (normalised)
  [9]  ancestor_rule_entropy    — Shannon entropy over rule distribution
                                   in the ancestor chain (higher = more
                                   diverse derivation path)
  [10] max_parent_depth         — deepest parent's depth (normalised)
  [11] branching_factor         — avg parent count across ancestors
  [12] chain_length             — length of inference chain (normalised)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyladr.core.clause import Clause

from .derivation_context import DerivationContext


@dataclass(frozen=True, slots=True)
class DerivationFeatureConfig:
    """Configuration for derivation feature extraction.

    Attributes:
        enabled: Master switch. When False, extract_features returns zeros.
        depth_normaliser: Divides raw depth to keep feature in [0, ~1].
        max_chain_length: Maximum inference chain length to consider.
        num_just_types: Number of JustType values for normalisation.
    """

    enabled: bool = True
    depth_normaliser: float = 50.0
    max_chain_length: int = 64
    num_just_types: int = 22  # len(JustType)


_DEFAULT_CONFIG = DerivationFeatureConfig()

DERIVATION_FEATURE_DIM = 13


@dataclass(frozen=True, slots=True)
class DerivationFeatures:
    """Extracted derivation feature vector with named access."""

    features: list[float]

    @property
    def depth(self) -> float:
        return self.features[0]

    @property
    def num_parents(self) -> float:
        return self.features[1]

    @property
    def is_input(self) -> bool:
        return self.features[3] > 0.5

    @property
    def dim(self) -> int:
        return len(self.features)


class DerivationFeatureExtractor:
    """Extracts fixed-size derivation features from clause justifications.

    Requires a DerivationContext that has already registered the clause
    and its ancestors.  If the clause is not registered, returns a zero
    vector (graceful degradation).

    Usage::

        ctx = DerivationContext()
        extractor = DerivationFeatureExtractor(ctx)

        # After registering clauses during search...
        features = extractor.extract(clause)
        # features.features is a list[float] of length DERIVATION_FEATURE_DIM
    """

    def __init__(
        self,
        context: DerivationContext,
        config: DerivationFeatureConfig | None = None,
    ) -> None:
        self._ctx = context
        self._config = config or _DEFAULT_CONFIG

    @property
    def feature_dim(self) -> int:
        return DERIVATION_FEATURE_DIM

    def extract(self, clause: Clause) -> DerivationFeatures:
        """Extract derivation features for a single clause.

        Returns a zero vector if derivation context is unavailable or
        the feature is disabled.
        """
        if not self._config.enabled:
            return DerivationFeatures(features=[0.0] * DERIVATION_FEATURE_DIM)

        info = self._ctx.get(clause.id)
        if info is None:
            # Not registered — try to register now
            info = self._ctx.register(clause)

        chain = self._ctx.get_inference_chain(
            clause.id, max_length=self._config.max_chain_length
        )

        norm_d = self._config.depth_normaliser
        norm_jt = float(self._config.num_just_types)

        # Basic features
        depth = info.depth / norm_d
        num_parents = float(len(info.parent_ids))
        num_simp = float(info.num_simplifications)

        # Rule category flags
        from pyladr.core.clause import JustType

        rule = info.primary_rule
        is_input = float(rule in (
            int(JustType.INPUT), int(JustType.GOAL), int(JustType.DENY),
        ))
        is_resolution = float(rule in (
            int(JustType.BINARY_RES), int(JustType.HYPER_RES), int(JustType.UR_RES),
        ))
        is_para = float(rule == int(JustType.PARA))
        is_factor = float(rule == int(JustType.FACTOR))

        # Check for demodulation in secondary steps
        is_demod = 0.0
        if clause.justification and len(clause.justification) > 1:
            for j in clause.justification[1:]:
                if j.just_type == JustType.DEMOD:
                    is_demod = 1.0
                    break

        primary_rule_norm = float(rule) / norm_jt

        # Ancestor chain entropy
        entropy = _chain_entropy(chain, self._config.num_just_types)

        # Max parent depth (normalised)
        max_parent_depth = 0.0
        for pid in info.parent_ids:
            pd = self._ctx.get_depth(pid)
            if pd > max_parent_depth:
                max_parent_depth = float(pd)
        max_parent_depth /= norm_d

        # Branching factor (average parent count in chain)
        branching = _chain_branching(self._ctx, chain, clause.id)

        # Chain length (normalised)
        chain_len = float(len(chain)) / float(self._config.max_chain_length)

        features = [
            depth,
            num_parents,
            num_simp,
            is_input,
            is_resolution,
            is_para,
            is_factor,
            is_demod,
            primary_rule_norm,
            entropy,
            max_parent_depth,
            branching,
            chain_len,
        ]

        return DerivationFeatures(features=features)

    def extract_batch(self, clauses: list[Clause]) -> list[list[float]]:
        """Extract derivation features for a batch of clauses.

        Returns a list of feature vectors (one per clause).
        """
        return [self.extract(c).features for c in clauses]


def _chain_entropy(chain: tuple[int, ...], num_types: int) -> float:
    """Shannon entropy of inference rule distribution in a chain.

    Higher entropy = more diverse derivation path.  Returns 0.0 for
    empty or single-element chains.
    """
    if len(chain) <= 1:
        return 0.0

    counts: dict[int, int] = {}
    for rule in chain:
        counts[rule] = counts.get(rule, 0) + 1

    n = float(len(chain))
    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)

    # Normalise by log2(num_types) to keep in [0, 1]
    max_entropy = math.log2(num_types) if num_types > 1 else 1.0
    return entropy / max_entropy


def _chain_branching(
    ctx: DerivationContext, chain: tuple[int, ...], start_id: int
) -> float:
    """Average branching factor (parent count) across the derivation chain."""
    if not chain:
        return 0.0

    total_parents = 0
    count = 0
    visited: set[int] = set()
    current = start_id

    while count < len(chain):
        info = ctx.get(current)
        if info is None or current in visited:
            break
        visited.add(current)
        total_parents += len(info.parent_ids)
        count += 1
        if info.parent_ids:
            current = info.parent_ids[0]
        else:
            break

    return (total_parents / count) if count > 0 else 0.0
