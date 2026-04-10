"""Repetition detection for clause selection bias.

Detects structurally repetitious clauses during given-clause search by
computing structural fingerprints and tracking their frequency. Clauses
that are structurally similar to many previously seen clauses receive
higher repetition scores, allowing the selection system to bias against
them in favor of more novel clause structures.

Structural fingerprints abstract away variable names and clause IDs,
capturing only the "shape" of a clause: predicate/function symbols,
arities, literal signs, and nesting structure. This means that clauses
like P(f(x,y)) and P(f(z,w)) share the same skeleton — they are
structurally identical modulo variable renaming.

Performance: fingerprint computation is O(n) in clause symbol count.
Frequency lookups are O(1) via hash tables. Designed for real-time use
in the inner loop of given-clause search.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term

logger = logging.getLogger(__name__)

# ── Structural fingerprinting ────────────────────────────────────────────────


def term_skeleton(t: Term) -> tuple:
    """Compute the structural skeleton of a term.

    Abstracts away variable identity: all variables map to a sentinel,
    preserving only the tree shape and rigid symbol identities.

    Returns a nested tuple encoding the term structure:
      - Variable  → ("V",)
      - Constant  → ("C", symnum)
      - Complex   → ("F", symnum, child_skeleton_0, child_skeleton_1, ...)

    O(n) in term symbol_count (visits each node once).
    """
    if t.is_variable:
        return ("V",)
    if t.is_constant:
        return ("C", t.symnum)
    child_skeletons = tuple(term_skeleton(a) for a in t.args)
    return ("F", t.symnum, *child_skeletons)


def literal_skeleton(lit: Literal) -> tuple:
    """Compute the structural skeleton of a literal.

    Encodes (sign, atom_skeleton). Two literals with the same skeleton
    have the same predicate, arity, sign, and argument structure shapes.
    """
    return (lit.sign, term_skeleton(lit.atom))


def clause_skeleton(c: Clause) -> tuple:
    """Compute the structural skeleton of a clause.

    The skeleton is a sorted tuple of literal skeletons. Sorting makes
    the skeleton invariant to literal ordering within the clause, so
    clauses that differ only in literal order share the same skeleton.

    Returns a hashable tuple suitable for use as a dictionary key.
    """
    lit_skels = tuple(sorted(literal_skeleton(lit) for lit in c.literals))
    return lit_skels


def subterm_skeletons(t: Term) -> list[tuple]:
    """Extract skeletons of all non-trivial subterms.

    Returns skeletons for every complex subterm (arity > 0). Variables
    and constants are excluded since they carry no structural pattern.
    Used for fine-grained repetition detection at the subterm level.
    """
    result: list[tuple] = []
    for sub in t.subterms():
        if sub.is_complex:
            result.append(term_skeleton(sub))
    return result


def clause_subterm_skeletons(c: Clause) -> list[tuple]:
    """Extract all non-trivial subterm skeletons from a clause.

    Collects complex subterm skeletons from every literal's atom.
    """
    result: list[tuple] = []
    for lit in c.literals:
        result.extend(subterm_skeletons(lit.atom))
    return result


# ── Repetition tracker ───────────────────────────────────────────────────────


@dataclass(slots=True)
class RepetitionConfig:
    """Configuration for repetition detection.

    Attributes:
        enabled: Master switch for repetition bias.
        clause_weight: How much whole-clause skeleton frequency contributes
            to the repetition score (vs subterm frequency). Range [0, 1].
        subterm_weight: How much subterm skeleton frequency contributes.
            clause_weight + subterm_weight should equal 1.0.
        decay_rate: Exponential decay rate for frequency counts. Older
            observations contribute less. 0.0 = no decay (all history
            weighted equally), 1.0 = only most recent counts matter.
            Typical value: 0.01–0.05.
        window_size: Number of recent clauses to track for frequency.
            0 = unlimited (track all). Limits memory usage.
        min_observations: Minimum total clause observations before
            repetition scores become active. Prevents premature bias
            during early search when everything looks "novel".
    """

    enabled: bool = True
    clause_weight: float = 0.6
    subterm_weight: float = 0.4
    decay_rate: float = 0.02
    window_size: int = 0  # 0 = unlimited
    min_observations: int = 20


@dataclass(slots=True)
class RepetitionStats:
    """Statistics for repetition detection monitoring."""

    clauses_observed: int = 0
    unique_skeletons: int = 0
    unique_subterm_skeletons: int = 0
    max_skeleton_frequency: int = 0
    max_subterm_frequency: int = 0
    total_penalty_applied: float = 0.0
    penalized_selections: int = 0

    def report(self) -> str:
        return (
            f"repetition: observed={self.clauses_observed}, "
            f"unique_skeletons={self.unique_skeletons}, "
            f"unique_subterms={self.unique_subterm_skeletons}, "
            f"max_skel_freq={self.max_skeleton_frequency}, "
            f"max_sub_freq={self.max_subterm_frequency}, "
            f"penalized={self.penalized_selections}"
        )


@dataclass(slots=True)
class RepetitionTracker:
    """Tracks structural repetition across the search.

    Maintains frequency counts of clause skeletons and subterm skeletons
    observed during the search. Computes a repetition score for candidate
    clauses based on how structurally similar they are to previously
    observed clauses.

    The repetition score is in [0, 1]:
      - 0.0 = completely novel structure (never seen before)
      - 1.0 = maximally repetitious (skeleton seen very frequently)

    Thread-safety: not thread-safe. Should be used from a single search
    thread (the main given-clause loop is single-threaded).
    """

    config: RepetitionConfig = field(default_factory=RepetitionConfig)
    stats: RepetitionStats = field(default_factory=RepetitionStats)

    # Frequency tables: skeleton → count
    _clause_freq: dict[tuple, float] = field(default_factory=lambda: defaultdict(float))
    _subterm_freq: dict[tuple, float] = field(default_factory=lambda: defaultdict(float))

    # Peak frequencies for normalization
    _max_clause_freq: float = field(default=0.0)
    _max_subterm_freq: float = field(default=0.0)

    def observe(self, c: Clause) -> None:
        """Record a clause as observed (kept or selected as given).

        Updates skeleton and subterm frequency counts. Should be called
        for every clause that enters the SOS or is selected as given.
        """
        self.stats.clauses_observed += 1
        decay = self.config.decay_rate

        # Apply decay to existing frequencies periodically
        if decay > 0 and self.stats.clauses_observed % 100 == 0:
            self._apply_decay(decay)

        # Record clause skeleton
        skel = clause_skeleton(c)
        self._clause_freq[skel] += 1.0
        freq = self._clause_freq[skel]
        if freq > self._max_clause_freq:
            self._max_clause_freq = freq
            self.stats.max_skeleton_frequency = int(freq)

        # Record subterm skeletons
        for sub_skel in clause_subterm_skeletons(c):
            self._subterm_freq[sub_skel] += 1.0
            sub_freq = self._subterm_freq[sub_skel]
            if sub_freq > self._max_subterm_freq:
                self._max_subterm_freq = sub_freq
                self.stats.max_subterm_frequency = int(sub_freq)

        # Update unique counts
        self.stats.unique_skeletons = len(self._clause_freq)
        self.stats.unique_subterm_skeletons = len(self._subterm_freq)

    def repetition_score(self, c: Clause) -> float:
        """Compute the repetition score for a candidate clause.

        Returns a value in [0, 1] where higher means more repetitious.
        Returns 0.0 if not enough observations have been made yet
        (controlled by config.min_observations).
        """
        if not self.config.enabled:
            return 0.0

        if self.stats.clauses_observed < self.config.min_observations:
            return 0.0

        clause_score = self._clause_repetition(c)
        subterm_score = self._subterm_repetition(c)

        cw = self.config.clause_weight
        sw = self.config.subterm_weight
        total = cw + sw
        if total == 0:
            return 0.0

        return (cw * clause_score + sw * subterm_score) / total

    def _clause_repetition(self, c: Clause) -> float:
        """Clause-level repetition: how often this exact skeleton has been seen.

        Normalized to [0, 1] by dividing by the maximum observed frequency.
        """
        if self._max_clause_freq <= 0:
            return 0.0
        skel = clause_skeleton(c)
        freq = self._clause_freq.get(skel, 0.0)
        return min(1.0, freq / self._max_clause_freq)

    def _subterm_repetition(self, c: Clause) -> float:
        """Subterm-level repetition: average frequency of subterm skeletons.

        Captures finer-grained structural repetition: even if the overall
        clause skeleton is novel, its subterms may be highly repetitious.
        """
        if self._max_subterm_freq <= 0:
            return 0.0

        sub_skels = clause_subterm_skeletons(c)
        if not sub_skels:
            return 0.0

        total_freq = sum(
            self._subterm_freq.get(s, 0.0) for s in sub_skels
        )
        avg_freq = total_freq / len(sub_skels)
        return min(1.0, avg_freq / self._max_subterm_freq)

    def _apply_decay(self, rate: float) -> None:
        """Apply exponential decay to all frequency counts.

        Multiplies all counts by (1 - rate), reducing the influence
        of older observations. Entries that decay below a threshold
        are pruned to limit memory usage.
        """
        factor = 1.0 - rate
        threshold = 0.1  # Prune entries below this

        # Decay clause frequencies
        to_remove: list[tuple] = []
        for skel, freq in self._clause_freq.items():
            new_freq = freq * factor
            if new_freq < threshold:
                to_remove.append(skel)
            else:
                self._clause_freq[skel] = new_freq
        for skel in to_remove:
            del self._clause_freq[skel]

        # Decay subterm frequencies
        to_remove.clear()
        for skel, freq in self._subterm_freq.items():
            new_freq = freq * factor
            if new_freq < threshold:
                to_remove.append(skel)
            else:
                self._subterm_freq[skel] = new_freq
        for skel in to_remove:
            del self._subterm_freq[skel]

        # Recompute max frequencies after decay
        self._max_clause_freq = max(self._clause_freq.values(), default=0.0)
        self._max_subterm_freq = max(self._subterm_freq.values(), default=0.0)

    def reset(self) -> None:
        """Reset all tracking state. Useful between search runs."""
        self._clause_freq.clear()
        self._subterm_freq.clear()
        self._max_clause_freq = 0.0
        self._max_subterm_freq = 0.0
        self.stats = RepetitionStats()
