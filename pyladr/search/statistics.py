"""Search statistics matching C search-structures.h Stats.

Tracks all counters and metrics during the given-clause search loop.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
import time


@dataclass(slots=True)
class SearchStatistics:
    """Search statistics matching C Stats struct in search-structures.h."""

    # Main counters
    given: int = 0
    generated: int = 0
    kept: int = 0
    subsumed: int = 0

    # Subsumption/demodulation details
    back_subsumed: int = 0
    demodulated: int = 0
    back_demodulated: int = 0
    new_demodulators: int = 0
    new_lex_demods: int = 0
    sos_limit_deleted: int = 0
    kept_by_rule: int = 0
    deleted_by_rule: int = 0

    # Proof tracking
    proofs: int = 0
    empty_clauses_found: int = 0

    # Non-unit subsumption test counter (moved from module-level global)
    nonunit_subsumption_tests: int = 0

    # Unit conflict
    unit_conflicts: int = 0

    # Penalty weight adjustment
    penalty_weight_adjusted: int = 0

    # Timing
    start_time: float = 0.0
    search_start_time: float = 0.0

    # Per-given-clause inference tracking:
    # - given_inference_counts: total clauses *generated* from each given clause
    #   (incremented in record_generated). Used to identify productive givens.
    # - given_available_counts: snapshot of how many clauses were available for
    #   inference when this given was selected (set in begin_given).
    given_inference_counts: dict[int, int] = field(default_factory=dict)
    given_available_counts: dict[int, int] = field(default_factory=dict)

    # Clause-level partnership tracking — tracks *which* specific clauses were
    # tried and succeeded as inference partners. Used for subsumption analysis
    # and compatibility reporting (get_given_compatibility_stats).
    given_attempted_partners: dict[int, set[int]] = field(default_factory=dict)
    given_successful_partners: dict[int, set[int]] = field(default_factory=dict)
    _current_given_id: int = 0
    # Track recent given IDs to cap partnership dict memory usage
    _recent_given_ids: deque = field(default_factory=lambda: deque(maxlen=1000))

    def start(self) -> None:
        """Record search start time."""
        self.start_time = time.monotonic()
        self.search_start_time = self.start_time

    def elapsed_seconds(self) -> float:
        """Seconds since search start."""
        return time.monotonic() - self.start_time

    def search_seconds(self) -> float:
        """Seconds since search loop started."""
        return time.monotonic() - self.search_start_time

    def begin_given(self, clause_id: int, available_count: int = 0) -> None:
        """Mark the start of inference generation for a given clause.

        Called once per given clause, right after stats.given is incremented.
        Initializes the counter for this clause and sets it as current.

        Args:
            clause_id: ID of the given clause
            available_count: Number of clauses available for inference
        """
        self._current_given_id = clause_id
        self.given_inference_counts[clause_id] = 0
        self.given_available_counts[clause_id] = available_count
        # Evict oldest tracked given clause to cap memory
        if len(self._recent_given_ids) == self._recent_given_ids.maxlen:
            old_id = self._recent_given_ids[0]  # will be evicted by deque
            self.given_attempted_partners.pop(old_id, None)
            self.given_successful_partners.pop(old_id, None)
            self.given_inference_counts.pop(old_id, None)
            self.given_available_counts.pop(old_id, None)
        self._recent_given_ids.append(clause_id)
        self.given_attempted_partners[clause_id] = set()
        self.given_successful_partners[clause_id] = set()
        logging.getLogger(__name__).debug(
            "begin_given: clause %d has %d available clauses", clause_id, available_count
        )

    def record_generated(self) -> None:
        """Record one generated clause, attributing it to the current given.

        Increments both the global generated counter and the per-given counter.
        """
        self.generated += 1
        if self._current_given_id != 0:
            self.given_inference_counts[self._current_given_id] = (
                self.given_inference_counts.get(self._current_given_id, 0) + 1
            )

    def record_attempted_partnership(self, partner_clause_id: int) -> None:
        """Record that we attempted inference with a specific partner clause.

        Args:
            partner_clause_id: ID of the clause we attempted inference with
        """
        if self._current_given_id != 0:
            if self._current_given_id not in self.given_attempted_partners:
                self.given_attempted_partners[self._current_given_id] = set()
            self.given_attempted_partners[self._current_given_id].add(partner_clause_id)

    def record_successful_partnership(self, partner_clause_id: int) -> None:
        """Record that we successfully inferred with a specific partner clause.

        Args:
            partner_clause_id: ID of the clause we successfully inferred with
        """
        if self._current_given_id != 0:
            if self._current_given_id not in self.given_successful_partners:
                self.given_successful_partners[self._current_given_id] = set()
            self.given_successful_partners[self._current_given_id].add(partner_clause_id)
            logging.getLogger(__name__).debug(
                "compatible: clause %d compatible count: %d -> %d",
                self._current_given_id,
                len(self.given_successful_partners[self._current_given_id]) - 1,
                len(self.given_successful_partners[self._current_given_id]),
            )

    def get_given_inference_count(self, clause_id: int) -> int:
        """Return how many clauses were generated from a specific given clause."""
        return self.given_inference_counts.get(clause_id, 0)

    def get_given_compatibility_stats(self, clause_id: int) -> tuple[int, int, float]:
        """Return compatibility statistics for a given clause.

        Returns:
            tuple of (successful_partnerships, attempted_partnerships, percentage)
            where percentage is 0-100 or 0.0 if no attempted partnerships
        """
        successful_partners = len(self.given_successful_partners.get(clause_id, set()))
        attempted_partners = len(self.given_attempted_partners.get(clause_id, set()))

        # Fallback for clauses that haven't been updated to new tracking yet
        if attempted_partners == 0:
            attempted_partners = self.given_available_counts.get(clause_id, 0)

        percentage = (successful_partners / attempted_partners * 100.0) if attempted_partners > 0 else 0.0
        # Debug: print(f"DEBUG GET_COMPATIBILITY: clause {clause_id} -> successful={successful_partners}, attempted={attempted_partners}, percentage={percentage:.1f}%")
        return successful_partners, attempted_partners, percentage

    def top_given_clauses(self, n: int = 10) -> list[tuple[int, int]]:
        """Return the top-N most productive given clauses by inference count.

        Returns list of (clause_id, count) sorted descending by count.
        """
        return sorted(
            self.given_inference_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:n]

    def report(self) -> str:
        """Summary report matching C fprint_all_stats format."""
        lines = [
            f"given={self.given}",
            f"generated={self.generated}",
            f"kept={self.kept}",
            f"subsumed={self.subsumed}",
            f"back_subsumed={self.back_subsumed}",
            f"proofs={self.proofs}",
            f"time={self.elapsed_seconds():.2f}s",
        ]
        return ", ".join(lines)
