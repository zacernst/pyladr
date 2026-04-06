"""Search statistics matching C search-structures.h Stats.

Tracks all counters and metrics during the given-clause search loop.
"""

from __future__ import annotations

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

    # Unit conflict
    unit_conflicts: int = 0

    # Timing
    start_time: float = 0.0
    search_start_time: float = 0.0

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
