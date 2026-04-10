"""Performance regression detection for online learning.

Monitors whether online learning is improving or degrading theorem proving
capability. Implements statistical comparison, rolling window analysis,
and automatic fallback triggers.

Usage:
    detector = LearningRegressionDetector()

    # Record static baseline performance:
    detector.set_baseline(proof_rate=0.85, avg_given=120, avg_kept_rate=0.35)

    # During search with online learning:
    detector.record_search_result(proved=True, given=105, kept_rate=0.38)
    detector.record_search_result(proved=True, given=130, kept_rate=0.32)

    # Check for regression:
    report = detector.check()
    if report.should_fallback:
        # Switch back to static model
        ...
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# ── Search result record ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SearchResultRecord:
    """Record of a single search result for regression analysis."""

    proved: bool
    given_count: int
    kept_rate: float  # kept / generated
    elapsed_seconds: float = 0.0
    timestamp: float = 0.0


# ── Baseline ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class LearningBaseline:
    """Baseline performance from static (non-learning) model.

    These values represent expected performance without online learning,
    used to determine whether learning is helping or hurting.
    """

    proof_rate: float = 0.0  # Fraction of problems proved
    avg_given: float = 0.0  # Average given clauses to proof
    avg_kept_rate: float = 0.0  # Average kept/generated ratio
    avg_elapsed: float = 0.0  # Average search time
    sample_count: int = 0  # How many results in the baseline

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_rate": self.proof_rate,
            "avg_given": self.avg_given,
            "avg_kept_rate": self.avg_kept_rate,
            "avg_elapsed": self.avg_elapsed,
            "sample_count": self.sample_count,
        }


# ── Regression report ─────────────────────────────────────────────────────


@dataclass(slots=True)
class LearningRegressionReport:
    """Report from a learning regression check."""

    # Comparison metrics
    baseline_proof_rate: float = 0.0
    current_proof_rate: float = 0.0
    baseline_avg_given: float = 0.0
    current_avg_given: float = 0.0
    baseline_avg_kept_rate: float = 0.0
    current_avg_kept_rate: float = 0.0

    # Verdicts
    proof_rate_delta: float = 0.0
    given_count_delta_pct: float = 0.0
    kept_rate_delta: float = 0.0

    is_regression: bool = False
    should_fallback: bool = False
    reasons: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "LEARNING REGRESSION REPORT",
            "=" * 60,
        ]

        lines.append(
            f"Proof rate:  {self.baseline_proof_rate:.3f} -> "
            f"{self.current_proof_rate:.3f} "
            f"(delta: {self.proof_rate_delta:+.3f})"
        )
        lines.append(
            f"Avg given:   {self.baseline_avg_given:.1f} -> "
            f"{self.current_avg_given:.1f} "
            f"({self.given_count_delta_pct:+.1f}%)"
        )
        lines.append(
            f"Kept rate:   {self.baseline_avg_kept_rate:.3f} -> "
            f"{self.current_avg_kept_rate:.3f} "
            f"(delta: {self.kept_rate_delta:+.3f})"
        )

        if self.is_regression:
            lines.append("")
            lines.append("*** REGRESSION DETECTED ***")
            for r in self.reasons:
                lines.append(f"  - {r}")
        elif self.improvements:
            lines.append("")
            lines.append("Improvements:")
            for i in self.improvements:
                lines.append(f"  + {i}")
        else:
            lines.append("")
            lines.append("No significant change detected.")

        if self.should_fallback:
            lines.append("")
            lines.append(">>> FALLBACK RECOMMENDED: revert to static model <<<")

        lines.append("=" * 60)
        return "\n".join(lines)


# ── Regression detector ───────────────────────────────────────────────────


class LearningRegressionDetector:
    """Detects performance regression caused by online learning.

    Maintains a rolling window of search results and compares against
    a static model baseline. Triggers fallback when learning degrades
    proof success rate, increases search cost, or reduces kept rate.

    Usage:
        detector = LearningRegressionDetector()
        detector.set_baseline(proof_rate=0.85, avg_given=120)

        # Record each search result:
        detector.record_search_result(proved=True, given=105, kept_rate=0.38)

        # Check periodically:
        report = detector.check()
        if report.should_fallback:
            revert_to_static_model()
    """

    __slots__ = (
        "_baseline", "_results", "_window_size",
        "_proof_rate_threshold", "_given_threshold",
        "_kept_rate_threshold", "_min_samples",
    )

    def __init__(
        self,
        window_size: int = 20,
        proof_rate_threshold: float = 0.10,
        given_threshold: float = 0.25,
        kept_rate_threshold: float = 0.10,
        min_samples: int = 5,
    ) -> None:
        """Initialize regression detector.

        Args:
            window_size: Rolling window of recent results to track.
            proof_rate_threshold: Acceptable drop in proof rate before regression.
            given_threshold: Acceptable fractional increase in given count.
            kept_rate_threshold: Acceptable drop in kept rate.
            min_samples: Minimum results before making comparisons.
        """
        self._baseline: LearningBaseline | None = None
        self._results: deque[SearchResultRecord] = deque(maxlen=window_size)
        self._window_size = window_size
        self._proof_rate_threshold = proof_rate_threshold
        self._given_threshold = given_threshold
        self._kept_rate_threshold = kept_rate_threshold
        self._min_samples = min_samples

    def set_baseline(
        self,
        proof_rate: float = 0.0,
        avg_given: float = 0.0,
        avg_kept_rate: float = 0.0,
        avg_elapsed: float = 0.0,
        sample_count: int = 0,
    ) -> None:
        """Set the static model baseline.

        Args:
            proof_rate: Fraction of problems proved by static model.
            avg_given: Average given clauses needed for proof.
            avg_kept_rate: Average kept/generated ratio.
            avg_elapsed: Average search time.
            sample_count: Number of samples in the baseline.
        """
        self._baseline = LearningBaseline(
            proof_rate=proof_rate,
            avg_given=avg_given,
            avg_kept_rate=avg_kept_rate,
            avg_elapsed=avg_elapsed,
            sample_count=sample_count,
        )

    def set_baseline_from_object(self, baseline: LearningBaseline) -> None:
        """Set the baseline directly from a LearningBaseline object."""
        self._baseline = baseline

    def record_search_result(
        self,
        proved: bool,
        given: int,
        kept_rate: float = 0.0,
        elapsed: float = 0.0,
    ) -> None:
        """Record a search result from an online-learning-enabled run.

        Args:
            proved: Whether a proof was found.
            given: Number of given clauses processed.
            kept_rate: kept/generated ratio.
            elapsed: Wall-clock time for search.
        """
        self._results.append(SearchResultRecord(
            proved=proved,
            given_count=given,
            kept_rate=kept_rate,
            elapsed_seconds=elapsed,
            timestamp=time.monotonic(),
        ))

    @property
    def result_count(self) -> int:
        return len(self._results)

    @property
    def has_baseline(self) -> bool:
        return self._baseline is not None

    def check(self) -> LearningRegressionReport:
        """Check for performance regression.

        Compares the rolling window of recent results against the
        static model baseline.

        Returns:
            LearningRegressionReport with verdicts and recommendations.
        """
        report = LearningRegressionReport()

        if self._baseline is None:
            report.reasons.append("No baseline set")
            return report

        if len(self._results) < self._min_samples:
            report.reasons.append(
                f"Insufficient samples: {len(self._results)} < {self._min_samples}"
            )
            return report

        b = self._baseline
        results = list(self._results)

        # Compute current metrics
        proved_count = sum(1 for r in results if r.proved)
        current_proof_rate = proved_count / len(results)

        proved_results = [r for r in results if r.proved]
        current_avg_given = (
            sum(r.given_count for r in proved_results) / len(proved_results)
            if proved_results else 0.0
        )

        current_avg_kept_rate = (
            sum(r.kept_rate for r in results) / len(results)
            if results else 0.0
        )

        # Fill report
        report.baseline_proof_rate = b.proof_rate
        report.current_proof_rate = current_proof_rate
        report.baseline_avg_given = b.avg_given
        report.current_avg_given = current_avg_given
        report.baseline_avg_kept_rate = b.avg_kept_rate
        report.current_avg_kept_rate = current_avg_kept_rate

        # Compute deltas
        report.proof_rate_delta = current_proof_rate - b.proof_rate
        if b.avg_given > 0 and current_avg_given > 0:
            report.given_count_delta_pct = (
                (current_avg_given - b.avg_given) / b.avg_given * 100
            )
        report.kept_rate_delta = current_avg_kept_rate - b.avg_kept_rate

        # Check for regression
        regression_detected = False

        # Proof rate drop
        if report.proof_rate_delta < -self._proof_rate_threshold:
            report.reasons.append(
                f"Proof rate dropped by {abs(report.proof_rate_delta):.3f} "
                f"(threshold: {self._proof_rate_threshold:.3f})"
            )
            regression_detected = True
        elif report.proof_rate_delta > self._proof_rate_threshold:
            report.improvements.append(
                f"Proof rate improved by {report.proof_rate_delta:.3f}"
            )

        # Given count increase (more work needed = regression)
        if b.avg_given > 0 and current_avg_given > 0:
            if report.given_count_delta_pct > self._given_threshold * 100:
                report.reasons.append(
                    f"Average given count increased by "
                    f"{report.given_count_delta_pct:.1f}% "
                    f"(threshold: {self._given_threshold * 100:.1f}%)"
                )
                regression_detected = True
            elif report.given_count_delta_pct < -self._given_threshold * 100:
                report.improvements.append(
                    f"Average given count decreased by "
                    f"{abs(report.given_count_delta_pct):.1f}%"
                )

        # Kept rate drop
        if report.kept_rate_delta < -self._kept_rate_threshold:
            report.reasons.append(
                f"Kept rate dropped by {abs(report.kept_rate_delta):.3f} "
                f"(threshold: {self._kept_rate_threshold:.3f})"
            )
            regression_detected = True
        elif report.kept_rate_delta > self._kept_rate_threshold:
            report.improvements.append(
                f"Kept rate improved by {report.kept_rate_delta:.3f}"
            )

        report.is_regression = regression_detected

        # Recommend fallback for severe regression
        # (proof rate drop + another metric worsening)
        if regression_detected and len(report.reasons) >= 2:
            report.should_fallback = True

        return report

    def reset(self) -> None:
        """Clear all recorded results (baseline is preserved)."""
        self._results.clear()
