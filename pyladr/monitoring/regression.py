"""Automated performance regression detection.

Compares current search performance against stored baselines
to detect regressions in throughput, memory, or clause statistics.

Usage:
    # Create a baseline from a known-good run:
    baseline = PerformanceBaseline.from_search_result(result, "v1.0")
    baseline.save("baselines/group_theory.json")

    # Later, check for regressions:
    detector = RegressionDetector(baseline)
    report = detector.check(current_result)
    if report.has_regressions:
        print(report)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pyladr.search.given_clause import SearchResult
    from pyladr.search.statistics import SearchStatistics


@dataclass(slots=True)
class PerformanceBaseline:
    """Stored performance baseline for regression comparison."""

    name: str = ""
    label: str = ""  # Version or commit label
    timestamp: str = ""

    # Timing
    elapsed_seconds: float = 0.0

    # Clause statistics
    given: int = 0
    generated: int = 0
    kept: int = 0
    subsumed: int = 0
    back_subsumed: int = 0

    # Derived rates
    generated_per_sec: float = 0.0
    kept_per_sec: float = 0.0

    # Exit code
    exit_code: int = 0
    proof_found: bool = False

    @classmethod
    def from_search_result(
        cls,
        result: SearchResult,
        label: str = "",
        name: str = "",
    ) -> PerformanceBaseline:
        """Create a baseline from a SearchResult."""
        from datetime import datetime

        stats = result.stats
        elapsed = stats.elapsed_seconds()

        return cls(
            name=name,
            label=label,
            timestamp=datetime.now().isoformat(),
            elapsed_seconds=elapsed,
            given=stats.given,
            generated=stats.generated,
            kept=stats.kept,
            subsumed=stats.subsumed,
            back_subsumed=stats.back_subsumed,
            generated_per_sec=stats.generated / elapsed if elapsed > 0 else 0.0,
            kept_per_sec=stats.kept / elapsed if elapsed > 0 else 0.0,
            exit_code=result.exit_code,
            proof_found=len(result.proofs) > 0,
        )

    @classmethod
    def from_stats(
        cls,
        stats: SearchStatistics,
        label: str = "",
        name: str = "",
    ) -> PerformanceBaseline:
        """Create a baseline from SearchStatistics."""
        from datetime import datetime

        elapsed = stats.elapsed_seconds()
        return cls(
            name=name,
            label=label,
            timestamp=datetime.now().isoformat(),
            elapsed_seconds=elapsed,
            given=stats.given,
            generated=stats.generated,
            kept=stats.kept,
            subsumed=stats.subsumed,
            back_subsumed=stats.back_subsumed,
            generated_per_sec=stats.generated / elapsed if elapsed > 0 else 0.0,
            kept_per_sec=stats.kept / elapsed if elapsed > 0 else 0.0,
        )

    def save(self, path: str | Path) -> None:
        """Save baseline to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> PerformanceBaseline:
        """Load baseline from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k != "_type"})

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return {
            "_type": "PerformanceBaseline",
            "name": self.name,
            "label": self.label,
            "timestamp": self.timestamp,
            "elapsed_seconds": self.elapsed_seconds,
            "given": self.given,
            "generated": self.generated,
            "kept": self.kept,
            "subsumed": self.subsumed,
            "back_subsumed": self.back_subsumed,
            "generated_per_sec": self.generated_per_sec,
            "kept_per_sec": self.kept_per_sec,
            "exit_code": self.exit_code,
            "proof_found": self.proof_found,
        }


@dataclass(slots=True)
class RegressionItem:
    """A single regression or improvement detected."""

    metric: str
    baseline_value: float
    current_value: float
    change_pct: float  # Positive = regression (worse), negative = improvement
    severity: str  # "critical", "warning", "info"
    description: str = ""


@dataclass(slots=True)
class RegressionReport:
    """Report from a regression check."""

    baseline_label: str = ""
    items: list[RegressionItem] = field(default_factory=list)
    behavioral_change: bool = False  # Different proof/exit status
    behavioral_notes: list[str] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        """True if any regressions detected."""
        return self.behavioral_change or any(
            i.change_pct > 0 and i.severity in ("critical", "warning")
            for i in self.items
        )

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.items if i.severity == "critical" and i.change_pct > 0)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.items if i.severity == "warning" and i.change_pct > 0)

    def summary(self) -> str:
        """Generate a regression report summary."""
        lines = [
            "=" * 60,
            "PERFORMANCE REGRESSION REPORT",
            "=" * 60,
            f"Baseline: {self.baseline_label}",
        ]

        if self.behavioral_change:
            lines.append("")
            lines.append("*** BEHAVIORAL CHANGE DETECTED ***")
            for note in self.behavioral_notes:
                lines.append(f"  {note}")

        regressions = [i for i in self.items if i.change_pct > 0]
        improvements = [i for i in self.items if i.change_pct < 0]
        neutral = [i for i in self.items if i.change_pct == 0]

        if regressions:
            lines.append("")
            lines.append("Regressions:")
            for item in sorted(regressions, key=lambda i: -i.change_pct):
                marker = "!!!" if item.severity == "critical" else " ! " if item.severity == "warning" else "   "
                lines.append(
                    f"  {marker} {item.metric}: "
                    f"{item.baseline_value:.4g} -> {item.current_value:.4g} "
                    f"(+{item.change_pct:.1f}%)"
                )
                if item.description:
                    lines.append(f"      {item.description}")

        if improvements:
            lines.append("")
            lines.append("Improvements:")
            for item in sorted(improvements, key=lambda i: i.change_pct):
                lines.append(
                    f"       {item.metric}: "
                    f"{item.baseline_value:.4g} -> {item.current_value:.4g} "
                    f"({item.change_pct:.1f}%)"
                )

        if not regressions and not improvements and not self.behavioral_change:
            lines.append("")
            lines.append("No regressions detected.")

        lines.append("=" * 60)
        return "\n".join(lines)


class RegressionDetector:
    """Automated performance regression detector.

    Compares current search results against a stored baseline
    to detect timing regressions, throughput drops, and behavioral
    changes.

    Usage:
        baseline = PerformanceBaseline.load("baselines/problem.json")
        detector = RegressionDetector(baseline)
        report = detector.check_stats(current_stats)
    """

    __slots__ = ("_baseline", "_time_threshold", "_count_threshold")

    def __init__(
        self,
        baseline: PerformanceBaseline,
        time_threshold: float = 0.20,   # 20% timing regression
        count_threshold: float = 0.05,  # 5% count change
    ) -> None:
        """Initialize regression detector.

        Args:
            baseline: Reference baseline to compare against.
            time_threshold: Fraction increase in timing to flag as regression.
            count_threshold: Fraction change in clause counts to flag.
        """
        self._baseline = baseline
        self._time_threshold = time_threshold
        self._count_threshold = count_threshold

    def check_stats(self, stats: SearchStatistics) -> RegressionReport:
        """Check current statistics against the baseline."""
        report = RegressionReport(baseline_label=self._baseline.label)
        b = self._baseline
        elapsed = stats.elapsed_seconds()

        # Timing check
        self._check_metric(
            report, "elapsed_seconds", b.elapsed_seconds, elapsed,
            self._time_threshold, "critical",
            "Search took significantly longer",
        )

        # Throughput checks
        if elapsed > 0:
            gen_rate = stats.generated / elapsed
            self._check_metric(
                report, "generated_per_sec", b.generated_per_sec, gen_rate,
                self._time_threshold, "warning",
                "Inference generation throughput dropped",
                lower_is_worse=True,
            )

        # Clause count checks (behavioral — exact match expected)
        self._check_exact(report, "given", b.given, stats.given)
        self._check_exact(report, "generated", b.generated, stats.generated)
        self._check_exact(report, "kept", b.kept, stats.kept)

        return report

    def check_result(self, result: SearchResult) -> RegressionReport:
        """Check a full SearchResult against the baseline."""
        report = self.check_stats(result.stats)
        b = self._baseline

        # Behavioral check: same proof status?
        current_proved = len(result.proofs) > 0
        if b.proof_found != current_proved:
            report.behavioral_change = True
            report.behavioral_notes.append(
                f"Proof status changed: baseline={'proved' if b.proof_found else 'not proved'}, "
                f"current={'proved' if current_proved else 'not proved'}"
            )

        if b.exit_code != 0 and b.exit_code != result.exit_code:
            report.behavioral_change = True
            report.behavioral_notes.append(
                f"Exit code changed: baseline={b.exit_code}, current={result.exit_code}"
            )

        return report

    def _check_metric(
        self,
        report: RegressionReport,
        name: str,
        baseline_val: float,
        current_val: float,
        threshold: float,
        severity: str,
        description: str = "",
        lower_is_worse: bool = False,
    ) -> None:
        """Check a single metric for regression."""
        if baseline_val == 0:
            return

        if lower_is_worse:
            # For rates: decrease is bad
            change_pct = (baseline_val - current_val) / baseline_val * 100
        else:
            # For times/counts: increase is bad
            change_pct = (current_val - baseline_val) / baseline_val * 100

        # Classify severity
        actual_severity = severity
        if abs(change_pct) < threshold * 100:
            actual_severity = "info"

        report.items.append(RegressionItem(
            metric=name,
            baseline_value=baseline_val,
            current_value=current_val,
            change_pct=change_pct,
            severity=actual_severity,
            description=description if change_pct > threshold * 100 else "",
        ))

    def _check_exact(
        self,
        report: RegressionReport,
        name: str,
        baseline_val: int,
        current_val: int,
    ) -> None:
        """Check an exact-match metric (clause counts should be identical)."""
        if baseline_val == current_val:
            report.items.append(RegressionItem(
                metric=name,
                baseline_value=float(baseline_val),
                current_value=float(current_val),
                change_pct=0.0,
                severity="info",
            ))
            return

        change_pct = (
            (current_val - baseline_val) / baseline_val * 100
            if baseline_val != 0
            else 100.0
        )

        report.items.append(RegressionItem(
            metric=name,
            baseline_value=float(baseline_val),
            current_value=float(current_val),
            change_pct=change_pct,
            severity="warning",
            description=f"Clause count changed: {baseline_val} -> {current_val}",
        ))
