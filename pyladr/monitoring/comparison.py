"""Performance comparison tools for C Prover9 vs Python PyLADR.

Provides structured comparison of search results between the C
reference implementation and the Python implementation, highlighting
behavioral equivalence and performance differences.

Usage:
    report = compare_search_results(c_stats, py_stats, problem_name="x2")
    print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ComparisonReport:
    """Structured comparison of C vs Python search results."""

    problem: str = ""

    # C results
    c_given: int = 0
    c_generated: int = 0
    c_kept: int = 0
    c_subsumed: int = 0
    c_seconds: float = 0.0
    c_proved: bool = False

    # Python results
    py_given: int = 0
    py_generated: int = 0
    py_kept: int = 0
    py_subsumed: int = 0
    py_seconds: float = 0.0
    py_proved: bool = False

    # Comparison metrics
    search_equivalent: bool = False
    slowdown_ratio: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a comparison summary."""
        lines = [
            "=" * 60,
            f"C vs Python COMPARISON: {self.problem}",
            "=" * 60,
            "",
            f"{'Metric':<25} {'C':>12} {'Python':>12} {'Match':>8}",
            "-" * 57,
            self._row("Proved", str(self.c_proved), str(self.py_proved),
                       self.c_proved == self.py_proved),
            self._row("Given", str(self.c_given), str(self.py_given),
                       self.c_given == self.py_given),
            self._row("Generated", str(self.c_generated), str(self.py_generated),
                       self.c_generated == self.py_generated),
            self._row("Kept", str(self.c_kept), str(self.py_kept),
                       self.c_kept == self.py_kept),
            self._row("Subsumed", str(self.c_subsumed), str(self.py_subsumed),
                       self.c_subsumed == self.py_subsumed),
            self._row("CPU seconds", f"{self.c_seconds:.4f}", f"{self.py_seconds:.4f}",
                       None),
        ]

        lines.append("")
        lines.append(f"Search equivalent: {'YES' if self.search_equivalent else 'NO'}")

        if self.slowdown_ratio > 0:
            lines.append(f"Slowdown ratio: {self.slowdown_ratio:.2f}x")

            if self.slowdown_ratio <= 2.0:
                lines.append("Performance: EXCELLENT (within 2x of C)")
            elif self.slowdown_ratio <= 4.0:
                lines.append("Performance: GOOD (within 4x, parallelizable to parity)")
            elif self.slowdown_ratio <= 10.0:
                lines.append("Performance: ACCEPTABLE (within 10x)")
            else:
                lines.append(f"Performance: SLOW ({self.slowdown_ratio:.1f}x)")

        for note in self.notes:
            lines.append(f"  Note: {note}")

        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def _row(label: str, c_val: str, py_val: str, match: bool | None) -> str:
        match_str = ""
        if match is True:
            match_str = "OK"
        elif match is False:
            match_str = "DIFF"
        return f"{label:<25} {c_val:>12} {py_val:>12} {match_str:>8}"

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return {
            "problem": self.problem,
            "c": {
                "given": self.c_given,
                "generated": self.c_generated,
                "kept": self.c_kept,
                "subsumed": self.c_subsumed,
                "seconds": self.c_seconds,
                "proved": self.c_proved,
            },
            "python": {
                "given": self.py_given,
                "generated": self.py_generated,
                "kept": self.py_kept,
                "subsumed": self.py_subsumed,
                "seconds": self.py_seconds,
                "proved": self.py_proved,
            },
            "search_equivalent": self.search_equivalent,
            "slowdown_ratio": self.slowdown_ratio,
            "notes": self.notes,
        }


def compare_search_results(
    c_stats: dict[str, Any],
    py_stats: dict[str, Any],
    problem_name: str = "",
) -> ComparisonReport:
    """Compare C and Python search results from stat dictionaries.

    Args:
        c_stats: Dict with keys: given, generated, kept, subsumed, seconds, proved.
        py_stats: Dict with same keys.
        problem_name: Name of the benchmark problem.

    Returns:
        ComparisonReport with equivalence and performance analysis.
    """
    report = ComparisonReport(problem=problem_name)

    # Extract C stats
    report.c_given = c_stats.get("given", 0)
    report.c_generated = c_stats.get("generated", 0)
    report.c_kept = c_stats.get("kept", 0)
    report.c_subsumed = c_stats.get("subsumed", 0)
    report.c_seconds = c_stats.get("seconds", 0.0)
    report.c_proved = c_stats.get("proved", False)

    # Extract Python stats
    report.py_given = py_stats.get("given", 0)
    report.py_generated = py_stats.get("generated", 0)
    report.py_kept = py_stats.get("kept", 0)
    report.py_subsumed = py_stats.get("subsumed", 0)
    report.py_seconds = py_stats.get("seconds", 0.0)
    report.py_proved = py_stats.get("proved", False)

    # Check search equivalence
    report.search_equivalent = (
        report.c_proved == report.py_proved
        and report.c_given == report.py_given
        and report.c_generated == report.py_generated
        and report.c_kept == report.py_kept
    )

    # Compute slowdown ratio
    if report.c_seconds > 0 and report.py_seconds > 0:
        report.slowdown_ratio = report.py_seconds / report.c_seconds
    elif report.c_seconds == 0 and report.py_seconds == 0:
        report.slowdown_ratio = 1.0
        report.notes.append("Both completed in <0.01s")

    # Generate notes for differences
    if report.c_proved != report.py_proved:
        report.notes.append("CRITICAL: Different theorem status!")

    if report.c_given != report.py_given:
        report.notes.append(
            f"Given clauses differ: C={report.c_given}, Py={report.py_given}"
        )

    if report.c_generated != report.py_generated:
        report.notes.append(
            f"Generated differ: C={report.c_generated}, Py={report.py_generated}"
        )

    return report
