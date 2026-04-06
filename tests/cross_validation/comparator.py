"""Comparator for validating Python vs C behavioral equivalence.

Compares structured outputs from the Python and C implementations
at multiple levels of granularity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .c_runner import ProverResult


@dataclass
class ComparisonResult:
    """Result of comparing Python and C outputs."""

    equivalent: bool = True
    differences: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def add_difference(self, msg: str) -> None:
        self.equivalent = False
        self.differences.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def __str__(self) -> str:
        if self.equivalent:
            status = "EQUIVALENT"
            if self.warnings:
                status += f" (with {len(self.warnings)} warnings)"
        else:
            status = f"DIFFERENT ({len(self.differences)} differences)"
        lines = [status]
        for d in self.differences:
            lines.append(f"  DIFF: {d}")
        for w in self.warnings:
            lines.append(f"  WARN: {w}")
        return "\n".join(lines)


def compare_theorem_result(
    c_result: ProverResult,
    py_result: ProverResult,
) -> ComparisonResult:
    """Compare the high-level theorem proving result.

    Checks: theorem proved/failed status, proof existence.
    """
    comp = ComparisonResult()

    if c_result.theorem_proved != py_result.theorem_proved:
        comp.add_difference(
            f"theorem_proved: C={c_result.theorem_proved}, "
            f"Python={py_result.theorem_proved}"
        )

    if c_result.search_failed != py_result.search_failed:
        comp.add_difference(
            f"search_failed: C={c_result.search_failed}, "
            f"Python={py_result.search_failed}"
        )

    return comp


def compare_proof_structure(
    c_result: ProverResult,
    py_result: ProverResult,
) -> ComparisonResult:
    """Compare proof structure: length, clause count, justification types.

    Note: Exact clause IDs may differ, but proof length and structure
    should be equivalent for deterministic search.
    """
    comp = ComparisonResult()

    if c_result.proof_length != py_result.proof_length:
        comp.add_difference(
            f"proof_length: C={c_result.proof_length}, "
            f"Python={py_result.proof_length}"
        )

    c_proof_len = len(c_result.proof_clauses)
    py_proof_len = len(py_result.proof_clauses)
    if c_proof_len != py_proof_len:
        comp.add_difference(
            f"proof_clauses count: C={c_proof_len}, Python={py_proof_len}"
        )

    # Compare justification types (e.g., "assumption", "resolve", "paramod")
    c_just_types = _extract_just_types(c_result)
    py_just_types = _extract_just_types(py_result)
    if c_just_types != py_just_types:
        comp.add_difference(
            f"justification_types: C={c_just_types}, Python={py_just_types}"
        )

    return comp


def compare_search_statistics(
    c_result: ProverResult,
    py_result: ProverResult,
    *,
    tolerance: float = 0.0,
) -> ComparisonResult:
    """Compare search statistics with optional tolerance.

    For deterministic search, statistics should match exactly (tolerance=0).
    For non-deterministic variants, allow configurable tolerance.
    """
    comp = ComparisonResult()

    stats = [
        ("clauses_given", c_result.clauses_given, py_result.clauses_given),
        ("clauses_generated", c_result.clauses_generated, py_result.clauses_generated),
        ("clauses_kept", c_result.clauses_kept, py_result.clauses_kept),
    ]

    for name, c_val, py_val in stats:
        if tolerance == 0.0:
            if c_val != py_val:
                comp.add_difference(f"{name}: C={c_val}, Python={py_val}")
        else:
            if c_val == 0 and py_val == 0:
                continue
            ratio = abs(c_val - py_val) / max(c_val, py_val, 1)
            if ratio > tolerance:
                comp.add_difference(
                    f"{name}: C={c_val}, Python={py_val} "
                    f"(ratio={ratio:.2%}, tolerance={tolerance:.2%})"
                )
            elif c_val != py_val:
                comp.add_warning(
                    f"{name}: C={c_val}, Python={py_val} "
                    f"(within tolerance)"
                )

    return comp


def compare_full(
    c_result: ProverResult,
    py_result: ProverResult,
    *,
    stats_tolerance: float = 0.0,
) -> ComparisonResult:
    """Full comparison: result + proof structure + search stats."""
    comp = ComparisonResult()

    sub_comparisons = [
        ("theorem_result", compare_theorem_result(c_result, py_result)),
        ("proof_structure", compare_proof_structure(c_result, py_result)),
        (
            "search_statistics",
            compare_search_statistics(
                c_result, py_result, tolerance=stats_tolerance
            ),
        ),
    ]

    for name, sub in sub_comparisons:
        if not sub.equivalent:
            for d in sub.differences:
                comp.add_difference(f"[{name}] {d}")
        for w in sub.warnings:
            comp.add_warning(f"[{name}] {w}")

    comp.details = {name: sub for name, sub in sub_comparisons}
    return comp


def _extract_just_types(result: ProverResult) -> list[str]:
    """Extract the type of each justification (first word)."""
    types = []
    for clause in result.proof_clauses:
        just = clause.get("justification", "")
        # Extract first word as the justification type
        just_type = just.split("(")[0].split(",")[0].strip()
        types.append(just_type)
    return types
