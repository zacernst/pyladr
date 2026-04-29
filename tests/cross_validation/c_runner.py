"""Runner for the C Prover9 reference implementation.

Provides utilities to invoke the C prover9 binary and capture structured output
for comparison against the Python implementation.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests.conftest import C_PROVER9_BIN, ConfigurationError, require_c_binary

# Re-export for backwards compatibility with tests that import from c_runner.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
__all__ = [
    "C_PROVER9_BIN",
    "ConfigurationError",
    "ProverResult",
    "extract_proof_ids",
    "extract_proof_justifications",
    "run_c_prover9",
    "run_c_prover9_from_string",
]


@dataclass
class ProverResult:
    """Structured result from a Prover9 run."""

    exit_code: int
    raw_output: str
    theorem_proved: bool = False
    search_failed: bool = False
    clauses_given: int = 0
    clauses_generated: int = 0
    clauses_kept: int = 0
    clauses_deleted: int = 0
    proof_length: int = 0
    proof_clauses: list[dict[str, Any]] = field(default_factory=list)
    input_clauses: list[str] = field(default_factory=list)
    initial_clauses: list[str] = field(default_factory=list)
    max_weight: float = 0.0
    user_cpu_time: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.theorem_proved and self.exit_code == 0


def run_c_prover9(
    input_file: Path | str,
    *,
    timeout: float = 30.0,
    extra_args: list[str] | None = None,
) -> ProverResult:
    """Run the C Prover9 binary on an input file and return structured results.

    Args:
        input_file: Path to LADR-format input file.
        timeout: Maximum seconds to allow the prover to run.
        extra_args: Additional command-line arguments.

    Returns:
        ProverResult with parsed output fields.

    Raises:
        ConfigurationError: If the C binary is missing or not executable.
        subprocess.TimeoutExpired: If the prover exceeds the timeout.
    """
    require_c_binary()

    cmd = [str(C_PROVER9_BIN), "-f", str(input_file)]
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    return _parse_output(proc.stdout + proc.stderr, proc.returncode)


def run_c_prover9_from_string(
    input_text: str,
    *,
    timeout: float = 30.0,
) -> ProverResult:
    """Run C Prover9 with input piped via stdin.

    Args:
        input_text: LADR-format input as a string.
        timeout: Maximum seconds to allow the prover to run.

    Returns:
        ProverResult with parsed output fields.

    Raises:
        ConfigurationError: If the C binary is missing or not executable.
    """
    require_c_binary()

    proc = subprocess.run(
        [str(C_PROVER9_BIN)],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    return _parse_output(proc.stdout + proc.stderr, proc.returncode)


def _parse_output(raw: str, exit_code: int) -> ProverResult:
    """Parse raw Prover9 output into a structured ProverResult."""
    result = ProverResult(exit_code=exit_code, raw_output=raw)

    result.theorem_proved = "THEOREM PROVED" in raw
    result.search_failed = "SEARCH FAILED" in raw

    # Extract search statistics (C format: "Given=12. Generated=118. Kept=23.")
    m = re.search(r"Given=(\d+)", raw)
    if m:
        result.clauses_given = int(m.group(1))

    m = re.search(r"Generated=(\d+)", raw)
    if m:
        result.clauses_generated = int(m.group(1))

    m = re.search(r"Kept=(\d+)", raw)
    if m:
        result.clauses_kept = int(m.group(1))

    m = re.search(r"Sos_limit_deleted=(\d+)", raw)
    if m:
        result.clauses_deleted = int(m.group(1))

    # Extract proof length
    m = re.search(r"Length of proof is (\d+)", raw)
    if m:
        result.proof_length = int(m.group(1))

    # Extract max weight
    m = re.search(r"max_weight=([0-9.]+)", raw)
    if m:
        result.max_weight = float(m.group(1))

    # Extract user CPU time
    m = re.search(r"User_CPU=([0-9.]+)", raw)
    if m:
        result.user_cpu_time = float(m.group(1))

    # Extract proof clauses (lines with clause IDs in the proof section)
    proof_section = False
    for line in raw.splitlines():
        if "PROOF" in line and "====" in line:
            proof_section = True
            continue
        if proof_section and "====" in line:
            proof_section = False
            continue
        if proof_section:
            clause_match = re.match(r"\s*(\d+)\s+(.+)\.\s+\[(.+)\]", line)
            if clause_match:
                result.proof_clauses.append(
                    {
                        "id": int(clause_match.group(1)),
                        "clause": clause_match.group(2).strip(),
                        "justification": clause_match.group(3).strip(),
                    }
                )

    # Extract input clauses
    in_input = False
    for line in raw.splitlines():
        if "INPUT" in line and "====" in line:
            in_input = True
            continue
        if in_input and "====" in line:
            in_input = False
            continue
        if in_input and line.strip() and not line.startswith("%"):
            result.input_clauses.append(line.strip())

    return result


def extract_proof_ids(result: ProverResult) -> list[int]:
    """Extract the clause IDs used in the proof, in order."""
    return [c["id"] for c in result.proof_clauses]


def extract_proof_justifications(result: ProverResult) -> list[str]:
    """Extract justification strings from the proof."""
    return [c["justification"] for c in result.proof_clauses]
