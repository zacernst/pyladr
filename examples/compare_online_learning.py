#!/usr/bin/env python3
"""Compare traditional vs online-learning mode across a suite of problems.

Runs each problem twice — once with static selection and once with
``--online-learning`` — and prints a side-by-side summary table.

Usage:
    python3 examples/compare_online_learning.py
    python3 examples/compare_online_learning.py --max-given 200
    python3 examples/compare_online_learning.py --problems lattice_distributivity group_inverse_product
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROBLEM_DIR = Path(__file__).resolve().parent / "sample_problems" / "online_learning"

# Default problem ordering: easy -> hard
DEFAULT_PROBLEMS = [
    "group_inverse_product",
    "set_theory_subset",
    "lattice_distributivity",
    "order_theory_lattice",
    "ring_nilpotent",
    "boolean_sheffer",
]


def run_prover(problem_path: Path, online: bool, max_given: int) -> dict:
    """Run the prover on a single problem and extract statistics."""
    args = [
        sys.executable, "-m", "pyladr.apps.prover9",
        "-f", str(problem_path),
        "-max_given", str(max_given),
        "--quiet",
    ]
    if online:
        args.append("--online-learning")

    start = time.monotonic()
    result = subprocess.run(args, capture_output=True, text=True, timeout=120)
    elapsed = time.monotonic() - start

    output = result.stdout + result.stderr
    proof_found = "THEOREM PROVED" in output

    given = generated = kept = 0
    for line in output.splitlines():
        if line.startswith("Given="):
            for part in line.split("."):
                part = part.strip()
                if part.startswith("Given="):
                    given = int(part.split("=")[1])
                elif part.startswith("Generated="):
                    generated = int(part.split("=")[1])
                elif part.startswith("Kept="):
                    kept = int(part.split("=")[1])

    return {
        "proof": proof_found,
        "given": given,
        "generated": generated,
        "kept": kept,
        "time": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare online learning vs traditional mode")
    parser.add_argument(
        "--max-given", type=int, default=200,
        help="Maximum given clauses per run (default: 200)",
    )
    parser.add_argument(
        "--problems", nargs="*", default=None,
        help="Specific problem names to run (without .in extension)",
    )
    args = parser.parse_args()

    problems = args.problems or DEFAULT_PROBLEMS

    print("PyLADR Online Learning Comparison")
    print("=" * 78)
    print(f"Max given clauses: {args.max_given}")
    print()

    # Header
    fmt = "{:<26s} {:>6s} {:>6s} {:>7s}  {:>6s} {:>6s} {:>7s}  {:>7s}"
    print(fmt.format(
        "Problem", "Proof", "Given", "Time",
        "Proof", "Given", "Time", "Savings",
    ))
    print(fmt.format(
        "", "--- Traditional ---", "", "",
        "--- Online Learning ---", "", "", "",
    ))
    print("-" * 78)

    row_fmt = "{:<26s} {:>6s} {:>6d} {:>6.2f}s  {:>6s} {:>6d} {:>6.2f}s  {:>7s}"

    for name in problems:
        path = PROBLEM_DIR / f"{name}.in"
        if not path.exists():
            print(f"  [skip] {name}.in not found")
            continue

        # Traditional
        try:
            trad = run_prover(path, online=False, max_given=args.max_given)
        except subprocess.TimeoutExpired:
            trad = {"proof": False, "given": 0, "generated": 0, "kept": 0, "time": 120.0}

        # Online learning
        try:
            ol = run_prover(path, online=True, max_given=args.max_given)
        except subprocess.TimeoutExpired:
            ol = {"proof": False, "given": 0, "generated": 0, "kept": 0, "time": 120.0}

        # Compute savings
        if trad["proof"] and ol["proof"] and trad["given"] > 0:
            pct = (1.0 - ol["given"] / trad["given"]) * 100
            savings = f"{pct:+.0f}%"
        elif ol["proof"] and not trad["proof"]:
            savings = "OL wins"
        elif trad["proof"] and not ol["proof"]:
            savings = "Trad wins"
        else:
            savings = "---"

        print(row_fmt.format(
            name[:26],
            "yes" if trad["proof"] else "no", trad["given"], trad["time"],
            "yes" if ol["proof"] else "no", ol["given"], ol["time"],
            savings,
        ))

    print()
    print("Savings = reduction in given clauses with online learning (positive = better).")
    print("Use --max-given N to adjust search depth.")


if __name__ == "__main__":
    main()
