#!/usr/bin/env python3
"""
Test the new vampire.in file with different approaches
"""

import subprocess
import sys
from pathlib import Path

def run_test(description, cmd):
    """Run a test and capture results."""
    print(f"\n🧪 {description}")
    print("─" * 60)
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Extract key metrics
    proof_found = "PROOF FOUND" in output
    given_clauses = None
    generated = None
    kept = None

    # Parse the output for metrics
    for line in output.split('\n'):
        if "Given Clauses:" in line:
            given_clauses = int(line.split(':')[1].strip())
        elif "Generated:" in line:
            generated = int(line.split(':')[1].strip())
        elif "Kept:" in line:
            kept = int(line.split(':')[1].strip())

    print(f"Results:")
    print(f"  Proof Found: {'✅ YES' if proof_found else '❌ NO'}")
    if given_clauses is not None:
        print(f"  Given Clauses: {given_clauses}")
    if generated is not None:
        print(f"  Generated: {generated}")
    if kept is not None:
        print(f"  Kept: {kept}")

    return proof_found, given_clauses, generated, kept

def main():
    if not Path("examples/vampire.in").exists():
        print("❌ examples/vampire.in not found")
        return 1

    print("🧛 New Vampire.in Test Results")
    print("=" * 60)
    print("Testing the new vampire.in file with various approaches...")

    # Test 1: Pure traditional (pyprover9)
    run_test(
        "Traditional PyProver9 (No ML)",
        ["python3", "-m", "pyladr.apps.prover9", "-f", "examples/vampire.in", "--quiet"]
    )

    # Test 2: Enhanced ML model
    run_test(
        "Enhanced ML Model (3,775 training pairs)",
        ["python3", "examples/simple_ml_usage.py", "examples/vampire.in", "--model", "vampire_enhanced_model.pt"]
    )

    # Test 3: Goal-aware ML model
    run_test(
        "Goal-Aware ML Model (2,095 goal-focused pairs)",
        ["python3", "examples/simple_ml_usage.py", "examples/vampire.in", "--model", "vampire_goal_aware_model.pt"]
    )

    print(f"\n💡 Analysis:")
    print("• All approaches should be compared for efficiency")
    print("• The new goal structure uses variables (x,y,z,w,v) vs constants")
    print("• Goal is positive P(...) instead of negative -P(...)")
    print("• This may be inherently easier than the original problem")

    return 0

if __name__ == "__main__":
    sys.exit(main())