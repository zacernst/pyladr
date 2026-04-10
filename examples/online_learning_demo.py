#!/usr/bin/env python3
"""
Demonstration of PyLADR's online learning capabilities.

This script shows how to use the new --online-learning flag to enable
real-time contrastive learning during theorem proving. The system learns
from proof search experience and adapts its clause selection strategy
in real-time.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_prover(args, description):
    """Run prover with given arguments and return result info."""
    print(f"\n🔍 {description}")
    print("=" * 60)

    start_time = time.time()
    result = subprocess.run(
        ["python3", "-m", "pyladr.apps.prover9"] + args,
        capture_output=True,
        text=True
    )
    end_time = time.time()

    # Extract key statistics
    output = result.stdout + result.stderr
    proof_found = "THEOREM PROVED" in output
    given_count = 0
    generated_count = 0

    for line in output.split('\n'):
        if line.startswith("Given="):
            # Parse: "Given=X. Generated=Y. Kept=Z. proofs=W."
            parts = line.split('.')
            for part in parts:
                part = part.strip()
                if part.startswith("Given="):
                    given_count = int(part.split('=')[1])
                elif part.startswith("Generated="):
                    generated_count = int(part.split('=')[1])

    return {
        'proof_found': proof_found,
        'given_count': given_count,
        'generated_count': generated_count,
        'time': end_time - start_time,
        'output': output
    }

def create_test_problem():
    """Create a moderately difficult test problem."""
    problem = """set(auto).

formulas(sos).
% Basic group theory axioms
% Identity element
e * x = x.
x * e = x.

% Inverse elements
x * i(x) = e.
i(x) * x = e.

% Associativity
(x * y) * z = x * (y * z).

% Goal: prove that inverse of inverse is identity
% i(i(x)) = x
end_of_list.

formulas(goals).
i(i(a)) = a.
end_of_list.
"""

    problem_path = Path("temp_group_theory.in")
    problem_path.write_text(problem)
    return problem_path

def main():
    print("🧠 PyLADR Online Learning Demo")
    print("=" * 60)
    print()
    print("This demonstrates PyLADR's new real-time contrastive learning")
    print("capabilities. The system learns from proof search experience")
    print("and adapts its clause selection strategy during the search.")
    print()

    # Create test problem
    problem_file = create_test_problem()
    print(f"📝 Created test problem: {problem_file}")

    try:
        # Test traditional mode
        print("\n" + "="*60)
        print("🔄 TRADITIONAL MODE (Static Selection)")
        print("="*60)

        traditional_result = run_prover([
            "-f", str(problem_file),
            "-max_given", "100",
            "--quiet"
        ], "Traditional static clause selection")

        # Test online learning mode
        print("\n" + "="*60)
        print("🧠 ONLINE LEARNING MODE (Adaptive Selection)")
        print("="*60)

        online_result = run_prover([
            "-f", str(problem_file),
            "--online-learning",
            "-max_given", "100",
            "--quiet"
        ], "Online learning with real-time adaptation")

        # Compare results
        print("\n" + "="*60)
        print("📊 COMPARISON RESULTS")
        print("="*60)

        print("\n🔄 Traditional Mode:")
        print(f"   Proof found: {'✅' if traditional_result['proof_found'] else '❌'}")
        print(f"   Given clauses: {traditional_result['given_count']}")
        print(f"   Generated clauses: {traditional_result['generated_count']}")
        print(f"   Time: {traditional_result['time']:.3f}s")

        print("\n🧠 Online Learning Mode:")
        print(f"   Proof found: {'✅' if online_result['proof_found'] else '❌'}")
        print(f"   Given clauses: {online_result['given_count']}")
        print(f"   Generated clauses: {online_result['generated_count']}")
        print(f"   Time: {online_result['time']:.3f}s")

        # Analysis
        print("\n💡 Analysis:")
        if online_result['proof_found'] and traditional_result['proof_found']:
            if online_result['given_count'] < traditional_result['given_count']:
                print(f"   ✨ Online learning found proof with {traditional_result['given_count'] - online_result['given_count']} fewer given clauses!")
            elif online_result['given_count'] > traditional_result['given_count']:
                print(f"   📊 Traditional mode was more efficient by {online_result['given_count'] - traditional_result['given_count']} given clauses")
            else:
                print("   🤝 Both approaches used the same number of given clauses")
        elif online_result['proof_found'] and not traditional_result['proof_found']:
            print("   🎯 Online learning found a proof where traditional mode failed!")
        elif traditional_result['proof_found'] and not online_result['proof_found']:
            print("   📈 Traditional mode found a proof where online learning didn't")
        else:
            print("   🤔 Neither approach found a proof within the given limits")

        print("\n🔧 How to use online learning in your problems:")
        print("   python3 -m pyladr.apps.prover9 --online-learning -f your_problem.in")
        print("   python3 -m pyladr.apps.prover9 --help  # See all options")

    finally:
        # Cleanup
        problem_file.unlink(missing_ok=True)
        print(f"\n🧹 Cleaned up {problem_file}")

if __name__ == "__main__":
    main()