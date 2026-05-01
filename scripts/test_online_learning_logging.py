#!/usr/bin/env python3
"""
Test the enhanced online learning logging system.

This script demonstrates all the logging features added to show
how online learning proceeds during theorem proving.
"""

import subprocess
import sys

def run_with_logging():
    """Run the prover with enhanced online learning logging."""

    print("🧠 Enhanced Online Learning Logging Demo")
    print("=" * 60)
    print()
    print("This demonstrates the new logging system that shows:")
    print("• Online learning initialization")
    print("• Search startup with ML hooks")
    print("• Experience collection progress")
    print("• Model update decisions and outcomes")
    print("• Learning trigger evaluations")
    print("• ML weight adaptations")
    print()

    cmd = [
        'python3', '-c', '''
from pyladr.apps.prover9 import main
import sys

sys.argv = ["prover9", "--online-learning", "--ml-weight", "0.5", "-f", "vampire.in", "-max_given", "25"]
try:
    main()
except SystemExit:
    pass
'''
    ]

    print("Running: python3 -m pyladr.apps.prover9 --online-learning --ml-weight 0.5 -f vampire.in -max_given 25")
    print()
    print("🔍 Watch for these logging indicators:")
    print("📚 = Online learning initialization")
    print("🚀 = Search startup")
    print("📊 = Learning progress updates")
    print("💾 = Experience collection milestones")
    print("📈 = Learning trigger evaluations")
    print("🧠 = Model update results")
    print("🎛️ = ML weight configuration")
    print()
    print("-" * 60)

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Extract and display only our logging messages and key search events
    lines = output.split('\n')

    for line in lines:
        # Show our custom logging messages
        if any(indicator in line for indicator in ['📚', '🚀', '📊', '💾', '📈', '🧠', '🎛️']):
            print(line)
        # Show given clause selections to see ML in action
        elif 'given #' in line and ('T+ML' in line or 'A+ML' in line):
            print(f"  🎯 {line.strip()}")
        # Show proof result
        elif 'THEOREM PROVED' in line:
            print(f"  ✅ {line.strip()}")
        elif 'Search stopped' in line:
            print(f"  🛑 {line.strip()}")

    print()
    print("-" * 60)
    print("💡 Key Observations:")
    print("• Initialization shows buffer capacity and learning thresholds")
    print("• Experience collection happens as clauses are kept during inference")
    print("• Learning triggers evaluate when enough experience is collected")
    print("• Model updates show acceptance/rejection with weight adjustments")
    print("• ML-guided selections (T+ML, A+ML) demonstrate active learning")

if __name__ == "__main__":
    run_with_logging()