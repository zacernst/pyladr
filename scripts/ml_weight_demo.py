#!/usr/bin/env python3
"""
Demonstrate the new --ml-weight command-line option.
"""

import subprocess
import re

def extract_ml_stats(output):
    """Extract ML vs traditional selection statistics."""
    pattern = r'given #(\d+) \(([^,]+),wt=([^)]+)\)'

    traditional = 0
    ml_guided = 0

    for match in re.finditer(pattern, output):
        selection_type = match.group(2)
        if 'ML' in selection_type:
            ml_guided += 1
        else:
            traditional += 1

    total = traditional + ml_guided
    ml_percentage = (ml_guided / total * 100) if total > 0 else 0

    return traditional, ml_guided, ml_percentage

def test_weight(weight):
    """Test a specific ML weight value."""
    cmd = [
        'python3', '-m', 'pyladr.apps.prover9',
        '--online-learning', '--ml-weight', str(weight),
        '-f', 'vampire.in', '-max_given', '12'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    trad, ml, pct = extract_ml_stats(output)

    # Show configuration message
    config_line = [line for line in output.split('\n') if '🎛️ Using fixed ML weight' in line]
    config_msg = config_line[0] if config_line else f"ML weight: {weight}"

    # Show some ML selections for comparison
    ml_selections = [line for line in output.split('\n') if 'T+ML' in line or 'A+ML' in line]

    return {
        'weight': weight,
        'config': config_msg,
        'traditional': trad,
        'ml_guided': ml,
        'percentage': pct,
        'ml_examples': ml_selections[:3]  # First 3 ML selections
    }

def main():
    print("🎛️ NEW FEATURE: --ml-weight Command-Line Option")
    print("=" * 60)
    print()
    print("You can now control ML influence directly from the command line!")
    print()

    # Test different weights
    weights = [0.1, 0.3, 0.6, 0.9]
    results = []

    for weight in weights:
        print(f"Testing --ml-weight {weight}...")
        result = test_weight(weight)
        results.append(result)

    # Show comparison table
    print(f"\n" + "=" * 60)
    print("📊 ML INFLUENCE COMPARISON")
    print("=" * 60)

    print(f"\n{'Weight':>8} {'Traditional':>12} {'ML-Guided':>11} {'ML %':>8} {'Example Selections'}")
    print("-" * 85)

    for r in results:
        examples = ' | '.join([ex.split(':')[0].strip() for ex in r['ml_examples']][:2])
        if not examples:
            examples = "(no ML selections)"

        print(f"{r['weight']:>8.1f} {r['traditional']:>12} {r['ml_guided']:>11} {r['percentage']:>7.1f}% {examples}")

    print(f"\n💡 USAGE EXAMPLES:")
    print("=" * 60)
    print()
    print("# Light ML influence (10%)")
    print("python3 -m pyladr.apps.prover9 --online-learning --ml-weight 0.1 -f problem.in")
    print()
    print("# Balanced ML influence (50%)")
    print("python3 -m pyladr.apps.prover9 --online-learning --ml-weight 0.5 -f problem.in")
    print()
    print("# Heavy ML influence (80%)")
    print("python3 -m pyladr.apps.prover9 --online-learning --ml-weight 0.8 -f problem.in")
    print()
    print("# Pure ML (experimental - 100%)")
    print("python3 -m pyladr.apps.prover9 --online-learning --ml-weight 1.0 -f problem.in")
    print()
    print("# Default adaptive (no --ml-weight flag)")
    print("python3 -m pyladr.apps.prover9 --online-learning -f problem.in")
    print("  → Uses adaptive weight: 0.1 gradually increasing to 0.5")

if __name__ == "__main__":
    main()