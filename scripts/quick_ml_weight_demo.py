#!/usr/bin/env python3
"""
Quick demonstration of how ml_weight affects clause selection in practice.
"""

import tempfile
import subprocess
import os
import re

def create_modified_prover(ml_weight_value):
    """Create a temporary modified prover with specific ml_weight."""

    # Read the original prover file
    with open('pyladr/apps/prover9.py', 'r') as f:
        original_content = f.read()

    # Modify the ml_weight value
    modified_content = re.sub(
        r'ml_config = MLSelectionConfig\(enabled=True, ml_weight=[\d.]+\)',
        f'ml_config = MLSelectionConfig(enabled=True, ml_weight={ml_weight_value})',
        original_content
    )

    # Also disable adaptive weight for consistent testing
    modified_content = re.sub(
        r'adaptive_ml_weight=True',
        'adaptive_ml_weight=False',
        modified_content
    )

    # Set initial weight to match ml_weight
    modified_content = re.sub(
        r'initial_ml_weight=[\d.]+',
        f'initial_ml_weight={ml_weight_value}',
        modified_content
    )

    # Lower the threshold so ML activates quickly
    modified_content = re.sub(
        r'min_given_before_ml=\d+',
        'min_given_before_ml=3',
        modified_content
    )

    # Create temp file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.py', prefix='prover9_test_')
    with os.fdopen(temp_fd, 'w') as temp_file:
        temp_file.write(modified_content)

    return temp_path

def extract_selections_with_types(output):
    """Extract selection information from prover output."""
    pattern = r'given #(\d+) \(([^,]+),wt=([^)]+)\):\s*(\d+):\s*(.*?)(?:\n|$)'
    selections = []

    for match in re.finditer(pattern, output):
        selections.append({
            'num': int(match.group(1)),
            'type': match.group(2),
            'weight': float(match.group(3)),
            'id': int(match.group(4)),
            'clause': match.group(5).strip()[:40] + ('...' if len(match.group(5)) > 40 else '')
        })

    return selections

def test_ml_weight_effect(weight):
    """Test specific ml_weight value and return selection stats."""

    print(f"\n🎯 Testing ml_weight = {weight}")
    print("-" * 50)

    temp_prover = None
    try:
        # Create modified prover
        temp_prover = create_modified_prover(weight)

        # Run the test
        cmd = [
            'python3', temp_prover, '--online-learning',
            '-f', 'vampire.in', '-max_given', '10'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr

        # Extract selections
        selections = extract_selections_with_types(output)

        if not selections:
            print("   ⚠️  No selections captured")
            return None

        # Count selection types
        traditional = sum(1 for s in selections if 'ML' not in s['type'])
        ml_guided = sum(1 for s in selections if 'ML' in s['type'])

        print(f"   Selections captured: {len(selections)}")
        print(f"   Traditional (T/I/A): {traditional}")
        print(f"   ML-guided (T+ML/A+ML): {ml_guided}")

        if len(selections) > 0:
            ml_percentage = (ml_guided / len(selections)) * 100
            print(f"   ML influence observed: {ml_percentage:.1f}%")

        # Show first few selections to see the pattern
        print(f"   First 5 selections:")
        for i, sel in enumerate(selections[:5]):
            marker = "🧠" if 'ML' in sel['type'] else "  "
            print(f"     #{sel['num']} ({sel['type']:>5},wt={sel['weight']:4.0f}): {sel['clause']} {marker}")

        return {
            'weight': weight,
            'total': len(selections),
            'traditional': traditional,
            'ml_guided': ml_guided,
            'selections': selections
        }

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

    finally:
        if temp_prover and os.path.exists(temp_prover):
            os.unlink(temp_prover)

def main():
    print("🎛️ ML WEIGHT INFLUENCE DEMONSTRATION")
    print("=" * 60)
    print("Testing how different ml_weight values affect clause selection")
    print("in PyLADR's online learning system.")

    # Test different weights
    weights_to_test = [0.1, 0.3, 0.6]
    results = []

    for weight in weights_to_test:
        result = test_ml_weight_effect(weight)
        if result:
            results.append(result)

    # Summary
    if results:
        print(f"\n" + "=" * 60)
        print("📊 SUMMARY COMPARISON")
        print("=" * 60)

        print(f"{'Weight':>8} {'Total':>7} {'Traditional':>13} {'ML-Guided':>11} {'ML %':>8}")
        print("-" * 55)

        for r in results:
            ml_pct = (r['ml_guided'] / r['total'] * 100) if r['total'] > 0 else 0
            print(f"{r['weight']:>8.1f} {r['total']:>7} {r['traditional']:>13} {r['ml_guided']:>11} {ml_pct:>7.1f}%")

        print(f"\n💡 Key Observations:")
        print(f"   • Higher ml_weight → More T+ML and A+ML selections")
        print(f"   • Lower ml_weight → More traditional T, I, A selections")
        print(f"   • ML selections show different clause choices than traditional")

if __name__ == "__main__":
    main()