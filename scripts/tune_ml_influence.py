#!/usr/bin/env python3
"""
Demonstration of tuning ML influence on clause selection.

Shows how different ml_weight settings affect theorem proving behavior.
"""

import subprocess
import re
from pathlib import Path

def extract_ml_selections(output_text):
    """Count ML vs traditional selections from prover output."""
    pattern = r'given #(\d+) \(([^,]+),wt=([^)]+)\)'

    traditional = 0
    ml_guided = 0

    for match in re.finditer(pattern, output_text):
        selection_type = match.group(2)
        if 'ML' in selection_type:
            ml_guided += 1
        else:
            traditional += 1

    return traditional, ml_guided

def test_ml_weight(weight_value, max_given=12):
    """Test prover with specific ml_weight setting."""

    print(f"\n🎯 Testing ml_weight = {weight_value}")
    print("-" * 50)

    # Create temporary config modification script
    config_script = f"""
# Temporary modification for testing
import sys
sys.path.insert(0, '.')

from pyladr.apps.prover9 import main
from pyladr.search.ml_selection import MLSelectionConfig
from pyladr.search.online_integration import OnlineIntegrationConfig

# Monkey patch to override ml_weight
original_ml_config = None
original_online_config = None

def patched_ml_config(**kwargs):
    global original_ml_config
    if original_ml_config is None:
        original_ml_config = MLSelectionConfig
    return original_ml_config(enabled=True, ml_weight={weight_value}, **kwargs)

def patched_online_config(**kwargs):
    global original_online_config
    if original_online_config is None:
        original_online_config = OnlineIntegrationConfig
    return original_online_config(
        enabled=True,
        adaptive_ml_weight=False,  # Disable adaptive to use fixed weight
        initial_ml_weight={weight_value},
        max_ml_weight={weight_value},
        min_given_before_ml=3,  # Start ML early to see effect
        **kwargs
    )

# Apply patches
import pyladr.search.ml_selection as ml_sel_mod
import pyladr.search.online_integration as online_mod
ml_sel_mod.MLSelectionConfig = patched_ml_config
online_mod.OnlineIntegrationConfig = patched_online_config

# Run with patched config
import sys
sys.argv = ['prover9', '--online-learning', '-f', 'vampire.in', '-max_given', str({max_given})]
try:
    main()
except SystemExit:
    pass
"""

    # Write and run test
    with open('temp_config_test.py', 'w') as f:
        f.write(config_script)

    try:
        result = subprocess.run(['python3', 'temp_config_test.py'],
                               capture_output=True, text=True)
        output = result.stdout + result.stderr

        traditional, ml_guided = extract_ml_selections(output)
        total = traditional + ml_guided

        if total > 0:
            ml_percentage = (ml_guided / total) * 100
            print(f"   Traditional selections: {traditional}")
            print(f"   ML-guided selections: {ml_guided}")
            print(f"   ML influence: {ml_percentage:.1f}%")

            # Look for specific patterns
            t_plus_ml = output.count('T+ML')
            a_plus_ml = output.count('A+ML')
            print(f"   T+ML selections: {t_plus_ml}")
            print(f"   A+ML selections: {a_plus_ml}")
        else:
            print(f"   No selections captured (check output)")

        return traditional, ml_guided, output

    finally:
        Path('temp_config_test.py').unlink(missing_ok=True)

def demonstrate_ml_influence_tuning():
    """Show how different ml_weight values affect selection behavior."""

    print("🎛️ ML INFLUENCE TUNING DEMONSTRATION")
    print("=" * 60)
    print()
    print("Testing different ml_weight values to show their impact")
    print("on clause selection behavior in PyLADR's online learning system.")
    print()

    # Test different weight values
    weight_values = [0.1, 0.3, 0.5, 0.7]
    results = []

    for weight in weight_values:
        trad, ml, output = test_ml_weight(weight, max_given=15)
        results.append((weight, trad, ml, output))

    # Summary analysis
    print(f"\n" + "=" * 60)
    print("📊 INFLUENCE COMPARISON SUMMARY")
    print("=" * 60)

    print(f"\n{'Weight':>8} {'Traditional':>12} {'ML-Guided':>11} {'ML %':>8}")
    print("-" * 45)

    for weight, trad, ml, _ in results:
        total = trad + ml
        ml_pct = (ml / total * 100) if total > 0 else 0
        print(f"{weight:>8.1f} {trad:>12} {ml:>11} {ml_pct:>7.1f}%")

    print(f"\n💡 Interpretation:")
    print(f"   • Lower weights (0.1-0.3): Conservative ML influence")
    print(f"   • Medium weights (0.4-0.6): Balanced ML/traditional blend")
    print(f"   • Higher weights (0.7-1.0): Aggressive ML guidance")
    print(f"   • Weight 0.0: Pure traditional (no ML)")
    print(f"   • Weight 1.0: Pure ML (no traditional)")

if __name__ == "__main__":
    demonstrate_ml_influence_tuning()