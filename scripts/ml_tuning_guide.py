#!/usr/bin/env python3
"""
ML Influence Tuning Guide for PyLADR

This script shows you exactly where and how to adjust ML influence parameters.
"""

def print_tuning_guide():
    print("🎛️ ML INFLUENCE TUNING GUIDE")
    print("=" * 70)
    print()
    print("You can adjust ML influence through several key parameters:")
    print()

    print("📍 1. PRIMARY CONTROL: ml_weight")
    print("-" * 40)
    print("FILE: pyladr/apps/prover9.py (line ~691)")
    print("CURRENT: ml_config = MLSelectionConfig(enabled=True, ml_weight=0.2)")
    print()
    print("EFFECT:")
    print("  • 0.0 = Pure traditional selection (0% ML)")
    print("  • 0.1 = Very light ML influence (10% ML, 90% traditional)")
    print("  • 0.2 = Light ML influence (20% ML, 80% traditional) ← CURRENT")
    print("  • 0.3 = Moderate ML influence (30% ML, 70% traditional)")
    print("  • 0.5 = Balanced blend (50% ML, 50% traditional)")
    print("  • 0.7 = Heavy ML influence (70% ML, 30% traditional)")
    print("  • 1.0 = Pure ML selection (100% ML)")
    print()

    print("📍 2. ADAPTIVE LEARNING WEIGHTS")
    print("-" * 40)
    print("FILE: pyladr/apps/prover9.py (lines ~701-702)")
    print("CURRENT:")
    print("  initial_ml_weight=0.1,  # Starting influence")
    print("  max_ml_weight=0.5,      # Maximum influence as system learns")
    print()
    print("EFFECT:")
    print("  • System starts with 10% ML influence")
    print("  • Gradually increases up to 50% as it learns")
    print("  • Set adaptive_ml_weight=False to use fixed ml_weight")
    print()

    print("📍 3. ML ACTIVATION THRESHOLD")
    print("-" * 40)
    print("FILE: pyladr/apps/prover9.py (line ~703)")
    print("CURRENT: min_given_before_ml=10")
    print()
    print("EFFECT:")
    print("  • ML starts influencing after 10 given clauses")
    print("  • Lower values = ML starts earlier")
    print("  • Higher values = More traditional baseline before ML kicks in")
    print()

    print("📍 4. ADVANCED ML SELECTION PARAMETERS")
    print("-" * 40)
    print("FILE: pyladr/search/ml_selection.py (MLSelectionConfig)")
    print("AVAILABLE PARAMETERS:")
    print("  • diversity_weight: Weight of diversity in ML scoring (default 0.5)")
    print("  • proof_potential_weight: Weight of proof potential (default 0.5)")
    print("  • diversity_window: Recent givens tracked for diversity (default 20)")
    print("  • min_sos_for_ml: Min SOS size before ML activates (default 10)")
    print()

    print("🔧 HOW TO MODIFY")
    print("=" * 70)
    print()
    print("QUICK ADJUSTMENT (recommended):")
    print("1. Edit pyladr/apps/prover9.py line 691:")
    print("   FROM: ml_config = MLSelectionConfig(enabled=True, ml_weight=0.2)")
    print("   TO:   ml_config = MLSelectionConfig(enabled=True, ml_weight=0.5)")
    print()
    print("2. For adaptive weights, edit lines 701-702:")
    print("   FROM: initial_ml_weight=0.1, max_ml_weight=0.5")
    print("   TO:   initial_ml_weight=0.2, max_ml_weight=0.8")
    print()
    print("3. To disable adaptive adjustment (use fixed weight):")
    print("   Change line 700: adaptive_ml_weight=False")
    print()

    print("🧪 TESTING DIFFERENT INFLUENCE LEVELS")
    print("=" * 70)

    test_cases = [
        (0.0, "Pure traditional", "No ML influence - baseline performance"),
        (0.1, "Minimal ML", "Very conservative ML guidance"),
        (0.2, "Light ML", "Current default - gentle ML influence"),
        (0.3, "Moderate ML", "Balanced traditional/ML blend"),
        (0.5, "Equal blend", "50/50 traditional and ML guidance"),
        (0.7, "Heavy ML", "Strong ML influence over traditional"),
        (1.0, "Pure ML", "Maximum ML guidance - experimental"),
    ]

    print("\nRecommended values to try:")
    for weight, name, description in test_cases:
        print(f"  {weight:3.1f} - {name:12} : {description}")

    print()
    print("💡 PRACTICAL RECOMMENDATIONS")
    print("=" * 70)
    print("• START WITH: ml_weight=0.3 for most problems")
    print("• CONSERVATIVE: ml_weight=0.1-0.2 for critical proofs")
    print("• AGGRESSIVE: ml_weight=0.5-0.7 for exploration")
    print("• EXPERIMENTAL: ml_weight=0.8+ for research")
    print()
    print("• Keep adaptive_ml_weight=True for automatic tuning")
    print("• Lower min_given_before_ml=5 to see ML effects sooner")
    print("• Set log_integration_events=True for detailed monitoring")

if __name__ == "__main__":
    print_tuning_guide()