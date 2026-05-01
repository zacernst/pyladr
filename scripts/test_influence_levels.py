#!/usr/bin/env python3
"""
Simple test showing how to modify ML influence and see the effects.
"""

def show_current_vs_modified():
    print("🎛️ ML INFLUENCE ADJUSTMENT DEMO")
    print("=" * 60)
    print()
    print("CURRENT SYSTEM CONFIGURATION:")
    print("  • ml_weight = 0.2 (20% ML, 80% traditional)")
    print("  • adaptive_ml_weight = True (grows from 0.1 to 0.5)")
    print("  • min_given_before_ml = 10 (ML starts after 10 clauses)")
    print()
    print("EFFECT: Light ML influence that grows over time")
    print()

    print("🔧 TO INCREASE ML INFLUENCE:")
    print("-" * 40)
    print("Edit pyladr/apps/prover9.py:")
    print()
    print("OPTION 1 - Fixed Higher Weight:")
    print("  Change line 691:")
    print("    FROM: ml_weight=0.2")
    print("    TO:   ml_weight=0.5    # 50% ML influence")
    print()
    print("  Change line 700:")
    print("    FROM: adaptive_ml_weight=True")
    print("    TO:   adaptive_ml_weight=False  # Use fixed weight")
    print()

    print("OPTION 2 - Aggressive Adaptive:")
    print("  Keep adaptive_ml_weight=True")
    print("  Change lines 701-702:")
    print("    FROM: initial_ml_weight=0.1, max_ml_weight=0.5")
    print("    TO:   initial_ml_weight=0.3, max_ml_weight=0.8")
    print()

    print("OPTION 3 - Earlier ML Start:")
    print("  Change line 703:")
    print("    FROM: min_given_before_ml=10")
    print("    TO:   min_given_before_ml=3    # ML starts sooner")
    print()

    print("🔧 TO DECREASE ML INFLUENCE:")
    print("-" * 40)
    print("CONSERVATIVE SETTINGS:")
    print("  ml_weight=0.1                  # Very light ML")
    print("  adaptive_ml_weight=False        # No growth")
    print("  min_given_before_ml=25          # Later ML start")
    print()

    print("PURE TRADITIONAL (NO ML):")
    print("  ml_weight=0.0                  # Disable ML completely")
    print()

    print("🧪 QUICK TEST AFTER MODIFICATION:")
    print("-" * 40)
    print("Run this to see the changes:")
    print("  python3 -m pyladr.apps.prover9 --online-learning -f vampire.in -max_given 12")
    print()
    print("Look for:")
    print("  • More/fewer 'T+ML' and 'A+ML' selections")
    print("  • Different clause choices compared to before")
    print("  • Earlier/later appearance of ML selections")

if __name__ == "__main__":
    show_current_vs_modified()