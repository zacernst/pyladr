#!/usr/bin/env python3
"""
Validate that ML contrastive online learning is actually affecting clause selection.

This script provides concrete evidence that the --online-learning flag produces
genuine ML-guided decisions, not just placeholder labels.
"""

import subprocess
import sys
import re
import json
from pathlib import Path

def extract_selection_sequence(output: str) -> list[tuple[int, str, int, float]]:
    """Extract (given_num, selection_type, clause_id, weight) from prover output."""
    selections = []

    # Pattern: "given #N (TYPE[+ML],wt=W): ID: clause"
    pattern = r'given #(\d+) \(([^,]+),wt=([^)]+)\):\s*(\d+):'

    # Debug: print lines that contain "given #"
    given_lines = [line for line in output.split('\n') if 'given #' in line]
    if len(given_lines) > 0:
        print(f"   DEBUG: Found {len(given_lines)} lines with 'given #'")
        for i, line in enumerate(given_lines[:3]):
            print(f"   DEBUG: Line {i+1}: {repr(line)}")

    for match in re.finditer(pattern, output):
        given_num = int(match.group(1))
        sel_type = match.group(2)
        weight = float(match.group(3))
        clause_id = int(match.group(4))
        selections.append((given_num, sel_type, clause_id, weight))

    print(f"   DEBUG: Extracted {len(selections)} selections")
    return selections

def run_search(problem_file: str, online_learning: bool = False, max_given: int = 25) -> dict:
    """Run search and return analysis of selection behavior."""

    cmd = ["python3", "-m", "pyladr.apps.prover9", "-f", str(problem_file), "-max_given", str(max_given)]
    if online_learning:
        cmd.append("--online-learning")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")

    # Combine stdout and stderr since some output may go to stderr
    full_output = result.stdout + result.stderr

    if result.returncode not in [0, 3]:  # 0=proof, 3=max_given reached
        print(f"   WARNING: Search returned code {result.returncode}")
        print(f"   STDERR: {result.stderr[:200]}...")

    selections = extract_selection_sequence(full_output)

    # Analyze selection types
    selection_types = [sel[1] for sel in selections]
    traditional_count = sum(1 for t in selection_types if '+ML' not in t)
    ml_count = sum(1 for t in selection_types if '+ML' in t)

    # Analyze weight trends for ML vs traditional
    ml_selections = [sel for sel in selections if '+ML' in sel[1]]
    traditional_selections = [sel for sel in selections if '+ML' not in sel[1]]

    ml_weights = [sel[3] for sel in ml_selections] if ml_selections else []
    trad_weights = [sel[3] for sel in traditional_selections] if traditional_selections else []

    return {
        'selections': selections,
        'selection_types': selection_types,
        'traditional_count': traditional_count,
        'ml_count': ml_count,
        'ml_weights': ml_weights,
        'traditional_weights': trad_weights,
        'total_selections': len(selections),
        'output': full_output,
        'stderr': result.stderr,
    }

def analyze_ml_impact(traditional_result: dict, ml_result: dict) -> dict:
    """Compare traditional vs ML results to detect learning impact."""

    # Compare selection sequences
    trad_seq = [f"id{sel[2]}(wt{sel[3]:.1f})" for sel in traditional_result['selections']]
    ml_seq = [f"id{sel[2]}(wt{sel[3]:.1f})" for sel in ml_result['selections']]

    # Find where selections diverge
    divergence_point = None
    for i, (t_sel, m_sel) in enumerate(zip(trad_seq, ml_seq)):
        if t_sel != m_sel:
            divergence_point = i + 1
            break

    # Analyze ML-specific selections
    ml_only_selections = [sel for sel in ml_result['selections'] if '+ML' in sel[1]]

    return {
        'sequences_identical': trad_seq == ml_seq,
        'divergence_point': divergence_point,
        'traditional_sequence': trad_seq,
        'ml_sequence': ml_seq,
        'ml_only_selections': ml_only_selections,
        'ml_selection_count': len(ml_only_selections),
        'proof_found_traditional': 'THEOREM PROVED' in traditional_result['output'],
        'proof_found_ml': 'THEOREM PROVED' in ml_result['output'],
    }

def validate_ml_selection_impact():
    """Main validation that ML is actually affecting clause selection."""

    print("🔍 Validating ML Contrastive Online Learning Impact")
    print("=" * 60)
    print()
    print("Testing whether --online-learning produces genuine ML-guided")
    print("clause selection decisions vs traditional static heuristics.")
    print()

    # Use vampire.in as our test problem
    test_problem = Path("vampire.in")
    if not test_problem.exists():
        print(f"❌ Test problem {test_problem} not found")
        return False

    print(f"📝 Test Problem: {test_problem}")
    print("🧪 Running traditional search...")

    # Run traditional search
    traditional_result = run_search(test_problem, online_learning=False, max_given=20)

    print("🧠 Running online learning search...")

    # Run ML search
    ml_result = run_search(test_problem, online_learning=True, max_given=20)

    # Analyze results
    analysis = analyze_ml_impact(traditional_result, ml_result)

    print("\n" + "=" * 60)
    print("🔎 VALIDATION RESULTS")
    print("=" * 60)

    print(f"\n📊 Selection Statistics:")
    print(f"   Traditional: {traditional_result['traditional_count']} traditional, 0 ML")
    print(f"   Online Learning: {ml_result['traditional_count']} traditional, {ml_result['ml_count']} ML")

    if ml_result['ml_count'] > 0:
        print(f"   ✅ ML selections detected: {ml_result['ml_count']} 'T+ML' or 'A+ML' selections")
        print(f"   📈 ML engagement: {100 * ml_result['ml_count'] / ml_result['total_selections']:.1f}%")
    else:
        print(f"   ⚠️  No ML selections found")

    print(f"\n🎯 Selection Behavior:")
    sequences_same = analysis['sequences_identical']
    if sequences_same:
        print("   ❌ Selection sequences identical - ML may not be affecting decisions")
    else:
        print(f"   ✅ Selection sequences differ - ML is changing clause selection!")
        if analysis['divergence_point']:
            print(f"   🔀 Sequences diverge at given clause #{analysis['divergence_point']}")

    # Show specific ML selections
    if analysis['ml_only_selections']:
        print(f"\n🧠 ML-Guided Selections:")
        for given_num, sel_type, clause_id, weight in analysis['ml_only_selections'][:5]:
            print(f"   given #{given_num}: {sel_type}, clause {clause_id}, weight {weight:.1f}")
        if len(analysis['ml_only_selections']) > 5:
            print(f"   ... and {len(analysis['ml_only_selections']) - 5} more ML selections")

    # Show divergence example
    if not sequences_same and analysis['divergence_point']:
        dp = analysis['divergence_point'] - 1  # Convert to 0-indexed
        print(f"\n🔀 First Divergence at Given #{analysis['divergence_point']}:")
        if dp < len(analysis['traditional_sequence']) and dp < len(analysis['ml_sequence']):
            print(f"   Traditional: {analysis['traditional_sequence'][dp]}")
            print(f"   Online Learning: {analysis['ml_sequence'][dp]}")

    # Overall assessment
    print(f"\n💡 Assessment:")
    ml_working = ml_result['ml_count'] > 0 and not sequences_same

    if ml_working:
        print("   ✅ ML contrastive online learning IS affecting clause selection!")
        print("   ✅ The 'T+ML' labels represent genuine ML-guided decisions")
        print("   ✅ Online learning is changing theorem proving strategy in real-time")
    else:
        if ml_result['ml_count'] == 0:
            print("   ❌ No ML selections found - learning may not be triggering")
            print("   💡 Try a harder problem or increase max_given for longer search")
        elif sequences_same:
            print("   ❌ ML selections present but not changing behavior")
            print("   💡 ML may need more experience or different parameters")
        else:
            print("   ⚠️  Mixed results - further investigation needed")

    return ml_working

def main():
    success = validate_ml_selection_impact()

    print(f"\n{'✅' if success else '❌'} Validation {'PASSED' if success else 'NEEDS ATTENTION'}")
    print()
    print("🔧 Next Steps:")
    if success:
        print("   - Try validation on different problem types")
        print("   - Test with longer searches (higher max_given)")
        print("   - Verify learning improves over time within single searches")
    else:
        print("   - Check ML configuration (ml_weight, min_sos_for_ml)")
        print("   - Verify embedding provider is functional")
        print("   - Test on problems requiring more given clauses")
        print("   - Check learning trigger thresholds")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())