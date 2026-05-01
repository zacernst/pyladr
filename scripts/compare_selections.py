#!/usr/bin/env python3
"""
Compare clause selections between traditional and ML modes.
"""

import sys
import os
sys.path.insert(0, '.')

from pyladr.apps.prover9 import main
import re
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

def run_mode(mode_name, use_online_learning=False, max_given=10):
    """Run prover in specified mode and capture given clause selections."""

    # Set up arguments
    args = ['prover9', '-f', 'vampire.in', '-max_given', str(max_given)]
    if use_online_learning:
        args.append('--online-learning')

    # Capture all output
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    # Save original sys.argv and replace
    original_argv = sys.argv[:]
    sys.argv = args

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            try:
                result = main()
            except SystemExit as e:
                result = e.code
    finally:
        sys.argv = original_argv

    # Get captured output
    full_output = stdout_capture.getvalue() + stderr_capture.getvalue()

    # Extract given clause information
    given_pattern = r'given #(\d+) \(([^,]+),wt=([^)]+)\):\s*(\d+):\s*(.*?)(?:\n|$)'
    selections = []

    for match in re.finditer(given_pattern, full_output):
        given_num = int(match.group(1))
        selection_type = match.group(2)
        weight = match.group(3)
        clause_id = int(match.group(4))
        clause_text = match.group(5).strip()

        selections.append({
            'given_num': given_num,
            'selection_type': selection_type,
            'weight': float(weight),
            'clause_id': clause_id,
            'clause_text': clause_text
        })

    return {
        'mode': mode_name,
        'selections': selections,
        'full_output': full_output,
        'result_code': result
    }

def compare_selections(trad_result, ml_result):
    """Compare the clause selections between two modes."""

    print("🔍 CLAUSE SELECTION COMPARISON")
    print("=" * 60)

    trad_sels = trad_result['selections']
    ml_sels = ml_result['selections']

    print(f"\n📊 Selection Counts:")
    print(f"   Traditional: {len(trad_sels)} clauses")
    print(f"   ML Mode: {len(ml_sels)} clauses")

    # Compare selection types
    trad_types = [sel['selection_type'] for sel in trad_sels]
    ml_types = [sel['selection_type'] for sel in ml_sels]

    print(f"\n🎯 Selection Types:")
    print(f"   Traditional: {set(trad_types)}")
    print(f"   ML Mode: {set(ml_types)}")

    if 'ML' in ''.join(ml_types):
        ml_count = sum(1 for t in ml_types if 'ML' in t)
        print(f"   ✅ ML selections found: {ml_count} out of {len(ml_types)}")

    # Find divergence point
    divergence_point = None
    for i in range(min(len(trad_sels), len(ml_sels))):
        trad_clause = trad_sels[i]['clause_id']
        ml_clause = ml_sels[i]['clause_id']

        if trad_clause != ml_clause:
            divergence_point = i + 1  # Convert to 1-indexed
            break

    print(f"\n🔀 Selection Comparison:")
    if divergence_point:
        print(f"   Sequences diverge at given clause #{divergence_point}")
        print(f"   ✅ ML is making different selection decisions!")
    else:
        print(f"   Selection sequences are identical up to min length")
        if len(trad_sels) != len(ml_sels):
            print(f"   But different total lengths: {len(trad_sels)} vs {len(ml_sels)}")

    # Show detailed comparison
    print(f"\n📋 DETAILED SELECTION COMPARISON:")
    print("-" * 60)

    max_len = max(len(trad_sels), len(ml_sels))

    for i in range(max_len):
        trad_sel = trad_sels[i] if i < len(trad_sels) else None
        ml_sel = ml_sels[i] if i < len(ml_sels) else None

        print(f"\nGiven #{i+1}:")

        if trad_sel:
            print(f"  TRAD: ({trad_sel['selection_type']},wt={trad_sel['weight']:.0f}) " +
                  f"clause {trad_sel['clause_id']}: {trad_sel['clause_text'][:50]}...")
        else:
            print(f"  TRAD: (no more selections)")

        if ml_sel:
            print(f"  ML:   ({ml_sel['selection_type']},wt={ml_sel['weight']:.0f}) " +
                  f"clause {ml_sel['clause_id']}: {ml_sel['clause_text'][:50]}...")
        else:
            print(f"  ML:   (no more selections)")

        # Highlight differences
        if trad_sel and ml_sel:
            if trad_sel['clause_id'] != ml_sel['clause_id']:
                print(f"        ⚡ DIFFERENT CLAUSES SELECTED! ⚡")
            elif trad_sel['selection_type'] != ml_sel['selection_type']:
                print(f"        🎯 Same clause, different selection mechanism!")

    return divergence_point is not None

def main():
    print("🧠 PyLADR Clause Selection Comparison")
    print("=" * 60)

    print("\n🔄 Running traditional mode...")
    trad_result = run_mode("Traditional", use_online_learning=False, max_given=12)

    print("\n🧠 Running ML online learning mode...")
    ml_result = run_mode("ML Online Learning", use_online_learning=True, max_given=12)

    # Compare results
    has_differences = compare_selections(trad_result, ml_result)

    print(f"\n{'✅' if has_differences else '❓'} CONCLUSION:")
    if has_differences:
        print("   ML online learning IS affecting clause selection decisions!")
        print("   The system is making different choices based on learned embeddings.")
    else:
        print("   No clear differences detected in this test.")
        print("   Try with a longer search or different problem.")

    return 0

if __name__ == "__main__":
    exit(main())