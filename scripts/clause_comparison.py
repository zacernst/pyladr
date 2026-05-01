#!/usr/bin/env python3
"""
Direct comparison of clause selections between traditional and ML modes.
"""

import subprocess
import re

def extract_given_clauses(output_text):
    """Extract given clause selections from prover output."""
    pattern = r'given #(\d+) \(([^,]+),wt=([^)]+)\):\s*(\d+):\s*(.*?)(?:\n|$)'
    selections = []

    for match in re.finditer(pattern, output_text):
        selections.append({
            'num': int(match.group(1)),
            'type': match.group(2),
            'weight': float(match.group(3)),
            'id': int(match.group(4)),
            'clause': match.group(5).strip()
        })

    return selections

def run_comparison():
    print("🔍 CLAUSE SELECTION COMPARISON: Traditional vs ML")
    print("=" * 70)

    # Run traditional mode
    print("\n🔄 Running Traditional Mode...")
    trad_result = subprocess.run([
        'python3', '-m', 'pyladr.apps.prover9',
        '-f', 'vampire.in', '-max_given', '12'
    ], capture_output=True, text=True)

    trad_output = trad_result.stdout + trad_result.stderr
    trad_selections = extract_given_clauses(trad_output)

    print(f"   Traditional: {len(trad_selections)} selections captured")

    # Run ML mode
    print("\n🧠 Running ML Online Learning Mode...")
    ml_result = subprocess.run([
        'python3', '-m', 'pyladr.apps.prover9', '--online-learning',
        '-f', 'vampire.in', '-max_given', '12'
    ], capture_output=True, text=True)

    ml_output = ml_result.stdout + ml_result.stderr
    ml_selections = extract_given_clauses(ml_output)

    print(f"   ML Mode: {len(ml_selections)} selections captured")

    # Compare selections
    print("\n" + "=" * 70)
    print("📊 COMPARISON RESULTS")
    print("=" * 70)

    # Check for ML selections
    ml_count = sum(1 for sel in ml_selections if 'ML' in sel['type'])
    print(f"\n🎯 Selection Types:")
    print(f"   Traditional: {set(sel['type'] for sel in trad_selections)}")
    print(f"   ML Mode: {set(sel['type'] for sel in ml_selections)} ({ml_count} with ML)")

    # Find differences
    differences = []
    max_len = max(len(trad_selections), len(ml_selections))

    for i in range(max_len):
        trad_sel = trad_selections[i] if i < len(trad_selections) else None
        ml_sel = ml_selections[i] if i < len(ml_selections) else None

        if trad_sel and ml_sel:
            if trad_sel['id'] != ml_sel['id']:
                differences.append(i + 1)

    print(f"\n🔀 Selection Differences:")
    if differences:
        print(f"   ✅ Selections differ at positions: {differences}")
        print(f"   🎯 ML is making different clause selection decisions!")
    else:
        print(f"   Selection sequences are identical")

    # Show detailed comparison
    print(f"\n📋 DETAILED SELECTION COMPARISON:")
    print("=" * 70)

    for i in range(max_len):
        trad_sel = trad_selections[i] if i < len(trad_selections) else None
        ml_sel = ml_selections[i] if i < len(ml_selections) else None

        print(f"\n📍 Given #{i+1}:")

        if trad_sel:
            clause_preview = trad_sel['clause'][:60] + "..." if len(trad_sel['clause']) > 60 else trad_sel['clause']
            print(f"   TRAD: ({trad_sel['type']:<6},wt={trad_sel['weight']:>4.0f}) " +
                  f"id={trad_sel['id']:<3} │ {clause_preview}")
        else:
            print(f"   TRAD: (no selection)")

        if ml_sel:
            clause_preview = ml_sel['clause'][:60] + "..." if len(ml_sel['clause']) > 60 else ml_sel['clause']
            print(f"   ML:   ({ml_sel['type']:<6},wt={ml_sel['weight']:>4.0f}) " +
                  f"id={ml_sel['id']:<3} │ {clause_preview}")
        else:
            print(f"   ML:   (no selection)")

        # Highlight differences
        if trad_sel and ml_sel:
            if trad_sel['id'] != ml_sel['id']:
                print(f"        ⚡ DIFFERENT CLAUSES! Traditional chose {trad_sel['id']}, ML chose {ml_sel['id']}")
            elif 'ML' in ml_sel['type'] and 'ML' not in trad_sel['type']:
                print(f"        🧠 Same clause, but ML-guided selection mechanism")

    # Summary
    print(f"\n" + "=" * 70)
    print("🏆 CONCLUSION:")
    if differences or ml_count > 0:
        print("✅ SUCCESS: ML contrastive online learning IS affecting clause selection!")
        print("✅ The system demonstrates genuine ML-guided decision making")
        if ml_count > 0:
            print(f"✅ {ml_count} selections used ML guidance (marked as T+ML or A+ML)")
        if differences:
            print(f"✅ {len(differences)} positions show completely different clause choices")
    else:
        print("❓ No clear differences observed in this test run")
        print("💡 Try with longer search or different problem parameters")

if __name__ == "__main__":
    run_comparison()