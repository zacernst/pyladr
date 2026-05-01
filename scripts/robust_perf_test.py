#!/usr/bin/env python3
"""Robust performance test with multiple runs for statistical accuracy."""

import time
import statistics
from pathlib import Path
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.core.symbol import SymbolTable
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

class NoEntropySearch(GivenClauseSearch):
    """Search without entropy calculation."""
    def _calculate_structural_entropy(self, clause):
        return 0.0

def run_multiple_tests(input_file, num_runs=5):
    """Run multiple performance tests for statistical accuracy."""

    if not Path(input_file).exists():
        print(f"Error: {input_file} not found")
        return None

    input_text = Path(input_file).read_text()
    symbol_table = SymbolTable()
    parser = LADRParser(symbol_table)
    parsed = parser.parse_input(input_text)

    search_opts = SearchOptions(
        print_given=False,
        quiet=True,
        max_given=50  # Consistent limit for better comparison
    )

    print(f"Performance test: {input_file} ({num_runs} runs)")
    print(f"Input: {len(parsed.sos)} SOS, {len(parsed.goals)} goals")

    baseline_times = []
    entropy_times = []
    baseline_stats = None
    entropy_stats = None

    # Multiple runs for statistical accuracy
    for run in range(num_runs):
        print(f"  Run {run+1}/{num_runs}... ", end='', flush=True)

        # Baseline test
        baseline_search = NoEntropySearch(options=search_opts, symbol_table=symbol_table)
        start = time.perf_counter()
        result = baseline_search.run(parsed.usable, parsed.sos)
        baseline_time = time.perf_counter() - start
        baseline_times.append(baseline_time)
        if run == 0:
            baseline_stats = result.stats

        # Entropy test
        entropy_search = GivenClauseSearch(options=search_opts, symbol_table=symbol_table)
        start = time.perf_counter()
        result = entropy_search.run(parsed.usable, parsed.sos)
        entropy_time = time.perf_counter() - start
        entropy_times.append(entropy_time)
        if run == 0:
            entropy_stats = result.stats

        print(f"B:{baseline_time:.3f}s E:{entropy_time:.3f}s")

    # Statistical analysis
    baseline_mean = statistics.mean(baseline_times)
    baseline_stdev = statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0
    entropy_mean = statistics.mean(entropy_times)
    entropy_stdev = statistics.stdev(entropy_times) if len(entropy_times) > 1 else 0

    overhead = entropy_mean - baseline_mean
    overhead_percent = (overhead / baseline_mean * 100) if baseline_mean > 0 else 0

    print(f"\\nResults:")
    print(f"  Baseline: {baseline_mean:.4f} ± {baseline_stdev:.4f}s")
    print(f"  Entropy:  {entropy_mean:.4f} ± {entropy_stdev:.4f}s")
    print(f"  Overhead: {overhead:.4f}s ({overhead_percent:+.1f}%)")

    # Check if difference is statistically significant (rough test)
    if baseline_stdev > 0 and entropy_stdev > 0:
        # Simple significance test - if difference > 2*combined_stdev, likely significant
        combined_stdev = (baseline_stdev + entropy_stdev) / 2
        if abs(overhead) > 2 * combined_stdev:
            print(f"  Significance: LIKELY significant (|{overhead:.4f}| > 2×{combined_stdev:.4f})")
        else:
            print(f"  Significance: NOT significant (|{overhead:.4f}| ≤ 2×{combined_stdev:.4f})")

    # Verify identical behavior
    if (baseline_stats.given == entropy_stats.given and
        baseline_stats.generated == entropy_stats.generated):
        print(f"  ✅ Identical behavior: {baseline_stats.given} given, {baseline_stats.generated} generated")
    else:
        print(f"  ⚠️  Different behavior!")

    return {
        'overhead_percent': overhead_percent,
        'baseline_mean': baseline_mean,
        'entropy_mean': entropy_mean,
        'significant': abs(overhead) > 2 * combined_stdev if combined_stdev > 0 else False
    }

def main():
    """Run robust performance tests."""

    results = {}

    test_files = ['vampire.in', 'entropy_test.in']
    for test_file in test_files:
        if Path(test_file).exists():
            print('\\n' + '='*50)
            results[test_file] = run_multiple_tests(test_file, num_runs=5)

    # Summary
    print('\\n' + '='*50)
    print('PERFORMANCE SUMMARY')
    print('='*50)

    overheads = [r['overhead_percent'] for r in results.values() if r]
    if overheads:
        avg_overhead = sum(overheads) / len(overheads)
        print(f"Average overhead: {avg_overhead:+.1f}%")

        if abs(avg_overhead) < 5.0:
            print("✅ PASS: Minimal performance impact (<5%)")
        else:
            print(f"⚠️  WARNING: {avg_overhead:.1f}% overhead detected")

if __name__ == "__main__":
    main()