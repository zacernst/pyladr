#!/usr/bin/env python3
"""Measure performance impact of structural entropy calculations."""

import time
from pathlib import Path
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.core.symbol import SymbolTable
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

class NoEntropySearch(GivenClauseSearch):
    """Modified search that skips entropy calculation for baseline measurement."""

    def _calculate_structural_entropy(self, clause):
        """Override to return 0.0 without calculation."""
        return 0.0

def run_performance_test(input_file, max_time_seconds=10):
    """Run performance comparison between entropy and no-entropy searches."""

    if not Path(input_file).exists():
        print(f"Error: {input_file} not found")
        return None

    input_text = Path(input_file).read_text()

    # Parse input once for both tests
    symbol_table = SymbolTable()
    parser = LADRParser(symbol_table)
    parsed = parser.parse_input(input_text)

    print(f"Performance test on {input_file}")
    print(f"Input: {len(parsed.sos)} SOS clauses, {len(parsed.goals)} goals")
    print(f"Test duration: {max_time_seconds} seconds each")
    print()

    # Test 1: Baseline (no entropy calculation)
    print("=== Baseline Test (No Entropy) ===")
    search_opts = SearchOptions(
        print_given=False,  # Disable output for clean timing
        quiet=True,
        max_seconds=max_time_seconds
    )

    baseline_search = NoEntropySearch(options=search_opts, symbol_table=symbol_table)

    start_time = time.perf_counter()
    baseline_result = baseline_search.run(parsed.usable, parsed.sos)
    baseline_time = time.perf_counter() - start_time

    print(f"Baseline time: {baseline_time:.3f} seconds")
    print(f"Given: {baseline_result.stats.given}")
    print(f"Generated: {baseline_result.stats.generated}")
    print(f"Kept: {baseline_result.stats.kept}")
    print()

    # Test 2: With entropy calculation
    print("=== With Entropy Test ===")
    entropy_search = GivenClauseSearch(options=search_opts, symbol_table=symbol_table)

    start_time = time.perf_counter()
    entropy_result = entropy_search.run(parsed.usable, parsed.sos)
    entropy_time = time.perf_counter() - start_time

    print(f"Entropy time: {entropy_time:.3f} seconds")
    print(f"Given: {entropy_result.stats.given}")
    print(f"Generated: {entropy_result.stats.generated}")
    print(f"Kept: {entropy_result.stats.kept}")
    print()

    # Analysis
    if baseline_time > 0:
        overhead_percent = ((entropy_time - baseline_time) / baseline_time) * 100
        throughput_baseline = baseline_result.stats.given / baseline_time if baseline_time > 0 else 0
        throughput_entropy = entropy_result.stats.given / entropy_time if entropy_time > 0 else 0

        print("=== Performance Analysis ===")
        print(f"Overhead: {entropy_time - baseline_time:.3f} seconds ({overhead_percent:+.1f}%)")
        print(f"Baseline throughput: {throughput_baseline:.1f} given clauses/sec")
        print(f"Entropy throughput: {throughput_entropy:.1f} given clauses/sec")

        if abs(overhead_percent) < 5.0:
            print("✅ Performance impact within acceptable limits (<5%)")
        else:
            print(f"⚠️  Performance impact {overhead_percent:.1f}% may be significant")

        # Check if search results are equivalent
        if (baseline_result.stats.given == entropy_result.stats.given and
            baseline_result.stats.generated == entropy_result.stats.generated and
            baseline_result.stats.kept == entropy_result.stats.kept):
            print("✅ Search behavior unchanged (same statistics)")
        else:
            print("⚠️  Search behavior differs - investigating...")
            print(f"   Baseline: G={baseline_result.stats.given}, Gen={baseline_result.stats.generated}, K={baseline_result.stats.kept}")
            print(f"   Entropy:  G={entropy_result.stats.given}, Gen={entropy_result.stats.generated}, K={entropy_result.stats.kept}")

    return {
        'baseline_time': baseline_time,
        'entropy_time': entropy_time,
        'overhead_percent': overhead_percent if baseline_time > 0 else 0,
        'baseline_given': baseline_result.stats.given,
        'entropy_given': entropy_result.stats.given
    }

def main():
    """Run performance tests on available input files."""

    test_files = ['vampire.in', 'entropy_test.in']
    results = {}

    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\\n{'='*60}")
            results[test_file] = run_performance_test(test_file, max_time_seconds=5)
        else:
            print(f"Skipping {test_file} (not found)")

    # Overall summary
    print(f"\\n{'='*60}")
    print("OVERALL PERFORMANCE SUMMARY")
    print('='*60)

    total_overhead = 0
    test_count = 0

    for filename, result in results.items():
        if result:
            print(f"{filename}: {result['overhead_percent']:+.1f}% overhead")
            total_overhead += result['overhead_percent']
            test_count += 1

    if test_count > 0:
        avg_overhead = total_overhead / test_count
        print(f"\\nAverage overhead: {avg_overhead:+.1f}%")

        if abs(avg_overhead) < 5.0:
            print("✅ PASS: Entropy feature has minimal performance impact")
        else:
            print(f"⚠️  WARNING: Average {avg_overhead:.1f}% overhead may be significant")

if __name__ == "__main__":
    main()