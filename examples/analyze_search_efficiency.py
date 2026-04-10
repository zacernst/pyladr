#!/usr/bin/env python3
"""
Analyze search efficiency between traditional and ML-guided approaches.
Usage: python analyze_search_efficiency.py traditional.json ml_guided.json
"""

import json
import sys
from pathlib import Path


def compare_search_efficiency(traditional_log: str, ml_log: str) -> None:
    """Compare search efficiency between traditional and ML-guided approaches."""

    try:
        with open(traditional_log) as f:
            trad_data = json.load(f)
        with open(ml_log) as f:
            ml_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Log file not found: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        return

    print("=== Search Efficiency Comparison ===")
    print(f"Traditional: {trad_data.get('given_clauses', 'N/A')} given clauses")
    print(f"ML-Guided:   {ml_data.get('given_clauses', 'N/A')} given clauses")

    if 'given_clauses' in trad_data and 'given_clauses' in ml_data:
        efficiency_gain = 1 - ml_data['given_clauses'] / trad_data['given_clauses']
        print(f"Efficiency gain: {efficiency_gain:.1%}")

    print(f"Novel lemmas found: {ml_data.get('novel_lemmas', 'N/A')} vs {trad_data.get('novel_lemmas', 'N/A')}")
    print(f"Proof diversity: {ml_data.get('proof_branching_factor', 'N/A'):.2f}")
    print(f"Search time: {ml_data.get('search_time_seconds', 'N/A')}s vs {trad_data.get('search_time_seconds', 'N/A')}s")

    if 'embedding_stats' in ml_data:
        stats = ml_data['embedding_stats']
        print(f"\nML-Specific Metrics:")
        print(f"Cache hit rate: {stats.get('cache_hit_rate', 'N/A'):.1%}")
        print(f"Average embedding confidence: {stats.get('avg_confidence', 'N/A'):.2f}")
        print(f"Novel patterns discovered: {stats.get('novel_patterns', 'N/A')}")
        print(f"Online learning convergence: {stats.get('learning_convergence', 'N/A'):.2f}")


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python analyze_search_efficiency.py traditional.json ml_guided.json")
        print("\nThis script compares search efficiency between traditional and ML-guided theorem proving.")
        print("Log files should contain JSON output from PyLADR runs.")
        sys.exit(1)

    traditional_log = sys.argv[1]
    ml_log = sys.argv[2]

    if not Path(traditional_log).exists():
        print(f"Error: Traditional log file '{traditional_log}' not found")
        sys.exit(1)

    if not Path(ml_log).exists():
        print(f"Error: ML log file '{ml_log}' not found")
        sys.exit(1)

    compare_search_efficiency(traditional_log, ml_log)


if __name__ == "__main__":
    main()