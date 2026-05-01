#!/usr/bin/env python3
"""Debug script to test entropy calculation and display."""

import sys
from pathlib import Path
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.core.symbol import SymbolTable
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

def debug_entropy():
    """Debug entropy calculation and search integration."""

    # Read and parse input file
    input_file = "entropy_test.in"
    input_text = Path(input_file).read_text()

    symbol_table = SymbolTable()
    parser = LADRParser(symbol_table)
    parsed = parser.parse_input(input_text)

    print(f"Parsed {len(parsed.sos)} SOS clauses, {len(parsed.goals)} goals")

    # Test entropy calculation directly
    search = GivenClauseSearch(symbol_table=symbol_table)

    print("\n=== Testing entropy calculation directly ===")
    for i, clause in enumerate(parsed.sos):
        entropy = search._calculate_structural_entropy(clause)
        print(f"SOS {i}: {clause} -> entropy: {entropy:.4f}")

    for i, clause in enumerate(parsed.goals):
        entropy = search._calculate_structural_entropy(clause)
        print(f"Goal {i}: {clause} -> entropy: {entropy:.4f}")

    # Test search options
    print(f"\n=== Search options test ===")
    opts = SearchOptions()
    print(f"print_given: {opts.print_given}")
    print(f"quiet: {opts.quiet}")

    # Test manual search step
    print(f"\n=== Manual search test ===")
    search_opts = SearchOptions(print_given=True, quiet=False, max_given=2)
    search_with_opts = GivenClauseSearch(options=search_opts, symbol_table=symbol_table)

    try:
        # Just test the first step of search
        result = search_with_opts.run(parsed.usable, parsed.sos)
        print(f"Search completed with exit code: {result.exit_code}")
        print(f"Stats: Given={result.stats.given}, Generated={result.stats.generated}")
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_entropy()