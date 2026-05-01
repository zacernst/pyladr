#!/usr/bin/env python3
"""Test entropy display on vampire.in file."""

import sys
from pathlib import Path
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.core.symbol import SymbolTable
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

def test_vampire_entropy():
    """Test entropy display on vampire.in."""

    input_file = "vampire.in"
    if not Path(input_file).exists():
        print(f"Error: {input_file} not found")
        return

    input_text = Path(input_file).read_text()

    symbol_table = SymbolTable()
    parser = LADRParser(symbol_table)
    parsed = parser.parse_input(input_text)

    print(f"Vampire.in: {len(parsed.sos)} SOS clauses, {len(parsed.goals)} goals")

    # Create search with limited given clauses to see entropy display
    search_opts = SearchOptions(
        print_given=True,
        quiet=False,
        max_given=5  # Limit to first 5 given clauses
    )
    search = GivenClauseSearch(options=search_opts, symbol_table=symbol_table)

    print("\n=== Vampire.in entropy display test ===")
    try:
        result = search.run(parsed.usable, parsed.sos)
        print(f"\nSearch result: {result.exit_code}")
        print(f"Given: {result.stats.given}, Generated: {result.stats.generated}, Kept: {result.stats.kept}")
        if result.proofs:
            print(f"Proofs found: {len(result.proofs)}")
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vampire_entropy()