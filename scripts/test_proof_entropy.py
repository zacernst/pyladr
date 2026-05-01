#!/usr/bin/env python3
"""Test entropy display in proof output."""

import sys
from pathlib import Path
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.core.symbol import SymbolTable
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

def test_proof_entropy():
    """Test entropy display in proof output."""

    # Create a simple problem that should generate a proof
    input_text = """
    set(auto).
    assign(max_proofs, 1).

    formulas(sos).
    P(x) | Q(x).
    -P(a).
    end_of_list.

    formulas(goals).
    Q(a).
    end_of_list.
    """

    symbol_table = SymbolTable()
    parser = LADRParser(symbol_table)
    parsed = parser.parse_input(input_text)

    print("=== Test case for proof entropy display ===")
    print(f"SOS clauses: {len(parsed.sos)}")
    print(f"Goals: {len(parsed.goals)}")

    # Create search that should find a proof
    search_opts = SearchOptions(
        print_given=True,
        quiet=False,
        max_proofs=1,
        max_given=10
    )

    search = GivenClauseSearch(options=search_opts, symbol_table=symbol_table)

    print("\\n=== Running search for proof ===")
    result = search.run(parsed.usable, parsed.sos)

    print(f"\\nSearch result: {result.exit_code}")
    print(f"Proofs found: {len(result.proofs)}")

    if result.proofs:
        proof = result.proofs[0]
        print(f"\\nProof length: {len(proof.clauses)} clauses")

        print("\\n=== Manual proof clause entropy display ===")
        for clause in proof.clauses:
            from pyladr.apps.prover9 import _calculate_structural_entropy
            entropy = _calculate_structural_entropy(clause)
            print(f"  {clause} [entropy: {entropy:.2f}]")

        print("\\n=== Testing formatted proof display ===")
        # This should now include entropy in the output if we run through prover9 app
        from pyladr.apps.prover9 import _print_proof
        from pyladr.apps.cli_common import format_clause_standard
        import io

        output = io.StringIO()
        _print_proof(proof, 1, result.stats.search_seconds(), symbol_table, output)
        proof_output = output.getvalue()
        print(proof_output)
    else:
        print("No proofs found - cannot test proof entropy display")

if __name__ == "__main__":
    test_proof_entropy()