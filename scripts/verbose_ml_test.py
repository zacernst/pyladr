#!/usr/bin/env python3
"""Test script with verbose ML logging enabled."""

import logging
import sys
from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions
from pyladr.search.ml_selection import EmbeddingEnhancedSelection, MLSelectionConfig

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Enable ML selection logging
ml_config = MLSelectionConfig(
    enabled=True,
    ml_weight=0.4,
    log_selections=True  # This enables detailed ML selection logging!
)

def run_verbose_test(input_file):
    """Run test with verbose ML logging."""
    print("=== VERBOSE ML SELECTION LOGGING ENABLED ===")

    # Parse input
    st = SymbolTable()
    parser = LADRParser(st)

    with open(input_file) as f:
        parsed = parser.parse_input(f.read())

    # Set up search options with goal-directed features
    opts = SearchOptions(
        goal_directed=True,
        goal_proximity_weight=0.6,
        online_learning=True,
        ml_weight=0.4,
        max_given=10,
        quiet=False  # Ensure non-quiet mode
    )

    # Create selection with verbose ML logging
    selection = EmbeddingEnhancedSelection(ml_config=ml_config)

    # Deny goals for refutation (same as prover9.py)
    from pyladr.core.clause import Literal, Justification, JustType

    usable = list(parsed.usable)
    sos = list(parsed.sos)

    for goal in parsed.goals:
        denied_lits = tuple(
            Literal(sign=not lit.sign, atom=lit.atom) for lit in goal.literals
        )
        from pyladr.core.clause import Clause
        denied = Clause(
            literals=denied_lits,
            justification=(Justification(just_type=JustType.DENY, clause_ids=(0,)),),
        )
        sos.append(denied)

    # Run search
    search = GivenClauseSearch(
        options=opts,
        symbol_table=st,
        selection=selection
    )
    result = search.run(usable=usable, sos=sos)

    print(f"\n=== VERBOSE SEARCH COMPLETED ===")
    print(f"Result: {result.exit_code}")
    print(f"ML Selection Stats: {selection.ml_stats.report()}")

if __name__ == "__main__":
    run_verbose_test(sys.argv[1] if len(sys.argv) > 1 else "test_embedding_dim.in")