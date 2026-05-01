#!/usr/bin/env python3
"""Test script for back-subsumption learning feature."""

import logging
import sys
from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

# Enable detailed logging to see back-subsumption events
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

def test_back_subsumption_feature():
    """Test the back-subsumption learning feature."""
    print("=== TESTING BACK-SUBSUMPTION LEARNING FEATURE ===")

    # Create a simple test problem that should trigger back-subsumption
    # Start with specific clauses, then introduce more general ones
    test_input = """
formulas(usable).
  P(a).           % Start with specific clause
end_of_list.

formulas(sos).
  P(x) -> Q(x).   % This should generate P(a) -> Q(a)
  P(x) | R(x).    % More general clause that should back-subsume P(a)
end_of_list.

formulas(goals).
  Q(a).
end_of_list.
"""

    # Parse input
    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(test_input)

    # Test 1: Without back-subsumption learning
    print("\n--- Test 1: Without back-subsumption learning ---")
    opts1 = SearchOptions(
        learn_from_back_subsumption=False,
        max_given=20,
        quiet=True
    )

    search1 = GivenClauseSearch(
        options=opts1,
        symbol_table=st,
    )

    usable = list(parsed.usable)
    sos = list(parsed.sos)

    # Add denied goals (convert goals to negated clauses)
    from pyladr.core.clause import Literal, Justification, JustType, Clause
    for goal in parsed.goals:
        denied_lits = tuple(
            Literal(sign=not lit.sign, atom=lit.atom) for lit in goal.literals
        )
        denied = Clause(
            literals=denied_lits,
            justification=(Justification(just_type=JustType.DENY, clause_ids=(0,)),),
        )
        sos.append(denied)

    result1 = search1.run(usable=usable, sos=sos)
    print(f"Result without learning: {result1.exit_code}")
    print(f"Back-subsumptions: {result1.stats.back_subsumed}")

    # Test 2: With back-subsumption learning (but no online learning active)
    print("\n--- Test 2: With back-subsumption learning flag enabled ---")
    opts2 = SearchOptions(
        learn_from_back_subsumption=True,  # Enable the flag
        max_given=20,
        quiet=True
    )

    search2 = GivenClauseSearch(
        options=opts2,
        symbol_table=st,
    )

    # Set up a simple callback to verify it gets called
    back_subsumption_events = []
    def callback(subsuming, subsumed):
        back_subsumption_events.append((subsuming.id, subsumed.id))
        print(f"🎯 Back-subsumption callback: clause {subsuming.id} subsumed clause {subsumed.id}")

    search2.set_back_subsumption_callback(callback)

    result2 = search2.run(usable=usable, sos=sos)
    print(f"Result with learning flag: {result2.exit_code}")
    print(f"Back-subsumptions: {result2.stats.back_subsumed}")
    print(f"Callback events: {len(back_subsumption_events)}")

    if back_subsumption_events:
        print("✅ Back-subsumption callback mechanism working!")
        for sub_id, subsumbed_id in back_subsumption_events:
            print(f"   - Clause {sub_id} back-subsumed clause {subsumbed_id}")
    else:
        print("⚠️  No back-subsumption events detected")

    print("\n=== BACK-SUBSUMPTION LEARNING FEATURE TEST COMPLETE ===")

if __name__ == "__main__":
    test_back_subsumption_feature()