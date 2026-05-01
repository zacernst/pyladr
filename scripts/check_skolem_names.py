#!/usr/bin/env python3
"""Check what names are being assigned to Skolem constants."""

from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.apps.prover9 import _deny_goals

def check_skolem_names():
    """Check the actual names assigned to Skolem constants."""
    print("🔍 CHECKING SKOLEM CONSTANT NAMES")
    print("=" * 40)

    input_text = """
    formulas(goals).
      Q(x,y).
    end_of_list.
    """

    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(input_text)

    print(f"Symbols before denial: {[s.name for s in st.all_symbols]}")

    usable, sos, _denied = _deny_goals(parsed, st)
    denied_goal = sos[-1]

    print(f"Symbols after denial: {[s.name for s in st.all_symbols]}")
    print(f"Denied goal: {denied_goal}")

    # Get the actual symbol names used in the denied goal
    atom = denied_goal.literals[0].atom
    for term in atom.subterms():
        if not term.is_variable:
            symnum = -term.private_symbol
            name = st.sn_to_str(symnum)
            print(f"Term {term} uses symbol '{name}' (symnum={symnum})")

if __name__ == "__main__":
    check_skolem_names()