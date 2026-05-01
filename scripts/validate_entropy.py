#!/usr/bin/env python3
"""Comprehensive validation of structural entropy calculations."""

import math
from pyladr.parsing.ladr_parser import parse_clause
from pyladr.core.symbol import SymbolTable
from pyladr.search.given_clause import GivenClauseSearch

def manual_entropy(node_counts):
    """Calculate entropy manually from node counts."""
    total = sum(node_counts.values())
    if total <= 1:
        return 0.0

    entropy = 0.0
    for count in node_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy

def validate_entropy():
    """Validate entropy calculations against manual computations."""

    st = SymbolTable()
    search = GivenClauseSearch(symbol_table=st)

    test_cases = [
        # Test case: clause, expected node counts, expected entropy
        ("P(x)", {'clause': 1, 'literal': 1, 'predicate': 1, 'variable': 1, 'function': 0, 'constant': 0}, 2.0),
        ("P(a)", {'clause': 1, 'literal': 1, 'predicate': 1, 'variable': 0, 'function': 0, 'constant': 1}, 2.0),
        ("P(f(x,y))", {'clause': 1, 'literal': 1, 'predicate': 1, 'function': 1, 'variable': 2, 'constant': 0}, None),
        ("P(f(x,y)) | Q(z)", {'clause': 1, 'literal': 2, 'predicate': 2, 'function': 1, 'variable': 3, 'constant': 0}, None),
        ("R(g(h(x)))", {'clause': 1, 'literal': 1, 'predicate': 1, 'function': 2, 'variable': 1, 'constant': 0}, None),
    ]

    print("=== Entropy Validation Test Results ===\\n")
    all_passed = True

    for clause_str, expected_counts, expected_entropy in test_cases:
        clause = parse_clause(clause_str, st)
        computed_entropy = search._calculate_structural_entropy(clause)

        # Calculate expected entropy if not provided
        if expected_entropy is None:
            expected_entropy = manual_entropy(expected_counts)

        # Test calculation
        passed = abs(computed_entropy - expected_entropy) < 0.0001
        all_passed = all_passed and passed

        print(f"Clause: {clause_str}")
        print(f"  Expected entropy: {expected_entropy:.4f}")
        print(f"  Computed entropy: {computed_entropy:.4f}")
        print(f"  Status: {'PASS' if passed else 'FAIL'}")
        print(f"  Expected nodes: {expected_counts}")
        print()

    # Edge case tests
    print("=== Edge Case Tests ===\\n")

    # Test empty handling (this might not be directly testable depending on parsing)
    # Test single constant
    clause = parse_clause("a", st)  # This might parse as P(a) or similar
    entropy = search._calculate_structural_entropy(clause)
    print(f"Simple constant clause: {clause}")
    print(f"  Entropy: {entropy:.4f}")
    print()

    # Test complex nested structure
    clause = parse_clause("P(f(g(h(x)), y))", st)
    entropy = search._calculate_structural_entropy(clause)
    print(f"Complex nested: {clause}")
    print(f"  Entropy: {entropy:.4f}")
    print()

    print(f"Overall validation: {'PASS' if all_passed else 'FAIL'}")
    return all_passed

if __name__ == "__main__":
    validate_entropy()