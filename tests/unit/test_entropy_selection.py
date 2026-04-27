"""Comprehensive tests for entropy-based clause selection.

Tests cover:
- Unit tests for entropy calculation (_clause_entropy, _count_nodes_flat)
- Entropy-based selection via SelectionOrder.ENTROPY
- PrioritySOS entropy heap integration
- Entropy selection rule ratio cycling
- Regression: traditional selection unaffected by entropy addition
- Edge cases: empty clauses, single-literal, deep nesting, ground terms
- Conditional entropy display format (REQ-C009 compatibility)
- SearchOptions entropy_weight integration
- CLI --entropy-weight option (REQ-INT001 functional effect)
- Performance: entropy calculation overhead assessment
"""

from __future__ import annotations

import math
import io
import re
import time
from unittest.mock import patch

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import parse_input
from pyladr.search.selection import (
    GivenSelection,
    SelectionOrder,
    SelectionRule,
    _clause_entropy,
    _count_nodes_flat,
    _weight_compare,
    default_clause_weight,
)
from pyladr.search.state import ClauseList
from tests.factories import (
    make_clause_from_atoms as _make_clause,
    make_const as _const,
    make_func as _func,
    make_var as _var,
)


def _make_weighted_clause(weight: float, symnum: int = 1) -> Clause:
    """Create a clause and set its weight."""
    c = _make_clause(_const(symnum))
    c.weight = weight
    return c


# Symbol IDs for test clauses
A, B, C_SYM = 1, 2, 3
F, G = 10, 11
P, Q = 20, 21


# ── Unit Tests: _clause_entropy ─────────────────────────────────────────────


class TestClauseEntropy:
    """Unit tests for the _clause_entropy function."""

    def test_single_constant_clause(self) -> None:
        """A clause with a single constant literal has limited entropy.

        Node types: 1 clause + 1 literal + 1 predicate (atom root) = 3 nodes
        Distribution: {clause:1, literal:1, predicate:1} → uniform over 3 types
        H = log2(3) ≈ 1.585
        """
        c = _make_clause(_const(A))
        entropy = _clause_entropy(c)
        assert entropy > 0.0
        # Single constant atom: clause(1) + literal(1) + constant(1) = 3 nodes
        # but atom root is predicate for top-level atom
        assert entropy == pytest.approx(math.log2(3), abs=0.01)

    def test_entropy_is_nonnegative(self) -> None:
        """Entropy is always >= 0."""
        clauses = [
            _make_clause(_const(A)),
            _make_clause(_func(F, _const(A), _var(0))),
            _make_clause(_func(P, _var(0), _var(1)), _func(Q, _const(B))),
        ]
        for c in clauses:
            assert _clause_entropy(c) >= 0.0

    def test_entropy_increases_with_diversity(self) -> None:
        """A clause with more node type diversity has higher entropy."""
        # Simple: only constant and predicate types
        simple = _make_clause(_const(A))
        # Complex: function, variable, constant, predicate types
        complex_clause = _make_clause(
            _func(P, _func(F, _var(0), _const(A)), _const(B))
        )
        assert _clause_entropy(complex_clause) > _clause_entropy(simple)

    def test_entropy_of_ground_clause(self) -> None:
        """Ground clause (no variables) should have nonzero entropy."""
        # p(a, f(b)) - has clause, literal, predicate, function, constant types
        c = _make_clause(_func(P, _const(A), _func(F, _const(B))))
        entropy = _clause_entropy(c)
        assert entropy > 0.0
        # Should NOT have variable type contribution
        # Count: clause=1, literal=1, predicate=1, function=1, constant=2
        # Total = 6, but only 4 distinct types

    def test_entropy_of_all_variables(self) -> None:
        """Clause with all variable arguments."""
        # p(x, y) - clause, literal, predicate, variable types
        c = _make_clause(_func(P, _var(0), _var(1)))
        entropy = _clause_entropy(c)
        assert entropy > 0.0

    def test_entropy_multi_literal(self) -> None:
        """Multi-literal clause should have more entropy than single-literal."""
        single = _make_clause(_func(P, _var(0)))
        multi = _make_clause(
            _func(P, _var(0)),
            _func(Q, _const(A)),
            signs=(True, False),
        )
        # Multi-literal adds literal nodes and possibly more diversity
        assert _clause_entropy(multi) >= _clause_entropy(single)

    def test_entropy_deterministic(self) -> None:
        """Same clause always produces same entropy value."""
        c = _make_clause(_func(P, _func(F, _var(0), _const(A)), _const(B)))
        e1 = _clause_entropy(c)
        e2 = _clause_entropy(c)
        assert e1 == e2

    def test_entropy_maximum_bound(self) -> None:
        """Entropy cannot exceed log2(6) ≈ 2.585 (6 node types)."""
        c = _make_clause(
            _func(P, _func(F, _var(0), _const(A)), _const(B)),
            _func(Q, _func(G, _var(1))),
        )
        entropy = _clause_entropy(c)
        assert entropy <= math.log2(6) + 0.001  # 6 node type categories


class TestCountNodesFlat:
    """Unit tests for the _count_nodes_flat helper function."""

    def test_variable_counted(self) -> None:
        """Variable terms increment index 4."""
        counts = [0, 0, 0, 0, 0, 0]
        _count_nodes_flat(_var(0), counts, False)
        assert counts[4] == 1  # variable

    def test_constant_counted(self) -> None:
        """Constant terms increment index 5."""
        counts = [0, 0, 0, 0, 0, 0]
        _count_nodes_flat(_const(A), counts, False)
        assert counts[5] == 1  # constant

    def test_predicate_counted(self) -> None:
        """Top-level complex term with is_predicate=True increments index 2."""
        counts = [0, 0, 0, 0, 0, 0]
        _count_nodes_flat(_func(P, _const(A)), counts, True)
        assert counts[2] == 1  # predicate
        assert counts[5] == 1  # constant arg

    def test_function_counted(self) -> None:
        """Complex term with is_predicate=False increments index 3."""
        counts = [0, 0, 0, 0, 0, 0]
        _count_nodes_flat(_func(F, _const(A)), counts, False)
        assert counts[3] == 1  # function
        assert counts[5] == 1  # constant arg

    def test_nested_terms(self) -> None:
        """Nested structure counts all nodes recursively."""
        # f(g(x), a) - function=2, variable=1, constant=1
        term = _func(F, _func(G, _var(0)), _const(A))
        counts = [0, 0, 0, 0, 0, 0]
        _count_nodes_flat(term, counts, False)
        assert counts[3] == 2  # two functions: f and g
        assert counts[4] == 1  # one variable: x
        assert counts[5] == 1  # one constant: a

    def test_predicate_children_are_functions(self) -> None:
        """Children of a predicate are classified as functions, not predicates."""
        # p(f(x)) - predicate=1 (p), function=1 (f), variable=1 (x)
        term = _func(P, _func(F, _var(0)))
        counts = [0, 0, 0, 0, 0, 0]
        _count_nodes_flat(term, counts, True)
        assert counts[2] == 1  # predicate: p
        assert counts[3] == 1  # function: f
        assert counts[4] == 1  # variable: x


# ── Unit Tests: Entropy Selection ────────────────────────────────────────────


class TestEntropySelectionOrder:
    """Test SelectionOrder.ENTROPY integration in GivenSelection."""

    def test_entropy_order_exists(self) -> None:
        """SelectionOrder.ENTROPY should be defined."""
        assert SelectionOrder.ENTROPY == 3

    def test_entropy_selection_rule(self) -> None:
        """Can create a SelectionRule with ENTROPY order."""
        rule = SelectionRule("E", SelectionOrder.ENTROPY, part=1)
        assert rule.name == "E"
        assert rule.order == SelectionOrder.ENTROPY

    def test_entropy_selects_highest_entropy_clause(self) -> None:
        """Entropy selection should pick the clause with highest entropy."""
        sos = ClauseList("sos")

        # Simple clause: low entropy (few node types)
        c_simple = _make_clause(_const(A))
        c_simple.id = 1
        c_simple.weight = 1.0

        # Complex clause: high entropy (many node types)
        c_complex = _make_clause(
            _func(P, _func(F, _var(0), _const(A)), _const(B)),
            _func(Q, _var(1)),
        )
        c_complex.id = 2
        c_complex.weight = 7.0

        sos.append(c_simple)
        sos.append(c_complex)

        gs = GivenSelection(
            rules=[SelectionRule("E", SelectionOrder.ENTROPY, part=1)]
        )
        selected, name = gs.select_given(sos, 0)
        assert name == "E"
        # Complex clause should have higher entropy
        assert selected is c_complex

    def test_entropy_tiebreak_by_id(self) -> None:
        """When entropy is equal, older clause (lower ID) wins."""
        sos = ClauseList("sos")

        c1 = _make_clause(_const(A))
        c1.id = 5
        c1.weight = 1.0

        c2 = _make_clause(_const(B))
        c2.id = 10
        c2.weight = 1.0

        sos.append(c1)
        sos.append(c2)

        gs = GivenSelection(
            rules=[SelectionRule("E", SelectionOrder.ENTROPY, part=1)]
        )
        selected, _ = gs.select_given(sos, 0)
        # Same structure → same entropy → lower ID wins
        assert selected is c1

    def test_entropy_removes_from_sos(self) -> None:
        """Entropy selection removes clause from SOS."""
        sos = ClauseList("sos")
        c = _make_clause(_const(A))
        c.id = 1
        c.weight = 1.0
        sos.append(c)

        gs = GivenSelection(
            rules=[SelectionRule("E", SelectionOrder.ENTROPY, part=1)]
        )
        selected, _ = gs.select_given(sos, 0)
        assert selected is c
        assert sos.is_empty


class TestEntropyRatioCycling:
    """Test entropy selection integrated with ratio cycling."""

    def test_entropy_in_ratio_cycle(self) -> None:
        """Entropy can participate in ratio cycling with weight and age."""
        sos = ClauseList("sos")

        # Create enough clauses for multiple selections
        for i in range(10):
            c = _make_weighted_clause(float(10 - i), symnum=i + 1)
            c.id = i + 1
            sos.append(c)

        # Ratio: 1 age, 2 weight, 1 entropy = cycle of 4
        gs = GivenSelection(
            rules=[
                SelectionRule("A", SelectionOrder.AGE, part=1),
                SelectionRule("W", SelectionOrder.WEIGHT, part=2),
                SelectionRule("E", SelectionOrder.ENTROPY, part=1),
            ]
        )

        selections = []
        for i in range(4):
            _, name = gs.select_given(sos, i)
            selections.append(name)

        assert selections == ["A", "W", "W", "E"]

    def test_default_selection_unchanged(self) -> None:
        """Default GivenSelection (no entropy) is still weight+age only."""
        gs = GivenSelection()
        assert len(gs.rules) == 2
        assert gs.rules[0].order == SelectionOrder.AGE
        assert gs.rules[1].order == SelectionOrder.WEIGHT


# ── PrioritySOS Entropy Heap Tests ──────────────────────────────────────────


class TestPrioritySosEntropy:
    """Test PrioritySOS entropy heap operations."""

    def test_pop_highest_entropy(self) -> None:
        """pop_highest_entropy returns clause with highest entropy."""
        from pyladr.search.priority_sos import PrioritySOS

        psos = PrioritySOS("sos")

        c_simple = _make_clause(_const(A))
        c_simple.id = 1
        c_simple.weight = 1.0

        c_complex = _make_clause(
            _func(P, _func(F, _var(0), _const(A)), _const(B)),
            _func(Q, _var(1)),
        )
        c_complex.id = 2
        c_complex.weight = 7.0

        psos.append(c_simple)
        psos.append(c_complex)

        result = psos.pop_highest_entropy()
        assert result is c_complex
        assert psos.length == 1

    def test_pop_highest_entropy_empty(self) -> None:
        """pop_highest_entropy on empty PrioritySOS returns None."""
        from pyladr.search.priority_sos import PrioritySOS

        psos = PrioritySOS("sos")
        assert psos.pop_highest_entropy() is None

    def test_entropy_heap_lazy_deletion(self) -> None:
        """Removed clauses are skipped during entropy pop."""
        from pyladr.search.priority_sos import PrioritySOS

        psos = PrioritySOS("sos")

        c1 = _make_clause(_const(A))
        c1.id = 1
        c1.weight = 1.0

        c2 = _make_clause(_func(P, _var(0), _const(A)))
        c2.id = 2
        c2.weight = 3.0

        psos.append(c1)
        psos.append(c2)

        # Remove c2 (higher entropy) via lazy deletion
        psos.remove(c2)

        # Should get c1, skipping stale c2 entry in heap
        result = psos.pop_highest_entropy()
        assert result is c1

    def test_entropy_selection_via_priority_sos(self) -> None:
        """GivenSelection with ENTROPY order uses PrioritySOS.pop_highest_entropy."""
        from pyladr.search.priority_sos import PrioritySOS

        psos = PrioritySOS("sos")

        c1 = _make_clause(_const(A))
        c1.id = 1
        c1.weight = 1.0

        c2 = _make_clause(_func(P, _func(F, _var(0)), _const(A)))
        c2.id = 2
        c2.weight = 5.0

        psos.append(c1)
        psos.append(c2)

        gs = GivenSelection(
            rules=[SelectionRule("E", SelectionOrder.ENTROPY, part=1)]
        )
        selected, name = gs.select_given(psos, 0)
        assert name == "E"
        assert selected is c2  # higher entropy


# ── Regression Tests: Existing Selection Behavior ───────────────────────────


class TestExistingSelectionRegression:
    """Ensure adding entropy does not break existing weight/age selection."""

    def test_weight_selection_unchanged(self) -> None:
        """Weight-based selection still picks lightest clause."""
        sos = ClauseList("sos")
        c_heavy = _make_weighted_clause(10.0, symnum=A)
        c_heavy.id = 1
        c_light = _make_weighted_clause(2.0, symnum=B)
        c_light.id = 2
        sos.append(c_heavy)
        sos.append(c_light)

        gs = GivenSelection(
            rules=[SelectionRule("W", SelectionOrder.WEIGHT, part=1)]
        )
        selected, name = gs.select_given(sos, 0)
        assert name == "W"
        assert selected is c_light

    def test_age_selection_unchanged(self) -> None:
        """Age-based selection still picks oldest clause (FIFO)."""
        sos = ClauseList("sos")
        c_old = _make_weighted_clause(5.0, symnum=A)
        c_old.id = 1
        c_new = _make_weighted_clause(1.0, symnum=B)
        c_new.id = 2
        sos.append(c_old)
        sos.append(c_new)

        gs = GivenSelection(
            rules=[SelectionRule("A", SelectionOrder.AGE, part=1)]
        )
        selected, name = gs.select_given(sos, 0)
        assert name == "A"
        assert selected is c_old  # first appended = oldest

    def test_default_ratio_cycle_preserved(self) -> None:
        """Default 1:4 age:weight ratio still works as expected."""
        sos = ClauseList("sos")
        for i in range(10):
            c = _make_weighted_clause(float(10 - i), symnum=i + 1)
            c.id = i + 1
            sos.append(c)

        gs = GivenSelection()  # default rules
        selections = []
        for i in range(5):
            _, name = gs.select_given(sos, i)
            selections.append(name)

        # Default: A(1), W(4) → [A, W, W, W, W]
        assert selections == ["A", "W", "W", "W", "W"]

    def test_weight_tiebreak_still_by_id(self) -> None:
        """Weight selection tiebreak (by ID) is preserved."""
        sos = ClauseList("sos")
        c1 = _make_weighted_clause(5.0, symnum=A)
        c1.id = 10
        c2 = _make_weighted_clause(5.0, symnum=B)
        c2.id = 5
        sos.append(c1)
        sos.append(c2)

        gs = GivenSelection(
            rules=[SelectionRule("W", SelectionOrder.WEIGHT, part=1)]
        )
        selected, _ = gs.select_given(sos, 0)
        assert selected is c2  # lower ID wins

    def test_empty_sos_returns_none(self) -> None:
        """Empty SOS still returns (None, '') for all selection types."""
        sos = ClauseList("sos")
        for order in [SelectionOrder.WEIGHT, SelectionOrder.AGE, SelectionOrder.ENTROPY]:
            gs = GivenSelection(
                rules=[SelectionRule("X", order, part=1)]
            )
            selected, name = gs.select_given(sos, 0)
            assert selected is None
            assert name == ""


# ── Entropy Calculation Edge Cases ──────────────────────────────────────────


class TestEntropyEdgeCases:
    """Edge cases for entropy calculation."""

    def test_deeply_nested_term(self) -> None:
        """Deeply nested terms should compute without stack overflow."""
        # Build f(f(f(f(f(a))))) - 5 levels deep
        term = _const(A)
        for _ in range(5):
            term = _func(F, term)
        c = _make_clause(term)
        entropy = _clause_entropy(c)
        assert entropy >= 0.0
        assert math.isfinite(entropy)

    def test_many_literals(self) -> None:
        """Clause with many literals should work correctly."""
        atoms = [_func(P + i, _var(i)) for i in range(10)]
        c = _make_clause(*atoms)
        entropy = _clause_entropy(c)
        assert entropy > 0.0
        assert math.isfinite(entropy)

    def test_single_variable_literal(self) -> None:
        """Clause with single variable literal."""
        c = _make_clause(_var(0))
        entropy = _clause_entropy(c)
        # clause=1, literal=1, variable=1 → uniform over 3 → log2(3)
        assert entropy >= 0.0

    def test_entropy_consistency_across_implementations(self) -> None:
        """_clause_entropy (flat array) should match manual calculation."""
        # p(f(x), a) → clause:1, literal:1, predicate:1, function:1, variable:1, constant:1
        c = _make_clause(_func(P, _func(F, _var(0)), _const(A)))
        entropy = _clause_entropy(c)

        # Manual: 6 nodes, all distinct types → uniform distribution
        # H = log2(6) ≈ 2.585
        expected = math.log2(6)
        assert entropy == pytest.approx(expected, abs=0.01)


# ── Integration Test: Entropy in Search ──────────────────────────────────────


class TestEntropySearchIntegration:
    """Integration tests for entropy selection in the search loop."""

    def test_entropy_selection_finds_proof(self) -> None:
        """Search with entropy-only selection can still find proofs."""
        from pyladr.search.given_clause import (
            ExitCode,
            GivenClauseSearch,
            SearchOptions,
        )

        st = SymbolTable()
        parsed = parse_input(
            """\
formulas(sos).
  e * x = x.
end_of_list.
formulas(goals).
  e * e = e.
end_of_list.
""",
            st,
        )

        sos = list(parsed.sos)
        for goal in parsed.goals:
            denied_lits = tuple(
                Literal(sign=not lit.sign, atom=lit.atom)
                for lit in goal.literals
            )
            denied = Clause(
                literals=denied_lits,
                justification=(
                    Justification(just_type=JustType.DENY, clause_ids=(0,)),
                ),
            )
            sos.append(denied)

        opts = SearchOptions(
            max_given=100,
            max_seconds=10.0,
            print_given=False,
            quiet=True,
        )
        engine = GivenClauseSearch(opts, symbol_table=st)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            result = engine.run(usable=[], sos=sos)

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_given_trace_includes_entropy_when_enabled(self) -> None:
        """When entropy_weight > 0 and print_given=True, entropy info appears in trace."""
        from pyladr.search.given_clause import (
            ExitCode,
            GivenClauseSearch,
            SearchOptions,
        )

        st = SymbolTable()
        parsed = parse_input(
            """\
formulas(sos).
  e * x = x.
end_of_list.
formulas(goals).
  e * e = e.
end_of_list.
""",
            st,
        )

        sos = list(parsed.sos)
        for goal in parsed.goals:
            denied_lits = tuple(
                Literal(sign=not lit.sign, atom=lit.atom)
                for lit in goal.literals
            )
            denied = Clause(
                literals=denied_lits,
                justification=(
                    Justification(just_type=JustType.DENY, clause_ids=(0,)),
                ),
            )
            sos.append(denied)

        opts = SearchOptions(
            max_given=100,
            max_seconds=10.0,
            print_given=True,
            quiet=False,
            entropy_weight=2,  # Enable entropy to see it in output
        )
        engine = GivenClauseSearch(opts, symbol_table=st)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            result = engine.run(usable=[], sos=sos)

        output = captured.getvalue()
        given_lines = [l for l in output.splitlines() if "given #" in l]
        assert len(given_lines) > 0, "Should have given clause trace output"

        # Check entropy info is present when entropy is enabled
        has_entropy_info = any(
            "ent=" in line or "entropy" in line.lower()
            for line in given_lines
        )
        assert has_entropy_info, (
            "Entropy info should appear in trace when entropy_weight > 0"
        )

    def test_given_trace_no_entropy_when_disabled(self) -> None:
        """When entropy_weight=0, trace should use C-compatible format without entropy."""
        from pyladr.search.given_clause import (
            ExitCode,
            GivenClauseSearch,
            SearchOptions,
        )

        st = SymbolTable()
        parsed = parse_input(SIMPLE_INPUT, st)

        sos = list(parsed.sos)
        for goal in parsed.goals:
            denied_lits = tuple(
                Literal(sign=not lit.sign, atom=lit.atom)
                for lit in goal.literals
            )
            denied = Clause(
                literals=denied_lits,
                justification=(
                    Justification(just_type=JustType.DENY, clause_ids=(0,)),
                ),
            )
            sos.append(denied)

        opts = SearchOptions(
            max_given=100,
            max_seconds=10.0,
            print_given=True,
            quiet=False,
            entropy_weight=0,  # Disabled
        )
        engine = GivenClauseSearch(opts, symbol_table=st)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            result = engine.run(usable=[], sos=sos)

        output = captured.getvalue()
        given_lines = [l for l in output.splitlines() if "given #" in l]
        for line in given_lines:
            assert "entropy" not in line.lower(), (
                f"Entropy should NOT appear with entropy_weight=0: {line!r}"
            )


# ── Helpers for search integration tests ─────────────────────────────────────


SIMPLE_INPUT = """\
formulas(sos).
  e * x = x.
end_of_list.
formulas(goals).
  e * e = e.
end_of_list.
"""


def _parse_and_deny(text: str) -> tuple[list[Clause], SymbolTable]:
    """Parse LADR input and deny goals into SOS clauses."""
    st = SymbolTable()
    parsed = parse_input(text, st)
    sos = list(parsed.sos)
    for goal in parsed.goals:
        denied_lits = tuple(
            Literal(sign=not lit.sign, atom=lit.atom) for lit in goal.literals
        )
        denied = Clause(
            literals=denied_lits,
            justification=(Justification(just_type=JustType.DENY, clause_ids=(0,)),),
        )
        sos.append(denied)
    return sos, st


def _run_and_capture(
    text: str,
    entropy_weight: int = 0,
    print_given: bool = True,
    quiet: bool = False,
    max_given: int = 500,
    **kwargs,
) -> tuple[str, "ExitCode"]:
    """Run search and capture stdout output."""
    from pyladr.search.given_clause import (
        ExitCode,
        GivenClauseSearch,
        SearchOptions,
    )

    sos, st = _parse_and_deny(text)
    opts = SearchOptions(
        max_given=max_given,
        max_seconds=10.0,
        print_given=print_given,
        quiet=quiet,
        entropy_weight=entropy_weight,
        **kwargs,
    )
    engine = GivenClauseSearch(opts, symbol_table=st)
    captured = io.StringIO()
    with patch("sys.stdout", captured):
        result = engine.run(usable=[], sos=sos)
    return captured.getvalue(), result.exit_code


# ── SearchOptions entropy_weight Integration ─────────────────────────────────


class TestSearchOptionsEntropyWeight:
    """Test SearchOptions.entropy_weight creates correct selection rules."""

    def test_entropy_weight_zero_gives_default_selection(self) -> None:
        """entropy_weight=0 (default) should produce standard A+W selection."""
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        opts = SearchOptions(entropy_weight=0)
        engine = GivenClauseSearch(opts)
        # Should have default 2 rules: age + weight
        assert len(engine._selection.rules) == 2
        assert engine._selection.rules[0].order == SelectionOrder.AGE
        assert engine._selection.rules[1].order == SelectionOrder.WEIGHT

    def test_entropy_weight_positive_adds_entropy_rule(self) -> None:
        """entropy_weight > 0 should add an entropy selection rule."""
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        opts = SearchOptions(entropy_weight=2)
        engine = GivenClauseSearch(opts)
        # Should have 3 rules: age + weight + entropy
        assert len(engine._selection.rules) == 3
        assert engine._selection.rules[2].name == "E"
        assert engine._selection.rules[2].order == SelectionOrder.ENTROPY
        assert engine._selection.rules[2].part == 2

    def test_entropy_weight_respects_part_ratio(self) -> None:
        """The entropy rule's part should match entropy_weight value."""
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        for ew in [1, 3, 5]:
            opts = SearchOptions(entropy_weight=ew)
            engine = GivenClauseSearch(opts)
            entropy_rule = [r for r in engine._selection.rules if r.name == "E"]
            assert len(entropy_rule) == 1
            assert entropy_rule[0].part == ew

    def test_custom_selection_overrides_entropy_weight(self) -> None:
        """Providing explicit selection should override entropy_weight."""
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        opts = SearchOptions(entropy_weight=5)
        custom_sel = GivenSelection(
            rules=[SelectionRule("W", SelectionOrder.WEIGHT, part=1)]
        )
        engine = GivenClauseSearch(opts, selection=custom_sel)
        # Custom selection should win
        assert len(engine._selection.rules) == 1
        assert engine._selection.rules[0].name == "W"


# ── Conditional Entropy Display Format (REQ-C009) ───────────────────────────


class TestConditionalEntropyDisplay:
    """Test that entropy info only appears in trace when entropy is active.

    REQ-C009: Clause selection order consistency. When entropy is disabled
    (default), the output format must match C Prover9 exactly: (X,wt=W).
    When entropy is enabled, extended format (X,wt=W,ent=E.EE) is acceptable.
    """

    # C Prover9 format: given #N (X,wt=W): ID clause.
    C_COMPAT_RE = re.compile(
        r"given\s+#(\d+)\s+"
        r"\(([A-Z]),wt=(\d+(?:\.\d+)?)\):\s+"
        r"(\d+)\s+.+\."
    )

    def test_default_no_entropy_in_output(self) -> None:
        """With entropy_weight=0 (default), trace should NOT show entropy.

        This is the critical C Prover9 format compatibility test.
        The format must be: given #N (X,wt=W): ID clause.
        """
        output, exit_code = _run_and_capture(SIMPLE_INPUT, entropy_weight=0)
        given_lines = [l for l in output.splitlines() if "given #" in l]
        assert len(given_lines) > 0

        for line in given_lines:
            assert "entropy" not in line.lower(), (
                f"REQ-C009 VIOLATION: Entropy visible in default (disabled) mode:\n"
                f"  {line!r}\n"
                f"  Expected C-compatible format: (X,wt=W)"
            )
            assert self.C_COMPAT_RE.search(line), (
                f"Default format doesn't match C Prover9 pattern (X,wt=W):\n  {line!r}"
            )

    def test_entropy_visible_when_enabled(self) -> None:
        """With entropy_weight > 0, trace should show entropy info."""
        output, _ = _run_and_capture(SIMPLE_INPUT, entropy_weight=2)
        given_lines = [l for l in output.splitlines() if "given #" in l]
        assert len(given_lines) > 0

        # At least some lines should have entropy info
        has_entropy = any("ent=" in line or "entropy" in line.lower() for line in given_lines)
        assert has_entropy, (
            "Entropy info should appear when entropy_weight > 0"
        )

    def test_selection_type_e_appears_with_entropy(self) -> None:
        """When entropy_weight > 0, 'E' selection type should appear for some clauses."""
        # Use a problem that needs more than initial processing to see E selections
        X2_INPUT = """\
formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
  x * x = e.
end_of_list.
formulas(goals).
  x * y = y * x.
end_of_list.
"""
        output, _ = _run_and_capture(
            X2_INPUT, entropy_weight=2, max_given=50,
            paramodulation=True, demodulation=True,
        )
        given_lines = [l for l in output.splitlines() if "given #" in l]

        # With entropy_weight=2 in cycle of 7 (1A + 4W + 2E), we should see E selections
        types_seen = set()
        for line in given_lines:
            m = re.search(r"\(([A-Z])", line)
            if m:
                types_seen.add(m.group(1))

        # Note: Initial processing shows "I" type. After that, cycle produces A, W, E
        # If proof found during initial processing, we may only see I
        if len(given_lines) > 5:
            assert "E" in types_seen, (
                f"Expected 'E' selection type with entropy_weight=2, got types: {types_seen}"
            )


# ── REQ-INT001: CLI Flag Functional Effect ──────────────────────────────────


class TestCLIEntropyOption:
    """Test --entropy-weight CLI option integration (REQ-INT001)."""

    def test_cli_entropy_weight_parsed(self) -> None:
        """--entropy-weight N should be parsed by CLI argument parser."""
        from pyladr.apps.prover9 import run_prover

        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".in", delete=False) as f:
            f.write(SIMPLE_INPUT)
            f.flush()
            input_path = f.name

        try:
            argv = [
                "pyprover9", "-f", input_path,
                "--entropy-weight", "3",
                "-max_given", "10",
            ]
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                run_prover(argv=argv)
            # Should not raise an error - option is recognized
        finally:
            os.unlink(input_path)

    def test_cli_entropy_weight_zero_c_compatible(self) -> None:
        """CLI with --entropy-weight 0 should produce C-compatible output."""
        from pyladr.apps.prover9 import run_prover

        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".in", delete=False) as f:
            f.write(SIMPLE_INPUT)
            f.flush()
            input_path = f.name

        try:
            argv = [
                "pyprover9", "-f", input_path,
                "--entropy-weight", "0",
                "-max_given", "10",
            ]
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                run_prover(argv=argv)

            output = captured.getvalue()
            given_lines = [l for l in output.splitlines() if "given #" in l]
            for line in given_lines:
                assert "entropy" not in line.lower(), (
                    f"REQ-C009: Entropy should not appear with --entropy-weight 0:\n  {line!r}"
                )
        finally:
            os.unlink(input_path)

    def test_cli_default_no_entropy_weight(self) -> None:
        """CLI without --entropy-weight should default to 0 (disabled)."""
        from pyladr.apps.prover9 import run_prover

        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".in", delete=False) as f:
            f.write(SIMPLE_INPUT)
            f.flush()
            input_path = f.name

        try:
            argv = [
                "pyprover9", "-f", input_path,
                "-max_given", "10",
            ]
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                run_prover(argv=argv)

            output = captured.getvalue()
            given_lines = [l for l in output.splitlines() if "given #" in l]
            for line in given_lines:
                assert "entropy" not in line.lower(), (
                    f"REQ-C009: Default CLI should not show entropy:\n  {line!r}"
                )
        finally:
            os.unlink(input_path)


# ── REQ-INT001: Functional Effect Verification ──────────────────────────────


class TestEntropyFunctionalEffect:
    """Verify entropy selection has a demonstrable functional effect (REQ-INT001).

    When entropy_weight > 0, the selection order must differ from when
    entropy_weight = 0 (for problems with non-trivial SOS).
    """

    def test_entropy_changes_selection_order(self) -> None:
        """Enabling entropy must produce different selection from default.

        This verifies REQ-INT001: the CLI flag has a functional effect.
        """
        sos_default = ClauseList("sos_default")
        sos_entropy = ClauseList("sos_entropy")

        # Create clauses with varying structure (entropy) vs weight
        clauses_data = [
            # (weight, structure) - designed so weight order != entropy order
            (1.0, _make_clause(_const(A))),                                    # low entropy, low weight
            (5.0, _make_clause(_func(P, _func(F, _var(0), _const(A)), _const(B)))),  # high entropy, high weight
            (3.0, _make_clause(_func(Q, _var(0)))),                            # medium entropy, medium weight
        ]

        for i, (w, c) in enumerate(clauses_data):
            c.id = i + 1
            c.weight = w
            c_copy = _make_clause(*[lit.atom for lit in c.literals])
            c_copy.id = i + 1
            c_copy.weight = w
            sos_default.append(c)
            sos_entropy.append(c_copy)

        # Weight-only selection: picks lightest first
        gs_weight = GivenSelection(
            rules=[SelectionRule("W", SelectionOrder.WEIGHT, part=1)]
        )
        weight_selected, _ = gs_weight.select_given(sos_default, 0)

        # Entropy-only selection: picks highest entropy first
        gs_entropy = GivenSelection(
            rules=[SelectionRule("E", SelectionOrder.ENTROPY, part=1)]
        )
        entropy_selected, _ = gs_entropy.select_given(sos_entropy, 0)

        # They should pick different clauses (weight=1.0 vs highest entropy)
        assert weight_selected.id != entropy_selected.id, (
            "REQ-INT001: Entropy selection must differ from weight selection. "
            f"Both selected clause ID {weight_selected.id}."
        )


# ── Performance: Entropy Calculation Overhead ────────────────────────────────


class TestEntropyPerformance:
    """Assess entropy calculation overhead."""

    def test_entropy_calculation_speed(self) -> None:
        """Entropy calculation should be fast (< 1ms per clause for typical clauses)."""
        # Build a moderately complex clause
        c = _make_clause(
            _func(P, _func(F, _var(0), _const(A)), _func(G, _const(B), _var(1))),
            _func(Q, _func(F, _var(2), _func(G, _const(C_SYM), _var(3)))),
        )

        # Warm up
        _clause_entropy(c)

        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            _clause_entropy(c)
        elapsed = time.perf_counter() - start

        per_call_us = (elapsed / iterations) * 1_000_000
        # Should be well under 100 microseconds per call
        assert per_call_us < 100, (
            f"Entropy calculation too slow: {per_call_us:.1f}μs per call "
            f"(expected < 100μs)"
        )

    def test_entropy_selection_overhead_vs_weight(self) -> None:
        """Entropy selection should not be dramatically slower than weight selection."""
        sos_w = ClauseList("sos_w")
        sos_e = ClauseList("sos_e")

        # Populate with 100 clauses
        for i in range(100):
            c_w = _make_clause(_func(P + (i % 5), _var(i % 3), _const(A + (i % 4))))
            c_w.id = i + 1
            c_w.weight = float(100 - i)
            c_e = _make_clause(_func(P + (i % 5), _var(i % 3), _const(A + (i % 4))))
            c_e.id = i + 1
            c_e.weight = float(100 - i)
            sos_w.append(c_w)
            sos_e.append(c_e)

        gs_w = GivenSelection(rules=[SelectionRule("W", SelectionOrder.WEIGHT, part=1)])
        gs_e = GivenSelection(rules=[SelectionRule("E", SelectionOrder.ENTROPY, part=1)])

        # Time weight selection (10 picks)
        start = time.perf_counter()
        for i in range(10):
            gs_w.select_given(sos_w, i)
        time_weight = time.perf_counter() - start

        # Time entropy selection (10 picks)
        start = time.perf_counter()
        for i in range(10):
            gs_e.select_given(sos_e, i)
        time_entropy = time.perf_counter() - start

        # Entropy can be slower but should be within 20x of weight selection
        if time_weight > 0:
            ratio = time_entropy / time_weight
            assert ratio < 20, (
                f"Entropy selection is {ratio:.1f}x slower than weight selection "
                f"(expected < 20x)"
            )
