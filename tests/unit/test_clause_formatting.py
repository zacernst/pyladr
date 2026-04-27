"""Unit tests for pyladr.search.clause_formatting — formatting and entropy."""

from __future__ import annotations

import math

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.parsing.ladr_parser import parse_clause
from pyladr.search.clause_formatting import (
    calculate_structural_entropy,
    format_clause_std,
)


# ── calculate_structural_entropy tests ───────────────────────────────────────


class TestStructuralEntropy:
    def test_ground_clause_nonnegative(self):
        """Ground clause P(a) gives non-negative finite entropy."""
        table = SymbolTable()
        c = parse_clause("P(a).", table)
        h = calculate_structural_entropy(c)
        assert h >= 0.0
        assert math.isfinite(h)

    def test_simple_atom_entropy_value(self):
        """P(a) has nodes: clause=1, literal=1, predicate=1, constant=1.
        Total=4, each p=0.25, H = -4*(0.25*log2(0.25)) = 2.0."""
        table = SymbolTable()
        c = parse_clause("P(a).", table)
        h = calculate_structural_entropy(c)
        assert abs(h - 2.0) < 1e-9

    def test_empty_clause_zero_entropy(self):
        """Empty clause ($F) has only 1 node → entropy 0."""
        c = Clause(literals=(), id=1)
        assert calculate_structural_entropy(c) == 0.0

    def test_variable_clause_does_not_crash(self):
        """Clause with variables computes without error."""
        table = SymbolTable()
        c = parse_clause("P(x).", table)
        h = calculate_structural_entropy(c)
        assert h >= 0.0
        assert math.isfinite(h)

    def test_complex_clause_higher_entropy(self):
        """More diverse node types → higher entropy than a simple clause."""
        table = SymbolTable()
        simple = parse_clause("P(a).", table)
        # P(f(x), a) has: clause, literal, predicate, function, variable, constant
        complex_c = parse_clause("P(f(x), a).", table)
        h_simple = calculate_structural_entropy(simple)
        h_complex = calculate_structural_entropy(complex_c)
        # Complex has all 6 node types populated → higher entropy
        assert h_complex > h_simple


# ── format_clause_std tests ──────────────────────────────────────────────────


def _parsed(text: str, table: SymbolTable, clause_id: int = 1) -> Clause:
    """Parse a clause and set its id."""
    c = parse_clause(text, table)
    c.id = clause_id
    return c


class TestFormatClauseStd:
    def test_basic_clause_contains_predicate(self):
        """Formatted clause contains the predicate name."""
        table = SymbolTable()
        c = _parsed("P(a).", table, clause_id=1)
        s = format_clause_std(table, c)
        assert "P" in s
        assert "a" in s

    def test_clause_id_in_output(self):
        """Clause ID appears at the start."""
        table = SymbolTable()
        c = _parsed("P(a).", table, clause_id=42)
        s = format_clause_std(table, c)
        assert s.startswith("42 ")

    def test_empty_clause_formats_as_dollar_f(self):
        """Empty clause formats as $F."""
        c = Clause(literals=(), id=1)
        s = format_clause_std(None, c)
        assert "$F" in s

    def test_disjunctive_clause_uses_pipe(self):
        """Multi-literal clause uses ' | ' separator."""
        table = SymbolTable()
        c = _parsed("P(a) | Q(b).", table)
        s = format_clause_std(table, c)
        assert " | " in s

    def test_negative_literal_has_dash(self):
        """Negative literal is prefixed with '-'."""
        table = SymbolTable()
        c = _parsed("-P(a).", table)
        s = format_clause_std(table, c)
        assert "-P" in s

    def test_none_symbol_table_does_not_crash(self):
        """Passing None as symbol table still produces output."""
        atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c = Clause(literals=(Literal(sign=True, atom=atom),), id=1)
        s = format_clause_std(None, c)
        assert isinstance(s, str)
        assert len(s) > 0
