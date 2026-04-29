"""Unit tests for pyladr.search.clause_formatting."""

from __future__ import annotations

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.parsing.ladr_parser import parse_clause
from pyladr.search.clause_formatting import format_clause_std


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
