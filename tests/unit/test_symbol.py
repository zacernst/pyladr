"""Tests for pyladr.core.symbol — Symbol table management."""

from __future__ import annotations

from pyladr.core.symbol import (
    ParseType,
    SymbolTable,
    SymbolType,
    UnifTheory,
    VariableStyle,
)


class TestSymbolTable:
    """Test symbol table creation and lookup."""

    def test_create_symbol(self):
        st = SymbolTable()
        sn = st.str_to_sn("f", 2)
        assert sn == 1
        assert st.sn_to_str(sn) == "f"
        assert st.sn_to_arity(sn) == 2

    def test_same_name_arity_returns_same_id(self):
        st = SymbolTable()
        sn1 = st.str_to_sn("f", 2)
        sn2 = st.str_to_sn("f", 2)
        assert sn1 == sn2

    def test_same_name_different_arity(self):
        """f/2 and f/1 are different symbols (C behavior)."""
        st = SymbolTable()
        sn1 = st.str_to_sn("f", 2)
        sn2 = st.str_to_sn("f", 1)
        assert sn1 != sn2

    def test_sequential_ids(self):
        st = SymbolTable()
        sn1 = st.str_to_sn("a", 0)
        sn2 = st.str_to_sn("b", 0)
        sn3 = st.str_to_sn("f", 2)
        assert sn1 == 1
        assert sn2 == 2
        assert sn3 == 3

    def test_is_symbol(self):
        st = SymbolTable()
        sn = st.str_to_sn("p", 1)
        assert st.is_symbol(sn, "p", 1)
        assert not st.is_symbol(sn, "p", 2)
        assert not st.is_symbol(sn, "q", 1)
        assert not st.is_symbol(999, "p", 1)

    def test_len(self):
        st = SymbolTable()
        assert len(st) == 0
        st.str_to_sn("a", 0)
        assert len(st) == 1
        st.str_to_sn("b", 0)
        assert len(st) == 2

    def test_contains(self):
        st = SymbolTable()
        sn = st.str_to_sn("a", 0)
        assert sn in st
        assert 999 not in st


class TestSymbolProperties:
    """Test symbol property modification."""

    def test_default_properties(self):
        st = SymbolTable()
        sn = st.str_to_sn("f", 2)
        sym = st.get_symbol(sn)
        assert sym.sym_type == SymbolType.UNSPECIFIED
        assert sym.parse_type == ParseType.ORDINARY
        assert sym.parse_prec == 0
        assert sym.unif_theory == UnifTheory.EMPTY_THEORY
        assert sym.kb_weight == 1
        assert not sym.skolem

    def test_set_parse_type(self):
        st = SymbolTable()
        sn = st.str_to_sn("+", 2)
        st.set_parse_type(sn, ParseType.INFIX_LEFT, 500)
        sym = st.get_symbol(sn)
        assert sym.parse_type == ParseType.INFIX_LEFT
        assert sym.parse_prec == 500

    def test_set_kb_weight(self):
        st = SymbolTable()
        sn = st.str_to_sn("f", 1)
        st.set_kb_weight(sn, 3)
        assert st.get_symbol(sn).kb_weight == 3

    def test_mark_skolem(self):
        st = SymbolTable()
        sn = st.str_to_sn("c1", 0)
        st.mark_skolem(sn)
        assert st.get_symbol(sn).skolem

    def test_increment_occurrences(self):
        st = SymbolTable()
        sn = st.str_to_sn("f", 2)
        st.increment_occurrences(sn)
        st.increment_occurrences(sn)
        assert st.get_symbol(sn).occurrences == 2


class TestVariableFormatting:
    """Test variable display in different styles."""

    def test_standard_style(self):
        st = SymbolTable()
        st.variable_style = VariableStyle.STANDARD
        assert st.format_variable(0) == "x"
        assert st.format_variable(1) == "y"
        assert st.format_variable(2) == "z"
        assert st.format_variable(3) == "u"
        assert st.format_variable(4) == "v"
        assert st.format_variable(5) == "w"
        assert st.format_variable(6) == "v6"
        assert st.format_variable(10) == "v10"

    def test_prolog_style(self):
        st = SymbolTable()
        st.variable_style = VariableStyle.PROLOG
        assert st.format_variable(0) == "A"
        assert st.format_variable(1) == "B"
        assert st.format_variable(25) == "Z"
        assert st.format_variable(26) == "V26"

    def test_integer_style(self):
        st = SymbolTable()
        st.variable_style = VariableStyle.INTEGER
        assert st.format_variable(0) == "0"
        assert st.format_variable(42) == "42"


class TestSymbolTableWithTerms:
    """Test symbol table integration with term formatting."""

    def test_term_to_str_with_table(self):
        from pyladr.core.term import get_rigid_term, get_variable_term

        st = SymbolTable()
        f_sn = st.str_to_sn("f", 2)
        a_sn = st.str_to_sn("a", 0)

        x = get_variable_term(0)
        a = get_rigid_term(a_sn, 0)
        fxa = get_rigid_term(f_sn, 2, (x, a))

        assert fxa.to_str(st) == "f(v0,a)"
