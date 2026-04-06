"""Tests for pyladr.parsing — LADR syntax parser.

Tests behavioral equivalence with C parse.c:
- Tokenization of LADR input
- Term parsing with operator precedence
- Formula/clause block parsing
- Variable detection and conversion
- Comment handling
- Error handling for malformed input
- Cross-validation against C reference parsing
"""

from __future__ import annotations

import pytest

from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import (
    LADRParser,
    ParseError,
    parse_clause,
    parse_input,
    parse_term,
)
from pyladr.parsing.tokenizer import TokenType, strip_comments, tokenize


# ── Tokenizer tests ─────────────────────────────────────────────────────────


class TestTokenizer:
    def test_empty(self):
        assert tokenize("") == []

    def test_ordinary_token(self):
        tokens = tokenize("abc")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.ORDINARY
        assert tokens[0].value == "abc"

    def test_special_token(self):
        tokens = tokenize("->")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.SPECIAL
        assert tokens[0].value == "->"

    def test_mixed_tokens(self):
        tokens = tokenize("a * b")
        assert len(tokens) == 3
        assert tokens[0].value == "a"
        assert tokens[1].value == "*"
        assert tokens[2].value == "b"

    def test_punctuation(self):
        tokens = tokenize("f(a,b)")
        types = [t.type for t in tokens]
        assert types == [
            TokenType.ORDINARY,
            TokenType.PUNC,
            TokenType.ORDINARY,
            TokenType.PUNC,
            TokenType.ORDINARY,
            TokenType.PUNC,
        ]

    def test_special_sequence(self):
        tokens = tokenize("!=")
        assert len(tokens) == 1
        assert tokens[0].value == "!="

    def test_prime_postfix(self):
        tokens = tokenize("x'")
        assert len(tokens) == 2
        assert tokens[0].value == "x"
        assert tokens[1].value == "'"

    def test_dollar_symbol(self):
        tokens = tokenize("$F")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.ORDINARY
        assert tokens[0].value == "$F"

    def test_whitespace_ignored(self):
        tokens = tokenize("  a   b  ")
        assert len(tokens) == 2

    def test_negation_prefix(self):
        """'-' is a special char and tokenizes separately from ordinary."""
        tokens = tokenize("-P(x)")
        assert tokens[0].type == TokenType.SPECIAL
        assert tokens[0].value == "-"

    def test_quoted_string(self):
        tokens = tokenize('"hello world"')
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == '"hello world"'


class TestStripComments:
    def test_no_comments(self):
        assert strip_comments("a * b") == "a * b"

    def test_line_comment(self):
        assert strip_comments("a * b % comment\nc") == "a * b \nc"

    def test_comment_only(self):
        assert strip_comments("% comment\n").strip() == ""

    def test_preserves_quoted(self):
        assert strip_comments('"has % inside"') == '"has % inside"'

    def test_multiple_comments(self):
        result = strip_comments("a % c1\nb % c2\nc")
        assert result == "a \nb \nc"


# ── Term parsing tests ──────────────────────────────────────────────────────


class TestTermParsing:
    def test_parse_constant(self):
        table = SymbolTable()
        t = parse_term("a.", table)
        assert t.is_constant
        assert table.sn_to_str(t.symnum) == "a"

    def test_parse_variable(self):
        """Variables start with u-z in STANDARD style."""
        table = SymbolTable()
        t = parse_term("x.", table)
        assert t.is_variable
        assert t.varnum == 0

    def test_parse_multiple_variables(self):
        """Different variables get different numbers."""
        table = SymbolTable()
        parser = LADRParser(table)
        t = parser.parse_term("f(x, y).")
        assert t.is_complex
        assert t.arg(0).is_variable
        assert t.arg(1).is_variable
        assert t.arg(0).varnum != t.arg(1).varnum

    def test_function_application(self):
        table = SymbolTable()
        t = parse_term("f(a, b).", table)
        assert t.is_complex
        assert t.arity == 2
        assert table.sn_to_str(t.symnum) == "f"
        assert t.arg(0).is_constant
        assert t.arg(1).is_constant

    def test_infix_multiplication(self):
        """'a * b' parses as *(a, b)."""
        table = SymbolTable()
        t = parse_term("a * b.", table)
        assert t.is_complex
        assert t.arity == 2
        assert table.sn_to_str(t.symnum) == "*"

    def test_infix_equality(self):
        """'a = b' parses as =(a, b)."""
        table = SymbolTable()
        t = parse_term("a = b.", table)
        assert t.is_complex
        assert t.arity == 2
        assert table.sn_to_str(t.symnum) == "="

    def test_prime_notation(self):
        """\"x'\" parses as '(x) — unary postfix."""
        table = SymbolTable()
        t = parse_term("x'.", table)
        assert t.is_complex
        assert t.arity == 1
        assert table.sn_to_str(t.symnum) == "'"
        assert t.arg(0).is_variable

    def test_nested_operations(self):
        """'(x * y) * z' groups correctly."""
        table = SymbolTable()
        t = parse_term("(x * y) * z.", table)
        assert t.is_complex
        assert table.sn_to_str(t.symnum) == "*"
        # Left arg should be (x * y)
        left = t.arg(0)
        assert left.is_complex
        assert table.sn_to_str(left.symnum) == "*"
        # Right arg is z
        assert t.arg(1).is_variable

    def test_negation_tilde(self):
        """'-P(x)' parses as negation of P(x)."""
        table = SymbolTable()
        t = parse_term("-P(a).", table)
        assert t.is_complex
        assert t.arity == 1
        assert table.sn_to_str(t.symnum) == "-"

    def test_inequality(self):
        """'a != b' parses as !=(a, b)."""
        table = SymbolTable()
        t = parse_term("a != b.", table)
        assert t.is_complex
        assert t.arity == 2
        assert table.sn_to_str(t.symnum) == "!="

    def test_equality_with_complex_terms(self):
        """'e * x = x'."""
        table = SymbolTable()
        t = parse_term("e * x = x.", table)
        assert t.is_complex
        assert table.sn_to_str(t.symnum) == "="
        # LHS is e * x
        lhs = t.arg(0)
        assert lhs.is_complex
        assert table.sn_to_str(lhs.symnum) == "*"

    def test_associativity_precedence(self):
        """'x * y * z' should parse as (x * y) * z (INFIX = left assoc by default)."""
        table = SymbolTable()
        t = parse_term("x * y * z.", table)
        assert t.is_complex
        # Top should be * with left arg being (* x y)
        assert table.sn_to_str(t.symnum) == "*"
        left = t.arg(0)
        assert left.is_complex
        assert table.sn_to_str(left.symnum) == "*"

    def test_precedence_eq_over_mult(self):
        """'e * x = x' should parse as =(*(e,x), x) since * (500) < = (700)."""
        table = SymbolTable()
        t = parse_term("e * x = x.", table)
        assert table.sn_to_str(t.symnum) == "="

    def test_precedence_impl_over_or(self):
        """'a | b -> c' should parse as ->( |(a,b), c) since | (790) < -> (800)."""
        table = SymbolTable()
        t = parse_term("a | b -> c.", table)
        assert table.sn_to_str(t.symnum) == "->"

    def test_inverse_in_equation(self):
        """\"x' * x = e\" — prime binds tighter than *."""
        table = SymbolTable()
        t = parse_term("x' * x = e.", table)
        assert table.sn_to_str(t.symnum) == "="
        lhs = t.arg(0)
        assert table.sn_to_str(lhs.symnum) == "*"
        # lhs of * should be '(x)
        inv = lhs.arg(0)
        assert table.sn_to_str(inv.symnum) == "'"

    def test_complex_group_axiom(self):
        """'(x * y) * z = x * (y * z)' — associativity axiom."""
        table = SymbolTable()
        t = parse_term("(x * y) * z = x * (y * z).", table)
        assert table.sn_to_str(t.symnum) == "="

    def test_false_symbol(self):
        table = SymbolTable()
        t = parse_term("$F.", table)
        assert t.is_constant
        assert table.sn_to_str(t.symnum) == "$F"


# ── Variable detection tests ────────────────────────────────────────────────


class TestVariableDetection:
    """Test that variable names are correctly identified (C variable_name)."""

    def test_standard_vars(self):
        """u, v, w, x, y, z are variables in standard style."""
        from pyladr.parsing.ladr_parser import is_variable_name

        for name in ["x", "y", "z", "u", "v", "w"]:
            assert is_variable_name(name), f"{name} should be a variable"

    def test_standard_non_vars(self):
        """a-t, A-Z are NOT variables in standard style."""
        from pyladr.parsing.ladr_parser import is_variable_name

        for name in ["a", "b", "e", "f", "P", "A", "0"]:
            assert not is_variable_name(name), f"{name} should not be a variable"

    def test_multi_char_vars(self):
        """x1, y2 etc start with variable chars."""
        from pyladr.parsing.ladr_parser import is_variable_name

        assert is_variable_name("x1")
        assert is_variable_name("y_prime")

    def test_c1_c2_are_constants(self):
        """c1, c2 do NOT start with u-z, so they are constants."""
        from pyladr.parsing.ladr_parser import is_variable_name

        assert not is_variable_name("c1")
        assert not is_variable_name("c2")


# ── Clause parsing tests ────────────────────────────────────────────────────


class TestClauseParsing:
    def test_unit_positive_clause(self):
        table = SymbolTable()
        c = parse_clause("P(a).", table)
        assert c.num_literals == 1
        assert c.literals[0].is_positive

    def test_unit_negative_clause(self):
        table = SymbolTable()
        c = parse_clause("-P(a).", table)
        assert c.num_literals == 1
        assert c.literals[0].is_negative

    def test_disjunctive_clause(self):
        """'P(x) | Q(x)' → clause with two positive literals."""
        table = SymbolTable()
        c = parse_clause("P(x) | Q(x).", table)
        assert c.num_literals == 2
        assert all(lit.is_positive for lit in c.literals)

    def test_mixed_clause(self):
        """-P(x) | Q(x) → one negative, one positive."""
        table = SymbolTable()
        c = parse_clause("-P(x) | Q(x).", table)
        assert c.num_literals == 2
        assert c.literals[0].is_negative
        assert c.literals[1].is_positive

    def test_equality_clause(self):
        """'e * x = x' is a unit clause with equality literal."""
        table = SymbolTable()
        c = parse_clause("e * x = x.", table)
        assert c.num_literals == 1
        assert c.literals[0].is_positive
        assert c.literals[0].is_eq_literal

    def test_inequality_clause(self):
        """'c2 * c1 != c1 * c2' — negated equality."""
        table = SymbolTable()
        c = parse_clause("c2 * c1 != c1 * c2.", table)
        assert c.num_literals == 1
        # != should produce a negative equality literal
        # (In C, != is parsed as a special binary, but for clause conversion
        # it represents a != b which is the negation of a = b)

    def test_empty_clause(self):
        """$F is the empty clause."""
        table = SymbolTable()
        c = parse_clause("$F.", table)
        assert c.is_empty


# ── Input file parsing tests ────────────────────────────────────────────────


class TestInputParsing:
    def test_parse_sos_block(self):
        text = """\
formulas(sos).
  e * x = x.
  x' * x = e.
end_of_list.
"""
        result = parse_input(text)
        assert len(result.sos) == 2
        assert len(result.goals) == 0

    def test_parse_goals_block(self):
        text = """\
formulas(goals).
  x * y = y * x.
end_of_list.
"""
        result = parse_input(text)
        assert len(result.goals) == 1

    def test_parse_complete_input(self):
        """Parse a complete input matching x2.in format."""
        text = """\
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
        result = parse_input(text)
        assert len(result.sos) == 4
        assert len(result.goals) == 1

    def test_parse_with_comments(self):
        text = """\
% This is a comment
formulas(sos).
  % Another comment
  e * x = x.
end_of_list.
"""
        result = parse_input(text)
        assert len(result.sos) == 1

    def test_parse_lattice_input(self):
        text = """\
formulas(sos).
  x ^ y = y ^ x.
  x v y = y v x.
  (x ^ y) ^ z = x ^ (y ^ z).
  (x v y) v z = x v (y v z).
  x ^ (x v y) = x.
  x v (x ^ y) = x.
end_of_list.

formulas(goals).
  x ^ x = x.
end_of_list.
"""
        result = parse_input(text)
        assert len(result.sos) == 6
        assert len(result.goals) == 1


# ── Cross-validation with C reference ────────────────────────────────────────


class TestCrossValidation:
    """Parse the same input files as C Prover9 and verify structure."""

    def test_parse_x2_input(self, c_examples_dir):
        """Parse the x2 example — the canonical Prover9 test."""
        input_file = c_examples_dir / "x2.in"
        if not input_file.exists():
            pytest.skip("x2.in not found")
        text = input_file.read_text()
        result = parse_input(text)
        # x2 has 4 sos axioms and 1 goal
        assert len(result.sos) == 4
        assert len(result.goals) == 1

    def test_parse_custom_simple_group(self, test_inputs_dir):
        input_file = test_inputs_dir / "simple_group.in"
        if not input_file.exists():
            pytest.skip("simple_group.in not found")
        text = input_file.read_text()
        result = parse_input(text)
        assert len(result.sos) == 4
        assert len(result.goals) == 1

    def test_parse_identity_only(self, test_inputs_dir):
        input_file = test_inputs_dir / "identity_only.in"
        if not input_file.exists():
            pytest.skip("identity_only.in not found")
        text = input_file.read_text()
        result = parse_input(text)
        assert len(result.sos) == 1
        assert len(result.goals) == 1

    def test_parse_lattice_absorption(self, test_inputs_dir):
        input_file = test_inputs_dir / "lattice_absorption.in"
        if not input_file.exists():
            pytest.skip("lattice_absorption.in not found")
        text = input_file.read_text()
        result = parse_input(text)
        assert len(result.sos) == 6
        assert len(result.goals) == 1


# ── Error handling tests ─────────────────────────────────────────────────────


class TestParseErrors:
    def test_unclosed_parenthesis(self):
        with pytest.raises(ParseError):
            parse_term("f(a, b.")

    def test_empty_input(self):
        with pytest.raises(ParseError):
            parse_term(".")

    def test_unexpected_close_paren(self):
        with pytest.raises(ParseError):
            parse_term(").")
