"""Tests for pyladr.core.clause — Clause and Literal structures.

Tests behavioral equivalence with C literals.h, topform.h:
- Literal construction and properties
- Clause construction and analysis
- Clause properties (horn, unit, positive, negative, mixed)
- String representation (fprint_clause)
"""

from __future__ import annotations

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.term import get_rigid_term, get_variable_term


def _make_atom(symnum: int, *args):
    """Helper to create an atomic term."""
    arg_terms = tuple(args)
    return get_rigid_term(symnum, len(arg_terms), arg_terms)


class TestLiteral:
    """Test literal creation and properties."""

    def test_positive_literal(self):
        atom = _make_atom(1, get_variable_term(0))
        lit = Literal(sign=True, atom=atom)
        assert lit.is_positive
        assert not lit.is_negative

    def test_negative_literal(self):
        atom = _make_atom(1, get_variable_term(0))
        lit = Literal(sign=False, atom=atom)
        assert not lit.is_positive
        assert lit.is_negative

    def test_complementary(self):
        atom = _make_atom(1, get_variable_term(0))
        pos = Literal(sign=True, atom=atom)
        neg = Literal(sign=False, atom=atom)
        assert pos.complementary(neg)
        assert neg.complementary(pos)

    def test_not_complementary_same_sign(self):
        atom = _make_atom(1, get_variable_term(0))
        p1 = Literal(sign=True, atom=atom)
        p2 = Literal(sign=True, atom=atom)
        assert not p1.complementary(p2)

    def test_not_complementary_different_atom(self):
        a1 = _make_atom(1, get_variable_term(0))
        a2 = _make_atom(2, get_variable_term(0))
        p = Literal(sign=True, atom=a1)
        n = Literal(sign=False, atom=a2)
        assert not p.complementary(n)

    def test_equality_literal(self):
        """=(a, b) has arity 2, recognized as equality."""
        a = get_rigid_term(2, 0)
        b = get_rigid_term(3, 0)
        eq_atom = get_rigid_term(1, 2, (a, b))
        lit = Literal(sign=True, atom=eq_atom)
        assert lit.is_eq_literal

    def test_negative_equality(self):
        """a != b is a negative equality literal."""
        a = get_rigid_term(2, 0)
        b = get_rigid_term(3, 0)
        eq_atom = get_rigid_term(1, 2, (a, b))
        lit = Literal(sign=False, atom=eq_atom)
        assert lit.is_eq_literal
        assert lit.is_negative

    def test_literal_str_positive(self):
        atom = _make_atom(1, get_variable_term(0))
        lit = Literal(sign=True, atom=atom)
        assert lit.to_str() == "s1(v0)"

    def test_literal_str_negative(self):
        atom = _make_atom(1, get_variable_term(0))
        lit = Literal(sign=False, atom=atom)
        assert lit.to_str() == "-s1(v0)"


class TestClauseProperties:
    """Test clause classification matching C behavior."""

    def test_empty_clause(self):
        c = Clause()
        assert c.is_empty
        assert c.num_literals == 0

    def test_unit_clause(self):
        atom = _make_atom(1, get_variable_term(0))
        lit = Literal(sign=True, atom=atom)
        c = Clause(literals=(lit,))
        assert c.is_unit
        assert not c.is_empty

    def test_positive_clause(self):
        a1 = _make_atom(1, get_variable_term(0))
        a2 = _make_atom(2, get_variable_term(1))
        c = Clause(literals=(
            Literal(sign=True, atom=a1),
            Literal(sign=True, atom=a2),
        ))
        assert c.is_positive
        assert not c.is_negative

    def test_negative_clause(self):
        a1 = _make_atom(1, get_variable_term(0))
        a2 = _make_atom(2, get_variable_term(1))
        c = Clause(literals=(
            Literal(sign=False, atom=a1),
            Literal(sign=False, atom=a2),
        ))
        assert c.is_negative
        assert not c.is_positive

    def test_mixed_clause(self):
        a1 = _make_atom(1, get_variable_term(0))
        a2 = _make_atom(2, get_variable_term(1))
        c = Clause(literals=(
            Literal(sign=True, atom=a1),
            Literal(sign=False, atom=a2),
        ))
        assert not c.is_positive
        assert not c.is_negative

    def test_horn_clause(self):
        """p(x) | -q(y) is Horn (1 positive literal)."""
        a1 = _make_atom(1, get_variable_term(0))
        a2 = _make_atom(2, get_variable_term(1))
        c = Clause(literals=(
            Literal(sign=True, atom=a1),
            Literal(sign=False, atom=a2),
        ))
        assert c.is_horn
        assert c.is_definite

    def test_non_horn_clause(self):
        a1 = _make_atom(1, get_variable_term(0))
        a2 = _make_atom(2, get_variable_term(1))
        c = Clause(literals=(
            Literal(sign=True, atom=a1),
            Literal(sign=True, atom=a2),
        ))
        assert not c.is_horn

    def test_ground_clause(self):
        a = get_rigid_term(2, 0)
        atom = _make_atom(1, a)
        c = Clause(literals=(Literal(sign=True, atom=atom),))
        assert c.is_ground

    def test_non_ground_clause(self):
        atom = _make_atom(1, get_variable_term(0))
        c = Clause(literals=(Literal(sign=True, atom=atom),))
        assert not c.is_ground

    def test_clause_variables(self):
        x, y = get_variable_term(0), get_variable_term(2)
        a1 = _make_atom(1, x, y)
        a2 = _make_atom(2, x)
        c = Clause(literals=(
            Literal(sign=True, atom=a1),
            Literal(sign=False, atom=a2),
        ))
        assert c.variables() == {0, 2}


class TestClauseString:
    """Test clause string representation (C fprint_clause)."""

    def test_empty_clause_str(self):
        c = Clause()
        assert c.to_str() == "$F."

    def test_unit_clause_str(self):
        atom = _make_atom(1, get_variable_term(0))
        c = Clause(literals=(Literal(sign=True, atom=atom),))
        assert c.to_str() == "s1(v0)."

    def test_clause_with_id(self):
        atom = _make_atom(1, get_variable_term(0))
        c = Clause(id=5, literals=(Literal(sign=True, atom=atom),))
        assert c.to_str() == "5: s1(v0)."

    def test_multi_literal_str(self):
        """p(x) | -q(y) format."""
        a1 = _make_atom(1, get_variable_term(0))
        a2 = _make_atom(2, get_variable_term(1))
        c = Clause(literals=(
            Literal(sign=True, atom=a1),
            Literal(sign=False, atom=a2),
        ))
        assert c.to_str() == "s1(v0) | -s2(v1)."


class TestJustification:
    def test_input_justification(self):
        j = Justification(just_type=JustType.INPUT)
        assert j.just_type == JustType.INPUT

    def test_binary_res_justification(self):
        j = Justification(just_type=JustType.BINARY_RES, clause_ids=(1, 2))
        assert j.just_type == JustType.BINARY_RES
        assert j.clause_ids == (1, 2)
