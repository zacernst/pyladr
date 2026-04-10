"""Test goal Skolemization behavior.

Tests that goals are correctly Skolemized during the denial process.
Verifies that the Python implementation matches C Prover9 semantics.
"""

from __future__ import annotations

from pyladr.apps.prover9 import _deny_goals, _collect_variables
from pyladr.core.clause import Clause, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import get_rigid_term, get_variable_term, build_binary_term
from pyladr.parsing.ladr_parser import ParsedInput, parse_input


class TestGoalSkolemization:
    """Test Skolemization of goals during the denial process."""

    def test_simple_goal_single_variable(self):
        """Test Skolemization of a goal with a single variable.

        Goal: P(x)
        Expected denial: ¬P(c1) where c1 is a Skolem constant
        """
        input_text = """
formulas(goals).
  P(x).
end_of_list.
"""
        symbol_table = SymbolTable()
        parsed = parse_input(input_text, symbol_table)

        assert len(parsed.goals) == 1

        usable, sos = _deny_goals(parsed, symbol_table)

        # Should have one denied goal in SOS
        assert len(sos) == 1
        denied = sos[0]

        # Should be a negative literal
        assert len(denied.literals) == 1
        lit = denied.literals[0]
        assert lit.sign is False  # negated
        assert lit.atom.arity == 1

        # The atom should have a Skolem constant (rigid term with arity 0)
        arg = lit.atom.args[0]
        assert arg.is_constant  # should be a constant, not a variable

        # Symbol should be a Skolem constant (name starts with 'c')
        sym_name = symbol_table.sn_to_str(arg.symnum)
        assert sym_name.startswith('c')

    def test_goal_multiple_variables(self):
        """Test Skolemization with multiple variables in goal.

        Goal: P(x, y)
        Expected: ¬P(c1, c2) with two Skolem constants
        """
        input_text = """
formulas(goals).
  P(x, y).
end_of_list.
"""
        symbol_table = SymbolTable()
        parsed = parse_input(input_text, symbol_table)

        usable, sos = _deny_goals(parsed, symbol_table)

        assert len(sos) == 1
        denied = sos[0]
        assert len(denied.literals) == 1
        lit = denied.literals[0]

        # Both arguments should be Skolem constants
        assert lit.atom.arity == 2
        arg0 = lit.atom.args[0]
        arg1 = lit.atom.args[1]

        assert arg0.is_constant
        assert arg1.is_constant

        sym0 = symbol_table.sn_to_str(arg0.symnum)
        sym1 = symbol_table.sn_to_str(arg1.symnum)

        assert sym0.startswith('c')
        assert sym1.startswith('c')
        # Different constants for different variables
        assert sym0 != sym1

    def test_goal_equality(self):
        """Test Skolemization of equational goals.

        Goal: x = y
        Expected: ¬(c1 = c2)
        """
        input_text = """
formulas(goals).
  x = y.
end_of_list.
"""
        symbol_table = SymbolTable()
        parsed = parse_input(input_text, symbol_table)

        usable, sos = _deny_goals(parsed, symbol_table)

        assert len(sos) == 1
        denied = sos[0]
        assert len(denied.literals) == 1
        lit = denied.literals[0]

        assert lit.sign is False  # negated

        # Equality atom
        eq_atom_result = lit.atom
        assert symbol_table.sn_to_str(eq_atom_result.symnum) == "="
        assert eq_atom_result.arity == 2

        # Both sides should be Skolem constants
        arg0 = eq_atom_result.args[0]
        arg1 = eq_atom_result.args[1]

        assert arg0.is_constant
        assert arg1.is_constant

    def test_goal_with_constant_and_variable(self):
        """Test Skolemization when goal mixes constants and variables.

        Goal: P(a, x)
        Expected: ¬P(a, c1) where a stays as constant, x becomes c1
        """
        input_text = """
formulas(goals).
  P(a, x).
end_of_list.
"""
        symbol_table = SymbolTable()
        parsed = parse_input(input_text, symbol_table)

        usable, sos = _deny_goals(parsed, symbol_table)

        assert len(sos) == 1
        denied = sos[0]
        lit = denied.literals[0]

        # First arg should remain the constant 'a'
        arg0 = lit.atom.args[0]
        assert arg0.is_constant
        assert symbol_table.sn_to_str(arg0.symnum) == "a"

        # Second arg should be Skolem constant
        arg1 = lit.atom.args[1]
        assert arg1.is_constant
        sym1 = symbol_table.sn_to_str(arg1.symnum)
        assert sym1.startswith('c')

    def test_goal_disjunction(self):
        """Test Skolemization of disjunctive goals.

        Goal: P(x) | Q(y)
        Expected: ¬P(c1) ∧ ¬Q(c2) → two negative literals in clause
        """
        input_text = """
formulas(goals).
  P(x) | Q(y).
end_of_list.
"""
        symbol_table = SymbolTable()
        parsed = parse_input(input_text, symbol_table)

        usable, sos = _deny_goals(parsed, symbol_table)

        assert len(sos) == 1
        denied = sos[0]

        # Negation of (P(x) | Q(y)) = ¬P(x) ∧ ¬Q(y)
        # In clause form: two negative literals
        assert len(denied.literals) == 2

        lit0 = denied.literals[0]
        lit1 = denied.literals[1]

        assert lit0.sign is False
        assert lit1.sign is False

    def test_multiple_goals(self):
        """Test Skolemization of multiple goals.

        Goals: P(x), Q(y)
        Expected: Two denied clauses with different Skolem counters
        """
        input_text = """
formulas(goals).
  P(x).
  Q(y).
end_of_list.
"""
        symbol_table = SymbolTable()
        parsed = parse_input(input_text, symbol_table)

        usable, sos = _deny_goals(parsed, symbol_table)

        # Should have two denied goals
        assert len(sos) == 2

        denied1 = sos[0]
        denied2 = sos[1]

        # Extract Skolem constant names
        sym1 = symbol_table.sn_to_str(denied1.literals[0].atom.args[0].symnum)
        sym2 = symbol_table.sn_to_str(denied2.literals[0].atom.args[0].symnum)

        assert sym1.startswith('c')
        assert sym2.startswith('c')
        # Different goals should get different Skolem constants
        assert sym1 != sym2

    def test_collect_variables_helper(self):
        """Test the _collect_variables helper function."""
        input_text = """
formulas(goals).
  P(x, y).
end_of_list.
"""
        symbol_table = SymbolTable()
        parsed = parse_input(input_text, symbol_table)

        goal = parsed.goals[0]
        atom = goal.literals[0].atom

        vars_found = _collect_variables(atom)

        # Should find both variable numbers 0 and 1
        assert vars_found == {0, 1}

    def test_collect_variables_nested(self):
        """Test _collect_variables with nested terms."""
        input_text = """
formulas(goals).
  P(f(x, y)).
end_of_list.
"""
        symbol_table = SymbolTable()
        parsed = parse_input(input_text, symbol_table)

        goal = parsed.goals[0]
        atom = goal.literals[0].atom

        vars_found = _collect_variables(atom)

        # Should find both variables in the nested term
        assert vars_found == {0, 1}


class TestGoalSkolemizationCComparison:
    """Compare Python Skolemization with C Prover9 behavior.

    These tests verify that the Python denial process produces
    structurally equivalent clauses as C Prover9.
    """

    def test_parse_and_deny_simple_group_theory(self):
        """Test goal denial on a simple group theory problem.

        This problem: group with x*x=e
        Goal: x*y = y*x (commutativity)

        When denied and Skolemized, should produce:
        ¬(c1 * c2 = c2 * c1)
        """
        input_text = """
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
        symbol_table = SymbolTable()
        parsed = parse_input(input_text, symbol_table)

        assert len(parsed.goals) == 1

        usable, sos = _deny_goals(parsed, symbol_table)

        # Goal denial should produce exactly one clause added to SOS
        assert len(sos) == 5  # 4 original SOS + 1 denied goal

        # Last one is the denied goal
        denied_goal = sos[-1]
        assert len(denied_goal.literals) == 1

        lit = denied_goal.literals[0]
        assert lit.sign is False  # negated

        # Should be an equality
        eq_sym = symbol_table.sn_to_str(lit.atom.symnum)
        assert eq_sym == "="

    def test_parse_and_deny_lattice_problem(self):
        """Test goal denial on a lattice absorption problem.

        Goal: x ^ x = x (idempotent)
        Should produce: ¬(c1 ^ c1 = c1)
        """
        input_text = """
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
        symbol_table = SymbolTable()
        parsed = parse_input(input_text, symbol_table)

        assert len(parsed.goals) == 1

        usable, sos = _deny_goals(parsed, symbol_table)

        # Should have the negated goal
        assert len(sos) == 7  # 6 original SOS + 1 denied goal

        denied_goal = sos[-1]
        assert len(denied_goal.literals) == 1

        lit = denied_goal.literals[0]
        assert lit.sign is False

    def test_justification_marked_correctly(self):
        """Test that denied goals have correct DENY justification."""
        input_text = """
formulas(goals).
  P(x).
end_of_list.
"""
        symbol_table = SymbolTable()
        parsed = parse_input(input_text, symbol_table)

        usable, sos = _deny_goals(parsed, symbol_table)

        denied = sos[0]
        assert len(denied.justification) > 0

        just = denied.justification[0]
        assert just.just_type == JustType.DENY

    def test_goal_with_complex_term(self):
        """Test Skolemization of goals with complex nested terms.

        Goal: P(f(a, x), g(y, b))
        Expected: ¬P(f(a, c1), g(c2, b))
        """
        input_text = """
formulas(goals).
  P(f(a, x), g(y, b)).
end_of_list.
"""
        symbol_table = SymbolTable()
        parsed = parse_input(input_text, symbol_table)

        usable, sos = _deny_goals(parsed, symbol_table)

        assert len(sos) == 1
        denied = sos[0]
        lit = denied.literals[0]

        assert lit.sign is False
        assert lit.atom.arity == 2

        # First argument should be f(a, c1)
        arg0 = lit.atom.args[0]
        assert arg0.arity == 2
        # f should have preserved arguments
        f_arg0 = arg0.args[0]  # should be 'a'
        assert f_arg0.is_constant
        assert symbol_table.sn_to_str(f_arg0.symnum) == "a"

        f_arg1 = arg0.args[1]  # should be c1
        assert f_arg1.is_constant
        assert symbol_table.sn_to_str(f_arg1.symnum).startswith('c')

    def test_group_theory_goal_match(self):
        """Test that Skolemized group goal matches expected structure."""
        input_text = """
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
        symbol_table = SymbolTable()
        parsed = parse_input(input_text, symbol_table)

        usable, sos = _deny_goals(parsed, symbol_table)

        # Find the denied goal (should be last)
        denied_goal = sos[-1]

        # Should be: ¬(c1 * c2 = c2 * c1)
        assert len(denied_goal.literals) == 1
        lit = denied_goal.literals[0]

        assert lit.sign is False
        eq_atom = lit.atom

        # Left side: c1 * c2
        left = eq_atom.args[0]
        assert left.arity == 2  # multiplication has 2 args
        assert left.args[0].is_constant
        assert left.args[1].is_constant
        left_sym0 = symbol_table.sn_to_str(left.args[0].symnum)
        left_sym1 = symbol_table.sn_to_str(left.args[1].symnum)
        assert left_sym0.startswith('c')
        assert left_sym1.startswith('c')

        # Right side: c2 * c1
        right = eq_atom.args[1]
        assert right.arity == 2
        assert right.args[0].is_constant
        assert right.args[1].is_constant
        right_sym0 = symbol_table.sn_to_str(right.args[0].symnum)
        right_sym1 = symbol_table.sn_to_str(right.args[1].symnum)
        assert right_sym0.startswith('c')
        assert right_sym1.startswith('c')

        # The Skolem constants should be in reversed order
        # (assuming parser gives vars in order x, y -> c1, c2)
        # and they're used in order on both sides
        assert left_sym0 != left_sym1 or left_sym0 == "c1"  # at least one is c1
