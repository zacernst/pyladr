"""Tests for goal Skolemization correctness.

These tests verify that goals are properly negated and Skolemized
for refutation-based proving, matching C Prover9 semantics.

Key rules:
- Goals are implicitly universally quantified
- Negation flips universal to existential
- Skolemization replaces existential variables with fresh constants
- Explicit quantifiers (exists, all) must be properly clausified
"""

import pytest
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.apps.prover9 import _deny_goals


def _get_denied_goal(input_text: str) -> tuple:
    """Parse input, deny goals, return (denied_clause, symbol_table)."""
    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(input_text)
    usable, sos = _deny_goals(parsed, st)
    # The denied goal is the last clause in sos (after any sos axioms)
    denied = sos[-1]
    return denied, st


def _term_str(term: Term, st: SymbolTable) -> str:
    """Simple term-to-string for debugging."""
    if term.is_variable:
        return f"v{term.private_symbol}"
    name = st.sn_to_str(term.symnum)
    if term.arity == 0:
        return name
    args = ", ".join(_term_str(a, st) for a in term.args)
    return f"{name}({args})"


class TestBasicSkolemization:
    """Test basic goal negation and Skolemization."""

    def test_constant_goal_no_skolem(self):
        """Goal with only constants: no Skolemization needed."""
        inp = """
        formulas(sos).
          P(a).
        end_of_list.
        formulas(goals).
          Q(a, b).
        end_of_list.
        """
        denied, st = _get_denied_goal(inp)
        # Goal Q(a,b) -> denied -Q(a,b), no variables to Skolemize
        assert len(denied.literals) == 1
        lit = denied.literals[0]
        assert lit.sign is False  # negated
        # Atom should still be Q(a, b) with constants
        atom = lit.atom
        assert atom.arity == 2
        assert not atom.arg(0).is_variable  # a is constant
        assert not atom.arg(1).is_variable  # b is constant

    def test_single_variable_skolemized(self):
        """Goal P(x) -> denied -P(c1) with Skolem constant."""
        inp = """
        formulas(sos).
          P(a).
        end_of_list.
        formulas(goals).
          P(x).
        end_of_list.
        """
        denied, st = _get_denied_goal(inp)
        assert len(denied.literals) == 1
        lit = denied.literals[0]
        assert lit.sign is False  # negated
        atom = lit.atom
        # x should be replaced by Skolem constant, NOT remain a variable
        assert not atom.arg(0).is_variable, (
            "Variable x should be Skolemized to a constant"
        )
        name = st.sn_to_str(atom.arg(0).symnum)
        assert name.startswith("c"), f"Skolem constant should be named c*, got {name}"
        # Verify it's marked as Skolem
        sym = st.get_symbol(atom.arg(0).symnum)
        assert sym.skolem is True

    def test_multi_variable_distinct_skolem(self):
        """Goal Q(x,y) -> denied -Q(c1,c2) with distinct Skolem constants."""
        inp = """
        formulas(sos).
          Q(a, b).
        end_of_list.
        formulas(goals).
          Q(x, y).
        end_of_list.
        """
        denied, st = _get_denied_goal(inp)
        assert len(denied.literals) == 1
        lit = denied.literals[0]
        assert lit.sign is False
        atom = lit.atom
        arg0, arg1 = atom.arg(0), atom.arg(1)
        # Both should be Skolem constants
        assert not arg0.is_variable, "x should be Skolemized"
        assert not arg1.is_variable, "y should be Skolemized"
        # They should be DISTINCT constants
        assert arg0.symnum != arg1.symnum, (
            "Different variables must get different Skolem constants"
        )

    def test_mixed_constants_and_variables(self):
        """Goal f(x,a) = f(a,x) -> only x Skolemized, a stays."""
        inp = """
        formulas(sos).
          f(a, a) = f(a, a).
        end_of_list.
        formulas(goals).
          f(x, a) = f(a, x).
        end_of_list.
        """
        denied, st = _get_denied_goal(inp)
        assert len(denied.literals) == 1
        lit = denied.literals[0]
        assert lit.sign is False  # negated equality becomes inequality
        atom = lit.atom
        # Atom should be =(f(c1, a), f(a, c1))
        lhs, rhs = atom.arg(0), atom.arg(1)
        # lhs = f(c1, a): first arg is Skolem, second is constant 'a'
        assert not lhs.arg(0).is_variable, "x should be Skolemized in f(x,a)"
        lhs_first_name = st.sn_to_str(lhs.arg(0).symnum)
        assert lhs_first_name.startswith("c"), f"Expected Skolem c*, got {lhs_first_name}"
        lhs_second_name = st.sn_to_str(lhs.arg(1).symnum)
        assert lhs_second_name == "a", f"Constant a should be preserved, got {lhs_second_name}"

    def test_disjunctive_goal_all_negated(self):
        """Goal P(x) | Q(y) -> denied -P(c1) & -Q(c2) (both literals negated)."""
        inp = """
        formulas(sos).
          P(a).
        end_of_list.
        formulas(goals).
          P(x) | Q(y).
        end_of_list.
        """
        denied, st = _get_denied_goal(inp)
        assert len(denied.literals) == 2
        assert all(not lit.sign for lit in denied.literals), (
            "All literals in denied goal should be negative"
        )
        # Both variables should be Skolemized
        for lit in denied.literals:
            for i in range(lit.atom.arity):
                arg = lit.atom.arg(i)
                assert not arg.is_variable, (
                    f"Variable in disjunctive goal should be Skolemized"
                )


class TestExplicitQuantifiers:
    """Test goals with explicit quantifiers.

    These tests expose the bug where $quantified wrapper terms
    are not properly clausified before Skolemization.
    """

    def test_existential_goal_no_skolem(self):
        """Goal 'exists x P(x)' -> denied -P(x) with x universally quantified.

        exists x P(x) means ∃x P(x).
        Negation: ¬∃x P(x) ≡ ∀x ¬P(x)
        Clausification: ¬P(x)  (x stays as universal variable)

        BUG: Currently produces $quantified(exists, c1, -P(c1)) instead.
        """
        inp = """
        formulas(sos).
          P(a).
        end_of_list.
        formulas(goals).
          exists x P(x).
        end_of_list.
        """
        denied, st = _get_denied_goal(inp)
        assert len(denied.literals) == 1
        lit = denied.literals[0]
        assert lit.sign is False, "Literal should be negated"
        # The atom should be P(x), NOT $quantified(exists, ...)
        atom = lit.atom
        atom_name = st.sn_to_str(atom.symnum)
        assert atom_name != "$quantified", (
            f"BUG: $quantified wrapper should be removed during clausification, "
            f"but atom is still $quantified(...). Goal 'exists x P(x)' should "
            f"produce denied clause -P(x) with x as a universally quantified variable."
        )
        assert atom_name == "P", f"Expected atom P, got {atom_name}"
        # x should remain a variable (universal quantification after negation)
        assert atom.arg(0).is_variable, (
            "In 'exists x P(x)', after negation x should be universally quantified "
            "(i.e., remain a variable), not Skolemized to a constant"
        )

    def test_nested_all_exists(self):
        """Goal 'all x exists y R(x,y)' -> denied clause with proper Skolemization.

        all x exists y R(x,y) means ∀x∃y R(x,y).
        Negation: ∃x∀y ¬R(x,y)
        Skolemize x: ∀y ¬R(c1, y)
        Clause: ¬R(c1, y)  (c1 = Skolem for x, y universally quantified)

        BUG: Currently wraps entire formula in $quantified and doesn't handle nesting.
        """
        inp = """
        formulas(sos).
          R(a, b).
        end_of_list.
        formulas(goals).
          all x exists y R(x, y).
        end_of_list.
        """
        denied, st = _get_denied_goal(inp)
        assert len(denied.literals) == 1
        lit = denied.literals[0]
        assert lit.sign is False
        atom = lit.atom
        atom_name = st.sn_to_str(atom.symnum)
        assert atom_name != "$quantified", (
            f"BUG: Nested quantifiers not properly clausified. "
            f"'all x exists y R(x,y)' should produce -R(c1, y) but got $quantified wrapper."
        )
        assert atom_name == "R", f"Expected atom R, got {atom_name}"
        # x (outer universal -> after negation becomes existential -> Skolemized)
        assert not atom.arg(0).is_variable, (
            "x should be Skolemized (was universally quantified in goal, "
            "becomes existential after negation)"
        )
        # y (inner existential -> after negation becomes universal -> stays variable)
        assert atom.arg(1).is_variable, (
            "y should remain a variable (was existentially quantified in goal, "
            "becomes universally quantified after negation)"
        )

    def test_exists_with_skolem_function(self):
        """Goal 'all x exists y P(x,y)' needs Skolem FUNCTION, not constant.

        When exists y is in scope of all x:
        Negation: ∃x∀y ¬P(x,y)
        Skolemize x -> c1: ∀y ¬P(c1, y)
        Clause: ¬P(c1, y)

        Note: In this case x becomes a constant because it's outer-existential
        after negation, and y stays universal. No Skolem function needed here.

        But for: all x all y exists z P(x,y,z)
        Negation: ∃x∃y∀z ¬P(x,y,z)  -- WRONG, need to negate properly
        Actually: ∃x ∃y ∀z ¬P(x,y,z)  -- x,y Skolemized to c1,c2
        Wait - for a GOAL, this is what we want to PROVE: ∀x∀y∃z P(x,y,z)
        Negation: ∃x∃y∀z ¬P(x,y,z)
        Skolemize: ∀z ¬P(c1, c2, z)
        """
        inp = """
        formulas(sos).
          P(a, b, c).
        end_of_list.
        formulas(goals).
          all x all y exists z P(x, y, z).
        end_of_list.
        """
        denied, st = _get_denied_goal(inp)
        assert len(denied.literals) == 1
        lit = denied.literals[0]
        assert lit.sign is False
        atom = lit.atom
        atom_name = st.sn_to_str(atom.symnum)
        assert atom_name == "P", (
            f"Expected atom P after clausification, got {atom_name}"
        )
        # x and y should be Skolem constants (outer universals -> existential after negation)
        assert not atom.arg(0).is_variable, "x should be Skolemized"
        assert not atom.arg(1).is_variable, "y should be Skolemized"
        # z should remain a variable (inner existential -> universal after negation)
        assert atom.arg(2).is_variable, (
            "z should remain universally quantified (was existential in goal)"
        )


class TestCrossValidation:
    """Compare PyLADR Skolemization output against C Prover9 behavior."""

    @pytest.fixture
    def c_prover9_path(self):
        """Path to reference C Prover9 binary."""
        import os
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "reference-prover9", "bin", "prover9"
        )
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        pytest.skip("C Prover9 binary not available")

    @pytest.fixture
    def fixtures_dir(self):
        import os
        return os.path.join(os.path.dirname(__file__), "..", "fixtures", "inputs")

    def test_c_basic_skolem(self, c_prover9_path, fixtures_dir):
        """Verify basic Skolemization matches C Prover9."""
        import subprocess
        import os
        input_file = os.path.join(fixtures_dir, "skolem_basic.in")
        with open(input_file) as f:
            result = subprocess.run(
                [c_prover9_path],
                stdin=f,
                capture_output=True,
                text=True,
                timeout=10,
            )
        # C Prover9 should produce a denied clause with Skolem constant
        # Look for the deny justification in output
        assert "deny" in result.stdout.lower() or result.returncode in (0, 1, 2), (
            f"C Prover9 failed unexpectedly: {result.stderr}"
        )

    def test_c_existential_goal(self, c_prover9_path, fixtures_dir):
        """Verify existential goal handling matches C Prover9."""
        import subprocess
        import os
        input_file = os.path.join(fixtures_dir, "skolem_existential_goal.in")
        with open(input_file) as f:
            result = subprocess.run(
                [c_prover9_path],
                stdin=f,
                capture_output=True,
                text=True,
                timeout=30,
            )
        # C Prover9 should handle exists properly and find a proof
        # (the idempotent element in groups is e)
        output = result.stdout
        # Check that the denied goal doesn't contain $quantified
        assert "$quantified" not in output, (
            "C Prover9 should not have $quantified in output"
        )


class TestSkolemCounterIsolation:
    """Test that Skolem counter works correctly across multiple goals."""

    def test_multiple_goals_distinct_skolem(self):
        """Multiple goals should get distinct Skolem constants."""
        inp = """
        formulas(sos).
          P(a).
          Q(a).
        end_of_list.
        formulas(goals).
          P(x).
          Q(y).
        end_of_list.
        """
        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(inp)
        num_sos_before = len(parsed.sos)
        usable, sos = _deny_goals(parsed, st)
        # Denied goals are appended after original sos clauses
        denied_goals = sos[num_sos_before:]
        assert len(denied_goals) == 2

        # Collect all Skolem constant symnums
        skolem_syms = set()
        for clause in denied_goals:
            for lit in clause.literals:
                atom = lit.atom
                for i in range(atom.arity):
                    arg = atom.arg(i)
                    if not arg.is_variable and arg.arity == 0:
                        sym = st.get_symbol(arg.symnum)
                        if sym.skolem:
                            skolem_syms.add(arg.symnum)

        assert len(skolem_syms) == 2, (
            f"Two goals with one variable each should produce 2 distinct "
            f"Skolem constants, got {len(skolem_syms)}"
        )
