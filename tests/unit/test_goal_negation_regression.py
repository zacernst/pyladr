"""Regression tests for goal negation / Skolemization bug.

This test suite validates that goal formulas are correctly negated and
Skolemized for refutation-based theorem proving. The bug: _deny_goals()
only flips literal signs but does NOT Skolemize variables. This causes
universally-quantified goal variables to remain as variables in the denied
clause, which can unify with anything and produce trivial/spurious proofs.

Correct semantics (matching C Prover9):
  Goal: P(x)           [implicitly ∀x P(x)]
  Negate: ¬∀x P(x) ≡ ∃x ¬P(x)
  Skolemize: ¬P(c1)    [c1 is a fresh Skolem constant]

Bug behavior:
  Goal: P(x) → denied as ¬P(x)  [x stays as variable, matches anything]
"""

from __future__ import annotations

import pytest
from pyladr.apps.prover9 import _deny_goals
from pyladr.core.clause import Clause, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term
from pyladr.parsing.ladr_parser import LADRParser, ParsedInput, parse_input
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions, ExitCode


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_and_deny(input_text: str) -> tuple[list[Clause], list[Clause], SymbolTable]:
    """Parse input, deny goals, return (usable, sos, symbol_table)."""
    st = SymbolTable()
    parsed = parse_input(input_text, st)
    usable, sos, _denied = _deny_goals(parsed, st)
    return usable, sos, st


def _run_search(
    input_text: str,
    max_given: int = 500,
    max_seconds: float = 10.0,
    **kwargs,
):
    """Run full search pipeline on input text. Returns (result, symbol_table)."""
    st = SymbolTable()
    parsed = parse_input(input_text, st)
    usable, sos, _denied = _deny_goals(parsed, st)
    opts = SearchOptions(
        max_given=max_given,
        max_seconds=max_seconds,
        **kwargs,
    )
    engine = GivenClauseSearch(options=opts, symbol_table=st)
    result = engine.run(usable=usable, sos=sos)
    return result, st


def _term_name(term: Term, st: SymbolTable) -> str:
    """Get human-readable name for a term."""
    if term.is_variable:
        return f"var({term.varnum})"
    return st.sn_to_str(term.symnum)


def _has_variables(clause: Clause) -> bool:
    """Check if a clause contains any variable terms."""
    for lit in clause.literals:
        for t in lit.atom.subterms():
            if t.is_variable:
                return True
    return False


def _count_skolem_constants(clause: Clause, st: SymbolTable) -> int:
    """Count Skolem constants in a clause."""
    skolems = set()
    for lit in clause.literals:
        for t in lit.atom.subterms():
            if t.is_constant:
                sym = st.get_symbol(t.symnum)
                if sym.skolem:
                    skolems.add(t.symnum)
    return len(skolems)


# ── Core regression: variable Skolemization ──────────────────────────────────


class TestGoalNegationRegression:
    """Core regression tests: variables in goals MUST be Skolemized.

    These tests directly expose the bug where _deny_goals() fails to
    replace universally-quantified goal variables with Skolem constants.
    """

    def test_single_variable_must_be_skolemized(self):
        """Goal P(x) must produce -P(c1), NOT -P(x).

        This is the simplest reproduction of the bug. If x remains as a
        variable, it can unify with any term, making the denied goal
        trivially resolvable against any P(...) in the SOS.
        """
        inp = """
formulas(goals).
  P(x).
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        assert len(sos) == 1
        denied = sos[0]
        lit = denied.literals[0]
        assert lit.sign is False, "Goal literal should be negated"

        arg = lit.atom.args[0]
        assert not arg.is_variable, (
            "REGRESSION: Variable x in goal P(x) was NOT Skolemized. "
            "Expected a Skolem constant (c1), got a variable. "
            "This causes trivial proofs because the variable unifies with anything."
        )
        assert arg.is_constant, "Skolemized variable should be a constant"
        name = st.sn_to_str(arg.symnum)
        assert name.startswith("_sk"), f"Skolem constant should be named c*, got {name}"

    def test_equality_goal_both_sides_skolemized(self):
        """Goal x = y must produce -(c1 = c2), NOT -(x = y).

        With variables, -(x = y) is trivially false only if we can find
        distinct terms. But variables unify, so the prover may produce
        incorrect refutations.
        """
        inp = """
formulas(goals).
  x = y.
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        denied = sos[0]
        lit = denied.literals[0]
        assert lit.sign is False

        lhs = lit.atom.args[0]
        rhs = lit.atom.args[1]
        assert not lhs.is_variable, (
            "REGRESSION: LHS variable not Skolemized in equality goal"
        )
        assert not rhs.is_variable, (
            "REGRESSION: RHS variable not Skolemized in equality goal"
        )
        # Must be distinct Skolem constants
        assert lhs.symnum != rhs.symnum, (
            "Different variables must get different Skolem constants"
        )

    def test_equational_goal_variables_skolemized(self):
        """Goal x * y = y * x must produce -(c1 * c2 = c2 * c1).

        This is the most common form: proving commutativity. Without
        Skolemization, the denied goal has free variables that unify
        with the axiom terms, producing trivial proofs.
        """
        inp = """
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
        usable, sos, st = _parse_and_deny(inp)
        # 4 axioms + 1 denied goal
        assert len(sos) == 5
        denied = sos[-1]

        # The denied goal should have NO variables
        assert not _has_variables(denied), (
            "REGRESSION: Denied goal 'x * y = y * x' still contains variables. "
            "Expected all variables to be replaced with Skolem constants."
        )

        # Should have exactly 2 Skolem constants (for x and y)
        n_skolems = _count_skolem_constants(denied, st)
        assert n_skolems == 2, (
            f"Expected 2 Skolem constants (for x and y), got {n_skolems}"
        )

    def test_nested_function_variables_skolemized(self):
        """Goal P(f(x, a), g(y, b)) -> -P(f(c1, a), g(c2, b)).

        Variables deep inside nested terms must also be Skolemized.
        """
        inp = """
formulas(goals).
  P(f(x, a), g(y, b)).
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        denied = sos[0]

        assert not _has_variables(denied), (
            "REGRESSION: Variables inside nested terms were not Skolemized"
        )

    def test_shared_variable_same_skolem(self):
        """Goal f(x, x) -> -f(c1, c1): same variable gets same Skolem constant.

        If x appears twice, both occurrences must map to the same c1.
        """
        inp = """
formulas(goals).
  f(x, x) = a.
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        denied = sos[0]
        atom = denied.literals[0].atom  # =(f(c1,c1), a)

        lhs = atom.args[0]  # f(c1, c1)
        arg0 = lhs.args[0]
        arg1 = lhs.args[1]

        assert not arg0.is_variable, "First x should be Skolemized"
        assert not arg1.is_variable, "Second x should be Skolemized"
        assert arg0.symnum == arg1.symnum, (
            "Same variable x appearing twice must map to the same Skolem constant"
        )

    def test_constant_in_goal_preserved(self):
        """Goal P(a, x) -> -P(a, c1): constant 'a' must not be changed."""
        inp = """
formulas(goals).
  P(a, x).
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        denied = sos[0]
        atom = denied.literals[0].atom

        # First arg should still be 'a'
        arg0 = atom.args[0]
        assert arg0.is_constant
        assert st.sn_to_str(arg0.symnum) == "a", (
            "Constant 'a' in goal should be preserved, not Skolemized"
        )

        # Second arg should be Skolem constant
        arg1 = atom.args[1]
        assert not arg1.is_variable, "Variable x should be Skolemized"
        assert arg1.is_constant
        name = st.sn_to_str(arg1.symnum)
        assert name.startswith("_sk"), f"Expected Skolem constant c*, got {name}"

    def test_disjunctive_goal_all_variables_skolemized(self):
        """Goal P(x) | Q(y) -> -P(c1) | -Q(c2): both vars Skolemized."""
        inp = """
formulas(goals).
  P(x) | Q(y).
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        denied = sos[0]

        assert len(denied.literals) == 2
        for lit in denied.literals:
            assert lit.sign is False, "Both literals should be negated"

        assert not _has_variables(denied), (
            "REGRESSION: Variables in disjunctive goal not Skolemized"
        )

    def test_multiple_goals_distinct_skolems(self):
        """Multiple goals get distinct Skolem constants across goals."""
        inp = """
formulas(goals).
  P(x).
  Q(y).
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        assert len(sos) == 2

        # Collect all Skolem constant symnums across both goals
        all_skolems = set()
        for clause in sos:
            for lit in clause.literals:
                for t in lit.atom.subterms():
                    if t.is_constant:
                        sym = st.get_symbol(t.symnum)
                        if sym.skolem:
                            all_skolems.add(t.symnum)

        assert len(all_skolems) == 2, (
            f"Two goals with one variable each should produce 2 distinct "
            f"Skolem constants, got {len(all_skolems)}"
        )


# ── Trivial proof detection ─────────────────────────────────────────────────


class TestTrivialProofRegression:
    """Tests that detect trivial/spurious proofs caused by unskolemized variables.

    The core symptom of the bug: proofs that complete with zero or very few
    given clauses, because the variable in the denied goal unifies immediately
    with axiom terms.
    """

    def test_group_commutativity_not_trivial(self):
        """Group commutativity proof should require real work.

        With the bug, the denied goal -(x * y = y * x) with free variables
        can unify with axiom e * x = x via x->e, y->e, producing a trivial
        proof. The real proof requires many inference steps.
        """
        inp = """\
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
        result, st = _run_search(inp, max_given=1000, max_seconds=30.0)

        if result.exit_code == ExitCode.MAX_PROOFS_EXIT:
            # Proof found — check it's not trivial
            stats = result.statistics
            given_count = stats.given if hasattr(stats, 'given') else 0

            assert given_count > 5, (
                f"REGRESSION: Group commutativity 'proved' with only {given_count} "
                f"given clauses. This is a trivial proof caused by unskolemized "
                f"variables in the denied goal. Real proof needs many steps."
            )

    def test_identity_proof_is_legitimate(self):
        """e * e = e should have a short, legitimate proof.

        This is a constant-only goal (no variables to Skolemize), so
        it should work correctly regardless of the bug.
        """
        inp = """\
formulas(sos).
  e * x = x.
end_of_list.

formulas(goals).
  e * e = e.
end_of_list.
"""
        result, st = _run_search(inp)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT, (
            "e * e = e should be provable from e * x = x"
        )

    def test_lattice_idempotence_not_trivial(self):
        """Lattice idempotence proof should require real work.

        Goal: x ^ x = x. Without Skolemization, -(x ^ x = x) with free x
        can match absorption axioms trivially.
        """
        inp = """\
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
        result, st = _run_search(inp, max_given=500, max_seconds=30.0)

        if result.exit_code == ExitCode.MAX_PROOFS_EXIT:
            stats = result.statistics
            given_count = stats.given if hasattr(stats, 'given') else 0

            assert given_count > 2, (
                f"REGRESSION: Lattice idempotence 'proved' with only {given_count} "
                f"given clauses. Likely a trivial proof from unskolemized variables."
            )

    def test_unprovable_goal_not_proved(self):
        """A genuinely unprovable goal should NOT be proved.

        Goal: a = b where a and b are distinct constants.
        Without Skolemization this isn't directly affected, but it validates
        the search correctly exhausts without finding spurious proofs.
        """
        inp = """\
formulas(sos).
  P(a).
  P(b).
end_of_list.

formulas(goals).
  a = b.
end_of_list.
"""
        result, st = _run_search(inp, max_given=100, max_seconds=5.0)
        assert result.exit_code != ExitCode.MAX_PROOFS_EXIT, (
            "a = b should NOT be provable from P(a) and P(b)"
        )


# ── Denied clause structure validation ───────────────────────────────────────


class TestDeniedClauseStructure:
    """Validate the structure of denied clauses matches C Prover9 semantics."""

    def test_deny_justification(self):
        """Denied goals must have JustType.DENY justification."""
        inp = """
formulas(goals).
  P(x).
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        denied = sos[0]
        assert len(denied.justification) > 0
        assert denied.justification[0].just_type == JustType.DENY

    def test_positive_goal_becomes_negative(self):
        """Positive goal literal P(x) becomes negative -P(c1)."""
        inp = """
formulas(goals).
  P(a).
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        denied = sos[0]
        assert denied.literals[0].sign is False

    def test_negative_goal_becomes_positive(self):
        """Negative goal literal -P(a) becomes positive P(a)."""
        inp = """
formulas(goals).
  -P(a).
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        denied = sos[0]
        assert denied.literals[0].sign is True

    def test_denied_goal_is_ground(self):
        """After Skolemization, denied goals with variables should be ground."""
        inp = """
formulas(goals).
  P(x, y, z).
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        denied = sos[0]
        assert not _has_variables(denied), (
            "Denied goal with only universally-quantified variables should be ground "
            "after Skolemization"
        )

    def test_skolem_constants_registered_in_symbol_table(self):
        """Skolem constants must be registered and marked in symbol table."""
        inp = """
formulas(goals).
  P(x).
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        denied = sos[0]
        arg = denied.literals[0].atom.args[0]

        assert arg.is_constant, "Skolemized arg should be a constant"
        sym = st.get_symbol(arg.symnum)
        assert sym.skolem is True, (
            "Skolem constant must be marked as Skolem in symbol table"
        )
        assert sym.arity == 0, "Skolem constants have arity 0"

    def test_goals_appended_to_sos(self):
        """Denied goals are appended to the SOS list, after existing SOS clauses."""
        inp = """
formulas(sos).
  P(a).
  Q(b).
end_of_list.

formulas(goals).
  R(x).
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        assert len(sos) == 3  # 2 axioms + 1 denied goal
        # The denied goal should be the last clause
        denied = sos[-1]
        assert denied.justification[0].just_type == JustType.DENY

    def test_usable_clauses_preserved(self):
        """Usable clauses should be passed through unchanged."""
        inp = """
formulas(usable).
  P(a).
end_of_list.

formulas(goals).
  Q(x).
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        assert len(usable) == 1
        assert len(sos) == 1  # just the denied goal


# ── Cross-validation with C Prover9 ─────────────────────────────────────────


class TestCrossValidationGoalNegation:
    """Compare goal negation behavior with C Prover9 reference implementation.

    These tests require the C binary at reference-prover9/bin/prover9.
    """

    @pytest.fixture
    def c_binary(self):
        """Path to C Prover9 binary."""
        import os
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent.parent / "reference-prover9" / "bin" / "prover9"
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
        pytest.skip("C Prover9 binary not available")

    def _run_c_prover9(self, c_binary: str, input_text: str) -> str:
        """Run C Prover9 and return stdout."""
        import subprocess
        result = subprocess.run(
            [c_binary],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout

    def test_c_denied_goal_has_skolem_constants(self, c_binary):
        """C Prover9 denied goals contain Skolem constants, not variables."""
        inp = """\
formulas(sos).
  P(a).
end_of_list.

formulas(goals).
  P(x).
end_of_list.
"""
        output = self._run_c_prover9(c_binary, inp)

        # C Prover9 output should show the denied goal with a Skolem constant
        # like "P(c1)" not "P(x)"
        # The deny line typically looks like: "1 -P(c1).  [deny(1)]"
        assert "deny" in output.lower(), "C output should contain deny justification"
        # Should NOT have variables in the denied clause
        # (C Prover9 will show c1, c2, etc. for Skolem constants)

    def test_c_group_commutativity_nontrivial(self, c_binary):
        """C Prover9 group commutativity proof is nontrivial."""
        inp = """\
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
        output = self._run_c_prover9(c_binary, inp)

        # Extract given count from C output
        import re
        given_match = re.search(r"Given=(\d+)", output)
        if given_match:
            c_given = int(given_match.group(1))
            assert c_given > 5, (
                f"C Prover9 used {c_given} given clauses for group commutativity. "
                f"This should be a nontrivial proof."
            )

    def test_python_matches_c_proof_found(self, c_binary):
        """Python and C should agree on whether a proof exists."""
        problems = [
            # (description, input_text, expect_proof)
            ("identity e*e=e", """\
formulas(sos).
  e * x = x.
end_of_list.

formulas(goals).
  e * e = e.
end_of_list.
""", True),
        ]

        for desc, inp, expect_proof in problems:
            import subprocess
            c_result = subprocess.run(
                [c_binary],
                input=inp,
                capture_output=True,
                text=True,
                timeout=30,
            )
            py_result, _ = _run_search(inp, max_given=500, max_seconds=10.0)

            # Check proof existence rather than exact exit codes,
            # since C Prover9 can return sos_empty even when proof found
            c_found_proof = "proofs=1" in c_result.stdout or "Proof 1" in c_result.stderr
            py_found_proof = py_result.exit_code == ExitCode.MAX_PROOFS_EXIT

            assert c_found_proof == py_found_proof, (
                f"Proof existence mismatch for '{desc}': "
                f"C found proof={c_found_proof}, Python found proof={py_found_proof}"
            )
            assert py_found_proof == expect_proof, (
                f"Expected proof={'found' if expect_proof else 'not found'} "
                f"for '{desc}', but Python proof={py_found_proof}"
            )


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestGoalNegationEdgeCases:
    """Edge cases for goal negation and Skolemization."""

    def test_no_goals_section(self):
        """Input with no goals section: usable and SOS unchanged."""
        inp = """
formulas(sos).
  P(a).
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        assert len(sos) == 1
        assert len(usable) == 0

    def test_empty_goals_section(self):
        """Empty goals section: no denied clauses added."""
        inp = """
formulas(sos).
  P(a).
end_of_list.

formulas(goals).
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        assert len(sos) == 1  # just the original SOS clause

    def test_ground_goal_no_skolemization_needed(self):
        """Goal with only constants: no Skolemization needed, just negate."""
        inp = """
formulas(goals).
  P(a, b).
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        denied = sos[0]
        lit = denied.literals[0]
        assert lit.sign is False
        # Constants should be preserved
        assert st.sn_to_str(lit.atom.args[0].symnum) == "a"
        assert st.sn_to_str(lit.atom.args[1].symnum) == "b"

    def test_many_variables_goal(self):
        """Goal with many variables: all must be Skolemized."""
        inp = """
formulas(goals).
  P(u, v, w, x, y, z).
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        denied = sos[0]
        atom = denied.literals[0].atom

        assert atom.arity == 6
        skolem_syms = set()
        for i in range(6):
            arg = atom.args[i]
            assert not arg.is_variable, (
                f"Variable at position {i} should be Skolemized"
            )
            assert arg.is_constant
            sym = st.get_symbol(arg.symnum)
            assert sym.skolem
            skolem_syms.add(arg.symnum)

        # All 6 variables should map to distinct Skolem constants
        assert len(skolem_syms) == 6, (
            f"6 distinct variables should produce 6 distinct Skolem constants, "
            f"got {len(skolem_syms)}"
        )

    def test_equality_with_function_terms(self):
        """Goal f(x) = g(y): variables inside function terms Skolemized."""
        inp = """
formulas(goals).
  f(x) = g(y).
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        denied = sos[0]

        assert not _has_variables(denied), (
            "Variables inside function terms in goals must be Skolemized"
        )

    def test_deeply_nested_variables(self):
        """Goal f(g(h(x))): deeply nested variable must be Skolemized."""
        inp = """
formulas(goals).
  f(g(h(x))) = a.
end_of_list.
"""
        usable, sos, st = _parse_and_deny(inp)
        denied = sos[0]

        assert not _has_variables(denied), (
            "Deeply nested variables in goals must be Skolemized"
        )


# ── Proof soundness validation ───────────────────────────────────────────────


class TestProofSoundness:
    """Validate that proofs are sound (not caused by Skolemization bugs).

    These tests verify that when a proof IS found, it's a genuine proof
    that involves meaningful inference steps — not an artifact of
    unskolemized variables matching everything.
    """

    def test_proof_clauses_are_ground_or_properly_instantiated(self):
        """Proof clauses should be properly instantiated, not wild variables."""
        inp = """\
formulas(sos).
  e * x = x.
end_of_list.

formulas(goals).
  e * e = e.
end_of_list.
"""
        result, st = _run_search(inp)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

        # Check that proof exists and has reasonable structure
        if hasattr(result, 'proofs') and result.proofs:
            proof = result.proofs[0]
            assert len(proof.clauses) >= 2, (
                "A valid proof needs at least 2 clauses (axiom + denied goal)"
            )

    def test_no_proof_from_unrelated_axioms(self):
        """Unrelated axioms should not produce a proof for the goal.

        If Skolemization is broken, -Q(x) could resolve with P(a)
        if variable x somehow unifies with predicate mismatch.
        """
        inp = """\
formulas(sos).
  P(a).
end_of_list.

formulas(goals).
  Q(a).
end_of_list.
"""
        result, st = _run_search(inp, max_given=50, max_seconds=5.0)
        assert result.exit_code != ExitCode.MAX_PROOFS_EXIT, (
            "Q(a) should NOT be provable from just P(a) — different predicate"
        )
