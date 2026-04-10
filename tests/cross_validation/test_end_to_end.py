"""End-to-end cross-validation tests using the parser → search pipeline.

These tests parse LADR input strings and run the full given-clause search,
verifying that PyLADR finds proofs for problems known to be provable by
C Prover9. This validates the complete pipeline equivalence.
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import LADRParser, parse_input
from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions


def _deny_goals(parsed) -> tuple[list[Clause], list[Clause]]:
    """Negate goals and add to SOS for refutation search."""
    usable = list(parsed.usable)
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
    return usable, sos


def _prove(ladr_input: str, *, max_given: int = 500, max_kept: int = 5000) -> ExitCode:
    """Parse LADR input and run search, returning the exit code."""
    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(ladr_input)

    usable, sos = _deny_goals(parsed)

    opts = SearchOptions(
        binary_resolution=True,
        paramodulation=True,
        demodulation=True,
        factoring=True,
        max_given=max_given,
        max_kept=max_kept,
    )

    search = GivenClauseSearch(options=opts, symbol_table=st)
    result = search.run(usable=usable, sos=sos)
    return result.exit_code


class TestGroupTheoryEndToEnd:
    """Group theory proofs: validated against C Prover9."""

    def test_x_squared_implies_commutativity(self):
        """The classic x2 problem: x*x=e implies commutativity.

        This is the canonical Prover9 test case.
        C Prover9: proves in ~10 given clauses.
        """
        result = _prove("""\
formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
  x * x = e.
end_of_list.

formulas(goals).
  x * y = y * x.
end_of_list.
""")
        assert result == ExitCode.MAX_PROOFS_EXIT

    def test_left_identity_right_identity(self):
        """Prove right identity from left identity + left inverse + assoc."""
        result = _prove("""\
formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
end_of_list.

formulas(goals).
  x * e = x.
end_of_list.
""")
        assert result == ExitCode.MAX_PROOFS_EXIT

    def test_left_inverse_right_inverse(self):
        """Prove right inverse from left identity + left inverse + assoc."""
        result = _prove("""\
formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
end_of_list.

formulas(goals).
  x * x' = e.
end_of_list.
""")
        assert result == ExitCode.MAX_PROOFS_EXIT

    def test_inverse_of_inverse(self):
        """Prove (x')' = x (double inverse cancellation)."""
        result = _prove("""\
formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
end_of_list.

formulas(goals).
  x'' = x.
end_of_list.
""")
        assert result == ExitCode.MAX_PROOFS_EXIT


class TestLatticeTheoryEndToEnd:
    """Lattice theory proofs: validated against C Prover9."""

    def test_meet_idempotence(self):
        """Prove x^x=x from lattice axioms (absorption laws)."""
        result = _prove("""\
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
""")
        assert result == ExitCode.MAX_PROOFS_EXIT

    def test_join_idempotence(self):
        """Prove x v x = x from lattice axioms."""
        result = _prove("""\
formulas(sos).
  x ^ y = y ^ x.
  x v y = y v x.
  (x ^ y) ^ z = x ^ (y ^ z).
  (x v y) v z = x v (y v z).
  x ^ (x v y) = x.
  x v (x ^ y) = x.
end_of_list.

formulas(goals).
  x v x = x.
end_of_list.
""")
        assert result == ExitCode.MAX_PROOFS_EXIT


class TestSimpleEquationalEndToEnd:
    """Simple equational proofs through the parser pipeline."""

    def test_transitivity(self):
        """a=b, b=c => a=c."""
        result = _prove("""\
formulas(sos).
  a = b.
  b = c.
end_of_list.

formulas(goals).
  a = c.
end_of_list.
""")
        assert result == ExitCode.MAX_PROOFS_EXIT

    def test_congruence(self):
        """a=b => f(a) = f(b)."""
        result = _prove("""\
formulas(sos).
  a = b.
end_of_list.

formulas(goals).
  f(a) = f(b).
end_of_list.
""")
        assert result == ExitCode.MAX_PROOFS_EXIT

    def test_simple_rewriting(self):
        """f(a) = b, prove f(f(a)) = f(b)."""
        result = _prove("""\
formulas(sos).
  f(a) = b.
end_of_list.

formulas(goals).
  f(f(a)) = f(b).
end_of_list.
""")
        assert result == ExitCode.MAX_PROOFS_EXIT


class TestResolutionEndToEnd:
    """Pure resolution problems through the pipeline."""

    def test_modus_ponens(self):
        """P, -P | Q => Q."""
        result = _prove("""\
formulas(sos).
  P.
  -P | Q.
end_of_list.

formulas(goals).
  Q.
end_of_list.
""")
        assert result == ExitCode.MAX_PROOFS_EXIT

    def test_chain_resolution(self):
        """P, -P | Q, -Q | R => R."""
        result = _prove("""\
formulas(sos).
  P.
  -P | Q.
  -Q | R.
end_of_list.

formulas(goals).
  R.
end_of_list.
""")
        assert result == ExitCode.MAX_PROOFS_EXIT


class TestEdgeCases:
    """Edge cases for the search pipeline."""

    def test_trivially_provable(self):
        """a=a is trivial. The denied goal a!=a should yield empty clause
        when combined with reflexivity. If not, SOS_EMPTY is acceptable
        since some provers handle reflexivity as a built-in simplification.
        """
        result = _prove("""\
formulas(sos).
  x = x.
end_of_list.

formulas(goals).
  a = a.
end_of_list.
""")
        assert result == ExitCode.MAX_PROOFS_EXIT

    def test_unsatisfiable_direct(self):
        """Contradiction: P and -P should derive empty clause."""
        result = _prove("""\
formulas(sos).
  P.
  -P.
end_of_list.
""")
        assert result == ExitCode.MAX_PROOFS_EXIT

    def test_identity_only(self):
        """Prove e*e=e from e*x=x."""
        result = _prove("""\
formulas(sos).
  e * x = x.
end_of_list.

formulas(goals).
  e * e = e.
end_of_list.
""")
        assert result == ExitCode.MAX_PROOFS_EXIT
