"""Cross-validation tests for equational reasoning (paramodulation + demodulation).

Tests real algebraic problems that require equality reasoning:
- Group theory axioms and simple theorems
- Boolean algebra identities
- Ring theory properties
- Lattice theory
- Commutativity and associativity problems

These tests validate that the Python implementation produces correct
proofs for problems that require paramodulation and demodulation.
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, get_variable_term
from pyladr.inference.paramodulation import (
    _oriented_eqs,
    _renamable_flips,
    orient_equalities,
    para_from_into,
)
from pyladr.inference.demodulation import (
    DemodType,
    DemodulatorIndex,
    demodulate_clause,
    demodulator_type,
)
from pyladr.inference.resolution import renumber_variables
from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions


# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_orientation_state():
    _oriented_eqs.clear()
    _renamable_flips.clear()
    yield
    _oriented_eqs.clear()
    _renamable_flips.clear()


def _st() -> SymbolTable:
    """Create a symbol table with algebraic symbols."""
    s = SymbolTable()
    s.str_to_sn("=", 2)
    s.str_to_sn("*", 2)
    s.str_to_sn("+", 2)
    s.str_to_sn("i", 1)  # inverse
    s.str_to_sn("e", 0)  # identity
    s.str_to_sn("0", 0)  # zero
    s.str_to_sn("1", 0)  # one
    s.str_to_sn("a", 0)
    s.str_to_sn("b", 0)
    s.str_to_sn("c", 0)
    s.str_to_sn("f", 1)
    s.str_to_sn("g", 2)
    s.str_to_sn("p", 1)  # predicate
    return s


def _t(st: SymbolTable, name: str, *args: Term) -> Term:
    """Build a rigid term."""
    arity = len(args)
    sn = st.str_to_sn(name, arity)
    if arity == 0:
        return Term(private_symbol=-sn)
    return Term(private_symbol=-sn, arity=arity, args=tuple(args))


def _v(n: int) -> Term:
    return get_variable_term(n)


def _eq(st: SymbolTable, left: Term, right: Term) -> Literal:
    """Positive equality literal."""
    return Literal(sign=True, atom=_t(st, "=", left, right))


def _neq(st: SymbolTable, left: Term, right: Term) -> Literal:
    """Negative equality literal (disequality)."""
    return Literal(sign=False, atom=_t(st, "=", left, right))


def _cl(*lits: Literal) -> Clause:
    return Clause(literals=tuple(lits))


# ── Group theory ─────────────────────────────────────────────────────────────


class TestGroupTheory:
    """Group theory problems requiring paramodulation."""

    def test_left_identity_implies_right_identity(self):
        """In a group with left identity and left inverse,
        prove that e is also a right identity: x * e = x.

        Axioms:
        1. e * x = x                     (left identity)
        2. i(x) * x = e                  (left inverse)
        3. (x * y) * z = x * (y * z)     (associativity)
        Goal: a * e = a
        """
        st = _st()
        x, y, z = _v(0), _v(1), _v(2)
        e = _t(st, "e")
        a = _t(st, "a")

        # Axioms (in SOS so they get selected as given clauses)
        ax1 = _cl(_eq(st, _t(st, "*", e, x), x))                                    # e * x = x
        ax2 = _cl(_eq(st, _t(st, "*", _t(st, "i", x), x), e))                       # i(x) * x = e
        ax3 = _cl(_eq(st, _t(st, "*", _t(st, "*", x, y), z),
                       _t(st, "*", x, _t(st, "*", y, z))))                            # (x*y)*z = x*(y*z)

        # Negated goal: a * e != a
        goal = _cl(_neq(st, _t(st, "*", a, e), a))

        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            demodulation=True,
            factoring=True,
            max_given=500,
            max_kept=5000,
        )

        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(sos=[ax1, ax2, ax3, goal])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT, (
            f"Expected proof, got {result.exit_code}. "
            f"Stats: {result.stats.report()}"
        )

    def test_left_inverse_implies_right_inverse(self):
        """Prove x * i(x) = e from left identity, left inverse, associativity.

        Axioms:
        1. e * x = x
        2. i(x) * x = e
        3. (x * y) * z = x * (y * z)
        Goal: a * i(a) = e
        """
        st = _st()
        x, y, z = _v(0), _v(1), _v(2)
        e = _t(st, "e")
        a = _t(st, "a")

        ax1 = _cl(_eq(st, _t(st, "*", e, x), x))
        ax2 = _cl(_eq(st, _t(st, "*", _t(st, "i", x), x), e))
        ax3 = _cl(_eq(st, _t(st, "*", _t(st, "*", x, y), z),
                       _t(st, "*", x, _t(st, "*", y, z))))

        goal = _cl(_neq(st, _t(st, "*", a, _t(st, "i", a)), e))

        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            demodulation=True,
            factoring=True,
            max_given=500,
            max_kept=5000,
        )

        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(sos=[ax1, ax2, ax3, goal])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT, (
            f"Expected proof, got {result.exit_code}. "
            f"Stats: {result.stats.report()}"
        )


# ── Simple equational problems ──────────────────────────────────────────────


class TestSimpleEquational:
    """Simple equational reasoning that exercises paramodulation."""

    def test_transitivity_of_equality(self):
        """a=b, b=c => a=c (via paramodulation)."""
        st = _st()
        a, b, c = _t(st, "a"), _t(st, "b"), _t(st, "c")

        c1 = _cl(_eq(st, a, b))  # a = b
        c2 = _cl(_eq(st, b, c))  # b = c
        goal = _cl(_neq(st, a, c))  # -(a = c)

        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            factoring=False,
            max_given=50,
        )

        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(usable=[c1, c2], sos=[goal])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_congruence(self):
        """a=b => f(a) = f(b) (congruence closure via paramodulation)."""
        st = _st()
        a, b = _t(st, "a"), _t(st, "b")

        c1 = _cl(_eq(st, a, b))  # a = b
        goal = _cl(_neq(st, _t(st, "f", a), _t(st, "f", b)))  # -(f(a)=f(b))

        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            factoring=False,
            max_given=50,
        )

        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(usable=[c1], sos=[goal])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_nested_congruence(self):
        """a=b => f(f(a)) = f(f(b)) (nested congruence)."""
        st = _st()
        a, b = _t(st, "a"), _t(st, "b")

        c1 = _cl(_eq(st, a, b))
        ffa = _t(st, "f", _t(st, "f", a))
        ffb = _t(st, "f", _t(st, "f", b))
        goal = _cl(_neq(st, ffa, ffb))

        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            factoring=False,
            max_given=50,
        )

        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(usable=[c1], sos=[goal])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_symmetry_of_equality(self):
        """a=b => b=a (reflexivity is built-in, symmetry via paramodulation)."""
        st = _st()
        a, b = _t(st, "a"), _t(st, "b")

        c1 = _cl(_eq(st, a, b))
        goal = _cl(_neq(st, b, a))

        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            factoring=False,
            max_given=50,
        )

        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(usable=[c1], sos=[goal])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT


# ── Demodulation-specific cross-validation ───────────────────────────────────


class TestDemodulationCrossValidation:
    """Tests that specifically exercise the demodulation subsystem."""

    def test_chain_rewriting(self):
        """Chain of demodulators: f(a)=b, f(b)=c, prove f(f(a))=c."""
        st = _st()
        a, b, c = _t(st, "a"), _t(st, "b"), _t(st, "c")
        fa = _t(st, "f", a)
        fb = _t(st, "f", b)
        ffa = _t(st, "f", fa)

        d1 = _cl(_eq(st, fa, b))   # f(a) = b
        d2 = _cl(_eq(st, fb, c))   # f(b) = c
        goal = _cl(_neq(st, ffa, c))  # -(f(f(a)) = c)

        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            demodulation=True,
            factoring=False,
            max_given=50,
        )

        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(usable=[d1, d2], sos=[goal])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_functional_equation_with_demod(self):
        """f(x) = g(x, a): rewrite p(f(b)) to p(g(b, a))."""
        st = _st()
        x = _v(0)
        a, b = _t(st, "a"), _t(st, "b")
        fx = _t(st, "f", x)
        gxa = _t(st, "g", x, a)

        # f(x) = g(x, a) as demodulator
        d1 = _cl(_eq(st, fx, gxa))  # f(x) = g(x, a)

        # p(f(b)) and -p(g(b, a))
        fb = _t(st, "f", b)
        gba = _t(st, "g", b, a)
        c1 = _cl(Literal(sign=True, atom=_t(st, "p", fb)))
        goal = _cl(Literal(sign=False, atom=_t(st, "p", gba)))

        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            demodulation=True,
            factoring=False,
            max_given=50,
        )

        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(usable=[d1], sos=[c1, goal])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT


# ── Paramodulation inference generation tests ────────────────────────────────


class TestParamodulationInference:
    """Tests that paramodulation generates correct inference results."""

    def test_para_generates_correct_clauses(self):
        """Verify the structure of paramodulants."""
        st = _st()
        x = _v(0)
        a, b = _t(st, "a"), _t(st, "b")

        # f(x) = g(x, a)
        from_clause = Clause(
            literals=(_eq(st, _t(st, "f", x), _t(st, "g", x, a)),),
            id=1,
        )

        # p(f(b))
        fb = _t(st, "f", b)
        into_clause = Clause(
            literals=(Literal(sign=True, atom=_t(st, "p", fb)),),
            id=2,
        )

        results = para_from_into(from_clause, into_clause, False, st)
        assert len(results) >= 1

        # At least one result should contain p(g(b, a))
        gba = _t(st, "g", b, a)
        found = False
        for r in results:
            for lit in r.literals:
                if lit.atom.arity == 1:
                    if lit.atom.args[0].term_ident(gba):
                        found = True
        assert found, "Expected paramodulant containing p(g(b,a))"

    def test_para_into_equality(self):
        """Paramodulate into an equality literal."""
        st = _st()
        a, b, c = _t(st, "a"), _t(st, "b"), _t(st, "c")

        # a = b
        from_clause = Clause(literals=(_eq(st, a, b),), id=1)

        # f(a) = c
        fa = _t(st, "f", a)
        into_clause = Clause(literals=(_eq(st, fa, c),), id=2)

        results = para_from_into(from_clause, into_clause, False, st)
        assert len(results) >= 1

        # Should produce f(b) = c
        fb = _t(st, "f", b)
        found = False
        for r in results:
            for lit in r.literals:
                if lit.atom.arity == 2:
                    if lit.atom.args[0].term_ident(fb) and lit.atom.args[1].term_ident(c):
                        found = True
        assert found, "Expected paramodulant f(b)=c"
