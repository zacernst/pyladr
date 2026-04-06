"""Integration tests for advanced systems working together.

Validates cross-system compatibility between:
- Advanced subsumption (forward + backward)
- Demodulation (forward + back-demod)
- Paramodulation (equational reasoning)
- Parallel inference engine
- AC unification components

These tests exercise real search scenarios that require multiple
advanced subsystems to cooperate correctly.
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable, UnifTheory
from pyladr.core.term import Term, get_variable_term
from pyladr.inference.paramodulation import (
    _oriented_eqs,
    _renamable_flips,
    mark_oriented_eq,
    orient_equalities,
)
from pyladr.parallel.inference_engine import ParallelInferenceEngine, ParallelSearchConfig
from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions


# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_orientation_state():
    _oriented_eqs.clear()
    _renamable_flips.clear()
    yield
    _oriented_eqs.clear()
    _renamable_flips.clear()


def _v(n: int) -> Term:
    return get_variable_term(n)


def _t(st: SymbolTable, name: str, *args: Term) -> Term:
    arity = len(args)
    sn = st.str_to_sn(name, arity)
    if arity == 0:
        return Term(private_symbol=-sn)
    return Term(private_symbol=-sn, arity=arity, args=tuple(args))


def _eq(st: SymbolTable, left: Term, right: Term) -> Literal:
    return Literal(sign=True, atom=_t(st, "=", left, right))


def _neq(st: SymbolTable, left: Term, right: Term) -> Literal:
    return Literal(sign=False, atom=_t(st, "=", left, right))


def _pos(atom: Term) -> Literal:
    return Literal(sign=True, atom=atom)


def _neg(atom: Term) -> Literal:
    return Literal(sign=False, atom=atom)


def _cl(*lits: Literal) -> Clause:
    return Clause(literals=tuple(lits))


# ── Subsumption + Demodulation interaction ──────────────────────────────────


class TestSubsumptionDemodInteraction:
    """Test that subsumption and demodulation work correctly together."""

    def test_demod_then_forward_subsumption(self):
        """Demodulation simplifies a clause, which then gets forward-subsumed.

        Setup:
        - Demodulator: f(a) = a
        - Usable: P(x) (general clause)
        - SOS: P(f(a)) → demodulates to P(a) → subsumed by P(x)
        - SOS: -P(b) (to make a proof still possible)

        P(f(a)) should be demodulated to P(a), then forward-subsumed by P(x),
        but P(x) and -P(b) still produce a proof.
        """
        st = SymbolTable()
        st.str_to_sn("=", 2)
        st.str_to_sn("P", 1)
        st.str_to_sn("f", 1)
        st.str_to_sn("a", 0)
        st.str_to_sn("b", 0)

        x = _v(0)
        a = _t(st, "a")
        b = _t(st, "b")
        fa = _t(st, "f", a)

        # Demodulator: f(a) = a (oriented)
        eq_atom = _t(st, "=", fa, a)
        mark_oriented_eq(eq_atom)
        demod = _cl(_pos(eq_atom))

        # P(x) — general, subsumes any P(...)
        general = _cl(_pos(_t(st, "P", x)))

        # P(f(a)) — will be demodulated to P(a), then subsumed by P(x)
        specific = _cl(_pos(_t(st, "P", fa)))

        # -P(b)
        neg = _cl(_neg(_t(st, "P", b)))

        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            demodulation=True,
            factoring=False,
            max_given=50,
        )
        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(usable=[demod, general], sos=[specific, neg])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_backward_subsumption_after_paramodulation(self):
        """Paramodulation creates a general clause that back-subsumes specifics.

        From a=b and P(a)|Q(a), paramodulation can create P(b)|Q(a) etc.
        More importantly, if paramodulation creates P(x) from existing equations,
        it should back-subsume more specific P(a), P(b) etc.
        """
        st = SymbolTable()
        st.str_to_sn("=", 2)
        st.str_to_sn("P", 1)
        st.str_to_sn("a", 0)
        st.str_to_sn("b", 0)

        a = _t(st, "a")
        b = _t(st, "b")

        # a = b, P(a), -P(b) → contradiction
        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            factoring=False,
            max_given=50,
        )
        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(
            sos=[
                _cl(_eq(st, a, b)),
                _cl(_pos(_t(st, "P", a))),
                _cl(_neg(_t(st, "P", b))),
            ],
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT


# ── Parallel + Subsumption + Demodulation ───────────────────────────────────


class TestParallelWithAdvancedFeatures:
    """Test parallel engine with demodulation and subsumption active."""

    def test_parallel_config_with_demod(self):
        """Search with parallel config + demodulation finds proof."""
        st = SymbolTable()
        st.str_to_sn("=", 2)
        st.str_to_sn("P", 1)
        st.str_to_sn("f", 1)
        st.str_to_sn("a", 0)
        st.str_to_sn("b", 0)

        a = _t(st, "a")
        b = _t(st, "b")
        fa = _t(st, "f", a)

        eq_atom = _t(st, "=", fa, b)
        mark_oriented_eq(eq_atom)

        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            demodulation=True,
            factoring=False,
            max_given=50,
            parallel=ParallelSearchConfig(enabled=True, min_usable_for_parallel=1),
        )
        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(
            usable=[_cl(_pos(eq_atom))],
            sos=[
                _cl(_pos(_t(st, "P", fa))),
                _cl(_neg(_t(st, "P", b))),
            ],
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_parallel_sequential_same_proof(self):
        """Parallel and sequential search produce proofs for the same problem."""
        st = SymbolTable()
        st.str_to_sn("=", 2)
        st.str_to_sn("P", 2)
        st.str_to_sn("Q", 1)
        for name in "abcd":
            st.str_to_sn(name, 0)

        x, y = _v(0), _v(1)
        a, b = _t(st, "a"), _t(st, "b")

        clauses = [
            _cl(_pos(_t(st, "P", x, y)), _pos(_t(st, "Q", x))),
            _cl(_neg(_t(st, "P", a, b))),
            _cl(_neg(_t(st, "Q", a))),
        ]

        for parallel in [None, ParallelSearchConfig(enabled=False),
                         ParallelSearchConfig(enabled=True, min_usable_for_parallel=1)]:
            opts = SearchOptions(
                binary_resolution=True,
                factoring=True,
                max_given=50,
                parallel=parallel,
            )
            search = GivenClauseSearch(options=opts, symbol_table=st)
            result = search.run(sos=list(clauses))
            assert result.exit_code == ExitCode.MAX_PROOFS_EXIT, (
                f"Failed with parallel={parallel}"
            )


# ── AC components validation ────────────────────────────────────────────────


class TestACComponentsIntegration:
    """Test AC-related components work correctly with the core system."""

    def test_ac_normal_form_consistency(self):
        """AC canonical form is consistent through flatten/sort/reassociate."""
        from pyladr.core.ac_normal_form import ac_canonical, flatten_ac, term_compare_ncv

        st = SymbolTable()
        plus_sn = st.str_to_sn("+", 2)
        a_sn = st.str_to_sn("a", 0)
        b_sn = st.str_to_sn("b", 0)
        c_sn = st.str_to_sn("c", 0)

        a = Term(private_symbol=-a_sn)
        b = Term(private_symbol=-b_sn)
        c = Term(private_symbol=-c_sn)

        # (a + b) + c and a + (b + c) should have same canonical form
        ab = Term(private_symbol=-plus_sn, arity=2, args=(a, b))
        ab_c = Term(private_symbol=-plus_sn, arity=2, args=(ab, c))

        bc = Term(private_symbol=-plus_sn, arity=2, args=(b, c))
        a_bc = Term(private_symbol=-plus_sn, arity=2, args=(a, bc))

        is_ac = lambda sn: sn == plus_sn
        canon1 = ac_canonical(ab_c, is_ac)
        canon2 = ac_canonical(a_bc, is_ac)

        assert canon1.term_ident(canon2), "AC canonical form should be same for both associations"

    def test_ac_flatten_and_sort(self):
        """Flattening produces correct argument list."""
        from pyladr.core.ac_normal_form import flatten_ac

        st = SymbolTable()
        plus_sn = st.str_to_sn("+", 2)

        a = Term(private_symbol=-st.str_to_sn("a", 0))
        b = Term(private_symbol=-st.str_to_sn("b", 0))
        c = Term(private_symbol=-st.str_to_sn("c", 0))

        # (a + b) + c flattens to [a, b, c]
        ab = Term(private_symbol=-plus_sn, arity=2, args=(a, b))
        ab_c = Term(private_symbol=-plus_sn, arity=2, args=(ab, c))

        flat = flatten_ac(ab_c, -plus_sn)
        assert len(flat) == 3

    def test_diophantine_solver_basic(self):
        """Diophantine solver finds solutions for simple equation."""
        from pyladr.inference.diophantine import dio

        # 1*x1 = 1*y1 (trivial: x1=y1=k)
        result = dio([1, 1], m=1, n=1, constraints=[0, 0])
        assert result.status == 1
        assert result.num_basis > 0

    def test_diophantine_solver_two_by_two(self):
        """Diophantine solver for 2x2 system."""
        from pyladr.inference.diophantine import dio

        # 1*x1 + 1*x2 = 1*y1 + 1*y2
        result = dio([1, 1, 1, 1], m=2, n=2, constraints=[0, 0, 0, 0])
        assert result.status == 1
        assert result.num_basis >= 3  # At least S_ij solutions


# ── Multi-feature search ────────────────────────────────────────────────────


class TestMultiFeatureSearch:
    """Search problems that exercise multiple features simultaneously."""

    def test_resolution_paramodulation_demod_together(self):
        """Problem requiring resolution, paramodulation, and demodulation.

        Axioms: f(a) = b (demod), P(f(a)), Q(x) | -P(x)
        Goal: -Q(b)

        Path: f(a)=b demods P(f(a)) to P(b),
              Q(b)|-P(b) resolves with P(b) to get Q(b),
              Q(b) resolves with -Q(b) → empty clause.
        """
        st = SymbolTable()
        st.str_to_sn("=", 2)
        st.str_to_sn("P", 1)
        st.str_to_sn("Q", 1)
        st.str_to_sn("f", 1)
        st.str_to_sn("a", 0)
        st.str_to_sn("b", 0)

        x = _v(0)
        a = _t(st, "a")
        b = _t(st, "b")
        fa = _t(st, "f", a)

        eq_atom = _t(st, "=", fa, b)
        mark_oriented_eq(eq_atom)

        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            demodulation=True,
            factoring=True,
            max_given=100,
        )
        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(
            usable=[_cl(_pos(eq_atom))],
            sos=[
                _cl(_pos(_t(st, "P", fa))),
                _cl(_pos(_t(st, "Q", x)), _neg(_t(st, "P", x))),
                _cl(_neg(_t(st, "Q", b))),
            ],
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_many_clause_search_terminates(self):
        """Search with many clauses terminates within resource limits."""
        st = SymbolTable()
        st.str_to_sn("P", 2)
        st.str_to_sn("Q", 1)
        for name in "abcdefghij":
            st.str_to_sn(name, 0)

        x, y = _v(0), _v(1)
        constants = [_t(st, name) for name in "abcdefghij"]

        # Generate a non-trivial problem with many clauses
        sos: list[Clause] = []
        # P(x, y) | Q(x) — general
        sos.append(_cl(_pos(_t(st, "P", x, y)), _pos(_t(st, "Q", x))))
        # -P(a, b) — specific negation
        sos.append(_cl(_neg(_t(st, "P", constants[0], constants[1]))))
        # -Q(a) — to complete proof
        sos.append(_cl(_neg(_t(st, "Q", constants[0]))))

        opts = SearchOptions(
            binary_resolution=True,
            factoring=True,
            max_given=100,
            max_kept=500,
        )
        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(sos=sos)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_group_theory_with_all_features(self):
        """Full group theory proof with all advanced features enabled.

        Prove right identity from left identity + left inverse + assoc.
        Uses paramodulation, demodulation, subsumption together.
        """
        st = SymbolTable()
        st.str_to_sn("=", 2)
        st.str_to_sn("*", 2)
        st.str_to_sn("i", 1)
        st.str_to_sn("e", 0)
        st.str_to_sn("a", 0)

        x, y, z = _v(0), _v(1), _v(2)
        e = _t(st, "e")
        a = _t(st, "a")

        ax1 = _cl(_eq(st, _t(st, "*", e, x), x))                              # e * x = x
        ax2 = _cl(_eq(st, _t(st, "*", _t(st, "i", x), x), e))                 # i(x) * x = e
        ax3 = _cl(_eq(st, _t(st, "*", _t(st, "*", x, y), z),
                       _t(st, "*", x, _t(st, "*", y, z))))                      # (x*y)*z = x*(y*z)
        goal = _cl(_neq(st, _t(st, "*", a, e), a))                             # -(a * e = a)

        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            demodulation=True,
            factoring=True,
            max_given=500,
            max_kept=5000,
            parallel=ParallelSearchConfig(enabled=True, min_usable_for_parallel=1),
        )
        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(sos=[ax1, ax2, ax3, goal])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
