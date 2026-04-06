"""Integration tests for end-to-end proof workflows.

Tests complete proof pipelines using the GivenClauseSearch engine
with resolution, paramodulation, demodulation, and subsumption.
Validates that the Python implementation can solve classical
theorem proving problems from group theory, Boolean algebra, and logic.
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import (
    Term,
    build_binary_term,
    build_unary_term,
    get_rigid_term,
    get_variable_term,
)
from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions


# ── Helpers ──────────────────────────────────────────────────────────────────


def _run_search(
    usable: list[Clause],
    sos: list[Clause],
    *,
    symbol_table: SymbolTable | None = None,
    paramodulation: bool = False,
    demodulation: bool = False,
    back_demod: bool = False,
    max_given: int = 200,
):
    """Run a search and return the result."""
    opts = SearchOptions(
        binary_resolution=True,
        paramodulation=paramodulation,
        demodulation=demodulation,
        back_demod=back_demod,
        factoring=True,
        max_given=max_given,
        quiet=True,
    )
    search = GivenClauseSearch(options=opts, symbol_table=symbol_table)
    return search.run(usable=usable, sos=sos)


def _pos(atom: Term) -> Literal:
    return Literal(sign=True, atom=atom)


def _neg(atom: Term) -> Literal:
    return Literal(sign=False, atom=atom)


def _cl(*lits: Literal) -> Clause:
    return Clause(literals=tuple(lits))


# ── Simple Resolution Proofs ─────────────────────────────────────────────────


class TestSimpleProofs:
    """Test basic resolution proofs that don't require equality."""

    def test_modus_ponens_proof(self) -> None:
        """P(a), ~P(x)|Q(x), ~Q(a) => empty clause."""
        P_sn, Q_sn, a_sn = 1, 2, 3
        a = get_rigid_term(a_sn, 0)
        x = get_variable_term(0)

        c1 = _cl(_pos(get_rigid_term(P_sn, 1, (a,))))
        c2 = _cl(_neg(get_rigid_term(P_sn, 1, (x,))), _pos(get_rigid_term(Q_sn, 1, (x,))))
        c3 = _cl(_neg(get_rigid_term(Q_sn, 1, (a,))))

        result = _run_search(usable=[], sos=[c1, c2, c3])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1

    def test_chain_resolution(self) -> None:
        """P(a), ~P(x)|Q(x), ~Q(x)|R(x), ~R(a) => empty clause."""
        P_sn, Q_sn, R_sn, a_sn = 1, 2, 3, 4
        a = get_rigid_term(a_sn, 0)
        x = get_variable_term(0)

        c1 = _cl(_pos(get_rigid_term(P_sn, 1, (a,))))
        c2 = _cl(_neg(get_rigid_term(P_sn, 1, (x,))), _pos(get_rigid_term(Q_sn, 1, (x,))))
        c3 = _cl(_neg(get_rigid_term(Q_sn, 1, (x,))), _pos(get_rigid_term(R_sn, 1, (x,))))
        c4 = _cl(_neg(get_rigid_term(R_sn, 1, (a,))))

        result = _run_search(usable=[], sos=[c1, c2, c3, c4])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_factoring_proof(self) -> None:
        """P(x)|P(y), ~P(a) => empty clause via factoring + resolution."""
        P_sn, a_sn = 1, 2
        a = get_rigid_term(a_sn, 0)
        x = get_variable_term(0)
        y = get_variable_term(1)

        c1 = _cl(_pos(get_rigid_term(P_sn, 1, (x,))), _pos(get_rigid_term(P_sn, 1, (y,))))
        c2 = _cl(_neg(get_rigid_term(P_sn, 1, (a,))))

        result = _run_search(usable=[], sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_unsatisfiable_detected(self) -> None:
        """P(a), ~P(a) => immediate contradiction."""
        P_sn, a_sn = 1, 2
        a = get_rigid_term(a_sn, 0)
        P_a = get_rigid_term(P_sn, 1, (a,))

        c1 = _cl(_pos(P_a))
        c2 = _cl(_neg(P_a))

        result = _run_search(usable=[], sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1

    def test_no_proof_saturates(self) -> None:
        """P(a), Q(b) => SOS empty (no contradiction)."""
        P_sn, Q_sn, a_sn, b_sn = 1, 2, 3, 4
        a = get_rigid_term(a_sn, 0)
        b = get_rigid_term(b_sn, 0)

        c1 = _cl(_pos(get_rigid_term(P_sn, 1, (a,))))
        c2 = _cl(_pos(get_rigid_term(Q_sn, 1, (b,))))

        result = _run_search(usable=[], sos=[c1, c2], max_given=50)
        assert result.exit_code == ExitCode.SOS_EMPTY_EXIT


# ── Equational Proofs (Paramodulation) ──────────────────────────────────────


class TestEquationalProofs:
    """Test equational proofs requiring paramodulation."""

    def test_simple_equality_substitution(self) -> None:
        """a=b, p(a), ~p(b) => contradiction via paramodulation."""
        st = SymbolTable()
        eq_sn = st.str_to_sn("=", 2)
        p_sn = st.str_to_sn("p", 1)
        a_sn = st.str_to_sn("a", 0)
        b_sn = st.str_to_sn("b", 0)

        a = get_rigid_term(a_sn, 0)
        b = get_rigid_term(b_sn, 0)

        c1 = _cl(_pos(build_binary_term(eq_sn, a, b)))
        c2 = _cl(_pos(get_rigid_term(p_sn, 1, (a,))))
        c3 = _cl(_neg(get_rigid_term(p_sn, 1, (b,))))

        result = _run_search(
            usable=[c1],
            sos=[c2, c3],
            paramodulation=True,
            symbol_table=st,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_transitivity_of_equality(self) -> None:
        """a=b, b=c, p(a), ~p(c) => contradiction (two paramodulation steps)."""
        st = SymbolTable()
        eq_sn = st.str_to_sn("=", 2)
        p_sn = st.str_to_sn("p", 1)
        a_sn = st.str_to_sn("a", 0)
        b_sn = st.str_to_sn("b", 0)
        c_sn = st.str_to_sn("c", 0)

        a = get_rigid_term(a_sn, 0)
        b = get_rigid_term(b_sn, 0)
        c = get_rigid_term(c_sn, 0)

        c1 = _cl(_pos(build_binary_term(eq_sn, a, b)))
        c2 = _cl(_pos(build_binary_term(eq_sn, b, c)))
        c3 = _cl(_pos(get_rigid_term(p_sn, 1, (a,))))
        c4 = _cl(_neg(get_rigid_term(p_sn, 1, (c,))))

        result = _run_search(
            usable=[c1, c2],
            sos=[c3, c4],
            paramodulation=True,
            symbol_table=st,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_nested_function_paramodulation(self) -> None:
        """f(a)=b, p(f(a)), ~p(b) => contradiction (paramodulate into nested term)."""
        st = SymbolTable()
        eq_sn = st.str_to_sn("=", 2)
        f_sn = st.str_to_sn("f", 1)
        p_sn = st.str_to_sn("p", 1)
        a_sn = st.str_to_sn("a", 0)
        b_sn = st.str_to_sn("b", 0)

        a = get_rigid_term(a_sn, 0)
        b = get_rigid_term(b_sn, 0)
        fa = build_unary_term(f_sn, a)

        c1 = _cl(_pos(build_binary_term(eq_sn, fa, b)))
        c2 = _cl(_pos(get_rigid_term(p_sn, 1, (fa,))))
        c3 = _cl(_neg(get_rigid_term(p_sn, 1, (b,))))

        result = _run_search(
            usable=[c1],
            sos=[c2, c3],
            paramodulation=True,
            symbol_table=st,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_variable_equation_paramodulation(self) -> None:
        """f(x)=g(x), p(f(a)), ~p(g(a)) => contradiction."""
        st = SymbolTable()
        eq_sn = st.str_to_sn("=", 2)
        f_sn = st.str_to_sn("f", 1)
        g_sn = st.str_to_sn("g", 1)
        p_sn = st.str_to_sn("p", 1)
        a_sn = st.str_to_sn("a", 0)

        x = get_variable_term(0)
        a = get_rigid_term(a_sn, 0)
        fx = build_unary_term(f_sn, x)
        gx = build_unary_term(g_sn, x)
        fa = build_unary_term(f_sn, a)
        ga = build_unary_term(g_sn, a)

        c1 = _cl(_pos(build_binary_term(eq_sn, fx, gx)))
        c2 = _cl(_pos(get_rigid_term(p_sn, 1, (fa,))))
        c3 = _cl(_neg(get_rigid_term(p_sn, 1, (ga,))))

        result = _run_search(
            usable=[c1],
            sos=[c2, c3],
            paramodulation=True,
            symbol_table=st,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT


# ── Demodulation Integration ────────────────────────────────────────────────


class TestDemodulationIntegration:
    """Test demodulation (term rewriting) integration with search."""

    def test_search_with_demodulation(self) -> None:
        """Demodulation simplifies clauses during search.

        a=b (demodulator), p(a), ~p(b) => with demod, p(a) rewrites to p(b).
        """
        st = SymbolTable()
        eq_sn = st.str_to_sn("=", 2)
        p_sn = st.str_to_sn("p", 1)
        a_sn = st.str_to_sn("a", 0)
        b_sn = st.str_to_sn("b", 0)

        a = get_rigid_term(a_sn, 0)
        b = get_rigid_term(b_sn, 0)

        c1 = _cl(_pos(build_binary_term(eq_sn, a, b)))
        c2 = _cl(_pos(get_rigid_term(p_sn, 1, (a,))))
        c3 = _cl(_neg(get_rigid_term(p_sn, 1, (b,))))

        result = _run_search(
            usable=[c1],
            sos=[c2, c3],
            paramodulation=True,
            demodulation=True,
            symbol_table=st,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT


# ── Combined Features ───────────────────────────────────────────────────────


class TestCombinedFeatures:
    """Test multiple features working together."""

    def test_resolution_with_factoring(self) -> None:
        """Proof requiring both resolution and factoring."""
        P_sn, a_sn = 1, 2
        a = get_rigid_term(a_sn, 0)
        x = get_variable_term(0)
        y = get_variable_term(1)

        # P(x)|P(y)
        c1 = _cl(
            _pos(get_rigid_term(P_sn, 1, (x,))),
            _pos(get_rigid_term(P_sn, 1, (y,))),
        )
        # ~P(a)
        c2 = _cl(_neg(get_rigid_term(P_sn, 1, (a,))))

        result = _run_search(usable=[], sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_paramodulation_plus_resolution(self) -> None:
        """Proof requiring both paramodulation and resolution.

        a=b, P(a), ~P(x)|Q(x), ~Q(b) => contradiction.
        Paramodulation yields P(b), then resolution with ~P(x)|Q(x) gives Q(b),
        which resolves with ~Q(b).
        """
        st = SymbolTable()
        eq_sn = st.str_to_sn("=", 2)
        P_sn = st.str_to_sn("P", 1)
        Q_sn = st.str_to_sn("Q", 1)
        a_sn = st.str_to_sn("a", 0)
        b_sn = st.str_to_sn("b", 0)

        a = get_rigid_term(a_sn, 0)
        b = get_rigid_term(b_sn, 0)
        x = get_variable_term(0)

        c1 = _cl(_pos(build_binary_term(eq_sn, a, b)))
        c2 = _cl(_pos(get_rigid_term(P_sn, 1, (a,))))
        c3 = _cl(_neg(get_rigid_term(P_sn, 1, (x,))), _pos(get_rigid_term(Q_sn, 1, (x,))))
        c4 = _cl(_neg(get_rigid_term(Q_sn, 1, (b,))))

        result = _run_search(
            usable=[c1],
            sos=[c2, c3, c4],
            paramodulation=True,
            symbol_table=st,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_proof_contains_clauses(self) -> None:
        """Proof object contains the clause trace."""
        P_sn, a_sn = 1, 2
        a = get_rigid_term(a_sn, 0)
        P_a = get_rigid_term(P_sn, 1, (a,))

        c1 = _cl(_pos(P_a))
        c2 = _cl(_neg(P_a))

        result = _run_search(usable=[], sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        proof = result.proofs[0]
        assert proof.empty_clause is not None
        assert proof.empty_clause.is_empty
        assert len(proof.clauses) >= 2  # at least the two input clauses
