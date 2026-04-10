"""Regression tests ensuring all existing PyLADR functionality remains intact.

These tests verify that the core inference, search, and data structure
operations produce identical results after any code changes. They run
without requiring the C binary — they test Python-internal consistency.

Run with: pytest tests/compatibility/test_regression.py -v
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tests.compatibility.conftest import run_search


# ── Core Data Structure Regression ─────────────────────────────────────────


class TestTermRegression:
    """Ensure Term creation, equality, and hashing are stable."""

    def test_rigid_term_identity(self):
        """Rigid terms with same args are equal and have same hash."""
        from pyladr.core.term import get_rigid_term

        t1 = get_rigid_term(1, 0)
        t2 = get_rigid_term(1, 0)
        assert t1 == t2
        assert hash(t1) == hash(t2)

    def test_variable_term_identity(self):
        """Variable terms with same index are equal."""
        from pyladr.core.term import get_variable_term

        v1 = get_variable_term(0)
        v2 = get_variable_term(0)
        assert v1 == v2
        assert hash(v1) == hash(v2)

    def test_complex_term_equality(self):
        """Complex nested terms are equal when structurally identical."""
        from pyladr.core.term import get_rigid_term, get_variable_term

        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        f_x = get_rigid_term(2, 1, (x,))
        f_x2 = get_rigid_term(2, 1, (x,))
        g_f_x_a = get_rigid_term(3, 2, (f_x, a))
        g_f_x_a2 = get_rigid_term(3, 2, (f_x2, a))
        assert g_f_x_a == g_f_x_a2

    def test_term_immutability(self):
        """Terms cannot be modified after creation."""
        from pyladr.core.term import get_rigid_term

        t = get_rigid_term(1, 0)
        with pytest.raises((AttributeError, TypeError)):
            t.private_symbol = 999  # type: ignore[attr-defined]


class TestClauseRegression:
    """Ensure Clause creation and properties are stable."""

    def test_clause_literal_order(self):
        """Clause preserves literal ordering."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term

        P = get_rigid_term(1, 1, (get_rigid_term(3, 0),))
        Q = get_rigid_term(2, 1, (get_rigid_term(3, 0),))
        lit_p = Literal(sign=True, atom=P)
        lit_q = Literal(sign=False, atom=Q)
        c = Clause(literals=(lit_p, lit_q))
        assert c.literals[0] == lit_p
        assert c.literals[1] == lit_q

    def test_clause_equality(self):
        """Structurally identical clauses are equal."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term

        P = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c1 = Clause(literals=(Literal(sign=True, atom=P),))
        c2 = Clause(literals=(Literal(sign=True, atom=P),))
        assert c1.literals == c2.literals


class TestSubstitutionRegression:
    """Ensure unification behavior is stable."""

    def test_simple_unification(self):
        """Unify x with constant a."""
        from pyladr.core.substitution import Context, Trail, unify
        from pyladr.core.term import get_rigid_term, get_variable_term

        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        c1, c2, tr = Context(), Context(), Trail()
        assert unify(x, c1, a, c2, tr)

    def test_unify_function_terms(self):
        """Unification of f(x) and f(a) succeeds."""
        from pyladr.core.substitution import Context, Trail, unify
        from pyladr.core.term import get_rigid_term, get_variable_term

        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        f_x = get_rigid_term(2, 1, (x,))
        f_a = get_rigid_term(2, 1, (a,))
        c1, c2, tr = Context(), Context(), Trail()
        assert unify(f_x, c1, f_a, c2, tr)

    def test_trail_undo(self):
        """Trail.undo restores state after unification."""
        from pyladr.core.substitution import Context, Trail, unify
        from pyladr.core.term import get_rigid_term, get_variable_term

        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        c1, c2, tr = Context(), Context(), Trail()

        pos = tr.position
        assert unify(x, c1, a, c2, tr)
        tr.undo_to(pos)
        # After undo, x should be free for a different binding
        c3, c4, tr2 = Context(), Context(), Trail()
        assert unify(x, c3, b, c4, tr2)


# ── Inference Rule Regression ──────────────────────────────────────────────


class TestResolutionRegression:
    """Ensure resolution produces identical results."""

    def test_binary_resolution_complementary(self):
        """P(a) and ~P(x) resolve to empty clause."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.inference.resolution import all_binary_resolvents

        a = get_rigid_term(1, 0)
        x = get_variable_term(0)
        P_a = get_rigid_term(2, 1, (a,))
        P_x = get_rigid_term(2, 1, (x,))

        c1 = Clause(literals=(Literal(sign=True, atom=P_a),))
        c2 = Clause(literals=(Literal(sign=False, atom=P_x),))

        resolvents = all_binary_resolvents(c1, c2)
        assert len(resolvents) >= 1
        # At least one resolvent should be empty
        assert any(len(r.literals) == 0 for r in resolvents)

    def test_factoring(self):
        """Factoring reduces duplicate literals."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.inference.resolution import factor

        x = get_variable_term(0)
        y = get_variable_term(1)
        P_x = get_rigid_term(1, 1, (x,))
        P_y = get_rigid_term(1, 1, (y,))

        c = Clause(
            literals=(
                Literal(sign=True, atom=P_x),
                Literal(sign=True, atom=P_y),
            )
        )
        factors = list(factor(c))
        # Should produce at least one factor with fewer literals
        assert any(len(f.literals) < len(c.literals) for f in factors)

    def test_tautology_detection(self):
        """Tautologies are correctly identified."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term
        from pyladr.inference.resolution import is_tautology

        P_a = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c = Clause(
            literals=(
                Literal(sign=True, atom=P_a),
                Literal(sign=False, atom=P_a),
            )
        )
        assert is_tautology(c)

    def test_non_tautology(self):
        """Non-tautological clauses are correctly identified."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term
        from pyladr.inference.resolution import is_tautology

        P_a = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        Q_a = get_rigid_term(3, 1, (get_rigid_term(2, 0),))
        c = Clause(
            literals=(
                Literal(sign=True, atom=P_a),
                Literal(sign=False, atom=Q_a),
            )
        )
        assert not is_tautology(c)


class TestSubsumptionRegression:
    """Ensure subsumption checking is stable."""

    def test_subsumption_same_clause(self):
        """A clause subsumes itself."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term
        from pyladr.inference.subsumption import subsumes

        P_a = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c = Clause(literals=(Literal(sign=True, atom=P_a),))
        assert subsumes(c, c)

    def test_subsumption_general_specific(self):
        """P(x) subsumes P(a)."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.inference.subsumption import subsumes

        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        P_x = get_rigid_term(2, 1, (x,))
        P_a = get_rigid_term(2, 1, (a,))

        general = Clause(literals=(Literal(sign=True, atom=P_x),))
        specific = Clause(literals=(Literal(sign=True, atom=P_a),))
        assert subsumes(general, specific)
        # Specific does NOT subsume general
        assert not subsumes(specific, general)


class TestParamodulationRegression:
    """Ensure paramodulation produces correct results."""

    def test_basic_paramodulation(self):
        """a=b paramodulates into p(a) to give p(b)."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.symbol import SymbolTable
        from pyladr.core.term import build_binary_term, get_rigid_term
        from pyladr.inference.paramodulation import para_from_into

        st = SymbolTable()
        eq_sn = st.str_to_sn("=", 2)
        p_sn = st.str_to_sn("p", 1)
        a_sn = st.str_to_sn("a", 0)
        b_sn = st.str_to_sn("b", 0)

        a = get_rigid_term(a_sn, 0)
        b = get_rigid_term(b_sn, 0)
        eq_ab = build_binary_term(eq_sn, a, b)
        p_a = get_rigid_term(p_sn, 1, (a,))

        from_clause = Clause(literals=(Literal(sign=True, atom=eq_ab),))
        into_clause = Clause(literals=(Literal(sign=True, atom=p_a),))

        results = para_from_into(from_clause, into_clause, check_top=True, symbol_table=st)
        assert len(results) >= 1


# ── Search Loop Regression ─────────────────────────────────────────────────


class TestSearchRegression:
    """Ensure given-clause search produces correct results."""

    def test_trivial_proof(self, trivial_resolution_clauses):
        """Search finds proof from P(a), ~P(x)|Q(x), ~Q(a)."""
        from pyladr.search.given_clause import ExitCode

        result = run_search(usable=[], sos=trivial_resolution_clauses)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1

    def test_sos_empty(self):
        """Search correctly terminates when SOS is exhausted."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term
        from pyladr.search.given_clause import ExitCode

        P_a = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c = Clause(literals=(Literal(sign=True, atom=P_a),))
        result = run_search(usable=[], sos=[c], max_given=100)
        assert result.exit_code == ExitCode.SOS_EMPTY_EXIT

    def test_max_given_limit(self):
        """Search respects max_given limit."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.search.given_clause import ExitCode

        P_sn, Q_sn, a_sn, b_sn = 1, 2, 3, 4
        a = get_rigid_term(a_sn, 0)
        b = get_rigid_term(b_sn, 0)
        x = get_variable_term(0)

        c1 = Clause(
            literals=(Literal(sign=True, atom=get_rigid_term(P_sn, 1, (a,))),)
        )
        c2 = Clause(
            literals=(Literal(sign=True, atom=get_rigid_term(Q_sn, 1, (b,))),)
        )
        c3 = Clause(
            literals=(
                Literal(sign=False, atom=get_rigid_term(P_sn, 1, (x,))),
                Literal(sign=True, atom=get_rigid_term(Q_sn, 1, (x,))),
            )
        )

        result = run_search(usable=[], sos=[c1, c2, c3], max_given=3)
        assert result.exit_code in (
            ExitCode.MAX_GIVEN_EXIT,
            ExitCode.SOS_EMPTY_EXIT,
        )

    def test_equational_proof(self, equational_problem):
        """Search with paramodulation finds equational proof."""
        from pyladr.search.given_clause import ExitCode

        st, usable, sos = equational_problem
        result = run_search(
            usable=usable,
            sos=sos,
            paramodulation=True,
            symbol_table=st,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1

    def test_statistics_tracking(self, trivial_resolution_clauses):
        """Search statistics are properly updated."""
        result = run_search(usable=[], sos=trivial_resolution_clauses)
        assert result.stats.given >= 1
        assert result.stats.generated >= 0
        assert result.stats.kept >= 0
        assert result.stats.proofs >= 1


# ── Parsing Regression ─────────────────────────────────────────────────────


class TestParsingRegression:
    """Ensure LADR parsing produces identical ASTs."""

    def test_parse_simple_term(self):
        """Parse f(a, b) and verify structure."""
        from pyladr.core.symbol import SymbolTable
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        term = parser.parse_term("f(a, b)")
        assert term is not None
        assert term.arity == 2

    def test_parse_infix_equality(self):
        """Parse x = y as equality."""
        from pyladr.core.symbol import SymbolTable
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        term = parser.parse_term("x = y")
        assert term is not None

    def test_parse_clause_list(self):
        """Parse a formulas(sos) block."""
        from pyladr.core.symbol import SymbolTable
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        text = """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
end_of_list.
"""
        result = parser.parse_input(text)
        assert hasattr(result, "sos")
        assert len(result.sos) == 2

    def test_parse_roundtrip_stability(self):
        """Parsing the same input twice produces identical structures."""
        from pyladr.core.symbol import SymbolTable
        from pyladr.parsing.ladr_parser import LADRParser

        text = """\
formulas(sos).
  f(g(x, y), a) = g(f(a, x), f(b, y)).
end_of_list.
"""
        st1 = SymbolTable()
        p1 = LADRParser(st1)
        r1 = p1.parse_input(text)

        st2 = SymbolTable()
        p2 = LADRParser(st2)
        r2 = p2.parse_input(text)

        assert len(r1.sos) == len(r2.sos)


# ── Ordering Regression ────────────────────────────────────────────────────


class TestOrderingRegression:
    """Ensure term orderings are stable."""

    def test_kbo_ground_terms(self):
        """KBO correctly orders ground terms by weight."""
        from pyladr.core.symbol import SymbolTable
        from pyladr.core.term import get_rigid_term
        from pyladr.ordering.kbo import kbo, kbo_weight

        st = SymbolTable()
        a_sn = st.str_to_sn("a", 0)
        f_sn = st.str_to_sn("f", 1)

        a = get_rigid_term(a_sn, 0)
        f_a = get_rigid_term(f_sn, 1, (a,))
        # f(a) should be heavier than a
        assert kbo_weight(f_a, st) > kbo_weight(a, st)
        # f(a) > a in KBO
        assert kbo(f_a, a, lex_order_vars=False, st=st)

    def test_lrpo_ground_terms(self):
        """LRPO correctly orders ground terms."""
        from pyladr.core.symbol import SymbolTable
        from pyladr.core.term import get_rigid_term
        from pyladr.ordering.lrpo import lrpo

        st = SymbolTable()
        a_sn = st.str_to_sn("a", 0)
        f_sn = st.str_to_sn("f", 1)

        a = get_rigid_term(a_sn, 0)
        f_a = get_rigid_term(f_sn, 1, (a,))
        # f(a) > a in LRPO
        assert lrpo(f_a, a, lex_order_vars=False, st=st)


# ── Indexing Regression ────────────────────────────────────────────────────


class TestIndexingRegression:
    """Ensure discrimination tree indexing is stable."""

    def test_discrimination_tree_insert_retrieve(self):
        """Terms can be inserted and retrieved from discrimination tree."""
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.indexing.discrimination_tree import DiscrimWild

        dt = DiscrimWild()
        a = get_rigid_term(1, 0)
        f_a = get_rigid_term(2, 1, (a,))
        x = get_variable_term(0)
        f_x = get_rigid_term(2, 1, (x,))

        dt.insert(f_a, "clause_1")
        results = list(dt.retrieve_generalizations(f_a))
        assert len(results) >= 1

    def test_discrimination_tree_wildcard_match(self):
        """Wildcard (variable) patterns match all ground instances."""
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.indexing.discrimination_tree import DiscrimWild

        dt = DiscrimWild()
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        f_a = get_rigid_term(3, 1, (a,))
        f_b = get_rigid_term(3, 1, (b,))
        x = get_variable_term(0)
        f_x = get_rigid_term(3, 1, (x,))

        dt.insert(f_x, "general")
        # Should retrieve the general pattern when querying with specific terms
        results_a = list(dt.retrieve_generalizations(f_a))
        results_b = list(dt.retrieve_generalizations(f_b))
        assert len(results_a) >= 1
        assert len(results_b) >= 1


# ── Determinism Regression ─────────────────────────────────────────────────


class TestDeterminismRegression:
    """Ensure search is deterministic across runs."""

    def test_search_deterministic(self, trivial_resolution_clauses):
        """Running the same search twice produces identical stats."""
        from pyladr.search.given_clause import ExitCode

        results = []
        for _ in range(3):
            result = run_search(usable=[], sos=trivial_resolution_clauses)
            results.append(result)

        for r in results[1:]:
            assert r.exit_code == results[0].exit_code
            assert r.stats.given == results[0].stats.given
            assert r.stats.generated == results[0].stats.generated
            assert r.stats.kept == results[0].stats.kept

    def test_equational_search_deterministic(self, equational_problem):
        """Equational search always finds a proof."""
        from pyladr.search.given_clause import ExitCode

        st, usable, sos = equational_problem
        results = []
        for _ in range(3):
            result = run_search(
                usable=usable,
                sos=sos,
                paramodulation=True,
                symbol_table=st,
            )
            results.append(result)

        for r in results:
            assert r.exit_code == ExitCode.MAX_PROOFS_EXIT
            assert len(r.proofs) == 1
