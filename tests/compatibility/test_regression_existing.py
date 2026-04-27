"""Regression tests ensuring all existing PyLADR functionality is preserved.

These tests exercise every major module to verify that ML integration
does not break any existing behavior. They run WITHOUT any ML features
enabled, confirming identical default behavior.

Run with: pytest tests/compatibility/test_regression_existing.py -v
"""

from __future__ import annotations

import pytest


# ── Core Module Regression ──────────────────────────────────────────────────


class TestTermRegression:
    """Verify core term operations are unchanged."""

    def test_variable_creation(self):
        from pyladr.core.term import get_variable_term
        v = get_variable_term(0)
        assert v.private_symbol == 0

    def test_rigid_term_creation(self):
        from pyladr.core.term import get_rigid_term
        t = get_rigid_term(1, 0)
        # Rigid terms store negative symbol number (C-compatible encoding)
        assert t.private_symbol == -1

    def test_complex_term_structure(self):
        from pyladr.core.term import get_rigid_term, get_variable_term
        a = get_rigid_term(1, 0)
        x = get_variable_term(0)
        f_ax = get_rigid_term(2, 2, (a, x))
        assert f_ax.private_symbol == -2
        assert len(f_ax.args) == 2

    def test_term_immutability(self):
        from pyladr.core.term import get_rigid_term
        t = get_rigid_term(1, 0)
        with pytest.raises((AttributeError, TypeError)):
            t.private_symbol = 99  # type: ignore[misc]


class TestClauseRegression:
    """Verify clause construction and operations."""

    def test_clause_creation(self):
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term

        atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c = Clause(literals=(Literal(sign=True, atom=atom),))
        assert len(c.literals) == 1
        assert c.literals[0].sign is True

    def test_clause_id_default(self):
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term

        atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c = Clause(literals=(Literal(sign=True, atom=atom),))
        assert c.id == 0  # default ID

    def test_justification_types(self):
        from pyladr.core.clause import JustType
        # Verify all C-matching justification types exist
        assert JustType.INPUT is not None
        assert JustType.BINARY_RES is not None
        assert JustType.PARA is not None
        assert JustType.DEMOD is not None


class TestSubstitutionRegression:
    """Verify unification and substitution operations."""

    def test_simple_unification(self):
        from pyladr.core.substitution import Context, Trail, unify
        from pyladr.core.term import get_rigid_term, get_variable_term

        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        c1, c2, tr = Context(), Context(), Trail()
        assert unify(x, c1, a, c2, tr)

    def test_unification_failure(self):
        from pyladr.core.substitution import Context, Trail, unify
        from pyladr.core.term import get_rigid_term

        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        c1, c2, tr = Context(), Context(), Trail()
        assert not unify(a, c1, b, c2, tr)

    def test_trail_undo(self):
        from pyladr.core.substitution import Context, Trail, unify
        from pyladr.core.term import get_rigid_term, get_variable_term

        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        c1, c2, tr = Context(), Context(), Trail()
        mark = tr.position
        unify(x, c1, a, c2, tr)
        tr.undo_to(mark)


class TestParsingRegression:
    """Verify LADR parsing is unchanged."""

    def test_parse_term(self):
        from pyladr.parsing.ladr_parser import LADRParser
        from pyladr.core.symbol import SymbolTable

        st = SymbolTable()
        parser = LADRParser(st)
        t = parser.parse_term("f(a, b)")
        assert t is not None

    def test_parse_clause(self):
        from pyladr.parsing.ladr_parser import LADRParser
        from pyladr.core.symbol import SymbolTable

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input("formulas(sos).\nP(a) | -Q(b).\nend_of_list.\n")
        assert len(parsed.sos) == 1
        assert len(parsed.sos[0].literals) == 2

    def test_parse_input_block(self):
        from pyladr.parsing.ladr_parser import LADRParser
        from pyladr.core.symbol import SymbolTable

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input("""\
formulas(sos).
  P(a).
  -P(x) | Q(x).
end_of_list.

formulas(goals).
  Q(a).
end_of_list.
""")
        assert len(parsed.sos) >= 2
        assert len(parsed.goals) >= 1


# ── Inference Rule Regression ──────────────────────────────────────────────


class TestResolutionRegression:
    """Verify binary resolution behavior is preserved."""

    def test_binary_resolution_produces_resolvent(self):
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.inference.resolution import binary_resolve

        P_sn, a_sn = 1, 2
        a = get_rigid_term(a_sn, 0)
        x = get_variable_term(0)

        c1 = Clause(literals=(Literal(sign=True, atom=get_rigid_term(P_sn, 1, (a,))),))
        c2 = Clause(literals=(Literal(sign=False, atom=get_rigid_term(P_sn, 1, (x,))),))

        resolvent = binary_resolve(c1, 0, c2, 0)
        assert resolvent is not None
        assert len(resolvent.literals) == 0  # empty clause

    def test_tautology_detection(self):
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term
        from pyladr.inference.resolution import is_tautology

        P = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c = Clause(literals=(
            Literal(sign=True, atom=P),
            Literal(sign=False, atom=P),
        ))
        assert is_tautology(c)

    def test_factoring(self):
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.inference.resolution import factor

        P_sn = 1
        x = get_variable_term(0)
        y = get_variable_term(1)

        c = Clause(literals=(
            Literal(sign=True, atom=get_rigid_term(P_sn, 1, (x,))),
            Literal(sign=True, atom=get_rigid_term(P_sn, 1, (y,))),
        ))
        factors = list(factor(c))
        assert len(factors) >= 1


class TestSubsumptionRegression:
    """Verify subsumption checking is preserved."""

    def test_simple_subsumption(self):
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.inference.subsumption import subsumes

        P_sn = 1
        a = get_rigid_term(2, 0)
        x = get_variable_term(0)

        general = Clause(literals=(Literal(sign=True, atom=get_rigid_term(P_sn, 1, (x,))),))
        specific = Clause(literals=(Literal(sign=True, atom=get_rigid_term(P_sn, 1, (a,))),))

        assert subsumes(general, specific)
        assert not subsumes(specific, general)


class TestParamodulationRegression:
    """Verify paramodulation is preserved."""

    def test_simple_paramodulation(self):
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

        eq_clause = Clause(
            literals=(Literal(sign=True, atom=build_binary_term(eq_sn, a, b)),)
        )
        target = Clause(
            literals=(Literal(sign=True, atom=get_rigid_term(p_sn, 1, (a,))),)
        )

        results = para_from_into(eq_clause, target, check_top=True, symbol_table=st)
        assert len(results) >= 1


# ── Search Engine Regression ──────────────────────────────────────────────


class TestSearchRegression:
    """Verify the given-clause search loop is unchanged."""

    def test_simple_proof_found(self):
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term
        from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions

        P = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c1 = Clause(literals=(Literal(sign=True, atom=P),))
        c2 = Clause(literals=(Literal(sign=False, atom=P),))

        opts = SearchOptions(binary_resolution=True, factoring=True, max_given=50, quiet=True)
        search = GivenClauseSearch(options=opts)
        result = search.run(usable=[], sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_sos_empty_terminates(self):
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term
        from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions

        P = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c = Clause(literals=(Literal(sign=True, atom=P),))

        opts = SearchOptions(binary_resolution=True, max_given=50, quiet=True)
        search = GivenClauseSearch(options=opts)
        result = search.run(usable=[], sos=[c])
        assert result.exit_code == ExitCode.SOS_EMPTY_EXIT

    def test_max_given_limit(self):
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions

        P_sn, Q_sn = 1, 2
        a = get_rigid_term(3, 0)
        b = get_rigid_term(4, 0)
        x = get_variable_term(0)

        clauses = [
            Clause(literals=(Literal(sign=True, atom=get_rigid_term(P_sn, 1, (a,))),)),
            Clause(literals=(Literal(sign=True, atom=get_rigid_term(Q_sn, 1, (b,))),)),
            Clause(literals=(
                Literal(sign=False, atom=get_rigid_term(P_sn, 1, (x,))),
                Literal(sign=True, atom=get_rigid_term(Q_sn, 1, (x,))),
            )),
        ]

        opts = SearchOptions(binary_resolution=True, max_given=2, quiet=True)
        search = GivenClauseSearch(options=opts)
        result = search.run(usable=[], sos=clauses)
        assert result.exit_code in (ExitCode.MAX_GIVEN_EXIT, ExitCode.SOS_EMPTY_EXIT)

    def test_search_statistics_populated(self):
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions

        P_sn, Q_sn = 1, 2
        a = get_rigid_term(3, 0)
        x = get_variable_term(0)

        c1 = Clause(literals=(Literal(sign=True, atom=get_rigid_term(P_sn, 1, (a,))),))
        c2 = Clause(literals=(
            Literal(sign=False, atom=get_rigid_term(P_sn, 1, (x,))),
            Literal(sign=True, atom=get_rigid_term(Q_sn, 1, (x,))),
        ))
        c3 = Clause(literals=(Literal(sign=False, atom=get_rigid_term(Q_sn, 1, (a,))),))

        opts = SearchOptions(binary_resolution=True, factoring=True, max_given=50, quiet=True)
        search = GivenClauseSearch(options=opts)
        result = search.run(usable=[], sos=[c1, c2, c3])

        assert result.stats.given >= 1
        assert result.stats.generated >= 0
        assert result.stats.kept >= 0

    def test_equational_proof_with_paramodulation(self):
        """Verify paramodulation-based proofs still work."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.symbol import SymbolTable
        from pyladr.core.term import build_binary_term, get_rigid_term
        from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions

        st = SymbolTable()
        eq_sn = st.str_to_sn("=", 2)
        p_sn = st.str_to_sn("p", 1)
        a_sn = st.str_to_sn("a", 0)
        b_sn = st.str_to_sn("b", 0)

        a = get_rigid_term(a_sn, 0)
        b = get_rigid_term(b_sn, 0)

        c1 = Clause(literals=(Literal(sign=True, atom=build_binary_term(eq_sn, a, b)),))
        c2 = Clause(literals=(Literal(sign=True, atom=get_rigid_term(p_sn, 1, (a,))),))
        c3 = Clause(literals=(Literal(sign=False, atom=get_rigid_term(p_sn, 1, (b,))),))

        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            factoring=True,
            max_given=50,
            quiet=True,
        )
        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(usable=[c1], sos=[c2, c3])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT


# ── Indexing Regression ──────────────────────────────────────────────────


class TestIndexingRegression:
    """Verify discrimination tree and indexing are preserved."""

    def test_discrimination_tree_insert_retrieve(self):
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.indexing.discrimination_tree import DiscrimWild

        tree = DiscrimWild()
        a = get_rigid_term(1, 0)
        f_a = get_rigid_term(2, 1, (a,))
        x = get_variable_term(0)
        f_x = get_rigid_term(2, 1, (x,))

        tree.insert(f_a, "clause_1")
        results = list(tree.retrieve_generalizations(f_a))
        assert "clause_1" in results

    def test_feature_index_basic(self):
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term
        from pyladr.indexing.feature_index import FeatureIndex

        idx = FeatureIndex()
        atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c = Clause(literals=(Literal(sign=True, atom=atom),), id=1)
        idx.insert(c)
        candidates = list(idx.forward_candidates(c))
        assert len(candidates) >= 0  # Just verify it doesn't crash


# ── Ordering Regression ──────────────────────────────────────────────────


class TestOrderingRegression:
    """Verify term orderings are preserved."""

    def test_lrpo_ordering(self):
        from pyladr.core.term import get_rigid_term
        from pyladr.core.symbol import SymbolTable
        from pyladr.ordering.lrpo import lrpo

        st = SymbolTable()
        sn_a = st.str_to_sn("a", 0)
        sn_f = st.str_to_sn("f", 1)
        a = get_rigid_term(sn_a, 0)
        f_a = get_rigid_term(sn_f, 1, (a,))
        # lrpo returns True if first term is greater
        assert lrpo(f_a, a, lex_order_vars=False, st=st) is True

    def test_kbo_ordering(self):
        from pyladr.core.term import get_rigid_term
        from pyladr.core.symbol import SymbolTable
        from pyladr.ordering.kbo import kbo

        st = SymbolTable()
        sn_a = st.str_to_sn("a", 0)
        sn_f = st.str_to_sn("f", 1)
        a = get_rigid_term(sn_a, 0)
        f_a = get_rigid_term(sn_f, 1, (a,))
        # kbo returns True if first term is greater
        assert kbo(f_a, a, lex_order_vars=False, st=st) is True
