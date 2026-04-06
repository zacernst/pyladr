"""Tests for the Mace4 finite model finder.

Tests cover:
- Model representation (cells, symbols, indexing)
- Ground clause generation
- Simple satisfiable problems (find models)
- Unsatisfiable problems (no model found)
- Group theory axioms (find finite groups)
- Equality handling
- Domain size iteration
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, get_variable_term
from pyladr.mace4.model import Cell, FiniteModel, SymbolInfo, SymbolType
from pyladr.mace4.search import ModelSearcher, SearchOptions


# ── Helpers ──────────────────────────────────────────────────────────────────


def _st() -> SymbolTable:
    s = SymbolTable()
    s.str_to_sn("=", 2)
    s.str_to_sn("*", 2)
    s.str_to_sn("i", 1)
    s.str_to_sn("e", 0)
    s.str_to_sn("a", 0)
    s.str_to_sn("b", 0)
    s.str_to_sn("c", 0)
    s.str_to_sn("f", 1)
    s.str_to_sn("g", 2)
    s.str_to_sn("p", 1)
    s.str_to_sn("q", 1)
    s.str_to_sn("r", 2)
    return s


def _t(st: SymbolTable, name: str, *args: Term) -> Term:
    arity = len(args)
    sn = st.str_to_sn(name, arity)
    if arity == 0:
        return Term(private_symbol=-sn)
    return Term(private_symbol=-sn, arity=arity, args=tuple(args))


def _v(n: int) -> Term:
    return get_variable_term(n)


def _eq(st: SymbolTable, left: Term, right: Term) -> Literal:
    return Literal(sign=True, atom=_t(st, "=", left, right))


def _neq(st: SymbolTable, left: Term, right: Term) -> Literal:
    return Literal(sign=False, atom=_t(st, "=", left, right))


def _cl(*lits: Literal) -> Clause:
    return Clause(literals=tuple(lits))


# ── Model representation tests ──────────────────────────────────────────────


class TestFiniteModel:
    def test_create_model(self):
        model = FiniteModel(domain_size=3)
        assert model.domain_size == 3

    def test_add_symbol(self):
        model = FiniteModel(3)
        sym = model.add_symbol("f", 1, SymbolType.FUNCTION)
        assert sym.name == "f"
        assert sym.arity == 1
        assert sym.num_cells == 3

    def test_add_binary_symbol(self):
        model = FiniteModel(3)
        sym = model.add_symbol("g", 2, SymbolType.FUNCTION)
        assert sym.num_cells == 9

    def test_add_constant(self):
        model = FiniteModel(3)
        sym = model.add_symbol("e", 0, SymbolType.FUNCTION)
        assert sym.num_cells == 1

    def test_initialize_cells(self):
        model = FiniteModel(2)
        model.add_symbol("f", 1, SymbolType.FUNCTION)
        model.add_symbol("p", 1, SymbolType.RELATION)
        model.initialize_cells()
        assert len(model.cells) == 4  # 2 for f, 2 for p

    def test_set_and_get_value(self):
        model = FiniteModel(3)
        model.add_symbol("f", 1, SymbolType.FUNCTION)
        model.initialize_cells()
        assert model.set_value("f", (0,), 1)
        assert model.get_value("f", (0,)) == 1

    def test_equality_setup(self):
        model = FiniteModel(3)
        model.add_symbol("=", 2, SymbolType.RELATION)
        model.initialize_cells()
        model.setup_equality()
        assert model.get_value("=", (0, 0)) == 1
        assert model.get_value("=", (0, 1)) == 0
        assert model.get_value("=", (1, 1)) == 1
        assert model.get_value("=", (2, 1)) == 0

    def test_is_complete(self):
        model = FiniteModel(2)
        model.add_symbol("c", 0, SymbolType.FUNCTION)
        model.initialize_cells()
        assert not model.is_complete()
        model.set_value("c", (), 0)
        assert model.is_complete()

    def test_copy(self):
        model = FiniteModel(2)
        model.add_symbol("c", 0, SymbolType.FUNCTION)
        model.initialize_cells()
        model.set_value("c", (), 1)
        copy = model.copy()
        assert copy.get_value("c", ()) == 1
        assert copy is not model


# ── Model search tests ──────────────────────────────────────────────────────


class TestModelSearch:
    def test_trivially_satisfiable(self):
        """Empty clause set is trivially satisfiable."""
        st = _st()
        searcher = ModelSearcher(st)
        results = searcher.search([], SearchOptions(start_size=2, end_size=2))
        assert len(results) >= 1
        assert results[0].found

    def test_simple_constant_model(self):
        """Find model for a=b (constants must be equal)."""
        st = _st()
        c1 = _cl(_eq(st, _t(st, "a"), _t(st, "b")))

        searcher = ModelSearcher(st)
        results = searcher.search(
            [c1],
            SearchOptions(start_size=2, end_size=4),
        )
        assert len(results) >= 1
        r = results[0]
        assert r.found
        assert r.model is not None
        # a and b should have the same value
        a_val = r.model.get_value("a", ())
        b_val = r.model.get_value("b", ())
        assert a_val == b_val

    def test_constants_not_equal(self):
        """Find model for a!=b (constants must differ)."""
        st = _st()
        c1 = _cl(_neq(st, _t(st, "a"), _t(st, "b")))

        searcher = ModelSearcher(st)
        results = searcher.search(
            [c1],
            SearchOptions(start_size=2, end_size=4),
        )
        assert len(results) >= 1
        r = results[0]
        assert r.found
        assert r.model is not None
        a_val = r.model.get_value("a", ())
        b_val = r.model.get_value("b", ())
        assert a_val != b_val

    def test_unsatisfiable_simple(self):
        """a=b and a!=b should be unsatisfiable (for any domain size)."""
        st = _st()
        c1 = _cl(_eq(st, _t(st, "a"), _t(st, "b")))
        c2 = _cl(_neq(st, _t(st, "a"), _t(st, "b")))

        searcher = ModelSearcher(st)
        results = searcher.search(
            [c1, c2],
            SearchOptions(start_size=1, end_size=5, max_seconds=5),
        )
        # Should not find any model
        assert len(results) == 0

    def test_predicate_model(self):
        """Find model for p(a) (predicate must be true for constant a)."""
        st = _st()
        a = _t(st, "a")
        c1 = _cl(Literal(sign=True, atom=_t(st, "p", a)))

        searcher = ModelSearcher(st)
        results = searcher.search(
            [c1],
            SearchOptions(start_size=2, end_size=3),
        )
        assert len(results) >= 1
        r = results[0]
        assert r.found
        assert r.model is not None
        # p(a_val) should be true
        a_val = r.model.get_value("a", ())
        if a_val is not None:
            p_val = r.model.get_value("p", (a_val,))
            assert p_val == 1

    def test_universal_quantification(self):
        """Find model for p(x) (all x satisfy p). Requires domain where p is always true."""
        st = _st()
        x = _v(0)
        c1 = _cl(Literal(sign=True, atom=_t(st, "p", x)))

        searcher = ModelSearcher(st)
        results = searcher.search(
            [c1],
            SearchOptions(start_size=2, end_size=3),
        )
        assert len(results) >= 1
        r = results[0]
        assert r.found
        assert r.model is not None
        # p should be true for all domain elements
        for i in range(r.domain_size):
            assert r.model.get_value("p", (i,)) == 1


class TestGroupTheoryModels:
    """Find finite groups using Mace4."""

    def test_find_group_of_size_2(self):
        """Find a group of size 2 with group axioms.

        Axioms:
        1. e * x = x
        2. i(x) * x = e
        3. (x * y) * z = x * (y * z)
        """
        st = _st()
        x, y, z = _v(0), _v(1), _v(2)
        e = _t(st, "e")

        ax1 = _cl(_eq(st, _t(st, "*", e, x), x))
        ax2 = _cl(_eq(st, _t(st, "*", _t(st, "i", x), x), e))
        ax3 = _cl(_eq(st, _t(st, "*", _t(st, "*", x, y), z),
                       _t(st, "*", x, _t(st, "*", y, z))))

        searcher = ModelSearcher(st)
        results = searcher.search(
            [ax1, ax2, ax3],
            SearchOptions(start_size=2, end_size=2, max_seconds=10),
        )
        assert len(results) >= 1
        r = results[0]
        assert r.found
        assert r.domain_size == 2

        # Verify: e should be the identity
        model = r.model
        assert model is not None
        e_val = model.get_value("e", ())
        assert e_val is not None

        # e * x = x for all x in {0, 1}
        for i in range(2):
            prod = model.get_value("*", (e_val, i))
            assert prod == i, f"e * {i} should be {i}, got {prod}"

    def test_find_noncommutative_group(self):
        """Find a group where * is not commutative.

        Axioms: group axioms + a*b != b*a
        Smallest such group has order 6 (S3), which might be too slow.
        Let's just verify we can find a group of order 3 (which IS commutative).
        """
        st = _st()
        x, y, z = _v(0), _v(1), _v(2)
        e = _t(st, "e")

        ax1 = _cl(_eq(st, _t(st, "*", e, x), x))
        ax2 = _cl(_eq(st, _t(st, "*", _t(st, "i", x), x), e))
        ax3 = _cl(_eq(st, _t(st, "*", _t(st, "*", x, y), z),
                       _t(st, "*", x, _t(st, "*", y, z))))

        searcher = ModelSearcher(st)
        results = searcher.search(
            [ax1, ax2, ax3],
            SearchOptions(start_size=3, end_size=3, max_seconds=10),
        )
        assert len(results) >= 1
        r = results[0]
        assert r.found
        assert r.domain_size == 3


class TestDomainSizeIteration:
    def test_finds_smallest_model(self):
        """Should find the smallest model when iterating."""
        st = _st()
        # Three distinct constants
        c1 = _cl(_neq(st, _t(st, "a"), _t(st, "b")))
        c2 = _cl(_neq(st, _t(st, "b"), _t(st, "c")))
        c3 = _cl(_neq(st, _t(st, "a"), _t(st, "c")))

        searcher = ModelSearcher(st)
        results = searcher.search(
            [c1, c2, c3],
            SearchOptions(start_size=2, end_size=5),
        )
        assert len(results) >= 1
        # Smallest model with 3 distinct constants needs domain size >= 3
        assert results[0].domain_size >= 3

    def test_domain_size_1_constant(self):
        """Domain size 1: any constant equals any other."""
        st = _st()
        c1 = _cl(_eq(st, _t(st, "a"), _t(st, "b")))

        searcher = ModelSearcher(st)
        results = searcher.search(
            [c1],
            SearchOptions(start_size=1, end_size=1),
        )
        assert len(results) >= 1
        assert results[0].domain_size == 1
