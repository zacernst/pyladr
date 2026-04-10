"""Tests for pyladr.ml.graph.clause_graph — clause to graph conversion.

Tests cover:
- Basic graph construction for unit, horn, ground, and multi-literal clauses
- Node type counts and feature dimensions
- Edge connectivity (clause→literal→term→symbol/variable)
- Variable sharing detection across literals
- Symbol table integration for rich features
- Configuration options (depth limits, feature toggles)
- Batch conversion
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import Symbol, SymbolTable, SymbolType
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.ml.graph.clause_graph import (
    ClauseGraphConfig,
    EdgeType,
    NodeType,
    _GraphBuilder,
    batch_clauses_to_heterograph,
    clause_to_heterograph,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _atom(symnum: int, *args) -> "Term":
    """Build an atomic term f(args...)."""
    return get_rigid_term(symnum, len(args), tuple(args))


def _const(symnum: int) -> "Term":
    """Build a constant term."""
    return get_rigid_term(symnum, 0)


def _var(n: int) -> "Term":
    """Build a variable term."""
    return get_variable_term(n)


def _pos_lit(atom) -> Literal:
    return Literal(sign=True, atom=atom)


def _neg_lit(atom) -> Literal:
    return Literal(sign=False, atom=atom)


def _make_symbol_table() -> SymbolTable:
    """Create a SymbolTable with common test symbols."""
    st = SymbolTable()
    # Register some symbols: symnum 1 = 'f'/2, symnum 2 = 'a'/0, etc.
    st.str_to_sn("f", 2)   # symnum 1
    st.str_to_sn("a", 0)   # symnum 2
    st.str_to_sn("b", 0)   # symnum 3
    st.str_to_sn("P", 1)   # symnum 4
    st.str_to_sn("Q", 2)   # symnum 5
    st.str_to_sn("=", 2)   # symnum 6
    return st


# ── Unit clause tests ─────────────────────────────────────────────────────


class TestUnitClause:
    """Test graph construction from unit clauses (single literal)."""

    def test_ground_unit_clause(self):
        """P(a) — single positive literal with constant argument."""
        a = _const(2)
        atom = _atom(4, a)
        clause = Clause(literals=(Literal(sign=True, atom=atom),))

        graph = clause_to_heterograph(clause)

        # 1 clause, 1 literal, 2 terms (P(a) and a), 2 symbols (P and a), 0 vars
        assert graph[NodeType.CLAUSE.value].num_nodes == 1
        assert graph[NodeType.LITERAL.value].num_nodes == 1
        assert graph[NodeType.TERM.value].num_nodes == 2
        assert graph[NodeType.SYMBOL.value].num_nodes == 2
        assert graph[NodeType.VARIABLE.value].num_nodes == 0

    def test_variable_unit_clause(self):
        """P(x) — single literal with a variable."""
        x = _var(0)
        atom = _atom(4, x)
        clause = Clause(literals=(_pos_lit(atom),))

        graph = clause_to_heterograph(clause)

        assert graph[NodeType.CLAUSE.value].num_nodes == 1
        assert graph[NodeType.LITERAL.value].num_nodes == 1
        # 2 terms: P(x) and x
        assert graph[NodeType.TERM.value].num_nodes == 2
        # 1 symbol: P (x is a variable, not a symbol)
        assert graph[NodeType.SYMBOL.value].num_nodes == 1
        assert graph[NodeType.VARIABLE.value].num_nodes == 1

    def test_nested_term(self):
        """P(f(a, b)) — nested function application."""
        a = _const(2)
        b = _const(3)
        f_ab = _atom(1, a, b)  # f(a, b)
        atom = _atom(4, f_ab)  # P(f(a, b))
        clause = Clause(literals=(_pos_lit(atom),))

        graph = clause_to_heterograph(clause)

        # Terms: P(f(a,b)), f(a,b), a, b = 4 terms
        assert graph[NodeType.TERM.value].num_nodes == 4
        # Symbols: P, f, a, b = 4 symbols
        assert graph[NodeType.SYMBOL.value].num_nodes == 4

    def test_empty_clause(self):
        """Empty clause (contradiction) — only a clause node."""
        clause = Clause(literals=())

        graph = clause_to_heterograph(clause)

        assert graph[NodeType.CLAUSE.value].num_nodes == 1
        assert graph[NodeType.LITERAL.value].num_nodes == 0
        assert graph[NodeType.TERM.value].num_nodes == 0


# ── Multi-literal clause tests ────────────────────────────────────────────


class TestMultiLiteralClause:
    """Test graph construction from clauses with multiple literals."""

    def test_two_literal_clause(self):
        """P(x) | Q(x, a) — two literals sharing variable x."""
        x = _var(0)
        a = _const(2)
        lit1 = _pos_lit(_atom(4, x))        # P(x)
        lit2 = _pos_lit(_atom(5, x, a))     # Q(x, a)
        clause = Clause(literals=(lit1, lit2))

        graph = clause_to_heterograph(clause)

        assert graph[NodeType.CLAUSE.value].num_nodes == 1
        assert graph[NodeType.LITERAL.value].num_nodes == 2
        # Only 1 variable node for x (de-duplicated)
        assert graph[NodeType.VARIABLE.value].num_nodes == 1

    def test_horn_clause(self):
        """-P(x) | Q(x) — horn clause with one positive literal."""
        x = _var(0)
        lit1 = _neg_lit(_atom(4, x))   # -P(x)
        lit2 = _pos_lit(_atom(5, x, _const(2)))  # Q(x, a)
        clause = Clause(literals=(lit1, lit2))

        graph = clause_to_heterograph(clause)

        assert graph[NodeType.LITERAL.value].num_nodes == 2
        assert clause.is_horn

    def test_negative_clause(self):
        """-P(x) | -Q(y) — all negative literals."""
        lit1 = _neg_lit(_atom(4, _var(0)))
        lit2 = _neg_lit(_atom(5, _var(1), _const(2)))
        clause = Clause(literals=(lit1, lit2))

        graph = clause_to_heterograph(clause)

        assert graph[NodeType.VARIABLE.value].num_nodes == 2
        assert clause.is_negative


# ── Edge connectivity tests ───────────────────────────────────────────────


class TestEdgeConnectivity:
    """Test that edges are constructed correctly."""

    def test_clause_to_literal_edges(self):
        """Verify CONTAINS_LITERAL edges from clause to each literal."""
        lit1 = _pos_lit(_atom(4, _var(0)))
        lit2 = _neg_lit(_atom(5, _const(2)))
        clause = Clause(literals=(lit1, lit2))

        graph = clause_to_heterograph(clause)

        edge_key = (
            NodeType.CLAUSE.value,
            EdgeType.CONTAINS_LITERAL.value,
            NodeType.LITERAL.value,
        )
        assert edge_key in graph.edge_types
        edge_index = graph[edge_key].edge_index
        # 2 edges: clause→lit0, clause→lit1
        assert edge_index.shape[1] == 2
        # All from clause node 0
        assert (edge_index[0] == 0).all()

    def test_literal_to_atom_edges(self):
        """Verify HAS_ATOM edges from literal to its atom term."""
        atom = _atom(4, _const(2))
        clause = Clause(literals=(_pos_lit(atom),))

        graph = clause_to_heterograph(clause)

        edge_key = (
            NodeType.LITERAL.value,
            EdgeType.HAS_ATOM.value,
            NodeType.TERM.value,
        )
        assert edge_key in graph.edge_types
        edge_index = graph[edge_key].edge_index
        assert edge_index.shape[1] == 1

    def test_has_arg_edges(self):
        """Verify HAS_ARG edges for function arguments."""
        a = _const(2)
        b = _const(3)
        f_ab = _atom(1, a, b)  # f(a, b) — 2 arguments
        clause = Clause(literals=(_pos_lit(_atom(4, f_ab)),))

        graph = clause_to_heterograph(clause)

        edge_key = (
            NodeType.TERM.value,
            EdgeType.HAS_ARG.value,
            NodeType.TERM.value,
        )
        assert edge_key in graph.edge_types
        edge_index = graph[edge_key].edge_index
        # P→f(a,b) is HAS_ARG, f→a is HAS_ARG, f→b is HAS_ARG = 3 total
        assert edge_index.shape[1] == 3

    def test_symbol_of_edges(self):
        """Verify SYMBOL_OF edges from terms to their symbols."""
        atom = _atom(4, _const(2))  # P(a)
        clause = Clause(literals=(_pos_lit(atom),))

        graph = clause_to_heterograph(clause)

        edge_key = (
            NodeType.TERM.value,
            EdgeType.SYMBOL_OF.value,
            NodeType.SYMBOL.value,
        )
        assert edge_key in graph.edge_types
        # 2 terms (P(a) and a), each links to a symbol
        assert graph[edge_key].edge_index.shape[1] == 2

    def test_var_occurrence_edges(self):
        """Verify VAR_OCCURRENCE edges from variables to terms."""
        x = _var(0)
        atom = _atom(4, x)  # P(x)
        clause = Clause(literals=(_pos_lit(atom),))

        graph = clause_to_heterograph(clause)

        edge_key = (
            NodeType.VARIABLE.value,
            EdgeType.VAR_OCCURRENCE.value,
            NodeType.TERM.value,
        )
        assert edge_key in graph.edge_types
        assert graph[edge_key].edge_index.shape[1] == 1


# ── Variable sharing tests ────────────────────────────────────────────────


class TestVariableSharing:
    """Test SHARED_VARIABLE edge construction."""

    def test_shared_variable_across_literals(self):
        """P(x) | Q(x) — x appears in both literals."""
        x = _var(0)
        lit1 = _pos_lit(_atom(4, x))
        lit2 = _pos_lit(_atom(5, x, _const(2)))
        clause = Clause(literals=(lit1, lit2))

        graph = clause_to_heterograph(clause)

        edge_key = (
            NodeType.VARIABLE.value,
            EdgeType.SHARED_VARIABLE.value,
            NodeType.VARIABLE.value,
        )
        assert edge_key in graph.edge_types
        assert graph[edge_key].edge_index.shape[1] >= 1

    def test_no_shared_variable_single_literal(self):
        """P(x) — variable in only one literal, no sharing."""
        clause = Clause(literals=(_pos_lit(_atom(4, _var(0))),))

        graph = clause_to_heterograph(clause)

        edge_key = (
            NodeType.VARIABLE.value,
            EdgeType.SHARED_VARIABLE.value,
            NodeType.VARIABLE.value,
        )
        assert edge_key not in graph.edge_types

    def test_disable_variable_sharing(self):
        """Config can disable shared-variable edges."""
        x = _var(0)
        lit1 = _pos_lit(_atom(4, x))
        lit2 = _pos_lit(_atom(5, x, _const(2)))
        clause = Clause(literals=(lit1, lit2))

        config = ClauseGraphConfig(include_variable_sharing=False)
        graph = clause_to_heterograph(clause, config=config)

        edge_key = (
            NodeType.VARIABLE.value,
            EdgeType.SHARED_VARIABLE.value,
            NodeType.VARIABLE.value,
        )
        assert edge_key not in graph.edge_types


# ── Feature dimension tests ───────────────────────────────────────────────


class TestFeatureDimensions:
    """Verify feature tensor shapes are consistent."""

    def test_clause_features_dim(self):
        """Clause features should be 7-dimensional."""
        clause = Clause(
            literals=(_pos_lit(_atom(4, _const(2))),),
            weight=3.5,
        )
        graph = clause_to_heterograph(clause)
        assert graph[NodeType.CLAUSE.value].x.shape == (1, 7)

    def test_literal_features_dim(self):
        """Literal features should be 3-dimensional."""
        clause = Clause(literals=(_pos_lit(_atom(4, _const(2))),))
        graph = clause_to_heterograph(clause)
        assert graph[NodeType.LITERAL.value].x.shape == (1, 3)

    def test_term_features_dim(self):
        """Term features should be 8-dimensional."""
        clause = Clause(literals=(_pos_lit(_atom(4, _const(2))),))
        graph = clause_to_heterograph(clause)
        # 2 terms: atom and constant
        assert graph[NodeType.TERM.value].x.shape == (2, 8)

    def test_symbol_features_dim(self):
        """Symbol features should be 6-dimensional."""
        clause = Clause(literals=(_pos_lit(_atom(4, _const(2))),))
        graph = clause_to_heterograph(clause)
        assert graph[NodeType.SYMBOL.value].x.shape == (2, 6)

    def test_variable_features_dim(self):
        """Variable features should be 1-dimensional."""
        clause = Clause(literals=(_pos_lit(_atom(4, _var(0))),))
        graph = clause_to_heterograph(clause)
        assert graph[NodeType.VARIABLE.value].x.shape == (1, 1)


# ── Feature value tests ──────────────────────────────────────────────────


class TestFeatureValues:
    """Verify specific feature values are extracted correctly."""

    def test_clause_weight_in_features(self):
        """Clause weight is captured in feature vector."""
        clause = Clause(
            literals=(_pos_lit(_atom(4, _const(2))),),
            weight=7.5,
        )
        graph = clause_to_heterograph(clause)
        features = graph[NodeType.CLAUSE.value].x[0]
        # weight is feature index 6
        assert features[6].item() == pytest.approx(7.5)

    def test_literal_sign_in_features(self):
        """Literal sign (polarity) is captured."""
        pos = _pos_lit(_atom(4, _const(2)))
        neg = _neg_lit(_atom(5, _const(3)))
        clause = Clause(literals=(pos, neg))
        graph = clause_to_heterograph(clause)
        features = graph[NodeType.LITERAL.value].x
        assert features[0, 0].item() == 1.0  # positive
        assert features[1, 0].item() == 0.0  # negative

    def test_term_type_features(self):
        """Term type flags (variable, constant, complex) are correct."""
        x = _var(0)
        a = _const(2)
        f_xa = _atom(1, x, a)
        atom = _atom(4, f_xa)
        clause = Clause(literals=(_pos_lit(atom),))
        graph = clause_to_heterograph(clause)

        term_features = graph[NodeType.TERM.value].x
        # Check that we have at least one variable term and one constant term
        is_var_col = term_features[:, 0]  # is_variable column
        is_const_col = term_features[:, 1]  # is_constant column
        is_complex_col = term_features[:, 2]  # is_complex column
        assert is_var_col.sum().item() >= 1  # at least x
        assert is_const_col.sum().item() >= 1  # at least a
        assert is_complex_col.sum().item() >= 2  # f(x,a) and P(f(x,a))


# ── Symbol table integration tests ───────────────────────────────────────


class TestSymbolTableIntegration:
    """Test that symbol metadata from SymbolTable enriches features."""

    def test_with_symbol_table(self):
        """Symbol features include arity, type etc. when table provided."""
        st = _make_symbol_table()
        # Mark symnum 4 (P) as predicate
        sym = st.get_symbol(4)
        sym.sym_type = SymbolType.PREDICATE
        sym.occurrences = 5

        a = _const(2)
        atom = _atom(4, a)
        clause = Clause(literals=(_pos_lit(atom),))
        graph = clause_to_heterograph(clause, symbol_table=st)

        sym_features = graph[NodeType.SYMBOL.value].x
        assert sym_features.shape[0] == 2  # P and a
        # Check that features are enriched (occurrences > 0 for P)
        # Find the P symbol node — it should have occurrences=5
        # Features: [symnum, arity, sym_type, is_skolem, kb_weight, occurrences]
        found = False
        for i in range(sym_features.shape[0]):
            if sym_features[i, 5].item() == 5.0:  # occurrences == 5
                found = True
                break
        assert found, "Symbol table metadata not reflected in features"

    def test_without_symbol_table(self):
        """Without symbol table, minimal features are still produced."""
        clause = Clause(literals=(_pos_lit(_atom(4, _const(2))),))
        graph = clause_to_heterograph(clause, symbol_table=None)

        sym_features = graph[NodeType.SYMBOL.value].x
        assert sym_features.shape == (2, 6)


# ── Configuration tests ──────────────────────────────────────────────────


class TestConfiguration:
    """Test ClauseGraphConfig options."""

    def test_max_term_depth_truncation(self):
        """Deep terms are truncated at max_term_depth."""
        # Build f(f(f(a))) — depth 3
        a = _const(2)
        f_a = _atom(1, a)
        f_f_a = _atom(1, f_a)
        f_f_f_a = _atom(1, f_f_a)
        atom = _atom(4, f_f_f_a)  # P(f(f(f(a))))
        clause = Clause(literals=(_pos_lit(atom),))

        # Without limit — all terms present
        graph_full = clause_to_heterograph(clause)
        full_terms = graph_full[NodeType.TERM.value].num_nodes

        # With depth limit of 2 — should truncate deeper terms
        config = ClauseGraphConfig(max_term_depth=2)
        graph_trunc = clause_to_heterograph(clause, config=config)
        trunc_terms = graph_trunc[NodeType.TERM.value].num_nodes

        # Truncated graph should have fewer terms
        assert trunc_terms < full_terms

    def test_disable_symbol_features(self):
        """Disabling symbol features uses minimal feature set."""
        st = _make_symbol_table()
        clause = Clause(literals=(_pos_lit(_atom(4, _const(2))),))

        config = ClauseGraphConfig(include_symbol_features=False)
        graph = clause_to_heterograph(clause, symbol_table=st, config=config)

        # Features should still exist but be minimal
        sym_features = graph[NodeType.SYMBOL.value].x
        assert sym_features.shape[1] == 6


# ── Batch conversion tests ───────────────────────────────────────────────


class TestBatchConversion:
    """Test batch_clauses_to_heterograph."""

    def test_batch_produces_list(self):
        """Batch conversion returns one graph per clause."""
        c1 = Clause(literals=(_pos_lit(_atom(4, _const(2))),))
        c2 = Clause(literals=(_neg_lit(_atom(5, _var(0), _const(3))),))

        graphs = batch_clauses_to_heterograph([c1, c2])

        assert len(graphs) == 2
        assert graphs[0][NodeType.CLAUSE.value].num_nodes == 1
        assert graphs[1][NodeType.CLAUSE.value].num_nodes == 1

    def test_batch_empty_list(self):
        """Empty list returns empty list."""
        graphs = batch_clauses_to_heterograph([])
        assert graphs == []

    def test_batch_independent_graphs(self):
        """Each graph is independent (no shared node indices)."""
        c1 = Clause(literals=(_pos_lit(_atom(4, _var(0))),))
        c2 = Clause(literals=(_pos_lit(_atom(5, _var(1), _const(2))),))

        graphs = batch_clauses_to_heterograph([c1, c2])

        # Each graph's clause node index starts at 0
        assert graphs[0][NodeType.CLAUSE.value].num_nodes == 1
        assert graphs[1][NodeType.CLAUSE.value].num_nodes == 1


# ── Equality literal tests ───────────────────────────────────────────────


class TestEqualityLiterals:
    """Test handling of equality literals (a = b)."""

    def test_equality_clause(self):
        """a = b — equality literal with two constant args."""
        a = _const(2)
        b = _const(3)
        eq_atom = _atom(6, a, b)  # =(a, b)
        clause = Clause(literals=(_pos_lit(eq_atom),))

        graph = clause_to_heterograph(clause)

        # Terms: =(a,b), a, b = 3
        assert graph[NodeType.TERM.value].num_nodes == 3
        # Symbols: =, a, b = 3
        assert graph[NodeType.SYMBOL.value].num_nodes == 3

        # Literal features should indicate equality
        lit_features = graph[NodeType.LITERAL.value].x[0]
        assert lit_features[2].item() == 1.0  # is_eq_literal

    def test_inequality_clause(self):
        """-(a = b) is a negative equality literal."""
        a = _const(2)
        b = _const(3)
        eq_atom = _atom(6, a, b)
        clause = Clause(literals=(_neg_lit(eq_atom),))

        graph = clause_to_heterograph(clause)

        lit_features = graph[NodeType.LITERAL.value].x[0]
        assert lit_features[0].item() == 0.0  # sign = negative
        assert lit_features[2].item() == 1.0  # is_eq_literal


# ── Tensor dtype tests ───────────────────────────────────────────────────


class TestTensorProperties:
    """Verify tensor data types and device placement."""

    def test_feature_dtype(self):
        """All feature tensors should be float32."""
        clause = Clause(literals=(_pos_lit(_atom(4, _var(0))),))
        graph = clause_to_heterograph(clause)

        for nt in NodeType:
            if graph[nt.value].num_nodes > 0:
                assert graph[nt.value].x.dtype == torch.float32

    def test_edge_index_dtype(self):
        """All edge index tensors should be int64 (long)."""
        clause = Clause(literals=(_pos_lit(_atom(4, _const(2))),))
        graph = clause_to_heterograph(clause)

        for edge_type in graph.edge_types:
            assert graph[edge_type].edge_index.dtype == torch.long

    def test_edge_index_shape(self):
        """Edge indices should be 2×N tensors."""
        clause = Clause(literals=(_pos_lit(_atom(4, _const(2))),))
        graph = clause_to_heterograph(clause)

        for edge_type in graph.edge_types:
            assert graph[edge_type].edge_index.shape[0] == 2
