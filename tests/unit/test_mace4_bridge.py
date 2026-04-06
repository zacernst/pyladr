"""Tests for FiniteModel ↔ Interpretation bridge."""

from __future__ import annotations

import pytest

from pyladr.core.interpretation import (
    Interpretation,
    OperationTable,
    TableType,
    compile_interp_from_text,
    eval_clause,
    format_interp_standard,
    isomorphic_interps,
    permute_interp,
)
from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, get_variable_term
from pyladr.mace4.bridge import (
    finitemodel_to_interpretation,
    interpretation_to_finitemodel,
)
from pyladr.mace4.model import FiniteModel, SymbolType


def _make_term(sym_id: int, args: tuple[Term, ...] = ()) -> Term:
    return Term(private_symbol=-sym_id, arity=len(args), args=args)


def _make_var(n: int) -> Term:
    return get_variable_term(n)


def _make_unit_eq(eq_id: int, lhs: Term, rhs: Term) -> Clause:
    atom = Term(private_symbol=-eq_id, arity=2, args=(lhs, rhs))
    lit = Literal(sign=True, atom=atom)
    return Clause(literals=(lit,))


# ── Helper: build a complete Z2 XOR FiniteModel ─────────────────────────────


def _build_z2_model() -> FiniteModel:
    """Z2 XOR group as a FiniteModel."""
    m = FiniteModel(domain_size=2)
    m.add_symbol("f", arity=2, stype=SymbolType.FUNCTION)
    m.add_symbol("e", arity=0, stype=SymbolType.FUNCTION)
    m.initialize_cells()
    # f is XOR: f(0,0)=0, f(0,1)=1, f(1,0)=1, f(1,1)=0
    m.set_value("f", (0, 0), 0)
    m.set_value("f", (0, 1), 1)
    m.set_value("f", (1, 0), 1)
    m.set_value("f", (1, 1), 0)
    # e = 0
    m.set_value("e", (), 0)
    return m


def _build_z3_model() -> FiniteModel:
    """Z3 addition group as a FiniteModel."""
    m = FiniteModel(domain_size=3)
    m.add_symbol("f", arity=2, stype=SymbolType.FUNCTION)
    m.initialize_cells()
    # f is addition mod 3
    for i in range(3):
        for j in range(3):
            m.set_value("f", (i, j), (i + j) % 3)
    return m


# ── Conversion tests ────────────────────────────────────────────────────────


class TestFiniteModelToInterpretation:
    def test_z2_basic_conversion(self):
        model = _build_z2_model()
        interp = finitemodel_to_interpretation(model)
        assert interp.size == 2
        assert "f" in interp.operations
        assert "e" in interp.operations
        assert interp.operations["f"].values == [0, 1, 1, 0]
        assert interp.operations["e"].values == [0]

    def test_z3_conversion(self):
        model = _build_z3_model()
        interp = finitemodel_to_interpretation(model)
        assert interp.size == 3
        assert interp.operations["f"].values == [0, 1, 2, 1, 2, 0, 2, 0, 1]

    def test_skips_equality(self):
        model = FiniteModel(domain_size=2)
        model.add_symbol("=", arity=2, stype=SymbolType.RELATION)
        model.add_symbol("f", arity=1, stype=SymbolType.FUNCTION)
        model.initialize_cells()
        model.setup_equality()
        model.set_value("f", (0,), 1)
        model.set_value("f", (1,), 0)
        interp = finitemodel_to_interpretation(model)
        assert "=" not in interp.operations
        assert "f" in interp.operations

    def test_relation_type_preserved(self):
        model = FiniteModel(domain_size=2)
        model.add_symbol("p", arity=1, stype=SymbolType.RELATION)
        model.initialize_cells()
        model.set_value("p", (0,), 1)
        model.set_value("p", (1,), 0)
        interp = finitemodel_to_interpretation(model)
        assert interp.operations["p"].table_type == TableType.RELATION

    def test_unassigned_cell_raises(self):
        model = FiniteModel(domain_size=2)
        model.add_symbol("f", arity=1, stype=SymbolType.FUNCTION)
        model.initialize_cells()
        # Don't assign f(1)
        model.set_value("f", (0,), 0)
        with pytest.raises(ValueError, match="Unassigned cell"):
            finitemodel_to_interpretation(model)


class TestInterpretationToFiniteModel:
    def test_roundtrip_z2(self):
        original = _build_z2_model()
        interp = finitemodel_to_interpretation(original)
        roundtripped = interpretation_to_finitemodel(interp)
        assert roundtripped.domain_size == 2
        # Check all f values match
        for i in range(2):
            for j in range(2):
                assert roundtripped.get_value("f", (i, j)) == original.get_value("f", (i, j))
        assert roundtripped.get_value("e", ()) == 0

    def test_roundtrip_z3(self):
        original = _build_z3_model()
        interp = finitemodel_to_interpretation(original)
        roundtripped = interpretation_to_finitemodel(interp)
        for i in range(3):
            for j in range(3):
                assert roundtripped.get_value("f", (i, j)) == (i + j) % 3


# ── Integration: bridge + evaluation ────────────────────────────────────────


class TestBridgeEvaluation:
    def test_commutativity_in_z2(self):
        """Evaluate commutativity on a FiniteModel through the bridge."""
        model = _build_z2_model()
        interp = finitemodel_to_interpretation(model)

        st = SymbolTable()
        eq = st.str_to_sn("=", 2)
        f = st.str_to_sn("f", 2)
        x, y = _make_var(0), _make_var(1)

        comm = _make_unit_eq(eq, _make_term(f, (x, y)), _make_term(f, (y, x)))
        assert eval_clause(comm, interp, st, eq)

    def test_associativity_in_z3(self):
        """Evaluate associativity on a FiniteModel through the bridge."""
        model = _build_z3_model()
        interp = finitemodel_to_interpretation(model)

        st = SymbolTable()
        eq = st.str_to_sn("=", 2)
        f = st.str_to_sn("f", 2)
        x, y, z = _make_var(0), _make_var(1), _make_var(2)

        lhs = _make_term(f, (_make_term(f, (x, y)), z))
        rhs = _make_term(f, (x, _make_term(f, (y, z))))
        assoc = _make_unit_eq(eq, lhs, rhs)
        assert eval_clause(assoc, interp, st, eq)


class TestBridgeIsomorphism:
    def test_permuted_z2_isomorphic(self):
        """A permuted Z2 from FiniteModel should be detected as isomorphic."""
        model = _build_z2_model()
        interp = finitemodel_to_interpretation(model)
        permuted = permute_interp(interp, [1, 0])
        assert isomorphic_interps(interp, permuted)


class TestBridgeFormatting:
    def test_standard_format_roundtrip(self):
        """FiniteModel → Interpretation → standard text → reparse should match."""
        model = _build_z3_model()
        interp = finitemodel_to_interpretation(model)
        text = format_interp_standard(interp)
        reparsed = compile_interp_from_text(text)
        for name, op in interp.operations.items():
            assert reparsed.operations[name].values == op.values


class TestBridgeWithTextParse:
    def test_text_parse_matches_bridge(self):
        """Text-based compilation should produce identical result to bridge."""
        Z2_TEXT = """interpretation(2, [number=1, seconds=0], [
  function(f(_,_), [0,1,1,0]),
  function(e, [0])
])."""
        from_text = compile_interp_from_text(Z2_TEXT)
        from_bridge = finitemodel_to_interpretation(_build_z2_model())

        for name in from_text.operations:
            assert from_text.operations[name].values == from_bridge.operations[name].values
            assert from_text.operations[name].arity == from_bridge.operations[name].arity
