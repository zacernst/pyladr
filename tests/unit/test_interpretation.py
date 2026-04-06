"""Tests for PyLADR interpretation evaluation system (Phase B).

Tests the core interpretation infrastructure: compilation, evaluation,
isomorphism checking, and output formatting.
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.interpretation import (
    Interpretation,
    OperationTable,
    SemanticsResult,
    TableType,
    compare_interp,
    compile_interp_from_text,
    copy_interp,
    eval_clause,
    eval_clause_false_instances,
    eval_clause_true_instances,
    factorial,
    format_interp_cooked,
    format_interp_portable,
    format_interp_raw,
    format_interp_standard,
    format_interp_standard2,
    format_interp_tabular,
    format_interp_tex,
    format_interp_xml,
    ident_interp,
    ident_interp_perm,
    int_power,
    isomorphic_interps,
    normal_interp,
    permute_interp,
    perms_required,
)
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term


# ── Fixtures ─────────────────────────────────────────────────────────────────

# A group of order 2 (Z2): f(x,y) = x XOR y, identity = 0
Z2_TEXT = """
interpretation(2, [number=1, seconds=0], [
  function(f(_,_), [0,1,1,0]),
  function(e, [0])
]).
"""

# Cyclic group of order 3 (Z3): f(x,y) = (x+y) mod 3
Z3_TEXT = """
interpretation(3, [number=1, seconds=0], [
  function(f(_,_), [0,1,2,1,2,0,2,0,1])
]).
"""

# Interpretation with a relation
Z2_WITH_REL = """
interpretation(2, [number=1, seconds=0], [
  function(f(_,_), [0,1,1,0]),
  relation(R(_,_), [1,0,0,1])
]).
"""

# Klein 4-group V4
V4_TEXT = """
interpretation(4, [number=1, seconds=0], [
  function(f(_,_), [0,1,2,3,1,0,3,2,2,3,0,1,3,2,1,0])
]).
"""


@pytest.fixture
def z2_interp():
    return compile_interp_from_text(Z2_TEXT)


@pytest.fixture
def z3_interp():
    return compile_interp_from_text(Z3_TEXT)


@pytest.fixture
def z2_rel_interp():
    return compile_interp_from_text(Z2_WITH_REL)


@pytest.fixture
def symbol_table():
    return SymbolTable()


def _make_term(sym_id: int, args: tuple[Term, ...] = ()) -> Term:
    return Term(private_symbol=-sym_id, arity=len(args), args=args)


def _make_var(n: int) -> Term:
    from pyladr.core.term import get_variable_term
    return get_variable_term(n)


def _make_unit_eq(eq_id: int, lhs: Term, rhs: Term) -> Clause:
    atom = Term(private_symbol=-eq_id, arity=2, args=(lhs, rhs))
    lit = Literal(sign=True, atom=atom)
    return Clause(literals=(lit,))


# ── Compilation Tests ────────────────────────────────────────────────────────


class TestCompilation:
    """Tests for compile_interp_from_text."""

    def test_compile_z2(self, z2_interp):
        assert z2_interp.size == 2
        assert "f" in z2_interp.operations
        assert "e" in z2_interp.operations

    def test_compile_z2_function_table(self, z2_interp):
        f = z2_interp.get_table("f", 2)
        assert f is not None
        assert f.values == [0, 1, 1, 0]
        assert f.table_type == TableType.FUNCTION

    def test_compile_z2_constant(self, z2_interp):
        e = z2_interp.get_table("e", 0)
        assert e is not None
        assert e.values == [0]

    def test_compile_z3(self, z3_interp):
        assert z3_interp.size == 3
        f = z3_interp.get_table("f", 2)
        assert f is not None
        assert len(f.values) == 9

    def test_compile_relation(self, z2_rel_interp):
        r = z2_rel_interp.get_table("R", 2)
        assert r is not None
        assert r.table_type == TableType.RELATION
        assert r.values == [1, 0, 0, 1]

    def test_compile_bad_size(self):
        with pytest.raises(ValueError, match="Domain size"):
            compile_interp_from_text("interpretation(0, [], [])")

    def test_compile_bad_format(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            compile_interp_from_text("not an interpretation")

    def test_compile_wrong_values_count(self):
        with pytest.raises(ValueError, match="expected"):
            compile_interp_from_text(
                "interpretation(2, [], [function(f(_,_), [0,1,1])])"
            )

    def test_compile_function_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            compile_interp_from_text(
                "interpretation(2, [], [function(f(_,_), [0,1,1,5])])"
            )

    def test_compile_relation_bad_value(self):
        with pytest.raises(ValueError, match="not in"):
            compile_interp_from_text(
                "interpretation(2, [], [relation(R(_,_), [0,1,1,2])])"
            )

    def test_compile_incomplete(self):
        interp = compile_interp_from_text(
            "interpretation(2, [], [function(f(_), [-,1])])",
            allow_incomplete=True,
        )
        assert interp.incomplete
        assert interp.get_table("f", 1).values == [-1, 1]

    def test_compile_incomplete_not_allowed(self):
        with pytest.raises(ValueError, match="Undefined"):
            compile_interp_from_text(
                "interpretation(2, [], [function(f(_), [-,1])])"
            )


# ── Table Value Tests ────────────────────────────────────────────────────────


class TestTableValue:
    """Tests for direct table lookups."""

    def test_constant_lookup(self, z2_interp):
        assert z2_interp.table_value("e", 0) == 0

    def test_binary_lookup(self, z2_interp):
        # f(0,0) = 0, f(0,1) = 1, f(1,0) = 1, f(1,1) = 0
        assert z2_interp.table_value("f", 2, 0, 0) == 0
        assert z2_interp.table_value("f", 2, 0, 1) == 1
        assert z2_interp.table_value("f", 2, 1, 0) == 1
        assert z2_interp.table_value("f", 2, 1, 1) == 0

    def test_z3_commutativity(self, z3_interp):
        for i in range(3):
            for j in range(3):
                assert z3_interp.table_value("f", 2, i, j) == z3_interp.table_value(
                    "f", 2, j, i
                )

    def test_missing_symbol(self, z2_interp):
        with pytest.raises(ValueError, match="not in"):
            z2_interp.table_value("g", 2, 0, 0)


# ── Evaluation Tests ─────────────────────────────────────────────────────────


class TestEvaluation:
    """Tests for clause evaluation in interpretations."""

    def test_eval_ground_equality_true(self, z2_interp, symbol_table):
        """f(0,1) = 1 should be true in Z2."""
        eq = symbol_table.str_to_sn("=", 2)
        f = symbol_table.str_to_sn("f", 2)
        zero = symbol_table.str_to_sn("0", 0)
        one = symbol_table.str_to_sn("1", 0)

        lhs = _make_term(f, (_make_term(zero), _make_term(one)))
        rhs = _make_term(one)
        clause = _make_unit_eq(eq, lhs, rhs)
        assert eval_clause(clause, z2_interp, symbol_table, eq)

    def test_eval_ground_equality_false(self, z2_interp, symbol_table):
        """f(0,1) = 0 should be false in Z2."""
        eq = symbol_table.str_to_sn("=", 2)
        f = symbol_table.str_to_sn("f", 2)
        zero = symbol_table.str_to_sn("0", 0)
        one = symbol_table.str_to_sn("1", 0)

        lhs = _make_term(f, (_make_term(zero), _make_term(one)))
        rhs = _make_term(zero)
        clause = _make_unit_eq(eq, lhs, rhs)
        assert not eval_clause(clause, z2_interp, symbol_table, eq)

    def test_eval_associativity_z2(self, z2_interp, symbol_table):
        """f(f(x,y),z) = f(x,f(y,z)) should hold in Z2."""
        eq = symbol_table.str_to_sn("=", 2)
        f = symbol_table.str_to_sn("f", 2)

        x, y, z = _make_var(0), _make_var(1), _make_var(2)
        lhs = _make_term(f, (_make_term(f, (x, y)), z))
        rhs = _make_term(f, (x, _make_term(f, (y, z))))
        clause = _make_unit_eq(eq, lhs, rhs)
        assert eval_clause(clause, z2_interp, symbol_table, eq)

    def test_eval_commutativity_z3(self, z3_interp, symbol_table):
        """f(x,y) = f(y,x) should hold in Z3."""
        eq = symbol_table.str_to_sn("=", 2)
        f = symbol_table.str_to_sn("f", 2)

        x, y = _make_var(0), _make_var(1)
        lhs = _make_term(f, (x, y))
        rhs = _make_term(f, (y, x))
        clause = _make_unit_eq(eq, lhs, rhs)
        assert eval_clause(clause, z3_interp, symbol_table, eq)

    def test_eval_false_law_z3(self, z3_interp, symbol_table):
        """f(x,x) = x (idempotency) should NOT hold in Z3."""
        eq = symbol_table.str_to_sn("=", 2)
        f = symbol_table.str_to_sn("f", 2)

        x = _make_var(0)
        lhs = _make_term(f, (x, x))
        rhs = x
        clause = _make_unit_eq(eq, lhs, rhs)
        # f(1,1) = 2 != 1, so this is false
        assert not eval_clause(clause, z3_interp, symbol_table, eq)

    def test_eval_empty_clause(self, z2_interp):
        """Empty clause ($F) should be false."""
        clause = Clause(literals=())
        assert not eval_clause(clause, z2_interp)

    def test_eval_true_instances_count(self, z3_interp, symbol_table):
        """f(x,x) = x has some true instances in Z3 (just x=0)."""
        eq = symbol_table.str_to_sn("=", 2)
        f = symbol_table.str_to_sn("f", 2)

        x = _make_var(0)
        lhs = _make_term(f, (x, x))
        rhs = x
        clause = _make_unit_eq(eq, lhs, rhs)
        true_count = eval_clause_true_instances(clause, z3_interp, symbol_table, eq)
        false_count = eval_clause_false_instances(clause, z3_interp, symbol_table, eq)
        assert true_count + false_count == 3  # domain size = 3, 1 variable
        assert true_count >= 1  # at least x=0

    def test_eval_all_true_instances(self, z2_interp, symbol_table):
        """f(x,y) = f(y,x) should have 4/4 true instances in Z2."""
        eq = symbol_table.str_to_sn("=", 2)
        f = symbol_table.str_to_sn("f", 2)

        x, y = _make_var(0), _make_var(1)
        lhs = _make_term(f, (x, y))
        rhs = _make_term(f, (y, x))
        clause = _make_unit_eq(eq, lhs, rhs)
        assert eval_clause_true_instances(clause, z2_interp, symbol_table, eq) == 4
        assert eval_clause_false_instances(clause, z2_interp, symbol_table, eq) == 0


# ── Isomorphism Tests ────────────────────────────────────────────────────────


class TestIsomorphism:
    """Tests for isomorphism checking."""

    def test_ident_interp_same(self, z2_interp):
        z2_copy = copy_interp(z2_interp)
        assert ident_interp(z2_interp, z2_copy)

    def test_ident_interp_different(self, z2_interp, z3_interp):
        assert not ident_interp(z2_interp, z3_interp)

    def test_ident_interp_perm_identity(self, z2_interp):
        z2_copy = copy_interp(z2_interp)
        perm = [0, 1]
        assert ident_interp_perm(z2_interp, z2_copy, perm)

    def test_permute_interp_identity(self, z2_interp):
        perm = [0, 1]
        permuted = permute_interp(z2_interp, perm)
        assert ident_interp(z2_interp, permuted)

    def test_permute_interp_swap(self, z2_interp):
        """Swapping domain elements 0<->1 in Z2 should give an isomorphic model."""
        perm = [1, 0]
        permuted = permute_interp(z2_interp, perm)
        # Permuted is isomorphic but NOT identical (e maps to 1 instead of 0)
        assert not ident_interp(z2_interp, permuted)
        assert isomorphic_interps(z2_interp, permuted)

    def test_isomorphic_self(self, z2_interp):
        assert isomorphic_interps(z2_interp, z2_interp)

    def test_isomorphic_copy(self, z2_interp):
        assert isomorphic_interps(z2_interp, copy_interp(z2_interp))

    def test_isomorphic_permuted(self, z3_interp):
        """A permutation of Z3 should be isomorphic to Z3."""
        perm = [1, 2, 0]  # rotate
        permuted = permute_interp(z3_interp, perm)
        assert isomorphic_interps(z3_interp, permuted)

    def test_not_isomorphic_different_size(self, z2_interp, z3_interp):
        assert not isomorphic_interps(z2_interp, z3_interp)

    def test_not_isomorphic_different_structure(self):
        """Z4 is not isomorphic to V4."""
        z4 = compile_interp_from_text(
            "interpretation(4, [], ["
            "function(f(_,_), [0,1,2,3,1,2,3,0,2,3,0,1,3,0,1,2])"
            "])"
        )
        v4 = compile_interp_from_text(V4_TEXT)
        assert not isomorphic_interps(z4, v4)

    def test_normal_interp(self, z2_interp):
        """Normalization should produce an isomorphic interpretation."""
        normed = normal_interp(z2_interp)
        assert isomorphic_interps(z2_interp, normed)

    def test_compare_interp_equal(self, z2_interp):
        assert compare_interp(z2_interp, copy_interp(z2_interp)) == 0

    def test_compare_interp_size(self, z2_interp, z3_interp):
        assert compare_interp(z2_interp, z3_interp) < 0
        assert compare_interp(z3_interp, z2_interp) > 0

    def test_perms_required_no_blocks(self, z3_interp):
        """Without blocks, need 3! = 6 permutations."""
        assert perms_required(z3_interp) == 6


# ── Formatting Tests ─────────────────────────────────────────────────────────


class TestFormatting:
    """Tests for interpretation formatters."""

    def test_format_standard(self, z2_interp):
        out = format_interp_standard(z2_interp)
        assert "interpretation( 2" in out
        assert "function" in out
        assert "])." in out

    def test_format_standard_roundtrip(self, z2_interp):
        """Standard format should reparse to identical interpretation."""
        out = format_interp_standard(z2_interp)
        reparsed = compile_interp_from_text(out)
        assert ident_interp(z2_interp, reparsed)

    def test_format_standard2(self, z2_interp):
        out = format_interp_standard2(z2_interp)
        assert "interpretation( 2" in out

    def test_format_portable(self, z2_interp):
        out = format_interp_portable(z2_interp)
        assert "[" in out
        assert '"f"' in out

    def test_format_tabular(self, z2_interp):
        out = format_interp_tabular(z2_interp)
        assert "function" in out
        assert "|" in out

    def test_format_cooked(self, z2_interp):
        out = format_interp_cooked(z2_interp)
        assert "f(0,0) = 0." in out
        assert "f(0,1) = 1." in out

    def test_format_cooked_relation(self, z2_rel_interp):
        out = format_interp_cooked(z2_rel_interp)
        assert "R(0,0)." in out
        assert "R(1,1)." in out
        # R(0,1) and R(1,0) are 0, should NOT appear
        assert "R(0,1)" not in out

    def test_format_raw(self, z2_interp):
        out = format_interp_raw(z2_interp)
        assert "function" in out

    def test_format_tex(self, z2_interp):
        out = format_interp_tex(z2_interp)
        assert "\\begin{tabular}" in out
        assert "\\hline" in out

    def test_format_xml(self, z2_interp):
        out = format_interp_xml(z2_interp)
        assert '<interp size="2">' in out
        assert "</interp>" in out

    def test_format_xml_relation(self, z2_rel_interp):
        out = format_interp_xml(z2_rel_interp)
        assert 'type="relation"' in out
        assert 'type="function"' in out


# ── Utility Tests ────────────────────────────────────────────────────────────


class TestUtilities:
    def test_int_power(self):
        assert int_power(2, 0) == 1
        assert int_power(2, 10) == 1024
        assert int_power(3, 3) == 27

    def test_factorial(self):
        assert factorial(0) == 1
        assert factorial(1) == 1
        assert factorial(5) == 120

    def test_occurrences_tracking(self, z2_interp):
        """Occurrence counts should be populated from function tables."""
        assert len(z2_interp.occurrences) == 2
        # f: [0,1,1,0] and e: [0] — 0 appears 3 times, 1 appears 2 times
        assert z2_interp.occurrences[0] == 3
        assert z2_interp.occurrences[1] == 2
