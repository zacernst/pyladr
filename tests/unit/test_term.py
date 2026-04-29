"""Tests for pyladr.core.term — Term representation.

Tests behavioral equivalence with C term.h/term.c:
- Term construction (variable, constant, complex)
- Term identity and comparison (term_ident)
- C-compatible hash function (hash_term)
- Term copying (copy_term)
- Tree properties (ground, depth, symbol_count, variables)
- String representation (fprint_term)
"""

from __future__ import annotations

import pytest

from pyladr.core.term import (
    Term,
    TermType,
    build_binary_term,
    build_unary_term,
    copy_term,
    get_rigid_term,
    get_variable_term,
)


class TestTermCreation:
    """Test basic term construction and type classification."""

    def test_variable_term(self):
        t = get_variable_term(0)
        assert t.is_variable
        assert not t.is_constant
        assert not t.is_complex
        assert t.term_type == TermType.VARIABLE
        assert t.varnum == 0
        assert t.arity == 0
        assert t.args == ()

    def test_variable_sharing(self):
        """Variables with the same number share the same object (C Shared_variables)."""
        t1 = get_variable_term(5)
        t2 = get_variable_term(5)
        assert t1 is t2

    def test_variable_different_numbers(self):
        t1 = get_variable_term(0)
        t2 = get_variable_term(1)
        assert t1 is not t2
        assert t1.varnum != t2.varnum

    def test_rigid_constant_interning(self):
        """get_rigid_term(sn, 0) returns the same object on repeated calls.

        Memory optimization (cycle 6 T8): arity-0 rigid terms share a
        single Term object per symbol, reducing per-clause footprint.
        """
        c1 = get_rigid_term(42, 0)
        c2 = get_rigid_term(42, 0)
        assert c1 is c2
        assert c1.is_constant
        assert c1.symnum == 42

    def test_complex_terms_not_interned(self):
        """Complex rigid terms (arity > 0) must NOT be cached — args make
        sharing fragile and masks identity-dependent code paths.
        """
        x = get_variable_term(0)
        f1 = get_rigid_term(2, 1, (x,))
        f2 = get_rigid_term(2, 1, (x,))
        assert f1 is not f2
        # Structural equality still holds
        assert f1.term_ident(f2)

    def test_term_has_no_container_field(self):
        """Term.container field was removed in cycle 6 T8 (never used,
        wasted one pointer slot per Term). Direct Term() and factory
        constructors both produce Terms without `container`.
        """
        t_factory = get_rigid_term(5, 0)
        t_direct = Term(private_symbol=-5)
        assert "container" not in Term.__slots__
        assert not hasattr(t_factory, "container")
        assert not hasattr(t_direct, "container")

    def test_constant_term(self):
        t = get_rigid_term(1, 0)
        assert t.is_constant
        assert not t.is_variable
        assert not t.is_complex
        assert t.term_type == TermType.CONSTANT
        assert t.symnum == 1

    def test_complex_term(self):
        x = get_variable_term(0)
        y = get_variable_term(1)
        f = get_rigid_term(2, 2, (x, y))
        assert f.is_complex
        assert not f.is_variable
        assert not f.is_constant
        assert f.term_type == TermType.COMPLEX
        assert f.symnum == 2
        assert f.arity == 2
        assert f.arg(0) is x
        assert f.arg(1) is y

    def test_nested_term(self):
        """f(g(x), a) where f=sn1, g=sn2, a=sn3, x=v0."""
        x = get_variable_term(0)
        a = get_rigid_term(3, 0)
        gx = get_rigid_term(2, 1, (x,))
        fga = get_rigid_term(1, 2, (gx, a))
        assert fga.is_complex
        assert fga.arg(0).is_complex
        assert fga.arg(1).is_constant

    def test_arity_mismatch_raises(self):
        with pytest.raises(ValueError, match="Arity"):
            Term(private_symbol=-1, arity=2, args=())

    def test_negative_varnum_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            get_variable_term(-1)

    def test_zero_symnum_raises(self):
        with pytest.raises(ValueError, match="positive"):
            get_rigid_term(0, 0)

    def test_symnum_on_variable_raises(self):
        t = get_variable_term(0)
        with pytest.raises(ValueError, match="SYMNUM"):
            _ = t.symnum

    def test_varnum_on_constant_raises(self):
        t = get_rigid_term(1, 0)
        with pytest.raises(ValueError, match="VARNUM"):
            _ = t.varnum


class TestTermProperties:
    """Test term tree properties."""

    def test_ground_variable(self):
        assert not get_variable_term(0).is_ground

    def test_ground_constant(self):
        assert get_rigid_term(1, 0).is_ground

    def test_ground_complex_with_var(self):
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        f = get_rigid_term(2, 2, (x, a))
        assert not f.is_ground

    def test_ground_complex_without_var(self):
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        f = get_rigid_term(3, 2, (a, b))
        assert f.is_ground

    def test_depth_constant(self):
        assert get_rigid_term(1, 0).depth == 0

    def test_depth_variable(self):
        assert get_variable_term(0).depth == 0

    def test_depth_complex(self):
        x = get_variable_term(0)
        gx = get_rigid_term(2, 1, (x,))
        fgx = get_rigid_term(1, 1, (gx,))
        assert gx.depth == 1
        assert fgx.depth == 2

    def test_symbol_count(self):
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        gx = get_rigid_term(2, 1, (x,))
        fgxa = get_rigid_term(3, 2, (gx, a))
        assert x.symbol_count == 1
        assert a.symbol_count == 1
        assert gx.symbol_count == 2
        assert fgxa.symbol_count == 4

    def test_biggest_variable(self):
        x = get_variable_term(3)
        y = get_variable_term(7)
        f = get_rigid_term(1, 2, (x, y))
        assert f.biggest_variable() == 7

    def test_biggest_variable_ground(self):
        a = get_rigid_term(1, 0)
        assert a.biggest_variable() == -1

    def test_variables_set(self):
        x = get_variable_term(0)
        y = get_variable_term(2)
        f = get_rigid_term(1, 2, (x, y))
        g = get_rigid_term(2, 2, (f, x))
        assert g.variables() == {0, 2}

    def test_occurs_in(self):
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        f = get_rigid_term(2, 2, (x, a))
        assert x.occurs_in(f)
        assert not a.occurs_in(get_rigid_term(3, 0))


class TestTermIdentity:
    """Test structural identity (C term_ident)."""

    def test_same_variable(self):
        assert get_variable_term(0).term_ident(get_variable_term(0))

    def test_different_variables(self):
        assert not get_variable_term(0).term_ident(get_variable_term(1))

    def test_same_constant(self):
        a1 = get_rigid_term(1, 0)
        a2 = get_rigid_term(1, 0)
        assert a1.term_ident(a2)

    def test_different_constants(self):
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        assert not a.term_ident(b)

    def test_same_complex(self):
        x = get_variable_term(0)
        f1 = get_rigid_term(1, 1, (x,))
        f2 = get_rigid_term(1, 1, (x,))
        assert f1.term_ident(f2)

    def test_different_args(self):
        x = get_variable_term(0)
        y = get_variable_term(1)
        f1 = get_rigid_term(1, 1, (x,))
        f2 = get_rigid_term(1, 1, (y,))
        assert not f1.term_ident(f2)

    def test_variable_vs_constant(self):
        assert not get_variable_term(0).term_ident(get_rigid_term(1, 0))

    def test_complex_arg_order_matters(self):
        """f(a, b) != f(b, a)."""
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        f1 = get_rigid_term(3, 2, (a, b))
        f2 = get_rigid_term(3, 2, (b, a))
        assert not f1.term_ident(f2)


class TestTermHash:
    """Test C-compatible hash function."""

    def test_variable_hash(self):
        """Variables hash to their variable number."""
        assert get_variable_term(0).c_hash() == 0
        assert get_variable_term(42).c_hash() == 42

    def test_constant_hash(self):
        """Constants hash to their symbol number."""
        assert get_rigid_term(5, 0).c_hash() == 5

    def test_hash_deterministic(self):
        x = get_variable_term(0)
        f = get_rigid_term(1, 1, (x,))
        h1 = f.c_hash()
        h2 = f.c_hash()
        assert h1 == h2

    def test_hash_structural(self):
        """Different structures produce different hashes."""
        x = get_variable_term(0)
        y = get_variable_term(1)
        f1 = get_rigid_term(1, 1, (x,))
        f2 = get_rigid_term(1, 1, (y,))
        assert f1.c_hash() != f2.c_hash()

    def test_hash_matches_c_algorithm(self):
        """Verify hash matches C: x = SYMNUM(t); x = (x<<3) ^ hash(arg)."""
        # f(v0) where f has symnum=3
        # C: x = 3; x = (3 << 3) ^ hash(v0) = 24 ^ 0 = 24
        x = get_variable_term(0)
        f = get_rigid_term(3, 1, (x,))
        assert f.c_hash() == (3 << 3) ^ 0  # 24

    def test_hash_nested(self):
        """g(f(v0)) where g=sn1, f=sn2: hash = (1<<3) ^ ((2<<3) ^ 0) = 8 ^ 16 = 24."""
        x = get_variable_term(0)
        f = get_rigid_term(2, 1, (x,))
        g = get_rigid_term(1, 1, (f,))
        expected = (1 << 3) ^ ((2 << 3) ^ 0)
        assert g.c_hash() == expected


class TestCopyTerm:
    def test_copy_variable(self):
        x = get_variable_term(0)
        c = copy_term(x)
        assert c is x  # Variables are shared

    def test_copy_constant(self):
        a = get_rigid_term(1, 0)
        c = copy_term(a)
        assert c is not a
        assert c.term_ident(a)

    def test_copy_complex(self):
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        f = get_rigid_term(2, 2, (x, a))
        c = copy_term(f)
        assert c is not f
        assert c.term_ident(f)
        assert c.arg(0) is x  # Shared variable


class TestTermString:
    def test_variable_str(self):
        assert get_variable_term(0).to_str() == "v0"
        assert get_variable_term(42).to_str() == "v42"

    def test_constant_str_no_table(self):
        assert get_rigid_term(1, 0).to_str() == "s1"

    def test_complex_str_no_table(self):
        x = get_variable_term(0)
        f = get_rigid_term(1, 2, (x, get_rigid_term(2, 0)))
        assert f.to_str() == "s1(v0,s2)"


class TestBuildHelpers:
    def test_build_binary(self):
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        f = build_binary_term(3, a, b)
        assert f.arity == 2
        assert f.symnum == 3
        assert f.arg(0) is a
        assert f.arg(1) is b

    def test_build_unary(self):
        x = get_variable_term(0)
        f = build_unary_term(1, x)
        assert f.arity == 1
        assert f.symnum == 1
        assert f.arg(0) is x
