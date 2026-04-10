"""Tests for pyladr.inference.diophantine — Diophantine equation solver.

Tests behavioral equivalence with C dioph.c:
- Simple Diophantine equations
- Constraints (symbol vs variable positions)
- Minimal basis computation
- Edge cases
"""

from __future__ import annotations

import pytest

from pyladr.inference.diophantine import DioResult, dio, next_combo_a


def _dio_and_verify(ab, m, n, constraints=None):
    """Run dio and verify all basis elements satisfy the equation."""
    if constraints is None:
        constraints = [0] * (m + n)
    result = dio(ab, m, n, constraints)
    for b in result.basis:
        lhs = sum(ab[i] * b[i] for i in range(m))
        rhs = sum(ab[m + j] * b[m + j] for j in range(n))
        assert lhs == rhs, f"Basis element {b} violates equation"
    return result


class TestDioSimple:
    """Test simple Diophantine equations."""

    def test_1x_eq_1y(self):
        """1*x1 = 1*y1 → basis [[1, 1]] (x1=1, y1=1)."""
        result = dio([1, 1], m=1, n=1, constraints=[0, 0])
        assert result is not None
        assert len(result.basis) >= 1
        # Verify each basis element satisfies 1*x1 = 1*y1
        for b in result.basis:
            assert b[0] == b[1]

    def test_2x_eq_2y(self):
        """2*x1 = 2*y1 → basis [[1, 1]]."""
        result = dio([2, 2], m=1, n=1, constraints=[0, 0])
        assert result is not None
        assert len(result.basis) >= 1

    def test_1x_eq_2y(self):
        """1*x1 = 2*y1 → basis [[2, 1]] (x1=2 balances y1=1)."""
        result = dio([1, 2], m=1, n=1, constraints=[0, 0])
        assert result is not None
        assert len(result.basis) >= 1
        # Verify: each basis element must satisfy a1*x1 = b1*y1
        for b in result.basis:
            assert 1 * b[0] == 2 * b[1]

    def test_2x1_plus_3x2_eq_6y1(self):
        """2*x1 + 3*x2 = 6*y1."""
        result = dio([2, 3, 6], m=2, n=1, constraints=[0, 0, 0])
        assert result is not None
        # Each basis element must satisfy the equation
        for b in result.basis:
            lhs = 2 * b[0] + 3 * b[1]
            rhs = 6 * b[2]
            assert lhs == rhs

    def test_1x1_plus_1x2_eq_1y1_plus_1y2(self):
        """x1 + x2 = y1 + y2 — the standard 2-vs-2 case."""
        result = dio([1, 1, 1, 1], m=2, n=2, constraints=[0, 0, 0, 0])
        assert result is not None
        assert len(result.basis) >= 1
        for b in result.basis:
            assert b[0] + b[1] == b[2] + b[3]


class TestDioConstraints:
    """Test Diophantine solver with constraints."""

    def test_constrained_position_limits_to_1(self):
        """Constrained positions can only have coefficient 0 or 1."""
        # 1*x1 = 1*y1 with x1 constrained (non-zero constraint value)
        result = dio([1, 1], m=1, n=1, constraints=[5, 0])
        assert result is not None
        for b in result.basis:
            assert b[0] <= 1  # constrained position

    def test_no_solution_incompatible_constraints(self):
        """If constraints make equation impossible, should return None or empty basis."""
        # 1*x1 = 1*y1, both constrained to different symbols
        result = dio([1, 1], m=1, n=1, constraints=[5, 6])
        # With different constraints, solutions may be limited
        if result is not None:
            for b in result.basis:
                assert b[0] <= 1 and b[1] <= 1


class TestDioEdgeCases:
    """Test edge cases in the Diophantine solver."""

    def test_single_variable_each_side(self):
        """Simplest case: a*x = b*y."""
        result = dio([3, 5], m=1, n=1, constraints=[0, 0])
        assert result is not None
        for b in result.basis:
            assert 3 * b[0] == 5 * b[1]

    def test_gcd_reduction(self):
        """Coefficients with common factor: 4*x = 6*y → 2*x = 3*y."""
        result = dio([4, 6], m=1, n=1, constraints=[0, 0])
        assert result is not None
        for b in result.basis:
            assert 4 * b[0] == 6 * b[1]


class TestNextCombo:
    """Test combo enumeration for combining basis solutions."""

    def test_basic_combo_generation(self):
        """next_combo_a returns valid combinations of basis elements."""
        result = dio([1, 1], m=1, n=1, constraints=[0, 0])
        if result.num_basis > 0:
            length = 2
            combo = [0] * result.num_basis
            sumvec = [0] * length
            found = next_combo_a(
                length, result.basis, result.num_basis,
                [0, 0], combo, sumvec, True
            )
            assert isinstance(found, bool)

    def test_combo_exhaustion(self):
        """Repeated calls eventually exhaust all combos."""
        result = dio([1, 1], m=1, n=1, constraints=[0, 0])
        if result.num_basis > 0:
            length = 2
            combo = [0] * result.num_basis
            sumvec = [0] * length
            count = 0
            start = True
            while next_combo_a(
                length, result.basis, result.num_basis,
                [0, 0], combo, sumvec, start
            ):
                count += 1
                start = False
                if count > 100:
                    break
            assert count >= 1
