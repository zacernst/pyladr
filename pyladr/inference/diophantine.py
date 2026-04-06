"""Diophantine equation solver for AC unification matching C dioph.c.

Solves linear homogeneous Diophantine equations of the form:
    a1*x1 + ... + am*xm = b1*y1 + ... + bn*yn

where xi, yj are non-negative integers.

Uses Huet's algorithm (Information Processing Letters 7(3) 1978) to find
the minimal basis of solutions. Combinations of basis solutions give
the complete set of AC unifiers.

== Usage ==

    # Solve a1*x1 + a2*x2 = b1*y1 + b2*y2
    ab = [a1, a2, b1, b2]  # coefficients
    constraints = [0, 0, 0, 0]  # 0=variable, else symbol number
    result = dio(ab, m=2, n=2, constraints=constraints)
    # result.basis contains minimal solutions
    # Use next_combo_a() to enumerate subsets
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd

MAX_COEF = 250
MAX_BASIS = 100


def lcm(x: int, y: int) -> int:
    """Least common multiple."""
    return (x * y) // gcd(x, y)


def _less_vec(a1: list[int], a2: list[int], length: int) -> bool:
    """True iff each component of a1 <= corresponding component of a2."""
    return all(a1[i] <= a2[i] for i in range(length))


def _var_check_1(constraints: list[int], xy: list[int], start: int, stop: int) -> bool:
    """Check: constrained positions must have coefficient <= 1.

    Otherwise an AC symbol would have to unify with another rigid symbol.
    """
    return all(
        not (constraints[i] and xy[i] > 1)
        for i in range(start, stop + 1)
    )


def _var_check_2(constraints: list[int], xy: list[int], start: int, stop: int) -> bool:
    """Check: all constrained non-zero positions must have the same constraint.

    Otherwise a variable would have to unify with 2 different symbols.
    """
    first_con = 0
    for i in range(start, stop + 1):
        if constraints[i] and xy[i]:
            if first_con == 0:
                first_con = constraints[i]
            elif constraints[i] != first_con:
                return False
    return True


@dataclass
class DioResult:
    """Result of Diophantine equation solving."""

    basis: list[list[int]] = field(default_factory=list)
    num_basis: int = 0
    status: int = 1  # 0=no solution, 1=ok, -1=too many


def dio(
    ab: list[int],
    m: int,
    n: int,
    constraints: list[int],
) -> DioResult:
    """Solve a linear homogeneous Diophantine equation.

    Matching C dio(). Uses Huet's algorithm.

    The equation: a1*x1 + ... + am*xm = b1*y1 + ... + bn*yn
    where ab = [a1,...,am, b1,...,bn].

    Args:
        ab: Coefficient vector. ab[0..m-1] are left, ab[m..m+n-1] are right.
        m: Number of left-side coefficients.
        n: Number of right-side coefficients.
        constraints: Type constraints (0=bindable variable, else symbol number).

    Returns:
        DioResult with basis of minimal solutions.
    """
    result = DioResult()

    if m == 0 or n == 0:
        return result

    length = m + n

    # Precompute d[i][j] = lcm(a_i, b_j) / a_i
    #           e[i][j] = lcm(a_i, b_j) / b_j
    d = [[0] * length for _ in range(length)]
    e = [[0] * length for _ in range(length)]

    for i in range(m):
        for j in range(m, m + n):
            a, b = ab[i], ab[j]
            t = lcm(a, b)
            d[i][j] = t // a
            e[i][j] = t // b

    max_a = max(ab[i] for i in range(m))
    max_b = max(ab[j] for j in range(m, m + n))

    xy = [0] * length
    max_y = [0] * length

    # Search for basis solutions — a-side
    xypos = m - 1
    go_a = True
    suma = 0

    while go_a:
        xy[xypos] += 1
        suma += ab[xypos]

        # Check a-side bounds
        in_bounds = True
        if xy[xypos] > max_b:  # Huet's (a)
            in_bounds = False
        elif not _var_check_1(constraints, xy, 0, m - 1):
            in_bounds = False
        elif not _var_check_2(constraints, xy, 0, m - 1):
            in_bounds = False
        else:
            # Build max_y vector
            for j in range(m, m + n):
                max_y[j] = max_a
                for i in range(m):
                    if xy[i] >= d[i][j]:
                        f = e[i][j] - 1
                        if f < max_y[j]:
                            max_y[j] = f
            bsum = sum(ab[j] * max_y[j] for j in range(m, m + n))
            if suma > bsum:  # Huet's (b)
                in_bounds = False

        if in_bounds:
            # Search b-side
            sumb = 0
            xypos = m + n - 1
            go_b = True
            while go_b:
                xy[xypos] += 1
                sumb += ab[xypos]

                # Check b-side bounds
                b_ok = (
                    sumb <= suma
                    and xy[xypos] <= max_y[xypos]
                    and _var_check_1(constraints, xy, 0, m + n - 1)
                    and _var_check_2(constraints, xy, 0, m + n - 1)
                )

                if b_ok and suma == sumb:
                    # Found a solution — add if minimal
                    if not _add_solution(xy, length, result):
                        result.status = -1
                        return result
                    backup = True
                elif b_ok:
                    backup = False
                else:
                    backup = True

                if backup:
                    sumb -= xy[xypos] * ab[xypos]
                    xy[xypos] = 0
                    xypos -= 1
                    if xypos < m:
                        go_b = False
                else:
                    xypos = m + n - 1

            xypos = m - 1
        else:
            suma -= xy[xypos] * ab[xypos]
            xy[xypos] = 0
            xypos -= 1
            if xypos < 0:
                go_a = False

    # Add special solutions S_ij
    for i in range(length):
        xy[i] = 0
    for i in range(m):
        for j in range(m, m + n):
            xy[i] = d[i][j]
            xy[j] = e[i][j]
            if (_var_check_1(constraints, xy, 0, m + n - 1)
                    and _var_check_2(constraints, xy, 0, m + n - 1)):
                if not _add_solution(xy, length, result):
                    result.status = -1
                    return result
            xy[i] = 0
            xy[j] = 0

    if result.num_basis == 0:
        result.status = 0

    return result


def _add_solution(xy: list[int], length: int, result: DioResult) -> bool:
    """Add solution if not dominated by existing basis solution.

    Matching C add_solution(). Returns False if too many solutions.
    """
    for i in range(result.num_basis):
        if _less_vec(result.basis[i], xy, length):
            return True  # Dominated — skip

    if result.num_basis >= MAX_BASIS:
        return False

    result.basis.append(xy[:length])
    result.num_basis += 1
    return True


def next_combo_a(
    length: int,
    basis: list[list[int]],
    num_basis: int,
    constraints: list[int],
    combo: list[int],
    sumvec: list[int],
    start_flag: bool,
) -> bool:
    """Generate next valid subset of basis solutions.

    Matching C next_combo_a() — Algorithm A.
    Enumerates subsets of basis solutions where each variable gets
    instantiated and there are no rigid symbol clashes.

    Args:
        length: Total number of coefficients (m+n).
        basis: Basis solutions from dio().
        num_basis: Number of basis solutions.
        constraints: Rigid symbol constraints.
        combo: Current subset (modified in place). Binary vector.
        sumvec: Sum of selected basis solutions (modified in place).
        start_flag: True for first call, False for subsequent.

    Returns:
        True if a valid combination was found, False if exhausted.
    """
    if start_flag:
        for i in range(length):
            sumvec[i] = 0
        for i in range(num_basis):
            combo[i] = 0
        pos = 0
    else:
        # Backtrack from the end
        pos = num_basis - 1
        while pos >= 0 and combo[pos] == 0:
            pos -= 1
        if pos < 0:
            return False
        # Remove current selection and advance
        combo[pos] = 0
        for j in range(length):
            sumvec[j] -= basis[pos][j]
        pos += 1

    go = True
    while go:
        if pos >= num_basis:
            # Check if this is a valid combination
            if _valid_combo(sumvec, length, constraints):
                return True
            # Backtrack
            pos -= 1
            while pos >= 0 and combo[pos] == 0:
                pos -= 1
            if pos < 0:
                return False
            combo[pos] = 0
            for j in range(length):
                sumvec[j] -= basis[pos][j]
            pos += 1
        else:
            # Try including basis[pos]
            combo[pos] = 1
            for j in range(length):
                sumvec[j] += basis[pos][j]
            pos += 1

    return False


def _valid_combo(sumvec: list[int], length: int, constraints: list[int]) -> bool:
    """Check if a combination is valid: all positions non-zero, constraints satisfied."""
    # Every coefficient must be non-zero (each variable gets a value)
    if any(sumvec[i] == 0 for i in range(length)):
        return False
    return (
        _var_check_1(constraints, sumvec, 0, length - 1)
        and _var_check_2(constraints, sumvec, 0, length - 1)
    )
