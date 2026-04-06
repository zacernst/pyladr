"""Tests for pyladr.inference.resolution — Binary resolution and factoring.

Tests behavioral equivalence with C resolve.c:
- Binary resolution between two clauses
- Factoring (same-sign literal merging)
- Merge literals (duplicate removal)
- Tautology detection
- Variable renumbering
- Cross-validation against C reference proofs
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, build_binary_term, get_rigid_term, get_variable_term
from pyladr.inference.resolution import (
    all_binary_resolvents,
    binary_resolve,
    factor,
    is_tautology,
    merge_literals,
    renumber_variables,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _atom(symnum: int, *args: Term) -> Term:
    """Build an atomic term P(a1, ..., an)."""
    return get_rigid_term(symnum, len(args), tuple(args))


def _clause(*lits: tuple[bool, Term], clause_id: int = 0) -> Clause:
    """Build a clause from (sign, atom) pairs."""
    return Clause(
        literals=tuple(Literal(sign=s, atom=a) for s, a in lits),
        id=clause_id,
    )


# Symbol numbers for testing
P_SN, Q_SN, R_SN = 1, 2, 3
F_SN, G_SN = 4, 5
A_SN, B_SN, C_SN, E_SN = 6, 7, 8, 9
EQ_SN = 10

# Constants
a = get_rigid_term(A_SN, 0)
b = get_rigid_term(B_SN, 0)
c = get_rigid_term(C_SN, 0)
e = get_rigid_term(E_SN, 0)

# Variables
x = get_variable_term(0)
y = get_variable_term(1)
z = get_variable_term(2)


# ── Binary Resolution tests ─────────────────────────────────────────────────


class TestBinaryResolve:
    """Test binary resolution matching C binary_resolvent() behavior."""

    def test_simple_ground_resolution(self):
        """P(a) resolves with ~P(a) to give empty clause."""
        c1 = _clause((True, _atom(P_SN, a)), clause_id=1)
        c2 = _clause((False, _atom(P_SN, a)), clause_id=2)

        r = binary_resolve(c1, 0, c2, 0)
        assert r is not None
        assert r.is_empty  # empty clause = proof found

    def test_resolution_with_unification(self):
        """P(x) resolves with ~P(a) to give empty clause with {x->a}."""
        c1 = _clause((True, _atom(P_SN, x)), clause_id=1)
        c2 = _clause((False, _atom(P_SN, a)), clause_id=2)

        r = binary_resolve(c1, 0, c2, 0)
        assert r is not None
        assert r.is_empty

    def test_resolution_with_remaining_literals(self):
        """P(x) | Q(x) resolves with ~P(a) to give Q(a)."""
        c1 = _clause(
            (True, _atom(P_SN, x)),
            (True, _atom(Q_SN, x)),
            clause_id=1,
        )
        c2 = _clause((False, _atom(P_SN, a)), clause_id=2)

        r = binary_resolve(c1, 0, c2, 0)
        assert r is not None
        assert r.num_literals == 1
        assert r.literals[0].is_positive
        # Q(a) — x should be replaced by a
        assert r.literals[0].atom.is_ground

    def test_resolution_both_have_remaining(self):
        """P(x) | Q(x) resolves with ~P(a) | R(a) to give Q(a) | R(a)."""
        c1 = _clause(
            (True, _atom(P_SN, x)),
            (True, _atom(Q_SN, x)),
            clause_id=1,
        )
        c2 = _clause(
            (False, _atom(P_SN, a)),
            (True, _atom(R_SN, a)),
            clause_id=2,
        )

        r = binary_resolve(c1, 0, c2, 0)
        assert r is not None
        assert r.num_literals == 2
        # Q(a) and R(a)

    def test_resolution_fails_same_sign(self):
        """Cannot resolve two positive literals."""
        c1 = _clause((True, _atom(P_SN, a)), clause_id=1)
        c2 = _clause((True, _atom(P_SN, a)), clause_id=2)

        r = binary_resolve(c1, 0, c2, 0)
        assert r is None

    def test_resolution_fails_no_unifier(self):
        """P(a) and ~P(b) don't unify."""
        c1 = _clause((True, _atom(P_SN, a)), clause_id=1)
        c2 = _clause((False, _atom(P_SN, b)), clause_id=2)

        r = binary_resolve(c1, 0, c2, 0)
        assert r is None

    def test_resolution_fails_different_predicates(self):
        """P(a) and ~Q(a) don't match."""
        c1 = _clause((True, _atom(P_SN, a)), clause_id=1)
        c2 = _clause((False, _atom(Q_SN, a)), clause_id=2)

        r = binary_resolve(c1, 0, c2, 0)
        assert r is None

    def test_standardization_apart(self):
        """Variables in c1 and c2 are kept separate.

        P(x) | Q(x) resolves with ~P(y) | R(y)
        Variables x and y from different clauses shouldn't conflict.
        Resolvent should be Q(a) | R(a) if x=y=a, or contain renamed vars.
        """
        c1 = _clause(
            (True, _atom(P_SN, x)),
            (True, _atom(Q_SN, x)),
            clause_id=1,
        )
        c2 = _clause(
            (False, _atom(P_SN, y)),
            (True, _atom(R_SN, y)),
            clause_id=2,
        )

        r = binary_resolve(c1, 0, c2, 0)
        assert r is not None
        assert r.num_literals == 2
        # After renumbering, the remaining lits should share variable references
        rn = renumber_variables(r)
        # Q(v0) and R(v0) — both refer to same variable
        v0 = rn.literals[0].atom.arg(0)
        v1 = rn.literals[1].atom.arg(0)
        assert v0.is_variable
        assert v1.is_variable
        assert v0.varnum == v1.varnum

    def test_justification_recorded(self):
        """Resolvent justification records parent clause IDs."""
        c1 = _clause((True, _atom(P_SN, a)), clause_id=5)
        c2 = _clause((False, _atom(P_SN, a)), clause_id=8)

        r = binary_resolve(c1, 0, c2, 0)
        assert r is not None
        assert len(r.justification) == 1
        j = r.justification[0]
        assert j.just_type == JustType.BINARY_RES
        assert j.clause_ids == (5, 8)

    def test_resolution_with_function_terms(self):
        """P(f(x)) resolves with ~P(f(a))."""
        fx = get_rigid_term(F_SN, 1, (x,))
        fa = get_rigid_term(F_SN, 1, (a,))
        c1 = _clause((True, _atom(P_SN, fx)), clause_id=1)
        c2 = _clause((False, _atom(P_SN, fa)), clause_id=2)

        r = binary_resolve(c1, 0, c2, 0)
        assert r is not None
        assert r.is_empty

    def test_resolution_standardization_apart_avoids_occur_check(self):
        """P(x) and ~P(f(x)) succeed because vars are in separate contexts.

        In the two-context approach, x in c1 and x in c2 are independent.
        The unification is x_ctx1 = f(x_ctx2), which is fine — no circularity.
        """
        fx = get_rigid_term(F_SN, 1, (x,))
        c1 = _clause((True, _atom(P_SN, x)), clause_id=1)
        c2 = _clause((False, _atom(P_SN, fx)), clause_id=2)

        r = binary_resolve(c1, 0, c2, 0)
        # Succeeds because standardization apart means different contexts
        assert r is not None
        assert r.is_empty


# ── All binary resolvents ───────────────────────────────────────────────────


class TestAllBinaryResolvents:
    def test_all_resolvents_simple(self):
        """P(a) | Q(b) with ~P(a) | ~Q(b) has two resolvents."""
        c1 = _clause(
            (True, _atom(P_SN, a)),
            (True, _atom(Q_SN, b)),
            clause_id=1,
        )
        c2 = _clause(
            (False, _atom(P_SN, a)),
            (False, _atom(Q_SN, b)),
            clause_id=2,
        )

        rs = all_binary_resolvents(c1, c2)
        # Resolve P with ~P: gives Q(b) | ~Q(b)
        # Resolve Q with ~Q: gives P(a) | ~P(a)
        assert len(rs) == 2

    def test_no_resolvents(self):
        """Two positive clauses produce no resolvents."""
        c1 = _clause((True, _atom(P_SN, a)), clause_id=1)
        c2 = _clause((True, _atom(Q_SN, b)), clause_id=2)

        rs = all_binary_resolvents(c1, c2)
        assert len(rs) == 0


# ── Factoring tests ─────────────────────────────────────────────────────────


class TestFactoring:
    def test_simple_factor(self):
        """P(x) | P(a) factors to P(a) (unify x=a, keep one copy)."""
        c = _clause(
            (True, _atom(P_SN, x)),
            (True, _atom(P_SN, a)),
            clause_id=1,
        )

        fs = factor(c)
        assert len(fs) >= 1
        # At least one factor with one literal
        assert any(f.num_literals == 1 for f in fs)

    def test_no_factor_different_signs(self):
        """P(a) | ~P(a) does not factor (different signs)."""
        c = _clause(
            (True, _atom(P_SN, a)),
            (False, _atom(P_SN, a)),
            clause_id=1,
        )

        fs = factor(c)
        assert len(fs) == 0

    def test_no_factor_different_predicates(self):
        """P(a) | Q(a) does not factor."""
        c = _clause(
            (True, _atom(P_SN, a)),
            (True, _atom(Q_SN, a)),
            clause_id=1,
        )

        fs = factor(c)
        assert len(fs) == 0

    def test_factor_justification(self):
        """Factor records justification."""
        c = _clause(
            (True, _atom(P_SN, x)),
            (True, _atom(P_SN, a)),
            clause_id=3,
        )

        fs = factor(c)
        assert len(fs) >= 1
        j = fs[0].justification[0]
        assert j.just_type == JustType.FACTOR
        assert j.clause_ids == (3,)


# ── Merge literals ──────────────────────────────────────────────────────────


class TestMergeLiterals:
    def test_no_duplicates(self):
        """No duplicates → same clause returned."""
        c = _clause(
            (True, _atom(P_SN, a)),
            (True, _atom(Q_SN, b)),
        )
        m = merge_literals(c)
        assert m.num_literals == 2

    def test_exact_duplicates(self):
        """P(a) | P(a) → P(a)."""
        c = _clause(
            (True, _atom(P_SN, a)),
            (True, _atom(P_SN, a)),
        )
        m = merge_literals(c)
        assert m.num_literals == 1

    def test_different_sign_not_duplicate(self):
        """P(a) | ~P(a) has no duplicates (different sign)."""
        c = _clause(
            (True, _atom(P_SN, a)),
            (False, _atom(P_SN, a)),
        )
        m = merge_literals(c)
        assert m.num_literals == 2

    def test_unit_clause_unchanged(self):
        c = _clause((True, _atom(P_SN, a)))
        m = merge_literals(c)
        assert m.num_literals == 1


# ── Tautology detection ────────────────────────────────────────────────────


class TestTautology:
    def test_simple_tautology(self):
        """P(a) | ~P(a) is a tautology."""
        c = _clause(
            (True, _atom(P_SN, a)),
            (False, _atom(P_SN, a)),
        )
        assert is_tautology(c)

    def test_not_tautology(self):
        """P(a) | Q(a) is not a tautology."""
        c = _clause(
            (True, _atom(P_SN, a)),
            (True, _atom(Q_SN, a)),
        )
        assert not is_tautology(c)

    def test_empty_not_tautology(self):
        """Empty clause is not a tautology."""
        c = Clause()
        assert not is_tautology(c)

    def test_unit_not_tautology(self):
        c = _clause((True, _atom(P_SN, a)))
        assert not is_tautology(c)

    def test_almost_complementary(self):
        """P(a) | ~P(b) is NOT a tautology (different args)."""
        c = _clause(
            (True, _atom(P_SN, a)),
            (False, _atom(P_SN, b)),
        )
        assert not is_tautology(c)


# ── Variable renumbering ───────────────────────────────────────────────────


class TestRenumberVariables:
    def test_already_normalized(self):
        """Clause with v0, v1 stays the same."""
        c = _clause((True, _atom(P_SN, x, y)))
        r = renumber_variables(c)
        assert r.literals[0].atom.arg(0).varnum == 0
        assert r.literals[0].atom.arg(1).varnum == 1

    def test_gap_in_variables(self):
        """Variables v0, v5 renumbered to v0, v1."""
        v5 = get_variable_term(5)
        c = _clause((True, _atom(P_SN, x, v5)))
        r = renumber_variables(c)
        assert r.literals[0].atom.arg(0).varnum == 0
        assert r.literals[0].atom.arg(1).varnum == 1

    def test_large_variable_numbers(self):
        """After resolution, variables may have large numbers from multiplier."""
        v100 = get_variable_term(100)
        v200 = get_variable_term(200)
        c = _clause((True, _atom(P_SN, v100, v200)))
        r = renumber_variables(c)
        assert r.literals[0].atom.arg(0).varnum == 0
        assert r.literals[0].atom.arg(1).varnum == 1

    def test_ground_clause_unchanged(self):
        """Ground clause has no variables to renumber."""
        c = _clause((True, _atom(P_SN, a, b)))
        r = renumber_variables(c)
        assert r.literals[0].atom.arg(0).is_constant
        assert r.literals[0].atom.arg(1).is_constant


# ── Integration: mini proof ─────────────────────────────────────────────────


class TestMiniProof:
    """Test a small resolution proof to verify correctness end-to-end."""

    def test_modus_ponens_style(self):
        """Given P(a) and ~P(x) | Q(x), derive Q(a).

        This is the fundamental resolution step that enables
        modus ponens: from P(a) and P(x)->Q(x), derive Q(a).
        """
        c1 = _clause((True, _atom(P_SN, a)), clause_id=1)
        c2 = _clause(
            (False, _atom(P_SN, x)),
            (True, _atom(Q_SN, x)),
            clause_id=2,
        )

        r = binary_resolve(c1, 0, c2, 0)
        assert r is not None
        assert r.num_literals == 1
        assert r.literals[0].is_positive
        # Should be Q(a) after instantiation
        r = renumber_variables(r)
        assert r.literals[0].atom.is_ground

    def test_two_step_proof(self):
        """P(a), ~P(x)|Q(x), ~Q(a) → derive empty clause in 2 steps."""
        c1 = _clause((True, _atom(P_SN, a)), clause_id=1)
        c2 = _clause(
            (False, _atom(P_SN, x)),
            (True, _atom(Q_SN, x)),
            clause_id=2,
        )
        c3 = _clause((False, _atom(Q_SN, a)), clause_id=3)

        # Step 1: resolve c1 with c2 on P
        r1 = binary_resolve(c1, 0, c2, 0)
        assert r1 is not None
        r1 = renumber_variables(r1)
        r1.id = 4

        # Step 2: resolve r1 (Q(a)) with c3 (~Q(a))
        r2 = binary_resolve(r1, 0, c3, 0)
        assert r2 is not None
        assert r2.is_empty  # Proof complete!
