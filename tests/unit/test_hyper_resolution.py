"""Tests for pyladr.inference.hyper_resolution — Positive hyper-resolution.

Tests behavioral equivalence with C hyper_res.c:
- Basic hyper-resolution with single negative literal (degenerates to binary)
- Multi-literal hyper-resolution (multiple negative literals resolved)
- All-positive result constraint
- Given clause as nucleus and as satellite
- Variable unification across multiple satellites
- Integration with search engine via SearchOptions.hyper_resolution
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.inference.hyper_resolution import (
    all_hyper_resolvents,
    hyper_resolve,
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


# Symbol numbers
P_SN, Q_SN, R_SN = 1, 2, 3
F_SN = 4
A_SN, B_SN, C_SN = 6, 7, 8

# Constants
a = get_rigid_term(A_SN, 0)
b = get_rigid_term(B_SN, 0)
c = get_rigid_term(C_SN, 0)

# Variables
x = get_variable_term(0)
y = get_variable_term(1)
z = get_variable_term(2)


# ── hyper_resolve tests ──────────────────────────────────────────────────────


class TestHyperResolve:
    """Test hyper_resolve() core algorithm."""

    def test_no_negative_literals(self):
        """Nucleus with no negative literals produces no resolvents."""
        nuc = _clause((True, _atom(P_SN, a)), clause_id=1)
        sats = [_clause((True, _atom(Q_SN, b)), clause_id=2)]
        assert hyper_resolve(nuc, sats) == []

    def test_single_neg_literal_ground(self):
        """Single negative literal: -P(a) | Q(b) resolved with P(a) → Q(b)."""
        nuc = _clause(
            (False, _atom(P_SN, a)),
            (True, _atom(Q_SN, b)),
            clause_id=1,
        )
        sat = _clause((True, _atom(P_SN, a)), clause_id=2)
        results = hyper_resolve(nuc, [sat])

        assert len(results) == 1
        r = results[0]
        assert len(r.literals) == 1
        assert r.literals[0].sign is True  # Q(b)
        # Justification references nucleus and satellite
        assert r.justification[0].just_type == JustType.HYPER_RES
        assert 1 in r.justification[0].clause_ids
        assert 2 in r.justification[0].clause_ids

    def test_single_neg_literal_with_unification(self):
        """Single negative literal with variable: -P(x) | Q(x) + P(a) → Q(a)."""
        nuc = _clause(
            (False, _atom(P_SN, x)),
            (True, _atom(Q_SN, x)),
            clause_id=1,
        )
        sat = _clause((True, _atom(P_SN, a)), clause_id=2)
        results = hyper_resolve(nuc, [sat])

        assert len(results) == 1
        r = results[0]
        assert len(r.literals) == 1
        assert r.literals[0].sign is True
        # Q(a) — the variable x should be instantiated to a
        assert r.literals[0].atom.args[0].is_constant or not r.literals[0].atom.args[0].is_variable

    def test_two_neg_literals_ground(self):
        """-P(a) | -Q(b) | R(c) resolved with P(a) and Q(b) → R(c)."""
        nuc = _clause(
            (False, _atom(P_SN, a)),
            (False, _atom(Q_SN, b)),
            (True, _atom(R_SN, c)),
            clause_id=1,
        )
        sat_p = _clause((True, _atom(P_SN, a)), clause_id=2)
        sat_q = _clause((True, _atom(Q_SN, b)), clause_id=3)
        results = hyper_resolve(nuc, [sat_p, sat_q])

        assert len(results) >= 1
        # At least one resolvent should be just R(c)
        found_rc = False
        for r in results:
            if len(r.literals) == 1 and r.literals[0].sign:
                found_rc = True
                assert r.justification[0].just_type == JustType.HYPER_RES
                assert 1 in r.justification[0].clause_ids
        assert found_rc

    def test_two_neg_literals_with_shared_variable(self):
        """-P(x) | -Q(x) | R(x) + P(a) + Q(a) → R(a).

        The shared variable x must be consistently bound across
        both resolution steps.
        """
        nuc = _clause(
            (False, _atom(P_SN, x)),
            (False, _atom(Q_SN, x)),
            (True, _atom(R_SN, x)),
            clause_id=1,
        )
        sat_p = _clause((True, _atom(P_SN, a)), clause_id=2)
        sat_q = _clause((True, _atom(Q_SN, a)), clause_id=3)
        results = hyper_resolve(nuc, [sat_p, sat_q])

        assert len(results) >= 1
        # The resolvent R(a) should have a ground constant arg
        found = False
        for r in results:
            if len(r.literals) == 1:
                found = True
        assert found

    def test_shared_variable_conflict(self):
        """-P(x) | -Q(x) | R(x) with P(a) + Q(b) → no resolvent.

        P(a) forces x=a, but Q(b) needs x=b — conflict.
        """
        nuc = _clause(
            (False, _atom(P_SN, x)),
            (False, _atom(Q_SN, x)),
            (True, _atom(R_SN, x)),
            clause_id=1,
        )
        sat_p = _clause((True, _atom(P_SN, a)), clause_id=2)
        sat_q = _clause((True, _atom(Q_SN, b)), clause_id=3)
        results = hyper_resolve(nuc, [sat_p, sat_q])

        # Should be empty since x can't be both a and b
        assert len(results) == 0

    def test_non_positive_result_discarded(self):
        """Satellite with extra negative literal → result not all-positive → discarded."""
        # Nucleus: -P(a) | Q(b)
        nuc = _clause(
            (False, _atom(P_SN, a)),
            (True, _atom(Q_SN, b)),
            clause_id=1,
        )
        # Satellite: P(a) | -R(c)  — has a negative literal
        sat = _clause(
            (True, _atom(P_SN, a)),
            (False, _atom(R_SN, c)),
            clause_id=2,
        )
        results = hyper_resolve(nuc, [sat])

        # Resolvent would be Q(b) | -R(c) which is not all-positive
        assert len(results) == 0

    def test_empty_clause_result(self):
        """Hyper-resolution can produce empty clause (proof).

        -P(a) resolved with P(a) → empty clause.
        """
        nuc = _clause((False, _atom(P_SN, a)), clause_id=1)
        sat = _clause((True, _atom(P_SN, a)), clause_id=2)
        results = hyper_resolve(nuc, [sat])

        assert len(results) == 1
        assert results[0].is_empty

    def test_satellite_with_multiple_positive_literals(self):
        """Satellite with multiple positive literals: only one is resolved.

        Remaining positive literals go into the resolvent.
        """
        nuc = _clause(
            (False, _atom(P_SN, a)),
            (True, _atom(R_SN, c)),
            clause_id=1,
        )
        sat = _clause(
            (True, _atom(P_SN, a)),
            (True, _atom(Q_SN, b)),
            clause_id=2,
        )
        results = hyper_resolve(nuc, [sat])

        assert len(results) == 1
        r = results[0]
        # R(c) from nucleus + Q(b) from satellite
        assert len(r.literals) == 2
        assert all(lit.sign for lit in r.literals)

    def test_multiple_candidate_satellites(self):
        """When multiple satellites can resolve a literal, get multiple resolvents."""
        nuc = _clause(
            (False, _atom(P_SN, x)),
            (True, _atom(Q_SN, x)),
            clause_id=1,
        )
        sat_a = _clause((True, _atom(P_SN, a)), clause_id=2)
        sat_b = _clause((True, _atom(P_SN, b)), clause_id=3)
        results = hyper_resolve(nuc, [sat_a, sat_b])

        # Two resolvents: Q(a) and Q(b)
        assert len(results) == 2


# ── all_hyper_resolvents tests ───────────────────────────────────────────────


class TestAllHyperResolvents:
    """Test all_hyper_resolvents() — given clause as nucleus or satellite."""

    def test_given_as_nucleus(self):
        """Given clause with negative literals acts as nucleus."""
        given = _clause(
            (False, _atom(P_SN, a)),
            (True, _atom(Q_SN, b)),
            clause_id=1,
        )
        usable = [
            _clause((True, _atom(P_SN, a)), clause_id=2),
        ]
        results = all_hyper_resolvents(given, usable)
        assert len(results) >= 1

    def test_given_as_satellite(self):
        """Given clause with positive literal acts as satellite for usable nucleus."""
        given = _clause((True, _atom(P_SN, a)), clause_id=2)
        nuc = _clause(
            (False, _atom(P_SN, a)),
            (True, _atom(Q_SN, b)),
            clause_id=1,
        )
        usable = [nuc, given]
        results = all_hyper_resolvents(given, usable)

        # given is a satellite — should generate resolvents where given participates
        assert len(results) >= 1

    def test_given_all_positive_only_satellite(self):
        """Purely positive given clause can only be a satellite, not nucleus."""
        given = _clause(
            (True, _atom(P_SN, a)),
            (True, _atom(Q_SN, b)),
            clause_id=1,
        )
        nuc = _clause(
            (False, _atom(P_SN, a)),
            (True, _atom(R_SN, c)),
            clause_id=2,
        )
        usable = [nuc, given]
        results = all_hyper_resolvents(given, usable)

        # Given has no neg lits → not a nucleus
        # But given can be satellite for nuc
        assert len(results) >= 1


# ── Search integration tests ────────────────────────────────────────────────


class TestHyperResolutionSearchIntegration:
    """Test hyper-resolution integration with given-clause search."""

    def test_search_options_has_hyper_resolution(self):
        """SearchOptions includes hyper_resolution flag."""
        from pyladr.search.given_clause import SearchOptions

        opts = SearchOptions()
        assert hasattr(opts, "hyper_resolution")
        assert opts.hyper_resolution is False

    def test_hyper_resolution_finds_proof(self):
        """Hyper-resolution can find a proof for a simple problem.

        Problem: -P(x) | -Q(x) | R(x), P(a), Q(a), -R(a)
        Hyper-resolution of clause 1 with clauses 2,3 gives R(a).
        Then R(a) + -R(a) gives empty clause.
        """
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        opts = SearchOptions(
            binary_resolution=True,
            hyper_resolution=True,
            max_given=100,
            quiet=True,
        )
        search = GivenClauseSearch(options=opts)

        sos = [
            _clause(
                (False, _atom(P_SN, x)),
                (False, _atom(Q_SN, x)),
                (True, _atom(R_SN, x)),
                clause_id=0,
            ),
            _clause((True, _atom(P_SN, a)), clause_id=0),
            _clause((True, _atom(Q_SN, a)), clause_id=0),
            _clause((False, _atom(R_SN, a)), clause_id=0),
        ]

        result = search.run(sos=sos)
        assert len(result.proofs) >= 1

    def test_justification_type_is_hyper_res(self):
        """Hyper-resolution resolvents have HYPER_RES justification."""
        nuc = _clause(
            (False, _atom(P_SN, a)),
            (True, _atom(Q_SN, b)),
            clause_id=1,
        )
        sat = _clause((True, _atom(P_SN, a)), clause_id=2)
        results = hyper_resolve(nuc, [sat])

        assert len(results) == 1
        assert results[0].justification[0].just_type == JustType.HYPER_RES
        assert results[0].justification[0].clause_ids == (1, 2)
