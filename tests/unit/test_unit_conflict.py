"""Unit tests for pyladr.search.unit_conflict — O(1) unit conflict index."""

from __future__ import annotations

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.search.unit_conflict import UnitConflictIndex, _term_key


# Symbol IDs
P_SN, F_SN, A_SN, B_SN, C_SN = 1, 2, 3, 4, 5


def _const(sn: int):
    return get_rigid_term(sn, 0)


def _func(sn: int, *args):
    return get_rigid_term(sn, len(args), tuple(args))


def _unit(sign: bool, atom, clause_id: int = 1) -> Clause:
    return Clause(literals=(Literal(sign=sign, atom=atom),), id=clause_id)


# ── _term_key tests ──────────────────────────────────────────────────────────


class TestTermKey:
    def test_distinct_terms_produce_distinct_keys(self):
        """f(a, b) and f(a, c) have different keys."""
        t1 = _func(F_SN, _const(A_SN), _const(B_SN))
        t2 = _func(F_SN, _const(A_SN), _const(C_SN))
        assert _term_key(t1) != _term_key(t2)

    def test_identical_structure_produces_same_key(self):
        """Two independently built f(a, b) terms have the same key."""
        t1 = _func(F_SN, _const(A_SN), _const(B_SN))
        t2 = _func(F_SN, _const(A_SN), _const(B_SN))
        assert _term_key(t1) == _term_key(t2)


# ── UnitConflictIndex tests ──────────────────────────────────────────────────


class TestUnitConflictIndex:
    def test_find_complement_detects_sign_mismatch(self):
        """Inserting P(a) (positive) and querying with -P(a) finds the conflict."""
        idx = UnitConflictIndex()
        atom = _func(P_SN, _const(A_SN))
        pos_clause = _unit(sign=True, atom=atom, clause_id=1)
        neg_clause = _unit(sign=False, atom=atom, clause_id=2)

        idx.insert(pos_clause)
        result = idx.find_complement(neg_clause)
        assert result is not None
        assert result.id == pos_clause.id

    def test_find_complement_returns_none_when_no_conflict(self):
        """No complement exists -> returns None."""
        idx = UnitConflictIndex()
        atom_a = _func(P_SN, _const(A_SN))
        atom_b = _func(P_SN, _const(B_SN))

        idx.insert(_unit(sign=True, atom=atom_a, clause_id=1))
        # Query with -P(b) — different atom, no conflict
        result = idx.find_complement(_unit(sign=False, atom=atom_b, clause_id=2))
        assert result is None

    def test_remove_prevents_future_lookup(self):
        """After remove, the clause no longer appears in results."""
        idx = UnitConflictIndex()
        atom = _func(P_SN, _const(A_SN))
        pos_clause = _unit(sign=True, atom=atom, clause_id=1)
        neg_clause = _unit(sign=False, atom=atom, clause_id=2)

        idx.insert(pos_clause)
        assert idx.find_complement(neg_clause) is not None

        idx.remove(pos_clause)
        assert idx.find_complement(neg_clause) is None

    def test_non_unit_clause_ignored_on_insert(self):
        """Non-unit clauses are silently ignored by insert."""
        idx = UnitConflictIndex()
        atom = _func(P_SN, _const(A_SN))
        multi_lit = Clause(
            literals=(
                Literal(sign=True, atom=atom),
                Literal(sign=False, atom=atom),
            ),
            id=1,
        )
        idx.insert(multi_lit)
        # Nothing was inserted, so complement query returns None
        result = idx.find_complement(_unit(sign=False, atom=atom, clause_id=2))
        assert result is None

    def test_find_complement_returns_none_for_non_unit_query(self):
        """find_complement returns None when the query clause is non-unit."""
        idx = UnitConflictIndex()
        atom = _func(P_SN, _const(A_SN))
        idx.insert(_unit(sign=True, atom=atom, clause_id=1))

        multi_lit = Clause(
            literals=(
                Literal(sign=False, atom=atom),
                Literal(sign=True, atom=_func(P_SN, _const(B_SN))),
            ),
            id=2,
        )
        assert idx.find_complement(multi_lit) is None
