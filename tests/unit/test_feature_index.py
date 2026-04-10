"""Tests for pyladr.indexing.feature_index — Feature vector subsumption prefiltering.

Tests behavioral equivalence with C's feature vector indexing:
- Feature vector computation from clauses
- Subsumption candidate prefiltering (can_subsume)
- Feature index insert/delete/retrieve operations
- Correctness of necessary conditions for subsumption
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.indexing.feature_index import FeatureIndex, FeatureVector


def _atom(sn: int, *args) -> object:
    return get_rigid_term(sn, len(args), tuple(args))


def _pos(sn: int, *args) -> Literal:
    return Literal(sign=True, atom=_atom(sn, *args))


def _neg(sn: int, *args) -> Literal:
    return Literal(sign=False, atom=_atom(sn, *args))


def _eq_lit(left, right, *, sign=True) -> Literal:
    """Build an equality literal."""
    eq_atom = get_rigid_term(1, 2, (left, right))  # sn=1 for =
    return Literal(sign=sign, atom=eq_atom)


class TestFeatureVectorComputation:
    """Test feature vector extraction from clauses."""

    def test_unit_positive_clause(self):
        """Single positive literal."""
        x = get_variable_term(0)
        c = Clause(literals=(_pos(2, x),))
        fv = FeatureVector.from_clause(c)
        assert fv.pos_lit_count == 1
        assert fv.neg_lit_count == 0

    def test_unit_negative_clause(self):
        """Single negative literal."""
        x = get_variable_term(0)
        c = Clause(literals=(_neg(2, x),))
        fv = FeatureVector.from_clause(c)
        assert fv.pos_lit_count == 0
        assert fv.neg_lit_count == 1

    def test_mixed_clause(self):
        """P(x) | -Q(y) — one positive, one negative."""
        x, y = get_variable_term(0), get_variable_term(1)
        c = Clause(literals=(_pos(2, x), _neg(3, y)))
        fv = FeatureVector.from_clause(c)
        assert fv.pos_lit_count == 1
        assert fv.neg_lit_count == 1

    def test_ground_clause_zero_variables(self):
        """Ground clause has no variables."""
        a = get_rigid_term(1, 0)
        c = Clause(literals=(_pos(2, a),))
        fv = FeatureVector.from_clause(c)
        assert fv.variable_count == 0

    def test_variable_count(self):
        """Count distinct variables."""
        x, y = get_variable_term(0), get_variable_term(1)
        a = get_rigid_term(1, 0)
        # P(x, y) | Q(x) — 2 distinct variables
        c = Clause(literals=(
            _pos(2, x, y),
            _pos(3, x),
        ))
        fv = FeatureVector.from_clause(c)
        assert fv.variable_count == 2

    def test_empty_clause(self):
        """Empty clause has zero for everything."""
        c = Clause()
        fv = FeatureVector.from_clause(c)
        assert fv.pos_lit_count == 0
        assert fv.neg_lit_count == 0
        assert fv.variable_count == 0


class TestCanSubsume:
    """Test the can_subsume necessary condition checker."""

    def test_unit_can_subsume_multi(self):
        """P(x) can potentially subsume P(a) | Q(b)."""
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        c1 = Clause(literals=(_pos(3, x),))
        c2 = Clause(literals=(_pos(3, a), _pos(4, b)))
        fv1 = FeatureVector.from_clause(c1)
        fv2 = FeatureVector.from_clause(c2)
        assert fv1.can_subsume(fv2)

    def test_multi_cannot_subsume_unit(self):
        """P(x) | Q(y) cannot subsume P(a) — more positive literals."""
        x, y = get_variable_term(0), get_variable_term(1)
        a = get_rigid_term(1, 0)
        c1 = Clause(literals=(_pos(3, x), _pos(4, y)))
        c2 = Clause(literals=(_pos(3, a),))
        fv1 = FeatureVector.from_clause(c1)
        fv2 = FeatureVector.from_clause(c2)
        assert not fv1.can_subsume(fv2)

    def test_negative_count_blocks(self):
        """-P(x) | -Q(y) cannot subsume -P(a) — more negative literals."""
        x, y = get_variable_term(0), get_variable_term(1)
        a = get_rigid_term(1, 0)
        c1 = Clause(literals=(_neg(3, x), _neg(4, y)))
        c2 = Clause(literals=(_neg(3, a),))
        fv1 = FeatureVector.from_clause(c1)
        fv2 = FeatureVector.from_clause(c2)
        assert not fv1.can_subsume(fv2)

    def test_same_structure_can_subsume(self):
        """Identical structure: P(x) can subsume P(a)."""
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        c1 = Clause(literals=(_pos(3, x),))
        c2 = Clause(literals=(_pos(3, a),))
        fv1 = FeatureVector.from_clause(c1)
        fv2 = FeatureVector.from_clause(c2)
        assert fv1.can_subsume(fv2)

    def test_self_can_subsume(self):
        """A clause's feature vector can_subsume itself."""
        x = get_variable_term(0)
        c = Clause(literals=(_pos(3, x), _neg(4, x)))
        fv = FeatureVector.from_clause(c)
        assert fv.can_subsume(fv)


class TestFeatureIndex:
    """Test the FeatureIndex container for clause storage and retrieval."""

    def test_forward_candidates(self):
        """Insert clauses and retrieve forward subsumption candidates."""
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)

        # Unit clause P(x) — should be a forward candidate for P(a) | Q(b)
        c1 = Clause(literals=(_pos(3, x),), id=1)
        # Two-literal clause P(a) | Q(b)
        c2 = Clause(literals=(_pos(3, a), _pos(4, b)), id=2)

        idx = FeatureIndex()
        idx.insert(c1)
        idx.insert(c2)

        # forward_candidates(c2) finds clauses that could subsume c2
        candidates = idx.forward_candidates(c2)
        cand_ids = {c.id for c in candidates}
        assert 1 in cand_ids  # c1 can potentially subsume c2

    def test_backward_candidates(self):
        """Backward candidates: find clauses that new clause could subsume."""
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)

        c1 = Clause(literals=(_pos(3, x),), id=1)
        c2 = Clause(literals=(_pos(3, a), _pos(4, b)), id=2)

        idx = FeatureIndex()
        idx.insert(c1)
        idx.insert(c2)

        # backward_candidates(c1) finds clauses that c1 could subsume
        candidates = idx.backward_candidates(c1)
        cand_ids = {c.id for c in candidates}
        assert 2 in cand_ids  # c1 could subsume c2

    def test_delete_removes_clause(self):
        """After deletion, clause is no longer returned."""
        x = get_variable_term(0)
        c1 = Clause(literals=(_pos(3, x),), id=1)

        idx = FeatureIndex()
        idx.insert(c1)
        idx.delete(c1)
        assert len(idx) == 0
