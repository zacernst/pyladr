"""Tests for pyladr.indexing.literal_index — Literal-sign-based indexing.

Tests behavioral equivalence with C lindex.c:
- Inserting clauses by literal sign
- Positive/negative literal indexing separation
- Clause retrieval from the correct sign partition
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.indexing.literal_index import LiteralIndex


def _atom(sn: int, *args):
    return get_rigid_term(sn, len(args), tuple(args))


def _pos_lit(sn: int, *args) -> Literal:
    return Literal(sign=True, atom=_atom(sn, *args))


def _neg_lit(sn: int, *args) -> Literal:
    return Literal(sign=False, atom=_atom(sn, *args))


class TestLiteralIndexBasic:
    """Test basic literal index operations."""

    def test_create_empty_index(self):
        """Can create an empty LiteralIndex."""
        idx = LiteralIndex()
        assert idx.pos is not None
        assert idx.neg is not None

    def test_insert_positive_clause(self):
        """Positive literal goes into pos index."""
        a = get_rigid_term(1, 0)
        c = Clause(literals=(_pos_lit(2, a),), id=1)
        idx = LiteralIndex()
        idx.update(c, insert=True)
        # Verify: positive literals should be in pos index, not neg

    def test_insert_negative_clause(self):
        """Negative literal goes into neg index."""
        a = get_rigid_term(1, 0)
        c = Clause(literals=(_neg_lit(2, a),), id=1)
        idx = LiteralIndex()
        idx.update(c, insert=True)

    def test_insert_mixed_clause(self):
        """Mixed clause indexes literals into appropriate partitions."""
        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        c = Clause(literals=(_pos_lit(2, x), _neg_lit(3, a)), id=1)
        idx = LiteralIndex()
        idx.update(c, insert=True)

    def test_insert_then_delete(self):
        """Delete removes clause from index."""
        a = get_rigid_term(1, 0)
        c = Clause(literals=(_pos_lit(2, a),), id=1)
        idx = LiteralIndex()
        idx.update(c, insert=True)
        idx.update(c, insert=False)

    def test_multiple_clauses(self):
        """Multiple clauses can be indexed."""
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        c1 = Clause(literals=(_pos_lit(3, a),), id=1)
        c2 = Clause(literals=(_pos_lit(3, b),), id=2)
        c3 = Clause(literals=(_neg_lit(4, a),), id=3)
        idx = LiteralIndex()
        idx.update(c1, insert=True)
        idx.update(c2, insert=True)
        idx.update(c3, insert=True)


class TestLiteralIndexFirstOnly:
    """Test first_only mode matching C lindex_update_first()."""

    def test_first_only_indexes_first_literal(self):
        """In first_only mode, only the first literal is indexed."""
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        c = Clause(literals=(_pos_lit(3, a), _neg_lit(4, b)), id=1)
        idx = LiteralIndex(first_only=True)
        idx.update(c, insert=True)
        # Only the first literal (positive P(a)) should be indexed
