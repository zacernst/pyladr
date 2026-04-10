"""Tests for PrioritySOS: heap-based selection with O(1) removal."""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.search.priority_sos import PrioritySOS
from pyladr.search.state import ClauseList


def _make_clause(id: int, weight: float) -> Clause:
    """Create a minimal clause with given id and weight."""
    x = get_variable_term(id % 10)
    a = get_rigid_term(1, 0)
    atom = get_rigid_term(2, 1, (x,))
    c = Clause(literals=(Literal(sign=True, atom=atom),), id=id)
    c.weight = weight
    return c


class TestPrioritySOS:
    """Core PrioritySOS functionality."""

    def test_empty(self):
        sos = PrioritySOS("sos")
        assert sos.is_empty
        assert sos.length == 0
        assert len(sos) == 0
        assert sos.first is None
        assert sos.pop_first() is None
        assert sos.pop_lightest() is None

    def test_append_and_length(self):
        sos = PrioritySOS("sos")
        c1 = _make_clause(1, 5.0)
        c2 = _make_clause(2, 3.0)
        sos.append(c1)
        sos.append(c2)
        assert sos.length == 2
        assert not sos.is_empty

    def test_pop_first_fifo_order(self):
        sos = PrioritySOS("sos")
        c1 = _make_clause(1, 5.0)
        c2 = _make_clause(2, 3.0)
        c3 = _make_clause(3, 7.0)
        sos.append(c1)
        sos.append(c2)
        sos.append(c3)
        assert sos.pop_first() is c1
        assert sos.pop_first() is c2
        assert sos.pop_first() is c3
        assert sos.is_empty

    def test_pop_lightest_weight_order(self):
        sos = PrioritySOS("sos")
        c1 = _make_clause(1, 5.0)
        c2 = _make_clause(2, 3.0)
        c3 = _make_clause(3, 7.0)
        sos.append(c1)
        sos.append(c2)
        sos.append(c3)
        assert sos.pop_lightest() is c2  # weight 3.0
        assert sos.pop_lightest() is c1  # weight 5.0
        assert sos.pop_lightest() is c3  # weight 7.0
        assert sos.is_empty

    def test_pop_lightest_tiebreak_by_id(self):
        """Same weight — smaller ID wins (matching C behavior)."""
        sos = PrioritySOS("sos")
        c1 = _make_clause(10, 5.0)
        c2 = _make_clause(3, 5.0)
        c3 = _make_clause(7, 5.0)
        sos.append(c1)
        sos.append(c2)
        sos.append(c3)
        assert sos.pop_lightest() is c2  # id=3
        assert sos.pop_lightest() is c3  # id=7
        assert sos.pop_lightest() is c1  # id=10

    def test_remove(self):
        sos = PrioritySOS("sos")
        c1 = _make_clause(1, 5.0)
        c2 = _make_clause(2, 3.0)
        sos.append(c1)
        sos.append(c2)
        assert sos.remove(c1) is True
        assert sos.length == 1
        assert sos.pop_lightest() is c2
        assert sos.is_empty

    def test_remove_nonexistent(self):
        sos = PrioritySOS("sos")
        c1 = _make_clause(1, 5.0)
        assert sos.remove(c1) is False

    def test_remove_skips_stale_in_pop_lightest(self):
        """Removed clauses are skipped by pop_lightest."""
        sos = PrioritySOS("sos")
        c1 = _make_clause(1, 1.0)
        c2 = _make_clause(2, 2.0)
        c3 = _make_clause(3, 3.0)
        sos.append(c1)
        sos.append(c2)
        sos.append(c3)
        sos.remove(c1)  # Remove the lightest
        assert sos.pop_lightest() is c2  # Should skip c1

    def test_remove_skips_stale_in_pop_first(self):
        """Removed clauses are skipped by pop_first."""
        sos = PrioritySOS("sos")
        c1 = _make_clause(1, 5.0)
        c2 = _make_clause(2, 3.0)
        sos.append(c1)
        sos.append(c2)
        sos.remove(c1)  # Remove the oldest
        assert sos.pop_first() is c2

    def test_first_property(self):
        sos = PrioritySOS("sos")
        c1 = _make_clause(1, 5.0)
        c2 = _make_clause(2, 3.0)
        sos.append(c1)
        sos.append(c2)
        assert sos.first is c1  # Oldest
        sos.remove(c1)
        assert sos.first is c2

    def test_contains(self):
        sos = PrioritySOS("sos")
        c1 = _make_clause(1, 5.0)
        c2 = _make_clause(2, 3.0)
        sos.append(c1)
        assert sos.contains(c1)
        assert not sos.contains(c2)

    def test_iteration(self):
        sos = PrioritySOS("sos")
        c1 = _make_clause(1, 5.0)
        c2 = _make_clause(2, 3.0)
        c3 = _make_clause(3, 7.0)
        sos.append(c1)
        sos.append(c2)
        sos.append(c3)
        sos.remove(c2)
        ids = {c.id for c in sos}
        assert ids == {1, 3}

    def test_compact(self):
        sos = PrioritySOS("sos")
        for i in range(100):
            sos.append(_make_clause(i + 1, float(i)))
        # Remove 90 of them
        for i in range(90):
            sos.remove(_make_clause(i + 1, float(i)))
        # Heap still has stale entries
        assert len(sos._heap) > 10
        sos.compact()
        assert len(sos._heap) == 10
        assert sos.length == 10

    def test_peek_lightest(self):
        sos = PrioritySOS("sos")
        c1 = _make_clause(1, 5.0)
        c2 = _make_clause(2, 3.0)
        sos.append(c1)
        sos.append(c2)
        assert sos.peek_lightest() is c2
        assert sos.length == 2  # Not removed


class TestPrioritySOS_vs_ClauseList:
    """Verify PrioritySOS produces identical weight-ordered extraction as ClauseList."""

    def test_identical_weight_extraction(self):
        """Extract by weight from both — must produce same order."""
        clauses = []
        for i in range(200):
            weight = float((i * 13 + 7) % 200)
            clauses.append(_make_clause(i + 1, weight))

        # Linear scan extraction
        linear_sos = ClauseList("sos")
        for c in clauses:
            linear_sos.append(c)
        linear_order = []
        for _ in range(50):
            best = None
            for c in linear_sos:
                if best is None or (c.weight, c.id) < (best.weight, best.id):
                    best = c
            if best:
                linear_sos.remove(best)
                linear_order.append(best.id)

        # Heap extraction
        heap_sos = PrioritySOS("sos")
        for c in clauses:
            heap_sos.append(c)
        heap_order = []
        for _ in range(50):
            best = heap_sos.pop_lightest()
            if best:
                heap_order.append(best.id)

        assert linear_order == heap_order

    def test_identical_age_extraction(self):
        """Extract by age from both — must produce same order."""
        clauses = [_make_clause(i + 1, float(i * 7 % 50)) for i in range(100)]

        linear_sos = ClauseList("sos")
        heap_sos = PrioritySOS("sos")
        for c in clauses:
            linear_sos.append(c)
            heap_sos.append(c)

        for _ in range(50):
            l = linear_sos.pop_first()
            h = heap_sos.pop_first()
            assert l is not None and h is not None
            assert l.id == h.id

    def test_interleaved_add_remove_extract(self):
        """Interleave adds, removes, and extractions."""
        clauses = [_make_clause(i + 1, float((i * 17 + 3) % 100)) for i in range(50)]

        linear_sos = ClauseList("sos")
        heap_sos = PrioritySOS("sos")

        # Add first 30
        for c in clauses[:30]:
            linear_sos.append(c)
            heap_sos.append(c)

        # Remove 5 specific ones
        for c in clauses[10:15]:
            linear_sos.remove(c)
            heap_sos.remove(c)

        # Add remaining 20
        for c in clauses[30:]:
            linear_sos.append(c)
            heap_sos.append(c)

        # Extract all by weight and compare
        linear_order = []
        while not linear_sos.is_empty:
            best = None
            for c in linear_sos:
                if best is None or (c.weight, c.id) < (best.weight, best.id):
                    best = c
            if best:
                linear_sos.remove(best)
                linear_order.append(best.id)

        heap_order = []
        while not heap_sos.is_empty:
            best = heap_sos.pop_lightest()
            if best:
                heap_order.append(best.id)

        assert linear_order == heap_order
