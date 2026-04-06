"""Tests for discrimination tree indexing.

Tests cover:
- Path encoding correctness
- Wild discrimination tree: insert, delete, retrieve
- Bind discrimination tree: insert, delete, retrieve with substitutions
- Mindex interface for all backends
- Thread safety (basic concurrent access)
- Performance characteristics on realistic workloads
"""

from __future__ import annotations

import threading

import pytest

from pyladr.core.substitution import Context
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.indexing.discrimination_tree import (
    DiscrimBind,
    DiscrimWild,
    IndexType,
    Mindex,
    _NodeType,
    _term_to_path,
    _term_to_wild_path,
)

# ── Helper factories ─────────────────────────────────────────────────────────

# Symbol IDs (arbitrary positive ints)
SYM_E = 1      # constant 'e'
SYM_STAR = 2   # binary '*'
SYM_INV = 3    # unary "'"
SYM_A = 4      # constant 'a'
SYM_B = 5      # constant 'b'
SYM_C = 6      # constant 'c'
SYM_F = 7      # binary 'f'
SYM_G = 8      # unary 'g'
SYM_H = 9      # ternary 'h'


def var(n: int) -> Term:
    """Create variable term x_n."""
    return get_variable_term(n)


def const(sym: int) -> Term:
    """Create constant term."""
    return get_rigid_term(sym, 0)


def func(sym: int, *args: Term) -> Term:
    """Create function application."""
    return get_rigid_term(sym, len(args), tuple(args))


# Commonly used terms
e = const(SYM_E)
a = const(SYM_A)
b = const(SYM_B)
c = const(SYM_C)
x = var(0)
y = var(1)
z = var(2)


# ── Path encoding tests ─────────────────────────────────────────────────────


class TestPathEncoding:
    """Test that terms are encoded as correct pre-order paths."""

    def test_variable_path(self):
        path = _term_to_path(var(3))
        assert path == [(_NodeType.DVARIABLE, 3)]

    def test_constant_path(self):
        path = _term_to_path(const(SYM_A))
        assert path == [(_NodeType.DRIGID, SYM_A)]

    def test_unary_path(self):
        # g(a)
        t = func(SYM_G, a)
        path = _term_to_path(t)
        assert path == [(_NodeType.DRIGID, SYM_G), (_NodeType.DRIGID, SYM_A)]

    def test_binary_path(self):
        # f(a, b)
        t = func(SYM_F, a, b)
        path = _term_to_path(t)
        assert path == [
            (_NodeType.DRIGID, SYM_F),
            (_NodeType.DRIGID, SYM_A),
            (_NodeType.DRIGID, SYM_B),
        ]

    def test_nested_path(self):
        # f(g(a), b)
        t = func(SYM_F, func(SYM_G, a), b)
        path = _term_to_path(t)
        assert path == [
            (_NodeType.DRIGID, SYM_F),
            (_NodeType.DRIGID, SYM_G),
            (_NodeType.DRIGID, SYM_A),
            (_NodeType.DRIGID, SYM_B),
        ]

    def test_variable_in_complex_path(self):
        # f(x, a)
        t = func(SYM_F, x, a)
        path = _term_to_path(t)
        assert path == [
            (_NodeType.DRIGID, SYM_F),
            (_NodeType.DVARIABLE, 0),
            (_NodeType.DRIGID, SYM_A),
        ]

    def test_wild_path_collapses_variables(self):
        # f(x, y) — both become DVARIABLE with symbol=0
        t = func(SYM_F, x, y)
        path = _term_to_wild_path(t)
        assert path == [
            (_NodeType.DRIGID, SYM_F),
            (_NodeType.DVARIABLE, 0),
            (_NodeType.DVARIABLE, 0),
        ]

    def test_bind_path_distinguishes_variables(self):
        # f(x, y) — DVARIABLE 0 and DVARIABLE 1
        t = func(SYM_F, x, y)
        path = _term_to_path(t)
        assert path == [
            (_NodeType.DRIGID, SYM_F),
            (_NodeType.DVARIABLE, 0),
            (_NodeType.DVARIABLE, 1),
        ]


# ── Wild Discrimination Tree tests ──────────────────────────────────────────


class TestDiscrimWild:
    """Tests for wild (imperfect filter) discrimination tree."""

    def test_empty_tree(self):
        tree = DiscrimWild()
        assert tree.size == 0
        assert tree.retrieve_generalizations_flat(a) == []

    def test_insert_and_retrieve_constant(self):
        tree = DiscrimWild()
        tree.insert(a, "obj_a")
        assert tree.size == 1
        results = tree.retrieve_generalizations_flat(a)
        assert results == ["obj_a"]

    def test_constant_no_match(self):
        tree = DiscrimWild()
        tree.insert(a, "obj_a")
        results = tree.retrieve_generalizations_flat(b)
        assert results == []

    def test_variable_generalizes_any_constant(self):
        """A stored variable x generalizes any ground term."""
        tree = DiscrimWild()
        tree.insert(x, "obj_x")
        assert tree.retrieve_generalizations_flat(a) == ["obj_x"]
        assert tree.retrieve_generalizations_flat(b) == ["obj_x"]
        assert tree.retrieve_generalizations_flat(func(SYM_F, a, b)) == ["obj_x"]

    def test_complex_exact_match(self):
        """f(a, b) retrieves from index keyed by f(a, b)."""
        tree = DiscrimWild()
        t = func(SYM_F, a, b)
        tree.insert(t, "fab")
        assert tree.retrieve_generalizations_flat(t) == ["fab"]

    def test_complex_no_match_different_symbol(self):
        tree = DiscrimWild()
        tree.insert(func(SYM_F, a, b), "fab")
        # g(a, b) should not match f(a, b)
        assert tree.retrieve_generalizations_flat(func(SYM_G, a)) == []

    def test_complex_with_variable_generalizes(self):
        """f(x, b) generalizes f(a, b)."""
        tree = DiscrimWild()
        tree.insert(func(SYM_F, x, b), "fxb")
        results = tree.retrieve_generalizations_flat(func(SYM_F, a, b))
        assert results == ["fxb"]

    def test_complex_variable_both_args(self):
        """f(x, y) generalizes f(a, b)."""
        tree = DiscrimWild()
        tree.insert(func(SYM_F, x, y), "fxy")
        results = tree.retrieve_generalizations_flat(func(SYM_F, a, b))
        assert results == ["fxy"]

    def test_nested_generalization(self):
        """f(g(x), b) generalizes f(g(a), b)."""
        tree = DiscrimWild()
        stored = func(SYM_F, func(SYM_G, x), b)
        query = func(SYM_F, func(SYM_G, a), b)
        tree.insert(stored, "fgxb")
        results = tree.retrieve_generalizations_flat(query)
        assert results == ["fgxb"]

    def test_variable_generalizes_complex_subterm(self):
        """f(x, b) generalizes f(g(a), b) — x matches g(a)."""
        tree = DiscrimWild()
        tree.insert(func(SYM_F, x, b), "fxb")
        query = func(SYM_F, func(SYM_G, a), b)
        results = tree.retrieve_generalizations_flat(query)
        assert results == ["fxb"]

    def test_multiple_results(self):
        """Multiple stored terms can generalize the same query."""
        tree = DiscrimWild()
        tree.insert(x, "var")            # x generalizes everything
        tree.insert(func(SYM_F, x, y), "fxy")   # f(x,y) generalizes f(a,b)
        tree.insert(func(SYM_F, a, y), "fay")   # f(a,y) generalizes f(a,b)
        tree.insert(func(SYM_F, a, b), "fab")   # exact match

        query = func(SYM_F, a, b)
        results = tree.retrieve_generalizations_flat(query)
        assert set(results) == {"var", "fxy", "fay", "fab"}

    def test_delete_removes_entry(self):
        tree = DiscrimWild()
        tree.insert(a, "obj_a")
        assert tree.size == 1
        assert tree.delete(a, "obj_a")
        assert tree.size == 0
        assert tree.retrieve_generalizations_flat(a) == []

    def test_delete_wrong_object_fails(self):
        tree = DiscrimWild()
        tree.insert(a, "obj_a")
        assert not tree.delete(a, "wrong")
        assert tree.size == 1

    def test_delete_wrong_term_fails(self):
        tree = DiscrimWild()
        tree.insert(a, "obj_a")
        assert not tree.delete(b, "obj_a")
        assert tree.size == 1

    def test_multiple_objects_same_term(self):
        """Multiple objects can be stored under the same term."""
        tree = DiscrimWild()
        tree.insert(a, "obj1")
        tree.insert(a, "obj2")
        assert tree.size == 2
        results = tree.retrieve_generalizations_flat(a)
        assert set(results) == {"obj1", "obj2"}

    def test_delete_one_of_multiple(self):
        tree = DiscrimWild()
        tree.insert(a, "obj1")
        tree.insert(a, "obj2")
        tree.delete(a, "obj1")
        assert tree.size == 1
        results = tree.retrieve_generalizations_flat(a)
        assert results == ["obj2"]

    def test_query_variable_matches_stored_variable_only(self):
        """When query is a variable, only stored variables match."""
        tree = DiscrimWild()
        tree.insert(x, "var")
        tree.insert(a, "const")
        # Query is variable y — only stored variable x (wildcard) matches
        results = tree.retrieve_generalizations_flat(y)
        assert results == ["var"]


# ── Bind Discrimination Tree tests ──────────────────────────────────────────


class TestDiscrimBind:
    """Tests for bind (perfect filter) discrimination tree."""

    def test_empty_tree(self):
        tree = DiscrimBind()
        assert tree.size == 0
        results = tree.retrieve_generalizations(a)
        assert results == []

    def test_constant_retrieval(self):
        tree = DiscrimBind()
        tree.insert(a, "obj_a")
        results = tree.retrieve_generalizations(a)
        assert len(results) == 1
        assert results[0][0] == "obj_a"

    def test_variable_generalizes_constant(self):
        """Stored x generalizes query a, binding x→a."""
        tree = DiscrimBind()
        tree.insert(x, "var")
        results = tree.retrieve_generalizations(a)
        assert len(results) == 1
        obj, subst = results[0]
        assert obj == "var"
        # x (varnum=0) should be bound to 'a'
        assert subst.is_bound(0)
        assert subst.terms[0] is not None
        assert subst.terms[0].term_ident(a)

    def test_variable_consistency(self):
        """f(x, x) should NOT generalize f(a, b) — variable must be consistent."""
        tree = DiscrimBind()
        tree.insert(func(SYM_F, x, x), "fxx")
        results = tree.retrieve_generalizations(func(SYM_F, a, b))
        assert len(results) == 0  # x can't be both a and b

    def test_variable_consistency_match(self):
        """f(x, x) SHOULD generalize f(a, a) — x→a consistent."""
        tree = DiscrimBind()
        tree.insert(func(SYM_F, x, x), "fxx")
        results = tree.retrieve_generalizations(func(SYM_F, a, a))
        assert len(results) == 1
        assert results[0][0] == "fxx"

    def test_two_different_variables(self):
        """f(x, y) generalizes f(a, b) with x→a, y→b."""
        tree = DiscrimBind()
        tree.insert(func(SYM_F, x, y), "fxy")
        results = tree.retrieve_generalizations(func(SYM_F, a, b))
        assert len(results) == 1
        obj, subst = results[0]
        assert obj == "fxy"
        assert subst.terms[0] is not None
        assert subst.terms[0].term_ident(a)  # x→a
        assert subst.terms[1] is not None
        assert subst.terms[1].term_ident(b)  # y→b

    def test_nested_binding(self):
        """f(x, b) generalizes f(g(a), b), binding x→g(a)."""
        tree = DiscrimBind()
        tree.insert(func(SYM_F, x, b), "fxb")
        query = func(SYM_F, func(SYM_G, a), b)
        results = tree.retrieve_generalizations(query)
        assert len(results) == 1
        obj, subst = results[0]
        assert obj == "fxb"
        bound = subst.terms[0]
        assert bound is not None
        assert bound.term_ident(func(SYM_G, a))

    def test_delete(self):
        tree = DiscrimBind()
        tree.insert(a, "obj_a")
        assert tree.delete(a, "obj_a")
        assert tree.size == 0

    def test_multiple_generalizations(self):
        """Multiple stored terms can generalize the same query."""
        tree = DiscrimBind()
        tree.insert(x, "var")
        tree.insert(func(SYM_F, x, y), "fxy")
        tree.insert(func(SYM_F, x, b), "fxb")

        query = func(SYM_F, a, b)
        results = tree.retrieve_generalizations(query)
        objs = {obj for obj, _ in results}
        assert objs == {"var", "fxy", "fxb"}


# ── Mindex tests ─────────────────────────────────────────────────────────────


class TestMindex:
    """Tests for the unified multi-index interface."""

    @pytest.mark.parametrize("index_type", [IndexType.LINEAR, IndexType.DISCRIM_WILD, IndexType.DISCRIM_BIND])
    def test_insert_and_size(self, index_type: IndexType):
        idx = Mindex(index_type)
        idx.insert(a, "obj_a")
        assert idx.size == 1

    @pytest.mark.parametrize("index_type", [IndexType.LINEAR, IndexType.DISCRIM_WILD, IndexType.DISCRIM_BIND])
    def test_retrieve_generalizations(self, index_type: IndexType):
        idx = Mindex(index_type)
        idx.insert(x, "var")
        idx.insert(a, "const_a")
        results = idx.retrieve_generalizations(a)
        assert "var" in results
        assert "const_a" in results

    @pytest.mark.parametrize("index_type", [IndexType.LINEAR, IndexType.DISCRIM_WILD, IndexType.DISCRIM_BIND])
    def test_delete(self, index_type: IndexType):
        idx = Mindex(index_type)
        idx.insert(a, "obj")
        assert idx.delete(a, "obj")
        assert idx.size == 0


# ── Thread safety tests ─────────────────────────────────────────────────────


class TestThreadSafety:
    """Basic thread safety tests for concurrent index access."""

    def test_concurrent_inserts(self):
        """Multiple threads can insert concurrently without corruption."""
        tree = DiscrimWild()
        n_threads = 4
        n_inserts = 100

        def insert_batch(thread_id: int):
            for i in range(n_inserts):
                t = const(thread_id * 1000 + i + 10)  # unique symnum per entry
                tree.insert(t, f"t{thread_id}_{i}")

        threads = [
            threading.Thread(target=insert_batch, args=(tid,))
            for tid in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tree.size == n_threads * n_inserts

    def test_concurrent_reads(self):
        """Multiple threads can read concurrently."""
        tree = DiscrimWild()
        # Insert some data first
        tree.insert(x, "var")
        for i in range(100):
            tree.insert(const(i + 10), f"c{i}")

        results = []
        lock = threading.Lock()

        def read_batch():
            for i in range(50):
                r = tree.retrieve_generalizations_flat(const(i + 10))
                with lock:
                    results.append(len(r))

        threads = [threading.Thread(target=read_batch) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 200  # 4 threads * 50 reads


# ── Performance micro-benchmarks ─────────────────────────────────────────────


class TestPerformance:
    """Micro-benchmarks for discrimination tree operations."""

    @pytest.mark.benchmark
    def test_insert_throughput(self):
        """Measure insertion rate."""
        import time

        tree = DiscrimWild()
        n = 1000
        terms = [func(SYM_F, const(i + 10), const(i + 1010)) for i in range(n)]

        start = time.perf_counter()
        for i, t in enumerate(terms):
            tree.insert(t, i)
        elapsed = time.perf_counter() - start

        assert tree.size == n
        rate = n / elapsed
        print(f"\nDiscrimWild insert: {rate:.0f} terms/sec ({elapsed:.4f}s for {n})")

    @pytest.mark.benchmark
    def test_retrieval_throughput(self):
        """Measure retrieval rate."""
        import time

        tree = DiscrimWild()
        n = 1000
        # Insert terms with variable in first arg position
        for i in range(n):
            tree.insert(func(SYM_F, x, const(i + 10)), i)

        queries = [func(SYM_F, const(50), const(i + 10)) for i in range(n)]
        start = time.perf_counter()
        total_results = 0
        for q in queries:
            results = tree.retrieve_generalizations_flat(q)
            total_results += len(results)
        elapsed = time.perf_counter() - start

        rate = n / elapsed
        print(
            f"\nDiscrimWild retrieve: {rate:.0f} queries/sec ({elapsed:.4f}s for {n}), "
            f"avg results: {total_results/n:.1f}"
        )
