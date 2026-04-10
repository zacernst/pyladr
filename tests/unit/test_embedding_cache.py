"""Tests for the high-performance embedding cache."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term
from pyladr.ml.embeddings.cache import (
    CacheConfig,
    CacheStatistics,
    EmbeddingCache,
    clause_structural_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _var(n: int) -> Term:
    """Create a variable term with varnum *n*."""
    return Term(private_symbol=n, arity=0, args=())


def _const(symnum: int) -> Term:
    """Create a constant (0-arity rigid symbol)."""
    return Term(private_symbol=-symnum, arity=0, args=())


def _func(symnum: int, *args: Term) -> Term:
    """Create a function application."""
    return Term(private_symbol=-symnum, arity=len(args), args=tuple(args))


def _lit(sign: bool, atom: Term) -> Literal:
    return Literal(sign=sign, atom=atom)


def _clause(*lits: Literal, clause_id: int = 0) -> Clause:
    c = Clause(literals=tuple(lits))
    c.id = clause_id
    return c


def _unit_eq(a: Term, b: Term, clause_id: int = 0) -> Clause:
    """Positive unit equality: a = b."""
    eq = _func(1, a, b)  # symnum=1 is '='
    return _clause(_lit(True, eq), clause_id=clause_id)


class FakeComputer:
    """Deterministic embedding computer for tests."""

    def __init__(self, dim: int = 8, device: str = "cpu") -> None:
        self.dim = dim
        self.device = device
        self.call_count = 0

    def compute_embeddings(self, clauses: Sequence[Clause]) -> torch.Tensor:
        self.call_count += 1
        n = len(clauses)
        # Deterministic: hash-based so same clause → same embedding.
        result = torch.zeros(n, self.dim)
        for i, c in enumerate(clauses):
            key = clause_structural_hash(c)
            seed = int(key[:8], 16) % (2**31)
            gen = torch.Generator().manual_seed(seed)
            result[i] = torch.randn(self.dim, generator=gen)
        return result.to(self.device)


# ---------------------------------------------------------------------------
# Structural hashing tests
# ---------------------------------------------------------------------------

class TestStructuralHash:
    def test_identical_clauses_same_hash(self) -> None:
        c1 = _unit_eq(_var(0), _const(2))
        c2 = _unit_eq(_var(0), _const(2))
        assert clause_structural_hash(c1) == clause_structural_hash(c2)

    def test_alpha_equivalent_same_hash(self) -> None:
        """Clauses differing only in variable numbering should collide."""
        c1 = _unit_eq(_var(0), _const(2))
        c2 = _unit_eq(_var(5), _const(2))  # different varnum
        assert clause_structural_hash(c1) == clause_structural_hash(c2)

    def test_different_structure_different_hash(self) -> None:
        c1 = _unit_eq(_var(0), _const(2))
        c2 = _unit_eq(_const(2), _var(0))  # swapped
        assert clause_structural_hash(c1) != clause_structural_hash(c2)

    def test_different_symbols_different_hash(self) -> None:
        c1 = _unit_eq(_const(2), _const(3))
        c2 = _unit_eq(_const(2), _const(4))
        assert clause_structural_hash(c1) != clause_structural_hash(c2)

    def test_clause_id_does_not_affect_hash(self) -> None:
        c1 = _unit_eq(_var(0), _const(2), clause_id=1)
        c2 = _unit_eq(_var(0), _const(2), clause_id=99)
        assert clause_structural_hash(c1) == clause_structural_hash(c2)

    def test_literal_order_does_not_affect_hash(self) -> None:
        """Clause = multiset of literals: order should not matter."""
        lit_a = _lit(True, _const(2))
        lit_b = _lit(False, _const(3))
        c1 = _clause(lit_a, lit_b)
        c2 = _clause(lit_b, lit_a)
        assert clause_structural_hash(c1) == clause_structural_hash(c2)

    def test_sign_matters(self) -> None:
        c1 = _clause(_lit(True, _const(2)))
        c2 = _clause(_lit(False, _const(2)))
        assert clause_structural_hash(c1) != clause_structural_hash(c2)

    def test_two_variables_distinguished(self) -> None:
        """x=y should differ from x=x."""
        c1 = _unit_eq(_var(0), _var(1))
        c2 = _unit_eq(_var(0), _var(0))
        assert clause_structural_hash(c1) != clause_structural_hash(c2)


# ---------------------------------------------------------------------------
# CacheStatistics tests
# ---------------------------------------------------------------------------

class TestCacheStatistics:
    def test_initial_state(self) -> None:
        s = CacheStatistics()
        assert s.hits == 0
        assert s.misses == 0
        assert s.hit_rate == 0.0

    def test_hit_rate_computation(self) -> None:
        s = CacheStatistics()
        for _ in range(8):
            s.record_hit()
        for _ in range(2):
            s.record_miss()
        assert s.hit_rate == pytest.approx(0.8)

    def test_snapshot(self) -> None:
        s = CacheStatistics()
        s.record_hit()
        snap = s.snapshot()
        assert snap["hits"] == 1
        assert snap["hit_rate"] == pytest.approx(1.0)

    def test_reset(self) -> None:
        s = CacheStatistics()
        s.record_hit()
        s.record_miss()
        s.reset()
        assert s.total_lookups == 0


# ---------------------------------------------------------------------------
# EmbeddingCache core tests
# ---------------------------------------------------------------------------

class TestEmbeddingCacheBasic:
    def test_empty_cache(self) -> None:
        cache = EmbeddingCache(CacheConfig(embedding_dim=8))
        assert len(cache) == 0

    def test_put_and_get(self) -> None:
        cache = EmbeddingCache(CacheConfig(embedding_dim=4))
        c = _unit_eq(_var(0), _const(2))
        emb = torch.randn(4)
        cache.put(c, emb)
        assert len(cache) == 1
        result = cache.get(c)
        assert result is not None
        assert torch.equal(result, emb)

    def test_get_miss_returns_none(self) -> None:
        cache = EmbeddingCache(CacheConfig(embedding_dim=4))
        c = _unit_eq(_var(0), _const(2))
        assert cache.get(c) is None

    def test_contains(self) -> None:
        cache = EmbeddingCache(CacheConfig(embedding_dim=4))
        c = _unit_eq(_var(0), _const(2))
        assert c not in cache
        cache.put(c, torch.randn(4))
        assert c in cache

    def test_alpha_equivalent_hit(self) -> None:
        """Structurally identical clause with different varnums should hit."""
        cache = EmbeddingCache(CacheConfig(embedding_dim=4))
        c1 = _unit_eq(_var(0), _const(2))
        emb = torch.randn(4)
        cache.put(c1, emb)

        c2 = _unit_eq(_var(7), _const(2))  # alpha-equivalent
        result = cache.get(c2)
        assert result is not None
        assert torch.equal(result, emb)


# ---------------------------------------------------------------------------
# LRU eviction tests
# ---------------------------------------------------------------------------

class TestLRUEviction:
    def test_evicts_oldest_when_full(self) -> None:
        cache = EmbeddingCache(CacheConfig(max_entries=3, embedding_dim=4))
        clauses = [_unit_eq(_const(i), _const(i + 10)) for i in range(4)]
        for c in clauses:
            cache.put(c, torch.randn(4))

        assert len(cache) == 3
        # First clause should have been evicted.
        assert cache.get(clauses[0]) is None
        # Last three should be present.
        for c in clauses[1:]:
            assert cache.get(c) is not None

    def test_get_refreshes_lru(self) -> None:
        cache = EmbeddingCache(CacheConfig(max_entries=3, embedding_dim=4))
        c0 = _unit_eq(_const(0), _const(10))
        c1 = _unit_eq(_const(1), _const(11))
        c2 = _unit_eq(_const(2), _const(12))
        c3 = _unit_eq(_const(3), _const(13))

        cache.put(c0, torch.randn(4))
        cache.put(c1, torch.randn(4))
        cache.put(c2, torch.randn(4))

        # Touch c0 to refresh it.
        cache.get(c0)

        # Now insert c3 – should evict c1 (oldest untouched).
        cache.put(c3, torch.randn(4))
        assert cache.get(c0) is not None
        assert cache.get(c1) is None

    def test_eviction_counter(self) -> None:
        cache = EmbeddingCache(CacheConfig(max_entries=2, embedding_dim=4))
        for i in range(5):
            cache.put(_unit_eq(_const(i), _const(i + 10)), torch.randn(4))
        assert cache.stats.evictions == 3


# ---------------------------------------------------------------------------
# Batch interface tests
# ---------------------------------------------------------------------------

class TestBatchInterface:
    def test_all_misses(self) -> None:
        computer = FakeComputer(dim=8)
        cache = EmbeddingCache(
            CacheConfig(embedding_dim=8), compute_fn=computer
        )
        clauses = [_unit_eq(_const(i), _const(i + 10)) for i in range(5)]
        result = cache.get_or_compute_batch(clauses)

        assert result.shape == (5, 8)
        assert computer.call_count == 1
        assert cache.stats.misses == 5
        assert cache.stats.hits == 0

    def test_all_hits(self) -> None:
        computer = FakeComputer(dim=8)
        cache = EmbeddingCache(
            CacheConfig(embedding_dim=8), compute_fn=computer
        )
        clauses = [_unit_eq(_const(i), _const(i + 10)) for i in range(3)]

        # Populate cache.
        cache.get_or_compute_batch(clauses)
        computer.call_count = 0

        # Second call should be all hits.
        result = cache.get_or_compute_batch(clauses)
        assert result.shape == (3, 8)
        assert computer.call_count == 0
        assert cache.stats.hits == 3  # 3 hits from second call only

    def test_mixed_hits_and_misses(self) -> None:
        computer = FakeComputer(dim=8)
        cache = EmbeddingCache(
            CacheConfig(embedding_dim=8), compute_fn=computer
        )
        c0 = _unit_eq(_const(0), _const(10))
        c1 = _unit_eq(_const(1), _const(11))
        c2 = _unit_eq(_const(2), _const(12))

        # Warm c0 only.
        cache.get_or_compute_batch([c0])
        computer.call_count = 0

        # Request c0 (hit), c1 (miss), c2 (miss).
        result = cache.get_or_compute_batch([c0, c1, c2])
        assert result.shape == (3, 8)
        assert computer.call_count == 1

    def test_no_compute_fn_raises(self) -> None:
        cache = EmbeddingCache(CacheConfig(embedding_dim=8))
        clauses = [_unit_eq(_const(0), _const(10))]
        with pytest.raises(RuntimeError, match="no compute_fn"):
            cache.get_or_compute_batch(clauses)


# ---------------------------------------------------------------------------
# Cache management tests
# ---------------------------------------------------------------------------

class TestCacheManagement:
    def test_invalidate_all(self) -> None:
        cache = EmbeddingCache(CacheConfig(embedding_dim=4))
        for i in range(5):
            cache.put(_unit_eq(_const(i), _const(i + 10)), torch.randn(4))
        assert len(cache) == 5
        count = cache.invalidate_all()
        assert count == 5
        assert len(cache) == 0

    def test_invalidate_clause(self) -> None:
        cache = EmbeddingCache(CacheConfig(embedding_dim=4))
        c = _unit_eq(_const(0), _const(10))
        cache.put(c, torch.randn(4))
        assert cache.invalidate_clause(c) is True
        assert cache.get(c) is None
        assert cache.invalidate_clause(c) is False

    def test_on_model_update(self) -> None:
        cache = EmbeddingCache(CacheConfig(embedding_dim=4))
        cache.put(_unit_eq(_const(0), _const(10)), torch.randn(4))
        assert cache.model_version == 0
        cache.on_model_update()
        assert cache.model_version == 1
        assert len(cache) == 0


# ---------------------------------------------------------------------------
# Preload / warmup tests
# ---------------------------------------------------------------------------

class TestPreload:
    def test_preload_computes_misses(self) -> None:
        computer = FakeComputer(dim=8)
        cache = EmbeddingCache(
            CacheConfig(embedding_dim=8), compute_fn=computer
        )
        clauses = [_unit_eq(_const(i), _const(i + 10)) for i in range(3)]
        already = cache.preload(clauses)
        assert already == 0
        assert len(cache) == 3
        assert computer.call_count == 1

    def test_preload_skips_existing(self) -> None:
        computer = FakeComputer(dim=8)
        cache = EmbeddingCache(
            CacheConfig(embedding_dim=8), compute_fn=computer
        )
        clauses = [_unit_eq(_const(i), _const(i + 10)) for i in range(3)]
        cache.preload(clauses)
        computer.call_count = 0

        already = cache.preload(clauses)
        assert already == 3
        assert computer.call_count == 0

    def test_preload_empty(self) -> None:
        cache = EmbeddingCache(CacheConfig(embedding_dim=8))
        assert cache.preload([]) == 0


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, tmp_path) -> None:
        path = str(tmp_path / "cache.pt")
        config = CacheConfig(embedding_dim=4, persist_path=path)

        cache1 = EmbeddingCache(config)
        c = _unit_eq(_const(0), _const(10))
        emb = torch.randn(4)
        cache1.put(c, emb)
        cache1.save()

        cache2 = EmbeddingCache(config)
        loaded = cache2.load()
        assert loaded == 1
        result = cache2.get(c)
        assert result is not None
        assert torch.allclose(result, emb)

    def test_load_wrong_dimension_discards(self, tmp_path) -> None:
        path = str(tmp_path / "cache.pt")
        config4 = CacheConfig(embedding_dim=4, persist_path=path)
        cache1 = EmbeddingCache(config4)
        cache1.put(_unit_eq(_const(0), _const(10)), torch.randn(4))
        cache1.save()

        config8 = CacheConfig(embedding_dim=8, persist_path=path)
        cache2 = EmbeddingCache(config8)
        loaded = cache2.load()
        assert loaded == 0

    def test_load_wrong_model_version_discards(self, tmp_path) -> None:
        path = str(tmp_path / "cache.pt")
        config = CacheConfig(embedding_dim=4, persist_path=path)

        cache1 = EmbeddingCache(config)
        cache1.put(_unit_eq(_const(0), _const(10)), torch.randn(4))
        cache1.save()

        cache2 = EmbeddingCache(config)
        cache2.on_model_update()  # version = 1
        loaded = cache2.load()
        assert loaded == 0


# ---------------------------------------------------------------------------
# Thread safety smoke tests
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_reads(self) -> None:
        """Multiple threads reading should not deadlock or corrupt."""
        import concurrent.futures

        cache = EmbeddingCache(CacheConfig(embedding_dim=4))
        clauses = [_unit_eq(_const(i), _const(i + 10)) for i in range(20)]
        for c in clauses:
            cache.put(c, torch.randn(4))

        def reader(idx: int) -> bool:
            c = clauses[idx % len(clauses)]
            return cache.get(c) is not None

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(reader, i) for i in range(100)]
            results = [f.result() for f in futures]
        assert all(results)

    def test_concurrent_writes(self) -> None:
        """Multiple threads writing should not lose entries."""
        import concurrent.futures

        cache = EmbeddingCache(CacheConfig(max_entries=1000, embedding_dim=4))

        def writer(idx: int) -> None:
            c = _unit_eq(_const(idx), _const(idx + 1000))
            cache.put(c, torch.randn(4))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(writer, i) for i in range(100)]
            for f in futures:
                f.result()

        # All 100 unique clauses should be present.
        assert len(cache) == 100
