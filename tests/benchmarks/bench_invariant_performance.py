"""Performance benchmark: invariant embeddings vs original.

Measures overhead of property-invariant feature extraction, graph construction,
structural hashing, and full provider pipeline against the original system.

Target: <10% overhead per Task #3 acceptance criteria.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term


# ── Clause factories ──────────────────────────────────────────────────────


def make_term(symnum: int, *args: Term) -> Term:
    return Term(private_symbol=-symnum, arity=len(args), args=tuple(args))


def make_var(varnum: int) -> Term:
    return Term(private_symbol=varnum)


def make_literal(sign: bool, atom: Term) -> Literal:
    return Literal(sign=sign, atom=atom)


def make_clause(literals: list[Literal], clause_id: int = 1) -> Clause:
    return Clause(id=clause_id, literals=tuple(literals), justification=())


def generate_simple_clauses(n: int) -> list[Clause]:
    """P_i(x, c_i) for diverse symbol IDs."""
    clauses = []
    for i in range(n):
        x = make_var(0)
        c = make_term(100 + i)
        atom = make_term(1 + i, x, c)
        clauses.append(make_clause([make_literal(True, atom)], clause_id=i))
    return clauses


def generate_complex_clauses(n: int) -> list[Clause]:
    """Multi-literal clauses with nested terms."""
    clauses = []
    for i in range(n):
        x, y, z = make_var(0), make_var(1), make_var(2)
        f_x = make_term(10 + i, x)
        g_y_z = make_term(20 + i, y, z)
        h_fx_gyz = make_term(30 + i, f_x, g_y_z)
        lit1 = make_literal(True, make_term(1 + i, h_fx_gyz, x))
        lit2 = make_literal(False, make_term(2 + i, y, make_term(40 + i, z)))
        lit3 = make_literal(True, make_term(3 + i, x, y, z))
        clauses.append(make_clause([lit1, lit2, lit3], clause_id=i))
    return clauses


def generate_renamed_pairs(n: int) -> list[tuple[Clause, Clause]]:
    """Pairs of structurally identical clauses with different symbol names."""
    pairs = []
    for i in range(n):
        x, y = make_var(0), make_var(1)
        atom1 = make_term(1 + i, make_term(100 + i, x), y)
        c1 = make_clause([make_literal(True, atom1)], clause_id=i)

        atom2 = make_term(500 + i, make_term(600 + i, x), y)
        c2 = make_clause([make_literal(True, atom2)], clause_id=1000 + i)
        pairs.append((c1, c2))
    return pairs


# ── Timing utilities ──────────────────────────────────────────────────────


@dataclass
class BenchResult:
    name: str
    original_ms: float
    invariant_ms: float
    overhead_pct: float
    iterations: int

    def __str__(self) -> str:
        status = "PASS" if self.overhead_pct < 10.0 else "FAIL"
        return (
            f"  {self.name:<45} "
            f"orig={self.original_ms:8.3f}ms  "
            f"inv={self.invariant_ms:8.3f}ms  "
            f"overhead={self.overhead_pct:+6.1f}%  "
            f"[{status}]"
        )


def time_fn(fn, iterations: int = 100) -> float:
    """Time a function over multiple iterations, return median ms."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return statistics.median(times)


# ── Benchmarks ────────────────────────────────────────────────────────────


def bench_graph_construction() -> list[BenchResult]:
    """Compare graph construction: original vs invariant."""
    from pyladr.ml.graph.clause_graph import (
        ClauseGraphConfig,
        batch_clauses_to_heterograph,
    )
    from pyladr.ml.invariant.invariant_graph import (
        batch_invariant_clauses_to_heterograph,
    )

    results = []
    cfg = ClauseGraphConfig()

    for label, clauses in [
        ("simple (100 clauses)", generate_simple_clauses(100)),
        ("complex (100 clauses)", generate_complex_clauses(100)),
        ("simple (500 clauses)", generate_simple_clauses(500)),
        ("complex (500 clauses)", generate_complex_clauses(500)),
    ]:
        orig_ms = time_fn(
            lambda c=clauses: batch_clauses_to_heterograph(c, config=cfg),
            iterations=50,
        )
        inv_ms = time_fn(
            lambda c=clauses: batch_invariant_clauses_to_heterograph(c, config=cfg),
            iterations=50,
        )
        overhead = ((inv_ms - orig_ms) / orig_ms) * 100 if orig_ms > 0 else 0
        results.append(BenchResult(
            name=f"graph_construction/{label}",
            original_ms=orig_ms,
            invariant_ms=inv_ms,
            overhead_pct=overhead,
            iterations=50,
        ))

    return results


def bench_structural_hashing() -> list[BenchResult]:
    """Compare structural hashing: original vs invariant."""
    from pyladr.ml.embeddings.cache import clause_structural_hash
    from pyladr.ml.invariant.invariant_features import (
        invariant_clause_structural_hash,
    )

    results = []

    for label, clauses in [
        ("simple (500 clauses)", generate_simple_clauses(500)),
        ("complex (500 clauses)", generate_complex_clauses(500)),
    ]:
        orig_ms = time_fn(
            lambda c=clauses: [clause_structural_hash(cl) for cl in c],
            iterations=100,
        )
        inv_ms = time_fn(
            lambda c=clauses: [invariant_clause_structural_hash(cl) for cl in c],
            iterations=100,
        )
        overhead = ((inv_ms - orig_ms) / orig_ms) * 100 if orig_ms > 0 else 0
        results.append(BenchResult(
            name=f"structural_hash/{label}",
            original_ms=orig_ms,
            invariant_ms=inv_ms,
            overhead_pct=overhead,
            iterations=100,
        ))

    return results


def bench_cache_sharing() -> BenchResult:
    """Measure cache hit rate improvement from invariant hashing on renamed pairs."""
    from pyladr.ml.invariant.invariant_provider import InvariantEmbeddingProvider
    from pyladr.ml.embedding_provider import GNNEmbeddingProvider

    pairs = generate_renamed_pairs(50)
    first_clauses = [p[0] for p in pairs]
    second_clauses = [p[1] for p in pairs]

    # Original provider: renamed clauses are cache misses
    orig_provider = GNNEmbeddingProvider.create()
    orig_provider.get_embeddings_batch(first_clauses)
    orig_stats_before = orig_provider.stats.copy()
    orig_provider.get_embeddings_batch(second_clauses)
    orig_stats_after = orig_provider.stats.copy()
    orig_second_hits = orig_stats_after["hits"] - orig_stats_before["hits"]

    # Invariant provider: renamed clauses should be cache hits
    inv_provider = InvariantEmbeddingProvider.create()
    inv_provider.get_embeddings_batch(first_clauses)
    inv_stats_before = inv_provider.stats.copy()
    inv_provider.get_embeddings_batch(second_clauses)
    inv_stats_after = inv_provider.stats.copy()
    inv_second_hits = inv_stats_after["hits"] - inv_stats_before["hits"]

    return BenchResult(
        name=f"cache_sharing (50 renamed pairs: orig_hits={orig_second_hits}, inv_hits={inv_second_hits})",
        original_ms=float(orig_second_hits),
        invariant_ms=float(inv_second_hits),
        overhead_pct=0.0,  # Not an overhead metric — higher is better
        iterations=1,
    )


def bench_full_provider() -> list[BenchResult]:
    """End-to-end provider benchmark: embedding computation."""
    from pyladr.ml.embedding_provider import GNNEmbeddingProvider
    from pyladr.ml.invariant.invariant_provider import InvariantEmbeddingProvider

    results = []

    for label, clauses in [
        ("simple (50 clauses)", generate_simple_clauses(50)),
        ("complex (50 clauses)", generate_complex_clauses(50)),
    ]:
        orig_provider = GNNEmbeddingProvider.create()
        inv_provider = InvariantEmbeddingProvider.create()

        orig_ms = time_fn(
            lambda p=orig_provider, c=clauses: p.compute_embeddings(c),
            iterations=30,
        )
        inv_ms = time_fn(
            lambda p=inv_provider, c=clauses: p.compute_embeddings(c),
            iterations=30,
        )
        overhead = ((inv_ms - orig_ms) / orig_ms) * 100 if orig_ms > 0 else 0
        results.append(BenchResult(
            name=f"full_provider/{label}",
            original_ms=orig_ms,
            invariant_ms=inv_ms,
            overhead_pct=overhead,
            iterations=30,
        ))

    return results


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 90)
    print("Property-Invariant Embeddings Performance Benchmark")
    print("Target: <10% overhead vs original embedding system")
    print("=" * 90)

    all_results: list[BenchResult] = []

    print("\n[1/4] Graph Construction")
    print("-" * 90)
    for r in bench_graph_construction():
        print(r)
        all_results.append(r)

    print("\n[2/4] Structural Hashing")
    print("-" * 90)
    for r in bench_structural_hashing():
        print(r)
        all_results.append(r)

    print("\n[3/4] Cache Sharing (renamed clause pairs)")
    print("-" * 90)
    cache_result = bench_cache_sharing()
    print(f"  Original provider cache hits on renamed clauses: {int(cache_result.original_ms)}/50")
    print(f"  Invariant provider cache hits on renamed clauses: {int(cache_result.invariant_ms)}/50")
    improvement = int(cache_result.invariant_ms) - int(cache_result.original_ms)
    print(f"  Cache sharing improvement: +{improvement} hits from symbol-independent hashing")

    print("\n[4/4] Full Provider Pipeline (end-to-end)")
    print("-" * 90)
    for r in bench_full_provider():
        print(r)
        all_results.append(r)

    # Summary
    overhead_results = [r for r in all_results if r.original_ms > 0.01]
    max_overhead = max(r.overhead_pct for r in overhead_results) if overhead_results else 0
    avg_overhead = statistics.mean(r.overhead_pct for r in overhead_results) if overhead_results else 0

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  Max overhead:     {max_overhead:+.1f}%")
    print(f"  Average overhead: {avg_overhead:+.1f}%")
    print(f"  Target:           <10%")
    print(f"  Result:           {'PASS' if max_overhead < 10.0 else 'FAIL'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
