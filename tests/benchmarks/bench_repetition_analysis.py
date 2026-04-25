"""Microbenchmarks for subformula repetition penalty hot paths.

Measures the performance-critical functions in repetition_penalty.py
to establish baselines and quantify optimization impact.

Targets Christopher's Phase 1/2 implementation:
  - compute_repetition_penalty() main entry point
  - _penalty_exact() using Term frozen hash/eq
  - _penalty_normalized() using _normalize_variables() DFS traversal
  - Integration path via penalty_override in _cl_process()

Usage:
    python -m tests.benchmarks.bench_repetition_analysis
    python -m tests.benchmarks.bench_repetition_analysis --iterations 5000
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.search.repetition_penalty import (
    RepetitionPenaltyConfig,
    compute_repetition_penalty,
    _penalty_exact,
    _penalty_normalized,
    _normalize_variables,
)


# ── Test clause construction ─────────────────────────────────────────────────

# Symbol IDs
P, Q, R = 1, 2, 3              # predicates
f_sym, g_sym, h_sym = 4, 5, 6  # functions
a_sym, b_sym, c_sym = 7, 8, 9  # constants


def _var(n: int) -> Term:
    return get_variable_term(n)


def _const(s: int) -> Term:
    return get_rigid_term(s, 0)


def _fn(s: int, *args: Term) -> Term:
    return get_rigid_term(s, len(args), tuple(args))


def _lit(sign: bool, atom: Term) -> Literal:
    return Literal(sign=sign, atom=atom)


def _clause(*lits: Literal) -> Clause:
    c = Clause(literals=tuple(lits))
    c.weight = sum(lit.atom.symbol_count for lit in lits)
    return c


# ── Clause generators with known repetition patterns ─────────────────────────


def build_no_repetition() -> Clause:
    """P(f(x, a), g(b, y)) -- no repeated subterms. 7 symbols."""
    atom = _fn(P, _fn(f_sym, _var(0), _const(a_sym)), _fn(g_sym, _const(b_sym), _var(1)))
    return _clause(_lit(True, atom))


def build_exact_repetition() -> Clause:
    """P(f(x,a), f(x,a)) -- exact repeated subterm f(x,a). 7 symbols."""
    sub = _fn(f_sym, _var(0), _const(a_sym))
    atom = _fn(P, sub, sub)
    return _clause(_lit(True, atom))


def build_var_only_repetition() -> Clause:
    """P(f(x,y), f(z,w)) -- same structure but different vars (normalized match only). 7 symbols."""
    atom = _fn(P, _fn(f_sym, _var(0), _var(1)), _fn(f_sym, _var(2), _var(3)))
    return _clause(_lit(True, atom))


def build_deep_repetition() -> Clause:
    """P(g(f(x,a), b), g(f(y,a), b)) -- deeper nesting, var-only difference. ~11 symbols."""
    left = _fn(g_sym, _fn(f_sym, _var(0), _const(a_sym)), _const(b_sym))
    right = _fn(g_sym, _fn(f_sym, _var(1), _const(a_sym)), _const(b_sym))
    atom = _fn(P, left, right)
    return _clause(_lit(True, atom))


def build_multi_literal_repetition() -> Clause:
    """P(f(x,a)) | Q(f(y,a)) | -R(f(z,a)) -- same subterm across literals. 12 symbols."""
    a1 = _fn(P, _fn(f_sym, _var(0), _const(a_sym)))
    a2 = _fn(Q, _fn(f_sym, _var(1), _const(a_sym)))
    a3 = _fn(R, _fn(f_sym, _var(2), _const(a_sym)))
    return _clause(_lit(True, a1), _lit(True, a2), _lit(False, a3))


def build_large_clause_no_rep() -> Clause:
    """Large clause with no repetition: P(f(g(x,a),h(b,y)), g(h(c,z),f(a,b))). ~15 symbols."""
    left = _fn(f_sym, _fn(g_sym, _var(0), _const(a_sym)), _fn(h_sym, _const(b_sym), _var(1)))
    right = _fn(g_sym, _fn(h_sym, _const(c_sym), _var(2)), _fn(f_sym, _const(a_sym), _const(b_sym)))
    atom = _fn(P, left, right)
    return _clause(_lit(True, atom))


def build_large_clause_heavy_rep() -> Clause:
    """Large clause with heavy repetition across 3 literals, ~20 symbols.
    P(f(x,a), f(x,a)) | Q(f(y,a), g(b,b)) | -R(f(z,a))
    f(_,a) appears 4 times (3 normalized-equivalent + 1 exact match).
    """
    sub1 = _fn(f_sym, _var(0), _const(a_sym))
    a1 = _fn(P, sub1, sub1)
    a2 = _fn(Q, _fn(f_sym, _var(1), _const(a_sym)), _fn(g_sym, _const(b_sym), _const(b_sym)))
    a3 = _fn(R, _fn(f_sym, _var(2), _const(a_sym)))
    return _clause(_lit(True, a1), _lit(True, a2), _lit(False, a3))


def build_small_clause() -> Clause:
    """P(x,a) -- tiny clause, should trigger early termination. 3 symbols."""
    return _clause(_lit(True, _fn(P, _var(0), _const(a_sym))))


def build_batch_clauses(n: int) -> list[Clause]:
    """Build a mixed batch of n clauses for throughput testing."""
    generators = [
        build_no_repetition,
        build_exact_repetition,
        build_var_only_repetition,
        build_deep_repetition,
        build_multi_literal_repetition,
        build_large_clause_no_rep,
        build_large_clause_heavy_rep,
        build_small_clause,
    ]
    return [generators[i % len(generators)]() for i in range(n)]


# ── Benchmark infrastructure ─────────────────────────────────────────────────


@dataclass
class BenchResult:
    """Result of a single microbenchmark."""
    name: str
    iterations: int
    total_ns: float
    per_call_ns: float
    median_ns: float
    stdev_ns: float
    min_ns: float
    max_ns: float

    def report(self) -> str:
        return (
            f"  {self.name:<55} "
            f"med={self.median_ns:>9.0f}ns  "
            f"avg={self.per_call_ns:>9.0f}ns  "
            f"min={self.min_ns:>9.0f}ns  "
            f"max={self.max_ns:>9.0f}ns  "
            f"({self.iterations} iters)"
        )


def bench(name: str, fn, *, iterations: int = 1000, warmup: int = 100) -> BenchResult:
    """Run a microbenchmark with warmup and statistical collection."""
    for _ in range(warmup):
        fn()

    batch_size = 10
    num_batches = max(1, iterations // batch_size)
    per_call_times: list[float] = []

    for _ in range(num_batches):
        start = time.perf_counter_ns()
        for _ in range(batch_size):
            fn()
        elapsed = time.perf_counter_ns() - start
        per_call_times.append(elapsed / batch_size)

    total_calls = num_batches * batch_size
    total_ns = sum(per_call_times) * batch_size

    return BenchResult(
        name=name,
        iterations=total_calls,
        total_ns=total_ns,
        per_call_ns=statistics.mean(per_call_times),
        median_ns=statistics.median(per_call_times),
        stdev_ns=statistics.stdev(per_call_times) if len(per_call_times) > 1 else 0,
        min_ns=min(per_call_times),
        max_ns=max(per_call_times),
    )


# ── Benchmark functions ──────────────────────────────────────────────────────


def bench_penalty_exact(iterations: int) -> list[BenchResult]:
    """Benchmark _penalty_exact() with various clause patterns."""
    results = []
    config = RepetitionPenaltyConfig(enabled=True, base_penalty=2.0, min_subterm_size=2)

    cases = [
        ("exact: small clause (early term)", build_small_clause()),
        ("exact: no repetition (7 nodes)", build_no_repetition()),
        ("exact: exact repetition (7 nodes)", build_exact_repetition()),
        ("exact: var-only diff (7 nodes)", build_var_only_repetition()),
        ("exact: deep nesting (11 nodes)", build_deep_repetition()),
        ("exact: multi-literal (12 nodes)", build_multi_literal_repetition()),
        ("exact: large no-rep (15 nodes)", build_large_clause_no_rep()),
        ("exact: large heavy-rep (20 nodes)", build_large_clause_heavy_rep()),
    ]

    for label, clause in cases:
        results.append(bench(
            f"_penalty_exact: {label}",
            lambda c=clause: _penalty_exact(c, config),
            iterations=iterations,
        ))

    return results


def bench_penalty_normalized(iterations: int) -> list[BenchResult]:
    """Benchmark _penalty_normalized() with various clause patterns."""
    results = []
    config = RepetitionPenaltyConfig(
        enabled=True, base_penalty=2.0, min_subterm_size=2, normalize_variables=True,
    )

    cases = [
        ("normalized: no repetition (7 nodes)", build_no_repetition()),
        ("normalized: exact repetition (7 nodes)", build_exact_repetition()),
        ("normalized: var-only diff (7 nodes)", build_var_only_repetition()),
        ("normalized: deep nesting (11 nodes)", build_deep_repetition()),
        ("normalized: multi-literal (12 nodes)", build_multi_literal_repetition()),
        ("normalized: large no-rep (15 nodes)", build_large_clause_no_rep()),
        ("normalized: large heavy-rep (20 nodes)", build_large_clause_heavy_rep()),
    ]

    for label, clause in cases:
        results.append(bench(
            f"_penalty_normalized: {label}",
            lambda c=clause: _penalty_normalized(c, config),
            iterations=iterations,
        ))

    return results


def bench_normalize_variables(iterations: int) -> list[BenchResult]:
    """Benchmark _normalize_variables() DFS traversal at different term sizes."""
    results = []

    # Small: f(x, a) -- 3 nodes
    small = _fn(f_sym, _var(0), _const(a_sym))
    results.append(bench(
        "_normalize_variables: small (3 nodes)",
        lambda: _normalize_variables(small),
        iterations=iterations,
    ))

    # Medium: g(f(x,a), h(f(y,b), g(z,c))) -- 11 nodes
    medium = _fn(g_sym,
        _fn(f_sym, _var(0), _const(a_sym)),
        _fn(h_sym, _fn(f_sym, _var(1), _const(b_sym)), _fn(g_sym, _var(2), _const(c_sym))))
    results.append(bench(
        "_normalize_variables: medium (11 nodes)",
        lambda: _normalize_variables(medium),
        iterations=iterations,
    ))

    # Large: deeply nested ~20 nodes
    large = _fn(f_sym,
        _fn(g_sym,
            _fn(h_sym, _fn(f_sym, _var(0), _const(a_sym)), _fn(g_sym, _var(1), _const(b_sym))),
            _fn(f_sym, _fn(g_sym, _var(2), _const(c_sym)), _const(a_sym))),
        _fn(h_sym,
            _fn(f_sym, _var(3), _fn(g_sym, _const(b_sym), _const(c_sym))),
            _fn(g_sym, _var(4), _const(a_sym))))
    results.append(bench(
        "_normalize_variables: large (20+ nodes)",
        lambda: _normalize_variables(large),
        iterations=iterations,
    ))

    # Ground term (no vars to map -- fast path?)
    ground = _fn(f_sym, _fn(g_sym, _const(a_sym), _const(b_sym)), _const(c_sym))
    results.append(bench(
        "_normalize_variables: ground (no vars, 5 nodes)",
        lambda: _normalize_variables(ground),
        iterations=iterations,
    ))

    return results


def bench_term_subterms(iterations: int) -> list[BenchResult]:
    """Benchmark Term.subterms() generator -- the traversal engine."""
    results = []

    small = _fn(P, _var(0), _const(a_sym))
    results.append(bench(
        "Term.subterms(): small (3 nodes)",
        lambda: list(small.subterms()),
        iterations=iterations,
    ))

    medium = _fn(P, _fn(f_sym, _var(0), _var(1)), _fn(g_sym, _const(a_sym), _const(b_sym)))
    results.append(bench(
        "Term.subterms(): medium (7 nodes)",
        lambda: list(medium.subterms()),
        iterations=iterations,
    ))

    large = _fn(P,
        _fn(f_sym,
            _fn(g_sym, _var(0), _const(a_sym)),
            _fn(h_sym, _fn(f_sym, _var(1), _const(b_sym)), _fn(g_sym, _var(2), _const(c_sym)))),
        _fn(f_sym, _var(0), _var(1)))
    results.append(bench(
        "Term.subterms(): large (15+ nodes)",
        lambda: list(large.subterms()),
        iterations=iterations,
    ))

    return results


def bench_term_hash_eq(iterations: int) -> list[BenchResult]:
    """Benchmark Term hash/eq -- the foundation of _penalty_exact dict ops."""
    results = []

    # Hash of medium term
    t1 = _fn(f_sym, _fn(g_sym, _var(0), _const(a_sym)), _const(b_sym))
    results.append(bench(
        "hash(Term): medium (5 nodes)",
        lambda: hash(t1),
        iterations=iterations,
    ))

    # Hash of large term
    t2 = _fn(f_sym,
        _fn(g_sym, _fn(h_sym, _var(0), _const(a_sym)), _const(b_sym)),
        _fn(h_sym, _var(1), _fn(g_sym, _const(c_sym), _var(2))))
    results.append(bench(
        "hash(Term): large (11 nodes)",
        lambda: hash(t2),
        iterations=iterations,
    ))

    # Equality comparison (same structure, different objects)
    t3a = _fn(f_sym, _fn(g_sym, _var(0), _const(a_sym)), _const(b_sym))
    t3b = _fn(f_sym, _fn(g_sym, _var(0), _const(a_sym)), _const(b_sym))
    results.append(bench(
        "Term == Term: equal (5 nodes)",
        lambda: t3a == t3b,
        iterations=iterations,
    ))

    # Equality comparison (different structure)
    t4 = _fn(f_sym, _fn(g_sym, _var(1), _const(b_sym)), _const(a_sym))
    results.append(bench(
        "Term == Term: not equal (5 nodes)",
        lambda: t3a == t4,
        iterations=iterations,
    ))

    return results


def bench_compute_repetition_penalty(iterations: int) -> list[BenchResult]:
    """Benchmark the main entry point compute_repetition_penalty()."""
    results = []

    config_exact = RepetitionPenaltyConfig(enabled=True, base_penalty=2.0, min_subterm_size=2)
    config_norm = RepetitionPenaltyConfig(
        enabled=True, base_penalty=2.0, min_subterm_size=2, normalize_variables=True,
    )

    # Exact mode
    clause = build_large_clause_heavy_rep()
    results.append(bench(
        "compute_repetition_penalty: exact, heavy rep",
        lambda: compute_repetition_penalty(clause, config_exact),
        iterations=iterations,
    ))

    results.append(bench(
        "compute_repetition_penalty: normalized, heavy rep",
        lambda: compute_repetition_penalty(clause, config_norm),
        iterations=iterations,
    ))

    # Early termination (small clause)
    small = build_small_clause()
    results.append(bench(
        "compute_repetition_penalty: early termination",
        lambda: compute_repetition_penalty(small, config_exact),
        iterations=iterations,
    ))

    return results


def bench_throughput(iterations: int) -> list[BenchResult]:
    """Benchmark batch throughput simulating _cl_process integration."""
    results = []
    clauses = build_batch_clauses(100)

    config_exact = RepetitionPenaltyConfig(enabled=True, base_penalty=2.0, min_subterm_size=2)

    def process_batch_exact():
        for c in clauses:
            compute_repetition_penalty(c, config_exact)

    results.append(bench(
        "throughput: 100 mixed clauses (exact mode)",
        process_batch_exact,
        iterations=min(iterations, 500),
    ))

    config_norm = RepetitionPenaltyConfig(
        enabled=True, base_penalty=2.0, min_subterm_size=2, normalize_variables=True,
    )

    def process_batch_normalized():
        for c in clauses:
            compute_repetition_penalty(c, config_norm)

    results.append(bench(
        "throughput: 100 mixed clauses (normalized mode)",
        process_batch_normalized,
        iterations=min(iterations, 500),
    ))

    return results


# ── Main ─────────────────────────────────────────────────────────────────────


def run_all_benchmarks(iterations: int = 2000) -> list[BenchResult]:
    """Run all repetition penalty benchmarks."""
    all_results: list[BenchResult] = []

    print("=" * 90)
    print("SUBFORMULA REPETITION PENALTY MICROBENCHMARKS")
    print(f"  iterations={iterations}")
    print("=" * 90)

    sections = [
        ("Term.subterms() Traversal", bench_term_subterms),
        ("Term Hash/Eq (dict key path)", bench_term_hash_eq),
        ("Phase 1: _penalty_exact()", bench_penalty_exact),
        ("Phase 2: _normalize_variables()", bench_normalize_variables),
        ("Phase 2: _penalty_normalized()", bench_penalty_normalized),
        ("Main Entry: compute_repetition_penalty()", bench_compute_repetition_penalty),
        ("Batch Throughput (simulated _cl_process)", bench_throughput),
    ]

    for section_name, bench_fn in sections:
        print(f"\n--- {section_name} ---")
        section_results = bench_fn(iterations)
        for r in section_results:
            print(r.report())
        all_results.extend(section_results)

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    sorted_results = sorted(all_results, key=lambda r: r.median_ns, reverse=True)
    print("\nTop 5 slowest operations (optimization targets):")
    for i, r in enumerate(sorted_results[:5], 1):
        us = r.median_ns / 1000
        print(f"  {i}. {r.name}: {r.median_ns:,.0f} ns ({us:,.1f} μs)")

    # Phase 1 vs Phase 2 comparison
    exact_heavy = next((r for r in all_results if "exact, heavy rep" in r.name), None)
    norm_heavy = next((r for r in all_results if "normalized, heavy rep" in r.name), None)
    if exact_heavy and norm_heavy:
        ratio = norm_heavy.median_ns / exact_heavy.median_ns
        print(f"\nPhase 2 overhead vs Phase 1: {ratio:.1f}x (normalized/exact on heavy-rep clause)")

    exact_tp = next((r for r in all_results if "exact mode" in r.name and "throughput" in r.name), None)
    norm_tp = next((r for r in all_results if "normalized mode" in r.name and "throughput" in r.name), None)
    if exact_tp and norm_tp:
        ratio = norm_tp.median_ns / exact_tp.median_ns
        exact_per = exact_tp.median_ns / 100_000  # per clause in μs
        norm_per = norm_tp.median_ns / 100_000
        print(f"Batch throughput: exact={exact_per:.1f} μs/clause, normalized={norm_per:.1f} μs/clause ({ratio:.1f}x overhead)")

    return all_results


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Repetition penalty microbenchmarks")
    parser.add_argument("--iterations", type=int, default=2000, help="Iterations per benchmark")
    args = parser.parse_args()

    run_all_benchmarks(args.iterations)


if __name__ == "__main__":
    main()
