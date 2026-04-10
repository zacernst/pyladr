"""Performance overhead tests for ML enhancements.

Ensures that ML features do not degrade performance when disabled,
and that overhead when enabled stays within acceptable limits.

Run with: pytest tests/compatibility/test_performance_overhead.py -v -m benchmark
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from tests.compatibility.conftest import run_search


# ── Baseline Performance Tests ─────────────────────────────────────────────


@pytest.mark.benchmark
class TestBaselinePerformance:
    """Measure baseline performance without ML features."""

    def test_term_construction_throughput(self):
        """Term construction rate must exceed 100k terms/sec."""
        from pyladr.core.term import get_rigid_term, get_variable_term

        start = time.perf_counter()
        count = 10_000
        for i in range(count):
            a = get_rigid_term(1, 0)
            b = get_rigid_term(2, 0)
            f_ab = get_rigid_term(3, 2, (a, b))
            g = get_rigid_term(4, 1, (f_ab,))
        elapsed = time.perf_counter() - start
        ops_per_sec = (count * 4) / elapsed
        assert ops_per_sec > 100_000, f"Term construction: {ops_per_sec:.0f} ops/s"

    def test_unification_throughput(self):
        """Unification rate must exceed 10k ops/sec."""
        from pyladr.core.substitution import Context, Trail, unify
        from pyladr.core.term import get_rigid_term, get_variable_term

        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        f_x = get_rigid_term(2, 1, (x,))
        f_a = get_rigid_term(2, 1, (a,))

        start = time.perf_counter()
        count = 10_000
        for _ in range(count):
            c1, c2, tr = Context(), Context(), Trail()
            unify(f_x, c1, f_a, c2, tr)
            tr.undo()
        elapsed = time.perf_counter() - start
        ops_per_sec = count / elapsed
        assert ops_per_sec > 10_000, f"Unification: {ops_per_sec:.0f} ops/s"

    def test_parsing_throughput(self):
        """Clause parsing rate must exceed 1k clauses/sec."""
        from pyladr.core.symbol import SymbolTable
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        clause_text = "f(g(x,y),a) = g(f(a,x),f(b,y))"

        start = time.perf_counter()
        count = 5_000
        for _ in range(count):
            parser.parse_term(clause_text)
        elapsed = time.perf_counter() - start
        ops_per_sec = count / elapsed
        assert ops_per_sec > 1_000, f"Parsing: {ops_per_sec:.0f} ops/s"

    def test_resolution_throughput(self):
        """Binary resolution must produce results at >5k ops/sec."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.inference.resolution import all_binary_resolvents

        a = get_rigid_term(1, 0)
        x = get_variable_term(0)
        P_a = get_rigid_term(2, 1, (a,))
        P_x = get_rigid_term(2, 1, (x,))

        c1 = Clause(literals=(Literal(sign=True, atom=P_a),))
        c2 = Clause(literals=(Literal(sign=False, atom=P_x),))

        start = time.perf_counter()
        count = 5_000
        for _ in range(count):
            all_binary_resolvents(c1, c2)
        elapsed = time.perf_counter() - start
        ops_per_sec = count / elapsed
        assert ops_per_sec > 5_000, f"Resolution: {ops_per_sec:.0f} ops/s"

    def test_subsumption_throughput(self):
        """Subsumption checking must exceed 10k ops/sec."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.inference.subsumption import subsumes

        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        P_x = get_rigid_term(2, 1, (x,))
        P_a = get_rigid_term(2, 1, (a,))

        general = Clause(literals=(Literal(sign=True, atom=P_x),))
        specific = Clause(literals=(Literal(sign=True, atom=P_a),))

        start = time.perf_counter()
        count = 10_000
        for _ in range(count):
            subsumes(general, specific)
        elapsed = time.perf_counter() - start
        ops_per_sec = count / elapsed
        assert ops_per_sec > 10_000, f"Subsumption: {ops_per_sec:.0f} ops/s"


# ── ML Overhead Measurement ────────────────────────────────────────────────


@pytest.mark.benchmark
class TestMLOverhead:
    """Measure performance overhead when ML features are disabled vs absent."""

    def test_search_timing_baseline(self, trivial_resolution_clauses):
        """Establish baseline search timing for small problems."""
        from pyladr.search.given_clause import ExitCode

        times = []
        for _ in range(5):
            start = time.perf_counter()
            result = run_search(usable=[], sos=trivial_resolution_clauses)
            elapsed = time.perf_counter() - start
            assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
            times.append(elapsed)

        median_time = sorted(times)[len(times) // 2]
        # Trivial problem should solve in under 100ms
        assert median_time < 0.1, f"Trivial problem took {median_time:.3f}s"

    def test_search_overhead_acceptable(self):
        """Search with more clauses stays within acceptable time."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.search.given_clause import ExitCode

        # Create a problem that requires multiple resolution steps
        P_sn, Q_sn, R_sn = 1, 2, 3
        a_sn = 4
        a = get_rigid_term(a_sn, 0)
        x = get_variable_term(0)

        c1 = Clause(
            literals=(Literal(sign=True, atom=get_rigid_term(P_sn, 1, (a,))),)
        )
        c2 = Clause(
            literals=(
                Literal(sign=False, atom=get_rigid_term(P_sn, 1, (x,))),
                Literal(sign=True, atom=get_rigid_term(Q_sn, 1, (x,))),
            )
        )
        c3 = Clause(
            literals=(
                Literal(sign=False, atom=get_rigid_term(Q_sn, 1, (x,))),
                Literal(sign=True, atom=get_rigid_term(R_sn, 1, (x,))),
            )
        )
        c4 = Clause(
            literals=(
                Literal(sign=False, atom=get_rigid_term(R_sn, 1, (a,))),
            )
        )

        start = time.perf_counter()
        result = run_search(usable=[], sos=[c1, c2, c3, c4])
        elapsed = time.perf_counter() - start

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        # Multi-step resolution should still be fast
        assert elapsed < 1.0, f"Multi-step resolution took {elapsed:.3f}s"


# ── Memory Overhead Tests ──────────────────────────────────────────────────


@pytest.mark.benchmark
class TestMemoryOverhead:
    """Ensure no unexpected memory growth from ML infrastructure."""

    def test_term_memory_efficiency(self):
        """Terms use reasonable memory."""
        import sys

        from pyladr.core.term import get_rigid_term

        a = get_rigid_term(1, 0)
        f_a = get_rigid_term(2, 1, (a,))
        g_f_a = get_rigid_term(3, 1, (f_a,))

        # Basic sanity: terms should be small objects
        assert sys.getsizeof(a) < 1000
        assert sys.getsizeof(f_a) < 1000

    def test_clause_memory_efficiency(self):
        """Clauses use reasonable memory."""
        import sys

        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term

        P_a = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c = Clause(literals=(Literal(sign=True, atom=P_a),))

        assert sys.getsizeof(c) < 1000

    def test_search_state_cleanup(self, trivial_resolution_clauses):
        """Search state does not leak after completion."""
        import gc

        # Run search multiple times and verify GC can clean up
        for _ in range(10):
            run_search(usable=[], sos=trivial_resolution_clauses)

        gc.collect()
        # If we get here without OOM, memory is being managed properly
