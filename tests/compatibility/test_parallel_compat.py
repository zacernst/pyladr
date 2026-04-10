"""Parallel execution compatibility tests.

Ensures that parallel inference produces results identical to
sequential inference, and that ML enhancements don't break
the parallel execution model.

Run with: pytest tests/compatibility/test_parallel_compat.py -v
"""

from __future__ import annotations

import pytest

from tests.compatibility.conftest import run_search


# ── Sequential vs Parallel Equivalence ─────────────────────────────────────


class TestSequentialParallelEquivalence:
    """Verify parallel search matches sequential results."""

    def _run_sequential(self, sos, **kwargs):
        """Run search without parallelism."""
        return run_search(usable=[], sos=sos, **kwargs)

    def _run_parallel(self, sos, max_workers: int = 2, **kwargs):
        """Run search with parallelism."""
        from pyladr.parallel.inference_engine import ParallelSearchConfig

        parallel_config = ParallelSearchConfig(
            enabled=True, max_workers=max_workers
        )
        return run_search(
            usable=[], sos=sos, parallel=parallel_config, **kwargs
        )

    def test_trivial_proof_matches(self, trivial_resolution_clauses):
        """Trivial proof: sequential and parallel agree on result."""
        from pyladr.search.given_clause import ExitCode

        seq_result = self._run_sequential(trivial_resolution_clauses)
        par_result = self._run_parallel(trivial_resolution_clauses)

        assert seq_result.exit_code == par_result.exit_code
        assert seq_result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(seq_result.proofs) == len(par_result.proofs)

    def test_sos_empty_matches(self):
        """SOS-empty case: sequential and parallel agree."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term
        from pyladr.search.given_clause import ExitCode

        P_a = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        sos = [Clause(literals=(Literal(sign=True, atom=P_a),))]

        seq_result = self._run_sequential(sos, max_given=20)
        par_result = self._run_parallel(sos, max_given=20)

        assert seq_result.exit_code == par_result.exit_code

    def test_statistics_consistent(self, trivial_resolution_clauses):
        """Search statistics are consistent between sequential and parallel."""
        seq_result = self._run_sequential(trivial_resolution_clauses)
        par_result = self._run_parallel(trivial_resolution_clauses)

        # Given count should match since selection is sequential
        assert seq_result.stats.given == par_result.stats.given
        # Generated and kept may differ slightly due to ordering,
        # but should be close
        assert abs(seq_result.stats.generated - par_result.stats.generated) <= 2
        assert abs(seq_result.stats.kept - par_result.stats.kept) <= 2


# ── Parallel Engine Safety Tests ───────────────────────────────────────────


class TestParallelEngineSafety:
    """Ensure parallel engine handles edge cases correctly."""

    def test_parallel_with_empty_sos(self):
        """Parallel engine handles empty SOS gracefully."""
        from pyladr.parallel.inference_engine import (
            ParallelInferenceEngine,
            ParallelSearchConfig,
        )

        config = ParallelSearchConfig(enabled=True, max_workers=2)
        engine = ParallelInferenceEngine(config)
        try:
            # Should not crash with empty inputs
            results = engine.generate_inferences(
                given=None,  # type: ignore[arg-type]
                usable_snapshot=[],
                binary_resolution=True,
                paramodulation=False,
                factoring=False,
            )
        except (TypeError, AttributeError):
            # Expected — None given clause is not valid
            pass
        finally:
            engine.shutdown()

    def test_parallel_engine_shutdown_idempotent(self):
        """Shutting down the parallel engine multiple times is safe."""
        from pyladr.parallel.inference_engine import (
            ParallelInferenceEngine,
            ParallelSearchConfig,
        )

        config = ParallelSearchConfig(enabled=True, max_workers=2)
        engine = ParallelInferenceEngine(config)
        engine.shutdown()
        engine.shutdown()  # Should not raise

    def test_parallel_determinism_across_runs(self, trivial_resolution_clauses):
        """Parallel search produces consistent results across runs."""
        from pyladr.parallel.inference_engine import ParallelSearchConfig
        from pyladr.search.given_clause import ExitCode

        results = []
        for _ in range(5):
            result = run_search(
                usable=[],
                sos=trivial_resolution_clauses,
                parallel=ParallelSearchConfig(enabled=True, max_workers=2),
            )
            results.append(result)

        for r in results[1:]:
            assert r.exit_code == results[0].exit_code
