"""Parallel inference generation engine for the given-clause loop.

This module parallelizes the most expensive part of the theorem prover:
generating inferences between the given clause and all usable clauses.

== Parallelization Strategy ==

The given-clause loop's inference generation is embarrassingly parallel:
for each usable clause U, computing all_binary_resolvents(given, U) and
para_from_into(given, U) are independent operations.

We split the usable set into chunks and process each chunk in a separate
thread. Results are collected and then processed sequentially (clause
processing requires sequential index updates).

== Architecture ==

    Sequential: select_given → move_to_usable → index
    Parallel:   generate_inferences(given, usable_chunks)  ← THIS MODULE
    Sequential: cl_process each inference → limbo_process

== Thread Safety ==

During inference generation:
- Usable list is READ-ONLY (snapshot taken before parallel phase)
- Each worker produces an independent list of new clauses
- No shared mutable state during generation
- Workers use their own Context/Trail objects (thread-local)

After inference generation:
- Results merged into single list
- Sequential cl_process handles index updates with proper locking

== Performance Notes ==

- Minimum 50 usable clauses before parallelizing (overhead not worth it below)
- Thread pool is reused across iterations (amortize creation cost)
- Falls back to sequential on non-free-threaded Python (GIL makes threads useless)
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyladr.core.clause import Clause
from pyladr.inference.resolution import all_binary_resolvents, factor
from pyladr.inference.subsumption import back_subsume_from_lists, subsumes
from pyladr.threading_guide import FREE_THREADING_AVAILABLE

if TYPE_CHECKING:
    from pyladr.core.symbol import SymbolTable


@dataclass(frozen=True, slots=True)
class ParallelSearchConfig:
    """Configuration for parallel search execution.

    Attributes:
        enabled: Enable parallel inference generation.
        max_workers: Maximum number of worker threads (None = cpu_count).
        min_usable_for_parallel: Minimum usable clauses before parallelizing.
        chunk_size: Number of usable clauses per work unit.
        parallel_back_subsumption: Enable parallel backward subsumption.
        min_clauses_for_parallel_back: Min clauses for parallel back subsumption.
    """

    enabled: bool = True
    max_workers: int | None = None
    min_usable_for_parallel: int = 50
    chunk_size: int = 25
    parallel_back_subsumption: bool = True
    min_clauses_for_parallel_back: int = 100

    @property
    def effective_workers(self) -> int:
        if self.max_workers is not None:
            return self.max_workers
        return min(os.cpu_count() or 4, 8)


class ParallelInferenceEngine:
    """Parallel inference generation engine.

    Generates inferences between a given clause and all usable clauses
    using thread-based parallelism on free-threaded Python.

    Usage:
        engine = ParallelInferenceEngine(config)
        inferences = engine.generate_inferences(
            given, usable_snapshot,
            binary_resolution=True,
            paramodulation=False,
        )
        # Process inferences sequentially
        for clause in inferences:
            cl_process(clause)
    """

    __slots__ = ("_config", "_pool")

    def __init__(self, config: ParallelSearchConfig | None = None) -> None:
        self._config = config or ParallelSearchConfig()
        self._pool: ThreadPoolExecutor | None = None

    def _get_pool(self) -> ThreadPoolExecutor:
        """Get or create the thread pool (lazy initialization)."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self._config.effective_workers,
                thread_name_prefix="pyladr-infer",
            )
        return self._pool

    def shutdown(self) -> None:
        """Shutdown the thread pool."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def should_parallelize(self, usable_count: int) -> bool:
        """Check if parallelization is worthwhile for this iteration."""
        return (
            self._config.enabled
            and FREE_THREADING_AVAILABLE
            and usable_count >= self._config.min_usable_for_parallel
        )

    def generate_inferences(
        self,
        given: Clause,
        usable_snapshot: list[Clause],
        *,
        binary_resolution: bool = True,
        paramodulation: bool = False,
        factoring: bool = True,
        para_into_vars: bool = False,
        symbol_table: SymbolTable | None = None,
    ) -> list[Clause]:
        """Generate all inferences from given clause against usable set.

        If parallelization conditions are met, splits the usable set
        into chunks and processes in parallel. Otherwise falls back
        to sequential generation.

        Args:
            given: The selected given clause.
            usable_snapshot: Snapshot of usable clauses (immutable during generation).
            binary_resolution: Enable resolution inference.
            paramodulation: Enable paramodulation inference.
            factoring: Enable factoring of the given clause.
            para_into_vars: Allow paramodulation into variables.
            symbol_table: Symbol table (needed for paramodulation).

        Returns:
            List of all generated inferences (unprocessed).
        """
        if self.should_parallelize(len(usable_snapshot)):
            return self._parallel_generate(
                given, usable_snapshot,
                binary_resolution=binary_resolution,
                paramodulation=paramodulation,
                factoring=factoring,
                para_into_vars=para_into_vars,
                symbol_table=symbol_table,
            )
        return self._sequential_generate(
            given, usable_snapshot,
            binary_resolution=binary_resolution,
            paramodulation=paramodulation,
            factoring=factoring,
            para_into_vars=para_into_vars,
            symbol_table=symbol_table,
        )

    def _sequential_generate(
        self,
        given: Clause,
        usable_snapshot: list[Clause],
        *,
        binary_resolution: bool,
        paramodulation: bool,
        factoring: bool,
        para_into_vars: bool,
        symbol_table: SymbolTable | None,
    ) -> list[Clause]:
        """Sequential inference generation (baseline)."""
        results: list[Clause] = []

        for usable_clause in usable_snapshot:
            if binary_resolution:
                results.extend(all_binary_resolvents(given, usable_clause))
                if usable_clause is not given:
                    results.extend(all_binary_resolvents(usable_clause, given))

            if paramodulation and symbol_table is not None:
                from pyladr.inference.paramodulation import para_from_into
                paras = para_from_into(
                    given, usable_clause, False, symbol_table,
                    para_into_vars,
                )
                results.extend(paras)
                if usable_clause is not given:
                    paras2 = para_from_into(
                        usable_clause, given, True, symbol_table,
                        para_into_vars,
                    )
                    results.extend(paras2)

        if factoring:
            results.extend(factor(given))

        return results

    def _parallel_generate(
        self,
        given: Clause,
        usable_snapshot: list[Clause],
        *,
        binary_resolution: bool,
        paramodulation: bool,
        factoring: bool,
        para_into_vars: bool,
        symbol_table: SymbolTable | None,
    ) -> list[Clause]:
        """Parallel inference generation using thread pool."""
        pool = self._get_pool()
        chunk_size = self._config.chunk_size
        chunks = [
            usable_snapshot[i:i + chunk_size]
            for i in range(0, len(usable_snapshot), chunk_size)
        ]

        futures = []
        for chunk in chunks:
            future = pool.submit(
                self._infer_chunk,
                given, chunk,
                binary_resolution=binary_resolution,
                paramodulation=paramodulation,
                para_into_vars=para_into_vars,
                symbol_table=symbol_table,
            )
            futures.append(future)

        # Collect results from all chunks, preserving order
        results: list[Clause] = []
        for future in futures:
            chunk_results = future.result()
            results.extend(chunk_results)

        # Factoring is done once on the given clause (not parallelized)
        if factoring:
            results.extend(factor(given))

        return results

    @staticmethod
    def _infer_chunk(
        given: Clause,
        usable_chunk: list[Clause],
        *,
        binary_resolution: bool,
        paramodulation: bool,
        para_into_vars: bool,
        symbol_table: SymbolTable | None,
    ) -> list[Clause]:
        """Process a chunk of usable clauses — runs in worker thread.

        Each call is independent: uses only read-only shared data (given, usable)
        and produces an independent list of new clauses.
        """
        results: list[Clause] = []

        for usable_clause in usable_chunk:
            if binary_resolution:
                results.extend(all_binary_resolvents(given, usable_clause))
                if usable_clause is not given:
                    results.extend(all_binary_resolvents(usable_clause, given))

            if paramodulation and symbol_table is not None:
                from pyladr.inference.paramodulation import para_from_into
                paras = para_from_into(
                    given, usable_clause, False, symbol_table,
                    para_into_vars,
                )
                results.extend(paras)
                if usable_clause is not given:
                    paras2 = para_from_into(
                        usable_clause, given, True, symbol_table,
                        para_into_vars,
                    )
                    results.extend(paras2)

        return results

    # ── Parallel backward subsumption ─────────────────────────────────────

    def parallel_back_subsume(
        self,
        new_clause: Clause,
        clause_lists: list[list[Clause]],
    ) -> list[Clause]:
        """Parallel backward subsumption check.

        Splits candidate clauses into chunks and checks each chunk
        in parallel for subsumption by the new clause.
        """
        # Flatten all candidates
        all_candidates = [c for clist in clause_lists for c in clist if c is not new_clause]

        if (
            not self._config.parallel_back_subsumption
            or not FREE_THREADING_AVAILABLE
            or len(all_candidates) < self._config.min_clauses_for_parallel_back
        ):
            return back_subsume_from_lists(new_clause, clause_lists)

        pool = self._get_pool()
        chunk_size = max(self._config.chunk_size, 50)
        chunks = [
            all_candidates[i:i + chunk_size]
            for i in range(0, len(all_candidates), chunk_size)
        ]

        futures = []
        nc = new_clause.num_literals
        for chunk in chunks:
            future = pool.submit(self._back_subsume_chunk, new_clause, nc, chunk)
            futures.append(future)

        victims: list[Clause] = []
        for future in futures:
            victims.extend(future.result())

        return victims

    @staticmethod
    def _back_subsume_chunk(
        new_clause: Clause,
        nc: int,
        candidates: list[Clause],
    ) -> list[Clause]:
        """Check a chunk of candidates for backward subsumption."""
        victims: list[Clause] = []
        for d in candidates:
            nd = d.num_literals
            if nc <= nd and subsumes(new_clause, d):
                victims.append(d)
        return victims
