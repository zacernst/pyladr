"""Tests for parallel inference generation engine."""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.parallel.inference_engine import (
    ParallelInferenceEngine,
    ParallelSearchConfig,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def symtab():
    return SymbolTable()


def _var(n: int) -> Term:
    return get_variable_term(n)


def _const(st: SymbolTable, name: str) -> Term:
    return get_rigid_term(st.str_to_sn(name, 0), 0)


def _func(st: SymbolTable, name: str, *args: Term) -> Term:
    return get_rigid_term(st.str_to_sn(name, len(args)), len(args), args)


def _pos_lit(atom: Term) -> Literal:
    return Literal(sign=True, atom=atom)


def _neg_lit(atom: Term) -> Literal:
    return Literal(sign=False, atom=atom)


def _clause(*lits: Literal, cid: int = 0) -> Clause:
    c = Clause(literals=tuple(lits))
    c.id = cid
    return c


# ── Config tests ─────────────────────────────────────────────────────────────


class TestParallelSearchConfig:
    def test_default_config(self):
        cfg = ParallelSearchConfig()
        assert cfg.enabled is True
        assert cfg.min_usable_for_parallel == 50
        assert cfg.effective_workers >= 1

    def test_custom_workers(self):
        cfg = ParallelSearchConfig(max_workers=2)
        assert cfg.effective_workers == 2


# ── Sequential generation tests ──────────────────────────────────────────────


class TestSequentialGeneration:
    """Test inference generation in sequential mode."""

    def test_empty_usable(self, symtab):
        """No usable clauses → no inferences."""
        a = _const(symtab, "a")
        p_a = _func(symtab, "P", a)
        given = _clause(_pos_lit(p_a))
        engine = ParallelInferenceEngine(ParallelSearchConfig(enabled=False))
        results = engine.generate_inferences(given, [])
        assert results == []

    def test_resolution_generates_resolvents(self, symtab):
        """Binary resolution between complementary unit clauses."""
        a = _const(symtab, "a")
        p_a = _func(symtab, "P", a)
        given = _clause(_pos_lit(p_a), cid=1)
        usable = _clause(_neg_lit(p_a), cid=2)
        engine = ParallelInferenceEngine(ParallelSearchConfig(enabled=False))
        results = engine.generate_inferences(
            given, [usable],
            binary_resolution=True,
            factoring=False,
        )
        # Should produce empty clause (resolvent of P(a) and ~P(a))
        assert any(c.is_empty for c in results)

    def test_factoring_produces_factors(self, symtab):
        """Factoring of a clause with duplicate-able literals."""
        x = _var(0)
        y = _var(1)
        p_x = _func(symtab, "P", x)
        p_y = _func(symtab, "P", y)
        given = _clause(_pos_lit(p_x), _pos_lit(p_y), cid=1)
        engine = ParallelInferenceEngine(ParallelSearchConfig(enabled=False))
        results = engine.generate_inferences(
            given, [],
            binary_resolution=False,
            factoring=True,
        )
        assert len(results) > 0

    def test_no_inference_different_predicates(self, symtab):
        """No resolution if predicates don't match."""
        a = _const(symtab, "a")
        p_a = _func(symtab, "P", a)
        q_a = _func(symtab, "Q", a)
        given = _clause(_pos_lit(p_a), cid=1)
        usable = _clause(_neg_lit(q_a), cid=2)
        engine = ParallelInferenceEngine(ParallelSearchConfig(enabled=False))
        results = engine.generate_inferences(
            given, [usable],
            binary_resolution=True,
            factoring=False,
        )
        assert all(not c.is_empty for c in results)


# ── Parallel mode tests ─────────────────────────────────────────────────────


class TestParallelGeneration:
    """Test parallel inference generation produces same results as sequential."""

    def test_parallel_matches_sequential(self, symtab):
        """Parallel and sequential should produce the same set of inferences."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        x = _var(0)
        p_x = _func(symtab, "P", x)
        p_a = _func(symtab, "P", a)
        p_b = _func(symtab, "P", b)
        q_a = _func(symtab, "Q", a)

        given = _clause(_neg_lit(p_x), cid=1)
        usable = [
            _clause(_pos_lit(p_a), cid=2),
            _clause(_pos_lit(p_b), cid=3),
            _clause(_pos_lit(q_a), cid=4),
        ]

        seq_engine = ParallelInferenceEngine(ParallelSearchConfig(enabled=False))
        seq_results = seq_engine.generate_inferences(
            given, usable,
            binary_resolution=True,
            factoring=False,
        )

        # Force parallel path by lowering threshold
        par_engine = ParallelInferenceEngine(
            ParallelSearchConfig(
                enabled=True,
                min_usable_for_parallel=1,
                chunk_size=2,
                max_workers=2,
            )
        )
        par_results = par_engine.generate_inferences(
            given, usable,
            binary_resolution=True,
            factoring=False,
        )
        par_engine.shutdown()

        # Same number of inferences
        assert len(par_results) == len(seq_results)

    def test_should_parallelize_threshold(self):
        """Parallelization decision based on usable count."""
        engine = ParallelInferenceEngine(
            ParallelSearchConfig(min_usable_for_parallel=50)
        )
        assert not engine.should_parallelize(10)
        assert not engine.should_parallelize(49)
        # On non-free-threaded Python, still False
        # On free-threaded Python, would be True for >= 50

    def test_shutdown_idempotent(self):
        """Shutdown is safe to call multiple times."""
        engine = ParallelInferenceEngine()
        engine.shutdown()
        engine.shutdown()  # Should not raise


# ── Parallel backward subsumption tests ──────────────────────────────────────


class TestParallelBackSubsumption:
    """Test parallel backward subsumption."""

    def test_finds_victims_sequentially(self, symtab):
        """Back subsumption finds subsumed clauses."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        x = _var(0)
        p_x = _func(symtab, "P", x)
        p_a = _func(symtab, "P", a)
        p_b = _func(symtab, "P", b)

        general = _clause(_pos_lit(p_x), cid=1)
        specific_a = _clause(_pos_lit(p_a), cid=2)
        specific_b = _clause(_pos_lit(p_b), cid=3)

        engine = ParallelInferenceEngine(
            ParallelSearchConfig(parallel_back_subsumption=False)
        )
        victims = engine.parallel_back_subsume(
            general, [[specific_a, specific_b]]
        )
        assert specific_a in victims
        assert specific_b in victims

    def test_parallel_back_subsume_matches_sequential(self, symtab):
        """Parallel and sequential backward subsumption give same results."""
        a = _const(symtab, "a")
        b = _const(symtab, "b")
        x = _var(0)
        p_x = _func(symtab, "P", x)
        p_a = _func(symtab, "P", a)
        p_b = _func(symtab, "P", b)

        general = _clause(_pos_lit(p_x), cid=1)
        specific_a = _clause(_pos_lit(p_a), cid=2)
        specific_b = _clause(_pos_lit(p_b), cid=3)

        # Sequential
        seq_engine = ParallelInferenceEngine(
            ParallelSearchConfig(parallel_back_subsumption=False)
        )
        seq_victims = seq_engine.parallel_back_subsume(
            general, [[specific_a, specific_b]]
        )

        # Force parallel path
        par_engine = ParallelInferenceEngine(
            ParallelSearchConfig(
                parallel_back_subsumption=True,
                min_clauses_for_parallel_back=1,
                chunk_size=1,
                max_workers=2,
            )
        )
        par_victims = par_engine.parallel_back_subsume(
            general, [[specific_a, specific_b]]
        )
        par_engine.shutdown()

        assert set(id(v) for v in seq_victims) == set(id(v) for v in par_victims)
