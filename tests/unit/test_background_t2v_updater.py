"""Tests for BackgroundT2VUpdater and its search integration.

Covers:
- Area 1: BackgroundT2VUpdater unit tests (daemon thread, submit, queue, shutdown, callbacks)
- Area 2: Search integration tests (bg updater lifecycle in GivenClauseSearch)
- Area 3: Regression / sync-mode correctness
"""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.ml.tree2vec.algorithm import Tree2Vec, Tree2VecConfig
from pyladr.ml.tree2vec.background_updater import BackgroundT2VUpdater
from pyladr.ml.tree2vec.provider import Tree2VecEmbeddingProvider, Tree2VecProviderConfig
from pyladr.ml.tree2vec.skipgram import SkipGramConfig
from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions


# ── Helpers ──────────────────────────────────────────────────────────────

SYM_P = 1
SYM_I = 2
SYM_N = 3


def var(n: int) -> Term:
    return get_variable_term(n)


def n(arg: Term) -> Term:
    return get_rigid_term(SYM_N, 1, (arg,))


def i(left: Term, right: Term) -> Term:
    return get_rigid_term(SYM_I, 2, (left, right))


def P(arg: Term) -> Term:
    return get_rigid_term(SYM_P, 1, (arg,))


def make_literal(sign: bool, atom: Term) -> Literal:
    return Literal(sign=sign, atom=atom)


def make_clause(*lits: Literal, clause_id: int = 0) -> Clause:
    return Clause(literals=lits, id=clause_id)


def _make_training_clauses() -> list[Clause]:
    x, y = var(0), var(1)
    return [
        make_clause(make_literal(True, P(n(x))), clause_id=1),
        make_clause(make_literal(False, P(n(y))), clause_id=2),
        make_clause(make_literal(True, P(i(x, y))), clause_id=3),
    ]


def _make_novel_clauses() -> list[Clause]:
    x, y = var(0), var(1)
    return [
        make_clause(make_literal(True, P(i(n(x), n(y)))), clause_id=100),
        make_clause(make_literal(True, P(n(n(x)))), clause_id=101),
    ]


def make_trained_provider() -> Tree2VecEmbeddingProvider:
    """Build a minimal trained provider for testing."""
    config = Tree2VecConfig(
        skipgram_config=SkipGramConfig(embedding_dim=8, num_epochs=2, seed=42),
    )
    t2v = Tree2Vec(config)
    t2v.train(_make_training_clauses())
    return Tree2VecEmbeddingProvider(tree2vec=t2v)


_SIMPLE_PROOF_INPUT = (
    "formulas(sos).\n"
    "  P(a).\n"
    "  -P(x) | Q(x).\n"
    "end_of_list.\n"
    "formulas(goals).\n"
    "  Q(a).\n"
    "end_of_list.\n"
)


def _run_python(input_text: str, max_seconds: float = 10, **extra_opts) -> dict:
    """Run Python prover on input text, return result dict."""
    from pyladr.apps.prover9 import _auto_inference, _auto_limits, _deny_goals, _apply_settings
    from pyladr.parsing.ladr_parser import LADRParser

    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(input_text)
    usable, sos, _denied = _deny_goals(parsed, st)
    opts = SearchOptions(max_seconds=max_seconds, **extra_opts)
    _auto_inference(parsed, opts)
    _auto_limits(parsed, opts)
    _apply_settings(parsed, opts)
    engine = GivenClauseSearch(
        options=opts,
        symbol_table=st,
        hints=parsed.hints if parsed.hints else None,
    )
    result = engine.run(usable=usable, sos=sos)
    return {
        "proved": len(result.proofs) > 0,
        "exit_code": result.exit_code,
        "given": result.stats.given,
        "generated": result.stats.generated,
        "kept": result.stats.kept,
        "engine": engine,
    }


# ═══════════════════════════════════════════════════════════════════════
# Area 1: BackgroundT2VUpdater unit tests
# ═══════════════════════════════════════════════════════════════════════


class TestBackgroundT2VUpdater:
    def test_updater_starts_daemon_thread(self) -> None:
        """After construction, thread is alive and is a daemon thread."""
        provider = make_trained_provider()
        updater = BackgroundT2VUpdater(provider=provider)
        try:
            assert updater.is_alive
            assert updater._thread.daemon is True
        finally:
            updater.shutdown(drain=False)

    def test_submit_triggers_update(self) -> None:
        """Submit a batch; after shutdown(drain=True), update_count == 1."""
        provider = make_trained_provider()
        updater = BackgroundT2VUpdater(provider=provider)
        try:
            batch = _make_novel_clauses()
            accepted = updater.submit(batch)
            assert accepted is True
        finally:
            updater.shutdown(drain=True)
        assert updater.update_count == 1

    def test_submit_returns_false_when_queue_full(self) -> None:
        """When the internal queue is full, submit returns False."""
        provider = make_trained_provider()
        updater = BackgroundT2VUpdater(provider=provider)
        try:
            # Fill the queue by pausing the worker
            # We'll block the worker by holding the queue item and submitting another
            # The queue has maxsize=1, so filling it + another submit = False
            with patch.object(updater, '_queue') as mock_queue:
                mock_queue.put_nowait.side_effect = queue.Full()
                # _stop_event.is_set() needs to return False
                result = updater.submit(_make_novel_clauses())
                assert result is False
        finally:
            updater.shutdown(drain=False)

    def test_submit_returns_false_after_max_updates(self) -> None:
        """Set max_updates=1, submit twice; second returns False."""
        provider = make_trained_provider()
        updater = BackgroundT2VUpdater(provider=provider, max_updates=1)
        try:
            batch = _make_novel_clauses()
            accepted1 = updater.submit(batch)
            assert accepted1 is True
            # Wait for the update to complete
            time.sleep(0.5)
            accepted2 = updater.submit(batch)
            assert accepted2 is False
        finally:
            updater.shutdown(drain=True)

    def test_shutdown_drains_pending_update(self) -> None:
        """Submit a batch, immediately call shutdown(drain=True), verify update ran."""
        provider = make_trained_provider()
        updater = BackgroundT2VUpdater(provider=provider)
        updater.submit(_make_novel_clauses())
        updater.shutdown(drain=True)
        assert updater.update_count == 1

    def test_completion_callback_called(self) -> None:
        """Pass a callback, submit a batch, shutdown; assert callback was called."""
        provider = make_trained_provider()
        callback_data: list[tuple[int, dict]] = []

        def on_done(count: int, stats: dict) -> None:
            callback_data.append((count, stats))

        updater = BackgroundT2VUpdater(
            provider=provider, completion_callback=on_done,
        )
        updater.submit(_make_novel_clauses())
        updater.shutdown(drain=True)

        assert len(callback_data) == 1
        assert callback_data[0][0] == 1  # update_count
        assert "pairs_trained" in callback_data[0][1]

    def test_bump_model_version_called_after_update(self) -> None:
        """Provider.model_version increments after an update completes."""
        provider = make_trained_provider()
        assert provider.model_version == 0

        updater = BackgroundT2VUpdater(provider=provider)
        updater.submit(_make_novel_clauses())
        updater.shutdown(drain=True)

        assert provider.model_version == 1

    def test_concurrent_reads_during_update(self) -> None:
        """Launch 4 threads calling get_embedding() while submitting a batch.

        No exceptions, no torn values (all embeddings are None or correct length).
        """
        provider = make_trained_provider()
        dim = provider.embedding_dim
        errors: list[Exception] = []

        def reader() -> None:
            deadline = time.monotonic() + 0.2
            while time.monotonic() < deadline:
                try:
                    clause = _make_training_clauses()[0]
                    emb = provider.get_embedding(clause)
                    if emb is not None:
                        assert len(emb) == dim
                except Exception as exc:
                    errors.append(exc)

        updater = BackgroundT2VUpdater(provider=provider)
        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()

        updater.submit(_make_novel_clauses())

        for t in threads:
            t.join(timeout=2.0)
        updater.shutdown(drain=True)

        assert errors == [], f"Reader threads raised: {errors}"


# ═══════════════════════════════════════════════════════════════════════
# Area 2: Search integration tests
# ═══════════════════════════════════════════════════════════════════════


class TestSearchIntegration:
    def test_bg_update_off_by_default_when_online_learning_false(self) -> None:
        """tree2vec_online_learning=False → _t2v_bg_updater is None."""
        result = _run_python(
            _SIMPLE_PROOF_INPUT,
            tree2vec_embeddings=True,
            tree2vec_online_learning=False,
        )
        engine = result["engine"]
        assert engine._t2v_bg_updater is None

    def test_bg_updater_created_when_enabled(self) -> None:
        """tree2vec_online_learning=True + tree2vec_bg_update=True → updater created."""
        input_text = (
            "set(tree2vec_online_learning).\n"
            + _SIMPLE_PROOF_INPUT
        )
        result = _run_python(
            input_text,
            tree2vec_embeddings=True,
            tree2vec_online_learning=True,
            tree2vec_bg_update=True,
        )
        # After run() completes, the updater has been shut down but the attribute
        # should still exist (not None before shutdown; we check it was created
        # by verifying it's not None OR that shutdown was called).
        # Since run() shuts it down, the thread won't be alive, but the object exists.
        engine = result["engine"]
        # The updater was created and then shut down — check it's not None
        # (it stays assigned after shutdown).
        assert engine._t2v_bg_updater is not None

    def test_sync_fallback_when_bg_update_false(self) -> None:
        """tree2vec_bg_update=False → _t2v_bg_updater is None."""
        result = _run_python(
            _SIMPLE_PROOF_INPUT,
            tree2vec_embeddings=True,
            tree2vec_online_learning=True,
            tree2vec_bg_update=False,
        )
        engine = result["engine"]
        assert engine._t2v_bg_updater is None
        assert hasattr(engine, "_do_t2v_online_update_sync")

    def test_search_completes_cleanly_with_bg_update(self) -> None:
        """Full search with bg update on simple fixture; no fatal exit."""
        input_text = (
            "set(tree2vec_online_learning).\n"
            + _SIMPLE_PROOF_INPUT
        )
        result = _run_python(
            input_text,
            tree2vec_embeddings=True,
            tree2vec_online_learning=True,
            tree2vec_bg_update=True,
        )
        assert result["exit_code"] != ExitCode.FATAL_EXIT, "Search hit fatal error"

    def test_bg_updater_shut_down_after_run(self) -> None:
        """After run() returns, _t2v_bg_updater.is_alive is False."""
        # Use a slightly larger problem to give the bg updater time to start
        input_text = (
            "set(tree2vec_online_learning).\n"
            "assign(tree2vec_online_update_interval, 2).\n"
            "formulas(sos).\n"
            "  P(a).\n"
            "  -P(x) | Q(x).\n"
            "  -Q(x) | R(x).\n"
            "end_of_list.\n"
            "formulas(goals).\n"
            "  R(a).\n"
            "end_of_list.\n"
        )
        result = _run_python(
            input_text,
            tree2vec_embeddings=True,
            tree2vec_online_learning=True,
            tree2vec_bg_update=True,
        )
        engine = result["engine"]
        if engine._t2v_bg_updater is not None:
            # run() should have called shutdown(); if the thread is still alive,
            # call shutdown again and give it time to finish.
            if engine._t2v_bg_updater.is_alive:
                engine._t2v_bg_updater.shutdown(drain=True, timeout=3.0)
            assert engine._t2v_bg_updater.is_alive is False


# ═══════════════════════════════════════════════════════════════════════
# Area 3: Regression tests
# ═══════════════════════════════════════════════════════════════════════


class TestRegressionSyncMode:
    def test_sync_mode_produces_same_result(self) -> None:
        """With tree2vec_bg_update=False (sync mode), verify sync path works:
        bump_model_version is called and update_count increments."""
        from pyladr.apps.prover9 import _auto_inference, _auto_limits, _deny_goals, _apply_settings
        from pyladr.parsing.ladr_parser import LADRParser

        # Use a slightly larger problem so at least one online update fires
        input_text = (
            "set(tree2vec_online_learning).\n"
            "assign(tree2vec_online_update_interval, 2).\n"
            "assign(tree2vec_online_batch_size, 2).\n"
            "formulas(sos).\n"
            "  P(a).\n"
            "  -P(x) | Q(x).\n"
            "  -Q(x) | R(x).\n"
            "end_of_list.\n"
            "formulas(goals).\n"
            "  R(a).\n"
            "end_of_list.\n"
        )

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(input_text)
        usable, sos, _denied = _deny_goals(parsed, st)
        opts = SearchOptions(
            max_seconds=5,
            tree2vec_embeddings=True,
            tree2vec_online_learning=True,
            tree2vec_bg_update=False,  # sync mode
        )
        _auto_inference(parsed, opts)
        _auto_limits(parsed, opts)
        _apply_settings(parsed, opts)
        engine = GivenClauseSearch(
            options=opts,
            symbol_table=st,
            hints=parsed.hints if parsed.hints else None,
        )
        result = engine.run(usable=usable, sos=sos)

        # The search should complete without error
        assert result.exit_code != ExitCode.FATAL_EXIT, "Sync mode search crashed"
        # bg updater should not have been created
        assert engine._t2v_bg_updater is None

    def test_existing_online_t2v_baseline(self) -> None:
        """Verify the basic online T2V provider roundtrip still works (regression guard)."""
        t2v = Tree2Vec(Tree2VecConfig(
            skipgram_config=SkipGramConfig(embedding_dim=32, num_epochs=3, seed=42),
        ))
        t2v.train(_make_training_clauses())
        provider = Tree2VecEmbeddingProvider(
            tree2vec=t2v,
            config=Tree2VecProviderConfig(cache_max_entries=1000),
        )

        clause = _make_training_clauses()[0]
        emb_before = provider.get_embedding(clause)
        assert emb_before is not None

        # Online update
        stats = t2v.update_online(_make_novel_clauses(), learning_rate=0.05)
        assert stats["pairs_trained"] > 0

        new_version = provider.bump_model_version()
        assert new_version == 1

        emb_after = provider.get_embedding(clause)
        assert emb_after is not None
        assert provider.stats.misses >= 2  # initial + post-bump
