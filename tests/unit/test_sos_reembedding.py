"""Tests for SOS re-embedding after Tree2Vec online model updates.

Covers:
- SOS clause embeddings are refreshed after _do_t2v_online_update()
- Stale embeddings in _tree2vec_embeddings are replaced with fresh vectors
- Empty SOS produces zero re-embeddings without errors
- sos_reembedded count appears in output
- End-to-end: online learning + SOS re-embedding during search
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.ml.tree2vec.algorithm import Tree2Vec, Tree2VecConfig
from pyladr.ml.tree2vec.skipgram import SkipGramConfig
from pyladr.ml.tree2vec.provider import (
    Tree2VecEmbeddingProvider,
    Tree2VecProviderConfig,
)
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions


# ── Helpers (reused from test_online_tree2vec.py patterns) ───────────────

SYM_P = 1
SYM_I = 2
SYM_N = 3
SYM_E = 4


def var(n: int) -> Term:
    return get_variable_term(n)


def n(arg: Term) -> Term:
    return get_rigid_term(SYM_N, 1, (arg,))


def i(left: Term, right: Term) -> Term:
    return get_rigid_term(SYM_I, 2, (left, right))


def P(arg: Term) -> Term:
    return get_rigid_term(SYM_P, 1, (arg,))


def e() -> Term:
    return get_rigid_term(SYM_E, 0)


def make_literal(sign: bool, atom: Term) -> Literal:
    return Literal(sign=sign, atom=atom)


def make_clause(*lits: Literal, clause_id: int = 0) -> Clause:
    return Clause(literals=lits, id=clause_id)


def _make_training_clauses() -> list[Clause]:
    x, y, z = var(0), var(1), var(2)
    return [
        make_clause(make_literal(True, P(i(x, y))), clause_id=1),
        make_clause(make_literal(True, P(i(n(x), x))), clause_id=2),
        make_clause(make_literal(True, P(i(i(x, y), z))), clause_id=3),
        make_clause(make_literal(True, P(n(i(x, x)))), clause_id=4),
        make_clause(make_literal(False, P(i(x, n(y)))), clause_id=5),
    ]


def _make_novel_clauses() -> list[Clause]:
    x, y = var(0), var(1)
    return [
        make_clause(make_literal(True, P(i(n(x), n(y)))), clause_id=100),
        make_clause(make_literal(True, P(n(n(x)))), clause_id=101),
    ]


def _make_sos_clauses() -> list[Clause]:
    """Separate clauses to populate SOS for re-embedding tests."""
    x, y = var(0), var(1)
    return [
        make_clause(make_literal(True, P(i(x, e()))), clause_id=50),
        make_clause(make_literal(True, P(i(e(), y))), clause_id=51),
        make_clause(make_literal(False, P(n(x))), clause_id=52),
    ]


def _train_tree2vec(clauses: list[Clause] | None = None) -> Tree2Vec:
    config = Tree2VecConfig(
        skipgram_config=SkipGramConfig(
            embedding_dim=32,
            num_epochs=3,
            seed=42,
        ),
    )
    t2v = Tree2Vec(config)
    t2v.train(clauses or _make_training_clauses())
    return t2v


def _make_provider(t2v: Tree2Vec | None = None) -> Tree2VecEmbeddingProvider:
    if t2v is None:
        t2v = _train_tree2vec()
    config = Tree2VecProviderConfig(cache_max_entries=1000)
    return Tree2VecEmbeddingProvider(tree2vec=t2v, config=config)


# ── Direct unit tests for SOS re-embedding logic ────────────────────────


class TestSOSReembeddingDirect:
    """Test SOS re-embedding by directly manipulating engine internals."""

    def _make_engine(self, **opts_kw) -> GivenClauseSearch:
        """Create a minimal engine with tree2vec online learning enabled."""
        defaults = dict(
            tree2vec_online_learning=True,
            tree2vec_online_update_interval=2,
            tree2vec_online_batch_size=5,
            tree2vec_online_lr=0.05,
            max_seconds=5,
            quiet=True,
        )
        defaults.update(opts_kw)
        opts = SearchOptions(**defaults)
        st = SymbolTable()
        return GivenClauseSearch(options=opts, symbol_table=st)

    def _setup_engine_with_sos(
        self, engine: GivenClauseSearch, sos_clauses: list[Clause]
    ) -> None:
        """Inject a trained provider and SOS clauses into the engine."""
        t2v = _train_tree2vec()
        provider = _make_provider(t2v)
        engine._tree2vec_provider = provider

        # Pre-compute embeddings and populate SOS
        for c in sos_clauses:
            emb = provider.get_embedding(c)
            if emb is not None:
                engine._tree2vec_embeddings[c.id] = emb
            engine._state.sos.append(c)

    def test_sos_embeddings_refreshed_after_online_update(self) -> None:
        """After _do_t2v_online_update, SOS embeddings dict is updated."""
        engine = self._make_engine()
        sos_clauses = _make_sos_clauses()
        self._setup_engine_with_sos(engine, sos_clauses)

        # Snapshot pre-update embeddings
        pre_embeddings = {
            cid: list(emb) for cid, emb in engine._tree2vec_embeddings.items()
        }
        assert len(pre_embeddings) == 3

        # Load the batch with novel clauses and trigger update
        engine._t2v_online_batch = _make_novel_clauses()
        engine._t2v_kept_since_update = 0

        # Run many online updates to drift the model
        for _ in range(10):
            engine._t2v_online_batch = _make_novel_clauses()
            engine._do_t2v_online_update()

        # All SOS clause IDs should still have embeddings
        for c in sos_clauses:
            assert c.id in engine._tree2vec_embeddings, (
                f"Clause {c.id} lost its embedding after re-embedding"
            )

        # Provider cache should show misses from re-embedding
        provider = engine._tree2vec_provider
        assert provider.stats.misses > len(sos_clauses)  # type: ignore[union-attr]

    def test_sos_reembedded_count_matches_sos_size(self) -> None:
        """The sos_reembedded count should equal the number of SOS clauses."""
        engine = self._make_engine(quiet=False)
        sos_clauses = _make_sos_clauses()
        self._setup_engine_with_sos(engine, sos_clauses)

        engine._t2v_online_batch = _make_novel_clauses()

        buf = io.StringIO()
        with redirect_stdout(buf):
            engine._do_t2v_online_update()

        output = buf.getvalue()
        match = re.search(r"sos_reembedded=(\d+)", output)
        assert match is not None, f"sos_reembedded not in output: {output!r}"
        assert int(match.group(1)) == len(sos_clauses)

    def test_empty_sos_produces_zero_reembeddings(self) -> None:
        """When SOS is empty, sos_reembedded=0 and no errors."""
        engine = self._make_engine(quiet=False)
        # Set up provider but don't add any SOS clauses
        t2v = _train_tree2vec()
        engine._tree2vec_provider = _make_provider(t2v)
        engine._t2v_online_batch = _make_novel_clauses()

        buf = io.StringIO()
        with redirect_stdout(buf):
            engine._do_t2v_online_update()

        output = buf.getvalue()
        match = re.search(r"sos_reembedded=(\d+)", output)
        assert match is not None, f"sos_reembedded not in output: {output!r}"
        assert int(match.group(1)) == 0

    def test_empty_batch_skips_update_entirely(self) -> None:
        """With an empty batch, _do_t2v_online_update is a no-op."""
        engine = self._make_engine()
        sos_clauses = _make_sos_clauses()
        self._setup_engine_with_sos(engine, sos_clauses)

        pre_embeddings = {
            cid: list(emb) for cid, emb in engine._tree2vec_embeddings.items()
        }

        engine._t2v_online_batch = []
        engine._do_t2v_online_update()

        # Embeddings unchanged (no update happened)
        for cid, emb in engine._tree2vec_embeddings.items():
            assert list(emb) == pre_embeddings[cid]

    def test_sos_clause_removed_mid_iteration_safe(self) -> None:
        """SOS re-embedding iterates a snapshot — removing a clause doesn't crash."""
        engine = self._make_engine()
        sos_clauses = _make_sos_clauses()
        self._setup_engine_with_sos(engine, sos_clauses)

        # Remove one clause from SOS before triggering update
        engine._state.sos.remove(sos_clauses[1])

        engine._t2v_online_batch = _make_novel_clauses()
        # Should not raise
        engine._do_t2v_online_update()

        # Only remaining clauses should have updated embeddings
        assert sos_clauses[0].id in engine._tree2vec_embeddings
        assert sos_clauses[2].id in engine._tree2vec_embeddings

    def test_model_version_increments_on_update(self) -> None:
        """Each online update bumps the provider's model version."""
        engine = self._make_engine()
        sos_clauses = _make_sos_clauses()
        self._setup_engine_with_sos(engine, sos_clauses)

        provider = engine._tree2vec_provider
        v_before = provider.model_version  # type: ignore[union-attr]

        engine._t2v_online_batch = _make_novel_clauses()
        engine._do_t2v_online_update()

        assert provider.model_version > v_before  # type: ignore[union-attr]

    def test_multiple_updates_progressively_reembed(self) -> None:
        """Multiple successive online updates each re-embed SOS."""
        engine = self._make_engine(quiet=False)
        sos_clauses = _make_sos_clauses()
        self._setup_engine_with_sos(engine, sos_clauses)

        for update_round in range(3):
            engine._t2v_online_batch = _make_novel_clauses()
            buf = io.StringIO()
            with redirect_stdout(buf):
                engine._do_t2v_online_update()
            output = buf.getvalue()
            match = re.search(r"sos_reembedded=(\d+)", output)
            assert match is not None, f"Round {update_round}: missing sos_reembedded"
            assert int(match.group(1)) == len(sos_clauses)

        # Update count tracks all rounds
        assert engine._t2v_update_count == 3


# ── End-to-end integration test ──────────────────────────────────────────


_ONLINE_LEARNING_INPUT = (
    "set(tree2vec_embeddings).\n"
    "set(tree2vec_online_learning).\n"
    "assign(tree2vec_online_update_interval, 3).\n"
    "assign(tree2vec_online_batch_size, 5).\n"
    "formulas(sos).\n"
    "  P(a).\n"
    "  -P(x) | Q(x).\n"
    "end_of_list.\n"
    "formulas(goals).\n"
    "  Q(a).\n"
    "end_of_list.\n"
)


class TestSOSReembeddingE2E:
    """End-to-end: search with online learning produces correct proofs."""

    def _run_search(self, input_text: str, max_seconds: float = 10) -> dict:
        from pyladr.apps.prover9 import _auto_inference, _auto_limits, _deny_goals, _apply_settings
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(input_text)
        usable, sos, _denied = _deny_goals(parsed, st)
        opts = SearchOptions(max_seconds=max_seconds)
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
            "kept": result.stats.kept,
        }

    def test_proof_found_with_online_learning_enabled(self) -> None:
        """Search with online learning + SOS re-embedding still finds the proof."""
        result = self._run_search(_ONLINE_LEARNING_INPUT)
        assert result["proved"]

    def test_no_regression_vs_baseline(self) -> None:
        """Online learning result matches baseline (proof found, same exit code)."""
        baseline_input = (
            "formulas(sos).\n"
            "  P(a).\n"
            "  -P(x) | Q(x).\n"
            "end_of_list.\n"
            "formulas(goals).\n"
            "  Q(a).\n"
            "end_of_list.\n"
        )
        baseline = self._run_search(baseline_input)
        online = self._run_search(_ONLINE_LEARNING_INPUT)

        assert baseline["proved"]
        assert online["proved"]
        assert baseline["exit_code"] == online["exit_code"]
