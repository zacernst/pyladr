"""Coverage for the torch-unavailable / ML-disabled fallback path.

Unlike ``tests/unit/test_embedding_provider.py`` (which gates everything on
``pytest.importorskip("torch")``), this file deliberately avoids requiring
torch at module scope so that these tests still execute in a torch-free
environment. The goal is to ensure the fallback path stays exercised:

  * ``NoOpEmbeddingProvider`` contracts (dim, single, batch, empty).
  * ``create_embedding_provider`` falling back on both gates:
      - ``_ML_AVAILABLE == False`` (torch not importable at module load)
      - GNN construction raises at runtime (model file corrupt, OOM, etc.)
  * ``pyladr.ml.rnn2vec.algorithm._require_torch`` raising a helpful
    ``ImportError`` when torch is missing.
  * ``EmbeddingEnhancedSelection`` surviving a provider that returns only
    ``None`` embeddings (records misses, falls back to traditional scoring,
    search continues).

Baseline coverage in ``tests/compatibility/test_ml_opt_in.py`` had 3 tests
for these paths; this file adds 8 more.
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import get_rigid_term
from pyladr.ml.embedding_provider import (
    NoOpEmbeddingProvider,
    create_embedding_provider,
)
from pyladr.protocols import EmbeddingProvider
from pyladr.search.ml_selection import (
    EmbeddingEnhancedSelection,
    MLSelectionConfig,
)
from pyladr.search.selection import SelectionOrder, SelectionRule
from pyladr.search.state import ClauseList


# ── Clause helpers (no torch required) ─────────────────────────────────────


def _const(symnum: int):
    return get_rigid_term(symnum, 0)


def _make_clause(symnum: int, cid: int, weight: float) -> Clause:
    atom = _const(symnum)
    c = Clause(literals=(Literal(sign=True, atom=atom),))
    c.id = cid
    c.weight = weight
    return c


def _make_sos(clauses: list[Clause]) -> ClauseList:
    cl = ClauseList("sos")
    for c in clauses:
        cl.append(c)
    return cl


# Weight-heavy rule mix so the ML path actually runs during the test.
_WEIGHT_RULES = [
    SelectionRule("W", SelectionOrder.WEIGHT, part=5),
    SelectionRule("A", SelectionOrder.AGE, part=1),
]


# ── NoOpEmbeddingProvider contract ─────────────────────────────────────────


class TestNoOpProviderContract:
    """Exercise the NoOp provider directly — no torch required at any point."""

    def test_default_embedding_dim_is_512(self):
        assert NoOpEmbeddingProvider().embedding_dim == 512

    def test_embedding_dim_custom_value_preserved(self):
        assert NoOpEmbeddingProvider(embedding_dim=128).embedding_dim == 128

    def test_get_embedding_always_returns_none(self):
        noop = NoOpEmbeddingProvider()
        c = _make_clause(symnum=1, cid=1, weight=1.0)
        assert noop.get_embedding(c) is None

    def test_get_embeddings_batch_returns_none_per_clause(self):
        noop = NoOpEmbeddingProvider()
        clauses = [_make_clause(i + 1, i, float(i)) for i in range(3)]
        assert noop.get_embeddings_batch(clauses) == [None, None, None]

    def test_get_embeddings_batch_empty_input_returns_empty_list(self):
        assert NoOpEmbeddingProvider().get_embeddings_batch([]) == []

    def test_satisfies_runtime_checkable_protocol(self):
        assert isinstance(NoOpEmbeddingProvider(), EmbeddingProvider)


# ── Factory fallback gates ──────────────────────────────────────────────────


class TestFactoryFallback:
    """``create_embedding_provider`` has two NoOp fallback gates."""

    def test_falls_back_when_ml_unavailable(self, monkeypatch):
        """Gate 1: ``_ML_AVAILABLE == False`` short-circuits to NoOp."""
        import pyladr.ml.embedding_provider as ep_mod

        monkeypatch.setattr(ep_mod, "_ML_AVAILABLE", False)
        provider = create_embedding_provider()
        assert isinstance(provider, NoOpEmbeddingProvider)

    def test_falls_back_when_gnn_construction_raises(self, monkeypatch):
        """Gate 2: GNNEmbeddingProvider.create() exception → NoOp fallback.

        Covers the try/except at the end of ``create_embedding_provider``.
        Simulates a runtime failure (e.g. corrupt checkpoint, import error
        inside the GNN module) even though ``_ML_AVAILABLE`` is True.
        """
        import pyladr.ml.embedding_provider as ep_mod

        monkeypatch.setattr(ep_mod, "_ML_AVAILABLE", True)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated GNN construction failure")

        monkeypatch.setattr(ep_mod.GNNEmbeddingProvider, "create", _boom)
        provider = create_embedding_provider()
        assert isinstance(provider, NoOpEmbeddingProvider)


# ── RNN2Vec torch guard ─────────────────────────────────────────────────────


class TestRNN2VecTorchGuard:
    """``_require_torch`` is the sole torch-missing signal for RNN2Vec training."""

    def test_require_torch_raises_with_install_hint(self, monkeypatch):
        import pyladr.ml.rnn2vec.algorithm as alg_mod

        monkeypatch.setattr(alg_mod, "_TORCH_AVAILABLE", False)
        with pytest.raises(ImportError, match="pip install torch"):
            alg_mod._require_torch()

    def test_require_torch_is_noop_when_torch_available(self, monkeypatch):
        import pyladr.ml.rnn2vec.algorithm as alg_mod

        monkeypatch.setattr(alg_mod, "_TORCH_AVAILABLE", True)
        alg_mod._require_torch()  # must not raise


# ── Selection integration with an all-None provider ────────────────────────


class TestSelectionToleratesNoOpProvider:
    """Selection must keep searching when the provider returns no embeddings."""

    def _run_selection(self, sos: ClauseList, provider: EmbeddingProvider):
        config = MLSelectionConfig(
            enabled=True,
            ml_weight=0.5,
            min_sos_for_ml=1,       # let ML attempt to fire immediately
            log_selections=False,
            complexity_normalization=False,
        )
        selection = EmbeddingEnhancedSelection(
            rules=list(_WEIGHT_RULES),
            embedding_provider=provider,
            ml_config=config,
        )
        # Cycle count=0 → first rule ("W", WEIGHT) is chosen, triggering ML path.
        return selection.select_given(sos, given_count=0), selection

    def test_all_none_embeddings_still_select_a_clause(self):
        clauses = [_make_clause(i + 1, i + 1, float(i + 1)) for i in range(3)]
        sos = _make_sos(clauses)
        (selected, sel_type), _ = self._run_selection(sos, NoOpEmbeddingProvider())
        # With all-None embeddings the blended score collapses to the
        # traditional weight score; the lightest clause must be chosen.
        assert selected is not None
        assert selected.weight == 1.0
        assert "W" in sel_type  # selection type still reports the rule

    def test_all_none_embeddings_record_embedding_miss_per_clause(self):
        clauses = [_make_clause(i + 1, i + 1, float(i + 1)) for i in range(3)]
        sos = _make_sos(clauses)
        (selected, _), selection = self._run_selection(sos, NoOpEmbeddingProvider())
        # One miss is recorded per scored clause inside _ml_select_inner.
        assert selected is not None
        assert selection.ml_stats.embedding_miss_count == 3
