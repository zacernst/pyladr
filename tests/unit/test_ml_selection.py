"""Tests for embedding-enhanced clause selection.

Verifies:
- Backward compatibility: disabled ML = identical to GivenSelection
- ML scoring: diversity, proof-potential, blended scoring
- Fallback behavior: graceful degradation on embedding failures
- Configuration: all settings work as documented
- Statistics: correct tracking of ML vs traditional selections
"""

from __future__ import annotations

import math

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import get_rigid_term
from pyladr.search.ml_selection import (
    EmbeddingEnhancedSelection,
    EmbeddingProvider,
    MLSelectionConfig,
    MLSelectionStats,
    _cosine_similarity,
)
from pyladr.search.selection import GivenSelection, SelectionOrder, SelectionRule
from pyladr.search.state import ClauseList


# Default weight-only rules for tests (avoids C Prover9 property-filtered defaults)
_WEIGHT_AGE_RULES = [
    SelectionRule("W", SelectionOrder.WEIGHT, part=5),
    SelectionRule("A", SelectionOrder.AGE, part=1),
]

# ── Helpers ──────────────────────────────────────────────────────────────────


def _const(symnum: int) -> "Term":
    return get_rigid_term(symnum, 0)


def _func(symnum: int, *args) -> "Term":
    return get_rigid_term(symnum, len(args), args)


def _make_clause(*atoms, signs=None) -> Clause:
    if signs is None:
        signs = (True,) * len(atoms)
    lits = tuple(Literal(sign=s, atom=a) for s, a in zip(signs, atoms))
    return Clause(literals=lits)


def _make_weighted_clause(weight: float, cid: int = 0) -> Clause:
    c = _make_clause(_const(1))
    c.weight = weight
    c.id = cid
    return c


def _make_sos(clauses: list[Clause]) -> ClauseList:
    cl = ClauseList("sos")
    for c in clauses:
        cl.append(c)
    return cl


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""

    def __init__(
        self,
        embeddings: dict[int, list[float]] | None = None,
        dim: int = 4,
        fail_on: set[int] | None = None,
        raise_error: bool = False,
    ):
        self._embeddings = embeddings or {}
        self._dim = dim
        self._fail_on = fail_on or set()
        self._raise_error = raise_error
        self.call_count = 0

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def get_embedding(self, clause: Clause) -> list[float] | None:
        self.call_count += 1
        if self._raise_error:
            raise RuntimeError("Provider error")
        if clause.id in self._fail_on:
            return None
        return self._embeddings.get(clause.id)

    def get_embeddings_batch(
        self, clauses: list[Clause],
    ) -> list[list[float] | None]:
        return [self.get_embedding(c) for c in clauses]


# ── Cosine similarity tests ─────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0, 0], [1, 0]) == pytest.approx(0.0)

    def test_similar_vectors(self) -> None:
        sim = _cosine_similarity([1, 1], [1, 0])
        assert 0.5 < sim < 1.0


# ── Backward compatibility tests ─────────────────────────────────────────────


class TestBackwardCompatibility:
    """ML-enhanced selection with ML disabled must behave identically."""

    def test_disabled_ml_matches_traditional(self) -> None:
        """With ML disabled, selection is pure traditional."""
        config = MLSelectionConfig(enabled=False)
        sel = EmbeddingEnhancedSelection(
            rules=[SelectionRule("W", SelectionOrder.WEIGHT, part=5),
                   SelectionRule("A", SelectionOrder.AGE, part=1)],
            ml_config=config,
        )

        clauses = [_make_weighted_clause(w, cid=i + 1) for i, w in enumerate([5.0, 3.0, 1.0, 4.0])]
        sos = _make_sos(clauses)

        # Weight-based should pick lightest (weight=1.0)
        c, name = sel.select_given(sos, 0)
        assert c is not None
        assert c.weight == 1.0
        assert name == "W"  # Default weight rule is named "W"

    def test_no_provider_matches_traditional(self) -> None:
        """With no embedding provider, selection is pure traditional."""
        config = MLSelectionConfig(enabled=True)
        sel = EmbeddingEnhancedSelection(
            rules=[SelectionRule("W", SelectionOrder.WEIGHT, part=5),
                   SelectionRule("A", SelectionOrder.AGE, part=1)],
            embedding_provider=None,
            ml_config=config,
        )

        clauses = [_make_weighted_clause(w, cid=i + 1) for i, w in enumerate([5.0, 3.0, 1.0])]
        sos = _make_sos(clauses)

        c, name = sel.select_given(sos, 0)
        assert c is not None
        assert c.weight == 1.0

    def test_age_selection_never_uses_ml(self) -> None:
        """Age-based selection steps always use FIFO, never ML."""
        config = MLSelectionConfig(enabled=True, ml_weight=1.0)
        provider = MockEmbeddingProvider(
            embeddings={1: [1, 0, 0, 0], 2: [0, 1, 0, 0], 3: [0, 0, 1, 0]},
        )
        # Rules: only age-based
        sel = EmbeddingEnhancedSelection(
            rules=[SelectionRule("A", SelectionOrder.AGE, part=1)],
            embedding_provider=provider,
            ml_config=config,
        )

        c1 = _make_weighted_clause(10.0, cid=1)
        c2 = _make_weighted_clause(1.0, cid=2)
        sos = _make_sos([c1, c2])

        c, name = sel.select_given(sos, 0)
        # Should pick c1 (oldest/first) regardless of weight
        assert c is c1
        assert name == "A"

    def test_ratio_cycling_preserved(self) -> None:
        """5:1 weight:age ratio cycling works correctly with ML."""
        config = MLSelectionConfig(enabled=False)
        sel = EmbeddingEnhancedSelection(
            rules=[SelectionRule("W", SelectionOrder.WEIGHT, part=5),
                   SelectionRule("A", SelectionOrder.AGE, part=1)],
            ml_config=config,
        )

        selections = []
        for i in range(6):
            c1 = _make_weighted_clause(10.0, cid=i * 2 + 1)
            c2 = _make_weighted_clause(1.0, cid=i * 2 + 2)
            sos = _make_sos([c1, c2])
            _, name = sel.select_given(sos, i)
            selections.append(name)

        # Default 5:1 ratio: WWWWWA
        assert selections == ["W", "W", "W", "W", "W", "A"]

    def test_empty_sos(self) -> None:
        sel = EmbeddingEnhancedSelection()
        sos = ClauseList("sos")
        c, name = sel.select_given(sos, 0)
        assert c is None
        assert name == ""


# ── ML selection tests ───────────────────────────────────────────────────────


class TestMLSelection:
    """Test ML-enhanced selection scoring."""

    def test_ml_prefers_diverse_clause(self) -> None:
        """ML should prefer clauses dissimilar to recent givens."""
        config = MLSelectionConfig(
            enabled=True,
            ml_weight=0.8,
            diversity_weight=1.0,
            proof_potential_weight=0.0,
            min_sos_for_ml=1,
        )
        # Recent givens cluster around [1,0,0,0]
        provider = MockEmbeddingProvider(embeddings={
            1: [1.0, 0.0, 0.0, 0.0],  # similar to recent
            2: [0.0, 0.0, 1.0, 0.0],  # diverse
        })
        sel = EmbeddingEnhancedSelection(
            rules=_WEIGHT_AGE_RULES,
            embedding_provider=provider,
            ml_config=config,
        )
        # Seed recent embeddings
        sel._recent_embeddings.append([1.0, 0.0, 0.0, 0.0])
        sel._recent_embeddings.append([0.9, 0.1, 0.0, 0.0])

        c1 = _make_weighted_clause(1.0, cid=1)  # similar to recent, same weight
        c2 = _make_weighted_clause(1.0, cid=2)  # diverse, same weight
        sos = _make_sos([c1, c2])

        c, name = sel.select_given(sos, 0)
        assert c is c2  # diverse clause preferred
        assert "+ML" in name

    def test_ml_blending_with_weight(self) -> None:
        """Heavy ML weight can override traditional weight preference."""
        config = MLSelectionConfig(
            enabled=True,
            ml_weight=0.9,
            diversity_weight=1.0,
            proof_potential_weight=0.0,
            min_sos_for_ml=1,
        )
        provider = MockEmbeddingProvider(embeddings={
            1: [1.0, 0.0, 0.0, 0.0],  # similar to recent
            2: [0.0, 1.0, 0.0, 0.0],  # diverse
        })
        sel = EmbeddingEnhancedSelection(
            rules=_WEIGHT_AGE_RULES,
            embedding_provider=provider,
            ml_config=config,
        )
        sel._recent_embeddings.append([1.0, 0.0, 0.0, 0.0])

        # c1 is lighter but similar; c2 is heavier but diverse
        c1 = _make_weighted_clause(1.0, cid=1)
        c2 = _make_weighted_clause(5.0, cid=2)
        sos = _make_sos([c1, c2])

        c, name = sel.select_given(sos, 0)
        # With 90% ML weight and strong diversity, c2 should win
        assert c is c2
        assert "+ML" in name

    def test_low_ml_weight_defers_to_traditional(self) -> None:
        """Low ML weight makes traditional weight dominate."""
        config = MLSelectionConfig(
            enabled=True,
            ml_weight=0.05,
            min_sos_for_ml=1,
        )
        provider = MockEmbeddingProvider(embeddings={
            1: [1.0, 0.0, 0.0, 0.0],
            2: [0.0, 1.0, 0.0, 0.0],
        })
        sel = EmbeddingEnhancedSelection(
            rules=_WEIGHT_AGE_RULES,
            embedding_provider=provider,
            ml_config=config,
        )
        sel._recent_embeddings.append([1.0, 0.0, 0.0, 0.0])

        c1 = _make_weighted_clause(1.0, cid=1)   # light but similar
        c2 = _make_weighted_clause(100.0, cid=2)  # heavy but diverse
        sos = _make_sos([c1, c2])

        c, name = sel.select_given(sos, 0)
        # With only 5% ML weight, lighter clause should win
        assert c is c1

    def test_min_sos_threshold(self) -> None:
        """ML not used when SOS is below min_sos_for_ml."""
        config = MLSelectionConfig(
            enabled=True,
            ml_weight=1.0,
            min_sos_for_ml=5,
        )
        provider = MockEmbeddingProvider(
            embeddings={1: [1, 0, 0, 0], 2: [0, 1, 0, 0]},
        )
        sel = EmbeddingEnhancedSelection(
            rules=_WEIGHT_AGE_RULES,
            embedding_provider=provider,
            ml_config=config,
        )

        # Only 2 clauses, below threshold of 5
        sos = _make_sos([
            _make_weighted_clause(5.0, cid=1),
            _make_weighted_clause(1.0, cid=2),
        ])

        c, name = sel.select_given(sos, 0)
        # Should use traditional (lightest)
        assert c is not None
        assert c.weight == 1.0
        assert "+ML" not in name


# ── Fallback tests ───────────────────────────────────────────────────────────


class TestFallback:
    """Test graceful fallback on embedding failures."""

    def test_fallback_on_provider_error(self) -> None:
        """Provider errors cause fallback to traditional selection."""
        config = MLSelectionConfig(
            enabled=True,
            ml_weight=1.0,
            min_sos_for_ml=1,
            fallback_on_error=True,
        )
        provider = MockEmbeddingProvider(raise_error=True)
        sel = EmbeddingEnhancedSelection(
            rules=_WEIGHT_AGE_RULES,
            embedding_provider=provider,
            ml_config=config,
        )

        clauses = [_make_weighted_clause(w, cid=i + 1) for i, w in enumerate([5.0, 1.0])]
        sos = _make_sos(clauses)

        c, name = sel.select_given(sos, 0)
        assert c is not None
        assert c.weight == 1.0  # traditional fallback picks lightest
        assert sel.ml_stats.fallback_count == 1

    def test_missing_embeddings_use_traditional_score(self) -> None:
        """Clauses without embeddings get scored purely on weight."""
        config = MLSelectionConfig(
            enabled=True,
            ml_weight=0.5,
            min_sos_for_ml=1,
        )
        # Only provide embedding for clause 2
        provider = MockEmbeddingProvider(
            embeddings={2: [0, 1, 0, 0]},
            fail_on={1},
        )
        sel = EmbeddingEnhancedSelection(
            rules=_WEIGHT_AGE_RULES,
            embedding_provider=provider,
            ml_config=config,
        )

        c1 = _make_weighted_clause(1.0, cid=1)  # no embedding
        c2 = _make_weighted_clause(2.0, cid=2)  # has embedding
        sos = _make_sos([c1, c2])

        c, name = sel.select_given(sos, 0)
        assert c is not None
        assert sel.ml_stats.embedding_miss_count >= 1


# ── Statistics tests ─────────────────────────────────────────────────────────


class TestMLSelectionStats:
    def test_stats_tracking(self) -> None:
        stats = MLSelectionStats()
        stats.record_ml_selection(0.8)
        stats.record_ml_selection(0.6)
        stats.record_traditional()
        stats.record_fallback()

        assert stats.ml_selections == 2
        assert stats.traditional_selections == 2  # 1 traditional + 1 fallback
        assert stats.fallback_count == 1
        assert stats.avg_ml_score == pytest.approx(0.7)

    def test_report_format(self) -> None:
        stats = MLSelectionStats()
        stats.record_ml_selection(0.5)
        report = stats.report()
        assert "ml_selection:" in report
        assert "ML" in report


# ── Diversity window tests ───────────────────────────────────────────────────


class TestDiversityWindow:
    def test_window_size_from_config(self) -> None:
        config = MLSelectionConfig(enabled=True, diversity_window=5)
        sel = EmbeddingEnhancedSelection(ml_config=config)
        assert sel._recent_embeddings.maxlen == 5

    def test_window_eviction(self) -> None:
        config = MLSelectionConfig(enabled=True, diversity_window=2)
        sel = EmbeddingEnhancedSelection(ml_config=config)
        sel._recent_embeddings.append([1, 0])
        sel._recent_embeddings.append([0, 1])
        sel._recent_embeddings.append([1, 1])  # evicts [1, 0]
        assert len(sel._recent_embeddings) == 2

    def test_no_recent_gives_neutral_diversity(self) -> None:
        config = MLSelectionConfig(enabled=True)
        sel = EmbeddingEnhancedSelection(ml_config=config)
        score = sel._diversity_score([1, 0, 0, 0])
        assert score == pytest.approx(0.5)
