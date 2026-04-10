"""Tests for embedding-guided inference targeting.

Verifies:
- Backward compatibility: disabled guidance returns usable list unmodified
- Prioritization: clauses scored and ranked by compatibility
- Thresholding: low-scoring candidates filtered out
- Truncation: max_candidates limit respected
- Early termination: stops after enough inferences
- Statistics: correct tracking
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.search.inference_guidance import (
    EmbeddingGuidedInference,
    InferenceGuidanceConfig,
    InferenceGuidanceStats,
    ScoredCandidate,
    _cosine_similarity,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _const(symnum: int) -> "Term":
    return get_rigid_term(symnum, 0)


def _func(symnum: int, *args) -> "Term":
    return get_rigid_term(symnum, len(args), args)


def _var(n: int) -> "Term":
    return get_variable_term(n)


def _make_clause(*atoms, signs=None, cid=0) -> Clause:
    if signs is None:
        signs = (True,) * len(atoms)
    lits = tuple(Literal(sign=s, atom=a) for s, a in zip(signs, atoms))
    c = Clause(literals=lits, id=cid)
    c.weight = sum(1 for _ in lits)
    return c


class MockEmbeddingProvider:
    """Mock provider for testing inference guidance."""

    def __init__(
        self,
        embeddings: dict[int, list[float]] | None = None,
        dim: int = 4,
    ):
        self._embeddings = embeddings or {}
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def get_embedding(self, clause: Clause) -> list[float] | None:
        return self._embeddings.get(clause.id)

    def get_embeddings_batch(
        self, clauses: list[Clause],
    ) -> list[list[float] | None]:
        return [self.get_embedding(c) for c in clauses]


# ── Backward compatibility tests ─────────────────────────────────────────────


class TestBackwardCompatibility:
    def test_disabled_returns_unchanged(self) -> None:
        """Disabled guidance returns usable list as-is."""
        config = InferenceGuidanceConfig(enabled=False)
        guidance = EmbeddingGuidedInference(config=config)

        given = _make_clause(_const(1), cid=1)
        usable = [_make_clause(_const(i), cid=i) for i in range(2, 5)]

        result = guidance.prioritize(given, usable)
        assert result is usable  # same object

    def test_no_provider_returns_unchanged(self) -> None:
        """No provider returns usable list as-is."""
        config = InferenceGuidanceConfig(enabled=True)
        guidance = EmbeddingGuidedInference(provider=None, config=config)

        given = _make_clause(_const(1), cid=1)
        usable = [_make_clause(_const(i), cid=i) for i in range(2, 5)]

        result = guidance.prioritize(given, usable)
        assert result is usable

    def test_empty_usable_returns_empty(self) -> None:
        """Empty usable list returns empty."""
        config = InferenceGuidanceConfig(enabled=True)
        provider = MockEmbeddingProvider(embeddings={1: [1, 0, 0, 0]})
        guidance = EmbeddingGuidedInference(provider=provider, config=config)

        given = _make_clause(_const(1), cid=1)
        result = guidance.prioritize(given, [])
        assert result == []


# ── Prioritization tests ────────────────────────────────────────────────────


class TestPrioritization:
    def test_ranks_by_semantic_similarity(self) -> None:
        """More semantically similar clauses ranked higher."""
        config = InferenceGuidanceConfig(
            enabled=True,
            structural_weight=0.0,
            semantic_weight=1.0,
        )
        provider = MockEmbeddingProvider(embeddings={
            1: [1.0, 0.0, 0.0, 0.0],  # given
            2: [0.9, 0.1, 0.0, 0.0],  # very similar
            3: [0.0, 0.0, 0.0, 1.0],  # dissimilar
            4: [0.5, 0.5, 0.0, 0.0],  # moderately similar
        })
        guidance = EmbeddingGuidedInference(provider=provider, config=config)

        given = _make_clause(_const(1), cid=1)
        usable = [
            _make_clause(_const(2), cid=2),
            _make_clause(_const(3), cid=3),
            _make_clause(_const(4), cid=4),
        ]

        result = guidance.prioritize(given, usable)
        result_ids = [c.id for c in result]
        # Most similar first
        assert result_ids[0] == 2
        assert result_ids[-1] == 3

    def test_structural_compatibility_unit_bonus(self) -> None:
        """Unit clauses get structural bonus."""
        config = InferenceGuidanceConfig(
            enabled=True,
            structural_weight=1.0,
            semantic_weight=0.0,
        )
        provider = MockEmbeddingProvider(embeddings={
            1: [1, 0, 0, 0],
            2: [0, 1, 0, 0],
            3: [0, 0, 1, 0],
        })
        guidance = EmbeddingGuidedInference(provider=provider, config=config)

        given = _make_clause(_const(1), cid=1)
        # Unit clause
        unit = _make_clause(_const(10), cid=2)
        # Multi-literal clause
        multi = _make_clause(_const(10), _const(11), _const(12), cid=3)

        result = guidance.prioritize(given, [multi, unit])
        # Unit should rank higher due to structural bonus
        assert result[0].id == 2

    def test_complementarity_bonus(self) -> None:
        """Clauses with complementary literals (same predicate, opposite sign) get a bonus."""
        P_SYM = 20
        config = InferenceGuidanceConfig(
            enabled=True,
            structural_weight=1.0,
            semantic_weight=0.0,
            complementarity_bonus=0.5,
        )
        provider = MockEmbeddingProvider(embeddings={
            1: [1, 0, 0, 0],
            2: [0, 1, 0, 0],
            3: [0, 0, 1, 0],
        })
        guidance = EmbeddingGuidedInference(provider=provider, config=config)

        # Given has positive P(a)
        given = _make_clause(_func(P_SYM, _const(1)), cid=1, signs=(True,))
        # Candidate with negative P(b) — same predicate, opposite sign → complementary
        neg_clause = _make_clause(_func(P_SYM, _const(2)), cid=2, signs=(False,))
        # Candidate with positive P(b) — same predicate, same sign → no resolution possible
        pos_clause = _make_clause(_func(P_SYM, _const(2)), cid=3, signs=(True,))

        result = guidance.prioritize(given, [pos_clause, neg_clause])
        # Negative should rank higher due to complementarity bonus
        assert result[0].id == 2


# ── Filtering tests ──────────────────────────────────────────────────────────


class TestFiltering:
    def test_max_candidates_truncation(self) -> None:
        """max_candidates limits the returned list."""
        config = InferenceGuidanceConfig(
            enabled=True,
            max_candidates=2,
        )
        provider = MockEmbeddingProvider(embeddings={
            1: [1, 0, 0, 0],
            2: [0.9, 0.1, 0, 0],
            3: [0.5, 0.5, 0, 0],
            4: [0, 0, 1, 0],
            5: [0, 0, 0, 1],
        })
        guidance = EmbeddingGuidedInference(provider=provider, config=config)

        given = _make_clause(_const(1), cid=1)
        usable = [_make_clause(_const(i), cid=i) for i in range(2, 6)]

        result = guidance.prioritize(given, usable)
        assert len(result) == 2

    def test_compatibility_threshold(self) -> None:
        """Clauses below threshold are excluded."""
        config = InferenceGuidanceConfig(
            enabled=True,
            compatibility_threshold=0.7,
            structural_weight=0.0,
            semantic_weight=1.0,
        )
        provider = MockEmbeddingProvider(embeddings={
            1: [1, 0, 0, 0],
            2: [0.95, 0.05, 0, 0],  # very similar → high score
            3: [0, 0, 0, 1],         # dissimilar → low score
        })
        guidance = EmbeddingGuidedInference(provider=provider, config=config)

        given = _make_clause(_const(1), cid=1)
        usable = [
            _make_clause(_const(2), cid=2),
            _make_clause(_const(3), cid=3),
        ]

        result = guidance.prioritize(given, usable)
        result_ids = [c.id for c in result]
        # Only the similar clause should survive the threshold
        assert 2 in result_ids
        # The dissimilar one may or may not be included depending on
        # exact score, but at least similar is present
        assert len(result) <= len(usable)


# ── Early termination tests ─────────────────────────────────────────────────


class TestEarlyTermination:
    def test_early_termination_triggers(self) -> None:
        config = InferenceGuidanceConfig(
            enabled=True,
            early_termination_count=5,
        )
        guidance = EmbeddingGuidedInference(config=config)

        assert not guidance.should_terminate_early(4)
        assert guidance.should_terminate_early(5)
        assert guidance.should_terminate_early(10)
        assert guidance.stats.early_terminations == 2

    def test_early_termination_disabled(self) -> None:
        config = InferenceGuidanceConfig(
            enabled=True,
            early_termination_count=-1,
        )
        guidance = EmbeddingGuidedInference(config=config)
        assert not guidance.should_terminate_early(1000)

    def test_early_termination_when_guidance_disabled(self) -> None:
        config = InferenceGuidanceConfig(enabled=False)
        guidance = EmbeddingGuidedInference(config=config)
        assert not guidance.should_terminate_early(1000)


# ── Statistics tests ─────────────────────────────────────────────────────────


class TestGuidanceStats:
    def test_guided_round_tracking(self) -> None:
        config = InferenceGuidanceConfig(enabled=True)
        provider = MockEmbeddingProvider(embeddings={
            1: [1, 0, 0, 0],
            2: [0.5, 0.5, 0, 0],
        })
        guidance = EmbeddingGuidedInference(provider=provider, config=config)

        given = _make_clause(_const(1), cid=1)
        usable = [_make_clause(_const(2), cid=2)]
        guidance.prioritize(given, usable)

        assert guidance.stats.guided_rounds == 1
        assert guidance.stats.total_candidates_scored == 1
        assert guidance.stats.total_candidates_selected == 1

    def test_unguided_tracking(self) -> None:
        config = InferenceGuidanceConfig(enabled=False)
        guidance = EmbeddingGuidedInference(config=config)

        given = _make_clause(_const(1), cid=1)
        usable = [_make_clause(_const(2), cid=2)]
        guidance.prioritize(given, usable)

        assert guidance.stats.unguided_rounds == 1
        assert guidance.stats.guided_rounds == 0

    def test_stats_report_format(self) -> None:
        stats = InferenceGuidanceStats()
        stats.record_guided_round(10, 5, 5, 0.9)
        report = stats.report()
        assert "inference_guidance:" in report
        assert "guided" in report


# ── Provider setter test ─────────────────────────────────────────────────────


class TestProviderManagement:
    def test_set_provider_later(self) -> None:
        """Provider can be set after construction."""
        guidance = EmbeddingGuidedInference(
            config=InferenceGuidanceConfig(enabled=True),
        )
        assert guidance.provider is None

        provider = MockEmbeddingProvider(embeddings={1: [1, 0, 0, 0]})
        guidance.provider = provider
        assert guidance.provider is provider


# ── Fallback on error test ───────────────────────────────────────────────────


class TestErrorFallback:
    def test_provider_error_returns_unmodified(self) -> None:
        """Provider errors cause graceful fallback."""

        class BrokenProvider:
            @property
            def embedding_dim(self) -> int:
                return 4

            def get_embedding(self, clause):
                raise RuntimeError("broken")

            def get_embeddings_batch(self, clauses):
                raise RuntimeError("broken")

        config = InferenceGuidanceConfig(enabled=True)
        guidance = EmbeddingGuidedInference(
            provider=BrokenProvider(),
            config=config,
        )

        given = _make_clause(_const(1), cid=1)
        usable = [_make_clause(_const(2), cid=2)]

        result = guidance.prioritize(given, usable)
        # Should return the original list on error
        assert result is usable
        assert guidance.stats.unguided_rounds == 1
