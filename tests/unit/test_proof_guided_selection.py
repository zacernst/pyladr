"""Comprehensive tests for proof-guided clause selection.

Tests cover: proof pattern memory, similarity scoring, exploration/exploitation
balance, CLI parameter integration, backward compatibility, PrioritySOS
proof-guided heap, multi-proof learning, and decay behavior.
"""

from __future__ import annotations

import math

import pytest

from pyladr.search.proof_pattern_memory import (
    ProofGuidedConfig,
    ProofPatternMemory,
    _cosine_similarity,
    proof_guided_score,
)
from pyladr.search.given_clause import GivenClauseSearch, Proof, SearchOptions
from pyladr.search.priority_sos import PrioritySOS
from pyladr.search.selection import (
    GivenSelection,
    SelectionOrder,
    SelectionRule,
)
from tests.factories import make_clause, make_const, make_func, make_neg_lit, make_pos_lit, make_var


# ── Cosine Similarity ──────────────────────────────────────────────────────


class TestCosineSimilarity:
    """Verify cosine similarity computation correctness."""

    def test_identical_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        assert _cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_both_zero_vectors(self) -> None:
        a = [0.0, 0.0]
        b = [0.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_non_unit_vectors(self) -> None:
        a = [3.0, 4.0]
        b = [6.0, 8.0]
        # Same direction, different magnitudes
        assert _cosine_similarity(a, b) == pytest.approx(1.0)

    def test_known_angle(self) -> None:
        # 45 degrees: cos(45°) ≈ 0.7071
        a = [1.0, 0.0]
        b = [1.0, 1.0]
        expected = 1.0 / math.sqrt(2)
        assert _cosine_similarity(a, b) == pytest.approx(expected, abs=1e-6)


# ── ProofGuidedConfig ─────────────────────────────────────────────────────


class TestProofGuidedConfig:
    """Verify config defaults and immutability."""

    def test_defaults(self) -> None:
        cfg = ProofGuidedConfig()
        assert cfg.enabled is True
        assert cfg.exploitation_ratio == 0.7
        assert cfg.max_patterns == 500
        assert cfg.decay_rate == 0.95
        assert cfg.min_similarity_threshold == 0.1
        assert cfg.warmup_proofs == 1

    def test_custom_config(self) -> None:
        cfg = ProofGuidedConfig(
            exploitation_ratio=0.9,
            max_patterns=100,
            decay_rate=0.8,
        )
        assert cfg.exploitation_ratio == 0.9
        assert cfg.max_patterns == 100
        assert cfg.decay_rate == 0.8

    def test_frozen(self) -> None:
        cfg = ProofGuidedConfig()
        with pytest.raises(AttributeError):
            cfg.exploitation_ratio = 0.5  # type: ignore[misc]


# ── ProofPatternMemory ────────────────────────────────────────────────────


class TestProofPatternMemory:
    """Verify pattern storage, decay, and scoring behavior."""

    def test_initial_state(self) -> None:
        memory = ProofPatternMemory()
        assert memory.proof_count == 0
        assert memory.pattern_count == 0
        assert not memory.is_warmed_up

    def test_record_proof_increments_count(self) -> None:
        memory = ProofPatternMemory()
        memory.record_proof([[1.0, 0.0]])
        assert memory.proof_count == 1
        assert memory.pattern_count == 1

    def test_record_empty_embeddings_noop(self) -> None:
        memory = ProofPatternMemory()
        memory.record_proof([])
        assert memory.proof_count == 0
        assert memory.pattern_count == 0

    def test_warmup_threshold(self) -> None:
        cfg = ProofGuidedConfig(warmup_proofs=2)
        memory = ProofPatternMemory(config=cfg)
        assert not memory.is_warmed_up

        memory.record_proof([[1.0, 0.0]])
        assert not memory.is_warmed_up

        memory.record_proof([[0.0, 1.0]])
        assert memory.is_warmed_up

    def test_exploitation_score_before_warmup(self) -> None:
        memory = ProofPatternMemory(config=ProofGuidedConfig(warmup_proofs=2))
        memory.record_proof([[1.0, 0.0]])
        # Not warmed up yet → neutral score
        assert memory.exploitation_score([1.0, 0.0]) == pytest.approx(0.5)

    def test_exploitation_score_exact_match(self) -> None:
        memory = ProofPatternMemory()
        memory.record_proof([[1.0, 0.0, 0.0]])
        assert memory.exploitation_score([1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_exploitation_score_orthogonal(self) -> None:
        memory = ProofPatternMemory(
            config=ProofGuidedConfig(min_similarity_threshold=0.0)
        )
        memory.record_proof([[1.0, 0.0, 0.0]])
        # Orthogonal vector → similarity = 0 → score = 0
        assert memory.exploitation_score([0.0, 1.0, 0.0]) == pytest.approx(0.0)

    def test_exploitation_score_below_threshold(self) -> None:
        memory = ProofPatternMemory(
            config=ProofGuidedConfig(min_similarity_threshold=0.5)
        )
        memory.record_proof([[1.0, 0.0, 0.0]])
        # Slight similarity but below threshold → 0
        emb = [0.3, 0.95, 0.0]  # cos(emb, [1,0,0]) = 0.3/|emb| ≈ 0.3
        score = memory.exploitation_score(emb)
        assert score == pytest.approx(0.0)

    def test_max_similarity_across_patterns(self) -> None:
        memory = ProofPatternMemory()
        memory.record_proof([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        # Query similar to second pattern
        score = memory.exploitation_score([0.0, 1.0, 0.0])
        assert score == pytest.approx(1.0)

    def test_decay_reduces_weights(self) -> None:
        cfg = ProofGuidedConfig(decay_rate=0.5)
        memory = ProofPatternMemory(config=cfg)
        memory.record_proof([[1.0, 0.0]])
        # First pattern weight = 1.0

        memory.record_proof([[0.0, 1.0]])
        # First pattern decayed to 0.5, second pattern = 1.0

        # Query matching first pattern: score = sim * weight = 1.0 * 0.5 = 0.5
        score = memory.exploitation_score([1.0, 0.0])
        assert score == pytest.approx(0.5)

    def test_no_decay_when_rate_is_one(self) -> None:
        cfg = ProofGuidedConfig(decay_rate=1.0)
        memory = ProofPatternMemory(config=cfg)
        memory.record_proof([[1.0, 0.0]])
        memory.record_proof([[0.0, 1.0]])

        # First pattern not decayed
        score = memory.exploitation_score([1.0, 0.0])
        assert score == pytest.approx(1.0)

    def test_max_patterns_eviction(self) -> None:
        cfg = ProofGuidedConfig(max_patterns=3)
        memory = ProofPatternMemory(config=cfg)
        memory.record_proof([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        assert memory.pattern_count == 3

        # Adding one more evicts the oldest
        memory.record_proof([[0.7, 0.7]])
        assert memory.pattern_count == 3

    def test_clear_resets_state(self) -> None:
        memory = ProofPatternMemory()
        memory.record_proof([[1.0, 0.0]])
        memory.clear()
        assert memory.proof_count == 0
        assert memory.pattern_count == 0
        assert not memory.is_warmed_up

    def test_centroid_score_basic(self) -> None:
        memory = ProofPatternMemory()
        memory.record_proof([[1.0, 0.0, 0.0]])
        # Centroid is the single pattern itself
        score = memory.centroid_score([1.0, 0.0, 0.0])
        assert score == pytest.approx(1.0)

    def test_centroid_score_before_warmup(self) -> None:
        cfg = ProofGuidedConfig(warmup_proofs=5)
        memory = ProofPatternMemory(config=cfg)
        memory.record_proof([[1.0, 0.0]])
        assert memory.centroid_score([1.0, 0.0]) == pytest.approx(0.5)

    def test_multiple_proofs_accumulate(self) -> None:
        memory = ProofPatternMemory()
        memory.record_proof([[1.0, 0.0]])
        memory.record_proof([[0.0, 1.0]])
        memory.record_proof([[0.5, 0.5]])
        assert memory.proof_count == 3
        assert memory.pattern_count == 3


# ── proof_guided_score ────────────────────────────────────────────────────


class TestProofGuidedScore:
    """Verify blended scoring function."""

    def test_returns_diversity_when_disabled(self) -> None:
        cfg = ProofGuidedConfig(enabled=False)
        memory = ProofPatternMemory(config=cfg)
        score = proof_guided_score([1.0, 0.0], memory, 0.8, cfg)
        assert score == pytest.approx(0.8)

    def test_returns_diversity_before_warmup(self) -> None:
        cfg = ProofGuidedConfig(warmup_proofs=3)
        memory = ProofPatternMemory(config=cfg)
        memory.record_proof([[1.0, 0.0]])  # Only 1 proof, need 3
        score = proof_guided_score([1.0, 0.0], memory, 0.6, cfg)
        assert score == pytest.approx(0.6)

    def test_pure_exploitation(self) -> None:
        cfg = ProofGuidedConfig(exploitation_ratio=1.0)
        memory = ProofPatternMemory(config=cfg)
        memory.record_proof([[1.0, 0.0, 0.0]])

        # Perfect match → exploitation = 1.0
        score = proof_guided_score([1.0, 0.0, 0.0], memory, 0.3, cfg)
        assert score == pytest.approx(1.0)

    def test_pure_exploration(self) -> None:
        cfg = ProofGuidedConfig(exploitation_ratio=0.0)
        memory = ProofPatternMemory(config=cfg)
        memory.record_proof([[1.0, 0.0]])

        score = proof_guided_score([1.0, 0.0], memory, 0.7, cfg)
        assert score == pytest.approx(0.7)

    def test_default_blend(self) -> None:
        # Default: 0.7 exploitation, 0.3 exploration
        cfg = ProofGuidedConfig()
        memory = ProofPatternMemory(config=cfg)
        memory.record_proof([[1.0, 0.0, 0.0]])

        exploitation = 1.0  # Perfect match
        diversity = 0.5
        expected = 0.7 * exploitation + 0.3 * diversity
        score = proof_guided_score([1.0, 0.0, 0.0], memory, diversity, cfg)
        assert score == pytest.approx(expected)

    def test_blend_with_partial_similarity(self) -> None:
        cfg = ProofGuidedConfig(exploitation_ratio=0.5, min_similarity_threshold=0.0)
        memory = ProofPatternMemory(config=cfg)
        memory.record_proof([[1.0, 0.0]])

        emb = [1.0, 1.0]  # cos similarity to [1,0] = 1/sqrt(2) ≈ 0.707
        diversity = 0.4
        exploitation = _cosine_similarity(emb, [1.0, 0.0])
        expected = 0.5 * exploitation + 0.5 * diversity
        score = proof_guided_score(emb, memory, diversity, cfg)
        assert score == pytest.approx(expected, abs=1e-6)


# ── SearchOptions Integration ─────────────────────────────────────────────


class TestSearchOptionsProofGuided:
    """Verify proof-guided SearchOptions fields and validation."""

    def test_defaults_disabled(self) -> None:
        opts = SearchOptions()
        assert opts.proof_guided is False
        assert opts.proof_guided_weight == 0.0
        assert opts.proof_guided_exploitation_ratio == 0.7
        assert opts.proof_guided_max_patterns == 500
        assert opts.proof_guided_decay_rate == 0.95
        assert opts.proof_guided_min_similarity == 0.1
        assert opts.proof_guided_warmup_proofs == 1

    def test_custom_values(self) -> None:
        opts = SearchOptions(
            proof_guided=True,
            forte_embeddings=True,
            proof_guided_weight=3.0,
            proof_guided_exploitation_ratio=0.9,
            proof_guided_max_patterns=200,
            proof_guided_decay_rate=0.8,
            proof_guided_min_similarity=0.2,
            proof_guided_warmup_proofs=3,
        )
        assert opts.proof_guided_exploitation_ratio == 0.9
        assert opts.proof_guided_max_patterns == 200

    def test_exploitation_ratio_bounds(self) -> None:
        with pytest.raises(ValueError, match="proof_guided_exploitation_ratio"):
            SearchOptions(proof_guided_exploitation_ratio=1.5)
        with pytest.raises(ValueError, match="proof_guided_exploitation_ratio"):
            SearchOptions(proof_guided_exploitation_ratio=-0.1)

    def test_decay_rate_bounds(self) -> None:
        with pytest.raises(ValueError, match="proof_guided_decay_rate"):
            SearchOptions(proof_guided_decay_rate=1.5)

    def test_max_patterns_bounds(self) -> None:
        with pytest.raises(ValueError, match="proof_guided_max_patterns"):
            SearchOptions(proof_guided_max_patterns=0)

    def test_warmup_proofs_bounds(self) -> None:
        with pytest.raises(ValueError, match="proof_guided_warmup_proofs"):
            SearchOptions(proof_guided_warmup_proofs=-1)

    def test_weight_bounds(self) -> None:
        with pytest.raises(ValueError, match="proof_guided_weight"):
            SearchOptions(proof_guided_weight=-1.0)

    def test_semantic_warning_without_forte(self) -> None:
        opts = SearchOptions(proof_guided=True, forte_embeddings=False)
        warnings = opts.validate()
        assert any("proof_guided" in w and "forte_embeddings" in w for w in warnings)

    def test_no_warning_with_forte(self) -> None:
        opts = SearchOptions(proof_guided=True, forte_embeddings=True)
        warnings = opts.validate()
        proof_guided_warnings = [w for w in warnings if "proof_guided" in w]
        assert len(proof_guided_warnings) == 0


# ── SelectionOrder.PROOF_GUIDED ──────────────────────────────────────────


class TestSelectionOrderProofGuided:
    """Verify PROOF_GUIDED enum and selection cycle integration."""

    def test_enum_value(self) -> None:
        assert SelectionOrder.PROOF_GUIDED == 6

    def test_selection_rule_creation(self) -> None:
        rule = SelectionRule("PG", SelectionOrder.PROOF_GUIDED, part=2)
        assert rule.name == "PG"
        assert rule.order == SelectionOrder.PROOF_GUIDED
        assert rule.part == 2

    def test_proof_guided_in_selection_cycle(self) -> None:
        rules = [
            SelectionRule("A", SelectionOrder.AGE, part=1),
            SelectionRule("W", SelectionOrder.WEIGHT, part=4),
            SelectionRule("PG", SelectionOrder.PROOF_GUIDED, part=2),
        ]
        sel = GivenSelection(rules=rules)
        assert sel._cycle_size == 7

    def test_search_options_creates_pg_rule(self) -> None:
        opts = SearchOptions(
            forte_embeddings=True,
            proof_guided=True,
            proof_guided_weight=2.0,
        )
        search = GivenClauseSearch(options=opts)
        rule_names = [r.name for r in search._selection.rules]
        assert "PG" in rule_names

    def test_no_pg_rule_when_weight_zero(self) -> None:
        opts = SearchOptions(
            forte_embeddings=True,
            proof_guided=True,
            proof_guided_weight=0.0,
        )
        search = GivenClauseSearch(options=opts)
        rule_names = [r.name for r in search._selection.rules]
        assert "PG" not in rule_names

    def test_no_pg_rule_when_disabled(self) -> None:
        opts = SearchOptions(forte_embeddings=True, proof_guided=False)
        search = GivenClauseSearch(options=opts)
        rule_names = [r.name for r in search._selection.rules]
        assert "PG" not in rule_names


# ── PrioritySOS Proof-Guided Heap ─────────────────────────────────────────


class TestPrioritySosProofGuided:
    """Verify PrioritySOS proof-guided heap operations."""

    def _make_sos_with_clauses(self) -> tuple[PrioritySOS, list[Clause]]:
        sos = PrioritySOS("test")
        c1 = make_clause(make_pos_lit(make_const(1)), weight=5.0, clause_id=1)
        c2 = make_clause(make_pos_lit(make_const(2)), weight=3.0, clause_id=2)
        c3 = make_clause(make_pos_lit(make_const(3)), weight=7.0, clause_id=3)
        for c in [c1, c2, c3]:
            sos.append(c)
        return sos, [c1, c2, c3]

    def test_pop_best_proof_guided_no_scorer_falls_back(self) -> None:
        """Without scorer, falls back to FORTE (which falls back to None without embeddings)."""
        sos, clauses = self._make_sos_with_clauses()
        result = sos.pop_best_proof_guided()
        # No scorer, no FORTE embeddings → None from FORTE fallback
        assert result is None

    def test_pop_best_proof_guided_with_scorer(self) -> None:
        sos, clauses = self._make_sos_with_clauses()
        # Scorer: clause 2 gets highest score
        scores = {1: 0.3, 2: 0.9, 3: 0.5}
        sos._proof_guided_scorer = lambda cid: scores.get(cid, 0.0)

        best = sos.pop_best_proof_guided()
        assert best is not None
        assert best.id == 2  # Highest score

    def test_pop_best_proof_guided_removes_from_active(self) -> None:
        sos, clauses = self._make_sos_with_clauses()
        scores = {1: 0.9, 2: 0.3, 3: 0.5}
        sos._proof_guided_scorer = lambda cid: scores.get(cid, 0.0)

        best = sos.pop_best_proof_guided()
        assert best.id == 1
        assert not sos.contains(clauses[0])
        assert sos.length == 2

    def test_pop_exhausts_heap(self) -> None:
        sos, clauses = self._make_sos_with_clauses()
        scores = {1: 0.3, 2: 0.9, 3: 0.5}
        sos._proof_guided_scorer = lambda cid: scores.get(cid, 0.0)

        results = []
        for _ in range(4):  # One more than available
            c = sos.pop_best_proof_guided()
            if c is None:
                break
            results.append(c.id)

        assert results == [2, 3, 1]  # Descending score order

    def test_lazy_initialization(self) -> None:
        sos, clauses = self._make_sos_with_clauses()
        scores = {1: 0.5, 2: 0.8, 3: 0.3}
        sos._proof_guided_scorer = lambda cid: scores.get(cid, 0.0)

        assert not sos._proof_guided_initialized
        sos.pop_best_proof_guided()
        assert sos._proof_guided_initialized

    def test_append_after_initialization(self) -> None:
        sos, clauses = self._make_sos_with_clauses()
        scores = {1: 0.5, 2: 0.3, 3: 0.2, 4: 0.9}
        sos._proof_guided_scorer = lambda cid: scores.get(cid, 0.0)

        # Force initialization
        sos.pop_best_proof_guided()  # pops clause 1 (score 0.5)

        # Append new clause after initialization
        c4 = make_clause(make_pos_lit(make_const(4)), weight=1.0, clause_id=4)
        sos.append(c4)

        # New clause should be in the heap
        best = sos.pop_best_proof_guided()
        assert best is not None
        assert best.id == 4  # Score 0.9, highest remaining

    def test_compact_preserves_proof_guided_heap(self) -> None:
        sos, clauses = self._make_sos_with_clauses()
        scores = {1: 0.3, 2: 0.9, 3: 0.5}
        sos._proof_guided_scorer = lambda cid: scores.get(cid, 0.0)

        # Initialize and remove one
        sos.pop_best_proof_guided()  # removes clause 2
        sos.compact()

        # Should still work after compact
        best = sos.pop_best_proof_guided()
        assert best is not None
        assert best.id == 3  # Next highest score


# ── GivenClauseSearch Integration ─────────────────────────────────────────


class TestGivenClauseSearchProofGuided:
    """Verify proof-guided integration with GivenClauseSearch."""

    def test_memory_initialized_with_forte(self) -> None:
        opts = SearchOptions(forte_embeddings=True, proof_guided=True)
        search = GivenClauseSearch(options=opts)
        assert search.proof_pattern_memory is not None

    def test_memory_none_without_forte(self) -> None:
        opts = SearchOptions(forte_embeddings=False, proof_guided=True)
        search = GivenClauseSearch(options=opts)
        assert search.proof_pattern_memory is None

    def test_memory_none_when_disabled(self) -> None:
        opts = SearchOptions(forte_embeddings=True, proof_guided=False)
        search = GivenClauseSearch(options=opts)
        assert search.proof_pattern_memory is None

    def test_memory_config_propagated(self) -> None:
        opts = SearchOptions(
            forte_embeddings=True,
            proof_guided=True,
            proof_guided_exploitation_ratio=0.9,
            proof_guided_max_patterns=200,
            proof_guided_decay_rate=0.8,
            proof_guided_min_similarity=0.2,
            proof_guided_warmup_proofs=3,
        )
        search = GivenClauseSearch(options=opts)
        memory = search.proof_pattern_memory
        assert memory.config.exploitation_ratio == 0.9
        assert memory.config.max_patterns == 200
        assert memory.config.decay_rate == 0.8
        assert memory.config.min_similarity_threshold == 0.2
        assert memory.config.warmup_proofs == 3

    def test_scorer_wired_to_priority_sos(self) -> None:
        opts = SearchOptions(
            forte_embeddings=True,
            proof_guided=True,
            proof_guided_weight=2.0,
        )
        search = GivenClauseSearch(options=opts)
        sos = search._state.sos
        assert isinstance(sos, PrioritySOS)
        assert sos._proof_guided_scorer is not None

    def test_scorer_not_wired_when_disabled(self) -> None:
        opts = SearchOptions(forte_embeddings=True, proof_guided=False)
        search = GivenClauseSearch(options=opts)
        sos = search._state.sos
        assert isinstance(sos, PrioritySOS)
        assert sos._proof_guided_scorer is None


# ── Backward Compatibility ────────────────────────────────────────────────


class TestBackwardCompatibility:
    """Ensure proof-guided features don't affect existing behavior."""

    def test_default_search_options_unchanged(self) -> None:
        opts = SearchOptions()
        assert not opts.proof_guided
        assert opts.proof_guided_weight == 0.0

    def test_default_selection_rules_unchanged(self) -> None:
        opts = SearchOptions()
        search = GivenClauseSearch(options=opts)
        rules = search._selection.rules
        assert len(rules) == 2
        assert rules[0].name == "A"
        assert rules[1].name == "W"

    def test_forte_only_selection_unchanged(self) -> None:
        opts = SearchOptions(forte_embeddings=True, forte_weight=1.0)
        search = GivenClauseSearch(options=opts)
        rule_names = [r.name for r in search._selection.rules]
        assert rule_names == ["A", "W", "F"]

    def test_priority_sos_unaffected_by_default(self) -> None:
        opts = SearchOptions()
        search = GivenClauseSearch(options=opts)
        sos = search._state.sos
        assert isinstance(sos, PrioritySOS)
        assert sos._proof_guided_scorer is None
        assert not sos._proof_guided_initialized


# ── Multi-Proof Learning ──────────────────────────────────────────────────


class TestMultiProofLearning:
    """Verify behavior with multiple proofs and pattern accumulation."""

    def test_accumulates_patterns_across_proofs(self) -> None:
        memory = ProofPatternMemory()
        memory.record_proof([[1.0, 0.0, 0.0]])
        memory.record_proof([[0.0, 1.0, 0.0]])
        memory.record_proof([[0.0, 0.0, 1.0]])
        assert memory.proof_count == 3
        assert memory.pattern_count == 3

    def test_multiple_embeddings_per_proof(self) -> None:
        memory = ProofPatternMemory()
        memory.record_proof([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ])
        assert memory.proof_count == 1
        assert memory.pattern_count == 3

    def test_decay_across_multiple_proofs(self) -> None:
        cfg = ProofGuidedConfig(decay_rate=0.5)
        memory = ProofPatternMemory(config=cfg)

        memory.record_proof([[1.0, 0.0]])  # weight = 1.0
        memory.record_proof([[0.0, 1.0]])  # first decayed to 0.5, second = 1.0
        memory.record_proof([[0.5, 0.5]])  # first: 0.25, second: 0.5, third: 1.0

        assert memory.proof_count == 3
        # Most recent pattern has highest weight
        score_recent = memory.exploitation_score([0.5, 0.5])
        score_oldest = memory.exploitation_score([1.0, 0.0])
        assert score_recent > score_oldest

    def test_bounded_memory_with_many_proofs(self) -> None:
        cfg = ProofGuidedConfig(max_patterns=5)
        memory = ProofPatternMemory(config=cfg)

        for i in range(10):
            emb = [0.0] * 4
            emb[i % 4] = 1.0
            memory.record_proof([emb])

        assert memory.pattern_count <= 5
        assert memory.proof_count == 10
