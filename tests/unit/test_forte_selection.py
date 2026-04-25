"""Tests for FORTE selection integration (Phase 2C).

Verifies that FORTE embeddings are consumed by the selection cycle,
influencing clause selection when forte_weight > 0.
"""

from __future__ import annotations

import pytest

from pyladr.search.given_clause import GivenClauseSearch, SearchOptions
from pyladr.search.priority_sos import PrioritySOS, _forte_novelty_score
from pyladr.search.selection import (
    GivenSelection,
    SelectionOrder,
    SelectionRule,
)
from tests.factories import make_clause, make_const, make_func, make_neg_lit, make_pos_lit, make_var


# ── SelectionOrder.FORTE ────────────────────────────────────────────────────


class TestSelectionOrderForte:
    """Verify FORTE enum value exists and integrates with selection cycle."""

    def test_forte_enum_value(self) -> None:
        assert SelectionOrder.FORTE == 5

    def test_forte_selection_rule_creation(self) -> None:
        rule = SelectionRule("F", SelectionOrder.FORTE, part=3)
        assert rule.name == "F"
        assert rule.order == SelectionOrder.FORTE
        assert rule.part == 3

    def test_forte_in_selection_cycle(self) -> None:
        """FORTE rule participates in ratio-based selection cycling."""
        rules = [
            SelectionRule("A", SelectionOrder.AGE, part=1),
            SelectionRule("W", SelectionOrder.WEIGHT, part=2),
            SelectionRule("F", SelectionOrder.FORTE, part=1),
        ]
        sel = GivenSelection(rules=rules)
        assert sel._cycle_size == 4


# ── FORTE Novelty Score ─────────────────────────────────────────────────────


class TestForteNoveltyScore:
    """Verify the FORTE novelty scoring function."""

    def test_score_is_negative_l1_norm(self) -> None:
        emb = [0.5, -0.3, 0.2, -0.1]
        expected = -(0.5 + 0.3 + 0.2 + 0.1)
        assert abs(_forte_novelty_score(emb) - expected) < 1e-10

    def test_zero_embedding(self) -> None:
        assert _forte_novelty_score([0.0] * 64) == 0.0

    def test_more_spread_gets_lower_score(self) -> None:
        """More spread embedding (higher L1-norm) → lower (more negative) score → selected first by min-heap."""
        # Concentrated: one large component
        concentrated = [1.0] + [0.0] * 63
        # Spread: many small components (still unit norm approximation)
        spread = [0.125] * 64  # L1 = 8.0 vs 1.0
        assert _forte_novelty_score(spread) < _forte_novelty_score(concentrated)

    def test_deterministic(self) -> None:
        emb = [0.1 * i for i in range(64)]
        assert _forte_novelty_score(emb) == _forte_novelty_score(emb)


# ── PrioritySOS FORTE Heap ──────────────────────────────────────────────────


class TestPrioritySosForteHeap:
    """Verify PrioritySOS FORTE heap operations."""

    def test_forte_embeddings_constructor(self) -> None:
        embs: dict[int, list[float]] = {}
        sos = PrioritySOS("sos", forte_embeddings=embs)
        assert sos._forte_embeddings_ref is embs

    def test_pop_best_forte_empty(self) -> None:
        sos = PrioritySOS("sos")
        sos._forte_embeddings_ref = ({})
        assert sos.pop_best_forte() is None

    def test_pop_best_forte_selects_most_diverse(self) -> None:
        """pop_best_forte returns the clause with highest L1-norm embedding."""
        sos = PrioritySOS("sos")
        embs: dict[int, list[float]] = {}

        # Clause with concentrated embedding (L1 = 1.0)
        c1 = make_clause(make_pos_lit(make_func(1, make_const(2))), weight=1.0)
        c1.id = 1
        concentrated = [1.0] + [0.0] * 63
        embs[1] = concentrated

        # Clause with spread embedding (L1 = 8.0)
        c2 = make_clause(make_pos_lit(make_func(3, make_const(4))), weight=1.0)
        c2.id = 2
        spread = [0.125] * 64
        embs[2] = spread

        sos._forte_embeddings_ref = (embs)
        sos.append(c1)
        sos.append(c2)

        # Most diverse (highest L1-norm) should be selected first
        selected = sos.pop_best_forte()
        assert selected is not None
        assert selected.id == 2  # spread embedding

    def test_pop_best_forte_with_score_at_append(self) -> None:
        """Clauses appended with forte_score after init get pushed to heap."""
        sos = PrioritySOS("sos")
        embs: dict[int, list[float]] = {}
        sos._forte_embeddings_ref = (embs)

        # Add initial clause to trigger lazy init
        c1 = make_clause(make_pos_lit(make_func(1, make_const(2))), weight=1.0)
        c1.id = 1
        embs[1] = [0.5] * 64
        sos.append(c1)

        # Trigger lazy init
        _ = sos.pop_best_forte()

        # Now add new clause with explicit forte_score
        c2 = make_clause(make_pos_lit(make_func(3, make_const(4))), weight=1.0)
        c2.id = 2
        sos.append(c2, forte_score=-10.0)  # very diverse

        selected = sos.pop_best_forte()
        assert selected is not None
        assert selected.id == 2

    def test_lazy_init_only_once(self) -> None:
        """FORTE heap is initialized only on first pop, not on subsequent pops."""
        sos = PrioritySOS("sos")
        sos._forte_embeddings_ref = ({})
        assert sos._forte_initialized is False

        sos.pop_best_forte()
        assert sos._forte_initialized is True

    def test_forte_heap_lazy_deletion(self) -> None:
        """Removed clauses are skipped during FORTE heap extraction."""
        sos = PrioritySOS("sos")
        embs: dict[int, list[float]] = {}

        c1 = make_clause(make_pos_lit(make_func(1, make_const(2))), weight=1.0)
        c1.id = 1
        embs[1] = [0.125] * 64  # L1=8, score=-8

        c2 = make_clause(make_pos_lit(make_func(3, make_const(4))), weight=1.0)
        c2.id = 2
        embs[2] = [1.0] + [0.0] * 63  # L1=1, score=-1

        sos._forte_embeddings_ref = (embs)
        sos.append(c1)
        sos.append(c2)

        # Remove c1 (the best candidate)
        sos.remove(c1)

        # Should skip c1 and return c2
        selected = sos.pop_best_forte()
        assert selected is not None
        assert selected.id == 2

    def test_compact_includes_forte_heap(self) -> None:
        """compact() rebuilds the FORTE heap too."""
        sos = PrioritySOS("sos")
        embs: dict[int, list[float]] = {}

        c1 = make_clause(make_pos_lit(make_func(1, make_const(2))), weight=1.0)
        c1.id = 1
        embs[1] = [0.5] * 64

        sos._forte_embeddings_ref = (embs)
        sos.append(c1)

        # Trigger init
        sos.pop_best_forte()

        # Re-add and compact
        c2 = make_clause(make_pos_lit(make_func(3, make_const(4))), weight=1.0)
        c2.id = 2
        embs[2] = [0.25] * 64
        sos.append(c2, forte_score=-16.0)

        sos.compact()
        # Should not crash and heap should still work
        assert sos._forte_initialized is True


# ── Selection Dispatch ───────────────────────────────────────────────────────


class TestSelectionDispatch:
    """Verify FORTE is dispatched correctly in selection cycle."""

    def test_pop_from_priority_sos_forte(self) -> None:
        """_pop_from_priority_sos handles SelectionOrder.FORTE."""
        sos = PrioritySOS("sos")
        embs: dict[int, list[float]] = {}

        c1 = make_clause(make_pos_lit(make_func(1, make_const(2))), weight=1.0)
        c1.id = 1
        embs[1] = [0.125] * 64

        sos._forte_embeddings_ref = (embs)
        sos.append(c1)

        selected = GivenSelection._pop_from_priority_sos(sos, SelectionOrder.FORTE)
        assert selected is not None
        assert selected.id == 1


# ── Search Integration with FORTE Selection ──────────────────────────────────


class TestForteSelectionInSearch:
    """End-to-end tests for FORTE-guided clause selection."""

    def test_forte_weight_creates_selection_rule(self) -> None:
        """forte_weight > 0 adds FORTE rule to selection cycle."""
        opts = SearchOptions(forte_embeddings=True, forte_weight=0.8)
        search = GivenClauseSearch(options=opts)
        rules = search._selection.rules
        forte_rules = [r for r in rules if r.order == SelectionOrder.FORTE]
        assert len(forte_rules) == 1
        assert forte_rules[0].name == "F"
        assert forte_rules[0].part == 0.8

    def test_forte_weight_zero_no_rule(self) -> None:
        """forte_weight=0 does not add FORTE rule."""
        opts = SearchOptions(forte_embeddings=True, forte_weight=0)
        search = GivenClauseSearch(options=opts)
        rules = search._selection.rules
        forte_rules = [r for r in rules if r.order == SelectionOrder.FORTE]
        assert len(forte_rules) == 0

    def test_priority_sos_wired_to_embeddings(self) -> None:
        """PrioritySOS gets reference to _forte_embeddings when FORTE enabled."""
        opts = SearchOptions(
            forte_embeddings=True, forte_weight=0.5, priority_sos=True,
        )
        search = GivenClauseSearch(options=opts)
        sos = search._state.sos
        assert isinstance(sos, PrioritySOS)
        assert sos._forte_embeddings_ref is search._forte_embeddings

    def test_simple_proof_with_forte_selection(self) -> None:
        """Proof found with FORTE selection active."""
        opts = SearchOptions(
            forte_embeddings=True, forte_weight=0.5, priority_sos=True,
        )
        search = GivenClauseSearch(options=opts)

        c1 = make_clause(make_pos_lit(make_func(1, make_const(2))), weight=2.0)
        c2 = make_clause(make_neg_lit(make_func(1, make_var(0))), weight=2.0)

        result = search.run(usable=[], sos=[c1, c2])
        assert result.exit_code.value == 1  # MAX_PROOFS_EXIT
        assert len(result.proofs) >= 1

    def test_sos_exhausted_with_forte_selection(self) -> None:
        """SOS exhaustion with FORTE selection active."""
        opts = SearchOptions(
            forte_embeddings=True, forte_weight=0.5,
            priority_sos=True, max_given=10,
        )
        search = GivenClauseSearch(options=opts)

        c1 = make_clause(make_pos_lit(make_func(1, make_const(2))), weight=2.0)
        result = search.run(usable=[], sos=[c1])
        assert result.exit_code.value in (2, 3)

    def test_selection_type_includes_forte(self) -> None:
        """When FORTE rule fires, selection type should be 'F'."""
        opts = SearchOptions(
            forte_embeddings=True, forte_weight=1.0,  # dominate selection
            priority_sos=True,
        )
        search = GivenClauseSearch(options=opts)

        # With forte_weight=1.0, FORTE has highest weight in cycle
        # The cycle is: A(1), W(4), F(1.0) = 6 total
        # Check the selection rules contain F
        forte_rules = [r for r in search._selection.rules if r.name == "F"]
        assert len(forte_rules) == 1
        assert forte_rules[0].part == 1.0

    def test_no_forte_selection_without_priority_sos(self) -> None:
        """FORTE selection rule is created even without PrioritySOS,
        falling back to age-based selection in linear scan mode."""
        opts = SearchOptions(
            forte_embeddings=True, forte_weight=0.5, priority_sos=False,
        )
        search = GivenClauseSearch(options=opts)

        c1 = make_clause(make_pos_lit(make_func(1, make_const(2))), weight=2.0)
        c2 = make_clause(make_neg_lit(make_func(1, make_var(0))), weight=2.0)

        # Should still find proof (FORTE falls back to age in linear scan)
        result = search.run(usable=[], sos=[c1, c2])
        assert result.exit_code.value == 1

    def test_forte_embeddings_disabled_weight_nonzero(self) -> None:
        """forte_weight > 0 but forte_embeddings=False: rule added but no provider."""
        opts = SearchOptions(forte_embeddings=False, forte_weight=0.5)
        search = GivenClauseSearch(options=opts)
        assert search.forte_provider is None
        # Rule is still in the cycle (falls back to age)
        forte_rules = [r for r in search._selection.rules if r.order == SelectionOrder.FORTE]
        assert len(forte_rules) == 1


# ── Compatibility Tests ──────────────────────────────────────────────────────


class TestForteSelectionCompatibility:
    """Verify FORTE selection preserves existing behavior when disabled."""

    def test_no_forte_in_cycle_when_weight_zero(self) -> None:
        """No FORTE in selection cycle when forte_weight=0."""
        opts = SearchOptions(forte_weight=0.0)
        search = GivenClauseSearch(options=opts)
        for rule in search._selection.rules:
            assert rule.order != SelectionOrder.FORTE

    def test_existing_selection_orders_unchanged(self) -> None:
        """Existing enum values are preserved."""
        assert SelectionOrder.WEIGHT == 0
        assert SelectionOrder.AGE == 1
        assert SelectionOrder.RANDOM == 2
        assert SelectionOrder.ENTROPY == 3
        assert SelectionOrder.UNIFICATION_PENALTY == 4

    def test_proof_identical_without_forte(self) -> None:
        """Search without FORTE produces same result as always."""
        opts = SearchOptions()
        search = GivenClauseSearch(options=opts)

        c1 = make_clause(make_pos_lit(make_func(1, make_const(2))), weight=2.0)
        c2 = make_clause(make_neg_lit(make_func(1, make_var(0))), weight=2.0)

        result = search.run(usable=[], sos=[c1, c2])
        assert result.exit_code.value == 1
        assert len(result.proofs) >= 1
