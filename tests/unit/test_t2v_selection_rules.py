"""Regression tests for Tree2Vec selection rule integration.

Covers:
- T2V SelectionRule present when tree2vec_weight > 0
- T2V annotations ("T2V") appear as selection type in ratio cycle
- tree2vec_include_position / tree2vec_include_depth don't break selection
- ML selection path (MLSelection) NOT triggered by tree2vec-only flags
- PrioritySOS T2V fallback to pop_first() when no FORTE scores
- Embeddings are non-None with position/depth flags after training
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions
from pyladr.search.priority_sos import PrioritySOS
from pyladr.search.selection import (
    GivenSelection,
    SelectionOrder,
    SelectionRule,
)
from pyladr.search.state import ClauseList


# ── Helpers ──────────────────────────────────────────────────────────────

SYM_P, SYM_A, SYM_B, SYM_F = 1, 2, 3, 10


def _const(symnum: int) -> Term:
    return get_rigid_term(symnum, 0)


def _func(symnum: int, *args: Term) -> Term:
    return get_rigid_term(symnum, len(args), args)


def _make_weighted_clause(weight: float, clause_id: int) -> Clause:
    c = Clause(
        literals=(Literal(sign=True, atom=_const(SYM_A)),),
        id=clause_id,
        justification=(Justification(just_type=JustType.INPUT),),
    )
    c.weight = weight
    return c


# ── T2V Rule Presence ────────────────────────────────────────────────────


class TestT2VRulePresence:
    """Verify T2V SelectionRule is created when tree2vec_weight > 0."""

    def test_t2v_rule_added_with_positive_weight(self) -> None:
        """tree2vec_weight > 0 produces a T2V rule in GivenSelection."""
        opts = SearchOptions(tree2vec_weight=2)
        search = GivenClauseSearch(options=opts)
        rules = search._selection.rules
        t2v_rules = [r for r in rules if r.name == "T2V"]
        assert len(t2v_rules) == 1
        assert t2v_rules[0].order == SelectionOrder.TREE2VEC

    def test_t2v_rule_part_matches_weight(self) -> None:
        """T2V rule part equals the configured tree2vec_weight."""
        opts = SearchOptions(tree2vec_weight=3)
        search = GivenClauseSearch(options=opts)
        t2v_rule = [r for r in search._selection.rules if r.name == "T2V"][0]
        assert t2v_rule.part == 3

    def test_no_t2v_rule_when_weight_zero(self) -> None:
        """Default tree2vec_weight=0 produces no T2V rule."""
        opts = SearchOptions()
        search = GivenClauseSearch(options=opts)
        t2v_rules = [r for r in search._selection.rules if r.name == "T2V"]
        assert len(t2v_rules) == 0

    def test_t2v_alongside_default_rules(self) -> None:
        """T2V rule coexists with default A and W rules."""
        opts = SearchOptions(tree2vec_weight=2)
        search = GivenClauseSearch(options=opts)
        names = [r.name for r in search._selection.rules]
        assert "A" in names
        assert "W" in names
        assert "T2V" in names

    def test_t2v_cycle_size_correct(self) -> None:
        """Cycle size includes T2V part: A(1) + W(4) + T2V(2) = 7."""
        opts = SearchOptions(tree2vec_weight=2)
        search = GivenClauseSearch(options=opts)
        assert search._selection._cycle_size == 7


# ── T2V Selection Annotations ────────────────────────────────────────────


class TestT2VSelectionAnnotations:
    """Verify T2V annotation appears in selection output."""

    def test_t2v_annotation_in_ratio_cycle(self) -> None:
        """T2V appears at correct positions in the ratio cycle."""
        gs = GivenSelection(rules=[
            SelectionRule("A", SelectionOrder.AGE, part=1),
            SelectionRule("W", SelectionOrder.WEIGHT, part=2),
            SelectionRule("T2V", SelectionOrder.TREE2VEC, part=1),
        ])
        sos = PrioritySOS("sos")
        for idx in range(8):
            c = _make_weighted_clause(5.0, idx + 1)
            sos.append(c)

        selections = []
        for idx in range(8):
            _, name = gs.select_given(sos, idx)
            selections.append(name)

        # Cycle: A(1), W(2), T2V(1) → total 4
        # Pattern: A, W, W, T2V, A, W, W, T2V
        assert selections == ["A", "W", "W", "T2V", "A", "W", "W", "T2V"]

    def test_t2v_only_rule(self) -> None:
        """T2V as the sole rule always returns 'T2V' annotation."""
        gs = GivenSelection(rules=[
            SelectionRule("T2V", SelectionOrder.TREE2VEC, part=1),
        ])
        sos = PrioritySOS("sos")
        for idx in range(3):
            c = _make_weighted_clause(5.0, idx + 1)
            sos.append(c)

        for _ in range(3):
            _, name = gs.select_given(sos, 0)
            assert name == "T2V"

    def test_t2v_selected_count_increments(self) -> None:
        """T2V rule's selected counter increments on each T2V pick."""
        gs = GivenSelection(rules=[
            SelectionRule("T2V", SelectionOrder.TREE2VEC, part=1),
        ])
        sos = PrioritySOS("sos")
        for idx in range(3):
            sos.append(_make_weighted_clause(5.0, idx + 1))

        t2v_rule = gs.rules[0]
        assert t2v_rule.selected == 0
        gs.select_given(sos, 0)
        assert t2v_rule.selected == 1
        gs.select_given(sos, 1)
        assert t2v_rule.selected == 2


# ── T2V Fallback Behavior ───────────────────────────────────────────────


class TestT2VFallbackBehavior:
    """Verify T2V selection falls back correctly without PrioritySOS."""

    def test_t2v_raises_without_priority_sos(self) -> None:
        """T2V on plain ClauseList raises ValueError (requires PrioritySOS)."""
        gs = GivenSelection(rules=[
            SelectionRule("T2V", SelectionOrder.TREE2VEC, part=1),
        ])
        sos = ClauseList("sos")
        c1 = _make_weighted_clause(10.0, 1)
        c2 = _make_weighted_clause(1.0, 2)
        sos.append(c1)
        sos.append(c2)

        with pytest.raises(ValueError, match="requires PrioritySOS"):
            gs.select_given(sos, 0)

    def test_t2v_priority_sos_fallback_to_pop_first(self) -> None:
        """T2V on PrioritySOS falls back to pop_first() when no FORTE scores."""
        psos = PrioritySOS("sos")
        c1 = _make_weighted_clause(10.0, 1)
        c2 = _make_weighted_clause(1.0, 2)
        psos.append(c1)
        psos.append(c2)

        gs = GivenSelection(rules=[
            SelectionRule("T2V", SelectionOrder.TREE2VEC, part=1),
        ])
        selected, name = gs.select_given(psos, 0)
        # Without FORTE scores, pop_best_forte() returns None, T2V falls back to pop_first()
        assert selected is not None
        assert name == "T2V"


# ── Position/Depth Flags Don't Break Selection ───────────────────────────


class TestPositionDepthFlags:
    """Verify position/depth flags don't interfere with selection logic."""

    def test_position_flag_creates_t2v_rule(self) -> None:
        """tree2vec_include_position=True with tree2vec_weight>0 still creates T2V rule."""
        opts = SearchOptions(
            tree2vec_weight=2,
            tree2vec_include_position=True,
        )
        search = GivenClauseSearch(options=opts)
        t2v_rules = [r for r in search._selection.rules if r.name == "T2V"]
        assert len(t2v_rules) == 1

    def test_depth_flag_creates_t2v_rule(self) -> None:
        """tree2vec_include_depth=True with tree2vec_weight>0 still creates T2V rule."""
        opts = SearchOptions(
            tree2vec_weight=2,
            tree2vec_include_depth=True,
        )
        search = GivenClauseSearch(options=opts)
        t2v_rules = [r for r in search._selection.rules if r.name == "T2V"]
        assert len(t2v_rules) == 1

    def test_both_flags_creates_t2v_rule(self) -> None:
        """Both position and depth flags with tree2vec_weight>0 still creates T2V rule."""
        opts = SearchOptions(
            tree2vec_weight=2,
            tree2vec_include_position=True,
            tree2vec_include_depth=True,
        )
        search = GivenClauseSearch(options=opts)
        t2v_rules = [r for r in search._selection.rules if r.name == "T2V"]
        assert len(t2v_rules) == 1
        assert t2v_rules[0].part == 2

    def test_position_depth_flags_without_weight_no_t2v_rule(self) -> None:
        """Position/depth flags alone (weight=0) don't create T2V rule."""
        opts = SearchOptions(
            tree2vec_include_position=True,
            tree2vec_include_depth=True,
        )
        search = GivenClauseSearch(options=opts)
        t2v_rules = [r for r in search._selection.rules if r.name == "T2V"]
        assert len(t2v_rules) == 0


# ── ML Selection Path Not Triggered ──────────────────────────────────────


class TestMLSelectionNotTriggered:
    """Verify tree2vec flags don't activate the MLSelection path."""

    def test_tree2vec_weight_uses_given_selection(self) -> None:
        """tree2vec_weight>0 alone uses GivenSelection, not MLSelection."""
        opts = SearchOptions(tree2vec_weight=2)
        search = GivenClauseSearch(options=opts)
        assert isinstance(search._selection, GivenSelection)

    def test_tree2vec_embeddings_flag_uses_given_selection(self) -> None:
        """tree2vec_embeddings=True alone uses default GivenSelection."""
        opts = SearchOptions(tree2vec_embeddings=True)
        search = GivenClauseSearch(options=opts)
        assert isinstance(search._selection, GivenSelection)

    def test_all_tree2vec_flags_no_ml_selection(self) -> None:
        """All T2V flags combined still use GivenSelection, not MLSelection."""
        opts = SearchOptions(
            tree2vec_weight=2,
            tree2vec_embeddings=True,
            tree2vec_include_position=True,
            tree2vec_include_depth=True,
            tree2vec_online_learning=True,
        )
        search = GivenClauseSearch(options=opts)
        assert isinstance(search._selection, GivenSelection)


# ── Embedding Non-None with Position/Depth ───────────────────────────────


class TestEmbeddingsWithPositionDepth:
    """Verify embeddings are non-None when position/depth flags match training."""

    def _train_and_embed(
        self, include_position: bool, include_depth: bool,
    ) -> list[float] | None:
        """Train Tree2Vec with flags and return embedding for a clause."""
        from pyladr.ml.tree2vec.algorithm import Tree2Vec, Tree2VecConfig
        from pyladr.ml.tree2vec.skipgram import SkipGramConfig
        from pyladr.ml.tree2vec.walks import WalkConfig

        x, y = get_variable_term(0), get_variable_term(1)
        clauses = [
            Clause(literals=(Literal(sign=True, atom=_func(SYM_P, _func(SYM_F, _const(SYM_A), _const(SYM_B)))),)),
            Clause(literals=(Literal(sign=True, atom=_func(SYM_P, _func(SYM_F, x, _const(SYM_A)))),)),
            Clause(literals=(Literal(sign=True, atom=_func(SYM_P, _func(SYM_F, x, y))),)),
        ]

        config = Tree2VecConfig(
            walk_config=WalkConfig(
                include_position=include_position,
                include_depth=include_depth,
            ),
            skipgram_config=SkipGramConfig(
                embedding_dim=32,
                num_epochs=5,
                seed=42,
            ),
        )
        t2v = Tree2Vec(config)
        t2v.train(clauses)
        return t2v.embed_clause(clauses[0])

    def test_embeddings_non_none_with_position(self) -> None:
        """Embeddings non-None when trained and queried with include_position=True."""
        emb = self._train_and_embed(include_position=True, include_depth=False)
        assert emb is not None
        assert len(emb) > 0

    def test_embeddings_non_none_with_depth(self) -> None:
        """Embeddings non-None when trained and queried with include_depth=True."""
        emb = self._train_and_embed(include_position=False, include_depth=True)
        assert emb is not None
        assert len(emb) > 0

    def test_embeddings_non_none_with_both(self) -> None:
        """Embeddings non-None with both position and depth flags."""
        emb = self._train_and_embed(include_position=True, include_depth=True)
        assert emb is not None
        assert len(emb) > 0

    def test_embeddings_non_none_without_flags(self) -> None:
        """Baseline: embeddings non-None without position/depth flags."""
        emb = self._train_and_embed(include_position=False, include_depth=False)
        assert emb is not None
        assert len(emb) > 0
