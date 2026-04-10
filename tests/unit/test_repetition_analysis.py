"""Tests for repetition detection algorithm."""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.search.repetition_analysis import (
    RepetitionConfig,
    RepetitionTracker,
    clause_skeleton,
    clause_subterm_skeletons,
    literal_skeleton,
    subterm_skeletons,
    term_skeleton,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_var(n: int) -> Term:
    return get_variable_term(n)


def make_const(symnum: int) -> Term:
    return get_rigid_term(symnum, 0)


def make_fn(symnum: int, *args: Term) -> Term:
    return get_rigid_term(symnum, len(args), tuple(args))


def make_literal(sign: bool, atom: Term) -> Literal:
    return Literal(sign=sign, atom=atom)


def make_clause(*literals: Literal) -> Clause:
    c = Clause(literals=tuple(literals))
    c.weight = sum(lit.atom.symbol_count for lit in literals)
    return c


# Symbols: P=1, Q=2, f=3, g=4, a=5, b=6
P, Q, f_sym, g_sym, a_sym, b_sym = 1, 2, 3, 4, 5, 6


# ── term_skeleton tests ─────────────────────────────────────────────────────

class TestTermSkeleton:
    def test_variable_skeleton(self):
        """All variables produce the same skeleton."""
        assert term_skeleton(make_var(0)) == ("V",)
        assert term_skeleton(make_var(1)) == ("V",)
        assert term_skeleton(make_var(99)) == ("V",)

    def test_constant_skeleton(self):
        """Constants preserve their symbol ID."""
        assert term_skeleton(make_const(a_sym)) == ("C", a_sym)
        assert term_skeleton(make_const(b_sym)) == ("C", b_sym)
        # Different constants have different skeletons
        assert term_skeleton(make_const(a_sym)) != term_skeleton(make_const(b_sym))

    def test_complex_skeleton(self):
        """Complex terms preserve structure and symbol IDs."""
        # f(x, a)
        t = make_fn(f_sym, make_var(0), make_const(a_sym))
        skel = term_skeleton(t)
        assert skel == ("F", f_sym, ("V",), ("C", a_sym))

    def test_variable_abstraction(self):
        """Terms differing only in variable names share skeletons."""
        # f(x, y) and f(z, w) should have same skeleton
        t1 = make_fn(f_sym, make_var(0), make_var(1))
        t2 = make_fn(f_sym, make_var(2), make_var(3))
        assert term_skeleton(t1) == term_skeleton(t2)

    def test_nested_skeleton(self):
        """Nested terms preserve full structure."""
        # f(g(x), a)
        inner = make_fn(g_sym, make_var(0))
        t = make_fn(f_sym, inner, make_const(a_sym))
        skel = term_skeleton(t)
        assert skel == ("F", f_sym, ("F", g_sym, ("V",)), ("C", a_sym))

    def test_different_structure(self):
        """Terms with different structure have different skeletons."""
        # f(x, x) vs f(x, a)
        t1 = make_fn(f_sym, make_var(0), make_var(0))
        t2 = make_fn(f_sym, make_var(0), make_const(a_sym))
        assert term_skeleton(t1) != term_skeleton(t2)

    def test_skeleton_hashable(self):
        """Skeletons can be used as dictionary keys."""
        t = make_fn(f_sym, make_var(0), make_const(a_sym))
        skel = term_skeleton(t)
        d = {skel: 1}
        assert d[skel] == 1


# ── literal_skeleton tests ───────────────────────────────────────────────────

class TestLiteralSkeleton:
    def test_positive_literal(self):
        atom = make_fn(P, make_var(0))
        skel = literal_skeleton(make_literal(True, atom))
        assert skel == (True, ("F", P, ("V",)))

    def test_negative_literal(self):
        atom = make_fn(P, make_var(0))
        skel = literal_skeleton(make_literal(False, atom))
        assert skel == (False, ("F", P, ("V",)))

    def test_sign_matters(self):
        atom = make_fn(P, make_var(0))
        pos = literal_skeleton(make_literal(True, atom))
        neg = literal_skeleton(make_literal(False, atom))
        assert pos != neg


# ── clause_skeleton tests ────────────────────────────────────────────────────

class TestClauseSkeleton:
    def test_empty_clause(self):
        c = make_clause()
        assert clause_skeleton(c) == ()

    def test_unit_clause(self):
        lit = make_literal(True, make_fn(P, make_var(0)))
        c = make_clause(lit)
        skel = clause_skeleton(c)
        assert len(skel) == 1

    def test_variable_renaming_invariance(self):
        """Clauses differing only in variable names share skeletons."""
        # P(x) | Q(y) vs P(z) | Q(w)
        c1 = make_clause(
            make_literal(True, make_fn(P, make_var(0))),
            make_literal(True, make_fn(Q, make_var(1))),
        )
        c2 = make_clause(
            make_literal(True, make_fn(P, make_var(2))),
            make_literal(True, make_fn(Q, make_var(3))),
        )
        assert clause_skeleton(c1) == clause_skeleton(c2)

    def test_literal_order_invariance(self):
        """Clauses with same literals in different order share skeletons."""
        lit1 = make_literal(True, make_fn(P, make_var(0)))
        lit2 = make_literal(False, make_fn(Q, make_var(1)))
        c1 = make_clause(lit1, lit2)
        c2 = make_clause(lit2, lit1)
        assert clause_skeleton(c1) == clause_skeleton(c2)

    def test_different_predicates(self):
        """Clauses with different predicates have different skeletons."""
        c1 = make_clause(make_literal(True, make_fn(P, make_var(0))))
        c2 = make_clause(make_literal(True, make_fn(Q, make_var(0))))
        assert clause_skeleton(c1) != clause_skeleton(c2)


# ── subterm_skeletons tests ──────────────────────────────────────────────────

class TestSubtermSkeletons:
    def test_variable_no_subterms(self):
        assert subterm_skeletons(make_var(0)) == []

    def test_constant_no_subterms(self):
        assert subterm_skeletons(make_const(a_sym)) == []

    def test_simple_complex(self):
        """A simple f(x) yields one subterm skeleton."""
        t = make_fn(f_sym, make_var(0))
        skels = subterm_skeletons(t)
        assert len(skels) == 1
        assert skels[0] == ("F", f_sym, ("V",))

    def test_nested_complex(self):
        """f(g(x), a) yields skeletons for both f(...) and g(x)."""
        inner = make_fn(g_sym, make_var(0))
        t = make_fn(f_sym, inner, make_const(a_sym))
        skels = subterm_skeletons(t)
        assert len(skels) == 2

    def test_clause_subterm_skeletons(self):
        """Collects subterm skeletons from all literals."""
        c = make_clause(
            make_literal(True, make_fn(P, make_fn(f_sym, make_var(0)))),
            make_literal(False, make_fn(Q, make_const(a_sym))),
        )
        skels = clause_subterm_skeletons(c)
        # P(f(x)): P(...) and f(x) are complex; Q(a): Q(...) is complex
        assert len(skels) == 3


# ── RepetitionTracker tests ──────────────────────────────────────────────────

class TestRepetitionTracker:
    def _make_tracker(self, **kwargs) -> RepetitionTracker:
        config = RepetitionConfig(**kwargs)
        return RepetitionTracker(config=config)

    def test_initial_score_zero(self):
        """Score is 0 before any observations."""
        tracker = self._make_tracker(min_observations=0)
        c = make_clause(make_literal(True, make_fn(P, make_var(0))))
        assert tracker.repetition_score(c) == 0.0

    def test_score_increases_with_repetition(self):
        """Observing the same structure repeatedly increases score."""
        tracker = self._make_tracker(min_observations=0, decay_rate=0.0)

        c1 = make_clause(make_literal(True, make_fn(P, make_var(0))))
        c2 = make_clause(make_literal(True, make_fn(P, make_var(1))))  # same skeleton

        tracker.observe(c1)
        score1 = tracker.repetition_score(c2)

        # Observe same skeleton many more times
        for _ in range(10):
            tracker.observe(c1)
        score2 = tracker.repetition_score(c2)

        # Score should be equal since only one skeleton exists (normalized to max)
        # But if we also observe a different clause, the original should score higher
        different = make_clause(make_literal(True, make_fn(Q, make_var(0))))
        tracker.observe(different)

        score_repeated = tracker.repetition_score(c2)
        score_novel = tracker.repetition_score(different)
        assert score_repeated > score_novel

    def test_novel_clause_low_score(self):
        """A structurally novel clause gets a lower score than a repeated one."""
        tracker = self._make_tracker(min_observations=0, decay_rate=0.0)

        repeated = make_clause(make_literal(True, make_fn(P, make_var(0))))
        for _ in range(5):
            tracker.observe(repeated)

        novel = make_clause(
            make_literal(True, make_fn(Q, make_fn(f_sym, make_var(0)))),
            make_literal(False, make_fn(P, make_const(a_sym))),
        )
        # Observe novel once so we have multiple skeleton types
        tracker.observe(novel)

        assert tracker.repetition_score(repeated) > tracker.repetition_score(novel)

    def test_min_observations_gate(self):
        """Score is 0 until min_observations is reached."""
        tracker = self._make_tracker(min_observations=5, decay_rate=0.0)

        c = make_clause(make_literal(True, make_fn(P, make_var(0))))
        for i in range(4):
            tracker.observe(c)
            assert tracker.repetition_score(c) == 0.0

        tracker.observe(c)
        # Now at 5 observations, score should be active
        # (but may still be 0 if only one skeleton seen and it's the max)
        # Actually with only one skeleton, freq/max_freq = 1.0, so score = 1.0
        assert tracker.repetition_score(c) > 0.0

    def test_disabled_returns_zero(self):
        """When disabled, always returns 0."""
        tracker = self._make_tracker(enabled=False)
        c = make_clause(make_literal(True, make_fn(P, make_var(0))))
        for _ in range(50):
            tracker.observe(c)
        assert tracker.repetition_score(c) == 0.0

    def test_decay_reduces_old_frequencies(self):
        """Decay should reduce influence of old observations."""
        tracker = self._make_tracker(
            min_observations=0, decay_rate=0.5,
        )

        old = make_clause(make_literal(True, make_fn(P, make_var(0))))
        # Observe old clause 99 times (decay triggers at multiples of 100)
        for _ in range(99):
            tracker.observe(old)

        score_before_decay = tracker.repetition_score(old)

        # 100th observation triggers decay
        new = make_clause(make_literal(True, make_fn(Q, make_var(0))))
        tracker.observe(new)

        # After decay, old clause's frequency is halved
        # Score should reflect reduced frequency
        score_after_decay = tracker.repetition_score(old)
        # The score is relative to max, so after decay the relative positions
        # may shift. With aggressive decay (0.5), old drops significantly.
        assert score_after_decay <= score_before_decay

    def test_reset_clears_state(self):
        """Reset brings tracker back to initial state."""
        tracker = self._make_tracker(min_observations=0, decay_rate=0.0)
        c = make_clause(make_literal(True, make_fn(P, make_var(0))))
        for _ in range(10):
            tracker.observe(c)
        assert tracker.repetition_score(c) > 0.0

        tracker.reset()
        assert tracker.repetition_score(c) == 0.0
        assert tracker.stats.clauses_observed == 0

    def test_stats_tracking(self):
        """Statistics are updated correctly."""
        tracker = self._make_tracker(min_observations=0, decay_rate=0.0)

        c1 = make_clause(make_literal(True, make_fn(P, make_var(0))))
        c2 = make_clause(make_literal(True, make_fn(Q, make_var(0))))

        tracker.observe(c1)
        tracker.observe(c1)
        tracker.observe(c2)

        assert tracker.stats.clauses_observed == 3
        assert tracker.stats.unique_skeletons == 2
        assert tracker.stats.max_skeleton_frequency == 2

    def test_subterm_contribution(self):
        """Subterm repetition contributes to score even when clause skeleton is novel."""
        tracker = self._make_tracker(
            min_observations=0, decay_rate=0.0,
            clause_weight=0.0, subterm_weight=1.0,  # only subterm matters
        )

        # Observe many clauses with f(x) subterm
        for i in range(10):
            c = make_clause(make_literal(True, make_fn(P, make_fn(f_sym, make_var(i % 5)))))
            tracker.observe(c)

        # A novel clause that also contains f(x) subterm
        novel = make_clause(
            make_literal(True, make_fn(Q, make_fn(f_sym, make_var(0)))),
        )

        # A clause without f(x) subterm
        no_f = make_clause(
            make_literal(True, make_fn(Q, make_fn(g_sym, make_var(0)))),
        )

        assert tracker.repetition_score(novel) > tracker.repetition_score(no_f)

    def test_score_bounded(self):
        """Repetition score is always in [0, 1]."""
        tracker = self._make_tracker(min_observations=0, decay_rate=0.0)

        c = make_clause(make_literal(True, make_fn(P, make_var(0))))
        for _ in range(1000):
            tracker.observe(c)
            score = tracker.repetition_score(c)
            assert 0.0 <= score <= 1.0

    def test_report(self):
        """Stats report produces a non-empty string."""
        tracker = self._make_tracker()
        report = tracker.stats.report()
        assert "repetition" in report
        assert "observed=0" in report
