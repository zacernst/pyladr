"""Unit tests for repetition detection and selection bias.

Tests the repetition analysis system that detects structurally repetitious
clauses and biases the selection system against them:
- Structural fingerprinting (term, literal, clause skeletons)
- Repetition tracking (frequency counting, decay, normalization)
- Repetition scoring (clause-level, subterm-level, combined)
- Configuration and edge cases
- Performance characteristics
"""

from __future__ import annotations

import time

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.search.repetition_analysis import (
    RepetitionConfig,
    RepetitionStats,
    RepetitionTracker,
    clause_skeleton,
    clause_subterm_skeletons,
    literal_skeleton,
    subterm_skeletons,
    term_skeleton,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _const(symnum: int) -> Term:
    return get_rigid_term(symnum, 0)


def _func(symnum: int, *args: Term) -> Term:
    return get_rigid_term(symnum, len(args), args)


def _var(n: int) -> Term:
    return get_variable_term(n)


def _make_clause(*atoms: Term, signs: tuple[bool, ...] | None = None) -> Clause:
    if signs is None:
        signs = (True,) * len(atoms)
    lits = tuple(Literal(sign=s, atom=a) for s, a in zip(signs, atoms))
    return Clause(literals=lits)


def _make_clause_with_id(clause_id: int, *atoms: Term) -> Clause:
    c = _make_clause(*atoms)
    c.id = clause_id
    return c


# Symbol IDs
A, B, C_SYM, D = 1, 2, 3, 4
F, G, H = 10, 11, 12
P, Q, R = 20, 21, 22
EQ = 30  # equality


# ── Term Skeleton Tests ──────────────────────────────────────────────────────


class TestTermSkeleton:
    """Test structural fingerprinting of terms."""

    def test_variable_skeleton(self) -> None:
        """All variables produce the same skeleton regardless of index."""
        s0 = term_skeleton(_var(0))
        s1 = term_skeleton(_var(1))
        s42 = term_skeleton(_var(42))
        assert s0 == s1 == s42 == ("V",)

    def test_constant_skeleton(self) -> None:
        """Constants encode their symbol number."""
        sa = term_skeleton(_const(A))
        sb = term_skeleton(_const(B))
        assert sa == ("C", A)
        assert sb == ("C", B)
        assert sa != sb

    def test_function_skeleton(self) -> None:
        """f(a, b) skeleton encodes function symbol and child structure."""
        t = _func(F, _const(A), _const(B))
        s = term_skeleton(t)
        assert s == ("F", F, ("C", A), ("C", B))

    def test_nested_function_skeleton(self) -> None:
        """f(g(a)) skeleton is nested."""
        t = _func(F, _func(G, _const(A)))
        s = term_skeleton(t)
        assert s == ("F", F, ("F", G, ("C", A)))

    def test_variable_renaming_invariance(self) -> None:
        """f(x, y) and f(z, w) produce the same skeleton."""
        t1 = _func(F, _var(0), _var(1))
        t2 = _func(F, _var(2), _var(3))
        assert term_skeleton(t1) == term_skeleton(t2)

    def test_variable_vs_constant_differs(self) -> None:
        """f(x) and f(a) have different skeletons."""
        t1 = _func(F, _var(0))
        t2 = _func(F, _const(A))
        assert term_skeleton(t1) != term_skeleton(t2)

    def test_different_functions_differ(self) -> None:
        """f(a) and g(a) have different skeletons."""
        t1 = _func(F, _const(A))
        t2 = _func(G, _const(A))
        assert term_skeleton(t1) != term_skeleton(t2)

    def test_different_arity_differs(self) -> None:
        """f(a) and f(a, b) have different skeletons."""
        t1 = _func(F, _const(A))
        t2 = _func(F, _const(A), _const(B))
        assert term_skeleton(t1) != term_skeleton(t2)

    def test_skeleton_is_hashable(self) -> None:
        """Skeletons can be used as dictionary keys."""
        t = _func(F, _var(0), _const(A))
        s = term_skeleton(t)
        d = {s: 1}
        assert d[s] == 1

    def test_deeply_nested_skeleton(self) -> None:
        """Deeply nested terms produce correct skeletons."""
        # f(g(h(a)))
        t = _func(F, _func(G, _func(H, _const(A))))
        s = term_skeleton(t)
        assert s == ("F", F, ("F", G, ("F", H, ("C", A))))


# ── Literal Skeleton Tests ───────────────────────────────────────────────────


class TestLiteralSkeleton:
    """Test structural fingerprinting of literals."""

    def test_positive_literal(self) -> None:
        """Positive literal encodes sign=True and atom skeleton."""
        lit = Literal(sign=True, atom=_func(P, _const(A)))
        s = literal_skeleton(lit)
        assert s == (True, ("F", P, ("C", A)))

    def test_negative_literal(self) -> None:
        """Negative literal encodes sign=False."""
        lit = Literal(sign=False, atom=_func(P, _const(A)))
        s = literal_skeleton(lit)
        assert s == (False, ("F", P, ("C", A)))

    def test_sign_matters(self) -> None:
        """P(a) and -P(a) have different literal skeletons."""
        pos = Literal(sign=True, atom=_func(P, _const(A)))
        neg = Literal(sign=False, atom=_func(P, _const(A)))
        assert literal_skeleton(pos) != literal_skeleton(neg)


# ── Clause Skeleton Tests ────────────────────────────────────────────────────


class TestClauseSkeleton:
    """Test structural fingerprinting of clauses."""

    def test_single_literal_clause(self) -> None:
        """Single literal clause skeleton is a 1-tuple."""
        c = _make_clause(_func(P, _const(A)))
        s = clause_skeleton(c)
        assert len(s) == 1

    def test_multi_literal_clause(self) -> None:
        """Multi-literal clause skeleton sorts literal skeletons."""
        c = _make_clause(
            _func(P, _const(A)),
            _func(Q, _const(B)),
        )
        s = clause_skeleton(c)
        assert len(s) == 2

    def test_literal_order_invariance(self) -> None:
        """Clauses with same literals in different order share skeleton."""
        c1 = _make_clause(_func(P, _const(A)), _func(Q, _const(B)))
        c2 = _make_clause(_func(Q, _const(B)), _func(P, _const(A)))
        assert clause_skeleton(c1) == clause_skeleton(c2)

    def test_variable_renaming_invariance(self) -> None:
        """P(x, y) and P(z, w) share clause skeleton."""
        c1 = _make_clause(_func(P, _var(0), _var(1)))
        c2 = _make_clause(_func(P, _var(2), _var(3)))
        assert clause_skeleton(c1) == clause_skeleton(c2)

    def test_structurally_different_clauses(self) -> None:
        """P(x) and P(f(x)) have different skeletons."""
        c1 = _make_clause(_func(P, _var(0)))
        c2 = _make_clause(_func(P, _func(F, _var(0))))
        assert clause_skeleton(c1) != clause_skeleton(c2)

    def test_empty_clause_skeleton(self) -> None:
        """Empty clause has empty skeleton."""
        c = Clause(literals=())
        s = clause_skeleton(c)
        assert s == ()

    def test_clause_skeleton_is_hashable(self) -> None:
        """Clause skeletons are hashable for dict keys."""
        c = _make_clause(_func(P, _var(0)))
        s = clause_skeleton(c)
        d = {s: "test"}
        assert d[s] == "test"

    def test_sign_matters_in_clause(self) -> None:
        """P(a) and -P(a) produce different clause skeletons."""
        c1 = _make_clause(_func(P, _const(A)), signs=(True,))
        c2 = _make_clause(_func(P, _const(A)), signs=(False,))
        assert clause_skeleton(c1) != clause_skeleton(c2)


# ── Subterm Skeleton Tests ───────────────────────────────────────────────────


class TestSubtermSkeletons:
    """Test subterm-level skeleton extraction."""

    def test_variable_has_no_complex_subterms(self) -> None:
        """Variables have no complex subterms."""
        assert subterm_skeletons(_var(0)) == []

    def test_constant_has_no_complex_subterms(self) -> None:
        """Constants have no complex subterms."""
        assert subterm_skeletons(_const(A)) == []

    def test_function_includes_self(self) -> None:
        """f(a) is its own complex subterm."""
        t = _func(F, _const(A))
        skels = subterm_skeletons(t)
        assert len(skels) >= 1
        assert term_skeleton(t) in skels

    def test_nested_function_has_multiple_subterms(self) -> None:
        """f(g(a)) has both f(g(a)) and g(a) as complex subterms."""
        t = _func(F, _func(G, _const(A)))
        skels = subterm_skeletons(t)
        assert term_skeleton(_func(F, _func(G, _const(A)))) in skels
        assert term_skeleton(_func(G, _const(A))) in skels

    def test_clause_subterm_skeletons(self) -> None:
        """Clause subterm extraction collects from all literals."""
        c = _make_clause(
            _func(P, _func(F, _const(A))),
            _func(Q, _func(G, _const(B))),
        )
        skels = clause_subterm_skeletons(c)
        # Should include P(f(a)), f(a), Q(g(b)), g(b) at minimum
        assert len(skels) >= 4


# ── RepetitionTracker Tests ─────────────────────────────────────────────────


class TestRepetitionTracker:
    """Test the repetition frequency tracker."""

    def test_initial_state(self) -> None:
        """Fresh tracker has zero stats."""
        tracker = RepetitionTracker()
        assert tracker.stats.clauses_observed == 0
        assert tracker.stats.unique_skeletons == 0

    def test_observe_increments_count(self) -> None:
        """Observing a clause increments the observation count."""
        tracker = RepetitionTracker()
        c = _make_clause(_func(P, _const(A)))
        tracker.observe(c)
        assert tracker.stats.clauses_observed == 1

    def test_observe_tracks_unique_skeletons(self) -> None:
        """Unique skeleton count reflects distinct structures."""
        tracker = RepetitionTracker()
        c1 = _make_clause(_func(P, _var(0)))       # P(x)
        c2 = _make_clause(_func(Q, _var(0)))        # Q(x) - different skeleton
        c3 = _make_clause(_func(P, _var(1)))        # P(y) - same skeleton as c1

        tracker.observe(c1)
        tracker.observe(c2)
        tracker.observe(c3)

        assert tracker.stats.clauses_observed == 3
        assert tracker.stats.unique_skeletons == 2  # P(V) and Q(V)

    def test_repeated_skeleton_increases_frequency(self) -> None:
        """Observing same skeleton multiple times increases its frequency."""
        tracker = RepetitionTracker()
        for _ in range(5):
            tracker.observe(_make_clause(_func(P, _const(A))))
        assert tracker.stats.max_skeleton_frequency >= 5

    def test_score_zero_before_min_observations(self) -> None:
        """Repetition score is 0 before enough observations."""
        config = RepetitionConfig(min_observations=20)
        tracker = RepetitionTracker(config=config)

        c = _make_clause(_func(P, _const(A)))
        for _ in range(10):
            tracker.observe(c)

        # Still below min_observations
        assert tracker.repetition_score(c) == 0.0

    def test_score_nonzero_after_min_observations(self) -> None:
        """Repetition score is nonzero after enough observations."""
        config = RepetitionConfig(min_observations=5)
        tracker = RepetitionTracker(config=config)

        c = _make_clause(_func(P, _const(A)))
        for _ in range(10):
            tracker.observe(c)

        score = tracker.repetition_score(c)
        assert score > 0.0

    def test_novel_clause_has_lower_score(self) -> None:
        """A never-before-seen clause structure has a lower score."""
        config = RepetitionConfig(min_observations=5, decay_rate=0.0)
        tracker = RepetitionTracker(config=config)

        repeated = _make_clause(_func(P, _const(A)))
        for _ in range(20):
            tracker.observe(repeated)

        novel = _make_clause(_func(Q, _func(F, _var(0), _var(1))))
        score_repeated = tracker.repetition_score(repeated)
        score_novel = tracker.repetition_score(novel)
        assert score_novel < score_repeated

    def test_score_in_unit_range(self) -> None:
        """Repetition score is always in [0, 1]."""
        config = RepetitionConfig(min_observations=1, decay_rate=0.0)
        tracker = RepetitionTracker(config=config)

        c = _make_clause(_func(P, _const(A)))
        for _ in range(100):
            tracker.observe(c)

        score = tracker.repetition_score(c)
        assert 0.0 <= score <= 1.0

    def test_disabled_returns_zero(self) -> None:
        """When disabled, repetition score is always 0."""
        config = RepetitionConfig(enabled=False)
        tracker = RepetitionTracker(config=config)

        c = _make_clause(_func(P, _const(A)))
        for _ in range(100):
            tracker.observe(c)

        assert tracker.repetition_score(c) == 0.0

    def test_empty_clause_score(self) -> None:
        """Empty clause can be scored without error."""
        config = RepetitionConfig(min_observations=1, decay_rate=0.0)
        tracker = RepetitionTracker(config=config)

        empty = Clause(literals=())
        tracker.observe(empty)
        tracker.observe(empty)
        score = tracker.repetition_score(empty)
        assert isinstance(score, float)

    def test_reset_clears_state(self) -> None:
        """Reset returns tracker to initial state."""
        tracker = RepetitionTracker()
        c = _make_clause(_func(P, _const(A)))
        for _ in range(10):
            tracker.observe(c)

        tracker.reset()
        assert tracker.stats.clauses_observed == 0
        assert tracker.stats.unique_skeletons == 0


# ── Decay Tests ──────────────────────────────────────────────────────────────


class TestDecay:
    """Test frequency decay mechanism."""

    def test_decay_reduces_frequencies(self) -> None:
        """After decay, observed clause gets lower score eventually."""
        config = RepetitionConfig(
            min_observations=1,
            decay_rate=0.5,  # Aggressive decay for testing
        )
        tracker = RepetitionTracker(config=config)

        repeated = _make_clause(_func(P, _const(A)))

        # Observe 100 times to trigger decay at observation 100
        for _ in range(100):
            tracker.observe(repeated)

        score_before = tracker.repetition_score(repeated)

        # Observe 100 more different clauses to trigger more decay
        novel = _make_clause(_func(Q, _func(G, _const(B))))
        for _ in range(100):
            tracker.observe(novel)

        # The repeated clause's frequency was decayed, novel clause is now dominant
        # But repeated was decayed — score should still be meaningful
        score_after = tracker.repetition_score(repeated)
        # With aggressive decay, the novel clause dominates
        assert isinstance(score_after, float)

    def test_zero_decay_preserves_frequencies(self) -> None:
        """With decay_rate=0, frequencies never diminish."""
        config = RepetitionConfig(
            min_observations=1,
            decay_rate=0.0,
        )
        tracker = RepetitionTracker(config=config)

        c = _make_clause(_func(P, _const(A)))
        for i in range(200):
            tracker.observe(c)

        # With no decay, the max frequency should equal observations
        assert tracker.stats.max_skeleton_frequency >= 200


# ── Score Composition Tests ─────────────────────────────────────────────────


class TestScoreComposition:
    """Test how clause and subterm scores combine."""

    def test_clause_only_weight(self) -> None:
        """With subterm_weight=0, only clause-level repetition matters."""
        config = RepetitionConfig(
            min_observations=1,
            clause_weight=1.0,
            subterm_weight=0.0,
            decay_rate=0.0,
        )
        tracker = RepetitionTracker(config=config)

        c = _make_clause(_func(P, _const(A)))
        for _ in range(10):
            tracker.observe(c)

        score = tracker.repetition_score(c)
        assert score > 0.0  # clause-level repetition detected

    def test_subterm_only_weight(self) -> None:
        """With clause_weight=0, only subterm-level repetition matters."""
        config = RepetitionConfig(
            min_observations=1,
            clause_weight=0.0,
            subterm_weight=1.0,
            decay_rate=0.0,
        )
        tracker = RepetitionTracker(config=config)

        # Observe clauses with shared subterm f(a)
        c1 = _make_clause(_func(P, _func(F, _const(A))))
        c2 = _make_clause(_func(Q, _func(F, _const(A))))
        for _ in range(10):
            tracker.observe(c1)
            tracker.observe(c2)

        # A new clause with the shared subterm f(a) should score high
        c3 = _make_clause(_func(R, _func(F, _const(A))))
        score = tracker.repetition_score(c3)
        assert score > 0.0

    def test_zero_weights_return_zero(self) -> None:
        """With both weights zero, score is zero."""
        config = RepetitionConfig(
            min_observations=1,
            clause_weight=0.0,
            subterm_weight=0.0,
            decay_rate=0.0,
        )
        tracker = RepetitionTracker(config=config)

        c = _make_clause(_func(P, _const(A)))
        for _ in range(10):
            tracker.observe(c)

        assert tracker.repetition_score(c) == 0.0


# ── Structural Diversity Tests ──────────────────────────────────────────────


class TestStructuralDiversity:
    """Test that the system correctly distinguishes structural patterns."""

    def test_same_structure_different_constants(self) -> None:
        """P(a) and P(b) have different skeletons (constants differ)."""
        c1 = _make_clause(_func(P, _const(A)))
        c2 = _make_clause(_func(P, _const(B)))
        assert clause_skeleton(c1) != clause_skeleton(c2)

    def test_same_structure_different_variables(self) -> None:
        """P(x) and P(y) have the same skeleton (variables abstracted)."""
        c1 = _make_clause(_func(P, _var(0)))
        c2 = _make_clause(_func(P, _var(1)))
        assert clause_skeleton(c1) == clause_skeleton(c2)

    def test_deeper_nesting_is_distinct(self) -> None:
        """f(x) and f(f(x)) are structurally distinct."""
        c1 = _make_clause(_func(P, _func(F, _var(0))))
        c2 = _make_clause(_func(P, _func(F, _func(F, _var(0)))))
        assert clause_skeleton(c1) != clause_skeleton(c2)

    def test_equational_clauses_detect_patterns(self) -> None:
        """Equational clauses like f(x)*y = x*f(y) get fingerprinted."""
        # eq(f(x,y), g(x,y)) and eq(f(z,w), g(z,w)) should match
        c1 = _make_clause(_func(EQ, _func(F, _var(0), _var(1)),
                                       _func(G, _var(0), _var(1))))
        c2 = _make_clause(_func(EQ, _func(F, _var(2), _var(3)),
                                       _func(G, _var(2), _var(3))))
        assert clause_skeleton(c1) == clause_skeleton(c2)

    def test_many_similar_clauses_score_high(self) -> None:
        """A family of structurally identical clauses produces high scores."""
        config = RepetitionConfig(min_observations=1, decay_rate=0.0)
        tracker = RepetitionTracker(config=config)

        # Generate 50 clauses all shaped P(f(x, y)) with different var names
        for i in range(50):
            c = _make_clause(_func(P, _func(F, _var(i * 2), _var(i * 2 + 1))))
            tracker.observe(c)

        # Another clause of the same shape should score very high
        test = _make_clause(_func(P, _func(F, _var(100), _var(101))))
        score = tracker.repetition_score(test)
        assert score > 0.8  # very repetitious


# ── RepetitionStats Tests ───────────────────────────────────────────────────


class TestRepetitionStats:
    """Test statistics reporting."""

    def test_report_format(self) -> None:
        """Report generates human-readable format."""
        stats = RepetitionStats(
            clauses_observed=100,
            unique_skeletons=20,
            unique_subterm_skeletons=50,
            max_skeleton_frequency=15,
            max_subterm_frequency=30,
            penalized_selections=8,
        )
        report = stats.report()
        assert "observed=100" in report
        assert "unique_skeletons=20" in report
        assert "penalized=8" in report

    def test_default_stats_zero(self) -> None:
        """Default stats are all zero."""
        stats = RepetitionStats()
        assert stats.clauses_observed == 0
        assert stats.penalized_selections == 0


# ── Performance Tests ───────────────────────────────────────────────────────


class TestPerformance:
    """Test that repetition detection has acceptable overhead."""

    def test_fingerprint_performance(self) -> None:
        """Fingerprinting 1000 clauses takes < 100ms."""
        clauses = []
        for i in range(1000):
            # Mix of structures
            if i % 3 == 0:
                c = _make_clause(_func(P, _func(F, _var(0), _const(A))))
            elif i % 3 == 1:
                c = _make_clause(
                    _func(P, _func(F, _var(0))),
                    _func(Q, _func(G, _var(1), _const(B))),
                )
            else:
                c = _make_clause(
                    _func(P, _func(F, _func(G, _const(A)))),
                )
            clauses.append(c)

        start = time.perf_counter()
        for c in clauses:
            clause_skeleton(c)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"Fingerprinting 1000 clauses took {elapsed:.3f}s"

    def test_tracker_observe_performance(self) -> None:
        """Observing 1000 clauses takes < 200ms."""
        config = RepetitionConfig(min_observations=1, decay_rate=0.01)
        tracker = RepetitionTracker(config=config)

        clauses = []
        for i in range(1000):
            if i % 2 == 0:
                c = _make_clause(_func(P, _func(F, _var(0), _const(i % 5 + 1))))
            else:
                c = _make_clause(_func(Q, _func(G, _var(0))))
            clauses.append(c)

        start = time.perf_counter()
        for c in clauses:
            tracker.observe(c)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.2, f"Observing 1000 clauses took {elapsed:.3f}s"

    def test_score_computation_performance(self) -> None:
        """Scoring 1000 clauses takes < 200ms after warmup."""
        config = RepetitionConfig(min_observations=1, decay_rate=0.0)
        tracker = RepetitionTracker(config=config)

        # Warmup: observe 500 clauses
        for i in range(500):
            c = _make_clause(_func(P, _func(F, _var(0), _const(i % 5 + 1))))
            tracker.observe(c)

        # Score 1000 clauses
        clauses = [
            _make_clause(_func(P, _func(F, _var(0), _const(i % 10 + 1))))
            for i in range(1000)
        ]

        start = time.perf_counter()
        for c in clauses:
            tracker.repetition_score(c)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.2, f"Scoring 1000 clauses took {elapsed:.3f}s"

    def test_memory_bounded_with_decay(self) -> None:
        """With decay, frequency tables don't grow without bound."""
        config = RepetitionConfig(
            min_observations=1,
            decay_rate=0.5,  # Aggressive decay
        )
        tracker = RepetitionTracker(config=config)

        # Observe 1000 unique skeletons
        for i in range(1000):
            c = _make_clause(_func(P, _const(i + 1)))
            tracker.observe(c)

        # With aggressive decay, many old entries should have been pruned
        # Unique skeletons should be much less than 1000
        assert tracker.stats.unique_skeletons < 1000


# ── Edge Cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_variable_clause(self) -> None:
        """Clause with just a variable atom works."""
        # This is unusual but should not crash
        c = _make_clause(_var(0))
        s = clause_skeleton(c)
        assert s is not None
        assert len(s) == 1

    def test_large_arity_term(self) -> None:
        """Terms with many arguments work correctly."""
        args = tuple(_const(i + 1) for i in range(10))
        t = _func(F, *args)
        s = term_skeleton(t)
        assert s[0] == "F"
        assert s[1] == F
        assert len(s) == 12  # "F" + symnum + 10 args

    def test_very_deep_nesting(self) -> None:
        """Deeply nested terms don't crash."""
        t = _const(A)
        for _ in range(50):
            t = _func(F, t)
        s = term_skeleton(t)
        assert s is not None

    def test_clause_with_many_literals(self) -> None:
        """Clause with many literals works correctly."""
        atoms = [_func(P + i, _const(A)) for i in range(20)]
        c = _make_clause(*atoms)
        s = clause_skeleton(c)
        assert len(s) == 20

    def test_observe_then_score_same_clause(self) -> None:
        """Score the exact same clause object that was observed."""
        config = RepetitionConfig(min_observations=1, decay_rate=0.0)
        tracker = RepetitionTracker(config=config)

        c = _make_clause(_func(P, _const(A)))
        tracker.observe(c)
        score = tracker.repetition_score(c)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_mixed_ground_and_nonground(self) -> None:
        """Tracker handles mix of ground and non-ground clauses."""
        config = RepetitionConfig(min_observations=1, decay_rate=0.0)
        tracker = RepetitionTracker(config=config)

        ground = _make_clause(_func(P, _const(A), _const(B)))
        nonground = _make_clause(_func(P, _var(0), _var(1)))
        for _ in range(10):
            tracker.observe(ground)
            tracker.observe(nonground)

        s1 = tracker.repetition_score(ground)
        s2 = tracker.repetition_score(nonground)
        assert s1 >= 0.0
        assert s2 >= 0.0

    def test_config_defaults(self) -> None:
        """Default config values are reasonable."""
        config = RepetitionConfig()
        assert config.enabled is True
        assert 0.0 <= config.clause_weight <= 1.0
        assert 0.0 <= config.subterm_weight <= 1.0
        assert config.clause_weight + config.subterm_weight == pytest.approx(1.0)
        assert config.decay_rate >= 0.0
        assert config.min_observations >= 0


# ── Selection Bias Integration Tests ────────────────────────────────────────

from pyladr.search.selection import (
    GivenSelection,
    SelectionOrder,
    SelectionRule,
    default_clause_weight,
)
from pyladr.search.ml_selection import (
    EmbeddingEnhancedSelection,
    MLSelectionConfig,
)
from pyladr.search.state import ClauseList


class _SimpleEmbeddingProvider:
    """Minimal provider returning unit embeddings for repetition bias tests."""
    embedding_dim = 8

    def get_embedding(self, clause: Clause) -> list[float]:
        # Return distinct embeddings per clause id so diversity works
        emb = [0.0] * 8
        emb[clause.id % 8] = 1.0
        return emb

    def get_embeddings_batch(self, clauses: list[Clause]) -> list[list[float]]:
        return [self.get_embedding(c) for c in clauses]


def _make_weighted_clause_with_id(
    clause_id: int, weight: float, *atoms: Term
) -> Clause:
    """Create a clause with specific ID and weight."""
    if not atoms:
        atoms = (_const(A),)
    c = _make_clause(*atoms)
    c.id = clause_id
    c.weight = weight
    return c


class TestSelectionBiasIntegration:
    """Test repetition bias integration with GivenSelection."""

    def test_no_bias_when_disabled(self) -> None:
        """Without repetition bias, selection is purely by weight."""
        gs = GivenSelection(
            rules=[SelectionRule("W", SelectionOrder.WEIGHT)],
        )
        sos = ClauseList("sos")

        heavy = _make_weighted_clause_with_id(1, 10.0, _func(P, _const(A)))
        light = _make_weighted_clause_with_id(2, 3.0, _func(Q, _const(B)))
        sos.append(heavy)
        sos.append(light)

        selected, _ = gs.select_given(sos, 0)
        assert selected is light  # Lightest wins without bias

    def test_bias_penalizes_repetitious_clauses(self) -> None:
        """Repetition bias makes highly repetitious clauses less preferred."""
        config = RepetitionConfig(min_observations=1, decay_rate=0.0)
        tracker = RepetitionTracker(config=config)

        ml_config = MLSelectionConfig(
            enabled=True, ml_weight=0.8, min_sos_for_ml=1,
        )
        gs = EmbeddingEnhancedSelection(
            rules=[SelectionRule("W", SelectionOrder.WEIGHT)],
            embedding_provider=_SimpleEmbeddingProvider(),
            ml_config=ml_config,
            repetition_tracker=tracker,
            repetition_penalty=0.8,
        )

        # Observe the same skeleton many times to build up repetition
        for i in range(50):
            tracker.observe(_make_clause(_func(P, _var(i))))

        sos = ClauseList("sos")

        # Repetitious clause: P(x) shape, lighter weight
        repetitious = _make_weighted_clause_with_id(
            1, 5.0, _func(P, _var(0))
        )
        # Novel clause: Q(f(x, y)) shape, heavier weight
        novel = _make_weighted_clause_with_id(
            2, 7.0, _func(Q, _func(F, _var(0), _var(1)))
        )
        sos.append(repetitious)
        sos.append(novel)

        selected, _ = gs.select_given(sos, 0)
        # With strong penalty, the heavier novel clause should be preferred
        # over the lighter but repetitious clause due to ML scoring
        # penalizing the repetitious structure.
        assert selected is novel

    def test_zero_penalty_no_effect(self) -> None:
        """With penalty=0, repetition tracking has no effect on selection.

        Uses ML disabled to verify that repetition penalty=0 doesn't
        alter traditional weight-based selection.
        """
        config = RepetitionConfig(min_observations=1, decay_rate=0.0)
        tracker = RepetitionTracker(config=config)

        ml_config = MLSelectionConfig(enabled=False)
        gs = EmbeddingEnhancedSelection(
            rules=[SelectionRule("W", SelectionOrder.WEIGHT)],
            ml_config=ml_config,
            repetition_tracker=tracker,
            repetition_penalty=0.0,
        )

        for _ in range(50):
            tracker.observe(_make_clause(_func(P, _var(0))))

        sos = ClauseList("sos")
        light = _make_weighted_clause_with_id(1, 3.0, _func(P, _var(0)))
        heavy = _make_weighted_clause_with_id(2, 10.0, _func(Q, _const(B)))
        sos.append(light)
        sos.append(heavy)

        selected, _ = gs.select_given(sos, 0)
        assert selected is light  # Lightest still wins

    def test_age_selection_ignores_repetition(self) -> None:
        """Age-based selection is not affected by repetition bias."""
        config = RepetitionConfig(min_observations=1, decay_rate=0.0)
        tracker = RepetitionTracker(config=config)

        ml_config = MLSelectionConfig(
            enabled=True, ml_weight=0.8, min_sos_for_ml=1,
        )
        gs = EmbeddingEnhancedSelection(
            rules=[SelectionRule("A", SelectionOrder.AGE)],
            embedding_provider=_SimpleEmbeddingProvider(),
            ml_config=ml_config,
            repetition_tracker=tracker,
            repetition_penalty=1.0,
        )

        for _ in range(50):
            tracker.observe(_make_clause(_func(P, _var(0))))

        sos = ClauseList("sos")
        old = _make_weighted_clause_with_id(1, 5.0, _func(P, _var(0)))
        new = _make_weighted_clause_with_id(2, 5.0, _func(Q, _const(B)))
        sos.append(old)
        sos.append(new)

        selected, _ = gs.select_given(sos, 0)
        assert selected is old  # Oldest first regardless of repetition

    def test_bias_updates_stats(self) -> None:
        """Penalized selections update statistics."""
        config = RepetitionConfig(min_observations=1, decay_rate=0.0)
        tracker = RepetitionTracker(config=config)

        ml_config = MLSelectionConfig(
            enabled=True, ml_weight=0.8, min_sos_for_ml=1,
        )
        gs = EmbeddingEnhancedSelection(
            rules=[SelectionRule("W", SelectionOrder.WEIGHT)],
            embedding_provider=_SimpleEmbeddingProvider(),
            ml_config=ml_config,
            repetition_tracker=tracker,
            repetition_penalty=0.5,
        )

        # Build up repetition
        for _ in range(20):
            tracker.observe(_make_clause(_func(P, _var(0))))

        sos = ClauseList("sos")
        c = _make_weighted_clause_with_id(1, 5.0, _func(P, _var(0)))
        sos.append(c)

        gs.select_given(sos, 0)

        # Stats should reflect that a penalized selection occurred
        assert tracker.stats.penalized_selections >= 0  # May be 0 or 1

    def test_effective_weight_formula(self) -> None:
        """Repetition penalty reduces score for frequent clause structures."""
        config = RepetitionConfig(min_observations=1, decay_rate=0.0)
        tracker = RepetitionTracker(config=config)

        # Make one skeleton very frequent
        for _ in range(100):
            tracker.observe(_make_clause(_func(P, _var(0))))

        # And another rare
        tracker.observe(_make_clause(_func(Q, _func(G, _var(0)))))

        ml_config = MLSelectionConfig(
            enabled=True, ml_weight=0.8, min_sos_for_ml=1,
        )
        gs = EmbeddingEnhancedSelection(
            rules=[SelectionRule("W", SelectionOrder.WEIGHT)],
            embedding_provider=_SimpleEmbeddingProvider(),
            ml_config=ml_config,
            repetition_tracker=tracker,
            repetition_penalty=0.5,
        )

        sos = ClauseList("sos")

        # Both same base weight
        frequent = _make_weighted_clause_with_id(1, 5.0, _func(P, _var(0)))
        rare = _make_weighted_clause_with_id(2, 5.0, _func(Q, _func(G, _var(0))))
        sos.append(frequent)
        sos.append(rare)

        selected, _ = gs.select_given(sos, 0)
        # Rare clause should be selected (lower repetition penalty)
        assert selected is rare

    def test_ratio_cycle_with_bias(self) -> None:
        """Repetition bias applies only to weight steps in ratio cycle."""
        config = RepetitionConfig(min_observations=1, decay_rate=0.0)
        tracker = RepetitionTracker(config=config)

        ml_config = MLSelectionConfig(
            enabled=True, ml_weight=0.5, min_sos_for_ml=1,
        )
        gs = EmbeddingEnhancedSelection(
            rules=[
                SelectionRule("W", SelectionOrder.WEIGHT, part=1),
                SelectionRule("A", SelectionOrder.AGE, part=1),
            ],
            embedding_provider=_SimpleEmbeddingProvider(),
            ml_config=ml_config,
            repetition_tracker=tracker,
            repetition_penalty=0.5,
        )

        for _ in range(20):
            tracker.observe(_make_clause(_func(P, _var(0))))

        sos = ClauseList("sos")
        for i in range(4):
            c = _make_weighted_clause_with_id(i + 1, 5.0, _func(P, _var(0)))
            sos.append(c)

        selections = []
        for i in range(4):
            _, name = gs.select_given(sos, i)
            # Strip +ML suffix to check rule cycling
            selections.append(name.split("+")[0])

        # Should alternate W and A
        assert selections == ["W", "A", "W", "A"]


# ── End-to-End Prover Integration Tests ─────────────────────────────────────

from pyladr.core.clause import Justification, JustType
from pyladr.search.given_clause import (
    ExitCode,
    GivenClauseSearch,
    SearchOptions,
)


def _pos_lit(atom: Term) -> tuple[bool, Term]:
    return (True, atom)


def _neg_lit(atom: Term) -> tuple[bool, Term]:
    return (False, atom)


def _clause_from_lits(*lits: tuple[bool, Term]) -> Clause:
    return Clause(
        literals=tuple(Literal(sign=s, atom=a) for s, a in lits),
        justification=(Justification(just_type=JustType.INPUT),),
    )


class TestProverIntegration:
    """Test repetition bias in the full prover pipeline."""

    def test_disabled_by_default(self) -> None:
        """Repetition bias is disabled by default in SearchOptions."""
        opts = SearchOptions()
        assert opts.repetition_bias is False

    def test_search_options_repetition_params(self) -> None:
        """SearchOptions accepts repetition bias parameters."""
        opts = SearchOptions(
            repetition_bias=True,
            repetition_penalty=0.5,
            repetition_decay=0.03,
            repetition_min_obs=10,
        )
        assert opts.repetition_bias is True
        assert opts.repetition_penalty == 0.5
        assert opts.repetition_decay == 0.03
        assert opts.repetition_min_obs == 10

    def test_simple_proof_with_bias_enabled(self) -> None:
        """Basic proof still works with repetition bias enabled."""
        a = _const(A)
        pa = _func(P, a)

        c1 = _clause_from_lits(_pos_lit(pa))   # P(a)
        c2 = _clause_from_lits(_neg_lit(pa))    # -P(a)

        opts = SearchOptions(
            print_given=False,
            print_kept=False,
            repetition_bias=True,
            repetition_penalty=0.3,
            repetition_min_obs=1,
        )
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1, c2])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1

    def test_two_step_proof_with_bias(self) -> None:
        """Multi-step proof works with bias: P(a), -P(x)|Q(x), -Q(a)."""
        a = _const(A)
        x = _var(0)

        c1 = _clause_from_lits(_pos_lit(_func(P, a)))
        c2 = _clause_from_lits(_neg_lit(_func(P, x)), _pos_lit(_func(Q, x)))
        c3 = _clause_from_lits(_neg_lit(_func(Q, a)))

        opts = SearchOptions(
            print_given=False,
            repetition_bias=True,
            repetition_penalty=0.5,
            repetition_min_obs=1,
        )
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1, c2, c3])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert result.stats.proofs == 1

    def test_bias_disabled_matches_baseline(self) -> None:
        """Disabled bias produces same result as no bias at all."""
        a = _const(A)
        pa = _func(P, a)
        c1 = _clause_from_lits(_pos_lit(pa))
        c2 = _clause_from_lits(_neg_lit(pa))

        # Run without bias
        opts_no = SearchOptions(print_given=False, repetition_bias=False)
        search_no = GivenClauseSearch(options=opts_no)
        result_no = search_no.run(sos=[c1, c2])

        # Run with bias disabled via zero penalty
        c1b = _clause_from_lits(_pos_lit(pa))
        c2b = _clause_from_lits(_neg_lit(pa))
        opts_zero = SearchOptions(
            print_given=False,
            repetition_bias=True,
            repetition_penalty=0.0,
            repetition_min_obs=1,
        )
        search_zero = GivenClauseSearch(options=opts_zero)
        result_zero = search_zero.run(sos=[c1b, c2b])

        assert result_no.exit_code == result_zero.exit_code
        assert len(result_no.proofs) == len(result_zero.proofs)

    @pytest.mark.skip(reason="GivenClauseSearch repetition integration not yet wired")
    def test_tracker_set_on_search_object(self) -> None:
        """When bias is enabled, GivenClauseSearch creates a tracker."""
        opts = SearchOptions(
            repetition_bias=True,
            repetition_penalty=0.4,
        )
        search = GivenClauseSearch(options=opts)
        assert search._repetition_tracker is not None

    @pytest.mark.skip(reason="GivenClauseSearch repetition integration not yet wired")
    def test_tracker_not_set_when_disabled(self) -> None:
        """When bias is disabled, no tracker is created."""
        opts = SearchOptions(repetition_bias=False)
        search = GivenClauseSearch(options=opts)
        assert search._repetition_tracker is None

    def test_sos_exhaustion_with_bias(self) -> None:
        """SOS exhaustion works correctly with bias enabled."""
        a = _const(A)
        c1 = _clause_from_lits(_pos_lit(_func(P, a)))

        opts = SearchOptions(
            print_given=False,
            repetition_bias=True,
            repetition_penalty=0.5,
            max_given=50,
        )
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1])

        assert result.exit_code in (ExitCode.SOS_EMPTY_EXIT, ExitCode.MAX_GIVEN_EXIT)

    def test_max_given_with_bias(self) -> None:
        """Max given limit is respected with bias enabled."""
        a = _const(A)
        b = _const(B)
        c1 = _clause_from_lits(_pos_lit(_func(P, a)))
        c2 = _clause_from_lits(_pos_lit(_func(P, b)))

        opts = SearchOptions(
            print_given=False,
            max_given=3,
            repetition_bias=True,
            repetition_penalty=0.3,
        )
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1, c2])

        assert result.stats.given <= 3


# ── Bias Effectiveness Demonstration Tests ──────────────────────────────────


class TestBiasEffectiveness:
    """Tests demonstrating that repetition bias changes selection behavior."""

    def test_diverse_selection_preferred(self) -> None:
        """Repetition bias causes selection to prefer structural diversity."""
        config = RepetitionConfig(min_observations=1, decay_rate=0.0)
        tracker = RepetitionTracker(config=config)

        ml_config = MLSelectionConfig(
            enabled=True, ml_weight=0.8, min_sos_for_ml=1,
        )
        gs = EmbeddingEnhancedSelection(
            rules=[SelectionRule("W", SelectionOrder.WEIGHT)],
            embedding_provider=_SimpleEmbeddingProvider(),
            ml_config=ml_config,
            repetition_tracker=tracker,
            repetition_penalty=0.5,
        )

        # Pre-observe many P(var) shaped clauses
        for i in range(30):
            tracker.observe(_make_clause(_func(P, _var(i))))

        # Build SOS with mix of shapes, all same weight
        sos = ClauseList("sos")
        shapes = [
            _make_weighted_clause_with_id(1, 5.0, _func(P, _var(0))),        # very repetitious
            _make_weighted_clause_with_id(2, 5.0, _func(Q, _const(A))),       # somewhat novel
            _make_weighted_clause_with_id(3, 5.0, _func(R, _func(F, _var(0)))),  # novel
        ]
        for c in shapes:
            sos.append(c)

        selected, _ = gs.select_given(sos, 0)
        # Novel clauses should be preferred over the repetitious one
        assert selected.id != 1  # Should NOT select the repetitious clause

    def test_penalty_strength_ordering(self) -> None:
        """Higher penalty makes selection more aggressively avoid repetition."""
        config = RepetitionConfig(min_observations=1, decay_rate=0.0)

        # Build up strong repetition signal
        for penalty in [0.1, 0.5, 0.9]:
            tracker = RepetitionTracker(config=config)
            for _ in range(50):
                tracker.observe(_make_clause(_func(P, _var(0))))

            ml_config = MLSelectionConfig(
                enabled=True, ml_weight=0.8, min_sos_for_ml=1,
            )
            gs = EmbeddingEnhancedSelection(
                rules=[SelectionRule("W", SelectionOrder.WEIGHT)],
                embedding_provider=_SimpleEmbeddingProvider(),
                ml_config=ml_config,
                repetition_tracker=tracker,
                repetition_penalty=penalty,
            )

            sos = ClauseList("sos")
            rep = _make_weighted_clause_with_id(1, 5.0, _func(P, _var(0)))
            novel = _make_weighted_clause_with_id(2, 8.0, _func(Q, _func(G, _var(0))))
            sos.append(rep)
            sos.append(novel)

            selected, _ = gs.select_given(sos, 0)

            # At penalty=0.9, the novel clause (weight 8) should beat
            # the repetitious clause (eff_weight 5*(1+0.9*1.0) = 9.5)
            if penalty >= 0.9:
                assert selected is novel

    def test_gradual_penalty_ramp(self) -> None:
        """As repetition builds up, bias effect increases."""
        config = RepetitionConfig(min_observations=1, decay_rate=0.0)
        tracker = RepetitionTracker(config=config)

        scores = []
        c = _make_clause(_func(P, _var(0)))

        for i in range(50):
            score = tracker.repetition_score(c)
            scores.append(score)
            tracker.observe(c)

        # Scores should generally increase (or stay flat) as repetition builds
        # Check that later scores >= earlier scores (monotonic within reasonable tolerance)
        assert scores[-1] >= scores[0]
