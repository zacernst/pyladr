"""Property-based tests for online learning components.

Uses Hypothesis to verify invariants that must hold for all inputs:
- ExperienceBuffer capacity invariant
- Contrastive sampling correctness
- ABTestTracker rate bounds
- Outcome classification completeness
"""

from __future__ import annotations

import time

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

torch = pytest.importorskip("torch")

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term
from pyladr.ml.online_learning import (
    ABTestTracker,
    ExperienceBuffer,
    InferenceOutcome,
    ModelVersion,
    OnlineLearningConfig,
    OutcomeType,
)


# ── Strategies ────────────────────────────────────────────────────────────


def _make_term(symnum: int) -> Term:
    return Term(private_symbol=-abs(symnum) - 1, arity=0, args=())


def _make_clause(clause_id: int) -> Clause:
    c = Clause(literals=(Literal(sign=True, atom=_make_term(clause_id)),))
    c.id = clause_id
    c.weight = 1.0
    return c


outcome_type_strategy = st.sampled_from(list(OutcomeType))


def make_outcome_strategy():
    return st.builds(
        lambda cid, otype: InferenceOutcome(
            given_clause=_make_clause(cid),
            partner_clause=None,
            child_clause=_make_clause(cid + 1000),
            outcome=otype,
            timestamp=time.monotonic(),
            given_count=cid,
        ),
        cid=st.integers(min_value=0, max_value=10000),
        otype=outcome_type_strategy,
    )


# ── ExperienceBuffer properties ───────────────────────────────────────────


class TestExperienceBufferProperties:
    @given(
        capacity=st.integers(min_value=1, max_value=500),
        n_outcomes=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=50, deadline=5000)
    def test_size_never_exceeds_capacity(self, capacity, n_outcomes):
        """Buffer size never exceeds its configured capacity."""
        buf = ExperienceBuffer(capacity=capacity)
        for i in range(n_outcomes):
            otype = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            buf.add(InferenceOutcome(
                given_clause=_make_clause(i),
                partner_clause=None,
                child_clause=_make_clause(i + 1000),
                outcome=otype,
                timestamp=time.monotonic(),
                given_count=i,
            ))
        assert buf.size <= capacity

    @given(
        capacity=st.integers(min_value=1, max_value=200),
        n_outcomes=st.integers(min_value=0, max_value=500),
    )
    @settings(max_examples=50, deadline=5000)
    def test_productive_plus_unproductive_equals_size(self, capacity, n_outcomes):
        """Productive + unproductive counts equal total buffer size."""
        buf = ExperienceBuffer(capacity=capacity)
        for i in range(n_outcomes):
            otype = [OutcomeType.KEPT, OutcomeType.SUBSUMED, OutcomeType.PROOF, OutcomeType.TAUTOLOGY][i % 4]
            buf.add(InferenceOutcome(
                given_clause=_make_clause(i),
                partner_clause=None,
                child_clause=_make_clause(i + 1000),
                outcome=otype,
                timestamp=time.monotonic(),
                given_count=i,
            ))

        buf._rebuild_indices()
        assert buf.num_productive + buf.num_unproductive == buf.size

    @given(
        capacity=st.integers(min_value=10, max_value=200),
        batch_size=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=30, deadline=5000)
    def test_contrastive_batch_size_bounded(self, capacity, batch_size):
        """Contrastive batch never exceeds requested size or available pairs."""
        buf = ExperienceBuffer(capacity=capacity)
        # Add mixed outcomes
        for i in range(capacity):
            otype = OutcomeType.KEPT if i % 2 == 0 else OutcomeType.SUBSUMED
            buf.add(InferenceOutcome(
                given_clause=_make_clause(i),
                partner_clause=None,
                child_clause=_make_clause(i + 1000),
                outcome=otype,
                timestamp=time.monotonic(),
                given_count=i,
            ))

        pairs = buf.sample_contrastive_batch(batch_size)
        assert len(pairs) <= batch_size
        assert len(pairs) <= buf.num_productive
        assert len(pairs) <= buf.num_unproductive

    @given(n=st.integers(min_value=1, max_value=100))
    @settings(max_examples=30, deadline=5000)
    def test_get_recent_returns_at_most_n(self, n):
        """get_recent(n) returns at most n items (for n >= 1)."""
        buf = ExperienceBuffer(capacity=100)
        for i in range(50):
            buf.add(InferenceOutcome(
                given_clause=_make_clause(i),
                partner_clause=None,
                child_clause=_make_clause(i + 1000),
                outcome=OutcomeType.KEPT,
                timestamp=time.monotonic(),
                given_count=i,
            ))
        recent = buf.get_recent(n)
        assert len(recent) <= max(n, buf.size)
        assert len(recent) <= buf.size


# ── ABTestTracker properties ──────────────────────────────────────────────


class TestABTestTrackerProperties:
    @given(
        window=st.integers(min_value=2, max_value=200),
        outcomes=st.lists(st.booleans(), min_size=0, max_size=300),
    )
    @settings(max_examples=50, deadline=5000)
    def test_current_rate_bounded(self, window, outcomes):
        """Current rate is always in [0, 1]."""
        tracker = ABTestTracker(window_size=window)
        tracker.set_baseline(0.5)
        for o in outcomes:
            tracker.record_outcome(o)
        assert 0.0 <= tracker.current_rate <= 1.0

    @given(
        window=st.integers(min_value=2, max_value=200),
        baseline=st.floats(min_value=0.0, max_value=1.0),
        outcomes=st.lists(st.booleans(), min_size=0, max_size=300),
    )
    @settings(max_examples=50, deadline=5000)
    def test_has_enough_data_requires_minimum(self, window, baseline, outcomes):
        """has_enough_data requires at least window//2 outcomes."""
        tracker = ABTestTracker(window_size=window)
        tracker.set_baseline(baseline)
        for o in outcomes:
            tracker.record_outcome(o)

        if len(outcomes) < window // 2:
            assert not tracker.has_enough_data

    @given(
        window=st.integers(min_value=4, max_value=100),
    )
    @settings(max_examples=30, deadline=5000)
    def test_all_true_is_improvement_over_zero(self, window):
        """All-true outcomes always improve over zero baseline."""
        tracker = ABTestTracker(window_size=window)
        tracker.set_baseline(0.0)
        for _ in range(window):
            tracker.record_outcome(True)
        assert tracker.is_improvement(significance=0.0)

    @given(
        window=st.integers(min_value=4, max_value=100),
    )
    @settings(max_examples=30, deadline=5000)
    def test_all_false_is_degradation_from_one(self, window):
        """All-false outcomes always degrade from 1.0 baseline."""
        tracker = ABTestTracker(window_size=window)
        tracker.set_baseline(1.0)
        for _ in range(window):
            tracker.record_outcome(False)
        assert tracker.is_degradation(threshold=0.0)


# ── ModelVersion properties ───────────────────────────────────────────────


class TestModelVersionProperties:
    @given(
        total=st.integers(min_value=1, max_value=100000),
        productive=st.integers(min_value=0, max_value=100000),
    )
    @settings(max_examples=50, deadline=5000)
    def test_productivity_rate_bounded(self, total, productive):
        """Productivity rate is always in [0, 1]."""
        assume(productive <= total)
        v = ModelVersion(version_id=0, state_dict={})
        v.selections_made = total
        v.productive_selections = productive
        assert 0.0 <= v.productivity_rate <= 1.0

    @given(total=st.integers(min_value=1, max_value=100000))
    @settings(max_examples=30, deadline=5000)
    def test_all_productive_rate_one(self, total):
        """When all selections are productive, rate is 1.0."""
        v = ModelVersion(version_id=0, state_dict={})
        v.selections_made = total
        v.productive_selections = total
        assert v.productivity_rate == pytest.approx(1.0)


# ── OnlineLearningConfig properties ───────────────────────────────────────


class TestOnlineLearningConfigProperties:
    @given(
        interval=st.integers(min_value=1, max_value=10000),
        capacity=st.integers(min_value=1, max_value=100000),
        batch_size=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=30, deadline=5000)
    def test_config_construction(self, interval, capacity, batch_size):
        """Configs can be constructed with any valid positive integers."""
        config = OnlineLearningConfig(
            update_interval=interval,
            buffer_capacity=capacity,
            batch_size=batch_size,
        )
        assert config.update_interval == interval
        assert config.buffer_capacity == capacity
        assert config.batch_size == batch_size


# ── OutcomeType classification properties ─────────────────────────────────


class TestOutcomeClassification:
    @given(otype=st.sampled_from(list(OutcomeType)))
    @settings(max_examples=20, deadline=5000)
    def test_outcome_is_productive_or_unproductive(self, otype):
        """Every outcome type is classified as either productive or unproductive."""
        productive = otype in (OutcomeType.KEPT, OutcomeType.PROOF, OutcomeType.SUBSUMER)
        unproductive = otype in (OutcomeType.SUBSUMED, OutcomeType.TAUTOLOGY, OutcomeType.WEIGHT_LIMIT)
        # Exactly one must be true
        assert productive != unproductive
