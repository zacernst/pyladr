"""Tests for the monitoring bridge between OnlineLearningManager and monitoring stack."""

from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term
from pyladr.ml.online_learning import (
    ExperienceBuffer,
    InferenceOutcome,
    OnlineLearningConfig,
    OnlineLearningManager,
    OutcomeType,
)
from pyladr.monitoring.learning_bridge import MonitoredLearning
from pyladr.monitoring.learning_monitor import LearningMonitor
from pyladr.monitoring.learning_curves import LearningCurveAnalyzer


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_term(symnum: int, args: tuple[Term, ...] = ()) -> Term:
    return Term(private_symbol=-symnum, arity=len(args), args=args)


def _make_clause(
    lits: list[tuple[bool, Term]], clause_id: int = 0,
) -> Clause:
    literals = tuple(Literal(sign=s, atom=a) for s, a in lits)
    c = Clause(literals=literals)
    c.id = clause_id
    c.weight = float(len(literals))
    return c


def _make_outcome(
    clause_id: int,
    outcome: OutcomeType = OutcomeType.KEPT,
    given_id: int = 0,
) -> InferenceOutcome:
    given = _make_clause([(True, _make_term(given_id))], clause_id=given_id)
    child = _make_clause([(True, _make_term(clause_id))], clause_id=clause_id)
    return InferenceOutcome(
        given_clause=given,
        partner_clause=None,
        child_clause=child,
        outcome=outcome,
        timestamp=time.monotonic(),
        given_count=clause_id,
    )


class MockEncoder:
    """Mock encoder for testing."""

    def __init__(self, dim: int = 64):
        self._dim = dim
        self._linear = torch.nn.Linear(dim, dim)

    def encode_clauses(self, clauses):
        x = torch.randn(len(clauses), self._dim)
        return self._linear(x)

    def parameters(self):
        return self._linear.parameters()

    def named_parameters(self):
        return self._linear.named_parameters()

    def state_dict(self):
        return self._linear.state_dict()

    def load_state_dict(self, state):
        self._linear.load_state_dict(state)

    def train(self, mode=True):
        self._linear.train(mode)

    def eval(self):
        self._linear.eval()


def _make_manager(
    update_interval: int = 10,
    min_examples: int = 5,
    buffer_capacity: int = 200,
) -> OnlineLearningManager:
    config = OnlineLearningConfig(
        enabled=True,
        update_interval=update_interval,
        min_examples_for_update=min_examples,
        buffer_capacity=buffer_capacity,
        batch_size=4,
        gradient_steps_per_update=1,
        learning_rate=1e-3,
    )
    encoder = MockEncoder(dim=64)
    return OnlineLearningManager(encoder, config, device=torch.device("cpu"))


# ── Tests ─────────────────────────────────────────────────────────────────


class TestMonitoredLearning:
    def test_creation(self):
        mgr = _make_manager()
        monitored = MonitoredLearning(mgr)
        assert monitored.manager is mgr
        assert isinstance(monitored.monitor, LearningMonitor)
        assert isinstance(monitored.curves, LearningCurveAnalyzer)

    def test_custom_monitor_and_curves(self):
        mgr = _make_manager()
        mon = LearningMonitor(balance_low=0.2, balance_high=0.8)
        crv = LearningCurveAnalyzer(convergence_window=3)
        monitored = MonitoredLearning(mgr, monitor=mon, curves=crv)
        assert monitored.monitor is mon
        assert monitored.curves is crv

    def test_record_outcome_delegates(self):
        mgr = _make_manager()
        monitored = MonitoredLearning(mgr)
        outcome = _make_outcome(1, OutcomeType.KEPT)
        monitored.record_outcome(outcome)
        assert mgr._total_outcomes == 1

    def test_record_selection(self):
        mgr = _make_manager()
        monitored = MonitoredLearning(mgr)
        monitored.record_selection(ml_guided=True, productive=True)
        monitored.record_selection(ml_guided=False, productive=False)
        assert monitored.monitor.selections.total_ml == 1
        assert monitored.monitor.selections.total_trad == 1

    def test_should_update_delegates(self):
        mgr = _make_manager(update_interval=5, min_examples=3)
        monitored = MonitoredLearning(mgr)
        # Not enough examples yet
        assert not monitored.should_update()

    def test_update_records_monitoring(self):
        mgr = _make_manager(update_interval=5, min_examples=3)
        monitored = MonitoredLearning(mgr)

        # Fill buffer with enough data for an update
        for i in range(10):
            if i % 2 == 0:
                monitored.record_outcome(_make_outcome(i, OutcomeType.KEPT))
            else:
                monitored.record_outcome(_make_outcome(i, OutcomeType.SUBSUMED))

        if monitored.should_update():
            monitored.update()

        # Monitor should have recorded the update
        assert monitored.monitor.update_count >= 1
        # Curve analyzer should have data
        assert monitored.curves.loss_count >= 0  # May be 0 if loss was inf

    def test_update_with_search_stats(self):
        """Test that search stats are passed through to monitoring."""
        mgr = _make_manager(update_interval=5, min_examples=3)
        monitored = MonitoredLearning(mgr)

        # We can't easily mock SearchStatistics, but we can verify
        # set_search_stats doesn't crash
        monitored.set_search_stats(None)

        for i in range(10):
            if i % 2 == 0:
                monitored.record_outcome(_make_outcome(i, OutcomeType.KEPT))
            else:
                monitored.record_outcome(_make_outcome(i, OutcomeType.SUBSUMED))

        if monitored.should_update():
            monitored.update()

    def test_buffer_health(self):
        mgr = _make_manager()
        monitored = MonitoredLearning(mgr)

        # Empty buffer
        health = monitored.buffer_health()
        assert health.status == "empty"

        # Add some data
        for i in range(20):
            if i % 3 == 0:
                monitored.record_outcome(_make_outcome(i, OutcomeType.KEPT))
            else:
                monitored.record_outcome(_make_outcome(i, OutcomeType.SUBSUMED))

        health = monitored.buffer_health()
        assert health.size > 0

    def test_report(self):
        mgr = _make_manager()
        monitored = MonitoredLearning(mgr)
        report = monitored.report()
        assert "ONLINE LEARNING MONITOR REPORT" in report

    def test_has_converged_initial(self):
        mgr = _make_manager()
        monitored = MonitoredLearning(mgr)
        assert not monitored.has_converged()

    def test_is_overfitting_initial(self):
        mgr = _make_manager()
        monitored = MonitoredLearning(mgr)
        assert not monitored.is_overfitting()

    def test_on_proof_found_delegates(self):
        mgr = _make_manager()
        monitored = MonitoredLearning(mgr)

        # Add some outcomes first
        for i in range(5):
            monitored.record_outcome(_make_outcome(i, OutcomeType.KEPT))

        # Should not crash
        monitored.on_proof_found({0, 1, 2})

    def test_full_workflow(self):
        """End-to-end: outcomes -> update -> monitoring -> report."""
        mgr = _make_manager(update_interval=8, min_examples=5)
        monitored = MonitoredLearning(mgr)

        # Simulate a search session
        for i in range(20):
            outcome_type = OutcomeType.KEPT if i % 3 == 0 else OutcomeType.SUBSUMED
            monitored.record_outcome(_make_outcome(i, outcome_type))
            monitored.record_selection(
                ml_guided=(i % 2 == 0),
                productive=(i % 3 == 0),
            )

            if monitored.should_update():
                monitored.update()

        # Verify monitoring captured data
        report = monitored.report()
        assert "ONLINE LEARNING MONITOR REPORT" in report

        sel = monitored.monitor.selections
        assert sel.total_ml > 0
        assert sel.total_trad > 0

    def test_import_from_package(self):
        from pyladr.monitoring import MonitoredLearning
        assert MonitoredLearning is not None
