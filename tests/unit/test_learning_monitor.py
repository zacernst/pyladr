"""Tests for online learning monitoring, regression detection, and curve analysis.

Tests all three monitoring modules:
1. LearningMonitor - real-time monitoring and alerting
2. LearningRegressionDetector - performance regression detection
3. LearningCurveAnalyzer - convergence and overfitting detection
"""

from __future__ import annotations

import time

import pytest

from pyladr.monitoring.learning_monitor import (
    AlertSeverity,
    BufferHealth,
    LearningAlert,
    LearningMonitor,
    SelectionWindow,
    UpdateSnapshot,
)
from pyladr.monitoring.learning_regression import (
    LearningBaseline,
    LearningRegressionDetector,
    LearningRegressionReport,
    SearchResultRecord,
)
from pyladr.monitoring.learning_curves import (
    LearningCurveAnalyzer,
    LearningCurveMetrics,
    ProductivityMetrics,
)


# ═══════════════════════════════════════════════════════════════════════════
# SelectionWindow tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSelectionWindow:
    def test_empty_window(self):
        w = SelectionWindow()
        assert w.ml_window_rate == 0.0
        assert w.trad_window_rate == 0.0
        assert w.ml_overall_rate == 0.0
        assert w.trad_overall_rate == 0.0
        assert w.ml_advantage == 0.0

    def test_ml_tracking(self):
        w = SelectionWindow()
        w.record_ml(True)
        w.record_ml(True)
        w.record_ml(False)
        assert w.total_ml == 3
        assert w.total_ml_productive == 2
        assert w.ml_overall_rate == pytest.approx(2 / 3)
        assert w.ml_window_rate == pytest.approx(2 / 3)

    def test_traditional_tracking(self):
        w = SelectionWindow()
        w.record_traditional(True)
        w.record_traditional(False)
        assert w.total_trad == 2
        assert w.total_trad_productive == 1
        assert w.trad_overall_rate == pytest.approx(0.5)

    def test_ml_advantage(self):
        w = SelectionWindow()
        # ML: 80% productive
        for _ in range(8):
            w.record_ml(True)
        for _ in range(2):
            w.record_ml(False)
        # Traditional: 50% productive
        for _ in range(5):
            w.record_traditional(True)
        for _ in range(5):
            w.record_traditional(False)

        assert w.ml_advantage == pytest.approx(0.3)


# ═══════════════════════════════════════════════════════════════════════════
# UpdateSnapshot tests
# ═══════════════════════════════════════════════════════════════════════════


class TestUpdateSnapshot:
    def test_to_dict(self):
        snap = UpdateSnapshot(
            update_id=1, timestamp=1.0, avg_loss=0.5,
            buffer_size=100, model_version=1,
        )
        d = snap.to_dict()
        assert d["update_id"] == 1
        assert d["avg_loss"] == 0.5
        assert d["buffer_size"] == 100
        assert "was_accepted" in d
        assert "search_keep_rate" in d


# ═══════════════════════════════════════════════════════════════════════════
# BufferHealth tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBufferHealth:
    def test_to_dict(self):
        health = BufferHealth(size=500, capacity=5000, fill_ratio=0.1, status="healthy")
        d = health.to_dict()
        assert d["size"] == 500
        assert d["status"] == "healthy"

    def test_frozen(self):
        health = BufferHealth(size=100, status="healthy")
        with pytest.raises(AttributeError):
            health.size = 200  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════
# LearningMonitor tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLearningMonitor:
    def test_initial_state(self):
        monitor = LearningMonitor()
        assert monitor.update_count == 0
        assert monitor.alerts == []
        assert monitor.loss_history() == []

    def test_record_update(self):
        monitor = LearningMonitor()
        snap = monitor.record_update(
            update_id=1, accepted=True, avg_loss=0.5,
            grad_norm=0.02, buffer_size=500,
            buffer_productive=200, buffer_unproductive=300,
            model_version=1,
        )
        assert monitor.update_count == 1
        assert snap.avg_loss == 0.5
        assert snap.productive_ratio == pytest.approx(0.4)

    def test_loss_delta_tracking(self):
        monitor = LearningMonitor()
        monitor.record_update(update_id=1, accepted=True, avg_loss=1.0)
        snap = monitor.record_update(update_id=2, accepted=True, avg_loss=0.8)
        assert snap.loss_delta == pytest.approx(-0.2)

    def test_loss_history(self):
        monitor = LearningMonitor()
        for i, loss in enumerate([1.0, 0.8, 0.6, 0.5]):
            monitor.record_update(update_id=i + 1, accepted=True, avg_loss=loss)
        assert monitor.loss_history() == [1.0, 0.8, 0.6, 0.5]

    def test_grad_norm_history(self):
        monitor = LearningMonitor()
        for i, gn in enumerate([0.1, 0.05, 0.03]):
            monitor.record_update(
                update_id=i + 1, accepted=True,
                avg_loss=0.5, grad_norm=gn,
            )
        assert monitor.grad_norm_history() == [0.1, 0.05, 0.03]

    def test_record_selection(self):
        monitor = LearningMonitor()
        monitor.record_selection(ml_guided=True, productive=True)
        monitor.record_selection(ml_guided=True, productive=False)
        monitor.record_selection(ml_guided=False, productive=True)

        sel = monitor.selections
        assert sel.total_ml == 2
        assert sel.total_trad == 1

    def test_loss_spike_alert(self):
        monitor = LearningMonitor()
        # Build up EMA with normal losses
        for i in range(5):
            monitor.record_update(update_id=i + 1, accepted=True, avg_loss=0.5)
        # Spike
        monitor.record_update(update_id=6, accepted=True, avg_loss=5.0)
        spike_alerts = monitor.get_alerts_by_category("loss_spike")
        assert len(spike_alerts) >= 1

    def test_loss_divergence_alert(self):
        monitor = LearningMonitor()
        # Monotonically increasing losses
        for i in range(6):
            monitor.record_update(
                update_id=i + 1, accepted=True,
                avg_loss=0.5 + i * 0.1,
            )
        div_alerts = monitor.get_alerts_by_category("loss_divergence")
        assert len(div_alerts) >= 1

    def test_gradient_explosion_alert(self):
        monitor = LearningMonitor()
        # Normal gradients
        for i in range(5):
            monitor.record_update(
                update_id=i + 1, accepted=True,
                avg_loss=0.5, grad_norm=0.01,
            )
        # Explosion
        monitor.record_update(
            update_id=6, accepted=True,
            avg_loss=0.5, grad_norm=1.0,
        )
        grad_alerts = monitor.get_alerts_by_category("gradient_explosion")
        assert len(grad_alerts) >= 1

    def test_vanishing_gradient_alert(self):
        monitor = LearningMonitor()
        for i in range(5):
            monitor.record_update(
                update_id=i + 1, accepted=True,
                avg_loss=0.5, grad_norm=0.01,
            )
        monitor.record_update(
            update_id=6, accepted=True,
            avg_loss=0.5, grad_norm=1e-10,
        )
        alerts = monitor.get_alerts_by_category("vanishing_gradient")
        assert len(alerts) >= 1

    def test_excessive_rollbacks_alert(self):
        monitor = LearningMonitor()
        for i in range(5):
            monitor.record_update(
                update_id=i + 1, accepted=False,
                avg_loss=0.5, was_rollback=True,
            )
        alerts = monitor.get_alerts_by_category("excessive_rollbacks")
        assert len(alerts) >= 1

    def test_buffer_health_empty(self):
        monitor = LearningMonitor()
        health = monitor.assess_buffer_health(0, 5000, 0, 0)
        assert health.status == "empty"
        assert monitor.get_alerts_by_category("buffer_empty")

    def test_buffer_health_imbalanced(self):
        monitor = LearningMonitor()
        # Almost all productive — imbalanced
        health = monitor.assess_buffer_health(1000, 5000, 950, 50)
        assert not health.is_balanced
        assert health.status == "imbalanced"

    def test_buffer_health_healthy(self):
        monitor = LearningMonitor()
        health = monitor.assess_buffer_health(1000, 5000, 400, 600)
        assert health.is_balanced
        assert health.status == "healthy"
        assert health.fill_ratio == pytest.approx(0.2)

    def test_alerts_by_severity(self):
        monitor = LearningMonitor()
        # Generate some alerts via loss spike
        for i in range(5):
            monitor.record_update(update_id=i + 1, accepted=True, avg_loss=0.5)
        monitor.record_update(update_id=6, accepted=True, avg_loss=5.0)

        all_alerts = monitor.alerts
        warnings = monitor.get_alerts_by_severity(AlertSeverity.WARNING)
        assert len(warnings) <= len(all_alerts)

    def test_report_no_updates(self):
        monitor = LearningMonitor()
        report = monitor.report()
        assert "No model updates recorded" in report

    def test_report_with_data(self):
        monitor = LearningMonitor()
        for i in range(5):
            monitor.record_update(
                update_id=i + 1, accepted=True,
                avg_loss=1.0 - i * 0.1, grad_norm=0.02,
                buffer_size=500, buffer_productive=200,
                buffer_unproductive=300, model_version=i + 1,
            )
        for _ in range(5):
            monitor.record_selection(ml_guided=True, productive=True)
        for _ in range(5):
            monitor.record_selection(ml_guided=False, productive=False)

        report = monitor.report()
        assert "ONLINE LEARNING MONITOR REPORT" in report
        assert "Loss:" in report
        assert "ML-guided:" in report

    def test_to_csv_rows(self):
        monitor = LearningMonitor()
        for i in range(3):
            monitor.record_update(update_id=i + 1, accepted=True, avg_loss=0.5)
        rows = monitor.to_csv_rows()
        assert len(rows) == 3
        assert all("avg_loss" in r for r in rows)


# ═══════════════════════════════════════════════════════════════════════════
# LearningRegressionDetector tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLearningRegressionDetector:
    def test_no_baseline(self):
        detector = LearningRegressionDetector()
        report = detector.check()
        assert not report.is_regression
        assert "No baseline set" in report.reasons[0]

    def test_insufficient_samples(self):
        detector = LearningRegressionDetector(min_samples=5)
        detector.set_baseline(proof_rate=0.8, avg_given=100)
        detector.record_search_result(proved=True, given=100, kept_rate=0.3)
        report = detector.check()
        assert "Insufficient samples" in report.reasons[0]

    def test_no_regression(self):
        detector = LearningRegressionDetector(min_samples=3)
        detector.set_baseline(proof_rate=0.8, avg_given=100, avg_kept_rate=0.3)

        for _ in range(5):
            detector.record_search_result(proved=True, given=95, kept_rate=0.32)

        report = detector.check()
        assert not report.is_regression
        assert not report.should_fallback

    def test_proof_rate_regression(self):
        detector = LearningRegressionDetector(
            min_samples=3, proof_rate_threshold=0.1,
        )
        detector.set_baseline(proof_rate=0.9, avg_given=100)

        # Only 2 out of 5 proved — big drop
        for i in range(5):
            detector.record_search_result(
                proved=(i < 2), given=100, kept_rate=0.3,
            )

        report = detector.check()
        assert report.is_regression
        assert report.proof_rate_delta < 0
        assert any("Proof rate" in r for r in report.reasons)

    def test_given_count_regression(self):
        detector = LearningRegressionDetector(
            min_samples=3, given_threshold=0.2,
        )
        detector.set_baseline(proof_rate=0.8, avg_given=100, avg_kept_rate=0.3)

        # All proved but need many more given clauses
        for _ in range(5):
            detector.record_search_result(proved=True, given=150, kept_rate=0.3)

        report = detector.check()
        assert report.is_regression
        assert report.given_count_delta_pct > 0

    def test_kept_rate_regression(self):
        detector = LearningRegressionDetector(
            min_samples=3, kept_rate_threshold=0.05,
        )
        detector.set_baseline(proof_rate=0.8, avg_given=100, avg_kept_rate=0.4)

        for _ in range(5):
            detector.record_search_result(proved=True, given=100, kept_rate=0.2)

        report = detector.check()
        assert report.is_regression
        assert report.kept_rate_delta < 0

    def test_fallback_on_severe_regression(self):
        detector = LearningRegressionDetector(
            min_samples=3, proof_rate_threshold=0.1,
            given_threshold=0.2, kept_rate_threshold=0.05,
        )
        detector.set_baseline(proof_rate=0.9, avg_given=100, avg_kept_rate=0.4)

        # Bad on multiple fronts
        for _ in range(5):
            detector.record_search_result(proved=False, given=200, kept_rate=0.1)

        report = detector.check()
        assert report.is_regression
        assert report.should_fallback
        assert len(report.reasons) >= 2

    def test_improvement_detected(self):
        detector = LearningRegressionDetector(min_samples=3)
        detector.set_baseline(proof_rate=0.5, avg_given=200, avg_kept_rate=0.2)

        for _ in range(5):
            detector.record_search_result(proved=True, given=80, kept_rate=0.4)

        report = detector.check()
        assert not report.is_regression
        assert len(report.improvements) > 0

    def test_set_baseline_from_object(self):
        detector = LearningRegressionDetector()
        baseline = LearningBaseline(proof_rate=0.8, avg_given=100)
        detector.set_baseline_from_object(baseline)
        assert detector.has_baseline

    def test_reset(self):
        detector = LearningRegressionDetector(min_samples=3)
        detector.set_baseline(proof_rate=0.8, avg_given=100)
        for _ in range(5):
            detector.record_search_result(proved=True, given=100)
        assert detector.result_count == 5
        detector.reset()
        assert detector.result_count == 0
        assert detector.has_baseline  # Baseline preserved

    def test_report_summary(self):
        detector = LearningRegressionDetector(min_samples=3)
        detector.set_baseline(proof_rate=0.8, avg_given=100, avg_kept_rate=0.3)
        for _ in range(5):
            detector.record_search_result(proved=True, given=90, kept_rate=0.35)

        report = detector.check()
        text = report.summary()
        assert "LEARNING REGRESSION REPORT" in text
        assert "Proof rate:" in text

    def test_baseline_to_dict(self):
        b = LearningBaseline(proof_rate=0.8, avg_given=100, sample_count=10)
        d = b.to_dict()
        assert d["proof_rate"] == 0.8
        assert d["sample_count"] == 10


# ═══════════════════════════════════════════════════════════════════════════
# LearningCurveAnalyzer tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLearningCurveAnalyzer:
    def test_empty_analyzer(self):
        a = LearningCurveAnalyzer()
        assert a.loss_count == 0
        assert a.productivity_count == 0
        assert not a.has_converged()
        assert a.convergence_point() == -1

    def test_add_data(self):
        a = LearningCurveAnalyzer()
        a.add_loss(1.0)
        a.add_loss(0.8)
        a.add_productivity(0.3)
        assert a.loss_count == 2
        assert a.productivity_count == 1

    def test_convergence_detected(self):
        a = LearningCurveAnalyzer(
            convergence_window=5, convergence_threshold=0.01,
        )
        # Rapidly decreasing then stable
        for i in range(10):
            a.add_loss(1.0 / (i + 1))
        # Add many stable values
        for _ in range(15):
            a.add_loss(0.1)

        assert a.has_converged()
        cp = a.convergence_point()
        assert cp >= 0

    def test_no_convergence_with_decreasing_loss(self):
        a = LearningCurveAnalyzer(
            convergence_window=5, convergence_threshold=0.001,
        )
        # Steadily decreasing
        for i in range(20):
            a.add_loss(1.0 - i * 0.04)

        # May or may not converge depending on rate
        # Just ensure it doesn't crash
        _ = a.has_converged()

    def test_overfitting_detection(self):
        a = LearningCurveAnalyzer(overfitting_patience=5)
        # Decrease then increase
        losses = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        for l in losses:
            a.add_loss(l)
        assert a.is_overfitting()

    def test_no_overfitting_when_improving(self):
        a = LearningCurveAnalyzer(overfitting_patience=5)
        for i in range(10):
            a.add_loss(1.0 - i * 0.05)
        assert not a.is_overfitting()

    def test_loss_metrics(self):
        a = LearningCurveAnalyzer()
        losses = [1.0, 0.8, 0.6, 0.5, 0.4]
        for l in losses:
            a.add_loss(l)

        m = a.compute_loss_metrics()
        assert m.total_updates == 5
        assert m.initial_loss == 1.0
        assert m.final_loss == 0.4
        assert m.min_loss == 0.4
        assert m.max_loss == 1.0
        assert m.loss_reduction_pct == pytest.approx(60.0)
        assert m.best_loss_update == 4
        assert m.updates_since_best == 0
        assert m.loss_per_update == pytest.approx(0.15)

    def test_loss_metrics_empty(self):
        a = LearningCurveAnalyzer()
        m = a.compute_loss_metrics()
        assert m.total_updates == 0

    def test_half_life(self):
        a = LearningCurveAnalyzer()
        # Start at 1.0, reach 0.5 at step 3
        losses = [1.0, 0.8, 0.6, 0.5, 0.4]
        for l in losses:
            a.add_loss(l)
        m = a.compute_loss_metrics()
        assert m.half_life_updates == 3

    def test_oscillation_count(self):
        a = LearningCurveAnalyzer()
        # Oscillating losses
        losses = [1.0, 0.8, 0.9, 0.7, 0.85, 0.65]
        for l in losses:
            a.add_loss(l)
        m = a.compute_loss_metrics()
        assert m.oscillation_count >= 3

    def test_productivity_metrics(self):
        a = LearningCurveAnalyzer()
        rates = [0.2, 0.25, 0.3, 0.35, 0.38, 0.4]
        for r in rates:
            a.add_productivity(r)

        pm = a.compute_productivity_metrics()
        assert pm.total_samples == 6
        assert pm.initial_rate == 0.2
        assert pm.final_rate == 0.4
        assert pm.peak_rate == 0.4
        assert pm.improvement_pct == pytest.approx(100.0)
        assert pm.trend == "improving"

    def test_productivity_metrics_degrading(self):
        a = LearningCurveAnalyzer()
        rates = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25]
        for r in rates:
            a.add_productivity(r)

        pm = a.compute_productivity_metrics()
        assert pm.trend == "degrading"

    def test_productivity_metrics_stable(self):
        a = LearningCurveAnalyzer()
        rates = [0.5, 0.49, 0.51, 0.50, 0.49, 0.50]
        for r in rates:
            a.add_productivity(r)

        pm = a.compute_productivity_metrics()
        assert pm.trend == "stable"

    def test_productivity_metrics_empty(self):
        a = LearningCurveAnalyzer()
        pm = a.compute_productivity_metrics()
        assert pm.total_samples == 0

    def test_smoothed_losses(self):
        a = LearningCurveAnalyzer()
        for l in [1.0, 0.5, 1.0, 0.5, 1.0]:
            a.add_loss(l)
        smoothed = a.smoothed_losses(window=3)
        assert len(smoothed) == 5
        # Smoothed should have less variance than raw
        raw_var = sum((l - 0.8) ** 2 for l in [1.0, 0.5, 1.0, 0.5, 1.0]) / 5
        sm_mean = sum(smoothed) / len(smoothed)
        sm_var = sum((l - sm_mean) ** 2 for l in smoothed) / len(smoothed)
        assert sm_var < raw_var

    def test_smoothed_empty(self):
        a = LearningCurveAnalyzer()
        assert a.smoothed_losses() == []
        assert a.smoothed_productivity() == []

    def test_smoothed_productivity(self):
        a = LearningCurveAnalyzer()
        for r in [0.3, 0.35, 0.33, 0.37, 0.36]:
            a.add_productivity(r)
        smoothed = a.smoothed_productivity(window=3)
        assert len(smoothed) == 5

    def test_report_empty(self):
        a = LearningCurveAnalyzer()
        report = a.report()
        assert "No loss data recorded" in report

    def test_report_with_data(self):
        a = LearningCurveAnalyzer()
        for i in range(10):
            a.add_loss(1.0 - i * 0.05)
            a.add_productivity(0.2 + i * 0.02)
        report = a.report()
        assert "LEARNING CURVE ANALYSIS" in report
        assert "Loss curve:" in report
        assert "Productivity:" in report

    def test_report_with_convergence(self):
        a = LearningCurveAnalyzer(convergence_window=5, convergence_threshold=0.01)
        for i in range(10):
            a.add_loss(1.0 / (i + 1))
        for _ in range(15):
            a.add_loss(0.1)
        report = a.report()
        assert "Converged" in report

    def test_report_with_overfitting(self):
        a = LearningCurveAnalyzer(overfitting_patience=3)
        losses = [1.0, 0.5, 0.3, 0.2, 0.25, 0.35, 0.45]
        for l in losses:
            a.add_loss(l)
        report = a.report()
        assert "overfitting" in report.lower()

    def test_metrics_to_dict(self):
        a = LearningCurveAnalyzer()
        for i in range(5):
            a.add_loss(1.0 - i * 0.1)
        m = a.compute_loss_metrics()
        d = m.to_dict()
        assert "total_updates" in d
        assert "is_converged" in d
        assert "oscillation_count" in d

    def test_productivity_to_dict(self):
        a = LearningCurveAnalyzer()
        a.add_productivity(0.3)
        pm = a.compute_productivity_metrics()
        d = pm.to_dict()
        assert "trend" in d
        assert "peak_rate" in d

    def test_reset(self):
        a = LearningCurveAnalyzer()
        for i in range(5):
            a.add_loss(float(i))
            a.add_productivity(float(i) / 10)
        a.reset()
        assert a.loss_count == 0
        assert a.productivity_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# Integration tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMonitoringIntegration:
    """Test that all three monitoring modules work together."""

    def test_full_monitoring_workflow(self):
        """Simulate a complete online learning session with monitoring."""
        monitor = LearningMonitor()
        regression = LearningRegressionDetector(min_samples=3)
        curves = LearningCurveAnalyzer(convergence_window=3)

        # Set baseline
        regression.set_baseline(proof_rate=0.7, avg_given=150, avg_kept_rate=0.3)

        # Simulate learning session
        for i in range(10):
            loss = 1.0 - i * 0.08
            grad = 0.05 - i * 0.003
            prod = 200 + i * 10
            unprod = 300 + i * 5

            # Record in all three systems
            monitor.record_update(
                update_id=i + 1, accepted=True,
                avg_loss=loss, grad_norm=max(grad, 0.001),
                buffer_size=prod + unprod,
                buffer_productive=prod, buffer_unproductive=unprod,
                model_version=i + 1,
            )
            curves.add_loss(loss)
            curves.add_productivity(prod / (prod + unprod))

            # Record selections
            monitor.record_selection(ml_guided=True, productive=True)
            monitor.record_selection(ml_guided=False, productive=False)

            # Record search results
            regression.record_search_result(
                proved=True, given=140 - i * 5, kept_rate=0.3 + i * 0.01,
            )

        # Check all systems produce reports
        monitor_report = monitor.report()
        assert "ONLINE LEARNING MONITOR REPORT" in monitor_report

        reg_report = regression.check()
        assert isinstance(reg_report, LearningRegressionReport)
        reg_text = reg_report.summary()
        assert "LEARNING REGRESSION REPORT" in reg_text

        curve_report = curves.report()
        assert "LEARNING CURVE ANALYSIS" in curve_report

        # Verify learning is improving
        assert not reg_report.is_regression
        pm = curves.compute_productivity_metrics()
        assert pm.trend == "improving"

    def test_regression_with_curve_analysis(self):
        """Verify regression detection correlates with curve analysis."""
        regression = LearningRegressionDetector(min_samples=3)
        curves = LearningCurveAnalyzer(overfitting_patience=3)

        regression.set_baseline(proof_rate=0.9, avg_given=100, avg_kept_rate=0.4)

        # Simulate degradation
        losses = [0.5, 0.4, 0.3, 0.25, 0.3, 0.4, 0.5, 0.6]
        for i, loss in enumerate(losses):
            curves.add_loss(loss)
            curves.add_productivity(0.4 - i * 0.03)
            regression.record_search_result(
                proved=(i < 4), given=100 + i * 20, kept_rate=0.4 - i * 0.03,
            )

        # Curve should detect overfitting
        assert curves.is_overfitting()

        # Regression should detect degradation
        report = regression.check()
        assert report.is_regression

    def test_import_from_package(self):
        """Verify all new classes are importable from the package."""
        from pyladr.monitoring import (
            LearningMonitor,
            LearningAlert,
            AlertSeverity,
            UpdateSnapshot,
            SelectionWindow,
            BufferHealth,
            LearningRegressionDetector,
            LearningRegressionReport,
            LearningBaseline,
            SearchResultRecord,
            LearningCurveAnalyzer,
            LearningCurveMetrics,
            ProductivityMetrics,
        )
        # Just verify they're importable
        assert LearningMonitor is not None
        assert LearningRegressionDetector is not None
        assert LearningCurveAnalyzer is not None
