"""Tests for the pyladr.monitoring package.

Tests all monitoring and profiling tools to ensure they:
1. Correctly collect and report metrics
2. Are non-intrusive (don't modify search state)
3. Handle edge cases gracefully
"""

from __future__ import annotations

import json
import time
import tempfile
from pathlib import Path

import pytest

from pyladr.monitoring.profiler import SearchProfiler, PhaseTimer, ProfileReport
from pyladr.monitoring.memory_monitor import MemoryMonitor, MemorySnapshot
from pyladr.monitoring.search_analyzer import SearchAnalyzer, IterationSnapshot
from pyladr.monitoring.diagnostics import DiagnosticLogger, Verbosity
from pyladr.monitoring.regression import (
    RegressionDetector,
    PerformanceBaseline,
    RegressionReport,
)
from pyladr.monitoring.comparison import ComparisonReport, compare_search_results
from pyladr.search.state import SearchState
from pyladr.search.statistics import SearchStatistics


# ── PhaseTimer tests ─────────────────────────────────────────────────────


class TestPhaseTimer:
    def test_basic_timing(self):
        timer = PhaseTimer(name="test")
        timer.begin()
        time.sleep(0.01)
        elapsed = timer.end()
        assert elapsed > 0
        assert timer.call_count == 1
        assert timer.total_seconds > 0
        assert timer.min_seconds > 0
        assert timer.max_seconds > 0

    def test_multiple_calls(self):
        timer = PhaseTimer(name="multi")
        for _ in range(3):
            timer.begin()
            timer.end()
        assert timer.call_count == 3
        assert timer.avg_seconds >= 0

    def test_no_calls(self):
        timer = PhaseTimer(name="empty")
        assert timer.call_count == 0
        assert timer.avg_seconds == 0.0
        assert "no calls" in timer.summary()

    def test_summary_format(self):
        timer = PhaseTimer(name="test_phase")
        timer.begin()
        timer.end()
        summary = timer.summary()
        assert "test_phase" in summary
        assert "total" in summary
        assert "avg=" in summary


# ── SearchProfiler tests ────────────────────────────────────────────────


class TestSearchProfiler:
    def test_basic_profiling(self):
        profiler = SearchProfiler()
        profiler.start()

        profiler.begin_phase("phase_a")
        profiler.end_phase("phase_a")

        profiler.begin_phase("phase_b")
        profiler.end_phase("phase_b")

        report = profiler.report()
        assert "phase_a" in report.phases
        assert "phase_b" in report.phases
        assert report.total_seconds >= 0

    def test_context_manager(self):
        profiler = SearchProfiler()
        profiler.start()

        with profiler.phase("ctx_phase"):
            time.sleep(0.001)

        report = profiler.report()
        assert "ctx_phase" in report.phases
        assert report.phases["ctx_phase"].call_count == 1

    def test_disabled_profiler(self):
        profiler = SearchProfiler(enabled=False)
        profiler.start()
        profiler.begin_phase("test")
        elapsed = profiler.end_phase("test")
        assert elapsed == 0.0

    def test_iteration_counting(self):
        profiler = SearchProfiler()
        profiler.start()
        for _ in range(5):
            profiler.increment_iteration()
        report = profiler.report()
        assert report.iteration_count == 5

    def test_custom_counters(self):
        profiler = SearchProfiler()
        profiler.increment_counter("resolutions", 10)
        profiler.increment_counter("resolutions", 5)
        profiler.increment_counter("paramodulations", 3)
        report = profiler.report()
        assert report.custom_counters["resolutions"] == 15
        assert report.custom_counters["paramodulations"] == 3

    def test_report_to_dict(self):
        profiler = SearchProfiler()
        profiler.start()
        with profiler.phase("test"):
            pass
        profiler.increment_counter("count", 1)
        report = profiler.report()
        d = report.to_dict()
        assert "total_seconds" in d
        assert "phases" in d
        assert "test" in d["phases"]
        assert "custom_counters" in d

    def test_report_summary(self):
        profiler = SearchProfiler()
        profiler.start()
        with profiler.phase("inference"):
            time.sleep(0.001)
        report = profiler.report()
        summary = report.summary()
        assert "SEARCH PROFILING REPORT" in summary
        assert "inference" in summary

    def test_reset(self):
        profiler = SearchProfiler()
        profiler.start()
        profiler.begin_phase("test")
        profiler.end_phase("test")
        profiler.increment_counter("x")
        profiler.reset()
        report = profiler.report()
        assert len(report.phases) == 0
        assert report.iteration_count == 0


# ── MemoryMonitor tests ─────────────────────────────────────────────────


class TestMemoryMonitor:
    def _make_state(self) -> SearchState:
        """Create a minimal SearchState for testing."""
        from pyladr.core.clause import Clause
        state = SearchState()
        # Add some clauses to usable and sos
        for i in range(5):
            c = Clause(literals=())
            state.assign_clause_id(c)
            state.usable.append(c)
        for i in range(10):
            c = Clause(literals=())
            state.assign_clause_id(c)
            state.sos.append(c)
        return state

    def test_snapshot(self):
        state = self._make_state()
        monitor = MemoryMonitor()
        snap = monitor.snapshot(state, iteration=0)
        assert snap.usable_count == 5
        assert snap.sos_count == 10
        assert snap.total_active_clauses == 15
        assert snap.clause_ids_assigned == 15

    def test_multiple_snapshots(self):
        state = self._make_state()
        monitor = MemoryMonitor(interval=1)
        for i in range(3):
            assert monitor.should_snapshot(i)
            monitor.snapshot(state, iteration=i)
        assert len(monitor.snapshots) == 3

    def test_interval_check(self):
        monitor = MemoryMonitor(interval=5)
        assert monitor.should_snapshot(0)
        assert not monitor.should_snapshot(1)
        assert not monitor.should_snapshot(4)
        assert monitor.should_snapshot(5)
        assert monitor.should_snapshot(10)

    def test_disabled_interval(self):
        monitor = MemoryMonitor(interval=0)
        assert not monitor.should_snapshot(0)
        assert not monitor.should_snapshot(100)

    def test_report(self):
        state = self._make_state()
        monitor = MemoryMonitor()
        monitor.snapshot(state, iteration=0)
        monitor.snapshot(state, iteration=10)
        report = monitor.report()
        assert "MEMORY USAGE REPORT" in report
        assert "usable=" in report

    def test_empty_report(self):
        monitor = MemoryMonitor()
        assert "No memory snapshots" in monitor.report()

    def test_snapshot_to_dict(self):
        snap = MemorySnapshot(
            iteration=5,
            usable_count=10,
            sos_count=20,
        )
        d = snap.to_dict()
        assert d["iteration"] == 5
        assert d["usable_count"] == 10

    def test_csv_export(self):
        state = self._make_state()
        monitor = MemoryMonitor()
        monitor.snapshot(state, 0)
        monitor.snapshot(state, 1)
        rows = monitor.to_csv_rows()
        assert len(rows) == 2
        assert "usable_count" in rows[0]

    def test_peak_rss(self):
        state = self._make_state()
        monitor = MemoryMonitor()
        monitor.snapshot(state)
        # Peak RSS should be non-negative (may be 0 if resource module unavailable)
        assert monitor.peak_rss_bytes >= 0
        assert monitor.peak_rss_mb >= 0.0


# ── SearchAnalyzer tests ────────────────────────────────────────────────


class TestSearchAnalyzer:
    def _make_stats(self, given=0, generated=0, kept=0, subsumed=0) -> SearchStatistics:
        stats = SearchStatistics()
        stats.given = given
        stats.generated = generated
        stats.kept = kept
        stats.subsumed = subsumed
        stats.start_time = time.monotonic() - 1.0  # 1 second ago
        return stats

    def test_record_iteration(self):
        analyzer = SearchAnalyzer()
        stats = self._make_stats(given=1, generated=10, kept=3)
        snap = analyzer.record_iteration(stats)
        assert snap.given == 1
        assert snap.generated == 10
        assert snap.kept == 3
        assert analyzer.iteration_count == 1

    def test_delta_computation(self):
        analyzer = SearchAnalyzer()
        stats1 = self._make_stats(given=1, generated=10, kept=3, subsumed=5)
        analyzer.record_iteration(stats1)

        stats2 = self._make_stats(given=2, generated=25, kept=7, subsumed=12)
        snap = analyzer.record_iteration(stats2)
        assert snap.delta_generated == 15
        assert snap.delta_kept == 4
        assert snap.delta_subsumed == 7

    def test_keep_rate(self):
        analyzer = SearchAnalyzer()
        stats1 = self._make_stats(generated=0, kept=0)
        analyzer.record_iteration(stats1)

        stats2 = self._make_stats(generated=100, kept=25)
        snap = analyzer.record_iteration(stats2)
        assert snap.keep_rate == pytest.approx(0.25)

    def test_with_state(self):
        from pyladr.core.clause import Clause
        state = SearchState()
        for _ in range(5):
            c = Clause(literals=())
            state.assign_clause_id(c)
            state.usable.append(c)

        analyzer = SearchAnalyzer()
        stats = self._make_stats(given=1, generated=10, kept=3)
        snap = analyzer.record_iteration(stats, state)
        assert snap.usable_size == 5

    def test_identify_hotspots(self):
        analyzer = SearchAnalyzer()
        for i in range(10):
            stats = self._make_stats(
                given=i + 1,
                generated=(i + 1) * (10 if i == 5 else 1),
                kept=(i + 1),
            )
            analyzer.record_iteration(stats)

        hotspots = analyzer.identify_hotspots(3)
        assert len(hotspots) <= 3

    def test_compute_rates(self):
        analyzer = SearchAnalyzer()
        stats1 = self._make_stats(given=0, generated=0, kept=0)
        analyzer.record_iteration(stats1)
        time.sleep(0.01)
        stats2 = self._make_stats(given=10, generated=100, kept=30)
        analyzer.record_iteration(stats2)

        rates = analyzer.compute_rates()
        assert "overall_generated_per_sec" in rates
        assert rates["overall_generated_per_sec"] > 0

    def test_empty_analyzer(self):
        analyzer = SearchAnalyzer()
        assert analyzer.iteration_count == 0
        assert analyzer.identify_hotspots() == []
        assert analyzer.compute_rates() == {}
        assert "No search iterations" in analyzer.report()

    def test_report(self):
        analyzer = SearchAnalyzer()
        for i in range(5):
            stats = self._make_stats(
                given=i + 1,
                generated=(i + 1) * 10,
                kept=(i + 1) * 3,
                subsumed=(i + 1) * 2,
            )
            analyzer.record_iteration(stats)

        report = analyzer.report()
        assert "SEARCH ANALYSIS REPORT" in report
        assert "Total iterations: 5" in report

    def test_csv_export(self):
        analyzer = SearchAnalyzer()
        stats = self._make_stats(given=1, generated=10, kept=3)
        analyzer.record_iteration(stats)
        rows = analyzer.to_csv_rows()
        assert len(rows) == 1
        assert "generated" in rows[0]


# ── DiagnosticLogger tests ──────────────────────────────────────────────


class TestDiagnosticLogger:
    def test_set_level(self):
        diag = DiagnosticLogger()
        diag.set_level("search", Verbosity.DEBUG)
        assert diag.get_level("search") == Verbosity.DEBUG

    def test_unknown_component(self):
        diag = DiagnosticLogger()
        with pytest.raises(ValueError, match="Unknown component"):
            diag.set_level("nonexistent", Verbosity.INFO)

    def test_global_level(self):
        diag = DiagnosticLogger()
        diag.set_global_level(Verbosity.INFO)
        for component in diag.components:
            assert diag.get_level(component) == Verbosity.INFO
        diag.reset()

    def test_log_message(self, capsys):
        import io
        output = io.StringIO()
        diag = DiagnosticLogger(output=output)
        diag.set_level("search", Verbosity.DEBUG)
        diag.log("search", Verbosity.INFO, "test message %d", 42)
        diag.reset()
        # Message should have been logged to the output
        text = output.getvalue()
        assert "test message 42" in text

    def test_log_suppressed(self):
        import io
        output = io.StringIO()
        diag = DiagnosticLogger(output=output)
        diag.set_level("search", Verbosity.ERROR)
        diag.log("search", Verbosity.DEBUG, "should not appear")
        diag.reset()
        assert "should not appear" not in output.getvalue()

    def test_silent_level(self):
        diag = DiagnosticLogger()
        # Default level is SILENT
        assert diag.get_level("search") == Verbosity.SILENT

    def test_components_list(self):
        diag = DiagnosticLogger()
        components = diag.components
        assert "search" in components
        assert "inference" in components
        assert "subsumption" in components

    def test_status(self):
        diag = DiagnosticLogger()
        assert "No diagnostic levels" in diag.status()
        diag.set_level("search", Verbosity.DEBUG)
        status = diag.status()
        assert "search" in status
        assert "DEBUG" in status
        diag.reset()

    def test_reset(self):
        diag = DiagnosticLogger()
        diag.set_level("search", Verbosity.DEBUG)
        diag.reset()
        assert diag.get_level("search") == Verbosity.SILENT


# ── PerformanceBaseline tests ───────────────────────────────────────────


class TestPerformanceBaseline:
    def test_from_stats(self):
        stats = SearchStatistics()
        stats.start_time = time.monotonic() - 2.0
        stats.given = 10
        stats.generated = 100
        stats.kept = 30
        stats.subsumed = 50

        baseline = PerformanceBaseline.from_stats(stats, label="test")
        assert baseline.given == 10
        assert baseline.generated == 100
        assert baseline.kept == 30
        assert baseline.label == "test"
        assert baseline.elapsed_seconds > 0

    def test_save_and_load(self, tmp_path):
        baseline = PerformanceBaseline(
            name="test_problem",
            label="v1.0",
            given=10,
            generated=100,
            kept=30,
            subsumed=50,
            elapsed_seconds=1.5,
            proof_found=True,
        )

        filepath = tmp_path / "baseline.json"
        baseline.save(filepath)

        loaded = PerformanceBaseline.load(filepath)
        assert loaded.name == "test_problem"
        assert loaded.given == 10
        assert loaded.generated == 100
        assert loaded.proof_found is True

    def test_to_dict(self):
        baseline = PerformanceBaseline(name="test", given=5)
        d = baseline.to_dict()
        assert d["name"] == "test"
        assert d["given"] == 5
        assert d["_type"] == "PerformanceBaseline"


# ── RegressionDetector tests ───────────────────────────────────────────


class TestRegressionDetector:
    def _make_baseline(self, **kwargs) -> PerformanceBaseline:
        defaults = {
            "name": "test",
            "label": "baseline",
            "elapsed_seconds": 1.0,
            "given": 10,
            "generated": 100,
            "kept": 30,
            "subsumed": 50,
            "generated_per_sec": 100.0,
            "proof_found": True,
            "exit_code": 1,
        }
        defaults.update(kwargs)
        return PerformanceBaseline(**defaults)

    def _make_stats(self, **kwargs) -> SearchStatistics:
        stats = SearchStatistics()
        elapsed = kwargs.pop("elapsed", 1.0)
        stats.start_time = time.monotonic() - elapsed
        for k, v in kwargs.items():
            setattr(stats, k, v)
        return stats

    def test_no_regression(self):
        baseline = self._make_baseline()
        detector = RegressionDetector(baseline)
        stats = self._make_stats(
            given=10, generated=100, kept=30, subsumed=50, elapsed=1.0,
        )
        report = detector.check_stats(stats)
        assert not report.has_regressions

    def test_timing_regression(self):
        baseline = self._make_baseline(elapsed_seconds=1.0)
        detector = RegressionDetector(baseline, time_threshold=0.20)
        # 50% slower
        stats = self._make_stats(
            given=10, generated=100, kept=30, subsumed=50, elapsed=1.5,
        )
        report = detector.check_stats(stats)
        # Should detect timing regression
        timing_items = [i for i in report.items if i.metric == "elapsed_seconds"]
        assert len(timing_items) == 1
        assert timing_items[0].change_pct > 0

    def test_clause_count_change(self):
        baseline = self._make_baseline(given=10, generated=100, kept=30)
        detector = RegressionDetector(baseline)
        stats = self._make_stats(
            given=12, generated=120, kept=35, subsumed=50, elapsed=1.0,
        )
        report = detector.check_stats(stats)
        # Clause count changes should be warnings
        changed = [i for i in report.items if i.severity == "warning"]
        assert len(changed) > 0

    def test_report_summary(self):
        baseline = self._make_baseline()
        detector = RegressionDetector(baseline)
        stats = self._make_stats(
            given=10, generated=100, kept=30, subsumed=50, elapsed=1.0,
        )
        report = detector.check_stats(stats)
        summary = report.summary()
        assert "PERFORMANCE REGRESSION REPORT" in summary

    def test_empty_report(self):
        report = RegressionReport(baseline_label="test")
        assert not report.has_regressions
        assert report.critical_count == 0


# ── ComparisonReport tests ──────────────────────────────────────────────


class TestComparisonReport:
    def test_equivalent_results(self):
        c_stats = {
            "given": 10, "generated": 100, "kept": 30,
            "subsumed": 50, "seconds": 0.1, "proved": True,
        }
        py_stats = {
            "given": 10, "generated": 100, "kept": 30,
            "subsumed": 50, "seconds": 0.5, "proved": True,
        }
        report = compare_search_results(c_stats, py_stats, "test")
        assert report.search_equivalent
        assert report.slowdown_ratio == pytest.approx(5.0)

    def test_different_results(self):
        c_stats = {
            "given": 10, "generated": 100, "kept": 30,
            "subsumed": 50, "seconds": 0.1, "proved": True,
        }
        py_stats = {
            "given": 12, "generated": 110, "kept": 35,
            "subsumed": 55, "seconds": 0.5, "proved": True,
        }
        report = compare_search_results(c_stats, py_stats, "test")
        assert not report.search_equivalent
        assert len(report.notes) > 0

    def test_different_proof_status(self):
        c_stats = {"given": 10, "generated": 100, "kept": 30,
                    "subsumed": 50, "seconds": 0.1, "proved": True}
        py_stats = {"given": 10, "generated": 100, "kept": 30,
                    "subsumed": 50, "seconds": 0.5, "proved": False}
        report = compare_search_results(c_stats, py_stats, "test")
        assert not report.search_equivalent
        assert any("CRITICAL" in n for n in report.notes)

    def test_summary_format(self):
        c_stats = {"given": 10, "generated": 100, "kept": 30,
                    "subsumed": 50, "seconds": 0.1, "proved": True}
        py_stats = {"given": 10, "generated": 100, "kept": 30,
                    "subsumed": 50, "seconds": 0.3, "proved": True}
        report = compare_search_results(c_stats, py_stats, "x2")
        summary = report.summary()
        assert "C vs Python" in summary
        assert "x2" in summary

    def test_to_dict(self):
        report = ComparisonReport(
            problem="test",
            c_given=10,
            py_given=10,
            search_equivalent=True,
        )
        d = report.to_dict()
        assert d["problem"] == "test"
        assert d["search_equivalent"] is True

    def test_zero_times(self):
        c_stats = {"given": 1, "generated": 5, "kept": 2,
                    "subsumed": 1, "seconds": 0.0, "proved": True}
        py_stats = {"given": 1, "generated": 5, "kept": 2,
                    "subsumed": 1, "seconds": 0.0, "proved": True}
        report = compare_search_results(c_stats, py_stats)
        assert report.slowdown_ratio == 1.0
