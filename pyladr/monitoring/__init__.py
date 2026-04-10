"""Monitoring and profiling tools for PyLADR.

Non-intrusive performance monitoring, profiling, memory tracking,
and regression detection for the theorem proving engine.

All tools are designed to observe without modifying search behavior,
maintaining perfect compatibility with C Prover9 output.
"""

from pyladr.monitoring.profiler import (
    PhaseTimer,
    SearchProfiler,
    ProfileReport,
)
from pyladr.monitoring.memory_monitor import MemoryMonitor, MemorySnapshot
from pyladr.monitoring.search_analyzer import SearchAnalyzer, IterationSnapshot
from pyladr.monitoring.diagnostics import DiagnosticLogger, Verbosity
from pyladr.monitoring.regression import (
    RegressionDetector,
    PerformanceBaseline,
    RegressionReport,
)
from pyladr.monitoring.comparison import ComparisonReport, compare_search_results
from pyladr.monitoring.learning_monitor import (
    LearningMonitor,
    LearningAlert,
    AlertSeverity,
    UpdateSnapshot,
    SelectionWindow,
    BufferHealth,
)
from pyladr.monitoring.learning_regression import (
    LearningRegressionDetector,
    LearningRegressionReport,
    LearningBaseline,
    SearchResultRecord,
)
from pyladr.monitoring.learning_curves import (
    LearningCurveAnalyzer,
    LearningCurveMetrics,
    ProductivityMetrics,
)
from pyladr.monitoring.learning_bridge import MonitoredLearning
from pyladr.monitoring.ml_memory import (
    MLMemoryTracker,
    MLMemorySnapshot,
    MemoryBudget,
)
from pyladr.monitoring.health import (
    HealthChecker,
    SystemHealth,
    ComponentHealth,
    HealthStatus,
    ProductionConfig,
    PRODUCTION_DEFAULTS,
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerAction,
)

__all__ = [
    "PhaseTimer",
    "SearchProfiler",
    "ProfileReport",
    "MemoryMonitor",
    "MemorySnapshot",
    "SearchAnalyzer",
    "IterationSnapshot",
    "DiagnosticLogger",
    "Verbosity",
    "RegressionDetector",
    "PerformanceBaseline",
    "RegressionReport",
    "ComparisonReport",
    "compare_search_results",
    # Online learning monitoring
    "LearningMonitor",
    "LearningAlert",
    "AlertSeverity",
    "UpdateSnapshot",
    "SelectionWindow",
    "BufferHealth",
    # Learning regression detection
    "LearningRegressionDetector",
    "LearningRegressionReport",
    "LearningBaseline",
    "SearchResultRecord",
    # Learning curve analysis
    "LearningCurveAnalyzer",
    "LearningCurveMetrics",
    "ProductivityMetrics",
    # Integration bridge
    "MonitoredLearning",
    # ML memory tracking
    "MLMemoryTracker",
    "MLMemorySnapshot",
    "MemoryBudget",
    # Production health checks
    "HealthChecker",
    "SystemHealth",
    "ComponentHealth",
    "HealthStatus",
    "ProductionConfig",
    "PRODUCTION_DEFAULTS",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerState",
    "CircuitBreakerAction",
]
