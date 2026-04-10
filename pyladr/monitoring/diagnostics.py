"""Configurable diagnostic logging for PyLADR components.

Provides fine-grained verbosity control for debugging different
components of the theorem prover without affecting search behavior.

Verbosity levels:
    SILENT  (0) - No diagnostic output
    ERROR   (1) - Errors only
    WARN    (2) - Warnings and errors
    INFO    (3) - Key events (given clause, proof found)
    DEBUG   (4) - Detailed operation logging
    TRACE   (5) - Full trace of all operations

Usage:
    diag = DiagnosticLogger()
    diag.set_level("search", Verbosity.DEBUG)
    diag.set_level("inference", Verbosity.INFO)
    diag.set_level("subsumption", Verbosity.TRACE)

    diag.log("search", Verbosity.DEBUG, "Selected given #%d", given_id)
"""

from __future__ import annotations

import logging
import sys
from enum import IntEnum
from typing import TextIO


class Verbosity(IntEnum):
    """Verbosity levels for diagnostic logging."""

    SILENT = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    DEBUG = 4
    TRACE = 5


# Map Verbosity to Python logging levels
_VERBOSITY_TO_LOGGING = {
    Verbosity.SILENT: logging.CRITICAL + 1,  # Above all levels
    Verbosity.ERROR: logging.ERROR,
    Verbosity.WARN: logging.WARNING,
    Verbosity.INFO: logging.INFO,
    Verbosity.DEBUG: logging.DEBUG,
    Verbosity.TRACE: logging.DEBUG - 5,  # Below DEBUG
}

# Component names and their logger paths
_COMPONENT_LOGGERS: dict[str, str] = {
    "search": "pyladr.search.given_clause",
    "selection": "pyladr.search.selection",
    "inference": "pyladr.inference",
    "resolution": "pyladr.inference.resolution",
    "paramodulation": "pyladr.inference.paramodulation",
    "demodulation": "pyladr.inference.demodulation",
    "subsumption": "pyladr.inference.subsumption",
    "indexing": "pyladr.indexing",
    "parsing": "pyladr.parsing",
    "parallel": "pyladr.parallel",
    "ordering": "pyladr.ordering",
    "monitoring": "pyladr.monitoring",
    "ml_learning": "pyladr.ml.online_learning",
    "ml_selection": "pyladr.search.ml_selection",
}


class DiagnosticLogger:
    """Configurable diagnostic logging for PyLADR components.

    Wraps Python's logging system with component-based verbosity
    control. Each component can have its own verbosity level.

    Does not affect search behavior — only controls what gets logged.

    Usage:
        diag = DiagnosticLogger(output=sys.stderr)
        diag.set_level("search", Verbosity.DEBUG)
        diag.set_level("subsumption", Verbosity.TRACE)

        # Or set all at once:
        diag.set_global_level(Verbosity.INFO)

        # Log a message (only emitted if component level >= message level):
        diag.log("search", Verbosity.DEBUG, "Given #%d weight=%f", gid, wt)
    """

    __slots__ = ("_levels", "_handlers", "_output", "_format")

    def __init__(
        self,
        output: TextIO | None = None,
        fmt: str = "[%(name)s] %(message)s",
    ) -> None:
        self._levels: dict[str, Verbosity] = {}
        self._handlers: dict[str, logging.Handler] = {}
        self._output = output or sys.stderr
        self._format = fmt

    @property
    def components(self) -> list[str]:
        """Available component names."""
        return sorted(_COMPONENT_LOGGERS.keys())

    def set_level(self, component: str, level: Verbosity) -> None:
        """Set verbosity level for a specific component.

        Args:
            component: Component name (e.g., "search", "inference").
            level: Verbosity level to set.
        """
        if component not in _COMPONENT_LOGGERS:
            raise ValueError(
                f"Unknown component {component!r}. "
                f"Available: {', '.join(sorted(_COMPONENT_LOGGERS))}"
            )

        self._levels[component] = level
        logger_name = _COMPONENT_LOGGERS[component]
        logger = logging.getLogger(logger_name)

        py_level = _VERBOSITY_TO_LOGGING.get(level, logging.DEBUG)
        logger.setLevel(py_level)

        # Ensure handler exists
        if component not in self._handlers:
            handler = logging.StreamHandler(self._output)
            handler.setFormatter(logging.Formatter(self._format))
            logger.addHandler(handler)
            self._handlers[component] = handler

    def set_global_level(self, level: Verbosity) -> None:
        """Set verbosity level for all components."""
        for component in _COMPONENT_LOGGERS:
            self.set_level(component, level)

    def get_level(self, component: str) -> Verbosity:
        """Get current verbosity level for a component."""
        return self._levels.get(component, Verbosity.SILENT)

    def log(
        self,
        component: str,
        level: Verbosity,
        msg: str,
        *args: object,
    ) -> None:
        """Log a diagnostic message for a component.

        The message is only emitted if the component's verbosity level
        is >= the message level.
        """
        current_level = self._levels.get(component, Verbosity.SILENT)
        if current_level < level:
            return

        logger_name = _COMPONENT_LOGGERS.get(component, f"pyladr.{component}")
        logger = logging.getLogger(logger_name)
        py_level = _VERBOSITY_TO_LOGGING.get(level, logging.DEBUG)

        if args:
            logger.log(py_level, msg, *args)
        else:
            logger.log(py_level, msg)

    def reset(self) -> None:
        """Remove all diagnostic handlers and reset levels."""
        for component, handler in self._handlers.items():
            logger_name = _COMPONENT_LOGGERS.get(component)
            if logger_name:
                logging.getLogger(logger_name).removeHandler(handler)
        self._handlers.clear()
        self._levels.clear()

    def status(self) -> str:
        """Show current verbosity settings for all configured components."""
        if not self._levels:
            return "No diagnostic levels configured."

        lines = ["Diagnostic verbosity levels:"]
        for component in sorted(self._levels):
            level = self._levels[component]
            lines.append(f"  {component}: {level.name} ({level.value})")
        return "\n".join(lines)
