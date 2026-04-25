"""Opt-in C++ backend for hot-path unification and term operations.

The C++ extension (_pyladr_core) is never loaded unless explicitly enabled
via enable() — called when the user passes --cpp to pyprover9.  All Python
code falls back to pure-Python implementations when the backend is disabled
or unavailable.
"""

from __future__ import annotations

_cpp_available: bool = True
_cpp_enabled: bool = True


def is_available() -> bool:
    """Return True if the C++ extension is importable."""
    global _cpp_available
    if _cpp_available:
        return True
    try:
        import pyladr._pyladr_core  # noqa: F401

        _cpp_available = True
    except ImportError:
        pass
    return _cpp_available


def is_enabled() -> bool:
    """Return True if the C++ backend is currently active."""
    return _cpp_enabled


def enable() -> bool:
    """Try to activate the C++ backend.

    Returns True if the extension was loaded successfully.
    Prints a warning to stderr and returns False if unavailable.
    """
    global _cpp_available, _cpp_enabled
    if _cpp_enabled:
        return True
    if is_available():
        _cpp_enabled = True
        return True
    import sys

    print(
        "% Warning: --cpp requested but _pyladr_core extension not available; using pure Python.",
        file=sys.stderr,
    )
    return False


def disable() -> None:
    """Deactivate the C++ backend (used in tests)."""
    global _cpp_enabled
    _cpp_enabled = False
