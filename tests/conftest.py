"""Shared test fixtures for pyladr tests."""

from __future__ import annotations

import os
from pathlib import Path
import pytest

# ── Path fixtures ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

C_PROVER9_BIN = PROJECT_ROOT / "reference-prover9" / "bin" / "prover9"
C_EXAMPLES_DIR = PROJECT_ROOT / "prover9.examples"
TEST_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TEST_INPUTS_DIR = TEST_FIXTURES_DIR / "inputs"
C_REFERENCE_DIR = TEST_FIXTURES_DIR / "c_reference"


class ConfigurationError(RuntimeError):
    """Raised when test infrastructure is misconfigured.

    Distinct from test failure: signals a setup/env bug that must be fixed,
    not a test assertion that didn't hold. Cycle 5 hid 175 cross-validation
    tests because a wrong path resolved to skipif(True) — loud-fail prevents
    that class of silent regression.
    """


def require_c_binary(bin_path: Path | None = None) -> Path:
    """Return the C Prover9 binary path, raising ConfigurationError if missing.

    Preferred over silent skip at call sites that genuinely need the binary.
    For legitimate opt-out on machines without the binary, set the env var
    PYLADR_ALLOW_MISSING_C_BINARY=1 before running pytest.
    """
    path = bin_path or C_PROVER9_BIN
    if not path.exists():
        raise ConfigurationError(
            f"C Prover9 binary not found at {path}. "
            f"Expected location: reference-prover9/bin/prover9 "
            f"(build with 'make all' from reference-prover9/). "
            f"Set PYLADR_ALLOW_MISSING_C_BINARY=1 to opt out (dev only)."
        )
    if not os.access(path, os.X_OK):
        raise ConfigurationError(
            f"C Prover9 binary at {path} is not executable. "
            f"Check permissions (chmod +x)."
        )
    return path


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def c_examples_dir() -> Path:
    """Return the path to the C Prover9 example inputs."""
    return C_EXAMPLES_DIR


@pytest.fixture
def test_inputs_dir() -> Path:
    """Return the path to our custom test input files."""
    return TEST_INPUTS_DIR


@pytest.fixture
def c_reference_dir() -> Path:
    """Return the path to captured C reference outputs."""
    return C_REFERENCE_DIR


# ── C binary availability ───────────────────────────────────────────────────

def c_binary_available() -> bool:
    """Check if the C Prover9 binary is built and executable."""
    return C_PROVER9_BIN.exists() and os.access(C_PROVER9_BIN, os.X_OK)


requires_c_binary = pytest.mark.skipif(
    not c_binary_available(),
    reason="C Prover9 binary not found (run 'make all' to build)",
)

# ── Standard LADR input strings ─────────────────────────────────────────────


@pytest.fixture
def simple_group_input() -> str:
    """Group theory: prove commutativity from x*x=e."""
    return """\
formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
  x * x = e.
end_of_list.

formulas(goals).
  x * y = y * x.
end_of_list.
"""


@pytest.fixture
def identity_input() -> str:
    """Trivial: prove e*e=e from left identity."""
    return """\
formulas(sos).
  e * x = x.
end_of_list.

formulas(goals).
  e * e = e.
end_of_list.
"""


@pytest.fixture
def lattice_absorption_input() -> str:
    """Lattice: prove idempotence from absorption laws."""
    return """\
formulas(sos).
  x ^ y = y ^ x.
  x v y = y v x.
  (x ^ y) ^ z = x ^ (y ^ z).
  (x v y) v z = x v (y v z).
  x ^ (x v y) = x.
  x v (x ^ y) = x.
end_of_list.

formulas(goals).
  x ^ x = x.
end_of_list.
"""


# ── Pytest configuration ────────────────────────────────────────────────────


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers and validate test infrastructure."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line(
        "markers", "cross_validation: tests that compare Python vs C"
    )
    config.addinivalue_line("markers", "property: property-based tests")
    config.addinivalue_line("markers", "benchmark: performance benchmark tests")
    config.addinivalue_line(
        "markers", "requires_models: tests that need core data models"
    )

    # §4.4 loud-fail infra guard (cycle 6): abort session loudly if the C
    # reference binary is missing rather than silently skipping tests.
    if os.environ.get("PYLADR_ALLOW_MISSING_C_BINARY") != "1":
        require_c_binary()
