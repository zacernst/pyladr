"""Shared test fixtures for pyladr tests."""

from __future__ import annotations

import os
from pathlib import Path
import pytest

# ── Path fixtures ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

C_PROVER9_BIN = PROJECT_ROOT / "bin" / "prover9"
C_EXAMPLES_DIR = PROJECT_ROOT / "prover9.examples"
TEST_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TEST_INPUTS_DIR = TEST_FIXTURES_DIR / "inputs"
C_REFERENCE_DIR = TEST_FIXTURES_DIR / "c_reference"


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
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line(
        "markers", "cross_validation: tests that compare Python vs C"
    )
    config.addinivalue_line("markers", "property: property-based tests")
    config.addinivalue_line("markers", "benchmark: performance benchmark tests")
    config.addinivalue_line(
        "markers", "requires_models: tests that need core data models"
    )
