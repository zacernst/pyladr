"""Tests for the §4.4 loud-fail ConfigurationError guard.

Cycle 5 hid 175 cross-validation tests behind a silent skipif because the
C Prover9 binary path was wrong (resolved to PROJECT_ROOT/bin/prover9 instead
of PROJECT_ROOT/reference-prover9/bin/prover9). Cycle 6 replaced the silent
skip with a loud ConfigurationError. These tests verify the guard fires on
misconfigured paths and succeeds on the real path.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

import pytest

from tests.conftest import (
    C_PROVER9_BIN,
    ConfigurationError,
    require_c_binary,
)


class TestConfigurationErrorGuard:
    """The loud-fail guard must raise on missing/bad binaries."""

    def test_default_path_points_to_reference_prover9_directory(self) -> None:
        # Regression test for cycle 5's hidden skip: the path must resolve
        # to reference-prover9/bin/prover9, not bin/prover9.
        assert C_PROVER9_BIN.parts[-3:] == ("reference-prover9", "bin", "prover9")

    def test_default_binary_exists_and_is_executable(self) -> None:
        # Sanity check that the fix works on this machine: the configured
        # path actually resolves to a real, executable binary.
        assert C_PROVER9_BIN.exists(), f"{C_PROVER9_BIN} missing"
        assert os.access(C_PROVER9_BIN, os.X_OK), f"{C_PROVER9_BIN} not executable"

    def test_guard_returns_path_when_binary_present(self) -> None:
        returned = require_c_binary()
        assert returned == C_PROVER9_BIN

    def test_guard_raises_configuration_error_on_missing_binary(self) -> None:
        nonexistent = Path("/tmp/definitely-not-a-prover9-binary-cycle6")
        assert not nonexistent.exists()

        with pytest.raises(ConfigurationError) as excinfo:
            require_c_binary(nonexistent)

        # Error message must be actionable: point at the expected location
        # and hint at the build step, not a generic "file not found".
        msg = str(excinfo.value)
        assert "C Prover9 binary not found" in msg
        assert str(nonexistent) in msg
        assert "reference-prover9/bin/prover9" in msg

    def test_guard_raises_configuration_error_on_non_executable_binary(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"not a binary")
            tmp_path = Path(tmp.name)
        try:
            # Strip execute bits.
            tmp_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            with pytest.raises(ConfigurationError) as excinfo:
                require_c_binary(tmp_path)
            assert "not executable" in str(excinfo.value)
        finally:
            tmp_path.unlink()

    def test_configuration_error_is_not_a_test_failure(self) -> None:
        # ConfigurationError signals infra misconfig, distinct from AssertionError
        # (test failure) or FileNotFoundError (ordinary missing file). Tests that
        # depend on the guard should get a loud distinct error type.
        assert issubclass(ConfigurationError, RuntimeError)
        assert not issubclass(ConfigurationError, AssertionError)
        assert not issubclass(ConfigurationError, FileNotFoundError)
