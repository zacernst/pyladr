"""Tests for the C runner cross-validation framework itself."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import C_REFERENCE_DIR, TEST_INPUTS_DIR, requires_c_binary

from .c_runner import (
    C_PROVER9_BIN,
    ProverResult,
    extract_proof_ids,
    extract_proof_justifications,
    run_c_prover9,
    run_c_prover9_from_string,
)


class TestCRunnerSmoke:
    """Verify the C runner can invoke prover9 and parse output."""

    @requires_c_binary
    def test_binary_exists(self) -> None:
        assert C_PROVER9_BIN.exists()

    @requires_c_binary
    def test_run_simple_group(self) -> None:
        input_file = TEST_INPUTS_DIR / "simple_group.in"
        result = run_c_prover9(input_file)

        assert result.succeeded
        assert result.theorem_proved
        assert not result.search_failed
        assert result.proof_length > 0
        assert len(result.proof_clauses) > 0

    @requires_c_binary
    def test_run_identity_only(self) -> None:
        input_file = TEST_INPUTS_DIR / "identity_only.in"
        result = run_c_prover9(input_file)

        assert result.succeeded
        assert result.theorem_proved
        assert result.proof_length > 0

    @requires_c_binary
    def test_run_lattice_absorption(self) -> None:
        input_file = TEST_INPUTS_DIR / "lattice_absorption.in"
        result = run_c_prover9(input_file)

        assert result.succeeded
        assert result.theorem_proved

    @requires_c_binary
    def test_run_from_string(self) -> None:
        input_text = """\
formulas(sos).
  e * x = x.
end_of_list.

formulas(goals).
  e * e = e.
end_of_list.
"""
        result = run_c_prover9_from_string(input_text)

        assert result.succeeded
        assert result.theorem_proved

    @requires_c_binary
    def test_proof_clause_extraction(self) -> None:
        input_file = TEST_INPUTS_DIR / "simple_group.in"
        result = run_c_prover9(input_file)

        ids = extract_proof_ids(result)
        justs = extract_proof_justifications(result)

        assert len(ids) > 0
        assert len(justs) == len(ids)
        # Proof IDs should be positive integers
        assert all(i > 0 for i in ids)

    @requires_c_binary
    def test_search_statistics_parsed(self) -> None:
        input_file = TEST_INPUTS_DIR / "simple_group.in"
        result = run_c_prover9(input_file)

        assert result.clauses_given > 0
        assert result.clauses_generated > 0
        assert result.clauses_kept > 0


class TestProverResultParsing:
    """Test output parsing with known reference data."""

    def test_parse_reference_x2(self) -> None:
        ref_file = C_REFERENCE_DIR / "x2_full_output.txt"
        if not ref_file.exists():
            pytest.skip("Reference output not captured yet")

        raw = ref_file.read_text()
        from .c_runner import _parse_output

        result = _parse_output(raw, 0)

        assert result.theorem_proved
        assert not result.search_failed
        assert result.proof_length > 0
        assert len(result.proof_clauses) > 0
        assert result.clauses_given > 0

    def test_parse_empty_output(self) -> None:
        from .c_runner import _parse_output

        result = _parse_output("", 1)

        assert not result.theorem_proved
        assert not result.search_failed
        assert result.proof_length == 0
        assert len(result.proof_clauses) == 0
