"""Tests for given clause trace display (REQ-R002).

Regression prevention: given clause traces must be printed to stdout during
search when ``print_given=True`` (the default) and suppressed when
``quiet=True`` or ``print_given=False``.

These tests cover:
- Given clause trace appears in stdout output
- Trace format matches C Prover9 output pattern
- Suppression via quiet and print_given flags
- Kept clause and proof found messages also reach stdout
- CLI flag integration (--no-print-given, --quiet, --print-kept)
- Cross-validation against C reference output format
"""

from __future__ import annotations

import io
import re
import sys
from unittest.mock import patch

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import get_rigid_term
from pyladr.parsing.ladr_parser import parse_input
from pyladr.search.given_clause import (
    ExitCode,
    GivenClauseSearch,
    SearchOptions,
)
from pyladr.search.selection import default_clause_weight


# ── Helpers ──────────────────────────────────────────────────────────────────


# Engine-level given clause line pattern (uses Clause.to_str() without symbol table):
#   given #N (X,wt=W): ID clause.
ENGINE_GIVEN_RE = re.compile(
    r"given\s+#(\d+)\s+"                    # given #N
    r"\(([A-Z]),wt=(\d+(?:\.\d+)?)\):\s+"   # (X,wt=W):
    r"(\d+)\s+"                             # clause ID
    r".+\."                                 # clause text ending with period
)

# CLI-level C Prover9 given clause line pattern (human-readable names + justification):
#   given #N (X,wt=W): ID clause_text.  [justification].
CLI_GIVEN_RE = re.compile(
    r"given\s+#(\d+)\s+"           # given #N
    r"\(([A-Z]),wt=(\d+(?:\.\d+)?)\):\s+"  # (X,wt=W):
    r"(\d+)\s+"                    # clause ID
    r".+\."                        # clause text ending with period
    r"\s+\[.+\]\."                 # [justification].
)


X2_INPUT = """\
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

SIMPLE_INPUT = """\
formulas(sos).
  e * x = x.
end_of_list.
formulas(goals).
  e * e = e.
end_of_list.
"""

RESOLUTION_INPUT = """\
formulas(sos).
  p(a).
  -p(x) | q(x).
end_of_list.
formulas(goals).
  q(a).
end_of_list.
"""


def _parse_and_deny(text: str) -> tuple[list[Clause], SymbolTable]:
    """Parse LADR input and deny goals into SOS clauses."""
    st = SymbolTable()
    parsed = parse_input(text, st)

    sos = list(parsed.sos)
    for goal in parsed.goals:
        denied_lits = tuple(
            Literal(sign=not lit.sign, atom=lit.atom) for lit in goal.literals
        )
        denied = Clause(
            literals=denied_lits,
            justification=(Justification(just_type=JustType.DENY, clause_ids=(0,)),),
        )
        sos.append(denied)

    return sos, st


def _run_and_capture(
    text: str,
    print_given: bool = True,
    print_kept: bool = False,
    quiet: bool = False,
    max_given: int = 500,
    max_seconds: float = 10.0,
    **kwargs,
) -> tuple[str, ExitCode]:
    """Run search and capture stdout output.

    Returns (captured_stdout, exit_code).
    """
    sos, st = _parse_and_deny(text)
    opts = SearchOptions(
        max_given=max_given,
        max_seconds=max_seconds,
        print_given=print_given,
        print_kept=print_kept,
        quiet=quiet,
        **kwargs,
    )
    engine = GivenClauseSearch(opts, symbol_table=st)

    captured = io.StringIO()
    with patch("sys.stdout", captured):
        result = engine.run(usable=[], sos=sos)

    return captured.getvalue(), result.exit_code


# ── Core trace display tests ────────────────────────────────────────────────


class TestGivenClauseTraceDisplay:
    """REQ-R002: Given clause traces must appear during search."""

    def test_given_clause_trace_appears_by_default(self):
        """Given clause lines must appear in stdout with default options."""
        output, exit_code = _run_and_capture(X2_INPUT)
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        assert "given #" in output, (
            "REGRESSION: given clause trace not appearing in output"
        )

    def test_given_clause_trace_for_simple_proof(self):
        """Even trivial proofs should show given clause traces."""
        output, exit_code = _run_and_capture(SIMPLE_INPUT)
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        assert "given #1" in output

    def test_multiple_given_clauses_appear(self):
        """Multi-step proofs should show multiple given clause lines."""
        output, exit_code = _run_and_capture(X2_INPUT)
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        given_lines = [l for l in output.splitlines() if "given #" in l]
        assert len(given_lines) >= 2, (
            f"Expected multiple given clauses, got {len(given_lines)}"
        )

    def test_given_numbers_are_sequential(self):
        """Given clause numbers should be sequential starting from 1."""
        output, _ = _run_and_capture(X2_INPUT)
        given_lines = [l for l in output.splitlines() if "given #" in l]
        numbers = []
        for line in given_lines:
            m = re.search(r"given\s+#(\d+)", line)
            if m:
                numbers.append(int(m.group(1)))
        assert numbers == list(range(1, len(numbers) + 1)), (
            f"Given clause numbers not sequential: {numbers}"
        )

    def test_resolution_problem_shows_given_trace(self):
        """Resolution-based proofs should also show given clause traces."""
        output, exit_code = _run_and_capture(
            RESOLUTION_INPUT, binary_resolution=True
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        assert "given #" in output


# ── Suppression tests ───────────────────────────────────────────────────────


class TestGivenClauseTraceSuppression:
    """Verify trace suppression via quiet and print_given flags."""

    def test_quiet_suppresses_given_trace(self):
        """quiet=True should suppress given clause output."""
        output, _ = _run_and_capture(X2_INPUT, quiet=True)
        assert "given #" not in output

    def test_print_given_false_suppresses_trace(self):
        """print_given=False should suppress given clause output."""
        output, _ = _run_and_capture(X2_INPUT, print_given=False)
        assert "given #" not in output

    def test_quiet_overrides_print_given(self):
        """quiet=True should suppress even when print_given=True."""
        output, _ = _run_and_capture(
            X2_INPUT, print_given=True, quiet=True
        )
        assert "given #" not in output


# ── Format validation tests ─────────────────────────────────────────────────


class TestGivenClauseTraceFormat:
    """Validate given clause trace format matches C Prover9."""

    def test_format_matches_engine_pattern(self):
        """Each given line should match: given #N (X,wt=W): ID: clause."""
        output, _ = _run_and_capture(X2_INPUT)
        given_lines = [l for l in output.splitlines() if "given #" in l]
        assert len(given_lines) > 0

        for line in given_lines:
            assert ENGINE_GIVEN_RE.search(line), (
                f"Given clause line doesn't match engine format:\n  {line!r}"
            )

    def test_selection_type_present(self):
        """Selection type (I, A, T, W) must be present."""
        output, _ = _run_and_capture(X2_INPUT)
        given_lines = [l for l in output.splitlines() if "given #" in l]
        for line in given_lines:
            m = re.search(r"\(([A-Z]),wt=", line)
            assert m, f"No selection type in: {line!r}"
            assert m.group(1) in ("I", "A", "T", "W"), (
                f"Unexpected selection type '{m.group(1)}' in: {line!r}"
            )

    def test_weight_is_numeric(self):
        """Weight should be a valid number."""
        output, _ = _run_and_capture(X2_INPUT)
        given_lines = [l for l in output.splitlines() if "given #" in l]
        for line in given_lines:
            m = re.search(r"wt=(\d+(?:\.\d+)?)", line)
            assert m, f"No weight in: {line!r}"
            float(m.group(1))  # should not raise

    def test_clause_id_present(self):
        """Each given line should include a clause ID after the weight info."""
        output, _ = _run_and_capture(X2_INPUT)
        given_lines = [l for l in output.splitlines() if "given #" in l]
        for line in given_lines:
            m = re.search(r"\):\s+(\d+)[:\s]", line)
            assert m, f"No clause ID in: {line!r}"
            assert int(m.group(1)) > 0

    def test_clause_text_ends_with_period(self):
        """Each given clause line should end with the clause period."""
        output, _ = _run_and_capture(X2_INPUT)
        given_lines = [l for l in output.splitlines() if "given #" in l]
        for line in given_lines:
            assert line.rstrip().endswith("."), (
                f"Given clause line should end with period: {line!r}"
            )


# ── Cross-validation with C reference output ────────────────────────────────


class TestCReferenceFormatCompatibility:
    """Cross-validate given clause format against C Prover9 reference output."""

    @pytest.fixture
    def c_reference_givens(self) -> list[str]:
        """Extract given clause lines from C reference x2 output."""
        import os
        ref_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..",
            "reference-prover9", "tests", "fixtures",
            "c_reference", "x2_full_output.txt",
        )
        if not os.path.exists(ref_path):
            pytest.skip("C reference output not available")
        with open(ref_path) as f:
            lines = f.readlines()
        return [l.strip() for l in lines if "given #" in l]

    def test_c_reference_has_given_lines(self, c_reference_givens):
        """Sanity: C reference output contains given clause lines."""
        assert len(c_reference_givens) > 0

    def test_c_reference_format_is_parseable(self, c_reference_givens):
        """C reference given lines should match our expected format regex."""
        for line in c_reference_givens:
            assert CLI_GIVEN_RE.search(line), (
                f"C reference given line doesn't match pattern:\n  {line!r}"
            )

    def test_python_format_matches_c_structure(self, c_reference_givens):
        """Python and C output both follow 'given #N (X,wt=W):' structure."""
        output, _ = _run_and_capture(
            X2_INPUT, paramodulation=True, demodulation=True
        )
        py_givens = [l.strip() for l in output.splitlines() if "given #" in l]

        if not py_givens:
            pytest.fail("No given clause lines in Python output")

        # Both should start with "given #1"
        assert py_givens[0].startswith("given #1"), (
            f"Python first given should be #1: {py_givens[0]!r}"
        )
        assert c_reference_givens[0].startswith("given #1"), (
            f"C first given should be #1: {c_reference_givens[0]!r}"
        )

        # Both should follow the (X,wt=W) structural pattern
        py_match = re.search(r"\(([A-Z]),wt=\d+\)", py_givens[0])
        c_match = re.search(r"\(([A-Z]),wt=\d+\)", c_reference_givens[0])
        assert py_match, f"Python format mismatch: {py_givens[0]!r}"
        assert c_match, f"C format mismatch: {c_reference_givens[0]!r}"


# ── Kept clause and proof message tests ─────────────────────────────────────


class TestRelatedTraceMessages:
    """Verify other trace messages (kept, proof found) also use stdout."""

    def test_kept_clause_suppressed_by_default(self):
        """print_kept=False (default) should not produce kept lines."""
        output, _ = _run_and_capture(X2_INPUT, print_kept=False)
        assert "kept:" not in output


# ── CLI integration tests ───────────────────────────────────────────────────


class TestCLIFlagIntegration:
    """Test CLI flags that control trace output."""

    def _run_cli(self, extra_args: list[str] | None = None) -> str:
        """Run prover9 CLI with x2 input and capture stdout."""
        from pyladr.apps.prover9 import run_prover

        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".in", delete=False
        ) as f:
            f.write(X2_INPUT)
            f.flush()
            input_path = f.name

        try:
            argv = [
                "pyprover9", "-f", input_path,
                "--paramodulation", "-max_given", "50",
            ]
            if extra_args:
                argv.extend(extra_args)

            captured = io.StringIO()
            with patch("sys.stdout", captured):
                run_prover(argv=argv)

            return captured.getvalue()
        finally:
            os.unlink(input_path)

    def test_default_cli_shows_given(self):
        """Default CLI invocation should show given clause traces."""
        output = self._run_cli()
        assert "given #" in output, (
            "REGRESSION: CLI default should display given clause traces"
        )

    def test_cli_given_format_is_valid(self):
        """CLI given clause lines should match the engine format pattern."""
        output = self._run_cli()
        given_lines = [l.strip() for l in output.splitlines() if "given #" in l]
        assert len(given_lines) > 0
        for line in given_lines:
            assert ENGINE_GIVEN_RE.search(line), (
                f"CLI given clause line doesn't match format:\n  {line!r}"
            )

    def test_no_print_given_flag(self):
        """--no-print-given should suppress given clause output."""
        output = self._run_cli(["--no-print-given"])
        assert "given #" not in output

    def test_quiet_flag(self):
        """--quiet should suppress given clause output."""
        output = self._run_cli(["--quiet"])
        assert "given #" not in output

    def test_print_kept_flag(self):
        """--print-kept should enable kept clause output."""
        output = self._run_cli(["--print-kept"])
        assert "kept:" in output


# ── Regression guard: stdout not logger ─────────────────────────────────────


class TestOutputMechanism:
    """Ensure trace output goes through print(), not logging.

    This is the root cause guard: if someone changes print() back to
    logger.info(), these tests will fail because logging has no configured
    handler by default.
    """

    def test_given_trace_uses_stdout_not_logging(self):
        """Given clause trace must reach stdout even without logging config.

        This is the critical regression test. Python's default logging level
        is WARNING, so logger.info() calls are silently dropped unless
        explicit logging config is set up. The trace must use print() or
        an equivalent mechanism that writes to stdout directly.
        """
        output, exit_code = _run_and_capture(SIMPLE_INPUT)
        assert exit_code == ExitCode.MAX_PROOFS_EXIT
        assert "given #1" in output, (
            "CRITICAL REGRESSION: Given clause trace not reaching stdout. "
            "Check that print() is used instead of logger.info() in "
            "GivenClauseSearch._make_inferences()."
        )

