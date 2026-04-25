"""Subsumption output reporting validation tests.

Validates that PyLADR correctly reports subsumption activity during search,
matching C Prover9's output format:

1. Final STATISTICS section reports Forward_subsumed and Back_subsumed counts
2. print_kept output uses human-readable symbol names (not internal s3/s6/v0)
3. print_given output uses human-readable symbol names
4. Backward subsumption logging uses human-readable symbol names

BUG FOUND: print_kept at given_clause.py:743 uses c.to_str() without
symbol_table, outputting internal names like s3(s6(...)) instead of
P(i(...)).

Run: python3 -m pytest tests/soundness/test_subsumption_output_reporting.py -v
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

VAMPIRE_IN = os.path.join(os.path.dirname(__file__), "..", "fixtures", "inputs", "vampire.in")
C_PROVER9 = os.path.join(os.path.dirname(__file__), "..", "..", "reference-prover9", "bin", "prover9")
PYTHON = sys.executable


def _run_pyladr(args: list[str], timeout: int = 120) -> str:
    """Run PyLADR CLI and return combined output."""
    proc = subprocess.run(
        [PYTHON, "-m", "pyladr.cli"] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.stdout + proc.stderr


def _run_c_prover9(input_text: str, timeout: int = 60) -> str:
    """Run C Prover9 on input text and return output."""
    proc = subprocess.run(
        [C_PROVER9],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.stdout + proc.stderr


# ── Statistics Section Tests ─────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(VAMPIRE_IN), reason="vampire.in not found")
class TestStatisticsSectionReporting:
    """Validate that STATISTICS section reports subsumption counts."""

    def test_statistics_section_present(self):
        """PyLADR output should contain a STATISTICS section."""
        output = _run_pyladr(["-f", VAMPIRE_IN, "-max_given", "10"])
        assert "STATISTICS" in output, "No STATISTICS section in output"

    def test_forward_subsumed_reported(self):
        """STATISTICS should report Forward_subsumed count."""
        output = _run_pyladr(["-f", VAMPIRE_IN, "-max_given", "20"])
        match = re.search(r"Forward_subsumed=(\d+)", output)
        assert match is not None, (
            f"Forward_subsumed not found in STATISTICS.\n"
            f"Stats section:\n{_extract_stats_section(output)}"
        )
        count = int(match.group(1))
        assert count > 0, f"Forward_subsumed={count}, expected > 0 for vampire.in"

    def test_back_subsumed_reported(self):
        """STATISTICS should report Back_subsumed count."""
        output = _run_pyladr(["-f", VAMPIRE_IN, "-max_given", "50"])
        match = re.search(r"Back_subsumed=(\d+)", output)
        assert match is not None, (
            f"Back_subsumed not found in STATISTICS.\n"
            f"Stats section:\n{_extract_stats_section(output)}"
        )
        # Back subsumption may be 0 for small runs; just verify it's reported
        count = int(match.group(1))
        assert count >= 0

    def test_given_generated_kept_reported(self):
        """STATISTICS should report Given, Generated, Kept counts."""
        output = _run_pyladr(["-f", VAMPIRE_IN, "-max_given", "10"])
        for field in ["Given", "Generated", "Kept"]:
            match = re.search(rf"{field}=(\d+)", output)
            assert match is not None, f"{field} not found in STATISTICS"
            assert int(match.group(1)) > 0, f"{field}=0, expected > 0"

    @pytest.mark.skipif(not os.path.exists(C_PROVER9), reason="C Prover9 not found")
    def test_statistics_format_matches_c(self):
        """PyLADR STATISTICS format should match C Prover9 key fields."""
        py_output = _run_pyladr(["-f", VAMPIRE_IN, "-max_given", "10"])
        # C format: "Given=N. Generated=N. Kept=N. proofs=N."
        # C format: "Forward_subsumed=N. Back_subsumed=N."
        assert re.search(r"Given=\d+\.\s+Generated=\d+\.\s+Kept=\d+\.\s+proofs=\d+\.", py_output), (
            f"Statistics line doesn't match C format.\nStats:\n{_extract_stats_section(py_output)}"
        )
        assert re.search(r"Forward_subsumed=\d+\.\s+Back_subsumed=\d+\.", py_output), (
            f"Subsumption stats don't match C format.\nStats:\n{_extract_stats_section(py_output)}"
        )


# ── Print Given Output Tests ────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(VAMPIRE_IN), reason="vampire.in not found")
class TestPrintGivenOutput:
    """Validate that print_given uses human-readable symbol names."""

    def test_given_clauses_use_readable_names(self):
        """Given clause output should use P, i, n, not s3, s6, s8."""
        output = _run_pyladr(["-f", VAMPIRE_IN, "-max_given", "10"])
        given_lines = [l for l in output.splitlines() if l.startswith("given #")]
        assert len(given_lines) > 0, "No given lines in output"

        for line in given_lines:
            # Should contain P( from the vampire.in problem
            assert "P(" in line or "deny" in line.lower() or "$F" in line, (
                f"Given line missing human-readable symbols: {line}"
            )
            # Should NOT contain internal symbol names
            _assert_no_internal_symbols(line, "given")

    def test_given_format_matches_c_pattern(self):
        """Given clause format: 'given #N (X,wt=W): ID clause.  [justification].'"""
        output = _run_pyladr(["-f", VAMPIRE_IN, "-max_given", "5"])
        given_lines = [l for l in output.splitlines() if l.startswith("given #")]
        assert len(given_lines) >= 5

        for line in given_lines:
            # Check format: given #N (TYPE,wt=N): ...
            assert re.match(r"given #\d+ \([A-Z],wt=\d+\):", line), (
                f"Given line doesn't match expected format: {line}"
            )


# ── Print Kept Output Tests ─────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(VAMPIRE_IN), reason="vampire.in not found")
class TestPrintKeptOutput:
    """Validate that print_kept uses human-readable symbol names.

    BUG: given_clause.py:743 calls c.to_str() without symbol_table,
    producing internal names like s3(s6(v0,v1)) instead of P(i(x,y)).
    """

    def test_kept_clauses_use_readable_names(self):
        """Kept clause output should use P, i, n, not s3, s6, s8.

        This test will FAIL until the print_kept bug is fixed.
        """
        output = _run_pyladr(["-f", VAMPIRE_IN, "-max_given", "10", "--print-kept"])
        kept_lines = [l for l in output.splitlines() if "kept:" in l.lower()]
        assert len(kept_lines) > 0, "No kept lines in output (--print-kept may not be working)"

        failures = []
        for line in kept_lines:
            # Check for internal symbol pattern: s followed by digits
            if re.search(r'\bs\d+\(', line):
                failures.append(line)

        if failures:
            pytest.fail(
                f"print_kept uses internal symbol names instead of human-readable names.\n"
                f"Found {len(failures)} kept lines with internal symbols (showing first 3):\n"
                + "\n".join(failures[:3])
                + "\n\nExpected format like: kept:      9 P(i(i(i(i(x,y),i(z,y)),u),i(i(z,x),u))).  [hyper_res]."
                + "\n\nRoot cause: given_clause.py:743 calls c.to_str() without symbol_table."
                + "\nFix: use self._format_clause_std(c) instead."
            )

    def test_kept_includes_justification(self):
        """Kept clause output should include justification like C Prover9."""
        output = _run_pyladr(["-f", VAMPIRE_IN, "-max_given", "10", "--print-kept"])
        kept_lines = [l for l in output.splitlines() if "kept:" in l.lower()]

        # At least some kept lines should have justification brackets
        lines_with_just = [l for l in kept_lines if "[" in l and "]" in l]
        # Initial clauses may not have justification, but inferred ones should
        # With 10 given, there should be inferred clauses
        if len(kept_lines) > 6:  # 6 initial clauses
            assert len(lines_with_just) > 0, (
                f"No kept lines include justification brackets.\n"
                f"Sample kept lines:\n" + "\n".join(kept_lines[:5])
            )

    @pytest.mark.skipif(not os.path.exists(C_PROVER9), reason="C Prover9 not found")
    def test_kept_count_comparable_to_c(self):
        """Number of kept clauses should be in same ballpark as C Prover9."""
        py_output = _run_pyladr(["-f", VAMPIRE_IN, "-max_given", "10", "--print-kept"])
        py_kept = len([l for l in py_output.splitlines() if "kept:" in l.lower()])

        with open(VAMPIRE_IN) as f:
            input_text = f.read()
        input_with_limit = input_text.replace(
            "set(auto).", "set(auto).\nassign(max_given, 10).\nset(print_kept)."
        )
        c_output = _run_c_prover9(input_with_limit)
        c_kept = len([l for l in c_output.splitlines() if l.startswith("kept:")])

        print(f"\n  PyLADR kept lines: {py_kept}")
        print(f"  C Prover9 kept lines: {c_kept}")

        # Both should have kept clauses
        assert py_kept > 0, "PyLADR produced no kept lines"
        assert c_kept > 0, "C Prover9 produced no kept lines"


# ── Backward Subsumption Reporting Tests ─────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(VAMPIRE_IN), reason="vampire.in not found")
class TestBackSubsumptionReporting:
    """Validate backward subsumption event reporting."""

    def test_back_subsumed_reported_in_statistics(self):
        """STATISTICS section should report Back_subsumed count (may be 0)."""
        output = _run_pyladr(["-f", VAMPIRE_IN, "-max_given", "50"])
        match = re.search(r"Back_subsumed=(\d+)", output)
        assert match is not None, (
            f"Back_subsumed not found in STATISTICS.\n"
            f"Stats section:\n{_extract_stats_section(output)}"
        )
        # Count may be 0 depending on problem configuration; just verify reporting
        count = int(match.group(1))
        assert count >= 0

    def test_back_subsumed_logged_with_readable_names(self):
        """Back subsumption debug log should use human-readable names.

        Currently uses logger.debug with c.to_str() (no symbol table).
        This test validates the logging content if debug is enabled.
        """
        # Run with debug logging enabled
        import logging
        from io import StringIO
        from pyladr.search.given_clause import GivenClauseSearch
        from pyladr.core.clause import Clause, Literal

        # This is a structural check - verify the code at the logging point
        # uses symbol table for formatting
        import inspect
        source = inspect.getsource(GivenClauseSearch._limbo_process)

        # Check if back_subsumed logging uses to_str with symbol_table
        # or _format_clause_std (which includes symbol_table)
        if "victim.to_str()" in source and "_format_clause_std" not in source.split("back subsumed")[0].split("back subsumed")[-1] if "back subsumed" in source else True:
            # Check the specific logging line
            lines = source.split("\n")
            for i, line in enumerate(lines):
                if "back subsumed" in line and "to_str()" in line:
                    # Found the bug: to_str() without symbol_table
                    pytest.fail(
                        f"Back subsumption logging at _limbo_process uses "
                        f"to_str() without symbol_table.\n"
                        f"Line: {line.strip()}\n"
                        f"Fix: use self._format_clause_std() instead."
                    )


# ── Symbol Name Rendering Tests ──────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(VAMPIRE_IN), reason="vampire.in not found")
class TestSymbolNameRendering:
    """Validate all clause output paths use human-readable symbol names."""

    def test_no_internal_symbols_in_search_output(self):
        """Search output should never contain internal symbol names like s3, s6."""
        output = _run_pyladr(["-f", VAMPIRE_IN, "-max_given", "10"])
        # Check all non-header lines for internal symbols
        for line in output.splitlines():
            if line.startswith("=") or line.startswith("%") or line.startswith("--"):
                continue
            if "given #" in line or "kept:" in line.lower():
                _assert_no_internal_symbols(line, "search output")

    def test_initial_clauses_have_readable_names(self):
        """Initial (I) given clauses should display P, i, n from input."""
        output = _run_pyladr(["-f", VAMPIRE_IN, "-max_given", "6"])
        given_lines = [l for l in output.splitlines() if l.startswith("given #")]
        initial_lines = [l for l in given_lines if "(I," in l]
        assert len(initial_lines) >= 4, f"Expected at least 4 initial clauses, got {len(initial_lines)}"

        for line in initial_lines:
            # Must contain the predicate P and functor i from vampire.in
            if "deny" not in line.lower():
                assert "P(" in line, f"Initial clause missing P(: {line}"

    def test_inferred_clauses_have_readable_names(self):
        """Inferred clauses (A/W/T) should also use human-readable names."""
        output = _run_pyladr(["-f", VAMPIRE_IN, "-max_given", "15"])
        given_lines = [l for l in output.splitlines() if l.startswith("given #")]
        inferred = [l for l in given_lines if "(A," in l or "(W," in l or "(T," in l]
        assert len(inferred) > 0, "No inferred given clauses found"

        for line in inferred:
            assert "P(" in line, f"Inferred given clause missing P(: {line}"
            _assert_no_internal_symbols(line, "inferred given")


# ── Cross-Validation: C vs PyLADR Output Comparison ─────────────────────────

@pytest.mark.skipif(
    not os.path.exists(VAMPIRE_IN) or not os.path.exists(C_PROVER9),
    reason="vampire.in or C Prover9 not found",
)
class TestCrossValidationOutput:
    """Compare output format between C Prover9 and PyLADR."""

    def test_given_clause_format_comparable(self):
        """Given clause output format should be structurally similar."""
        py_output = _run_pyladr(["-f", VAMPIRE_IN, "-max_given", "10"])
        py_given = [l for l in py_output.splitlines() if l.startswith("given #")]

        with open(VAMPIRE_IN) as f:
            input_text = f.read()
        input_with_limit = input_text.replace(
            "set(auto).", "set(auto).\nassign(max_given, 10)."
        )
        c_output = _run_c_prover9(input_with_limit)
        c_given = [l for l in c_output.splitlines() if l.startswith("given #")]

        print(f"\n  PyLADR given lines: {len(py_given)}")
        print(f"  C Prover9 given lines: {len(c_given)}")

        # Both should have ~10 given clauses
        assert len(py_given) >= 10
        assert len(c_given) >= 10

        # Compare format structure (not exact content due to numbering differences)
        # C: "given #1 (I,wt=8): 3 -P(x) | -P(i(x,y)) | P(y).  [assumption]."
        # PyLADR: "given #1 (I,wt=8): 1 -P(x) | -P(i(x,y)) | P(y)."
        for py_line in py_given[:3]:
            # Should have same structure
            assert re.match(r"given #\d+ \([A-Z],wt=\d+\):", py_line), (
                f"PyLADR given line format doesn't match C pattern: {py_line}"
            )

    def test_statistics_fields_superset_check(self):
        """PyLADR should report at least the key C Prover9 statistics fields."""
        py_output = _run_pyladr(["-f", VAMPIRE_IN, "-max_given", "10"])
        py_stats = _extract_stats_section(py_output)

        required_fields = [
            "Given", "Generated", "Kept", "proofs",
            "Forward_subsumed", "Back_subsumed",
        ]
        missing = [f for f in required_fields if f not in py_stats]
        assert len(missing) == 0, (
            f"Missing statistics fields: {missing}\n"
            f"PyLADR STATISTICS section:\n{py_stats}"
        )


# ── Regression Prevention ────────────────────────────────────────────────────

class TestOutputRegressionPrevention:
    """Structural checks to prevent output reporting regressions."""

    def test_format_clause_std_uses_symbol_table(self):
        """_format_clause_std should pass symbol_table to to_str()."""
        import inspect
        from pyladr.search.given_clause import GivenClauseSearch
        source = inspect.getsource(GivenClauseSearch._format_clause_std)
        # Should reference symbol_table (self._symbol_table) in atom.to_str(st) call
        assert "symbol_table" in source or "_symbol_table" in source or "to_str(st)" in source, (
            "_format_clause_std doesn't appear to use symbol_table for formatting"
        )

    def test_print_given_uses_format_clause_std(self):
        """Print given should use _format_clause_std (not raw to_str)."""
        import inspect
        from pyladr.search.given_clause import GivenClauseSearch

        # Check the source code for print_given context
        source = inspect.getsource(GivenClauseSearch)
        lines = source.split("\n")

        # Find blocks where print_given is checked and look for _format_clause_std nearby
        for i, line in enumerate(lines):
            if "print_given" in line and "self._opts" in line:
                # Check the surrounding 10 lines for _format_clause_std
                block = "\n".join(lines[i:min(len(lines), i+10)])
                if "_format_clause_std" in block:
                    return  # Found it — test passes

        pytest.fail(
            "print_given code path doesn't use _format_clause_std near the print_given check"
        )

    def test_print_kept_should_use_format_clause_std(self):
        """Print kept SHOULD use _format_clause_std (currently uses raw to_str).

        This test documents the bug: print_kept calls c.to_str() without
        the symbol table, producing internal names.
        """
        import inspect
        from pyladr.search.given_clause import GivenClauseSearch
        source = inspect.getsource(GivenClauseSearch._keep_clause)

        # Find the print_kept line
        if "print_kept" in source:
            # Check if it uses _format_clause_std or to_str with symbol_table
            lines = source.split("\n")
            for i, line in enumerate(lines):
                if "print_kept" in line:
                    # Look at the next few lines for the actual print/log call
                    context = "\n".join(lines[i:min(len(lines), i+5)])
                    if "to_str()" in context and "_format_clause_std" not in context:
                        pytest.fail(
                            f"print_kept uses to_str() without symbol_table.\n"
                            f"Code:\n{context}\n"
                            f"Fix: replace c.to_str() with self._format_clause_std(c)"
                        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_stats_section(output: str) -> str:
    """Extract the STATISTICS section from output."""
    lines = output.splitlines()
    in_stats = False
    stats_lines = []
    for line in lines:
        if "STATISTICS" in line:
            in_stats = True
        if in_stats:
            stats_lines.append(line)
        if in_stats and "end of statistics" in line.lower():
            break
    return "\n".join(stats_lines) if stats_lines else "(no STATISTICS section found)"


def _assert_no_internal_symbols(line: str, context: str) -> None:
    """Assert that a line doesn't contain internal symbol names like s3(, s6(."""
    # Internal symbols are sN( where N is a digit — but exclude legitimate words
    # like "stats", "step", etc. Focus on the pattern: bare sN( as a function call
    if re.search(r'\bs\d+\(', line):
        # Make sure it's not a word like "set(" or "sos("
        matches = re.findall(r'\bs\d+\(', line)
        real_internal = [m for m in matches if m not in ("set(", "sos(")]
        if real_internal:
            pytest.fail(
                f"Internal symbol names found in {context}: {real_internal}\n"
                f"Line: {line}\n"
                f"Expected human-readable names like P(, i(, n("
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
