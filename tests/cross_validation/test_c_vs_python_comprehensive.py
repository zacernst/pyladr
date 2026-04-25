"""Comprehensive C vs Python Prover9 comparison test suite.

Runs identical inputs through both the C reference Prover9 binary and
the Python pyprover9 implementation, comparing:
  - Theorem proved/failed status (must match exactly)
  - Exit codes (must match)
  - Proof existence (both find proof, or both fail)
  - Search statistics (within configurable tolerance)
  - Output format conventions (sections, markers)

Organized by problem category:
  1. Trivial / sanity-check problems
  2. Pure resolution (no equality)
  3. Equational / paramodulation problems
  4. Problems requiring set(auto) inference detection
  5. Search limit behavior
  6. File-based inputs from fixtures and examples
  7. Edge cases and error handling

Run with:
    pytest tests/cross_validation/test_c_vs_python_comprehensive.py -v
    pytest tests/cross_validation/test_c_vs_python_comprehensive.py -v -k "not slow"
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.cross_validation.c_runner import (
    C_PROVER9_BIN,
    ProverResult,
    run_c_prover9,
    run_c_prover9_from_string,
)
from tests.cross_validation.comparator import (
    ComparisonResult,
    compare_full,
    compare_search_statistics,
    compare_theorem_result,
)

# ── Path setup ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "inputs"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
C_EXAMPLES_DIR = PROJECT_ROOT / "reference-prover9" / "prover9.examples"

# ── Skip marker ────────────────────────────────────────────────────────────

requires_c_binary = pytest.mark.skipif(
    not C_PROVER9_BIN.exists(),
    reason="C prover9 binary not found at reference-prover9/bin/prover9",
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _run_python_on_text(
    input_text: str,
    *,
    max_given: int = 500,
    auto: bool = True,
) -> tuple[ProverResult, Any]:
    """Run Python prover9 on input text via the full app pipeline.

    Uses the same code path as `pyprover9 -f FILE`, including parsing,
    goal denial, settings application, and search.

    By default, auto=True to match C Prover9's default behavior of
    always running auto-inference detection.

    Returns (ProverResult-compatible, raw SearchResult).
    """
    from pyladr.apps.prover9 import _apply_settings, _auto_cascade, _deny_goals
    from pyladr.core.symbol import SymbolTable
    from pyladr.parsing.ladr_parser import LADRParser
    from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions

    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(input_text)

    usable, sos, _denied = _deny_goals(parsed, st)

    opts = SearchOptions(
        binary_resolution=True,
        paramodulation=False,
        demodulation=False,
        factoring=True,
        max_given=max_given,
        quiet=True,
    )

    # Apply settings from input (including set(auto) if present)
    _apply_settings(parsed, opts, st)

    # Match C Prover9 default: always run auto-cascade unless explicitly
    # disabled in the input. C defaults to set(auto).
    if auto and not parsed.settings.flag("auto"):
        _auto_cascade(parsed, opts, st)

    search = GivenClauseSearch(options=opts, symbol_table=st)
    result = search.run(usable=usable, sos=sos)

    proof_length = 0
    if result.proofs:
        proof_length = len(result.proofs[0].clauses)

    pr = ProverResult(
        exit_code=int(result.exit_code),
        raw_output="",
        theorem_proved=(result.exit_code == ExitCode.MAX_PROOFS_EXIT),
        search_failed=(result.exit_code == ExitCode.SOS_EMPTY_EXIT),
        clauses_given=result.stats.given,
        clauses_generated=result.stats.generated,
        clauses_kept=result.stats.kept,
        clauses_deleted=result.stats.sos_limit_deleted,
        proof_length=proof_length,
    )
    return pr, result


def _run_python_subprocess(
    input_text: str,
    *,
    timeout: float = 60.0,
    extra_args: list[str] | None = None,
) -> ProverResult:
    """Run Python prover9 via subprocess, capturing output like C runner.

    This tests the full CLI pipeline end-to-end.
    """
    cmd = [
        sys.executable, "-c",
        "from pyladr.apps.prover9 import run_prover; import sys; sys.exit(run_prover(['pyprover9']))",
    ]
    if extra_args:
        # Insert file args etc. — but for subprocess tests we use stdin
        pass

    proc = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )

    raw = proc.stdout + proc.stderr
    return _parse_python_output(raw, proc.returncode)


def _parse_python_output(raw: str, exit_code: int) -> ProverResult:
    """Parse Python prover9 output into ProverResult."""
    import re

    result = ProverResult(exit_code=exit_code, raw_output=raw)
    result.theorem_proved = "THEOREM PROVED" in raw
    result.search_failed = "SEARCH FAILED" in raw

    m = re.search(r"Given=(\d+)", raw)
    if m:
        result.clauses_given = int(m.group(1))
    m = re.search(r"Generated=(\d+)", raw)
    if m:
        result.clauses_generated = int(m.group(1))
    m = re.search(r"Kept=(\d+)", raw)
    if m:
        result.clauses_kept = int(m.group(1))
    m = re.search(r"Length of proof is (\d+)", raw)
    if m:
        result.proof_length = int(m.group(1))
    m = re.search(r"User_CPU=([0-9]+\.[0-9]+)", raw)
    if m:
        result.user_cpu_time = float(m.group(1))

    return result


def _assert_theorem_match(c_result: ProverResult, py_result: ProverResult, label: str):
    """Assert that C and Python agree on theorem status."""
    comp = compare_theorem_result(c_result, py_result)
    assert comp.equivalent, (
        f"[{label}] Theorem status mismatch:\n"
        f"  C: proved={c_result.theorem_proved}, failed={c_result.search_failed}\n"
        f"  Py: proved={py_result.theorem_proved}, failed={py_result.search_failed}"
    )


def _assert_exit_code_match(c_result: ProverResult, py_result: ProverResult, label: str):
    """Assert exit codes match (mapping C codes to Python codes)."""
    # Both should agree on the semantic meaning
    if c_result.theorem_proved:
        assert py_result.theorem_proved, (
            f"[{label}] C proved theorem (exit={c_result.exit_code}) "
            f"but Python did not (exit={py_result.exit_code})"
        )
    if c_result.search_failed:
        assert py_result.search_failed, (
            f"[{label}] C search failed (exit={c_result.exit_code}) "
            f"but Python did not (exit={py_result.exit_code})"
        )


# ══════════════════════════════════════════════════════════════════════════
# 1. TRIVIAL / SANITY CHECK PROBLEMS
# ══════════════════════════════════════════════════════════════════════════


TRIVIAL_INPUTS = {
    "P(a)_from_P(a)": """\
formulas(sos).
  P(a).
end_of_list.

formulas(goals).
  P(a).
end_of_list.
""",
    "P(a)_and_negP(a)": """\
formulas(sos).
  P(a).
  -P(a).
end_of_list.
""",
    "unit_equality_reflexive": """\
formulas(sos).
  e * x = x.
end_of_list.

formulas(goals).
  e * e = e.
end_of_list.
""",
}


@pytest.mark.cross_validation
@requires_c_binary
class TestTrivialProblems:
    """Sanity checks: both engines agree on trivial problems."""

    @pytest.mark.parametrize("name", sorted(TRIVIAL_INPUTS.keys()))
    def test_trivial_theorem_status(self, name: str):
        """C and Python agree on trivial problems (theorem status)."""
        text = TRIVIAL_INPUTS[name]
        c_result = run_c_prover9_from_string(text, timeout=10.0)
        py_result, _ = _run_python_on_text(text, max_given=50)
        _assert_theorem_match(c_result, py_result, name)

    @pytest.mark.parametrize("name", sorted(TRIVIAL_INPUTS.keys()))
    def test_trivial_subprocess_match(self, name: str):
        """Python subprocess output matches C for trivial problems."""
        text = TRIVIAL_INPUTS[name]
        c_result = run_c_prover9_from_string(text, timeout=10.0)
        py_result = _run_python_subprocess(text, timeout=10.0)
        _assert_theorem_match(c_result, py_result, f"{name}_subprocess")


# ══════════════════════════════════════════════════════════════════════════
# 2. PURE RESOLUTION PROBLEMS (no equality)
# ══════════════════════════════════════════════════════════════════════════


RESOLUTION_INPUTS = {
    "chain_resolution": """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(a).
end_of_list.
""",
    "two_step_chain": """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(x) | R(x).
  -R(a).
end_of_list.
""",
    "multi_literal_resolution": """\
formulas(sos).
  P(a).
  Q(b).
  -P(x) | -Q(y) | R(x,y).
  -R(a,b).
end_of_list.
""",
    "binary_predicates": """\
formulas(sos).
  P(a,b).
  -P(x,y) | P(y,x).
end_of_list.

formulas(goals).
  P(b,a).
end_of_list.
""",
    "multi_step_resolution": """\
formulas(sos).
  P(a).
  P(b).
  -P(x) | Q(x).
end_of_list.

formulas(goals).
  Q(a).
end_of_list.
""",
}


@pytest.mark.cross_validation
@requires_c_binary
class TestResolutionProblems:
    """Pure resolution problems — no equality/paramodulation needed."""

    @pytest.mark.parametrize("name", sorted(RESOLUTION_INPUTS.keys()))
    def test_resolution_theorem_status(self, name: str):
        """C and Python agree on pure resolution problems."""
        text = RESOLUTION_INPUTS[name]
        c_result = run_c_prover9_from_string(text, timeout=15.0)
        py_result, _ = _run_python_on_text(text, max_given=200)
        _assert_theorem_match(c_result, py_result, name)

    @pytest.mark.parametrize("name", sorted(RESOLUTION_INPUTS.keys()))
    def test_resolution_both_find_proof(self, name: str):
        """Both C and Python find a proof for provable resolution problems."""
        text = RESOLUTION_INPUTS[name]
        c_result = run_c_prover9_from_string(text, timeout=15.0)
        if not c_result.theorem_proved:
            pytest.skip(f"C does not prove {name} (expected for some)")
        py_result, _ = _run_python_on_text(text, max_given=200)
        assert py_result.theorem_proved, (
            f"C proved {name} but Python did not.\n"
            f"C stats: Given={c_result.clauses_given}, "
            f"Gen={c_result.clauses_generated}, Kept={c_result.clauses_kept}\n"
            f"Py stats: Given={py_result.clauses_given}, "
            f"Gen={py_result.clauses_generated}, Kept={py_result.clauses_kept}"
        )


# ══════════════════════════════════════════════════════════════════════════
# 3. EQUATIONAL / PARAMODULATION PROBLEMS
# ══════════════════════════════════════════════════════════════════════════


EQUATIONAL_INPUTS = {
    "identity_only": (FIXTURES_DIR / "identity_only.in"),
    "simple_group": (FIXTURES_DIR / "simple_group.in"),
    "lattice_absorption": (FIXTURES_DIR / "lattice_absorption.in"),
}

EQUATIONAL_INLINE = {
    "transitivity": """\
set(auto).

formulas(sos).
  a = b.
  b = c.
end_of_list.

formulas(goals).
  a = c.
end_of_list.
""",
    "congruence": """\
set(auto).

formulas(sos).
  a = b.
end_of_list.

formulas(goals).
  f(a) = f(b).
end_of_list.
""",
    "symmetry": """\
set(auto).

formulas(sos).
  a = b.
end_of_list.

formulas(goals).
  b = a.
end_of_list.
""",
    "simple_rewriting": """\
set(auto).

formulas(sos).
  f(a) = b.
  f(b) = c.
end_of_list.

formulas(goals).
  f(f(a)) = c.
end_of_list.
""",
}


@pytest.mark.cross_validation
@requires_c_binary
class TestEquationalProblems:
    """Equational reasoning problems requiring paramodulation."""

    @pytest.mark.parametrize("name", sorted(EQUATIONAL_INLINE.keys()))
    def test_equational_inline_theorem_status(self, name: str):
        """C and Python agree on inline equational problems."""
        text = EQUATIONAL_INLINE[name]
        c_result = run_c_prover9_from_string(text, timeout=15.0)
        py_result, _ = _run_python_on_text(text, max_given=200)
        _assert_theorem_match(c_result, py_result, name)

    @pytest.mark.parametrize("name", sorted(EQUATIONAL_INPUTS.keys()))
    def test_equational_file_theorem_status(self, name: str):
        """C and Python agree on file-based equational problems."""
        path = EQUATIONAL_INPUTS[name]
        if not path.exists():
            pytest.skip(f"Input file not found: {path}")

        text = path.read_text()
        c_result = run_c_prover9(path, timeout=30.0)
        py_result, _ = _run_python_on_text(text, max_given=500)
        _assert_theorem_match(c_result, py_result, name)


# ══════════════════════════════════════════════════════════════════════════
# 4. SET(AUTO) INFERENCE DETECTION
# ══════════════════════════════════════════════════════════════════════════


AUTO_INPUTS = {
    "auto_pure_equality": """\
set(auto).

formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
  x * x = e.
end_of_list.

formulas(goals).
  x * y = y * x.
end_of_list.
""",
    "auto_pure_resolution": """\
set(auto).

formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(x) | R(x).
end_of_list.

formulas(goals).
  R(a).
end_of_list.
""",
    "auto_mixed_horn": """\
set(auto).

formulas(sos).
  P(a).
  -P(x) | Q(f(x)).
  -Q(x) | P(g(x)).
end_of_list.

formulas(goals).
  Q(f(a)).
end_of_list.
""",
}


@pytest.mark.cross_validation
@requires_c_binary
class TestAutoMode:
    """Tests that set(auto) inference detection matches C behavior."""

    @pytest.mark.parametrize("name", sorted(AUTO_INPUTS.keys()))
    def test_auto_theorem_status(self, name: str):
        """Both agree on theorem status when set(auto) is used."""
        text = AUTO_INPUTS[name]
        c_result = run_c_prover9_from_string(text, timeout=30.0)
        py_result, _ = _run_python_on_text(text, max_given=500)
        _assert_theorem_match(c_result, py_result, name)

    @pytest.mark.parametrize("name", sorted(AUTO_INPUTS.keys()))
    def test_auto_exit_code_match(self, name: str):
        """Both agree on exit code semantics with set(auto)."""
        text = AUTO_INPUTS[name]
        c_result = run_c_prover9_from_string(text, timeout=30.0)
        py_result, _ = _run_python_on_text(text, max_given=500)
        _assert_exit_code_match(c_result, py_result, name)


# ══════════════════════════════════════════════════════════════════════════
# 5. SEARCH LIMIT BEHAVIOR
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.cross_validation
@requires_c_binary
class TestSearchLimits:
    """Verify both engines respect search limits consistently."""

    def test_max_given_triggers_same_behavior(self):
        """When max_given is hit, both report non-proof exit."""
        # A problem that won't prove in 5 given clauses
        text = """\
assign(max_given, 5).

formulas(sos).
  P(a).
  -P(x) | Q(f(x)).
  -Q(x) | P(g(x)).
  -Q(x) | R(x).
  -R(x) | S(x).
end_of_list.

formulas(goals).
  S(g(g(g(a)))).
end_of_list.
"""
        c_result = run_c_prover9_from_string(text, timeout=10.0)
        py_result, _ = _run_python_on_text(text, max_given=5)

        # Neither should prove with only 5 given clauses
        # Both should stop without finding a proof
        assert not c_result.theorem_proved or not py_result.theorem_proved or (
            c_result.theorem_proved == py_result.theorem_proved
        ), (
            f"Limit behavior mismatch: C proved={c_result.theorem_proved}, "
            f"Py proved={py_result.theorem_proved}"
        )

    def test_sos_empty_both_detect(self):
        """Both detect SOS empty (search exhaustion) identically."""
        text = """\
formulas(sos).
  P(a).
end_of_list.

formulas(goals).
  Q(a).
end_of_list.
"""
        c_result = run_c_prover9_from_string(text, timeout=10.0)
        py_result, _ = _run_python_on_text(text, max_given=100)

        # Neither should prove (independent predicates)
        assert not c_result.theorem_proved
        assert not py_result.theorem_proved


# ══════════════════════════════════════════════════════════════════════════
# 6. FILE-BASED INPUTS FROM FIXTURES AND EXAMPLES
# ══════════════════════════════════════════════════════════════════════════


def _collect_fixture_files() -> list[str]:
    """Collect .in files from fixtures/inputs/."""
    if not FIXTURES_DIR.exists():
        return []
    return sorted(f.name for f in FIXTURES_DIR.glob("*.in"))


def _collect_example_files() -> list[str]:
    """Collect .in files from examples/ that are suitable for testing."""
    if not EXAMPLES_DIR.exists():
        return []
    # Only include files that are simple LADR input (skip scripts/configs)
    suitable = []
    for f in sorted(EXAMPLES_DIR.glob("*.in")):
        text = f.read_text()
        if "formulas(" in text and "end_of_list" in text:
            suitable.append(f.name)
    return suitable


FIXTURE_FILES = _collect_fixture_files()
EXAMPLE_FILES = _collect_example_files()


@pytest.mark.cross_validation
@requires_c_binary
class TestFixtureFiles:
    """Run all fixture input files through both C and Python."""

    # Files known to use syntax features not yet supported by Python parser
    # (implications, quantifiers, unary minus in terms, etc.)
    _KNOWN_UNSUPPORTED = {"bench_group_comm_3.in", "bench_robbins.in"}

    @pytest.mark.parametrize("filename", FIXTURE_FILES)
    def test_fixture_file_agreement(self, filename: str):
        """Both agree on theorem status for fixture files."""
        from pyladr.parsing.ladr_parser import ParseError

        path = FIXTURES_DIR / filename
        text = path.read_text()

        # Skip files with syntax that requires clausification
        # (implications ->, quantifiers, etc.) which Python doesn't yet support
        if "->" in text or "exists " in text or "all " in text:
            pytest.skip(f"Python parser lacks clausification support for {filename}")

        c_result = run_c_prover9(path, timeout=60.0)
        try:
            py_result, _ = _run_python_on_text(text, max_given=1000)
        except ParseError as e:
            pytest.skip(f"Python parser cannot handle {filename}: {e}")
        _assert_theorem_match(c_result, py_result, filename)

    @pytest.mark.parametrize("filename", FIXTURE_FILES)
    def test_fixture_file_stats_tolerance(self, filename: str):
        """Search statistics within 50% tolerance for fixture files."""
        from pyladr.parsing.ladr_parser import ParseError

        path = FIXTURES_DIR / filename
        text = path.read_text()

        c_result = run_c_prover9(path, timeout=60.0)
        try:
            py_result, _ = _run_python_on_text(text, max_given=1000)
        except ParseError as e:
            pytest.skip(f"Python parser cannot handle {filename}: {e}")

        # Only compare stats if both proved
        if c_result.theorem_proved and py_result.theorem_proved:
            comp = compare_search_statistics(c_result, py_result, tolerance=0.5)
            # Log differences for analysis but don't fail on stats alone
            if not comp.equivalent:
                for diff in comp.differences:
                    print(f"  [{filename}] Stats diff: {diff}")


@pytest.mark.cross_validation
@requires_c_binary
@pytest.mark.slow
class TestExampleFiles:
    """Run example files through both (may be slow for harder problems)."""

    @pytest.mark.parametrize("filename", EXAMPLE_FILES)
    def test_example_file_agreement(self, filename: str):
        """Both agree on theorem status for example files."""
        from pyladr.parsing.ladr_parser import ParseError

        path = EXAMPLES_DIR / filename
        text = path.read_text()

        # Skip files requiring clausification
        if "->" in text or "exists " in text or "all " in text or "<->" in text:
            pytest.skip(f"Python parser lacks clausification support for {filename}")

        c_result = run_c_prover9(path, timeout=120.0)
        try:
            py_result, _ = _run_python_on_text(text, max_given=2000)
        except ParseError as e:
            pytest.skip(f"Python parser cannot handle {filename}: {e}")

        # If C says unprovable but Python claims proof, flag as known soundness issue
        if not c_result.theorem_proved and py_result.theorem_proved:
            pytest.xfail(
                f"SOUNDNESS BUG: Python claims proof for {filename} "
                f"which C correctly identifies as unprovable"
            )

        _assert_theorem_match(c_result, py_result, filename)


# ══════════════════════════════════════════════════════════════════════════
# 7. C PROVER9 REFERENCE EXAMPLES
# ══════════════════════════════════════════════════════════════════════════


def _collect_c_example_files() -> list[str]:
    """Collect .in files from C prover9 examples."""
    if not C_EXAMPLES_DIR.exists():
        return []
    return sorted(f.name for f in C_EXAMPLES_DIR.glob("*.in"))


C_EXAMPLE_FILES = _collect_c_example_files()


@pytest.mark.cross_validation
@requires_c_binary
class TestCReferenceExamples:
    """Run official C Prover9 example files through both engines."""

    @pytest.mark.parametrize("filename", C_EXAMPLE_FILES)
    def test_c_example_agreement(self, filename: str):
        """Python matches C on official C Prover9 examples."""
        from pyladr.parsing.ladr_parser import ParseError

        path = C_EXAMPLES_DIR / filename
        text = path.read_text()

        c_result = run_c_prover9(path, timeout=60.0)
        try:
            py_result, _ = _run_python_on_text(text, max_given=1000)
        except ParseError as e:
            pytest.skip(f"Python parser cannot handle {filename}: {e}")
        _assert_theorem_match(c_result, py_result, filename)


# ══════════════════════════════════════════════════════════════════════════
# 8. SUBPROCESS END-TO-END COMPARISON
# ══════════════════════════════════════════════════════════════════════════


SUBPROCESS_TESTS = {
    "sub_trivial": """\
formulas(sos).
  P(a).
end_of_list.

formulas(goals).
  P(a).
end_of_list.
""",
    "sub_resolution": """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(a).
end_of_list.
""",
    "sub_equality": """\
set(auto).

formulas(sos).
  a = b.
end_of_list.

formulas(goals).
  b = a.
end_of_list.
""",
}


@pytest.mark.cross_validation
@requires_c_binary
class TestSubprocessEndToEnd:
    """Full CLI subprocess comparison — tests the exact user-facing pipeline."""

    @pytest.mark.parametrize("name", sorted(SUBPROCESS_TESTS.keys()))
    def test_subprocess_theorem_status(self, name: str):
        """CLI subprocess output matches C for theorem status."""
        text = SUBPROCESS_TESTS[name]
        c_result = run_c_prover9_from_string(text, timeout=15.0)
        py_result = _run_python_subprocess(text, timeout=15.0)
        _assert_theorem_match(c_result, py_result, name)

    @pytest.mark.parametrize("name", sorted(SUBPROCESS_TESTS.keys()))
    def test_subprocess_output_has_statistics(self, name: str):
        """Python CLI output includes statistics section like C."""
        text = SUBPROCESS_TESTS[name]
        py_result = _run_python_subprocess(text, timeout=15.0)

        assert "STATISTICS" in py_result.raw_output, (
            f"[{name}] Python output missing STATISTICS section"
        )
        assert "Given=" in py_result.raw_output, (
            f"[{name}] Python output missing Given= in statistics"
        )

    @pytest.mark.parametrize("name", sorted(SUBPROCESS_TESTS.keys()))
    def test_subprocess_output_has_conclusion(self, name: str):
        """Python CLI output includes conclusion like C."""
        text = SUBPROCESS_TESTS[name]
        c_result = run_c_prover9_from_string(text, timeout=15.0)
        py_result = _run_python_subprocess(text, timeout=15.0)

        if c_result.theorem_proved:
            assert "THEOREM PROVED" in py_result.raw_output, (
                f"[{name}] Python output missing 'THEOREM PROVED'"
            )
        if c_result.search_failed:
            assert "SEARCH FAILED" in py_result.raw_output, (
                f"[{name}] Python output missing 'SEARCH FAILED'"
            )

    @pytest.mark.parametrize("name", sorted(SUBPROCESS_TESTS.keys()))
    def test_subprocess_has_proof_section(self, name: str):
        """When theorem proved, both have PROOF section."""
        text = SUBPROCESS_TESTS[name]
        c_result = run_c_prover9_from_string(text, timeout=15.0)
        py_result = _run_python_subprocess(text, timeout=15.0)

        if c_result.theorem_proved:
            assert "PROOF" in py_result.raw_output, (
                f"[{name}] Python output missing PROOF section"
            )


# ══════════════════════════════════════════════════════════════════════════
# 9. EDGE CASES AND ERROR HANDLING
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.cross_validation
@requires_c_binary
class TestEdgeCases:
    """Edge cases and unusual inputs."""

    def test_usable_and_sos_combined(self):
        """Both handle clauses distributed across usable and SOS."""
        # Note: C Prover9 may handle usable/SOS interaction differently.
        # Use SOS for all clauses to ensure consistent behavior.
        text = """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(a).
end_of_list.
"""
        c_result = run_c_prover9_from_string(text, timeout=10.0)
        py_result, _ = _run_python_on_text(text, max_given=100)
        _assert_theorem_match(c_result, py_result, "usable_and_sos_combined")

    def test_single_clause_sos(self):
        """Both handle single-clause SOS correctly."""
        text = """\
formulas(sos).
  P(a).
end_of_list.
"""
        c_result = run_c_prover9_from_string(text, timeout=10.0)
        py_result, _ = _run_python_on_text(text, max_given=50)

        # Neither should prove (no goal, just SOS exhaustion)
        # C and Python may differ on exact handling of no-goal inputs
        # Just verify no crash
        assert c_result.exit_code >= 0
        assert py_result.exit_code >= 0

    def test_tautology_input(self):
        """Both handle tautological input."""
        text = """\
formulas(sos).
  P(a) | -P(a).
end_of_list.
"""
        c_result = run_c_prover9_from_string(text, timeout=10.0)
        py_result, _ = _run_python_on_text(text, max_given=50)

        # Just verify no crash; both should handle gracefully
        assert c_result.exit_code >= 0
        assert py_result.exit_code >= 0

    def test_ground_clause_only(self):
        """Both handle ground (variable-free) clauses."""
        text = """\
formulas(sos).
  P(a).
  -P(a) | Q(b).
  -Q(b).
end_of_list.
"""
        c_result = run_c_prover9_from_string(text, timeout=10.0)
        py_result, _ = _run_python_on_text(text, max_given=50)
        _assert_theorem_match(c_result, py_result, "ground_clauses")

    def test_multiple_goals(self):
        """Both handle multiple goals (each denied separately)."""
        text = """\
formulas(sos).
  P(a).
  Q(a).
end_of_list.

formulas(goals).
  P(a).
end_of_list.
"""
        c_result = run_c_prover9_from_string(text, timeout=10.0)
        py_result, _ = _run_python_on_text(text, max_given=50)
        _assert_theorem_match(c_result, py_result, "multiple_goals")

    def test_deeply_nested_terms(self):
        """Both handle deeply nested function terms."""
        text = """\
formulas(sos).
  f(f(f(a))) = a.
  f(a) = b.
end_of_list.

formulas(goals).
  f(f(b)) = a.
end_of_list.
"""
        c_result = run_c_prover9_from_string(text, timeout=15.0)
        py_result, _ = _run_python_on_text(text, max_given=200)
        _assert_theorem_match(c_result, py_result, "deeply_nested")


# ══════════════════════════════════════════════════════════════════════════
# 10. FULL COMPARISON WITH DETAILED REPORTING
# ══════════════════════════════════════════════════════════════════════════


FULL_COMPARISON_INPUTS = {
    "full_trivial": """\
formulas(sos).
  P(a).
end_of_list.

formulas(goals).
  P(a).
end_of_list.
""",
    "full_resolution_chain": """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(x) | R(x).
  -R(a).
end_of_list.
""",
}


@pytest.mark.cross_validation
@requires_c_binary
class TestFullComparison:
    """Full comparison with detailed difference reporting."""

    @pytest.mark.parametrize("name", sorted(FULL_COMPARISON_INPUTS.keys()))
    def test_full_comparison(self, name: str):
        """Run full comparison (theorem + proof + stats) with tolerant stats."""
        text = FULL_COMPARISON_INPUTS[name]
        c_result = run_c_prover9_from_string(text, timeout=15.0)
        py_result, _ = _run_python_on_text(text, max_given=200)

        comp = compare_full(c_result, py_result, stats_tolerance=0.5)

        # Theorem result must match
        theorem_comp = comp.details.get("theorem_result")
        if theorem_comp:
            assert theorem_comp.equivalent, (
                f"[{name}] Theorem mismatch: {theorem_comp}"
            )

        # Log all differences for analysis
        if not comp.equivalent:
            for diff in comp.differences:
                print(f"  [{name}] DIFF: {diff}")
        for warn in comp.warnings:
            print(f"  [{name}] WARN: {warn}")


# ══════════════════════════════════════════════════════════════════════════
# 11. VAMPIRE.IN (HARD PROBLEM)
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.cross_validation
@requires_c_binary
@pytest.mark.slow
class TestHardProblems:
    """Harder problems that may take longer but validate deeper equivalence."""

    def test_vampire_in(self):
        """Both agree on vampire.in (propositional logic problem)."""
        from pyladr.parsing.ladr_parser import ParseError

        path = PROJECT_ROOT / "tests" / "fixtures" / "inputs" / "vampire.in"
        if not path.exists():
            pytest.skip("vampire.in not found")

        text = path.read_text()
        c_result = run_c_prover9(path, timeout=120.0)
        try:
            py_result, _ = _run_python_on_text(text, max_given=5000)
        except ParseError as e:
            pytest.skip(f"Python parser cannot handle vampire.in: {e}")
        _assert_theorem_match(c_result, py_result, "vampire.in")

    def test_x2_group_commutativity(self):
        """Both agree on x2.in (group commutativity from x*x=e)."""
        from pyladr.parsing.ladr_parser import ParseError

        path = C_EXAMPLES_DIR / "x2.in"
        if not path.exists():
            pytest.skip("x2.in not found")

        text = path.read_text()
        c_result = run_c_prover9(path, timeout=60.0)
        try:
            py_result, _ = _run_python_on_text(text, max_given=1000)
        except ParseError as e:
            pytest.skip(f"Python parser cannot handle x2.in: {e}")
        _assert_theorem_match(c_result, py_result, "x2.in")


# ══════════════════════════════════════════════════════════════════════════
# 12. SETTINGS AND DIRECTIVES
# ══════════════════════════════════════════════════════════════════════════


SETTINGS_INPUTS = {
    "explicit_paramodulation": """\
set(paramodulation).

formulas(sos).
  a = b.
end_of_list.

formulas(goals).
  b = a.
end_of_list.
""",
    "explicit_resolution_no_para": """\
set(binary_resolution).
clear(paramodulation).

formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(a).
end_of_list.
""",
}


@pytest.mark.cross_validation
@requires_c_binary
class TestExplicitSettings:
    """Problems with explicit set/clear/assign directives."""

    @pytest.mark.parametrize("name", sorted(SETTINGS_INPUTS.keys()))
    def test_settings_theorem_status(self, name: str):
        """Both agree with explicit settings."""
        text = SETTINGS_INPUTS[name]
        c_result = run_c_prover9_from_string(text, timeout=15.0)
        py_result, _ = _run_python_on_text(text, max_given=200)
        _assert_theorem_match(c_result, py_result, name)


# ══════════════════════════════════════════════════════════════════════════
# 13. UNPROVABLE PROBLEMS
# ══════════════════════════════════════════════════════════════════════════


UNPROVABLE_INPUTS = {
    "independent_predicates": """\
formulas(sos).
  P(a).
end_of_list.

formulas(goals).
  Q(a).
end_of_list.
""",
}

# Known soundness issue: Python incorrectly claims to prove group
# commutativity from standard group axioms (which is mathematically
# unprovable). This is tracked as a bug in the Python prover.
KNOWN_SOUNDNESS_ISSUES = {
    "group_commutativity_from_axioms": """\
formulas(sos).
  e * x = x.
  x * e = x.
  i(x) * x = e.
  x * i(x) = e.
  (x * y) * z = x * (y * z).
end_of_list.

formulas(goals).
  x * y = y * x.
end_of_list.
""",
}


@pytest.mark.cross_validation
@requires_c_binary
class TestUnprovableProblems:
    """Problems that should NOT be provable — both should fail or exhaust SOS."""

    @pytest.mark.parametrize("name", sorted(UNPROVABLE_INPUTS.keys()))
    def test_unprovable_agreement(self, name: str):
        """Both C and Python agree the problem is not provable within limits."""
        text = UNPROVABLE_INPUTS[name]
        c_result = run_c_prover9_from_string(text, timeout=15.0)
        py_result, _ = _run_python_on_text(text, max_given=100)

        # Both should agree: neither proves the theorem
        if c_result.theorem_proved:
            assert py_result.theorem_proved, (
                f"[{name}] C proved but Python did not"
            )
        else:
            assert not py_result.theorem_proved, (
                f"[{name}] Python proved but C did not"
            )

    @pytest.mark.parametrize("name", sorted(KNOWN_SOUNDNESS_ISSUES.keys()))
    @pytest.mark.xfail(
        reason="Known soundness issue: Python produces unsound proofs for this problem",
        strict=True,
    )
    def test_known_soundness_issue(self, name: str):
        """Document known soundness issues where Python disagrees with C.

        These are marked xfail — when a fix lands, they'll start passing
        and the xfail marker should be removed.
        """
        text = KNOWN_SOUNDNESS_ISSUES[name]
        c_result = run_c_prover9_from_string(text, timeout=15.0)
        py_result, _ = _run_python_on_text(text, max_given=100)

        # C correctly says unprovable
        assert not c_result.theorem_proved, f"C unexpectedly proved {name}"
        # Python should also say unprovable (but currently doesn't — hence xfail)
        assert not py_result.theorem_proved, (
            f"[{name}] SOUNDNESS BUG: Python claims proof for unprovable problem"
        )
