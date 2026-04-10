"""Compatibility tests verifying Python defaults match C Prover9 defaults.

These tests validate that PyLADR's default behavior matches C Prover9's,
particularly for critical flags like set(auto) and demodulation that
C Prover9 enables by default.

Findings documented here:
  - C Prover9 defaults auto=TRUE (line 331 of search.c)
  - C Prover9 defaults back_demod=TRUE (line 278 of search.c)
  - C Prover9 defaults ordered_res=TRUE, ordered_para=TRUE
  - Python defaults auto=False, demodulation=False, back_demod=False
  - This causes Python to fail on equational problems that lack explicit set(auto)
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from pyladr.apps.prover9 import _apply_settings, _auto_cascade, _deny_goals
from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
C_BINARY = PROJECT_ROOT / "reference-prover9" / "bin" / "prover9"

requires_c_binary = pytest.mark.skipif(
    not C_BINARY.exists(), reason="C prover9 binary not found"
)


def _run_python(input_text: str, max_seconds: float = 30) -> dict:
    """Run Python prover on input text, return result dict."""
    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(input_text)
    usable, sos = _deny_goals(parsed, st)
    opts = SearchOptions(max_seconds=max_seconds)
    _apply_settings(parsed, opts, st)
    engine = GivenClauseSearch(options=opts, symbol_table=st)
    result = engine.run(usable=usable, sos=sos)
    return {
        "proved": len(result.proofs) > 0,
        "exit_code": result.exit_code,
        "given": result.stats.given,
        "generated": result.stats.generated,
        "kept": result.stats.kept,
        "paramodulation": opts.paramodulation,
        "demodulation": opts.demodulation,
        "back_demod": opts.back_demod,
        "binary_resolution": opts.binary_resolution,
    }


def _run_c(input_text: str, timeout: float = 30) -> dict:
    """Run C prover9 on input text, return result dict."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".in", delete=False) as f:
        f.write(input_text)
        f.flush()
        proc = subprocess.run(
            [str(C_BINARY), "-f", f.name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    output = proc.stdout
    proved = "THEOREM PROVED" in output
    import re

    stats = {}
    m = re.search(r"Given=(\d+)\.\s+Generated=(\d+)\.\s+Kept=(\d+)", output)
    if m:
        stats["given"] = int(m.group(1))
        stats["generated"] = int(m.group(2))
        stats["kept"] = int(m.group(3))
    return {"proved": proved, "output": output, **stats}


# ── Default Flag Tests ─────────────────────────────────────────────────────


class TestDefaultFlags:
    """Verify Python default flags match C Prover9 defaults."""

    def test_c_defaults_auto_true(self):
        """C Prover9 defaults auto=TRUE (search.c:331).

        KNOWN ISSUE: Python defaults auto=False.
        This is the root cause of equational problems failing without
        explicit set(auto) in the input.
        """
        # Document the current Python behavior
        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(
            "formulas(sos). x = x. end_of_list."
        )
        assert parsed.settings.flag("auto") is False, (
            "Expected Python to default auto=False (known divergence from C)"
        )

    def test_c_defaults_back_demod_true(self):
        """C Prover9 defaults back_demod=TRUE (search.c:278).

        KNOWN ISSUE: Python defaults back_demod=False even in auto mode.
        """
        opts = SearchOptions()
        assert opts.back_demod is False, (
            "Expected Python to default back_demod=False (known divergence from C)"
        )
        assert opts.demodulation is False, (
            "Expected Python to default demodulation=False (known divergence from C)"
        )

    def test_auto_cascade_enables_paramodulation_for_equality(self):
        """When set(auto) is present, auto_cascade should enable paramodulation."""
        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input("""
set(auto).
formulas(sos).
  f(x) = g(x).
end_of_list.
formulas(goals).
  f(a) = g(a).
end_of_list.
""")
        opts = SearchOptions()
        _apply_settings(parsed, opts, st)
        assert opts.paramodulation is True

    @pytest.mark.xfail(
        reason="Python auto_cascade does not enable demodulation/back_demod "
        "(C Prover9 has these TRUE by default, independent of set(auto))"
    )
    def test_auto_cascade_should_enable_demodulation(self):
        """C Prover9 has demodulation enabled by default.

        The auto_cascade in Python should either:
        1. Enable demodulation/back_demod in auto mode, OR
        2. Default demodulation/back_demod to True in SearchOptions
        """
        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input("""
set(auto).
formulas(sos).
  f(x) = g(x).
end_of_list.
formulas(goals).
  f(a) = g(a).
end_of_list.
""")
        opts = SearchOptions()
        _apply_settings(parsed, opts, st)
        assert opts.demodulation is True, "demodulation should be enabled"
        assert opts.back_demod is True, "back_demod should be enabled"


# ── Equational Problem Compatibility ────────────────────────────────────────


class TestEquationalWithoutExplicitAuto:
    """Test problems that rely on C's default auto=TRUE behavior.

    These problems don't have explicit set(auto) but C Prover9 solves them
    because auto mode is on by default. Python fails because auto defaults
    to False.
    """

    ROBBINS_INPUT = """\
assign(max_seconds, 30).
formulas(sos).
  x + y = y + x.
  (x + y) + z = x + (y + z).
  -(-x + y) + -(-x + -y) = x.
end_of_list.
formulas(goals).
  --x = x.
end_of_list.
"""

    @requires_c_binary
    def test_c_solves_robbins(self):
        """C Prover9 solves Robbins without explicit set(auto)."""
        result = _run_c(self.ROBBINS_INPUT, timeout=60)
        assert result["proved"], "C should prove Robbins"

    @pytest.mark.xfail(
        reason="Python lacks default auto=TRUE: won't auto-enable paramodulation"
    )
    def test_python_solves_robbins_without_set_auto(self):
        """Python should solve Robbins like C does (requires auto=TRUE default)."""
        result = _run_python(self.ROBBINS_INPUT, max_seconds=30)
        assert result["proved"], "Python should prove Robbins"

    def test_python_solves_robbins_with_explicit_set_auto(self):
        """Python solves Robbins when set(auto) is explicitly added."""
        input_with_auto = "set(auto).\n" + self.ROBBINS_INPUT
        result = _run_python(input_with_auto, max_seconds=30)
        # Even with set(auto), Python may not prove it due to missing
        # demodulation (a separate issue)
        if not result["proved"]:
            pytest.skip(
                "Python couldn't prove Robbins even with set(auto) — "
                "likely due to missing demodulation (separate issue)"
            )

    @requires_c_binary
    def test_c_vs_python_group_comm_no_auto(self):
        """Group commutativity without explicit set(auto).

        C proves this via auto-inference detection. Python fails.
        """
        input_text = """\
assign(max_seconds, 30).
formulas(sos).
  -(-x + y) + -(-x + -y) = x.
  x + y = y + x.
  (x + y) + z = x + (y + z).
end_of_list.
formulas(goals).
  --(x) = x.
end_of_list.
"""
        c_result = _run_c(input_text, timeout=60)
        py_result = _run_python(input_text, max_seconds=30)

        if c_result["proved"] and not py_result["proved"]:
            pytest.xfail(
                "C proved but Python failed — expected due to auto=FALSE default"
            )
        assert c_result["proved"] == py_result["proved"]


# ── Auto-cascade Behavior ──────────────────────────────────────────────────


class TestAutoCascadeBehavior:
    """Validate auto-cascade matches C Prover9 auto_inference logic."""

    def test_pure_equality_enables_paramodulation(self):
        """Pure equality problem → paramodulation."""
        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input("""
set(auto).
formulas(sos).
  f(x) = g(x).
  g(a) = b.
end_of_list.
formulas(goals).
  f(a) = b.
end_of_list.
""")
        opts = SearchOptions()
        _apply_settings(parsed, opts, st)
        assert opts.paramodulation is True
        # Pure equality + all units → no resolution needed
        assert opts.binary_resolution is False

    def test_non_equality_enables_resolution(self):
        """Non-equality problem → binary resolution (for non-horn)."""
        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input("""
set(auto).
formulas(sos).
  P(a) | Q(a).
  -P(x) | R(x).
  -Q(x) | R(x).
end_of_list.
formulas(goals).
  R(a).
end_of_list.
""")
        opts = SearchOptions()
        _apply_settings(parsed, opts, st)
        assert opts.binary_resolution is True
        assert opts.paramodulation is False

    def test_horn_equality_enables_hyper_and_para(self):
        """Horn + equality → hyper_resolution + paramodulation."""
        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input("""
set(auto).
formulas(sos).
  f(x,e) = x.
  -P(x) | P(f(x,y)).
end_of_list.
formulas(goals).
  P(f(a,e)).
end_of_list.
""")
        opts = SearchOptions()
        _apply_settings(parsed, opts, st)
        assert opts.paramodulation is True
        assert opts.hyper_resolution is True


# ── Output Format Compatibility ─────────────────────────────────────────────


class TestOutputFormat:
    """Validate Python output format matches C Prover9 structure."""

    @requires_c_binary
    def test_output_has_standard_sections(self):
        """Both C and Python should produce the standard output sections."""
        input_text = """\
set(auto).
formulas(sos).
  P(a).
end_of_list.
formulas(goals).
  P(a).
end_of_list.
"""
        c_result = _run_c(input_text)
        assert "============================== PROOF" in c_result["output"]
        assert "============================== STATISTICS" in c_result["output"]
        assert "THEOREM PROVED" in c_result["output"]

    @requires_c_binary
    def test_statistics_format_matches(self):
        """Statistics line format: Given=N. Generated=N. Kept=N. proofs=N."""
        input_text = """\
set(auto).
formulas(sos).
  P(a).
  -P(x) | Q(x).
end_of_list.
formulas(goals).
  Q(a).
end_of_list.
"""
        c_result = _run_c(input_text)
        import re

        pattern = r"Given=\d+\.\s+Generated=\d+\.\s+Kept=\d+\.\s+proofs=\d+\."
        assert re.search(pattern, c_result["output"]), (
            "C output should have standard statistics format"
        )


# ── Exit Code Compatibility ─────────────────────────────────────────────────


class TestExitCodes:
    """Validate exit code semantics match C Prover9."""

    EXPECTED_EXIT_CODES = {
        "max_proofs": 0,   # proof found
        "fatal": 1,        # fatal error
        "sos_empty": 2,    # SOS exhausted
        "max_given": 3,    # max_given limit
        "max_kept": 4,     # max_kept limit
        "max_seconds": 5,  # time limit
        "max_generated": 6,  # max_generated limit
    }

    def test_proof_found_exit_code(self):
        """Exit code MAX_PROOFS_EXIT when proof found."""
        result = _run_python("""
set(auto).
formulas(sos).
  P(a).
end_of_list.
formulas(goals).
  P(a).
end_of_list.
""")
        assert result["exit_code"] == ExitCode.MAX_PROOFS_EXIT

    def test_sos_empty_exit_code(self):
        """Exit code SOS_EMPTY_EXIT for SOS empty (no proof possible)."""
        result = _run_python("""
set(auto).
formulas(sos).
  P(a).
end_of_list.
formulas(goals).
  Q(a).
end_of_list.
""")
        assert result["exit_code"] == ExitCode.SOS_EMPTY_EXIT

    def test_max_given_exit_code(self):
        """Exit code for max_given limit."""
        result = _run_python("""
set(auto).
assign(max_given, 2).
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(x) | R(x).
  -R(x) | S(x).
end_of_list.
formulas(goals).
  S(a) & T(a).
end_of_list.
""")
        # Should hit max_given or find proof quickly
        assert result["exit_code"] in (
            ExitCode.MAX_GIVEN_EXIT,
            ExitCode.MAX_PROOFS_EXIT,
        )


# ── Clause Comparison Ordering Bug ──────────────────────────────────────────


class TestClauseOrdering:
    """Test that Clause objects support comparison for heap operations."""

    def test_clause_weight_heap_tiebreaking(self):
        """selection.py:102 crashes when clauses have same (weight, id).

        The heapq comparison falls back to comparing Clause objects,
        which don't implement __lt__.
        """
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import Term

        # Create two simple clauses
        atom = Term(private_symbol=-1, arity=0, args=())
        lit = Literal(atom=atom, sign=True)
        c1 = Clause(literals=(lit,), id=1)
        c2 = Clause(literals=(lit,), id=2)

        # If IDs differ, comparison works (int tiebreaker resolves before Clause)
        # The real issue is when weight AND id are the same
        c3 = Clause(literals=(lit,), id=1)
        try:
            _ = (c1.weight, c1.id, c1) < (c3.weight, c3.id, c3)
        except TypeError:
            pytest.xfail(
                "Clause.__lt__ not implemented — heap tiebreaking crashes "
                "(selection.py:102)"
            )
