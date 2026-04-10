"""Tests for auto-inference cascade and Horn problem detection (REQ-R003).

Validates the complete auto-inference decision tree:
- Problem analysis (_analyze_problem)
- Depth difference computation (_neg_pos_depth_difference)
- Auto-inference rule activation (_apply_settings)
- Auto-inference trace messages in stdout
- Hyper-resolution end-to-end proof finding

See tests/AUTO_INFERENCE_TEST_STRATEGY.md for full design.
"""

from __future__ import annotations

import io
import re
from unittest.mock import patch

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.apps.prover9 import _analyze_problem, _apply_settings, _neg_pos_depth_difference
from pyladr.parsing.ladr_parser import LADRParser, parse_input
from pyladr.search.given_clause import (
    ExitCode,
    GivenClauseSearch,
    SearchOptions,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _analyze(text: str) -> tuple[dict, SearchOptions, str]:
    """Parse input, run auto-cascade, return (analysis, opts, stdout).

    Returns analysis dict, configured SearchOptions, and captured stdout.
    """
    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(text)
    all_clauses = list(parsed.sos) + list(parsed.goals) + list(parsed.usable)
    is_horn, has_eq, all_units = _analyze_problem(all_clauses, st)

    opts = SearchOptions()
    captured = io.StringIO()
    with patch("sys.stdout", captured):
        _apply_settings(parsed, opts, st)

    analysis = {"is_horn": is_horn, "has_equality": has_eq, "all_units": all_units}
    return analysis, opts, captured.getvalue()


def _get_depth_diff(text: str) -> int:
    """Parse input and compute neg-pos depth difference."""
    st = SymbolTable()
    parsed = parse_input(text, st)
    all_clauses = list(parsed.sos) + list(parsed.goals) + list(parsed.usable)
    return _neg_pos_depth_difference(all_clauses)


def _run_and_capture(
    text: str,
    max_given: int = 200,
    max_seconds: float = 10.0,
    **kwargs,
) -> tuple[str, ExitCode]:
    """Run search via _apply_settings and capture stdout."""
    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(text)

    # Deny goals
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

    opts = SearchOptions(
        max_given=max_given,
        max_seconds=max_seconds,
        **kwargs,
    )
    _apply_settings(parsed, opts, st)

    engine = GivenClauseSearch(opts)
    captured = io.StringIO()
    with patch("sys.stdout", captured):
        result = engine.run(usable=[], sos=sos)

    return captured.getvalue(), result.exit_code


# ── Test inputs ──────────────────────────────────────────────────────────────

# Horn Non-Equality (vampire.in pattern)
VAMPIRE_INPUT = """\
formulas(sos).
-P(x) | -P(i(x,y)) | P(y).
P(i(i(x,y),i(i(y,z),i(x,z)))).
P(i(i(n(x),x),x)).
P(i(x,i(n(x),y))).
end_of_list.
formulas(goals).
P(i(x,x)).
end_of_list.
"""

# Unit equality (x2 problem)
UNIT_EQUALITY_INPUT = """\
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

# Non-Horn, no equality
NONHORN_INPUT = """\
formulas(sos).
P(a) | Q(a).
-P(x) | R(x).
-Q(x) | R(x).
end_of_list.
formulas(goals).
R(a).
end_of_list.
"""

# Non-unit Horn with equality
NONUNIT_HORN_EQ_INPUT = """\
formulas(sos).
f(e,x) = x.
f(x,e) = x.
f(f(x,y),z) = f(x,f(y,z)).
-P(x) | f(x,e) = x.
P(a).
end_of_list.
formulas(goals).
f(a,e) = a.
end_of_list.
"""

# Simple Horn for modus ponens
SIMPLE_HORN_INPUT = """\
formulas(sos).
-P(x) | Q(x).
P(a).
end_of_list.
formulas(goals).
Q(a).
end_of_list.
"""

# HNE with flat depth (depth_diff = 0)
HNE_FLAT_INPUT = """\
formulas(sos).
-P(x) | Q(x).
P(a).
end_of_list.
formulas(goals).
Q(a).
end_of_list.
"""

# HNE with deep negative literals
HNE_DEEP_NEG_INPUT = """\
formulas(sos).
-P(f(f(x))) | Q(x).
P(f(f(a))).
end_of_list.
formulas(goals).
Q(a).
end_of_list.
"""


# ── TestProblemAnalysis ──────────────────────────────────────────────────────


class TestProblemAnalysis:
    """Test _analyze_problem correctness."""

    def test_unit_equality(self):
        """Unit equality: Horn, has equality, all units."""
        analysis, _, _ = _analyze(UNIT_EQUALITY_INPUT)
        assert analysis["is_horn"] is True
        assert analysis["has_equality"] is True
        assert analysis["all_units"] is True

    def test_nonhorn_no_equality(self):
        """Non-Horn disjunction: not Horn, no equality."""
        analysis, _, _ = _analyze(NONHORN_INPUT)
        assert analysis["is_horn"] is False
        assert analysis["has_equality"] is False

    def test_horn_no_equality(self):
        """Pure Horn, no equality (vampire-style)."""
        analysis, _, _ = _analyze(VAMPIRE_INPUT)
        assert analysis["is_horn"] is True
        assert analysis["has_equality"] is False

    def test_nonunit_horn_equality(self):
        """Non-unit Horn with equality."""
        analysis, _, _ = _analyze(NONUNIT_HORN_EQ_INPUT)
        assert analysis["is_horn"] is True
        assert analysis["has_equality"] is True
        assert analysis["all_units"] is False

    def test_simple_horn_no_equality(self):
        """Simple Horn clause set, no equality."""
        analysis, _, _ = _analyze(SIMPLE_HORN_INPUT)
        assert analysis["is_horn"] is True
        assert analysis["has_equality"] is False

    def test_negative_equality_not_counted(self):
        """Negative equality literals should NOT trigger has_equality."""
        text = """\
formulas(sos).
  -(x = y) | P(x,y).
  a = b.
end_of_list.
"""
        analysis, _, _ = _analyze(text)
        # a = b is positive equality, so has_equality should be True
        assert analysis["has_equality"] is True

    def test_only_negative_equality(self):
        """Only negative equality → has_equality is False."""
        text = """\
formulas(sos).
  -(a = b).
end_of_list.
"""
        analysis, _, _ = _analyze(text)
        assert analysis["has_equality"] is False


# ── TestDepthDifference ──────────────────────────────────────────────────────


class TestDepthDifference:
    """Test _neg_pos_depth_difference computation."""

    def test_vampire_style_positive_diff(self):
        """vampire.in pattern: negative literals are deeper → diff > 0."""
        diff = _get_depth_diff(VAMPIRE_INPUT)
        assert diff > 0, f"Expected positive depth_diff for vampire-style HNE, got {diff}"

    def test_flat_mixed_clause(self):
        """Flat mixed clause: -P(x)|Q(x) → depth_diff = 0."""
        diff = _get_depth_diff(HNE_FLAT_INPUT)
        assert diff == 0

    def test_deep_negative_literals(self):
        """Deep negative literals → positive depth_diff."""
        diff = _get_depth_diff(HNE_DEEP_NEG_INPUT)
        assert diff > 0

    def test_pure_positive_ignored(self):
        """Pure positive clauses have no mixed clauses → diff = 0."""
        text = """\
formulas(sos).
P(a).
Q(f(f(b))).
end_of_list.
"""
        diff = _get_depth_diff(text)
        assert diff == 0

    def test_pure_negative_ignored(self):
        """Pure negative clauses are not mixed → diff = 0."""
        text = """\
formulas(sos).
-P(a).
end_of_list.
"""
        diff = _get_depth_diff(text)
        assert diff == 0

    def test_deep_positive_gives_negative_diff(self):
        """Deep positive, shallow negative → negative diff."""
        text = """\
formulas(sos).
-P(a) | Q(f(f(f(a)))).
end_of_list.
"""
        diff = _get_depth_diff(text)
        assert diff < 0, f"Expected negative depth_diff, got {diff}"


# ── TestAutoCascadeDecisions ─────────────────────────────────────────────────


class TestAutoCascadeDecisions:
    """Test _apply_settings produces correct inference rule configuration."""

    def test_unit_equality_only_para(self):
        """Unit equality: only paramodulation, no hyper or binary."""
        _, opts, _ = _analyze(UNIT_EQUALITY_INPUT)
        assert opts.paramodulation is True
        assert opts.demodulation is True
        assert opts.hyper_resolution is False
        assert opts.binary_resolution is False

    def test_hne_depth_positive_hyper(self):
        """HNE with depth_diff > 0 → hyper_resolution."""
        _, opts, _ = _analyze(VAMPIRE_INPUT)
        assert opts.hyper_resolution is True
        assert opts.binary_resolution is False
        assert opts.paramodulation is False

    def test_hne_depth_zero_binary(self):
        """HNE with depth_diff = 0 → binary_resolution."""
        _, opts, _ = _analyze(HNE_FLAT_INPUT)
        assert opts.binary_resolution is True
        assert opts.hyper_resolution is False

    def test_nonhorn_binary(self):
        """Non-Horn → binary_resolution."""
        _, opts, _ = _analyze(NONHORN_INPUT)
        assert opts.binary_resolution is True
        assert opts.hyper_resolution is False

    def test_nonunit_horn_eq_both(self):
        """Non-unit Horn + equality → para + hyper."""
        _, opts, _ = _analyze(NONUNIT_HORN_EQ_INPUT)
        assert opts.paramodulation is True
        assert opts.hyper_resolution is True
        assert opts.binary_resolution is False

    def test_nonhorn_factoring_enabled(self):
        """Non-Horn auto_process enables factoring."""
        _, opts, _ = _analyze(NONHORN_INPUT)
        assert opts.factoring is True

    def test_auto_limits_applied(self):
        """Auto-limits set max_weight=100, sos_limit=20000."""
        _, opts, _ = _analyze(SIMPLE_HORN_INPUT)
        assert opts.max_weight == 100.0
        assert opts.sos_limit == 20000

    def test_explicit_max_weight_preserved(self):
        """Explicit max_weight is not overridden by auto."""
        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(SIMPLE_HORN_INPUT)
        opts = SearchOptions(max_weight=50.0)  # explicit
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            _apply_settings(parsed, opts, st)
        assert opts.max_weight == 50.0


# ── TestAutoInferenceOutput ──────────────────────────────────────────────────


class TestAutoInferenceOutput:
    """Test that auto-inference trace messages appear in stdout."""

    def test_auto_inference_header(self):
        """Auto_inference settings header appears."""
        _, _, output = _analyze(UNIT_EQUALITY_INPUT)
        assert "Auto_inference settings:" in output

    def test_auto_process_header(self):
        """Auto_process settings header appears."""
        _, _, output = _analyze(UNIT_EQUALITY_INPUT)
        assert "Auto_process settings:" in output

    def test_paramodulation_message(self):
        """Equality input shows paramodulation message."""
        _, _, output = _analyze(UNIT_EQUALITY_INPUT)
        assert "set(paramodulation)" in output
        assert "positive equality literals" in output

    def test_hyper_resolution_message_hne(self):
        """HNE with depth>0 shows hyper_resolution message."""
        _, _, output = _analyze(VAMPIRE_INPUT)
        assert "set(hyper_resolution)" in output
        assert "HNE" in output

    def test_binary_resolution_message_nonhorn(self):
        """Non-Horn shows binary_resolution message."""
        _, _, output = _analyze(NONHORN_INPUT)
        assert "set(binary_resolution)" in output
        assert "non-Horn" in output

    def test_depth_diff_shown_in_hne(self):
        """HNE message includes depth_diff value."""
        _, _, output = _analyze(VAMPIRE_INPUT)
        assert re.search(r"depth_diff=\d+", output), (
            f"Expected depth_diff=N in output:\n{output}"
        )

    def test_quiet_suppresses_auto_messages(self):
        """quiet=True suppresses auto-inference messages."""
        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(UNIT_EQUALITY_INPUT)
        opts = SearchOptions(quiet=True)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            _apply_settings(parsed, opts, st)
        assert "Auto_inference" not in captured.getvalue()


# ── TestHyperResolutionEndToEnd ──────────────────────────────────────────────


class TestHyperResolutionEndToEnd:
    """Test hyper-resolution actually finds proofs."""

    def test_simple_horn_proof_via_hyper(self):
        """Simple Horn problem: -P(x)|Q(x), P(a) → Q(a) via hyper-resolution."""
        output, exit_code = _run_and_capture(
            SIMPLE_HORN_INPUT,
            hyper_resolution=True,
            binary_resolution=False,
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT, (
            f"Expected proof via hyper-resolution, got {exit_code}"
        )

    def test_modus_ponens_chain(self):
        """Chain: -P(x)|Q(x), -Q(x)|R(x), P(a) → R(a)."""
        text = """\
formulas(sos).
-P(x) | Q(x).
-Q(x) | R(x).
P(a).
end_of_list.
formulas(goals).
R(a).
end_of_list.
"""
        output, exit_code = _run_and_capture(
            text,
            hyper_resolution=True,
            binary_resolution=False,
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_multi_neg_literal_nucleus(self):
        """Multi-negative nucleus: -A|-B|C resolved with A and B."""
        text = """\
formulas(sos).
-P(x) | -Q(x) | R(x).
P(a).
Q(a).
end_of_list.
formulas(goals).
R(a).
end_of_list.
"""
        output, exit_code = _run_and_capture(
            text,
            hyper_resolution=True,
            binary_resolution=False,
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_vampire_in_proof(self):
        """vampire.in: condensed detachment finds P(i(x,x))."""
        output, exit_code = _run_and_capture(
            VAMPIRE_INPUT,
            max_given=500,
            max_seconds=30.0,
        )
        # Auto-cascade should select hyper_resolution for this HNE problem
        assert exit_code == ExitCode.MAX_PROOFS_EXIT, (
            f"Expected vampire.in to find proof, got {exit_code}"
        )

    def test_hyper_with_given_trace(self):
        """Hyper-resolution search shows given clause traces."""
        output, exit_code = _run_and_capture(
            SIMPLE_HORN_INPUT,
            hyper_resolution=True,
            binary_resolution=False,
        )
        assert "given #" in output, (
            "Given clause trace missing during hyper-resolution search"
        )

    def test_hyper_respects_max_given(self):
        """Hyper-resolution respects max_given limit."""
        output, exit_code = _run_and_capture(
            VAMPIRE_INPUT,
            max_given=5,
        )
        # Should stop at limit (may or may not find proof in 5 givens)
        assert exit_code in (
            ExitCode.MAX_PROOFS_EXIT,
            ExitCode.MAX_GIVEN_EXIT,
            ExitCode.SOS_EMPTY_EXIT,
        )


# ── TestCrossValidationReadiness ─────────────────────────────────────────────


class TestCrossValidationReadiness:
    """Smoke tests to verify C cross-validation infrastructure works.

    These test that the C runner and comparison tools are available.
    Full cross-validation tests are in tests/cross_validation/.
    """

    @pytest.fixture
    def c_binary_available(self) -> bool:
        import os
        return os.path.exists("reference-prover9/bin/prover9")

    def test_c_binary_exists(self, c_binary_available):
        """C Prover9 binary should be available for cross-validation."""
        if not c_binary_available:
            pytest.skip("C Prover9 binary not found")
        assert c_binary_available

    def test_c_runner_importable(self):
        """Cross-validation runner should be importable."""
        try:
            from tests.cross_validation.c_runner import run_c_prover9_from_string
        except ImportError:
            pytest.skip("c_runner not available")

    def test_c_reference_fixtures_exist(self):
        """C reference output fixtures should exist."""
        import os
        ref_dir = "reference-prover9/tests/fixtures/c_reference"
        if not os.path.isdir(ref_dir):
            pytest.skip("C reference fixtures not found")
        files = os.listdir(ref_dir)
        assert len(files) > 0
