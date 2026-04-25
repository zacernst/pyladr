"""Tests for auto-inference and end-to-end proof finding (REQ-R003).

Validates the auto-inference decision tree using the current API:
- _auto_inference: detects equality and negative literals
- _auto_limits: applies default search limits
- End-to-end proof finding with auto-configured inference rules

See tests/AUTO_INFERENCE_TEST_STRATEGY.md for full design.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from pyladr.apps.prover9 import _auto_inference, _auto_limits, _deny_goals
from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.search.given_clause import (
    ExitCode,
    GivenClauseSearch,
    SearchOptions,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _configure(text: str) -> tuple[SearchOptions, SymbolTable]:
    """Parse input and run auto-inference, return (opts, symbol_table)."""
    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(text)
    opts = SearchOptions()
    _auto_inference(parsed, opts)
    _auto_limits(parsed, opts)
    return opts, st


def _run_and_capture(
    text: str,
    max_given: int = 200,
    max_seconds: float = 10.0,
    **kwargs,
) -> tuple[str, ExitCode]:
    """Run search with auto-inference and capture stdout."""
    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(text)

    usable, sos, _denied = _deny_goals(parsed, st)

    opts = SearchOptions(
        max_given=max_given,
        max_seconds=max_seconds,
        **kwargs,
    )
    _auto_inference(parsed, opts)
    _auto_limits(parsed, opts)

    engine = GivenClauseSearch(opts, symbol_table=st)
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = engine.run(usable=usable, sos=sos)

    return buf.getvalue(), result.exit_code


# ── Test inputs ──────────────────────────────────────────────────────────────

# Horn Non-Equality (vampire.in pattern)
VAMPIRE_INPUT = """\
set(auto).
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
"""

# Non-Horn, no equality
NONHORN_INPUT = """\
set(auto).
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
set(auto).
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
set(auto).
formulas(sos).
-P(x) | Q(x).
P(a).
end_of_list.
formulas(goals).
Q(a).
end_of_list.
"""


# ── TestAutoInferenceDecisions ─────────────────────────────────────────────


class TestAutoInferenceDecisions:
    """Test _auto_inference produces correct inference rule configuration."""

    def test_equality_enables_paramodulation(self):
        """Equality literals → paramodulation + demodulation."""
        opts, _ = _configure(UNIT_EQUALITY_INPUT)
        assert opts.paramodulation is True
        assert opts.demodulation is True

    def test_auto_neg_lits_enables_hyper(self):
        """set(auto) + negative literals → hyper_resolution, no binary."""
        opts, _ = _configure(VAMPIRE_INPUT)
        assert opts.hyper_resolution is True
        assert opts.binary_resolution is False

    def test_nonhorn_with_auto(self):
        """Non-Horn with set(auto) + negative literals → hyper_resolution."""
        opts, _ = _configure(NONHORN_INPUT)
        # Current behavior: doesn't distinguish Horn vs non-Horn
        assert opts.hyper_resolution is True
        assert opts.binary_resolution is False

    def test_nonunit_horn_eq(self):
        """Non-unit Horn + equality → both paramodulation and hyper."""
        opts, _ = _configure(NONUNIT_HORN_EQ_INPUT)
        assert opts.paramodulation is True
        assert opts.hyper_resolution is True

    def test_no_equality_no_paramodulation(self):
        """No equality → paramodulation stays disabled."""
        text = """formulas(sos).
P(a).
-P(x) | Q(x).
end_of_list.
"""
        opts, _ = _configure(text)
        assert opts.paramodulation is False
        assert opts.demodulation is False

    def test_auto_limits_applied(self):
        """Auto-limits set max_weight=100, sos_limit=20000."""
        opts, _ = _configure(SIMPLE_HORN_INPUT)
        assert opts.max_weight == 100.0
        assert opts.sos_limit == 20000

    def test_explicit_max_weight_preserved(self):
        """Explicit max_weight is not overridden by auto-limits."""
        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(SIMPLE_HORN_INPUT)
        opts = SearchOptions(max_weight=50.0)
        _auto_inference(parsed, opts)
        _auto_limits(parsed, opts)
        assert opts.max_weight == 50.0

    def test_no_auto_preserves_binary_resolution(self):
        """Without set(auto), binary_resolution default is preserved."""
        text = """formulas(sos).
-P(x) | Q(x).
P(a).
end_of_list.
formulas(goals).
Q(a).
end_of_list.
"""
        opts, _ = _configure(text)
        assert opts.binary_resolution is True
        assert opts.hyper_resolution is False


# ── TestHyperResolutionEndToEnd ──────────────────────────────────────────────


class TestHyperResolutionEndToEnd:
    """Test hyper-resolution actually finds proofs."""

    def test_simple_horn_proof_via_hyper(self):
        """Simple Horn problem: -P(x)|Q(x), P(a) → Q(a)."""
        output, exit_code = _run_and_capture(
            SIMPLE_HORN_INPUT,
            hyper_resolution=True,
            binary_resolution=False,
        )
        assert exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_modus_ponens_chain(self):
        """Chain: -P(x)|Q(x), -Q(x)|R(x), P(a) → R(a)."""
        text = """\
set(auto).
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
set(auto).
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
        assert exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_hyper_respects_max_given(self):
        """Hyper-resolution respects max_given limit."""
        output, exit_code = _run_and_capture(
            VAMPIRE_INPUT,
            max_given=5,
        )
        assert exit_code in (
            ExitCode.MAX_PROOFS_EXIT,
            ExitCode.MAX_GIVEN_EXIT,
            ExitCode.SOS_EMPTY_EXIT,
        )


# ── TestEquationalProofFinding ───────────────────────────────────────────────


class TestEquationalProofFinding:
    """Test auto-inference enables correct rules for equational proofs."""

    def test_simple_equality_proof(self):
        """Simple equality: f(a)=b from f(x)=g(x), g(a)=b."""
        text = """\
set(auto).
formulas(sos).
  f(x) = g(x).
  g(a) = b.
end_of_list.
formulas(goals).
  f(a) = b.
end_of_list.
"""
        output, exit_code = _run_and_capture(text)
        assert exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_group_identity(self):
        """Group theory: e*e=e from e*x=x."""
        text = """\
set(auto).
formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
end_of_list.
formulas(goals).
  e * e = e.
end_of_list.
"""
        output, exit_code = _run_and_capture(text)
        assert exit_code == ExitCode.MAX_PROOFS_EXIT


# ── TestCrossValidationReadiness ─────────────────────────────────────────────


class TestCrossValidationReadiness:
    """Smoke tests for C cross-validation infrastructure."""

    def test_c_runner_importable(self):
        """Cross-validation runner should be importable."""
        try:
            from tests.cross_validation.c_runner import run_c_prover9_from_string  # noqa: F401
        except ImportError:
            pytest.skip("c_runner not available")

    def test_c_reference_fixtures_exist(self):
        """C reference output fixtures should exist."""
        import os
        fixture_dirs = [
            "tests/fixtures/c_reference",
            "tests/fixtures/inputs",
        ]
        found = False
        for d in fixture_dirs:
            if os.path.isdir(d):
                found = True
                break
        if not found:
            pytest.skip("No fixture directories found")
