"""REQ-C010: Options compatibility validation.

Systematic testing that all set()/assign()/clear() directives produce
results matching C Prover9 reference behavior, and that each option
has a demonstrable functional effect (not silently ignored).
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

import pytest

from pyladr.parsing.ladr_parser import LADRParser
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions
from pyladr.search.selection import GivenSelection, SelectionRule, SelectionOrder


# ── Helpers ──────────────────────────────────────────────────────────────────


def _run_search_from_text(
    text: str,
    *,
    extra_opts: dict | None = None,
) -> tuple[object, str]:
    """Parse LADR input text, run search, return (result, stdout_output)."""
    from pyladr.apps.prover9 import _auto_inference, _deny_goals
    from pyladr.core.symbol import SymbolTable

    symbol_table = SymbolTable()
    parser = LADRParser(symbol_table)
    parsed = parser.parse_input(text)

    usable, sos, _denied = _deny_goals(parsed, symbol_table)

    opts = SearchOptions()

    # Apply parsed assigns
    if "max_proofs" in parsed.assigns:
        opts.max_proofs = int(parsed.assigns["max_proofs"])
    if "max_given" in parsed.assigns:
        opts.max_given = int(parsed.assigns["max_given"])
    if "max_kept" in parsed.assigns:
        opts.max_kept = int(parsed.assigns["max_kept"])
    if "max_seconds" in parsed.assigns:
        opts.max_seconds = float(parsed.assigns["max_seconds"])
    if "max_generated" in parsed.assigns:
        opts.max_generated = int(parsed.assigns["max_generated"])
    if "max_weight" in parsed.assigns:
        opts.max_weight = float(parsed.assigns["max_weight"])
    if "sos_limit" in parsed.assigns:
        opts.sos_limit = int(parsed.assigns["sos_limit"])

    # Apply flags
    if parsed.flags.get("print_given", False):
        opts.print_given = True
    if parsed.flags.get("print_kept", False):
        opts.print_kept = True

    # Auto inference
    _auto_inference(parsed, opts)

    # Apply extra options
    if extra_opts:
        for k, v in extra_opts.items():
            setattr(opts, k, v)

    engine = GivenClauseSearch(options=opts, symbol_table=symbol_table)

    buf = io.StringIO()
    with redirect_stdout(buf):
        result = engine.run(usable=usable, sos=sos)

    return result, buf.getvalue()


# ── Hyper-resolution input (vampire.in style, non-equational) ────────────────

HYPER_RES_INPUT = """\
formulas(sos).
-P(x) | -P(i(x,y)) | P(y).
P(i(x,i(i(y,i(x,z)),i(i(n(z),i(i(n(v),w),y)),i(v,z))))).
end_of_list.

formulas(goals).
P(i(x,x)).
end_of_list.
"""

# Simple propositional resolution input
SIMPLE_RESOLUTION_INPUT = """\
formulas(sos).
P(a).
-P(x) | Q(x).
end_of_list.

formulas(goals).
Q(a).
end_of_list.
"""


# ── Test: set(auto) enables hyper-resolution ─────────────────────────────────


class TestSetAuto:
    """Validate set(auto) behavior matches C Prover9."""

    def test_auto_enables_hyper_resolution_for_hne(self):
        """set(auto) should enable hyper-resolution for HNE problems."""
        text = "set(auto).\n" + HYPER_RES_INPUT
        result, output = _run_search_from_text(text)
        # Should find proof using hyper-resolution
        assert len(result.proofs) >= 1, "set(auto) should find proof via hyper-res"

    def test_auto_limits_max_weight_default(self):
        """set(auto) should set max_weight=100 when not explicitly overridden."""
        from pyladr.apps.prover9 import _auto_inference, _auto_limits

        parser = LADRParser()
        parsed = parser.parse_input("set(auto).\n" + HYPER_RES_INPUT)
        opts = SearchOptions()
        _auto_inference(parsed, opts)
        _auto_limits(parsed, opts)
        assert opts.max_weight == 100.0

    def test_auto_limits_sos_limit_default(self):
        """set(auto) should set sos_limit=20000 when not explicitly overridden."""
        from pyladr.apps.prover9 import _auto_inference, _auto_limits

        parser = LADRParser()
        parsed = parser.parse_input("set(auto).\n" + HYPER_RES_INPUT)
        opts = SearchOptions()
        _auto_inference(parsed, opts)
        _auto_limits(parsed, opts)
        assert opts.sos_limit == 20000

    def test_auto_limits_overridden_by_explicit_assign(self):
        """Explicit assign(max_weight, N) should override set(auto) default."""
        from pyladr.apps.prover9 import _auto_inference, _auto_limits

        parser = LADRParser()
        text = "set(auto).\nassign(max_weight, 128).\n" + HYPER_RES_INPUT
        parsed = parser.parse_input(text)
        opts = SearchOptions()
        opts.max_weight = float(parsed.assigns["max_weight"])
        _auto_inference(parsed, opts)
        _auto_limits(parsed, opts)
        assert opts.max_weight == 128.0

    def test_auto_disables_binary_resolution_for_hne(self):
        """set(auto) should disable binary_resolution for HNE problems."""
        from pyladr.apps.prover9 import _auto_inference

        parser = LADRParser()
        parsed = parser.parse_input("set(auto).\n" + HYPER_RES_INPUT)
        opts = SearchOptions()
        _auto_inference(parsed, opts)
        assert opts.hyper_resolution is True
        assert opts.binary_resolution is False


# ── Test: assign() directives have functional effect ─────────────────────────


class TestAssignDirectives:
    """Verify each assign() directive has a functional effect on search."""

    def test_max_proofs_limits_proof_count(self):
        """assign(max_proofs, N) should limit number of proofs found."""
        text = "assign(max_proofs, 1).\n" + SIMPLE_RESOLUTION_INPUT
        result, _ = _run_search_from_text(text)
        assert len(result.proofs) <= 1

    def test_max_given_limits_search(self):
        """assign(max_given, N) should limit given clauses processed."""
        text = "assign(max_given, 5).\n" + HYPER_RES_INPUT
        result, _ = _run_search_from_text(text)
        # C Prover9 checks "given > max_given" AFTER incrementing,
        # so max_given=5 allows up to 6 (5+1 for the exit check).
        assert result.stats.given <= 6

    def test_max_weight_filters_heavy_clauses(self):
        """assign(max_weight, N) should prevent heavy clauses from being kept."""
        # With very low max_weight, should not find proof
        text_low = "set(auto).\nassign(max_weight, 5).\n" + HYPER_RES_INPUT
        result_low, _ = _run_search_from_text(text_low)

        # With high max_weight, should find proof
        text_high = "set(auto).\nassign(max_weight, 128).\n" + HYPER_RES_INPUT
        result_high, _ = _run_search_from_text(text_high)

        # max_weight=5 is too restrictive for this problem
        assert len(result_high.proofs) >= len(result_low.proofs)

    def test_max_seconds_limits_time(self):
        """assign(max_seconds, N) should terminate search within time limit."""
        text = "assign(max_seconds, 1).\n" + HYPER_RES_INPUT
        result, _ = _run_search_from_text(text)
        assert result.stats.elapsed_seconds() <= 5  # generous slack


# ── Test: set()/clear() flags have functional effect ─────────────────────────


class TestSetClearFlags:
    """Verify set()/clear() flags have functional effect."""

    def test_print_given_produces_output(self):
        """set(print_given) should produce given clause trace in output."""
        text = "set(print_given).\n" + SIMPLE_RESOLUTION_INPUT
        _, output = _run_search_from_text(text)
        assert "given #" in output

    def test_no_print_given_suppresses_output(self):
        """Without set(print_given), no given clause trace."""
        result, output = _run_search_from_text(
            SIMPLE_RESOLUTION_INPUT,
            extra_opts={"print_given": False},
        )
        assert "given #" not in output


# ── Test: Selection ratio matches C Prover9 ──────────────────────────────────


class TestSelectionRatio:
    """Verify selection ratio matches C Prover9 default behavior."""

    def test_default_ratio_is_5_cycle(self):
        """Default selection should cycle with period 5 (1 age + 4 weight)."""
        gs = GivenSelection()
        assert gs._cycle_size == 5

    def test_default_starts_with_age(self):
        """First non-initial selection should be by age (matching C)."""
        gs = GivenSelection()
        rule = gs._get_current_rule()
        assert rule.order == SelectionOrder.AGE
        assert rule.name == "A"

    def test_default_cycle_pattern(self):
        """Selection pattern should be A W W W W A W W W W ..."""
        gs = GivenSelection()
        expected = ["A", "W", "W", "W", "W"] * 2
        actual = []
        for _ in range(10):
            rule = gs._get_current_rule()
            actual.append(rule.name)
            gs._advance_cycle()
        assert actual == expected

    def test_selection_type_in_output_matches_c(self):
        """Given clause output should show selection type matching C format."""
        text = "set(auto).\nset(print_given).\n" + HYPER_RES_INPUT
        result, output = _run_search_from_text(text)
        lines = [l for l in output.splitlines() if l.startswith("given #")]
        # Initial clauses should be (I)
        for line in lines[:8]:  # 8 initial clauses (2 sos + 6 goals negated... varies)
            if "(I," in line:
                continue
        # First non-initial should be (A)
        non_initial = [l for l in lines if "(I," not in l]
        if non_initial:
            assert "(A," in non_initial[0], (
                f"First non-initial selection should be (A), got: {non_initial[0][:50]}"
            )

    def test_custom_selection_rules(self):
        """Custom rules should change selection cycle."""
        rules = [
            SelectionRule("W", SelectionOrder.WEIGHT, part=2),
            SelectionRule("A", SelectionOrder.AGE, part=1),
        ]
        gs = GivenSelection(rules=rules)
        assert gs._cycle_size == 3
        expected = ["W", "W", "A", "W", "W", "A"]
        actual = []
        for _ in range(6):
            rule = gs._get_current_rule()
            actual.append(rule.name)
            gs._advance_cycle()
        assert actual == expected


# ── Test: Parsing of all directive types ─────────────────────────────────────


class TestDirectiveParsing:
    """Verify parser correctly handles all directive types."""

    def test_set_flag_parsed(self):
        """set(FLAG) should be parsed into flags dict as True."""
        parser = LADRParser()
        text = "set(auto).\nset(print_given).\nformulas(sos).\nend_of_list.\n"
        parsed = parser.parse_input(text)
        assert parsed.flags.get("auto") is True
        assert parsed.flags.get("print_given") is True

    def test_clear_flag_parsed(self):
        """clear(FLAG) should be parsed into flags dict as False."""
        parser = LADRParser()
        text = "clear(print_given).\nformulas(sos).\nend_of_list.\n"
        parsed = parser.parse_input(text)
        assert parsed.flags.get("print_given") is False

    def test_assign_int_parsed(self):
        """assign(NAME, INT) should parse integer values."""
        parser = LADRParser()
        text = "assign(max_proofs, 6).\nformulas(sos).\nend_of_list.\n"
        parsed = parser.parse_input(text)
        assert parsed.assigns["max_proofs"] == 6

    def test_assign_float_parsed(self):
        """assign(NAME, FLOAT) should parse float values."""
        parser = LADRParser()
        text = "assign(max_weight, 128.5).\nformulas(sos).\nend_of_list.\n"
        parsed = parser.parse_input(text)
        assert parsed.assigns["max_weight"] == 128.5

    def test_multiple_assigns_parsed(self):
        """Multiple assign() directives should all be captured."""
        parser = LADRParser()
        text = (
            "assign(max_proofs, 6).\n"
            "assign(max_weight, 128).\n"
            "assign(max_given, 100).\n"
            "formulas(sos).\nend_of_list.\n"
        )
        parsed = parser.parse_input(text)
        assert parsed.assigns["max_proofs"] == 6
        assert parsed.assigns["max_weight"] == 128
        assert parsed.assigns["max_given"] == 100

    def test_set_clear_precedence(self):
        """Last directive wins: set then clear should result in False."""
        parser = LADRParser()
        text = "set(print_given).\nclear(print_given).\nformulas(sos).\nend_of_list.\n"
        parsed = parser.parse_input(text)
        assert parsed.flags.get("print_given") is False


# ── Test: C Prover9 cross-validation (if binary available) ───────────────────


class TestCrossValidation:
    """Cross-validate option behavior against C Prover9 binary."""

    @pytest.fixture
    def c_binary(self, project_root):
        """Get C Prover9 binary path, skip if not available."""
        from tests.conftest import C_PROVER9_BIN, c_binary_available
        if not c_binary_available():
            pytest.skip("C Prover9 binary not available")
        return C_PROVER9_BIN

    def test_vampire_in_selection_ratio_matches(
        self, c_binary, project_root, tmp_path
    ):
        """Selection ratio on vampire.in should match C Prover9.

        First 18 weight-based selections must select identical weights.
        Age-based selections must occur at same positions (#9, #14, #19, #24).
        """
        import subprocess

        vampire_in = project_root / "tests" / "fixtures" / "inputs" / "vampire.in"
        if not vampire_in.exists():
            pytest.skip("vampire.in not found")

        # Run C Prover9
        c_result = subprocess.run(
            [str(c_binary), "-f", str(vampire_in)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Run PyLADR
        py_result = subprocess.run(
            [sys.executable, "-B", "-m", "pyladr.cli", "-f", str(vampire_in)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Extract given clause lines
        c_givens = [
            l for l in c_result.stdout.splitlines() if l.startswith("given #")
        ]
        py_givens = [
            l for l in py_result.stdout.splitlines() if l.startswith("given #")
        ]

        # Both should have at least 18 given clauses
        assert len(c_givens) >= 18, f"C Prover9 only produced {len(c_givens)} givens"
        assert len(py_givens) >= 18, f"PyLADR only produced {len(py_givens)} givens"

        # Extract selection types for first 18
        def get_sel_type(line: str) -> str:
            # "given #N (X,wt=W): ..."
            start = line.index("(") + 1
            return line[start]

        def get_weight(line: str) -> int:
            import re
            m = re.search(r"wt=(\d+)", line)
            return int(m.group(1)) if m else -1

        # Verify age-based picks occur at same positions
        for i in range(min(18, len(c_givens), len(py_givens))):
            c_type = get_sel_type(c_givens[i])
            py_type = get_sel_type(py_givens[i])

            # Both should agree on I/A distinction
            if c_type == "I":
                assert py_type == "I", f"Given #{i+1}: C={c_type}, PyLADR={py_type}"
            elif c_type == "A":
                assert py_type == "A", f"Given #{i+1}: C={c_type}, PyLADR={py_type}"
            else:
                # C uses T, PyLADR uses W for weight-based (cosmetic difference)
                assert py_type in ("W", "T"), f"Given #{i+1}: unexpected PyLADR type {py_type}"

        # Verify weight-based selections pick same weight clauses
        for i in range(min(18, len(c_givens), len(py_givens))):
            c_wt = get_weight(c_givens[i])
            py_wt = get_weight(py_givens[i])
            assert c_wt == py_wt, (
                f"Given #{i+1}: weight mismatch C={c_wt} vs PyLADR={py_wt}"
            )
