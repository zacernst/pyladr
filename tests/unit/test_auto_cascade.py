"""Tests for auto-cascade behavior matching C Prover9.

Validates that Python auto_inference decisions match the C Prover9
implementation in reference-prover9/provers.src/search.c lines 2335-2411.

C Prover9 auto_inference algorithm:
1. If equality → set(paramodulation)
2. If (!equality || !unit):
   a. Horn + equality → hyper_resolution (+ neg_ur_resolution)
   b. Horn + !equality (HNE) → hyper_resolution (if depth_diff > 0)
   c. !Horn → binary_resolution
"""

from __future__ import annotations

from pyladr.core.symbol import SymbolTable
from pyladr.apps.prover9 import _analyze_problem, _apply_settings
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.search.given_clause import SearchOptions


def _get_auto_config(text: str) -> tuple[dict, SearchOptions]:
    """Parse input and run auto-cascade, return (analysis, opts)."""
    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(text)
    all_clauses = parsed.sos + parsed.goals + parsed.usable
    is_horn, has_eq, all_units = _analyze_problem(all_clauses, st)
    opts = SearchOptions()
    # Only apply auto settings if set(auto) is in the input (C default is TRUE,
    # but the test helper is explicit: no set(auto) → no cascade)
    has_auto = "set(auto)" in text
    _apply_settings(parsed, opts, st, auto=has_auto)
    analysis = {"is_horn": is_horn, "has_equality": has_eq, "all_units": all_units}
    return analysis, opts


class TestAutoCascadeVsCProver9:
    """Validate auto-cascade decisions match C Prover9 ground truth."""

    def test_vampire_in_hne(self):
        """vampire.in: Horn, non-equality → hyper_resolution only.

        C Prover9 output:
          Auto_inference settings:
            % set(hyper_resolution).  % (HNE depth_diff=1)
        """
        text = """set(auto).
formulas(sos).
-P(x) | -P(i(x,y)) | P(y).
P(i(i(x,y),i(i(y,z),i(x,z)))).
P(i(i(n(x),x),x)).
P(i(x,i(n(x),y))).
-P(i(a,i(i(b,i(a,c)),i(i(n(c),i(i(n(d),e),b)),i(d,c))))).
end_of_list.
"""
        analysis, opts = _get_auto_config(text)
        assert analysis["is_horn"] is True
        assert analysis["has_equality"] is False
        assert opts.hyper_resolution is True
        assert opts.binary_resolution is False
        assert opts.paramodulation is False
        assert opts.max_weight == 100.0
        assert opts.sos_limit == 20000

    def test_unit_equality(self):
        """Unit equality: only paramodulation, nothing else.

        C Prover9 output:
          Auto_inference settings:
            % set(paramodulation).  % (positive equality literals)
        """
        text = """set(auto).
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
        analysis, opts = _get_auto_config(text)
        assert analysis["is_horn"] is True
        assert analysis["has_equality"] is True
        assert analysis["all_units"] is True
        assert opts.paramodulation is True
        assert opts.binary_resolution is False
        assert opts.hyper_resolution is False

    def test_nonhorn_noneq(self):
        """Non-Horn, non-equality → binary_resolution.

        C Prover9 output:
          Auto_inference settings:
            % set(binary_resolution).  % (non-Horn)
            % set(neg_ur_resolution).  % (non-Horn, less than 100 clauses)
        """
        text = """set(auto).
formulas(sos).
P(a) | Q(a).
-P(x) | R(x).
-Q(x) | R(x).
end_of_list.
formulas(goals).
R(a).
end_of_list.
"""
        analysis, opts = _get_auto_config(text)
        assert analysis["is_horn"] is False
        assert analysis["has_equality"] is False
        assert opts.binary_resolution is True
        assert opts.hyper_resolution is False
        assert opts.paramodulation is False

    def test_nonunit_horn_equality(self):
        """Non-unit Horn + equality → paramodulation + hyper_resolution.

        C Prover9 with predicate elimination may reduce this to unit equality,
        but without predicate elimination the correct choice is hyper + para.
        """
        text = """set(auto).
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
        analysis, opts = _get_auto_config(text)
        assert analysis["is_horn"] is True
        assert analysis["has_equality"] is True
        assert analysis["all_units"] is False
        assert opts.paramodulation is True
        assert opts.hyper_resolution is True

    def test_auto_limits(self):
        """Auto-limits: max_weight=100, sos_limit=20000."""
        text = """set(auto).
formulas(sos).
P(a).
end_of_list.
formulas(goals).
P(a).
end_of_list.
"""
        _, opts = _get_auto_config(text)
        assert opts.max_weight == 100.0
        assert opts.sos_limit == 20000

    def test_explicit_limits_not_overridden(self):
        """Explicit assign() limits should not be overridden by auto."""
        text = """set(auto).
assign(max_weight, 50).
assign(sos_limit, 5000).
formulas(sos).
P(a).
end_of_list.
formulas(goals).
P(a).
end_of_list.
"""
        _, opts = _get_auto_config(text)
        assert opts.max_weight == 50
        assert opts.sos_limit == 5000

    def test_no_auto_preserves_defaults(self):
        """Without set(auto), SearchOptions defaults are preserved."""
        text = """formulas(sos).
P(a).
end_of_list.
formulas(goals).
P(a).
end_of_list.
"""
        _, opts = _get_auto_config(text)
        # Without auto, Python defaults apply (binary_resolution=True, factoring=True)
        assert opts.binary_resolution is True
        assert opts.factoring is True
        assert opts.hyper_resolution is False
        assert opts.paramodulation is False
        assert opts.max_weight == -1.0  # no limit
        assert opts.sos_limit == -1  # no limit

    def test_print_given_from_settings(self):
        """set(print_given) correctly applied."""
        text = """set(print_given).
set(auto).
formulas(sos).
P(a).
end_of_list.
"""
        _, opts = _get_auto_config(text)
        assert opts.print_given is True
