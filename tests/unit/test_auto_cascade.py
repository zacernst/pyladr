"""Tests for auto-inference behavior matching C Prover9.

Validates that Python _auto_inference decisions produce correct inference
rule configuration based on problem characteristics.

Current _auto_inference algorithm:
1. If equality literals present → enable paramodulation + demodulation
2. If set(auto) and negative literals present → enable hyper_resolution,
   disable binary_resolution
"""

from __future__ import annotations

import pytest

from pyladr.apps.prover9 import _auto_inference, _auto_limits
from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.search.given_clause import SearchOptions


def _get_auto_config(text: str) -> SearchOptions:
    """Parse input and run auto-inference, return configured opts."""
    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(text)
    opts = SearchOptions()
    _auto_inference(parsed, opts)
    _auto_limits(parsed, opts)
    return opts


class TestAutoInferenceBehavior:
    """Validate _auto_inference produces correct inference configuration."""

    def test_vampire_in_hne(self):
        """vampire.in: Horn, non-equality with negative literals.

        With set(auto) and negative literals → hyper_resolution enabled.
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
        opts = _get_auto_config(text)
        assert opts.hyper_resolution is True
        assert opts.binary_resolution is False
        assert opts.paramodulation is False

    def test_unit_equality(self):
        """Unit equality: paramodulation + demodulation enabled.

        With set(auto) and equality → paramodulation + demodulation.
        Note: auto also enables hyper since goals create negative literals.
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
        opts = _get_auto_config(text)
        assert opts.paramodulation is True
        assert opts.demodulation is True

    def test_nonhorn_noneq_with_auto(self):
        """Non-Horn, non-equality with set(auto) and negative literals.

        Current behavior: auto + neg lits → hyper_resolution enabled.
        Note: C Prover9 would use binary_resolution for non-Horn,
        but current _auto_inference doesn't distinguish Horn vs non-Horn.
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
        opts = _get_auto_config(text)
        # Current simplified behavior: neg lits + auto → hyper
        assert opts.hyper_resolution is True
        assert opts.binary_resolution is False

    def test_nonunit_horn_equality(self):
        """Non-unit Horn + equality → paramodulation + hyper_resolution."""
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
        opts = _get_auto_config(text)
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
        opts = _get_auto_config(text)
        assert opts.max_weight == 100.0
        assert opts.sos_limit == 20000

    def test_no_auto_preserves_defaults(self):
        """Without set(auto), SearchOptions defaults are preserved."""
        text = """formulas(sos).
P(a).
end_of_list.
formulas(goals).
P(a).
end_of_list.
"""
        opts = _get_auto_config(text)
        # Without auto, defaults apply (binary_resolution=True, factoring=True)
        assert opts.binary_resolution is True
        assert opts.factoring is True
        assert opts.hyper_resolution is False
        assert opts.paramodulation is False

    def test_equality_enables_demodulation(self):
        """Equality literals enable both paramodulation and demodulation."""
        text = """formulas(sos).
f(x) = g(x).
end_of_list.
"""
        opts = _get_auto_config(text)
        assert opts.paramodulation is True
        assert opts.demodulation is True

    def test_no_equality_no_paramodulation(self):
        """Without equality, paramodulation stays disabled."""
        text = """formulas(sos).
P(a).
-P(x) | Q(x).
end_of_list.
"""
        opts = _get_auto_config(text)
        assert opts.paramodulation is False
        assert opts.demodulation is False
