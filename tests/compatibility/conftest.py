"""Shared fixtures for compatibility testing."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import C_PROVER9_BIN, PROJECT_ROOT, TEST_INPUTS_DIR

# ── Paths ──────────────────────────────────────────────────────────────────

COMPAT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = PROJECT_ROOT / "prover9.examples"
C_REFERENCE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "c_reference"
BENCH_INPUTS_DIR = PROJECT_ROOT / "tests" / "fixtures" / "inputs"


# ── ML availability detection ──────────────────────────────────────────────


def _ml_modules_available() -> bool:
    """Check if ML enhancement modules are importable."""
    try:
        importlib.import_module("torch")
        return True
    except ImportError:
        return False


ml_available = _ml_modules_available()

skip_without_ml = pytest.mark.skipif(
    not ml_available,
    reason="ML dependencies (torch) not installed",
)


# ── Standard problem fixtures ──────────────────────────────────────────────


@pytest.fixture
def trivial_resolution_clauses():
    """P(a), ~P(x)|Q(x), ~Q(a) — resolves to empty clause."""
    from pyladr.core.clause import Clause, Literal
    from pyladr.core.term import get_rigid_term, get_variable_term

    P_sn, Q_sn, a_sn = 1, 2, 3
    a = get_rigid_term(a_sn, 0)
    x = get_variable_term(0)

    c1 = Clause(literals=(Literal(sign=True, atom=get_rigid_term(P_sn, 1, (a,))),))
    c2 = Clause(
        literals=(
            Literal(sign=False, atom=get_rigid_term(P_sn, 1, (x,))),
            Literal(sign=True, atom=get_rigid_term(Q_sn, 1, (x,))),
        )
    )
    c3 = Clause(
        literals=(Literal(sign=False, atom=get_rigid_term(Q_sn, 1, (a,))),)
    )
    return [c1, c2, c3]


@pytest.fixture
def group_theory_problem():
    """Group theory: prove commutativity from x*x=e.

    Returns (symbol_table, usable, sos).
    """
    from pyladr.core.clause import Clause, Literal
    from pyladr.core.symbol import SymbolTable
    from pyladr.core.term import (
        build_binary_term,
        build_unary_term,
        get_rigid_term,
        get_variable_term,
    )

    st = SymbolTable()
    eq_sn = st.str_to_sn("=", 2)
    mult_sn = st.str_to_sn("*", 2)
    inv_sn = st.str_to_sn("'", 1)
    e_sn = st.str_to_sn("e", 0)

    e = get_rigid_term(e_sn, 0)
    x, y, z = get_variable_term(0), get_variable_term(1), get_variable_term(2)
    mult = lambda a, b: build_binary_term(mult_sn, a, b)
    inv = lambda a: build_unary_term(inv_sn, a)
    eq = lambda a, b: build_binary_term(eq_sn, a, b)

    def pos_lit(atom):
        return Literal(sign=True, atom=atom)

    def neg_lit(atom):
        return Literal(sign=False, atom=atom)

    c1 = Clause(literals=(pos_lit(eq(mult(e, x), x)),))
    c2 = Clause(literals=(pos_lit(eq(mult(inv(x), x), e)),))
    c3 = Clause(
        literals=(pos_lit(eq(mult(mult(x, y), z), mult(x, mult(y, z)))),)
    )
    c4 = Clause(literals=(pos_lit(eq(mult(x, x), e)),))

    a_sn = st.str_to_sn("a", 0)
    b_sn = st.str_to_sn("b", 0)
    a_const = get_rigid_term(a_sn, 0)
    b_const = get_rigid_term(b_sn, 0)
    goal_denial = Clause(
        literals=(
            neg_lit(eq(mult(a_const, b_const), mult(b_const, a_const))),
        )
    )

    return st, [c1, c2, c3, c4], [goal_denial]


@pytest.fixture
def equational_problem():
    """Simple equational problem: a=b, p(a), ~p(b).

    Returns (symbol_table, usable, sos).
    """
    from pyladr.core.clause import Clause, Literal
    from pyladr.core.symbol import SymbolTable
    from pyladr.core.term import build_binary_term, get_rigid_term

    st = SymbolTable()
    eq_sn = st.str_to_sn("=", 2)
    p_sn = st.str_to_sn("p", 1)
    a_sn = st.str_to_sn("a", 0)
    b_sn = st.str_to_sn("b", 0)

    a = get_rigid_term(a_sn, 0)
    b = get_rigid_term(b_sn, 0)
    eq_ab = build_binary_term(eq_sn, a, b)
    p_a = get_rigid_term(p_sn, 1, (a,))
    p_b = get_rigid_term(p_sn, 1, (b,))

    c1 = Clause(literals=(Literal(sign=True, atom=eq_ab),))
    c2 = Clause(literals=(Literal(sign=True, atom=p_a),))
    c3 = Clause(literals=(Literal(sign=False, atom=p_b),))

    return st, [c1], [c2, c3]


# ── Search runner helpers ──────────────────────────────────────────────────


def run_search(
    usable: list,
    sos: list,
    *,
    symbol_table=None,
    binary_resolution: bool = True,
    paramodulation: bool = False,
    demodulation: bool = False,
    factoring: bool = True,
    max_given: int = 200,
    quiet: bool = True,
    **extra_opts,
):
    """Run the Python search engine with given options."""
    from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

    opts = SearchOptions(
        binary_resolution=binary_resolution,
        paramodulation=paramodulation,
        demodulation=demodulation,
        factoring=factoring,
        max_given=max_given,
        quiet=quiet,
        **extra_opts,
    )
    search = GivenClauseSearch(options=opts, symbol_table=symbol_table)
    return search.run(usable=usable, sos=sos)


def run_search_from_file(input_path: Path, **opts):
    """Parse an input file and run the Python search engine.

    Returns the SearchResult from the Python engine.
    """
    from pyladr.core.symbol import SymbolTable
    from pyladr.parsing.ladr_parser import LADRParser
    from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

    text = input_path.read_text()
    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(text)

    search_opts = SearchOptions(
        binary_resolution=True,
        paramodulation=opts.get("paramodulation", False),
        demodulation=opts.get("demodulation", False),
        factoring=True,
        max_given=opts.get("max_given", 500),
        quiet=True,
    )

    search = GivenClauseSearch(options=search_opts, symbol_table=st)
    return search.run(
        usable=parsed.usable or [],
        sos=parsed.sos or [],
    )
