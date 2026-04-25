"""Behavioral equivalence tests: C++ _pyladr_core vs pure-Python implementation.

All tests are automatically skipped when the C++ extension is not built.
"""

from __future__ import annotations
from pathlib import Path
import pytest

cpp = pytest.importorskip("pyladr._pyladr_core",
                           reason="_pyladr_core not built; run build_cpp.sh")

from pyladr.core.term import Term as PyTerm, get_rigid_term, get_variable_term
from pyladr.core.substitution import (
    Context as PyContext, Trail as PyTrail,
    unify as py_unify, match as py_match,
    apply_substitution as py_apply_substitution,
    dereference as py_dereference,
)
from pyladr.cpp_utils import py_term_to_cpp


# ── Term fixtures ────────────────────────────────────────────────────────────
SYM_P, SYM_I, SYM_N, SYM_A = 1, 2, 3, 4   # positive symnums

def py_var(n): return get_variable_term(n)
def py_n(a): return get_rigid_term(SYM_N, 1, (a,))
def py_i(a, b): return get_rigid_term(SYM_I, 2, (a, b))
def py_const(s): return get_rigid_term(s, 0, ())

def cpp_var(n): return cpp.Term.make_variable(n)
def cpp_n(a): return cpp.Term.make_rigid(SYM_N, 1, [a])
def cpp_i(a, b): return cpp.Term.make_rigid(SYM_I, 2, [a, b])
def cpp_const(s): return cpp.Term.make_rigid(s, 0, [])


# ── 1. Term construction and properties ──────────────────────────────────────

class TestCppTermProperties:
    def test_variable_is_variable(self):
        t = cpp_var(0)
        assert t.is_variable
        assert not t.is_constant
        assert not t.is_complex
        assert t.varnum() == 0
        assert t.arity == 0
        assert t.symbol_count == 1

    def test_constant_is_constant(self):
        t = cpp_const(SYM_A)
        assert not t.is_variable
        assert t.is_constant
        assert not t.is_complex
        assert t.symnum() == SYM_A
        assert t.arity == 0
        assert t.symbol_count == 1

    def test_complex_is_complex(self):
        t = cpp_n(cpp_var(0))
        assert not t.is_variable
        assert not t.is_constant
        assert t.is_complex
        assert t.arity == 1
        assert t.symbol_count == 2

    def test_symbol_count_nested(self):
        # i(n(x), y) has 4 nodes
        t = cpp_i(cpp_n(cpp_var(0)), cpp_var(1))
        assert t.symbol_count == 4

    def test_variable_interning(self):
        # Same varnum → same object via term_ident
        a = cpp_var(5)
        b = cpp_var(5)
        assert a.term_ident(b)

    def test_term_ident_variables(self):
        assert cpp_var(0).term_ident(cpp_var(0))
        assert not cpp_var(0).term_ident(cpp_var(1))

    def test_term_ident_rigid(self):
        t1 = cpp_n(cpp_var(0))
        t2 = cpp_n(cpp_var(0))
        assert t1.term_ident(t2)
        t3 = cpp_n(cpp_var(1))
        assert not t1.term_ident(t3)

    def test_args_length(self):
        t = cpp_i(cpp_var(0), cpp_var(1))
        assert len(t.args) == 2

    def test_variables_set(self):
        t = cpp_i(cpp_n(cpp_var(0)), cpp_var(1))
        assert cpp_var(0).variables() == {0}
        assert t.variables() == {0, 1}
        assert cpp_const(SYM_A).variables() == set()


# ── 2. Unification equivalence ───────────────────────────────────────────────

def cpp_unify_fresh(ct1, ct2):
    """Attempt unification with fresh contexts; return bool."""
    c1, c2, tr = cpp.Context(), cpp.Context(), cpp.Trail()
    return cpp.unify(ct1, c1, ct2, c2, tr)

def py_unify_fresh(pt1, pt2):
    c1, c2, tr = PyContext(), PyContext(), PyTrail()
    return py_unify(pt1, c1, pt2, c2, tr)

UNIFY_CASES = [
    ("x_y",          lambda: (py_var(0),          py_var(1)),           True),
    ("x_x",          lambda: (py_var(0),          py_var(0)),           True),
    ("nx_ny",        lambda: (py_n(py_var(0)),    py_n(py_var(1))),     True),
    ("ixy_iyx",      lambda: (py_i(py_var(0), py_var(1)),
                               py_i(py_var(1), py_var(0))),              True),
    ("x_nx_diff_ctx", lambda: (py_var(0),          py_n(py_var(0))),     True),
    ("const_var",    lambda: (py_const(SYM_A),   py_var(0)),            True),
    ("var_const",    lambda: (py_var(0),          py_const(SYM_A)),     True),
    ("const_const",  lambda: (py_const(SYM_A),   py_const(SYM_A)),     True),
    ("diff_consts",  lambda: (py_const(SYM_A),   py_const(SYM_N)),     False),
    ("ixy_ia",       lambda: (py_i(py_var(0), py_var(1)),
                               py_i(py_const(SYM_A), py_const(SYM_A))), True),
    ("deep_nested",  lambda: (py_i(py_n(py_var(0)), py_var(1)),
                               py_i(py_var(0), py_const(SYM_A))),        True),
]

@pytest.mark.parametrize("name,terms_fn,expected", UNIFY_CASES, ids=[c[0] for c in UNIFY_CASES])
def test_cpp_unify_matches_expected(name, terms_fn, expected):
    pt1, pt2 = terms_fn()
    ct1, ct2 = py_term_to_cpp(pt1), py_term_to_cpp(pt2)
    result = cpp_unify_fresh(ct1, ct2)
    assert result == expected, f"cpp unify({name}): expected {expected}, got {result}"

@pytest.mark.parametrize("name,terms_fn,expected", UNIFY_CASES, ids=[c[0] for c in UNIFY_CASES])
def test_cpp_unify_matches_python(name, terms_fn, expected):
    pt1, pt2 = terms_fn()
    ct1, ct2 = py_term_to_cpp(pt1), py_term_to_cpp(pt2)
    py_result  = py_unify_fresh(pt1, pt2)
    cpp_result = cpp_unify_fresh(ct1, ct2)
    assert cpp_result == py_result, (
        f"cpp and python disagree on unify({name}): cpp={cpp_result} py={py_result}"
    )


# ── 3. Match equivalence ─────────────────────────────────────────────────────

def cpp_match_fresh(pattern, target):
    ctx, tr = cpp.Context(), cpp.Trail()
    return cpp.match_term(pattern, ctx, target, tr)

def py_match_fresh(pattern, target):
    ctx, tr = PyContext(), PyTrail()
    return py_match(pattern, ctx, target, tr)

MATCH_CASES = [
    ("x_matches_a",  lambda: (py_var(0),       py_const(SYM_A)),  True),
    ("x_matches_nx", lambda: (py_var(0),       py_n(py_var(1))),  True),
    ("nx_matches_na",lambda: (py_n(py_var(0)), py_n(py_const(SYM_A))), True),
    ("nx_no_match_a",lambda: (py_n(py_var(0)), py_const(SYM_A)), False),
    ("a_matches_a",  lambda: (py_const(SYM_A), py_const(SYM_A)), True),
    ("a_no_match_b", lambda: (py_const(SYM_A), py_const(SYM_N)), False),
]

@pytest.mark.parametrize("name,terms_fn,expected", MATCH_CASES, ids=[c[0] for c in MATCH_CASES])
def test_cpp_match_matches_python(name, terms_fn, expected):
    pt1, pt2 = terms_fn()
    ct1, ct2 = py_term_to_cpp(pt1), py_term_to_cpp(pt2)
    py_result  = py_match_fresh(pt1, pt2)
    cpp_result = cpp_match_fresh(ct1, ct2)
    assert cpp_result == py_result, (
        f"cpp and python disagree on match({name}): cpp={cpp_result} py={py_result}"
    )


# ── 4. apply_substitution equivalence ────────────────────────────────────────

def test_apply_substitution_binds_variable():
    """Binding x→n(a) and applying should give n(a)."""
    x_cpp = cpp_var(0)
    na_cpp = cpp_n(cpp_const(SYM_A))
    c1, c2, tr = cpp.Context(), cpp.Context(), cpp.Trail()
    assert cpp.unify(x_cpp, c1, na_cpp, c2, tr)
    result = cpp.apply_substitution(x_cpp, c1)
    assert result.term_ident(na_cpp)

def test_apply_substitution_unbound_renamed():
    """Unbound variable in a context gets multiplier-renamed."""
    x_cpp = cpp_var(0)
    c = cpp.Context()
    result = cpp.apply_substitution(x_cpp, c)
    assert result.is_variable
    assert result.varnum() == c.multiplier * cpp.MAX_VARS + 0

def test_apply_substitution_no_context():
    """apply_substitution with no context returns same variable."""
    x_cpp = cpp_var(7)
    result = cpp.apply_substitution(x_cpp, None)
    assert result.is_variable
    assert result.varnum() == 7


# ── 5. Trail undo ─────────────────────────────────────────────────────────────

def test_trail_undo_restores_context():
    x_cpp = cpp_var(0)
    na_cpp = cpp_n(cpp_const(SYM_A))
    c1, c2 = cpp.Context(), cpp.Context()
    tr = cpp.Trail()
    assert not c1.is_bound(0)
    assert cpp.unify(x_cpp, c1, na_cpp, c2, tr)
    assert c1.is_bound(0)
    tr.undo()
    assert not c1.is_bound(0)

def test_trail_undo_to():
    x_cpp = cpp_var(0)
    y_cpp = cpp_var(1)
    a_cpp = cpp_const(SYM_A)
    b_cpp = cpp_const(SYM_N)
    c = cpp.Context()
    tr = cpp.Trail()
    # Bind x→a
    cpp.unify(x_cpp, c, a_cpp, cpp.Context(), tr)
    pos = tr.position()
    # Bind y→b
    cpp.unify(y_cpp, c, b_cpp, cpp.Context(), tr)
    assert c.is_bound(0) and c.is_bound(1)
    tr.undo_to(pos)
    assert c.is_bound(0) and not c.is_bound(1)


# ── 6. py_term_to_cpp round-trip ─────────────────────────────────────────────

def test_py_to_cpp_variable():
    py_t = py_var(3)
    cpp_t = py_term_to_cpp(py_t)
    assert cpp_t.is_variable
    assert cpp_t.varnum() == 3

def test_py_to_cpp_complex():
    py_t = py_i(py_n(py_var(0)), py_const(SYM_A))
    cpp_t = py_term_to_cpp(py_t)
    assert cpp_t.is_complex
    assert cpp_t.symnum() == SYM_I
    assert cpp_t.arity == 2
    assert cpp_t.args[0].is_complex
    assert cpp_t.args[1].is_constant


# ── 7. --cpp CLI flag ─────────────────────────────────────────────────────────

def test_cli_cpp_flag_enables_backend():
    from pyladr.apps.prover9 import run_prover
    from pyladr import cpp_backend
    cpp_backend.disable()
    fixture = Path(__file__).parent.parent / "fixtures" / "inputs" / "simple_group.in"
    if not fixture.exists():
        pytest.skip("simple_group.in fixture not found")
    # --cpp should not crash
    exit_code = run_prover(["pyprover9", "-f", str(fixture),
                             "--cpp", "-max_seconds", "3"])
    assert exit_code != 1
    assert cpp_backend.is_enabled()
    cpp_backend.disable()  # cleanup

def test_cli_no_cpp_flag_leaves_backend_disabled():
    from pyladr.apps.prover9 import run_prover
    from pyladr import cpp_backend
    cpp_backend.disable()
    fixture = Path(__file__).parent.parent / "fixtures" / "inputs" / "simple_group.in"
    if not fixture.exists():
        pytest.skip("simple_group.in fixture not found")
    run_prover(["pyprover9", "-f", str(fixture), "-max_seconds", "2"])
    assert not cpp_backend.is_enabled()
