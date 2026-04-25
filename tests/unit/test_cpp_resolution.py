"""Phase 2 behavioral equivalence tests: C++ resolution kernel vs pure Python.

Skipped automatically when _pyladr_core is not built.
"""
from __future__ import annotations
from pathlib import Path
import pytest

cpp = pytest.importorskip("pyladr._pyladr_core",
                           reason="_pyladr_core not built; run build_cpp.sh")

from pyladr.core.term import Term as PyTerm, get_rigid_term, get_variable_term
from pyladr.core.clause import Clause, Literal
from pyladr.cpp_utils import py_term_to_cpp, cpp_term_to_py
from pyladr import cpp_backend

# ── Helpers ──────────────────────────────────────────────────────────────────
SYM_P, SYM_I, SYM_N, SYM_A, SYM_B = 1, 2, 3, 4, 5

def var(n): return get_variable_term(n)
def n_term(a): return get_rigid_term(SYM_N, 1, (a,))
def i_term(a, b): return get_rigid_term(SYM_I, 2, (a, b))
def const(s): return get_rigid_term(s, 0, ())

def make_clause(*lits): return Clause(literals=tuple(lits))
def lit(sign, atom): return Literal(sign=sign, atom=atom)
def pos(atom): return lit(True, atom)
def neg(atom): return lit(False, atom)

def to_cpp_lits(clause):
    return [(l.sign, py_term_to_cpp(l.atom)) for l in clause.literals]

def cpp_lits_to_py(cpp_lits):
    return tuple(Literal(sign=s, atom=cpp_term_to_py(a)) for s, a in cpp_lits)


# ── 1. binary_resolve_lits ───────────────────────────────────────────────────

class TestCppBinaryResolve:
    def test_unit_resolution_produces_empty_clause(self):
        # +P(x) resolved with -P(a) → empty clause
        c1 = make_clause(pos(n_term(var(0))))
        c2 = make_clause(neg(n_term(const(SYM_A))))
        result = cpp.binary_resolve_lits(to_cpp_lits(c1), 0, to_cpp_lits(c2), 0)
        assert result is not None
        assert len(result) == 0

    def test_same_sign_returns_none(self):
        # Both positive — not complementary
        c1 = make_clause(pos(n_term(var(0))))
        c2 = make_clause(pos(n_term(const(SYM_A))))
        result = cpp.binary_resolve_lits(to_cpp_lits(c1), 0, to_cpp_lits(c2), 0)
        assert result is None

    def test_multi_literal_resolvent(self):
        # +i(x,y), -n(x)  resolved with  +n(a)
        # Resolving idx=1 (-n(x)) against idx=0 (+n(a)) → +i(a,y)
        c1 = make_clause(pos(i_term(var(0), var(1))), neg(n_term(var(0))))
        c2 = make_clause(pos(n_term(const(SYM_A))))
        result = cpp.binary_resolve_lits(to_cpp_lits(c1), 1, to_cpp_lits(c2), 0)
        assert result is not None
        assert len(result) == 1
        sign, atom = result[0]
        assert sign is True
        py_atom = cpp_term_to_py(atom)
        assert py_atom.symnum == SYM_I
        assert py_atom.args[0].symnum == SYM_A   # x was bound to a

    def test_occur_check_cross_context(self):
        # +P(x) vs -P(n(x)): different contexts so x in c1 != x in c2
        # This SHOULD succeed (cross-context, no occur check fires)
        c1 = make_clause(pos(var(0)))
        c2 = make_clause(neg(n_term(var(0))))
        result = cpp.binary_resolve_lits(to_cpp_lits(c1), 0, to_cpp_lits(c2), 0)
        assert result is not None  # cross-context: succeeds

    def test_unification_failure_returns_none(self):
        # n(a) vs n(b): different constants, unification fails
        c1 = make_clause(pos(n_term(const(SYM_A))))
        c2 = make_clause(neg(n_term(const(SYM_B))))
        result = cpp.binary_resolve_lits(to_cpp_lits(c1), 0, to_cpp_lits(c2), 0)
        assert result is None


# ── 2. all_binary_resolvents_lits ────────────────────────────────────────────

class TestCppAllBinaryResolvents:
    def test_no_complementary_pairs(self):
        # Both positive clauses → no resolvents
        c1 = make_clause(pos(n_term(var(0))))
        c2 = make_clause(pos(n_term(var(1))))
        result = cpp.all_binary_resolvents_lits(to_cpp_lits(c1), to_cpp_lits(c2))
        assert result == []

    def test_one_pair_produces_one_resolvent(self):
        c1 = make_clause(pos(n_term(var(0))))
        c2 = make_clause(neg(n_term(var(1))))
        result = cpp.all_binary_resolvents_lits(to_cpp_lits(c1), to_cpp_lits(c2))
        assert len(result) == 1
        assert len(result[0]) == 0

    def test_matches_python_on_multi_literal_clauses(self):
        """C++ and Python produce same number of resolvents."""
        from pyladr.inference.resolution import all_binary_resolvents
        c1 = make_clause(pos(i_term(var(0), var(1))), neg(n_term(var(0))), pos(n_term(var(1))))
        c2 = make_clause(pos(n_term(const(SYM_A))), neg(i_term(var(2), var(3))))
        cpp_lits1, cpp_lits2 = to_cpp_lits(c1), to_cpp_lits(c2)
        cpp_results = cpp.all_binary_resolvents_lits(cpp_lits1, cpp_lits2)
        py_results = all_binary_resolvents(c1, c2)
        assert len(cpp_results) == len(py_results), (
            f"C++ produced {len(cpp_results)} resolvents, Python {len(py_results)}"
        )

    def test_resolvent_literal_count_matches_python(self):
        """For each resolvent pair, literal counts match between C++ and Python."""
        from pyladr.inference.resolution import all_binary_resolvents
        c1 = make_clause(pos(n_term(var(0))), pos(i_term(var(0), var(1))))
        c2 = make_clause(neg(n_term(var(2))), neg(i_term(var(3), var(4))))
        cpp_lits1, cpp_lits2 = to_cpp_lits(c1), to_cpp_lits(c2)
        cpp_results = cpp.all_binary_resolvents_lits(cpp_lits1, cpp_lits2)
        py_results = all_binary_resolvents(c1, c2)
        assert len(cpp_results) == len(py_results)
        cpp_sizes = sorted(len(r) for r in cpp_results)
        py_sizes  = sorted(len(r.literals) for r in py_results)
        assert cpp_sizes == py_sizes


# ── 3. factor_lits ───────────────────────────────────────────────────────────

class TestCppFactor:
    def test_no_factorable_pair(self):
        # Two lits with different signs — no factoring
        c = make_clause(pos(n_term(var(0))), neg(n_term(var(1))))
        result = cpp.factor_lits(to_cpp_lits(c))
        assert result == []

    def test_same_sign_factorable(self):
        # +P(x), +P(y) → factor on x=y → +P(x)
        c = make_clause(pos(n_term(var(0))), pos(n_term(var(1))))
        result = cpp.factor_lits(to_cpp_lits(c))
        assert len(result) == 1
        assert len(result[0]) == 1
        sign, atom = result[0][0]
        assert sign is True

    def test_matches_python_factor_count(self):
        from pyladr.inference.resolution import factor
        c = make_clause(pos(n_term(var(0))), pos(n_term(var(1))), neg(i_term(var(0), var(2))))
        cpp_results = cpp.factor_lits(to_cpp_lits(c))
        py_results  = factor(c)
        assert len(cpp_results) == len(py_results), (
            f"C++ factor: {len(cpp_results)}, Python factor: {len(py_results)}"
        )


# ── 4. End-to-end: --cpp flag produces valid proof ───────────────────────────

class TestCppEndToEnd:
    def test_cpp_proof_exit_code_matches_python(self):
        """Both --cpp and pure Python find a proof on simple_proof.in."""
        from pyladr.apps.prover9 import run_prover
        fixture = Path(__file__).parent.parent / "fixtures" / "inputs" / "simple_proof.in"
        if not fixture.exists():
            pytest.skip("simple_proof.in not found")

        cpp_backend.disable()
        py_exit = run_prover(["pyprover9", "-f", str(fixture), "-max_seconds", "10"])

        cpp_backend.disable()
        cpp_exit = run_prover(["pyprover9", "-f", str(fixture), "--cpp", "-max_seconds", "10"])
        cpp_backend.disable()

        assert cpp_exit == py_exit, (
            f"--cpp exit code {cpp_exit} != pure Python exit code {py_exit}"
        )
        assert py_exit == 0, "Expected proof found (exit 0)"

    def test_cpp_flag_prints_backend_message(self, capsys):
        from pyladr.apps.prover9 import run_prover
        fixture = Path(__file__).parent.parent / "fixtures" / "inputs" / "simple_proof.in"
        if not fixture.exists():
            pytest.skip("simple_proof.in not found")
        cpp_backend.disable()
        run_prover(["pyprover9", "-f", str(fixture), "--cpp", "-max_seconds", "3"])
        captured = capsys.readouterr()
        assert "C++ backend enabled" in captured.out
        cpp_backend.disable()
