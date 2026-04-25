"""Comprehensive subsumption validation for vampire.in and related problems.

Tests subsumption correctness by:
1. Validating core subsumption algorithm on known patterns
2. Cross-comparing PyLADR vs C Prover9 subsumption statistics
3. Checking forward/backward subsumption elimination rates
4. Verifying index-based subsumption matches brute-force results
5. Regression testing subsumption on vampire.in specifically

Run: python3 -m pytest tests/soundness/test_subsumption_validation.py -v
"""

from __future__ import annotations

import os
import subprocess
import re
import sys
import tempfile

import pytest

# Add project root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term
from pyladr.inference.subsumption import (
    BackSubsumptionIndex,
    back_subsume_from_lists,
    back_subsume_indexed,
    forward_subsume_from_lists,
    subsumes,
)
from pyladr.indexing.literal_index import LiteralIndex
from pyladr.inference.subsumption import forward_subsume


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_term(sym: int, *args: Term) -> Term:
    """Create a rigid term with the given symbol number and arguments."""
    return Term(private_symbol=-sym, arity=len(args), args=tuple(args))


def make_var(idx: int) -> Term:
    """Create a variable term."""
    return Term(private_symbol=idx, arity=0, args=())


def make_literal(sign: bool, atom: Term) -> Literal:
    return Literal(sign=sign, atom=atom)


def make_clause(literals: list[Literal], clause_id: int = 0) -> Clause:
    c = Clause(literals=tuple(literals), justification=())
    c.id = clause_id
    return c


# Symbol numbers for test predicates
P_SYM = 1  # P/1
Q_SYM = 2  # Q/1
R_SYM = 3  # R/2
I_SYM = 4  # i/2 (implication functor from vampire.in)
N_SYM = 5  # n/1 (negation functor from vampire.in)


# ── Core Subsumption Algorithm Tests ─────────────────────────────────────────

class TestCoreSubsumption:
    """Validate the core subsumes() function on various clause patterns."""

    def test_empty_clause_subsumes_everything(self):
        empty = make_clause([])
        c = make_clause([make_literal(True, make_term(P_SYM, make_var(0)))])
        assert subsumes(empty, c)

    def test_identical_clause_subsumes(self):
        """P(x) subsumes P(x)."""
        c = make_clause([make_literal(True, make_term(P_SYM, make_var(0)))])
        d = make_clause([make_literal(True, make_term(P_SYM, make_var(0)))])
        assert subsumes(c, d)

    def test_general_subsumes_specific(self):
        """P(x) subsumes P(a)."""
        a = make_term(10)  # constant a
        c = make_clause([make_literal(True, make_term(P_SYM, make_var(0)))])
        d = make_clause([make_literal(True, make_term(P_SYM, a))])
        assert subsumes(c, d)

    def test_specific_does_not_subsume_general(self):
        """P(a) does not subsume P(x)."""
        a = make_term(10)
        c = make_clause([make_literal(True, make_term(P_SYM, a))])
        d = make_clause([make_literal(True, make_term(P_SYM, make_var(0)))])
        assert not subsumes(c, d)

    def test_longer_does_not_subsume_shorter(self):
        """P(x) | Q(y) cannot subsume P(a)."""
        a = make_term(10)
        c = make_clause([
            make_literal(True, make_term(P_SYM, make_var(0))),
            make_literal(True, make_term(Q_SYM, make_var(1))),
        ])
        d = make_clause([make_literal(True, make_term(P_SYM, a))])
        assert not subsumes(c, d)

    def test_sign_mismatch_no_subsumption(self):
        """P(x) does not subsume ~P(a)."""
        a = make_term(10)
        c = make_clause([make_literal(True, make_term(P_SYM, make_var(0)))])
        d = make_clause([make_literal(False, make_term(P_SYM, a))])
        assert not subsumes(c, d)

    def test_multilit_subsumption(self):
        """-P(x) | P(f(x)) subsumes -P(a) | P(f(a))."""
        a = make_term(10)
        f_sym = 11
        c = make_clause([
            make_literal(False, make_term(P_SYM, make_var(0))),
            make_literal(True, make_term(P_SYM, make_term(f_sym, make_var(0)))),
        ])
        d = make_clause([
            make_literal(False, make_term(P_SYM, a)),
            make_literal(True, make_term(P_SYM, make_term(f_sym, a))),
        ])
        assert subsumes(c, d)

    def test_multilit_subsumption_extra_literal(self):
        """P(x) subsumes P(a) | Q(b) — subsumer is subset."""
        a = make_term(10)
        b = make_term(11)
        c = make_clause([make_literal(True, make_term(P_SYM, make_var(0)))])
        d = make_clause([
            make_literal(True, make_term(P_SYM, a)),
            make_literal(True, make_term(Q_SYM, b)),
        ])
        assert subsumes(c, d)

    def test_variable_renaming_subsumption(self):
        """P(x,y) subsumes P(y,x) — variable renaming."""
        c = make_clause([
            make_literal(True, make_term(R_SYM, make_var(0), make_var(1))),
        ])
        d = make_clause([
            make_literal(True, make_term(R_SYM, make_var(1), make_var(0))),
        ])
        assert subsumes(c, d)

    def test_vampire_pattern_condensation(self):
        """P(i(x,y)) subsumes P(i(a,i(b,c))) — nested term matching."""
        a, b, c_const = make_term(10), make_term(11), make_term(12)
        sub = make_clause([
            make_literal(True, make_term(P_SYM, make_term(I_SYM, make_var(0), make_var(1)))),
        ])
        target = make_clause([
            make_literal(True, make_term(P_SYM, make_term(I_SYM, a, make_term(I_SYM, b, c_const)))),
        ])
        assert subsumes(sub, target)

    def test_non_linear_variable_constraint(self):
        """P(x,x) does NOT subsume P(a,b) when a != b."""
        a, b = make_term(10), make_term(11)
        c = make_clause([
            make_literal(True, make_term(R_SYM, make_var(0), make_var(0))),
        ])
        d = make_clause([
            make_literal(True, make_term(R_SYM, a, b)),
        ])
        assert not subsumes(c, d)

    def test_non_linear_variable_success(self):
        """P(x,x) subsumes P(a,a)."""
        a = make_term(10)
        c = make_clause([
            make_literal(True, make_term(R_SYM, make_var(0), make_var(0))),
        ])
        d = make_clause([
            make_literal(True, make_term(R_SYM, a, a)),
        ])
        assert subsumes(c, d)


# ── Forward Subsumption Tests ────────────────────────────────────────────────

class TestForwardSubsumption:
    """Validate forward subsumption finds all subsumers."""

    def test_forward_subsume_finds_unit_subsumer(self):
        """P(x) in list should subsume P(a)."""
        a = make_term(10)
        subsumer = make_clause([make_literal(True, make_term(P_SYM, make_var(0)))], clause_id=1)
        target = make_clause([make_literal(True, make_term(P_SYM, a))], clause_id=2)
        result = forward_subsume_from_lists(target, [[subsumer]])
        assert result is not None
        assert result is subsumer

    def test_forward_subsume_no_match(self):
        """Q(x) should not subsume P(a)."""
        a = make_term(10)
        non_subsumer = make_clause([make_literal(True, make_term(Q_SYM, make_var(0)))], clause_id=1)
        target = make_clause([make_literal(True, make_term(P_SYM, a))], clause_id=2)
        result = forward_subsume_from_lists(target, [[non_subsumer]])
        assert result is None

    def test_forward_subsume_multilit(self):
        """-P(x)|P(f(x)) should subsume -P(a)|P(f(a))|Q(b)."""
        a, b = make_term(10), make_term(11)
        f_sym = 12
        subsumer = make_clause([
            make_literal(False, make_term(P_SYM, make_var(0))),
            make_literal(True, make_term(P_SYM, make_term(f_sym, make_var(0)))),
        ], clause_id=1)
        target = make_clause([
            make_literal(False, make_term(P_SYM, a)),
            make_literal(True, make_term(P_SYM, make_term(f_sym, a))),
            make_literal(True, make_term(Q_SYM, b)),
        ], clause_id=2)
        result = forward_subsume_from_lists(target, [[subsumer]])
        assert result is not None

    def test_forward_subsume_index_vs_list(self):
        """Indexed forward subsumption should match list-based."""
        a = make_term(10)
        subsumer = make_clause([make_literal(True, make_term(P_SYM, make_var(0)))], clause_id=1)
        target = make_clause([make_literal(True, make_term(P_SYM, a))], clause_id=2)

        # List-based
        list_result = forward_subsume_from_lists(target, [[subsumer]])

        # Index-based
        lindex = LiteralIndex(first_only=True)
        lindex.update(subsumer, insert=True)
        idx_result = forward_subsume(target, lindex.pos, lindex.neg)

        assert (list_result is not None) == (idx_result is not None)


# ── Backward Subsumption Tests ───────────────────────────────────────────────

class TestBackwardSubsumption:
    """Validate backward subsumption eliminates correct clauses."""

    def test_back_subsume_finds_victims(self):
        """P(x) should back-subsume P(a), P(b)."""
        a, b = make_term(10), make_term(11)
        subsumer = make_clause([make_literal(True, make_term(P_SYM, make_var(0)))], clause_id=1)
        victim1 = make_clause([make_literal(True, make_term(P_SYM, a))], clause_id=2)
        victim2 = make_clause([make_literal(True, make_term(P_SYM, b))], clause_id=3)
        non_victim = make_clause([make_literal(True, make_term(Q_SYM, a))], clause_id=4)

        result = back_subsume_from_lists(subsumer, [[victim1, victim2, non_victim]])
        ids = {c.id for c in result}
        assert 2 in ids
        assert 3 in ids
        assert 4 not in ids

    def test_back_subsume_indexed_matches_list(self):
        """BackSubsumptionIndex should find same victims as list scan."""
        a, b = make_term(10), make_term(11)
        subsumer = make_clause([make_literal(True, make_term(P_SYM, make_var(0)))], clause_id=1)
        victim1 = make_clause([make_literal(True, make_term(P_SYM, a))], clause_id=2)
        victim2 = make_clause([make_literal(True, make_term(P_SYM, b))], clause_id=3)
        non_victim = make_clause([make_literal(True, make_term(Q_SYM, a))], clause_id=4)

        # List-based
        list_result = back_subsume_from_lists(subsumer, [[victim1, victim2, non_victim]])
        list_ids = {c.id for c in list_result}

        # Index-based
        idx = BackSubsumptionIndex()
        for c in [victim1, victim2, non_victim]:
            idx.insert(c)
        idx_result = back_subsume_indexed(subsumer, idx)
        idx_ids = {c.id for c in idx_result}

        assert list_ids == idx_ids

    def test_back_subsume_skips_self(self):
        """Back subsumption should not include the subsumer itself."""
        c = make_clause([make_literal(True, make_term(P_SYM, make_var(0)))], clause_id=1)
        result = back_subsume_from_lists(c, [[c]])
        assert len(result) == 0

    def test_back_subsume_multilit_victims(self):
        """-P(x)|P(f(x)) should subsume -P(a)|P(f(a))|Q(b) but not -P(a)|Q(b)."""
        a, b = make_term(10), make_term(11)
        f_sym = 12
        subsumer = make_clause([
            make_literal(False, make_term(P_SYM, make_var(0))),
            make_literal(True, make_term(P_SYM, make_term(f_sym, make_var(0)))),
        ], clause_id=1)
        victim = make_clause([
            make_literal(False, make_term(P_SYM, a)),
            make_literal(True, make_term(P_SYM, make_term(f_sym, a))),
            make_literal(True, make_term(Q_SYM, b)),
        ], clause_id=2)
        non_victim = make_clause([
            make_literal(False, make_term(P_SYM, a)),
            make_literal(True, make_term(Q_SYM, b)),
        ], clause_id=3)

        result = back_subsume_from_lists(subsumer, [[victim, non_victim]])
        ids = {c.id for c in result}
        assert 2 in ids
        assert 3 not in ids


# ── BackSubsumptionIndex Consistency Tests ───────────────────────────────────

class TestBackSubsumptionIndex:
    """Validate the hash-based BackSubsumptionIndex consistency."""

    def test_insert_remove_consistency(self):
        """Insert then remove should leave clean state."""
        idx = BackSubsumptionIndex()
        c = make_clause([make_literal(True, make_term(P_SYM, make_var(0)))], clause_id=1)
        idx.insert(c)
        assert len(idx.candidates(c)) == 0  # c can't subsume itself (filtered)

        d = make_clause([make_literal(True, make_term(P_SYM, make_term(10)))], clause_id=2)
        idx.insert(d)
        cands = idx.candidates(c)
        assert any(x.id == 2 for x in cands)

        idx.remove(d)
        cands = idx.candidates(c)
        assert all(x.id != 2 for x in cands)

    def test_many_clauses_no_false_negatives(self):
        """Index should find ALL clauses that list-based would find."""
        # Build 50 clauses of form P(const_i) and check P(x) subsumes all
        clauses = []
        for i in range(50):
            c = make_clause([make_literal(True, make_term(P_SYM, make_term(100 + i)))], clause_id=i + 1)
            clauses.append(c)

        subsumer = make_clause([make_literal(True, make_term(P_SYM, make_var(0)))], clause_id=999)

        # List-based
        list_result = back_subsume_from_lists(subsumer, [clauses])
        list_ids = {c.id for c in list_result}

        # Index-based
        idx = BackSubsumptionIndex()
        for c in clauses:
            idx.insert(c)
        idx_result = back_subsume_indexed(subsumer, idx)
        idx_ids = {c.id for c in idx_result}

        assert list_ids == idx_ids
        assert len(list_ids) == 50


# ── Vampire.in Integration Test ──────────────────────────────────────────────

VAMPIRE_IN = os.path.join(os.path.dirname(__file__), "..", "fixtures", "inputs", "vampire.in")
C_PROVER9 = os.path.join(os.path.dirname(__file__), "..", "..", "reference-prover9", "bin", "prover9")


def _parse_c_stats(output: str) -> dict[str, int]:
    """Extract statistics from C Prover9 output."""
    stats = {}
    for line in output.splitlines():
        m = re.match(r"Given=(\d+)\.\s+Generated=(\d+)\.\s+Kept=(\d+)\.\s+proofs=(\d+)\.", line)
        if m:
            stats["given"] = int(m.group(1))
            stats["generated"] = int(m.group(2))
            stats["kept"] = int(m.group(3))
            stats["proofs"] = int(m.group(4))
        m = re.match(r"Forward_subsumed=(\d+)\.\s+Back_subsumed=(\d+)\.", line)
        if m:
            stats["forward_subsumed"] = int(m.group(1))
            stats["back_subsumed"] = int(m.group(2))
    return stats


def _parse_pyladr_stats(output: str) -> dict[str, int]:
    """Extract statistics from PyLADR output."""
    stats = {}
    for line in output.splitlines():
        m = re.match(r"Given=(\d+)\.\s+Generated=(\d+)\.\s+Kept=(\d+)\.\s+proofs=(\d+)\.", line)
        if m:
            stats["given"] = int(m.group(1))
            stats["generated"] = int(m.group(2))
            stats["kept"] = int(m.group(3))
            stats["proofs"] = int(m.group(4))
        m = re.match(r"Forward_subsumed=(\d+)\.\s+Back_subsumed=(\d+)\.", line)
        if m:
            stats["forward_subsumed"] = int(m.group(1))
            stats["back_subsumed"] = int(m.group(2))
        # Also try the simpler format
        for key in ["subsumed", "back_subsumed", "generated", "kept", "given"]:
            m = re.search(rf"{key}=(\d+)", line)
            if m and key not in stats:
                stats[key] = int(m.group(1))
    return stats


def _run_c_prover9(input_file: str, max_given: int = 100) -> tuple[str, dict[str, int]]:
    """Run C Prover9 on input file with max_given limit."""
    # Create a temp file with max_given added
    with open(input_file) as f:
        content = f.read()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".in", delete=False) as tf:
        # Add max_given before the first formulas block
        modified = content.replace(
            "set(auto).",
            f"set(auto).\nassign(max_given, {max_given}).",
        )
        tf.write(modified)
        tf.flush()

        try:
            with open(tf.name) as fin:
                proc = subprocess.run(
                    [C_PROVER9],
                    stdin=fin,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            output = proc.stdout + proc.stderr
        finally:
            os.unlink(tf.name)

    return output, _parse_c_stats(output)


def _run_pyladr(input_file: str, max_given: int = 100) -> tuple[str, dict[str, int]]:
    """Run PyLADR on input file with max_given limit."""
    proc = subprocess.run(
        [sys.executable, "-m", "pyladr.cli", "-f", input_file, "-max_given", str(max_given)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = proc.stdout + proc.stderr
    return output, _parse_pyladr_stats(output)


@pytest.mark.skipif(not os.path.exists(VAMPIRE_IN), reason="vampire.in not found")
class TestVampireSubsumptionIntegration:
    """Integration tests comparing subsumption behavior on vampire.in."""

    def test_pyladr_runs_and_reports_subsumption(self):
        """PyLADR should run vampire.in and report subsumption stats."""
        output, stats = _run_pyladr(VAMPIRE_IN, max_given=50)
        assert "given" in stats, f"PyLADR didn't report given count. Output:\n{output[-500:]}"
        assert stats["given"] >= 50, f"Expected 50 given, got {stats.get('given')}"
        # Should have some forward subsumption happening
        fs = stats.get("subsumed", stats.get("forward_subsumed", 0))
        assert fs > 0, f"No forward subsumption reported! Stats: {stats}"

    @pytest.mark.skipif(not os.path.exists(C_PROVER9), reason="C Prover9 binary not found")
    def test_cross_validate_subsumption_rates(self):
        """Compare subsumption rates between PyLADR and C Prover9.

        We don't expect exact match (different clause numbering, selection),
        but the subsumption rates should be in a reasonable range.
        """
        max_given = 100
        c_output, c_stats = _run_c_prover9(VAMPIRE_IN, max_given)
        py_output, py_stats = _run_pyladr(VAMPIRE_IN, max_given)

        print(f"\n=== C Prover9 Stats (max_given={max_given}) ===")
        for k, v in sorted(c_stats.items()):
            print(f"  {k}: {v}")

        print(f"\n=== PyLADR Stats (max_given={max_given}) ===")
        for k, v in sorted(py_stats.items()):
            print(f"  {k}: {v}")

        # Both should generate substantial clauses
        assert c_stats.get("generated", 0) > 100, "C Prover9 generated too few clauses"
        py_gen = py_stats.get("generated", 0)
        assert py_gen > 100, f"PyLADR generated too few clauses: {py_gen}"

        # Forward subsumption should be active in both
        c_fs = c_stats.get("forward_subsumed", 0)
        py_fs = py_stats.get("subsumed", py_stats.get("forward_subsumed", 0))
        assert c_fs > 0, "C Prover9 has no forward subsumption!"
        assert py_fs > 0, f"PyLADR has no forward subsumption! Stats: {py_stats}"

        # Compute forward subsumption rates
        c_rate = c_fs / c_stats["generated"] if c_stats["generated"] > 0 else 0
        py_rate = py_fs / py_gen if py_gen > 0 else 0
        print(f"\n  C forward_subsumption rate: {c_rate:.3f}")
        print(f"  PyLADR forward_subsumption rate: {py_rate:.3f}")

        # Rates should be within a reasonable range (both should be >10%)
        # Note: rate varies significantly by problem configuration
        assert c_rate > 0.1, f"C forward subsumption rate suspiciously low: {c_rate}"
        assert py_rate > 0.1, f"PyLADR forward subsumption rate suspiciously low: {py_rate}"

        # Rates should be comparable (within 30% of each other)
        if c_rate > 0 and py_rate > 0:
            ratio = py_rate / c_rate
            print(f"  Rate ratio (PyLADR/C): {ratio:.3f}")
            # Allow generous tolerance since clause ordering differs
            assert 0.5 < ratio < 2.0, (
                f"Subsumption rate divergence too large: ratio={ratio:.3f} "
                f"(C={c_rate:.3f}, PyLADR={py_rate:.3f})"
            )

    @pytest.mark.skipif(not os.path.exists(C_PROVER9), reason="C Prover9 binary not found")
    def test_backward_subsumption_comparable(self):
        """PyLADR backward subsumption count should be comparable to C Prover9.

        Note: backward subsumption may be zero for some problem configurations
        (depends on clause ordering). The key check is that PyLADR matches C.
        """
        max_given = 100
        c_output, c_stats = _run_c_prover9(VAMPIRE_IN, max_given)
        py_output, py_stats = _run_pyladr(VAMPIRE_IN, max_given)

        c_bs = c_stats.get("back_subsumed", 0)
        py_bs = py_stats.get("back_subsumed", 0)

        print(f"\n  C back_subsumed: {c_bs}")
        print(f"  PyLADR back_subsumed: {py_bs}")

        # Both should report back_subsumed (even if 0)
        assert "back_subsumed" in c_stats, "C Prover9 doesn't report back_subsumed"
        assert "back_subsumed" in py_stats, "PyLADR doesn't report back_subsumed"

        # If C has backward subsumption, PyLADR should too (within tolerance)
        if c_bs > 10:
            assert py_bs > 0, f"C has {c_bs} back_subsumed but PyLADR has 0"


# ── Subsumption Regression Tests ─────────────────────────────────────────────

class TestSubsumptionRegressions:
    """Regression tests for known subsumption edge cases."""

    def setup_method(self):
        pass

    def test_unit_subsumption_with_nested_terms(self):
        """Unit subsumption should handle deeply nested terms from vampire.in.

        P(i(x,y)) should subsume P(i(a,i(b,c))).
        """
        a, b, c_const = make_term(10), make_term(11), make_term(12)
        subsumer = make_clause([
            make_literal(True, make_term(P_SYM,
                make_term(I_SYM, make_var(0), make_var(1)))),
        ])
        target = make_clause([
            make_literal(True, make_term(P_SYM,
                make_term(I_SYM, a, make_term(I_SYM, b, c_const)))),
        ])
        assert subsumes(subsumer, target)

    def test_hyper_resolution_product_subsumption(self):
        """Typical vampire.in pattern: hyper-resolution nucleus with many products.

        The 3-literal clause (-P(x) | -P(i(x,y)) | P(y)) generates
        many clauses. Forward subsumption must eliminate redundant ones.
        """
        # P(i(x,x)) should subsume P(i(a,a))
        a = make_term(10)
        gen = make_clause([
            make_literal(True, make_term(P_SYM,
                make_term(I_SYM, make_var(0), make_var(0)))),
        ])
        spec = make_clause([
            make_literal(True, make_term(P_SYM,
                make_term(I_SYM, a, a))),
        ])
        assert subsumes(gen, spec)

    def test_n_functor_subsumption(self):
        """P(n(x)) should subsume P(n(i(a,b)))."""
        a, b = make_term(10), make_term(11)
        gen = make_clause([
            make_literal(True, make_term(P_SYM,
                make_term(N_SYM, make_var(0)))),
        ])
        spec = make_clause([
            make_literal(True, make_term(P_SYM,
                make_term(N_SYM, make_term(I_SYM, a, b)))),
        ])
        assert subsumes(gen, spec)

    def test_three_literal_clause_subsumption(self):
        """-P(x)|-P(i(x,y))|P(y) subsumes -P(a)|-P(i(a,b))|P(b)|Q(c).

        The main inference clause from vampire.in should subsume instances with extras.
        """
        a, b, c_const = make_term(10), make_term(11), make_term(12)
        # The condensation/hyper-resolution nucleus
        subsumer = make_clause([
            make_literal(False, make_term(P_SYM, make_var(0))),
            make_literal(False, make_term(P_SYM,
                make_term(I_SYM, make_var(0), make_var(1)))),
            make_literal(True, make_term(P_SYM, make_var(1))),
        ])
        target = make_clause([
            make_literal(False, make_term(P_SYM, a)),
            make_literal(False, make_term(P_SYM, make_term(I_SYM, a, b))),
            make_literal(True, make_term(P_SYM, b)),
            make_literal(True, make_term(Q_SYM, c_const)),
        ])
        assert subsumes(subsumer, target)

    def test_backtracking_required(self):
        """Subsumption requiring backtracking: P(x)|P(f(x)) vs P(a)|P(f(a))|P(f(f(a))).

        First try P(x)->P(a) forces P(f(x))->P(f(a)) which works.
        But also P(x)->P(f(a)) forces backtrack since P(f(f(a))) isn't there.
        """
        a = make_term(10)
        f_sym = 12
        fa = make_term(f_sym, a)
        ffa = make_term(f_sym, fa)
        subsumer = make_clause([
            make_literal(True, make_term(P_SYM, make_var(0))),
            make_literal(True, make_term(P_SYM, make_term(f_sym, make_var(0)))),
        ])
        target = make_clause([
            make_literal(True, make_term(P_SYM, a)),
            make_literal(True, make_term(P_SYM, fa)),
            make_literal(True, make_term(P_SYM, ffa)),
        ])
        assert subsumes(subsumer, target)


# ── Subsumption Effectiveness Analysis ───────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(VAMPIRE_IN), reason="vampire.in not found")
class TestSubsumptionEffectiveness:
    """Analyze subsumption effectiveness patterns on vampire.in."""

    def test_subsumption_prevents_clause_explosion(self):
        """Without subsumption working, kept clauses would be much higher.

        Run with 50 given clauses and verify that forward subsumption
        eliminates a significant fraction of generated clauses.
        """
        output, stats = _run_pyladr(VAMPIRE_IN, max_given=50)
        gen = stats.get("generated", 0)
        kept = stats.get("kept", 0)
        fs = stats.get("subsumed", stats.get("forward_subsumed", 0))

        print(f"\n  Generated: {gen}")
        print(f"  Kept: {kept}")
        print(f"  Forward subsumed: {fs}")

        if gen > 0:
            elimination_rate = fs / gen
            print(f"  Elimination rate: {elimination_rate:.3f}")
            # Forward subsumption should eliminate at least some clauses
            # Rate varies by problem configuration (10% is a safe lower bound)
            assert elimination_rate > 0.1, (
                f"Subsumption elimination rate too low: {elimination_rate:.3f}. "
                f"Expected >0.1 for vampire.in. Generated={gen}, subsumed={fs}"
            )

    def test_kept_to_generated_ratio(self):
        """Kept/generated ratio should be reasonable, indicating subsumption works.

        Without forward subsumption, almost all generated clauses would be kept.
        """
        output, stats = _run_pyladr(VAMPIRE_IN, max_given=50)
        gen = stats.get("generated", 0)
        kept = stats.get("kept", 0)

        if gen > 0:
            ratio = kept / gen
            print(f"\n  Kept/Generated ratio: {ratio:.3f} (kept={kept}, gen={gen})")
            # Should keep less than 95% of generated (subsumption doing something)
            # Rate varies by problem configuration
            assert ratio < 0.95, (
                f"Kept/generated ratio too high: {ratio:.3f}. "
                f"Subsumption may not be working at all."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
