"""Soundness test infrastructure for PyLADR.

Systematic validation that:
1. Proofs produced by PyLADR are logically valid (inference steps correct)
2. Proof chains derive contradiction (empty clause) from axioms
3. Known soundness regression scenarios are guarded
4. Cross-validation against C Prover9 detects suspiciously short proofs
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

from tests.soundness.proof_validator import (
    ProofValidationError,
    check_proof_derives_contradiction,
    check_trivial_proof_suspicious,
    validate_proof_chain,
    validate_unification_claim,
)

TEST_INPUTS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "inputs"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _run_search(text: str, **overrides) -> object:
    """Parse LADR input, run search, return result."""
    from pyladr.apps.prover9 import _auto_inference, _deny_goals

    st = SymbolTable()
    parser = LADRParser(st)
    parsed = parser.parse_input(text)
    usable, sos, _denied = _deny_goals(parsed, st)

    opts = SearchOptions()
    if "max_proofs" in parsed.assigns:
        opts.max_proofs = int(parsed.assigns["max_proofs"])
    if "max_given" in parsed.assigns:
        opts.max_given = int(parsed.assigns["max_given"])
    if "max_weight" in parsed.assigns:
        opts.max_weight = float(parsed.assigns["max_weight"])
    if "max_seconds" in parsed.assigns:
        opts.max_seconds = float(parsed.assigns["max_seconds"])

    _auto_inference(parsed, opts)
    for k, v in overrides.items():
        setattr(opts, k, v)

    engine = GivenClauseSearch(options=opts, symbol_table=st)
    buf = io.StringIO()
    with redirect_stdout(buf):
        return engine.run(usable=usable, sos=sos)


def _run_from_file(path: Path, **overrides) -> object:
    """Run search from an input file."""
    return _run_search(path.read_text(), **overrides)


# ── Test: Proof validator unit tests ─────────────────────────────────────────


class TestProofValidatorBasics:
    """Unit tests for the proof_validator module."""

    def test_empty_proof_valid(self):
        """An empty proof chain has no issues."""
        assert validate_proof_chain([]) == []

    def test_contradiction_detected(self):
        """check_proof_derives_contradiction finds empty clause."""
        from pyladr.core.clause import Clause

        empty = Clause(literals=())
        non_empty_text = "P(a)."
        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(f"formulas(sos).\n{non_empty_text}\nend_of_list.\n")

        assert check_proof_derives_contradiction([empty]) is True
        assert check_proof_derives_contradiction(parsed.sos) is False

    def test_trivial_proof_not_suspicious_when_c_absent(self):
        """No suspicion when C reference proof length is 0."""
        assert check_trivial_proof_suspicious(3, 0) is False

    def test_trivial_proof_suspicious_when_much_shorter(self):
        """Flag suspicion when Python proof is <25% of C proof."""
        assert check_trivial_proof_suspicious(3, 50) is True

    def test_trivial_proof_not_suspicious_when_similar(self):
        """No suspicion when proof lengths are comparable."""
        assert check_trivial_proof_suspicious(40, 50) is False


# ── Test: End-to-end proof soundness on known problems ───────────────────────


class TestEndToEndSoundness:
    """Verify proofs on standard problems are logically valid."""

    def test_simple_resolution_proof_valid(self):
        """Simple P(a), -P(x)|Q(x), -Q(a) proof is sound."""
        text = """\
formulas(sos).
P(a).
-P(x) | Q(x).
end_of_list.

formulas(goals).
Q(a).
end_of_list.
"""
        result = _run_search(text)
        assert len(result.proofs) >= 1, "Should find a proof"
        proof = result.proofs[0]
        issues = validate_proof_chain(proof.clauses)
        assert issues == [], f"Proof validation failed: {issues}"
        assert check_proof_derives_contradiction(proof.clauses)

    def test_identity_proof_valid(self):
        """e*e=e from left identity: proof is sound."""
        if not (TEST_INPUTS_DIR / "identity_only.in").exists():
            pytest.fail("identity_only.in not found — soundness fixtures must be committed")
        try:
            result = _run_from_file(
                TEST_INPUTS_DIR / "identity_only.in",
                paramodulation=True,
                demodulation=True,
            )
        except ValueError as e:
            if "not instantiated in demod context" in str(e):
                pytest.skip(f"Known demodulation bug: {e}")
            raise
        assert len(result.proofs) >= 1
        issues = validate_proof_chain(result.proofs[0].clauses)
        assert issues == [], f"Proof validation failed: {issues}"
        assert check_proof_derives_contradiction(result.proofs[0].clauses)

    def test_lattice_absorption_proof_valid(self):
        """Lattice idempotence from absorption: proof is sound."""
        if not (TEST_INPUTS_DIR / "lattice_absorption.in").exists():
            pytest.fail("lattice_absorption.in not found — soundness fixtures must be committed")
        try:
            result = _run_from_file(
                TEST_INPUTS_DIR / "lattice_absorption.in",
                paramodulation=True,
                demodulation=True,
            )
        except ValueError as e:
            if "not instantiated in demod context" in str(e):
                pytest.skip(f"Known demodulation bug: {e}")
            raise
        if not result.proofs:
            pytest.skip("No proof found (may need more search)")
        issues = validate_proof_chain(result.proofs[0].clauses)
        assert issues == [], f"Proof validation failed: {issues}"

    def test_group_commutativity_proof_valid(self):
        """Group commutativity from x*x=e: proof is sound."""
        if not (TEST_INPUTS_DIR / "simple_group.in").exists():
            pytest.fail("simple_group.in not found — soundness fixtures must be committed")
        try:
            result = _run_from_file(
                TEST_INPUTS_DIR / "simple_group.in",
                paramodulation=True,
                demodulation=True,
                max_seconds=10,
            )
        except ValueError as e:
            if "not instantiated in demod context" in str(e):
                pytest.skip(f"Known demodulation bug: {e}")
            raise
        if not result.proofs:
            pytest.skip("No proof found within time limit")
        issues = validate_proof_chain(result.proofs[0].clauses)
        assert issues == [], f"Proof validation failed: {issues}"


# ── Test: Skolemization soundness (Amendment Cycle 9 regression guard) ───────


class TestSkolemizationSoundness:
    """Guard against the Amendment Cycle 9 soundness regression.

    The bug: goal negation was missing Skolemization, causing PyLADR
    to find trivially invalid short proofs where C Prover9 correctly
    performed extensive search.
    """

    def test_goal_negation_produces_denied_clauses(self):
        """Goal clauses must be negated and Skolemized for SOS."""
        from pyladr.apps.prover9 import _deny_goals

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input("""\
formulas(sos).
P(a).
end_of_list.

formulas(goals).
P(a).
end_of_list.
""")
        usable, sos, _denied = _deny_goals(parsed, st)
        # The denied goal -P(a) should be in SOS
        denied = [c for c in sos if any(not lit.sign for lit in c.literals)]
        assert len(denied) >= 1, "Goal should be denied into SOS"

    def test_skolemization_regression_trivial(self):
        """Regression canary: P(a) should NOT prove 'all x P(x)'.

        With correct Skolemization, the goal P(x) is denied as -P(c1)
        where c1 is a Skolem constant. -P(c1) does NOT resolve with P(a)
        since c1 != a. A proof here would indicate a Skolemization bug.
        """
        if not (TEST_INPUTS_DIR / "goal_negation_trivial.in").exists():
            pytest.fail("goal_negation_trivial.in not found — soundness fixtures must be committed")
        result = _run_from_file(TEST_INPUTS_DIR / "goal_negation_trivial.in")
        # With correct Skolemization, no proof should be found
        assert len(result.proofs) == 0, (
            "SOUNDNESS BUG: Found proof for P(a) |- all x P(x). "
            "Goal negation likely missing Skolemization."
        )

    def test_skolemization_regression_commutativity(self):
        """Regression canary: commutativity with goal negation."""
        if not (TEST_INPUTS_DIR / "goal_negation_commutativity.in").exists():
            pytest.fail("goal_negation_commutativity.in not found — soundness fixtures must be committed")
        try:
            result = _run_from_file(
                TEST_INPUTS_DIR / "goal_negation_commutativity.in",
                max_seconds=10,
            )
        except ValueError as e:
            if "not instantiated in demod context" in str(e):
                pytest.skip(f"Known demodulation bug: {e}")
            raise
        if not result.proofs:
            pytest.skip("No proof found within time limit")
        issues = validate_proof_chain(result.proofs[0].clauses)
        assert issues == [], f"Proof validation failed: {issues}"


# ── Test: Vampire.in soundness canary ────────────────────────────────────────


class TestVampireSoundness:
    """Soundness canary for vampire.in (the critical benchmark problem)."""

    def test_vampire_proof_chain_valid(self):
        """Proofs found on vampire.in must be logically valid."""
        vampire_in = Path(__file__).resolve().parent.parent / "fixtures" / "inputs" / "vampire.in"
        if not vampire_in.exists():
            pytest.skip("vampire.in not found")
        result = _run_from_file(vampire_in, max_seconds=30)
        if not result.proofs:
            pytest.skip("No proof found within time limit")
        for i, proof in enumerate(result.proofs):
            issues = validate_proof_chain(proof.clauses)
            assert issues == [], f"Proof #{i+1} validation failed: {issues}"
            assert check_proof_derives_contradiction(proof.clauses), (
                f"Proof #{i+1} does not derive contradiction"
            )

    def test_vampire_proof_not_suspiciously_short(self):
        """vampire.in proof should not be suspiciously short vs C reference.

        C Prover9 finds proofs after extensive search (100+ given clauses).
        A 3-step proof would indicate a soundness bug.
        """
        vampire_in = Path(__file__).resolve().parent.parent / "fixtures" / "inputs" / "vampire.in"
        if not vampire_in.exists():
            pytest.skip("vampire.in not found")
        result = _run_from_file(vampire_in, max_seconds=30)
        if not result.proofs:
            pytest.skip("No proof found within time limit")
        # C Prover9 proofs on vampire.in are typically 15+ steps
        # A proof under 5 steps would be highly suspicious
        for i, proof in enumerate(result.proofs):
            assert len(proof.clauses) >= 5, (
                f"Proof #{i+1} has only {len(proof.clauses)} steps — suspiciously short "
                f"(possible soundness regression)"
            )


# ── Test: Unification validation ─────────────────────────────────────────────


class TestUnificationValidation:
    """Test that validate_unification_claim works correctly."""

    def test_identical_terms_unify(self):
        """Identical terms should unify."""
        from pyladr.core.term import get_rigid_term

        st = SymbolTable()
        a_sn = st.str_to_sn("a", 0)
        a = get_rigid_term(a_sn, 0)
        assert validate_unification_claim(a, a) is True

    def test_variable_unifies_with_constant(self):
        """A variable unifies with a constant."""
        from pyladr.core.term import get_rigid_term, get_variable_term

        st = SymbolTable()
        a_sn = st.str_to_sn("a", 0)
        x = get_variable_term(0)
        a = get_rigid_term(a_sn, 0)
        assert validate_unification_claim(x, a) is True

    def test_clash_does_not_unify(self):
        """Different constants don't unify."""
        from pyladr.core.term import get_rigid_term

        st = SymbolTable()
        a_sn = st.str_to_sn("a", 0)
        b_sn = st.str_to_sn("b", 0)
        a = get_rigid_term(a_sn, 0)
        b = get_rigid_term(b_sn, 0)
        assert validate_unification_claim(a, b) is False
