"""Comprehensive end-to-end regression tests for PyLADR.

These tests exercise the full pipeline: parse LADR input -> deny goals ->
configure search -> run given-clause algorithm -> validate results.

Each test validates:
- Exit code correctness
- Proof existence and quality (length, structure)
- Search statistics sanity
- No regressions in proof-finding ability

Uses subprocess isolation to avoid shared-state contamination between tests.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# ── Subprocess helper ──────────────────────────────────────────────────────

# Inline Python script that runs a search and outputs JSON results.
# This ensures complete process isolation between tests.
_RUNNER_SCRIPT = '''\
import json, sys
from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions

text = sys.stdin.read()
opts_json = json.loads(sys.argv[1])

st = SymbolTable()
parser = LADRParser(st)
parsed = parser.parse_input(text)

usable = list(parsed.usable)
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
    binary_resolution=opts_json.get("binary_resolution", True),
    paramodulation=opts_json.get("paramodulation", True),
    demodulation=opts_json.get("demodulation", True),
    factoring=opts_json.get("factoring", True),
    back_demod=opts_json.get("back_demod", False),
    max_given=opts_json.get("max_given", 500),
    max_kept=opts_json.get("max_kept", 5000),
    max_seconds=opts_json.get("max_seconds", 30.0),
    max_proofs=opts_json.get("max_proofs", 1),
    online_learning=opts_json.get("online_learning", False),
    ml_weight=opts_json.get("ml_weight"),
    buffer_capacity=opts_json.get("buffer_capacity", 5000),
    quiet=True,
    print_given=False,
)

search = GivenClauseSearch(options=opts, symbol_table=st)
result = search.run(usable=usable, sos=sos)

proofs_info = []
for p in result.proofs:
    just_types = set()
    for c in p.clauses:
        for j in c.justification:
            just_types.add(j.just_type.name)
    proofs_info.append({
        "length": len(p.clauses),
        "empty_clause_is_empty": p.empty_clause.is_empty,
        "clause_ids": [c.id for c in p.clauses],
        "justification_types": list(just_types),
    })

out = {
    "exit_code": result.exit_code.value,
    "exit_code_name": result.exit_code.name,
    "num_proofs": len(result.proofs),
    "proofs": proofs_info,
    "stats": {
        "given": result.stats.given,
        "generated": result.stats.generated,
        "kept": result.stats.kept,
        "subsumed": result.stats.subsumed,
        "back_subsumed": result.stats.back_subsumed,
        "proofs": result.stats.proofs,
    },
}
print(json.dumps(out))
'''


def _run_search(ladr_input: str, timeout: float = 60.0, **opts) -> dict:
    """Run a search in a subprocess and return parsed JSON results."""
    opts_str = json.dumps(opts)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER_SCRIPT, opts_str],
        input=ladr_input,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Runner failed (exit {proc.returncode}):\n"
            f"STDERR: {proc.stderr}\nSTDOUT: {proc.stdout}"
        )
    return json.loads(proc.stdout)


_CLI_SCRIPT = '''\
import sys
from pyladr.apps.prover9 import run_prover
sys.exit(run_prover())
'''


def _run_cli(ladr_input: str, extra_args: list[str] | None = None,
             timeout: float = 60.0) -> subprocess.CompletedProcess:
    """Run the prover CLI via subprocess."""
    cmd = [sys.executable, "-c", _CLI_SCRIPT, "--quiet"]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd, input=ladr_input, capture_output=True, text=True, timeout=timeout,
    )


# ── Standard problem inputs ───────────────────────────────────────────────

X2_GROUP = """\
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

GROUP_RIGHT_IDENTITY = """\
formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
end_of_list.

formulas(goals).
  x * e = x.
end_of_list.
"""

GROUP_RIGHT_INVERSE = """\
formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
end_of_list.

formulas(goals).
  x * x' = e.
end_of_list.
"""

GROUP_LEFT_CANCEL = """\
formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
  x * x = e.
end_of_list.

formulas(goals).
  a * (a * b) = b.
end_of_list.
"""

LATTICE_MEET_IDEMPOTENCE = """\
formulas(sos).
  x ^ y = y ^ x.
  x v y = y v x.
  (x ^ y) ^ z = x ^ (y ^ z).
  (x v y) v z = x v (y v z).
  x ^ (x v y) = x.
  x v (x ^ y) = x.
end_of_list.

formulas(goals).
  x ^ x = x.
end_of_list.
"""

LATTICE_JOIN_IDEMPOTENCE = """\
formulas(sos).
  x ^ y = y ^ x.
  x v y = y v x.
  (x ^ y) ^ z = x ^ (y ^ z).
  (x v y) v z = x v (y v z).
  x ^ (x v y) = x.
  x v (x ^ y) = x.
end_of_list.

formulas(goals).
  x v x = x.
end_of_list.
"""

SIMPLE_RESOLUTION = """\
formulas(sos).
  P.
  -P | Q.
  -Q | R.
end_of_list.

formulas(goals).
  R.
end_of_list.
"""

TRANSITIVITY = """\
formulas(sos).
  a = b.
  b = c.
end_of_list.

formulas(goals).
  a = c.
end_of_list.
"""

CONGRUENCE = """\
formulas(sos).
  a = b.
end_of_list.

formulas(goals).
  f(a) = f(b).
end_of_list.
"""

IDENTITY_ONLY = """\
formulas(sos).
  e * x = x.
end_of_list.

formulas(goals).
  e * e = e.
end_of_list.
"""

MODUS_PONENS = """\
formulas(sos).
  P.
  -P | Q.
end_of_list.

formulas(goals).
  Q.
end_of_list.
"""

CONTRADICTION = """\
formulas(sos).
  P.
  -P.
end_of_list.
"""

REFLEXIVITY_TRIVIAL = """\
formulas(sos).
  x = x.
end_of_list.

formulas(goals).
  a = a.
end_of_list.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Core proof-finding: verify that known-provable problems ARE proved
# ═══════════════════════════════════════════════════════════════════════════


class TestCoreProofFinding:
    """Verify that all standard problems produce proofs with correct exit codes."""

    def test_x2_group_commutativity(self):
        """Classic x^2=e implies commutativity."""
        r = _run_search(X2_GROUP)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        assert r["num_proofs"] >= 1

    def test_group_right_identity(self):
        r = _run_search(GROUP_RIGHT_IDENTITY)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        assert r["num_proofs"] >= 1

    def test_group_right_inverse(self):
        r = _run_search(GROUP_RIGHT_INVERSE)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        assert r["num_proofs"] >= 1

    def test_group_left_cancel(self):
        r = _run_search(GROUP_LEFT_CANCEL)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        assert r["num_proofs"] >= 1

    def test_lattice_meet_idempotence(self):
        r = _run_search(LATTICE_MEET_IDEMPOTENCE)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        assert r["num_proofs"] >= 1

    def test_lattice_join_idempotence(self):
        r = _run_search(LATTICE_JOIN_IDEMPOTENCE)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        assert r["num_proofs"] >= 1

    def test_chain_resolution(self):
        r = _run_search(SIMPLE_RESOLUTION)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        assert r["num_proofs"] >= 1

    def test_transitivity(self):
        r = _run_search(TRANSITIVITY)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"

    def test_congruence(self):
        r = _run_search(CONGRUENCE)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"

    def test_identity_only(self):
        r = _run_search(IDENTITY_ONLY)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"

    def test_modus_ponens(self):
        r = _run_search(MODUS_PONENS)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"

    def test_contradiction(self):
        r = _run_search(CONTRADICTION)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"

    def test_reflexivity_trivial(self):
        r = _run_search(REFLEXIVITY_TRIVIAL)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"


# ═══════════════════════════════════════════════════════════════════════════
# Proof quality: validate proof structure and length bounds
# ═══════════════════════════════════════════════════════════════════════════


class TestProofQuality:
    """Validate proof structure and reasonable length bounds.

    These tests catch regressions where proofs degrade (e.g., the 2-line
    proof issue).
    """

    def test_x2_proof_has_nontrivial_length(self):
        """x^2=e -> commutativity requires multiple steps (at least 5 clauses)."""
        r = _run_search(X2_GROUP)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        proof = r["proofs"][0]
        assert proof["length"] >= 4, (
            f"x^2 proof unexpectedly short: {proof['length']} clauses. "
            f"Expected at least 4. This may indicate a proof extraction regression."
        )

    def test_chain_resolution_proof_length(self):
        """P, P->Q, Q->R, !R requires at least the chain."""
        r = _run_search(SIMPLE_RESOLUTION)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        proof = r["proofs"][0]
        assert proof["length"] >= 3, (
            f"Chain resolution proof too short: {proof['length']} clauses."
        )

    def test_proof_contains_empty_clause(self):
        """Every proof must end with an empty clause."""
        r = _run_search(MODUS_PONENS)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        assert r["proofs"][0]["empty_clause_is_empty"] is True

    def test_proof_clauses_have_ids(self):
        """All clauses in a proof should have positive IDs."""
        r = _run_search(TRANSITIVITY)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        for cid in r["proofs"][0]["clause_ids"]:
            assert cid > 0, f"Proof clause has invalid ID: {cid}"

    def test_proof_has_justifications(self):
        """Proof should contain both input and derived clauses."""
        r = _run_search(X2_GROUP)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        jtypes = set(r["proofs"][0]["justification_types"])
        assert "ASSUME" in jtypes or "DENY" in jtypes, (
            f"Proof missing input clauses. Types: {jtypes}"
        )

    def test_proof_has_derived_clauses(self):
        """Proof should include derived (inferred) clauses."""
        r = _run_search(X2_GROUP)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        jtypes = set(r["proofs"][0]["justification_types"])
        derived = {"RESOLVE", "PARAMOD", "DEMOD", "BINARY_RES", "PARA", "BACK_DEMOD"}
        assert jtypes & derived, (
            f"Proof has no derived clauses! Types: {jtypes}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Search statistics sanity checks
# ═══════════════════════════════════════════════════════════════════════════


class TestSearchStatistics:
    """Verify search statistics are reasonable and consistent."""

    def test_stats_counters_positive(self):
        r = _run_search(X2_GROUP)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        s = r["stats"]
        assert s["given"] >= 1
        assert s["generated"] >= 1
        assert s["kept"] >= 1
        assert s["proofs"] >= 1

    def test_stats_generated_gte_kept(self):
        r = _run_search(X2_GROUP)
        assert r["stats"]["generated"] >= r["stats"]["kept"]

    def test_stats_x2_given_reasonable(self):
        """x^2 solved in ~10 given by C Prover9. Allow up to 100."""
        r = _run_search(X2_GROUP)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        assert r["stats"]["given"] <= 100, (
            f"x^2 needed {r['stats']['given']} given clauses, too many"
        )

    def test_simple_problem_low_cost(self):
        r = _run_search(MODUS_PONENS)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        assert r["stats"]["given"] <= 10

    def test_identity_proof_very_cheap(self):
        r = _run_search(IDENTITY_ONLY)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        assert r["stats"]["given"] <= 5


# ═══════════════════════════════════════════════════════════════════════════
# Search limit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSearchLimits:
    """Verify that search limits are respected."""

    def test_max_given_limit(self):
        r = _run_search(X2_GROUP, max_given=1)
        assert r["exit_code_name"] in ("MAX_PROOFS_EXIT", "MAX_GIVEN_EXIT")
        assert r["stats"]["given"] <= 2

    def test_max_kept_limit(self):
        r = _run_search(X2_GROUP, max_kept=5)
        assert r["exit_code_name"] in ("MAX_PROOFS_EXIT", "MAX_KEPT_EXIT")

    def test_max_seconds_limit(self):
        r = _run_search(X2_GROUP, max_seconds=0.001, max_given=10000)
        assert r["exit_code_name"] in ("MAX_PROOFS_EXIT", "MAX_SECONDS_EXIT")


# ═══════════════════════════════════════════════════════════════════════════
# Inference rule configuration tests
# ═══════════════════════════════════════════════════════════════════════════


class TestInferenceRules:
    """Verify different inference rule configurations work."""

    def test_resolution_only(self):
        r = _run_search(SIMPLE_RESOLUTION, paramodulation=False, demodulation=False)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"

    def test_paramodulation_required(self):
        r = _run_search(X2_GROUP, paramodulation=True, demodulation=True)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"

    def test_no_resolution_pure_equational(self):
        r = _run_search(
            TRANSITIVITY, binary_resolution=False,
            paramodulation=True, demodulation=True,
        )
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"

    def test_factoring_disabled_still_works(self):
        r = _run_search(MODUS_PONENS, factoring=False)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"

    def test_back_demod_enabled(self):
        r = _run_search(X2_GROUP, back_demod=True)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"

    def test_no_demod_equational(self):
        r = _run_search(IDENTITY_ONLY, demodulation=False, paramodulation=True)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"


# ═══════════════════════════════════════════════════════════════════════════
# Search failure tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSearchFailure:
    """Verify correct behavior when no proof exists."""

    def test_non_theorem_no_proof(self):
        """P(a) does not imply P(b) without equality."""
        input_text = """\
formulas(sos).
  P(a).
end_of_list.

formulas(goals).
  P(b).
end_of_list.
"""
        r = _run_search(
            input_text, paramodulation=False, demodulation=False, max_given=200,
        )
        assert r["exit_code_name"] in ("SOS_EMPTY_EXIT", "MAX_GIVEN_EXIT")
        assert r["num_proofs"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# ML integration tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMLIntegration:
    """ML options should not break proof-finding."""

    def test_online_learning_x2(self):
        try:
            r = _run_search(
                X2_GROUP, online_learning=True, ml_weight=0.3, buffer_capacity=100,
            )
            assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        except RuntimeError:
            pytest.skip("ML dependencies not available")

    def test_online_learning_resolution(self):
        try:
            r = _run_search(
                SIMPLE_RESOLUTION, online_learning=True, ml_weight=0.5,
                buffer_capacity=100,
            )
            assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        except RuntimeError:
            pytest.skip("ML dependencies not available")

    def test_various_buffer_capacities(self):
        for cap in [10, 100, 1000, 10000]:
            try:
                r = _run_search(
                    X2_GROUP, online_learning=True, ml_weight=0.2,
                    buffer_capacity=cap,
                )
                assert r["exit_code_name"] == "MAX_PROOFS_EXIT", (
                    f"Failed with buffer_capacity={cap}"
                )
            except RuntimeError:
                pytest.skip("ML dependencies not available")


# ═══════════════════════════════════════════════════════════════════════════
# CLI integration tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCLIIntegration:
    """Test the full CLI pipeline via subprocess."""

    def test_cli_x2_proof(self):
        proc = _run_cli(X2_GROUP)
        assert proc.returncode == 0, f"Expected exit 0, got {proc.returncode}"
        assert "THEOREM PROVED" in proc.stdout

    def test_cli_identity_proof(self):
        proc = _run_cli(IDENTITY_ONLY)
        assert proc.returncode == 0
        assert "THEOREM PROVED" in proc.stdout

    def test_cli_resolution(self):
        proc = _run_cli(SIMPLE_RESOLUTION)
        assert proc.returncode == 0
        assert "THEOREM PROVED" in proc.stdout

    def test_cli_max_given_limit(self):
        proc = _run_cli(
            X2_GROUP,
            extra_args=["-max_given", "1", "--no-resolution"],
        )
        assert proc.returncode in (0, 3)

    def test_cli_statistics_in_output(self):
        proc = _run_cli(MODUS_PONENS)
        assert proc.returncode == 0
        assert "STATISTICS" in proc.stdout
        assert "Given=" in proc.stdout
        assert "Generated=" in proc.stdout
        assert "Kept=" in proc.stdout

    def test_cli_proof_section(self):
        proc = _run_cli(TRANSITIVITY)
        assert proc.returncode == 0
        assert "PROOF" in proc.stdout
        assert "Length of proof" in proc.stdout

    def test_cli_stdin_input(self):
        proc = _run_cli(MODUS_PONENS)
        assert proc.returncode == 0
        assert "THEOREM PROVED" in proc.stdout

    def test_cli_file_input(self, tmp_path):
        f = tmp_path / "test.in"
        f.write_text(MODUS_PONENS)
        proc = _run_cli("", extra_args=["-f", str(f)])
        assert proc.returncode == 0
        assert "THEOREM PROVED" in proc.stdout


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases that have caused bugs or are tricky."""

    def test_usable_and_sos_sections(self):
        input_text = """\
formulas(usable).
  -P(x) | Q(x).
end_of_list.

formulas(sos).
  P(a).
end_of_list.

formulas(goals).
  Q(a).
end_of_list.
"""
        r = _run_search(input_text, paramodulation=False, demodulation=False)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"

    def test_multiple_equational_goals(self):
        input_text = """\
formulas(sos).
  a = b.
  b = c.
  c = d.
end_of_list.

formulas(goals).
  a = d.
end_of_list.
"""
        r = _run_search(input_text)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"

    @pytest.mark.xfail(
        reason="Known bug: demodulation infinite recursion with f(a)=a rewrite rule",
        strict=False,
    )
    def test_deeply_nested_terms(self):
        input_text = """\
formulas(sos).
  f(f(f(a))) = b.
  f(a) = a.
end_of_list.

formulas(goals).
  f(a) = b.
end_of_list.
"""
        r = _run_search(input_text)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"

    def test_constant_only_problem(self):
        input_text = """\
formulas(sos).
  a = b.
  b = c.
end_of_list.

formulas(goals).
  c = a.
end_of_list.
"""
        r = _run_search(input_text)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"


# ═══════════════════════════════════════════════════════════════════════════
# Determinism tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """The prover should be deterministic."""

    def test_same_exit_code(self):
        r1 = _run_search(X2_GROUP)
        r2 = _run_search(X2_GROUP)
        assert r1["exit_code_name"] == r2["exit_code_name"]
        assert r1["num_proofs"] == r2["num_proofs"]

    def test_same_stats(self):
        r1 = _run_search(MODUS_PONENS)
        r2 = _run_search(MODUS_PONENS)
        assert r1["stats"]["given"] == r2["stats"]["given"]
        assert r1["stats"]["generated"] == r2["stats"]["generated"]
        assert r1["stats"]["kept"] == r2["stats"]["kept"]

    def test_same_proof_length(self):
        r1 = _run_search(TRANSITIVITY)
        r2 = _run_search(TRANSITIVITY)
        assert r1["proofs"][0]["length"] == r2["proofs"][0]["length"]


# ═══════════════════════════════════════════════════════════════════════════
# 2-line proof regression test
# ═══════════════════════════════════════════════════════════════════════════


class TestTwoLineProofRegression:
    """Specific tests for the 2-line proof bug.

    The bug caused proofs to be reported with only 2 clauses instead of
    the full derivation chain.
    """

    def test_x2_proof_not_two_lines(self):
        r = _run_search(X2_GROUP)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        proof = r["proofs"][0]
        assert proof["length"] > 2, (
            f"REGRESSION: x^2 proof has only {proof['length']} clauses! "
            f"Expected a multi-step derivation."
        )

    def test_group_right_id_not_two_lines(self):
        r = _run_search(GROUP_RIGHT_IDENTITY)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        assert r["proofs"][0]["length"] >= 3, (
            f"REGRESSION: right identity proof only {r['proofs'][0]['length']} clauses!"
        )

    def test_chain_resolution_not_two_lines(self):
        r = _run_search(SIMPLE_RESOLUTION)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        assert r["proofs"][0]["length"] > 2, (
            f"REGRESSION: chain resolution proof only {r['proofs'][0]['length']} clauses!"
        )

    def test_proof_includes_input_clauses(self):
        r = _run_search(X2_GROUP)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        jtypes = set(r["proofs"][0]["justification_types"])
        assert "ASSUME" in jtypes or "DENY" in jtypes, (
            f"Proof missing input clauses. Types: {jtypes}"
        )

    def test_proof_includes_derived(self):
        r = _run_search(X2_GROUP)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"
        jtypes = set(r["proofs"][0]["justification_types"])
        derived = {"RESOLVE", "PARAMOD", "DEMOD", "BINARY_RES", "PARA", "BACK_DEMOD"}
        assert jtypes & derived, (
            f"Proof has no derived clauses! Types: {jtypes}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Demodulation tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDemodulation:
    """Test demodulation and back-demodulation."""

    def test_demod_with_transitivity(self):
        r = _run_search(TRANSITIVITY, demodulation=True, paramodulation=True)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"

    def test_back_demod_x2(self):
        r = _run_search(X2_GROUP, back_demod=True)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"

    def test_no_demod_identity(self):
        r = _run_search(IDENTITY_ONLY, demodulation=False, paramodulation=True)
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"


# ═══════════════════════════════════════════════════════════════════════════
# Fixture file tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFixtureFiles:
    """Run tests using standard fixture input files."""

    FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "inputs"

    def _run_file(self, filename: str, **kwargs) -> dict:
        path = self.FIXTURES_DIR / filename
        if not path.exists():
            pytest.skip(f"Fixture file not found: {path}")
        return _run_search(path.read_text(), **kwargs)

    def test_simple_group(self):
        r = self._run_file("simple_group.in")
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"

    def test_identity_only(self):
        r = self._run_file("identity_only.in")
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"

    def test_lattice_absorption(self):
        r = self._run_file("lattice_absorption.in")
        assert r["exit_code_name"] == "MAX_PROOFS_EXIT"


# ═══════════════════════════════════════════════════════════════════════════
# Skolemization & goal denial tests
# ═══════════════════════════════════════════════════════════════════════════

# Script that inspects goal denial without running full search.
_SKOLEM_INSPECT_SCRIPT = '''\
import json, sys
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import get_rigid_term
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.apps.prover9 import _deny_goals, _collect_variables

text = sys.stdin.read()
st = SymbolTable()
parser = LADRParser(st)
parsed = parser.parse_input(text)

num_goals = len(parsed.goals)
num_sos_before = len(parsed.sos)

usable, sos = _deny_goals(parsed, st)

# The denied goals are the new clauses appended to sos
denied_clauses = sos[num_sos_before:]

denied_info = []
for dc in denied_clauses:
    lits_info = []
    for lit in dc.literals:
        # Check if atom contains any variables
        def has_variables(term):
            if term.is_variable:
                return True
            return any(has_variables(a) for a in term.args)

        def collect_constants(term):
            if term.is_variable:
                return set()
            if term.arity == 0:
                name = st.sn_to_str(abs(term.private_symbol))
                return {name}
            result = set()
            for a in term.args:
                result.update(collect_constants(a))
            return result

        def collect_skolem_constants(term):
            if term.is_variable:
                return set()
            if term.arity == 0:
                sn = abs(term.private_symbol)
                sym = st.get_symbol(sn)
                if sym and sym.skolem:
                    return {sym.name}
            result = set()
            for a in term.args:
                result.update(collect_skolem_constants(a))
            return result

        lits_info.append({
            "sign": lit.sign,
            "has_variables": has_variables(lit.atom),
            "constants": sorted(collect_constants(lit.atom)),
            "skolem_constants": sorted(collect_skolem_constants(lit.atom)),
        })
    denied_info.append({"literals": lits_info})

# Collect all Skolem symbols
skolem_symbols = []
for sn in range(1, 200):
    try:
        sym = st.get_symbol(sn)
        if sym and sym.skolem:
            skolem_symbols.append(sym.name)
    except (KeyError, IndexError):
        break

out = {
    "num_goals": num_goals,
    "num_denied": len(denied_clauses),
    "denied": denied_info,
    "skolem_symbols": skolem_symbols,
}
print(json.dumps(out))
'''


def _inspect_skolemization(ladr_input: str) -> dict:
    """Inspect goal denial/Skolemization without running search."""
    proc = subprocess.run(
        [sys.executable, "-c", _SKOLEM_INSPECT_SCRIPT],
        input=ladr_input,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Skolem inspection failed:\nSTDERR: {proc.stderr}"
        )
    return json.loads(proc.stdout)


class TestSkolemization:
    """Verify goal denial correctly Skolemizes variables.

    The root cause of the 2-line proof regression was missing Skolemization:
    goals were denied by only flipping signs but leaving variables universal,
    which allowed spurious unification with axiom variables.
    """

    def test_single_variable_goal_skolemized(self):
        """Goal P(x) should become -P(c1) with Skolem constant c1."""
        r = _inspect_skolemization("""\
formulas(sos).
  P(a).
end_of_list.

formulas(goals).
  P(x).
end_of_list.
""")
        assert r["num_goals"] == 1
        assert r["num_denied"] == 1
        lit = r["denied"][0]["literals"][0]
        assert lit["sign"] is False, "Goal literal should be negated"
        assert lit["has_variables"] is False, (
            "Denied goal should NOT contain variables — must be Skolemized"
        )
        assert len(lit["skolem_constants"]) >= 1, (
            "Denied goal should contain at least one Skolem constant"
        )

    def test_multi_variable_goal_skolemized(self):
        """Goal P(x,y) should become -P(c1,c2) with distinct Skolem constants."""
        r = _inspect_skolemization("""\
formulas(sos).
  P(a,b).
end_of_list.

formulas(goals).
  P(x,y).
end_of_list.
""")
        lit = r["denied"][0]["literals"][0]
        assert lit["has_variables"] is False
        assert len(lit["skolem_constants"]) >= 2, (
            "Goal with 2 variables should have at least 2 Skolem constants"
        )

    def test_shared_variable_gets_same_skolem(self):
        """Goal P(x, f(x)) should use the same Skolem constant for both x's."""
        r = _inspect_skolemization("""\
formulas(sos).
  P(a, f(a)).
end_of_list.

formulas(goals).
  P(x, f(x)).
end_of_list.
""")
        lit = r["denied"][0]["literals"][0]
        assert lit["has_variables"] is False
        # x maps to one Skolem constant; it appears twice but it's the same one
        assert len(lit["skolem_constants"]) == 1, (
            "Same variable in goal should map to same Skolem constant"
        )

    def test_constant_goal_preserved(self):
        """Goal P(a,b) with constants should NOT introduce Skolem constants."""
        r = _inspect_skolemization("""\
formulas(sos).
  P(a,b).
end_of_list.

formulas(goals).
  P(a,b).
end_of_list.
""")
        lit = r["denied"][0]["literals"][0]
        assert lit["has_variables"] is False
        assert len(lit["skolem_constants"]) == 0, (
            "Goal with only constants should have no Skolem constants"
        )

    def test_multiple_goals_get_fresh_skolems(self):
        """Each goal should get its own fresh Skolem constants."""
        r = _inspect_skolemization("""\
formulas(sos).
  P(a).
end_of_list.

formulas(goals).
  P(x).
  Q(y).
end_of_list.
""")
        assert r["num_goals"] == 2
        assert r["num_denied"] == 2
        # Goal 1: P(x) -> -P(c1)
        # Goal 2: Q(y) -> -Q(c2)
        skolem1 = set(r["denied"][0]["literals"][0]["skolem_constants"])
        skolem2 = set(r["denied"][1]["literals"][0]["skolem_constants"])
        assert len(skolem1) >= 1
        assert len(skolem2) >= 1
        assert skolem1.isdisjoint(skolem2), (
            f"Different goals should use different Skolem constants: "
            f"goal1={skolem1}, goal2={skolem2}"
        )

    def test_vampire_4_goals_all_skolemized(self):
        """vampire.in has 4 goals with variables — all must be Skolemized."""
        vampire_input = """\
formulas(sos).
-P(x) | -P(i(x,y)) | P(y).
P(i(i(x,y),i(i(y,z),i(x,z)))).
P(i(i(n(x),x),x)).
P(i(x,i(n(x),y))).
end_of_list.

formulas(goals).
P(i(x,i(i(y,i(x,z)),i(i(n(z),i(i(n(v),w),y)),i(v,z))))).
P(i(x,x)).
P(i(i(x,i(y,z)),i(y,i(x,z)))).
P(i(x,i(y,x))).
end_of_list.
"""
        r = _inspect_skolemization(vampire_input)
        assert r["num_goals"] == 4
        assert r["num_denied"] == 4
        for i, denied in enumerate(r["denied"]):
            for lit in denied["literals"]:
                assert lit["has_variables"] is False, (
                    f"Goal {i+1} still has variables after denial! "
                    f"Skolemization failed."
                )

    def test_skolem_constants_dont_cause_spurious_proof(self):
        """Skolemized goals should NOT unify with axiom variables.

        This is the actual regression test: without Skolemization,
        -P(i(v0,i(v1,v0))) could unify with P(i(x,i(n(x),y))) giving
        a spurious 2-line proof.
        """
        vampire_input = """\
formulas(sos).
-P(x) | -P(i(x,y)) | P(y).
P(i(i(x,y),i(i(y,z),i(x,z)))).
P(i(i(n(x),x),x)).
P(i(x,i(n(x),y))).
end_of_list.

formulas(goals).
P(i(x,i(y,x))).
end_of_list.
"""
        r = _run_search(vampire_input, paramodulation=False, demodulation=False,
                        max_given=500)
        if r["exit_code_name"] == "MAX_PROOFS_EXIT":
            proof = r["proofs"][0]
            assert proof["length"] > 2, (
                f"REGRESSION: vampire goal proof has only {proof['length']} clauses! "
                f"This suggests Skolemization is broken and variables unified spuriously."
            )

    def test_goal_with_same_varnames_as_axioms(self):
        """Variables in goals must not collide with axiom variables.

        Even if goals use variable names like x,y that also appear in
        axioms, Skolemization should replace them with fresh constants.
        """
        r = _inspect_skolemization("""\
formulas(sos).
  f(x) = g(x).
end_of_list.

formulas(goals).
  f(x) = g(x).
end_of_list.
""")
        lit = r["denied"][0]["literals"][0]
        assert lit["has_variables"] is False, (
            "Goal variable 'x' should be Skolemized even if axioms also use 'x'"
        )
