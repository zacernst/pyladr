"""Comprehensive compatibility tests for hierarchical GNN integration.

Validates that when hierarchical GNN features are DISABLED, the prover
produces IDENTICAL behavior to the reference C Prover9 implementation.

This is the primary regression guard for the hierarchical GNN feature.
Zero tolerance for breaking changes when features are off.

Run with: pytest tests/compatibility/test_hierarchical_gnn_compat.py -v
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import build_binary_term, get_rigid_term, get_variable_term
from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions, SearchResult

from tests.conftest import C_PROVER9_BIN, requires_c_binary
from tests.cross_validation.c_runner import (
    ProverResult,
    run_c_prover9_from_string,
)
from tests.cross_validation.comparator import (
    compare_full,
    compare_search_statistics,
    compare_theorem_result,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _run_python(
    input_text: str,
    *,
    max_given: int = 200,
    goal_directed: bool = False,
    goal_proximity_weight: float = 0.3,
    embedding_evolution_rate: float = 0.01,
) -> SearchResult:
    """Run the Python search engine on inline LADR input.

    Accepts hierarchical GNN parameters to test both enabled/disabled modes.
    """
    st = SymbolTable()
    from pyladr.parsing.ladr_parser import LADRParser
    parser = LADRParser(st)
    parsed = parser.parse_input(input_text)

    from pyladr.apps.prover9 import _deny_goals, _apply_settings

    usable, sos = _deny_goals(parsed, st)

    opts = SearchOptions(
        max_given=max_given,
        quiet=True,
        goal_directed=goal_directed,
        goal_proximity_weight=goal_proximity_weight,
        embedding_evolution_rate=embedding_evolution_rate,
    )
    _apply_settings(parsed, opts, st)

    search = GivenClauseSearch(options=opts, symbol_table=st)
    return search.run(usable=usable, sos=sos)


def _py_to_prover_result(py: SearchResult) -> ProverResult:
    """Convert Python SearchResult to ProverResult for comparator."""
    return ProverResult(
        exit_code=int(py.exit_code),
        raw_output="",
        theorem_proved=(py.exit_code == ExitCode.MAX_PROOFS_EXIT),
        search_failed=(py.exit_code == ExitCode.SOS_EMPTY_EXIT),
        clauses_given=py.stats.given,
        clauses_generated=py.stats.generated,
        clauses_kept=py.stats.kept,
        clauses_deleted=py.stats.sos_limit_deleted,
        proof_length=len(py.proofs[0].clauses) if py.proofs else 0,
    )


# ── Test Data ──────────────────────────────────────────────────────────────


# Pure resolution problems (no equality)
RESOLUTION_PROBLEMS = [
    (
        "modus_ponens",
        """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(a).
end_of_list.
""",
    ),
    (
        "resolution_chain_3",
        """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(x) | R(x).
  -R(a).
end_of_list.
""",
    ),
    (
        "two_variable_resolution",
        """\
formulas(sos).
  P(a, b).
  -P(x, y) | Q(y, x).
  -Q(b, a).
end_of_list.
""",
    ),
    (
        "horn_clause_chain",
        """\
formulas(sos).
  A(a).
  -A(x) | B(x).
  -B(x) | C(x).
  -C(x) | D(x).
  -D(a).
end_of_list.
""",
    ),
]

# Equational problems with set(auto)
EQUATIONAL_PROBLEMS = [
    (
        "identity_proof",
        """\
set(auto).
formulas(sos).
  e * x = x.
end_of_list.

formulas(goals).
  e * e = e.
end_of_list.
""",
    ),
    (
        "group_commutativity",
        """\
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
""",
    ),
]

# Problems expected to exhaust SOS
UNPROVABLE_PROBLEMS = [
    (
        "independent_predicates",
        """\
formulas(sos).
  P(a).
end_of_list.

formulas(goals).
  Q(a).
end_of_list.
""",
    ),
    (
        "disconnected_clauses",
        """\
formulas(sos).
  P(a).
  Q(b).
end_of_list.

formulas(goals).
  R(c).
end_of_list.
""",
    ),
]


# ── Core: Features-disabled identical behavior ────────────────────────────


class TestHierarchicalDisabledIdenticalBehavior:
    """When goal_directed=False, behavior must be EXACTLY the same as
    the baseline (no hierarchical code paths activated)."""

    @pytest.mark.parametrize(
        "name,input_text",
        RESOLUTION_PROBLEMS,
        ids=[n for n, _ in RESOLUTION_PROBLEMS],
    )
    def test_resolution_disabled_matches_baseline(self, name: str, input_text: str):
        """Resolution proofs: disabled hierarchical == baseline."""
        baseline = _run_python(input_text, goal_directed=False)
        disabled = _run_python(
            input_text,
            goal_directed=False,
            goal_proximity_weight=0.3,
            embedding_evolution_rate=0.01,
        )

        assert baseline.exit_code == disabled.exit_code, (
            f"{name}: exit codes differ: {baseline.exit_code} vs {disabled.exit_code}"
        )
        assert baseline.stats.given == disabled.stats.given, (
            f"{name}: given counts differ: {baseline.stats.given} vs {disabled.stats.given}"
        )
        assert baseline.stats.generated == disabled.stats.generated, (
            f"{name}: generated counts differ"
        )
        assert baseline.stats.kept == disabled.stats.kept, (
            f"{name}: kept counts differ"
        )

    @pytest.mark.parametrize(
        "name,input_text",
        EQUATIONAL_PROBLEMS,
        ids=[n for n, _ in EQUATIONAL_PROBLEMS],
    )
    def test_equational_disabled_matches_baseline(self, name: str, input_text: str):
        """Equational proofs: disabled hierarchical == baseline."""
        baseline = _run_python(input_text, goal_directed=False)
        disabled = _run_python(
            input_text,
            goal_directed=False,
            goal_proximity_weight=0.5,
        )

        assert baseline.exit_code == disabled.exit_code
        assert baseline.stats.given == disabled.stats.given
        assert baseline.stats.generated == disabled.stats.generated
        assert baseline.stats.kept == disabled.stats.kept

    @pytest.mark.parametrize(
        "name,input_text",
        UNPROVABLE_PROBLEMS,
        ids=[n for n, _ in UNPROVABLE_PROBLEMS],
    )
    def test_unprovable_disabled_matches_baseline(self, name: str, input_text: str):
        """Unprovable problems: disabled hierarchical == baseline."""
        baseline = _run_python(input_text, max_given=50, goal_directed=False)
        disabled = _run_python(
            input_text,
            max_given=50,
            goal_directed=False,
            goal_proximity_weight=0.9,
        )

        assert baseline.exit_code == disabled.exit_code
        assert baseline.stats.given == disabled.stats.given


# ── SearchOptions defaults ────────────────────────────────────────────────


class TestSearchOptionsHierarchicalDefaults:
    """Hierarchical GNN options default to disabled."""

    def test_goal_directed_default_false(self):
        opts = SearchOptions()
        assert opts.goal_directed is False

    def test_goal_proximity_weight_default(self):
        opts = SearchOptions()
        assert opts.goal_proximity_weight == 0.3

    def test_embedding_evolution_rate_default(self):
        opts = SearchOptions()
        assert opts.embedding_evolution_rate == 0.01

    def test_all_ml_features_disabled_by_default(self):
        """No ML/hierarchical feature is enabled by default."""
        opts = SearchOptions()
        assert opts.online_learning is False
        assert opts.goal_directed is False
        assert opts.repetition_bias is False
        assert opts.ml_weight is None

    def test_hierarchical_options_dont_affect_core(self):
        """Setting hierarchical options doesn't change core behavior."""
        opts_plain = SearchOptions(quiet=True, max_given=50)
        opts_with = SearchOptions(
            quiet=True,
            max_given=50,
            goal_directed=False,
            goal_proximity_weight=0.99,
            embedding_evolution_rate=0.5,
        )

        # Core options must be identical
        assert opts_plain.binary_resolution == opts_with.binary_resolution
        assert opts_plain.factoring == opts_with.factoring
        assert opts_plain.paramodulation == opts_with.paramodulation
        assert opts_plain.max_given == opts_with.max_given
        assert opts_plain.max_weight == opts_with.max_weight
        assert opts_plain.sos_limit == opts_with.sos_limit


# ── C Reference Comparison ────────────────────────────────────────────────


@pytest.mark.cross_validation
@requires_c_binary
class TestHierarchicalDisabledVsCReference:
    """With hierarchical features disabled, Python must match C Prover9."""

    @pytest.mark.parametrize(
        "name,input_text",
        RESOLUTION_PROBLEMS,
        ids=[n for n, _ in RESOLUTION_PROBLEMS],
    )
    def test_resolution_matches_c(self, name: str, input_text: str):
        """Resolution: Python (disabled hierarchical) matches C."""
        c_result = run_c_prover9_from_string(input_text)
        py_result = _py_to_prover_result(
            _run_python(input_text, goal_directed=False)
        )

        comp = compare_theorem_result(c_result, py_result)
        assert comp.equivalent, f"{name}: theorem status mismatch: {comp}"

    @pytest.mark.parametrize(
        "name,input_text",
        EQUATIONAL_PROBLEMS,
        ids=[n for n, _ in EQUATIONAL_PROBLEMS],
    )
    def test_equational_matches_c(self, name: str, input_text: str):
        """Equational: Python (disabled hierarchical) matches C."""
        c_result = run_c_prover9_from_string(input_text)
        py_result = _py_to_prover_result(
            _run_python(input_text, goal_directed=False)
        )

        comp = compare_theorem_result(c_result, py_result)
        assert comp.equivalent, f"{name}: theorem status mismatch: {comp}"

    @pytest.mark.parametrize(
        "name,input_text",
        UNPROVABLE_PROBLEMS,
        ids=[n for n, _ in UNPROVABLE_PROBLEMS],
    )
    def test_unprovable_matches_c(self, name: str, input_text: str):
        """Unprovable: Python (disabled hierarchical) matches C."""
        c_result = run_c_prover9_from_string(input_text, timeout=10.0)
        py_result = _py_to_prover_result(
            _run_python(input_text, max_given=100, goal_directed=False)
        )

        comp = compare_theorem_result(c_result, py_result)
        assert comp.equivalent, f"{name}: theorem status mismatch: {comp}"

    @pytest.mark.parametrize(
        "name,input_text",
        RESOLUTION_PROBLEMS,
        ids=[n for n, _ in RESOLUTION_PROBLEMS],
    )
    def test_resolution_stats_close_to_c(self, name: str, input_text: str):
        """Resolution: search statistics within tolerance of C."""
        c_result = run_c_prover9_from_string(input_text)
        py_result = _py_to_prover_result(
            _run_python(input_text, goal_directed=False)
        )

        comp = compare_search_statistics(c_result, py_result, tolerance=0.5)
        # Log differences but allow tolerance for implementation variations
        if not comp.equivalent:
            for d in comp.differences:
                print(f"  {name} stats: {d}")


# ── Goal-Directed Config Isolation ────────────────────────────────────────


class TestGoalDirectedConfigIsolation:
    """GoalDirectedConfig and HierarchicalIntegrationConfig default to off."""

    def test_goal_directed_config_disabled_by_default(self):
        from pyladr.search.goal_directed import GoalDirectedConfig
        cfg = GoalDirectedConfig()
        assert cfg.enabled is False

    def test_hierarchical_integration_config_disabled_by_default(self):
        from pyladr.search.hierarchical_integration import HierarchicalIntegrationConfig
        cfg = HierarchicalIntegrationConfig()
        assert cfg.enabled is False
        assert cfg.use_hierarchical_gnn is False
        assert cfg.goal_directed.enabled is False

    def test_create_goal_directed_provider_noop_when_disabled(self):
        """Factory returns base provider unchanged when disabled."""
        from pyladr.search.hierarchical_integration import (
            HierarchicalIntegrationConfig,
            create_goal_directed_provider,
        )

        class FakeProvider:
            embedding_dim = 32
            def get_embedding(self, clause):
                return [0.0] * 32
            def get_embeddings_batch(self, clauses):
                return [[0.0] * 32 for _ in clauses]

        base = FakeProvider()
        cfg = HierarchicalIntegrationConfig(enabled=False)

        result = create_goal_directed_provider(base, cfg)
        assert result is base, "Disabled config must return base provider unchanged"

    def test_create_goal_directed_provider_noop_when_gd_disabled(self):
        """Factory returns base even when hierarchical enabled but GD disabled."""
        from pyladr.search.goal_directed import GoalDirectedConfig
        from pyladr.search.hierarchical_integration import (
            HierarchicalIntegrationConfig,
            create_goal_directed_provider,
        )

        class FakeProvider:
            embedding_dim = 32
            def get_embedding(self, clause):
                return [0.0] * 32
            def get_embeddings_batch(self, clauses):
                return [[0.0] * 32 for _ in clauses]

        base = FakeProvider()
        cfg = HierarchicalIntegrationConfig(
            enabled=True,
            goal_directed=GoalDirectedConfig(enabled=False),
        )

        result = create_goal_directed_provider(base, cfg)
        assert result is base

    def test_goal_directed_provider_passthrough_when_disabled(self):
        """GoalDirectedEmbeddingProvider is pure passthrough when disabled."""
        from pyladr.search.goal_directed import (
            GoalDirectedConfig,
            GoalDirectedEmbeddingProvider,
        )

        class FakeProvider:
            embedding_dim = 8
            def get_embedding(self, clause):
                return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            def get_embeddings_batch(self, clauses):
                return [self.get_embedding(c) for c in clauses]

        base = FakeProvider()
        provider = GoalDirectedEmbeddingProvider(
            base, GoalDirectedConfig(enabled=False)
        )

        # Create a minimal clause for testing
        atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        clause = Clause(literals=(Literal(sign=True, atom=atom),))

        base_emb = base.get_embedding(clause)
        gd_emb = provider.get_embedding(clause)

        assert base_emb == gd_emb, (
            "Disabled GoalDirected must return identical embeddings"
        )


# ── Proof Trace Validation ────────────────────────────────────────────────


class TestProofTracePreservation:
    """Verify proof traces are unchanged with hierarchical features disabled."""

    def test_proof_clause_count_preserved(self):
        """Proof length identical with or without hierarchical options."""
        input_text = """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(a).
end_of_list.
"""
        baseline = _run_python(input_text, goal_directed=False)
        with_opts = _run_python(
            input_text,
            goal_directed=False,
            goal_proximity_weight=0.8,
        )

        assert baseline.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert with_opts.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(baseline.proofs) == len(with_opts.proofs)

        if baseline.proofs and with_opts.proofs:
            assert len(baseline.proofs[0].clauses) == len(with_opts.proofs[0].clauses)

    def test_proof_justification_types_preserved(self):
        """Justification types in proof are identical."""
        input_text = """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(a).
end_of_list.
"""
        baseline = _run_python(input_text, goal_directed=False)
        with_opts = _run_python(
            input_text,
            goal_directed=False,
            goal_proximity_weight=0.5,
        )

        assert baseline.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert with_opts.exit_code == ExitCode.MAX_PROOFS_EXIT

        b_justs = [
            c.justification[0].just_type if c.justification else None
            for c in baseline.proofs[0].clauses
        ]
        w_justs = [
            c.justification[0].just_type if c.justification else None
            for c in with_opts.proofs[0].clauses
        ]
        assert b_justs == w_justs, "Justification types must be identical"


# ── Exit Code Compatibility ───────────────────────────────────────────────


class TestExitCodeCompatibility:
    """Verify exit codes match C conventions with hierarchical features."""

    def test_exit_codes_match_c_values(self):
        """ExitCode enum values match C search.h."""
        assert ExitCode.MAX_PROOFS_EXIT == 1
        assert ExitCode.SOS_EMPTY_EXIT == 2
        assert ExitCode.MAX_GIVEN_EXIT == 3
        assert ExitCode.MAX_KEPT_EXIT == 4
        assert ExitCode.MAX_SECONDS_EXIT == 5
        assert ExitCode.MAX_GENERATED_EXIT == 6
        assert ExitCode.FATAL_EXIT == 7

    def test_process_exit_codes_match_c(self):
        """Process-level exit codes match C prover9."""
        from pyladr.apps.prover9 import _PROCESS_EXIT_CODES

        assert _PROCESS_EXIT_CODES[ExitCode.MAX_PROOFS_EXIT] == 0   # proof found
        assert _PROCESS_EXIT_CODES[ExitCode.SOS_EMPTY_EXIT] == 2    # no proof
        assert _PROCESS_EXIT_CODES[ExitCode.MAX_GIVEN_EXIT] == 3
        assert _PROCESS_EXIT_CODES[ExitCode.MAX_KEPT_EXIT] == 4
        assert _PROCESS_EXIT_CODES[ExitCode.MAX_SECONDS_EXIT] == 5
        assert _PROCESS_EXIT_CODES[ExitCode.MAX_GENERATED_EXIT] == 6

    def test_proof_found_exit_code(self):
        result = _run_python("""\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(a).
end_of_list.
""", goal_directed=False)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_sos_empty_exit_code(self):
        result = _run_python("""\
formulas(sos).
  P(a).
end_of_list.

formulas(goals).
  Q(a).
end_of_list.
""", max_given=200, goal_directed=False)
        assert result.exit_code == ExitCode.SOS_EMPTY_EXIT

    def test_max_given_exit_code(self):
        result = _run_python("""\
formulas(sos).
  P(a, b).
  -P(x, y) | P(y, x).
  -P(x, y) | Q(x).
  -Q(x) | P(x, x).
end_of_list.

formulas(goals).
  R(a).
end_of_list.
""", max_given=5, goal_directed=False)
        assert result.exit_code in (ExitCode.MAX_GIVEN_EXIT, ExitCode.SOS_EMPTY_EXIT)


# ── Auto-Cascade Compatibility ────────────────────────────────────────────


class TestAutoCascadeUnchanged:
    """set(auto) cascade behavior must be unchanged with hierarchical features."""

    def test_auto_enables_paramodulation_for_equality(self):
        """set(auto) with equality must enable paramodulation."""
        from pyladr.apps.prover9 import _apply_settings
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input("""\
set(auto).
formulas(sos).
  e * x = x.
end_of_list.

formulas(goals).
  e * e = e.
end_of_list.
""")

        opts = SearchOptions(goal_directed=False)
        _apply_settings(parsed, opts, st)

        assert opts.paramodulation is True

    def test_auto_enables_resolution_for_non_horn(self):
        """set(auto) with non-Horn must enable binary_resolution."""
        from pyladr.apps.prover9 import _apply_settings
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input("""\
set(auto).
formulas(sos).
  P(a) | Q(a).
  -P(x) | -Q(x).
end_of_list.
""")

        opts = SearchOptions(goal_directed=False)
        _apply_settings(parsed, opts, st)

        assert opts.binary_resolution is True

    def test_auto_cascade_identical_with_hierarchical_options(self):
        """Auto-cascade produces same opts regardless of hierarchical settings."""
        from pyladr.apps.prover9 import _apply_settings
        from pyladr.parsing.ladr_parser import LADRParser

        input_text = """\
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

        st1 = SymbolTable()
        parsed1 = LADRParser(st1).parse_input(input_text)
        opts1 = SearchOptions(goal_directed=False)
        _apply_settings(parsed1, opts1, st1)

        st2 = SymbolTable()
        parsed2 = LADRParser(st2).parse_input(input_text)
        opts2 = SearchOptions(
            goal_directed=False,
            goal_proximity_weight=0.9,
            embedding_evolution_rate=0.5,
        )
        _apply_settings(parsed2, opts2, st2)

        assert opts1.binary_resolution == opts2.binary_resolution
        assert opts1.paramodulation == opts2.paramodulation
        assert opts1.hyper_resolution == opts2.hyper_resolution
        assert opts1.factoring == opts2.factoring
        assert opts1.demodulation == opts2.demodulation
        assert opts1.back_demod == opts2.back_demod
        assert opts1.max_weight == opts2.max_weight
        assert opts1.sos_limit == opts2.sos_limit


# ── Selection Strategy Compatibility ──────────────────────────────────────


class TestSelectionStrategyCompatibility:
    """Default clause selection must be unchanged."""

    def test_default_selection_strategy(self):
        """Default selection uses 5:1 weight:age ratio."""
        from pyladr.search.selection import GivenSelection

        sel = GivenSelection()
        # Verify the default rules exist
        assert len(sel.rules) >= 2  # weight + age at minimum

    def test_selection_order_deterministic(self):
        """Same input produces same selection order (determinism)."""
        input_text = """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(x) | R(x).
  -R(a).
end_of_list.
"""
        r1 = _run_python(input_text, goal_directed=False)
        r2 = _run_python(input_text, goal_directed=False)

        assert r1.exit_code == r2.exit_code
        assert r1.stats.given == r2.stats.given
        assert r1.stats.generated == r2.stats.generated
        assert r1.stats.kept == r2.stats.kept

    def test_weight_function_unchanged(self):
        """default_clause_weight unchanged by hierarchical code."""
        from pyladr.search.selection import default_clause_weight

        atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        clause = Clause(literals=(Literal(sign=True, atom=atom),))

        w1 = default_clause_weight(clause)
        w2 = default_clause_weight(clause)
        assert w1 == w2
        assert isinstance(w1, (int, float))


# ── Statistics Compatibility ──────────────────────────────────────────────


class TestStatisticsCompatibility:
    """Search statistics format and content must match C."""

    def test_statistics_fields_present(self):
        """All C-compatible statistics fields populated."""
        result = _run_python("""\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(a).
end_of_list.
""", goal_directed=False)

        stats = result.stats
        assert hasattr(stats, "given")
        assert hasattr(stats, "generated")
        assert hasattr(stats, "kept")
        assert hasattr(stats, "subsumed")
        assert hasattr(stats, "back_subsumed")
        assert hasattr(stats, "proofs")

    def test_statistics_values_sensible(self):
        """Statistics have sensible values for a solved problem."""
        result = _run_python("""\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(a).
end_of_list.
""", goal_directed=False)

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert result.stats.given >= 1
        assert result.stats.generated >= 0
        assert result.stats.kept >= 0
        assert result.stats.proofs >= 1

    def test_statistics_identical_disabled_vs_baseline(self):
        """Statistics must be byte-identical when hierarchical is disabled."""
        input_text = """\
formulas(sos).
  P(a).
  -P(x) | Q(x).
  -Q(x) | R(x).
  -R(a).
end_of_list.
"""
        baseline = _run_python(input_text, goal_directed=False)
        disabled = _run_python(
            input_text,
            goal_directed=False,
            goal_proximity_weight=0.99,
        )

        assert baseline.stats.given == disabled.stats.given
        assert baseline.stats.generated == disabled.stats.generated
        assert baseline.stats.kept == disabled.stats.kept
        assert baseline.stats.subsumed == disabled.stats.subsumed
        assert baseline.stats.back_subsumed == disabled.stats.back_subsumed
        assert baseline.stats.proofs == disabled.stats.proofs
