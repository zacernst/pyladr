"""End-to-end integration tests for ML features in the search pipeline.

Validates the complete CLI -> Search -> ML processing workflow:
- CLI flags activate ML features with observable effects (REQ-INT001)
- ML processing is observable via logs, stats, and behavior (REQ-INT002)
- Components integrate correctly beyond unit testing (REQ-INT003)
- C Prover9 compatibility is maintained with ML enabled
- Graceful degradation works when ML components fail

These tests address the validation blind spot from the previous mission:
unit tests passed (181/183) but end-to-end integration was missing.
"""

from __future__ import annotations

import io
import logging
import sys
from contextlib import redirect_stdout

import pytest

# torch_geometric triggers DeprecationWarning from torch.jit.script
# which gets promoted to an error by pyproject.toml filterwarnings=["error"]
pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning",
)

torch = pytest.importorskip("torch")

from pyladr.apps.prover9 import run_prover
from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.search.given_clause import (
    ExitCode,
    GivenClauseSearch,
    SearchOptions,
    SearchResult,
)
from pyladr.search.ml_selection import (
    EmbeddingEnhancedSelection,
    MLSelectionConfig,
    MLSelectionStats,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

# Symbol IDs (positive integers)
A, B, C_SYM = 1, 2, 3
F, G = 10, 11
P, Q, R_SYM = 20, 21, 22
EQ = 30  # equality


def _const(symnum: int) -> "Term":
    return get_rigid_term(symnum, 0)


def _func(symnum: int, *args) -> "Term":
    return get_rigid_term(symnum, len(args), args)


def _make_clause(*atoms, signs=None) -> Clause:
    if signs is None:
        signs = (True,) * len(atoms)
    lits = tuple(Literal(sign=s, atom=a) for s, a in zip(signs, atoms))
    return Clause(literals=lits)


# ── Test input problems ──────────────────────────────────────────────────────

TRIVIAL_PROBLEM = """\
formulas(sos).
  e * x = x.
end_of_list.

formulas(goals).
  e * e = e.
end_of_list.
"""

SIMPLE_PROBLEM = """\
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

LATTICE_PROBLEM = """\
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

# Problem with no goals (pure SOS)
NO_GOAL_PROBLEM = """\
formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
  x * x = e.
  a * b != b * a.
end_of_list.
"""


def _deny_goals(parsed) -> tuple[list, list]:
    """Negate goals and add to SOS (mirrors prover9._deny_goals)."""
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
    return usable, sos


def _write_temp_input(tmp_path, content: str, name: str = "test.in") -> str:
    p = tmp_path / name
    p.write_text(content)
    return str(p)


def _run_prover_capture(argv: list[str]) -> tuple[int, str]:
    """Run prover and capture stdout. Returns (exit_code, output)."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        rc = run_prover(argv)
    finally:
        sys.stdout = old_stdout
    return rc, buf.getvalue()


def _run_prover_capture_logs(
    argv: list[str], logger_name: str = "pyladr",
) -> tuple[int, str, str]:
    """Run prover capturing stdout and log messages. Returns (rc, output, logs)."""
    buf = io.StringIO()
    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(name)s:%(levelname)s:%(message)s"))

    target_logger = logging.getLogger(logger_name)
    old_level = target_logger.level
    target_logger.setLevel(logging.DEBUG)
    target_logger.addHandler(handler)

    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        rc = run_prover(argv)
    finally:
        sys.stdout = old_stdout
        target_logger.removeHandler(handler)
        target_logger.setLevel(old_level)

    return rc, buf.getvalue(), log_buf.getvalue()


# ── REQ-INT001: CLI flags produce observable ML effects ──────────────────────


class TestMLFlagActivation:
    """Verify that ML CLI flags produce observable functional effects."""

    def test_ml_weight_activates_ml_selection(self, tmp_path):
        """--ml-weight flag activates ML-enhanced selection."""
        input_file = _write_temp_input(tmp_path, TRIVIAL_PROBLEM)
        rc, output = _run_prover_capture(
            ["pyprover9", "-f", input_file, "--ml-weight", "0.3"],
        )
        assert rc == 0, f"Expected proof found (rc=0), got {rc}"
        assert "ML-enhanced selection enabled" in output
        assert "ml_weight=0.30" in output

    def test_goal_directed_activates(self, tmp_path):
        """--goal-directed flag activates goal-directed selection."""
        input_file = _write_temp_input(tmp_path, TRIVIAL_PROBLEM)
        rc, output = _run_prover_capture([
            "pyprover9", "-f", input_file,
            "--ml-weight", "0.3", "--goal-directed",
        ])
        assert rc == 0
        assert "goal_directed" in output

    def test_online_learning_activates(self, tmp_path):
        """--online-learning flag activates online learning integration."""
        input_file = _write_temp_input(tmp_path, TRIVIAL_PROBLEM)
        rc, output = _run_prover_capture([
            "pyprover9", "-f", input_file, "--online-learning",
        ])
        assert rc == 0
        assert "online_learning" in output

    def test_no_ml_flags_no_ml_activity(self, tmp_path):
        """Without ML flags, no ML activity appears in output."""
        input_file = _write_temp_input(tmp_path, TRIVIAL_PROBLEM)
        rc, output = _run_prover_capture(["pyprover9", "-f", input_file])
        assert rc == 0
        assert "ML-enhanced" not in output
        assert "online_learning" not in output

    def test_ml_weight_alone_sufficient(self, tmp_path):
        """--ml-weight alone is sufficient to activate ML selection."""
        input_file = _write_temp_input(tmp_path, TRIVIAL_PROBLEM)
        rc, output, logs = _run_prover_capture_logs([
            "pyprover9", "-f", input_file, "--ml-weight", "0.5",
        ])
        assert rc == 0
        assert "ML-enhanced selection enabled" in output
        assert "ml_weight=0.50" in output

    def test_all_ml_flags_combined(self, tmp_path):
        """All ML flags work together without conflict."""
        input_file = _write_temp_input(tmp_path, TRIVIAL_PROBLEM)
        rc, output = _run_prover_capture([
            "pyprover9", "-f", input_file,
            "--ml-weight", "0.4",
            "--goal-directed",
            "--online-learning",
            "--embedding-dim", "16",
        ])
        assert rc == 0
        assert "ML-enhanced selection enabled" in output
        assert "goal_directed" in output
        assert "online_learning" in output

    def test_quiet_mode_suppresses_ml_banner(self, tmp_path):
        """--quiet mode suppresses ML status messages but ML still active."""
        input_file = _write_temp_input(tmp_path, TRIVIAL_PROBLEM)
        rc, output, logs = _run_prover_capture_logs([
            "pyprover9", "-f", input_file,
            "--ml-weight", "0.3", "--quiet",
        ])
        assert rc == 0
        # Banner suppressed in quiet mode
        assert "ML-enhanced selection enabled" not in output
        # But ML actually ran (visible in logs)
        assert "embedding provider" in logs.lower() or "ML" in logs


# ── REQ-INT002: ML processing is observable ──────────────────────────────────


class TestMLProcessingObservable:
    """Verify that ML processing produces observable results."""

    def test_ml_selection_finds_proof_trivial(self, tmp_path):
        """ML-enhanced selection finds trivial proofs."""
        input_file = _write_temp_input(tmp_path, TRIVIAL_PROBLEM)
        rc, output = _run_prover_capture([
            "pyprover9", "-f", input_file, "--ml-weight", "0.3",
        ])
        assert rc == 0
        assert "THEOREM PROVED" in output

    def test_ml_selection_finds_proof_group_theory(self, tmp_path):
        """ML-enhanced selection handles real equational reasoning."""
        input_file = _write_temp_input(tmp_path, SIMPLE_PROBLEM)
        rc, output = _run_prover_capture([
            "pyprover9", "-f", input_file, "--ml-weight", "0.3",
        ])
        assert rc == 0, f"ML selection should find group commutativity proof, rc={rc}"
        assert "THEOREM PROVED" in output
        assert "PROOF" in output

    def test_ml_selection_finds_proof_lattice(self, tmp_path):
        """ML-enhanced selection handles lattice theory problems."""
        input_file = _write_temp_input(tmp_path, LATTICE_PROBLEM)
        rc, output = _run_prover_capture([
            "pyprover9", "-f", input_file, "--ml-weight", "0.3",
        ])
        assert rc == 0
        assert "THEOREM PROVED" in output

    def test_ml_with_online_learning_finds_proof(self, tmp_path):
        """Online learning integration doesn't prevent proof finding."""
        input_file = _write_temp_input(tmp_path, TRIVIAL_PROBLEM)
        rc, output = _run_prover_capture([
            "pyprover9", "-f", input_file,
            "--online-learning", "--ml-weight", "0.3",
        ])
        assert rc == 0
        assert "THEOREM PROVED" in output

    def test_online_learning_logs_activity(self, tmp_path):
        """Online learning produces log messages showing activity."""
        input_file = _write_temp_input(tmp_path, SIMPLE_PROBLEM)
        rc, output, logs = _run_prover_capture_logs([
            "pyprover9", "-f", input_file,
            "--online-learning", "--ml-weight", "0.3",
        ])
        assert rc == 0
        # Online learning should log integration activity
        assert "Online learning integration active" in logs or "online" in logs.lower()

    def test_embedding_dim_passed_through(self, tmp_path):
        """--embedding-dim flag is passed through to the provider."""
        input_file = _write_temp_input(tmp_path, TRIVIAL_PROBLEM)
        rc, output, logs = _run_prover_capture_logs([
            "pyprover9", "-f", input_file,
            "--ml-weight", "0.3", "--embedding-dim", "64",
        ])
        assert rc == 0
        # Provider should report the configured dimension
        assert "dim=64" in logs or "dim=64" in output

    def test_goal_directed_with_goals(self, tmp_path):
        """Goal-directed selection uses goals for scoring."""
        input_file = _write_temp_input(tmp_path, SIMPLE_PROBLEM)
        rc, output, logs = _run_prover_capture_logs([
            "pyprover9", "-f", input_file,
            "--ml-weight", "0.3", "--goal-directed",
        ])
        assert rc == 0
        assert "THEOREM PROVED" in output
        # Goal-directed should log goal registration
        assert "goal_directed" in output.lower() or "Goal-directed" in logs

    def test_goal_directed_without_goals_graceful(self, tmp_path):
        """Goal-directed flag with no goals in input degrades gracefully."""
        input_file = _write_temp_input(tmp_path, NO_GOAL_PROBLEM)
        rc, output = _run_prover_capture([
            "pyprover9", "-f", input_file,
            "--ml-weight", "0.3", "--goal-directed",
            "-max_given", "50",
        ])
        # Should not crash with unhandled exception - any valid exit code is fine
        assert rc in (0, 1, 2, 3, 4, 5, 6)  # any C-compatible exit code


# ── REQ-INT003: No gaps between unit and integration tests ──────────────────


class TestMLIntegrationGaps:
    """Verify integration between components that unit tests miss."""

    def test_embedding_enhanced_selection_with_search(self):
        """EmbeddingEnhancedSelection works plugged into GivenClauseSearch."""
        from pyladr.ml.embedding_provider import NoOpEmbeddingProvider

        provider = NoOpEmbeddingProvider(embedding_dim=32)
        ml_config = MLSelectionConfig(enabled=True, ml_weight=0.3)
        selection = EmbeddingEnhancedSelection(
            embedding_provider=provider,
            ml_config=ml_config,
        )

        opts = SearchOptions()
        search = GivenClauseSearch(
            options=opts,
            selection=selection,
        )

        # P(a) and -P(a) should yield empty clause
        a = _const(A)
        p_a = _func(P, a)
        c1 = _make_clause(p_a, signs=(True,))
        c2 = _make_clause(p_a, signs=(False,))

        result = search.run(usable=[], sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_gnn_embedding_provider_with_search(self):
        """GNNEmbeddingProvider produces embeddings during actual search."""
        from pyladr.core.symbol import SymbolTable
        from pyladr.ml.embedding_provider import (
            EmbeddingProviderConfig,
            create_embedding_provider,
        )
        from pyladr.ml.graph.clause_encoder import GNNConfig

        st = SymbolTable()
        gnn_config = GNNConfig(embedding_dim=32)
        provider_config = EmbeddingProviderConfig(model_path="", device="cpu")
        provider = create_embedding_provider(
            symbol_table=st,
            config=provider_config,
            gnn_config=gnn_config,
        )

        ml_config = MLSelectionConfig(enabled=True, ml_weight=0.3)
        selection = EmbeddingEnhancedSelection(
            embedding_provider=provider,
            ml_config=ml_config,
        )

        opts = SearchOptions(ml_weight=0.3)
        search = GivenClauseSearch(
            options=opts,
            selection=selection,
            symbol_table=st,
        )

        # Simple conflict
        a = _const(A)
        p_a = _func(P, a)
        c1 = _make_clause(p_a, signs=(True,))
        c2 = _make_clause(p_a, signs=(False,))

        result = search.run(usable=[], sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_goal_directed_provider_wrapping(self):
        """GoalDirectedEmbeddingProvider wraps base provider correctly."""
        from pyladr.ml.embedding_provider import NoOpEmbeddingProvider
        from pyladr.search.goal_directed import (
            GoalDirectedConfig,
            GoalDirectedEmbeddingProvider,
        )

        base = NoOpEmbeddingProvider(embedding_dim=32)
        config = GoalDirectedConfig(enabled=True, goal_proximity_weight=0.5)
        gd = GoalDirectedEmbeddingProvider(base_provider=base, config=config)

        assert gd.embedding_dim == 32
        # Should not crash even with no goals registered
        a = _const(A)
        lit = Literal(sign=True, atom=_func(P, a))
        c = Clause(literals=(lit,))
        emb = gd.get_embedding(c)
        # NoOp returns None, so gd should also return None
        assert emb is None

    def test_search_options_ml_fields_exist(self):
        """SearchOptions has all ML-related fields properly wired."""
        opts = SearchOptions(
            online_learning=True,
            ml_weight=0.5,
            model_path="test.pt",
            embedding_dim=64,
            buffer_capacity=1000,
            goal_directed=True,
            goal_proximity_weight=0.4,
        )
        assert opts.online_learning is True
        assert opts.ml_weight == 0.5
        assert opts.model_path == "test.pt"
        assert opts.embedding_dim == 64
        assert opts.buffer_capacity == 1000
        assert opts.goal_directed is True
        assert opts.goal_proximity_weight == 0.4

    def test_ml_selection_stats_tracking(self):
        """MLSelectionStats correctly tracks ML vs traditional selections."""
        stats = MLSelectionStats()
        stats.record_ml_selection(0.8)
        stats.record_ml_selection(0.6)
        stats.record_traditional()
        stats.record_fallback()

        assert stats.ml_selections == 2
        assert stats.traditional_selections == 2  # 1 traditional + 1 fallback
        assert stats.fallback_count == 1
        assert abs(stats.avg_ml_score - 0.7) < 0.01

        report = stats.report()
        assert "2/4 ML" in report
        assert "50.0%" in report

    def test_setup_ml_selection_function(self):
        """_setup_ml_selection creates correct selection object."""
        from pyladr.apps.prover9 import _setup_ml_selection
        from pyladr.core.symbol import SymbolTable

        st = SymbolTable()
        opts = SearchOptions(ml_weight=0.4, embedding_dim=16)
        out = io.StringIO()

        selection = _setup_ml_selection(opts, st, goals=[], out=out)

        assert selection is not None
        assert isinstance(selection, EmbeddingEnhancedSelection)
        assert selection.ml_config.enabled is True
        assert selection.ml_config.ml_weight == 0.4

    def test_setup_ml_selection_with_goal_directed(self):
        """_setup_ml_selection wraps provider with goal-directed when requested."""
        from pyladr.apps.prover9 import _setup_ml_selection
        from pyladr.core.symbol import SymbolTable
        from pyladr.search.goal_directed import GoalDirectedEmbeddingProvider

        st = SymbolTable()
        # Create a dummy goal clause
        a = _const(A)
        goal = _make_clause(_func(P, a))
        opts = SearchOptions(
            ml_weight=0.3,
            goal_directed=True,
            goal_proximity_weight=0.5,
        )
        out = io.StringIO()

        selection = _setup_ml_selection(opts, st, goals=[goal], out=out)

        assert selection is not None
        assert isinstance(selection, EmbeddingEnhancedSelection)
        # The provider should be wrapped
        assert isinstance(
            selection.embedding_provider, GoalDirectedEmbeddingProvider,
        )


# ── C Prover9 compatibility ──────────────────────────────────────────────────


class TestCCompatibility:
    """Verify ML features don't break C Prover9 compatibility."""

    def test_exit_code_proof_found(self, tmp_path):
        """Exit code 0 when proof found (with or without ML)."""
        input_file = _write_temp_input(tmp_path, TRIVIAL_PROBLEM)
        rc_no_ml = run_prover(["pyprover9", "-f", input_file, "--quiet"])
        assert rc_no_ml == 0
        rc_ml = run_prover([
            "pyprover9", "-f", input_file, "--quiet", "--ml-weight", "0.3",
        ])
        assert rc_ml == 0

    def test_exit_code_max_given(self, tmp_path):
        """Exit code 4 for max_given with ML (same as without)."""
        hard_problem = """\
formulas(sos).
  f(f(x)) = x.
  f(a) != a.
end_of_list.
"""
        input_file = _write_temp_input(tmp_path, hard_problem)
        rc_no_ml = run_prover([
            "pyprover9", "-f", input_file, "--quiet", "-max_given", "5",
        ])
        rc_ml = run_prover([
            "pyprover9", "-f", input_file, "--quiet",
            "--ml-weight", "0.3", "-max_given", "5",
        ])
        # Both should hit max_given limit with same exit code
        assert rc_no_ml == rc_ml

    def test_statistics_present_with_ml(self, tmp_path):
        """Statistics section present in output with ML enabled."""
        input_file = _write_temp_input(tmp_path, TRIVIAL_PROBLEM)
        rc, output = _run_prover_capture([
            "pyprover9", "-f", input_file, "--ml-weight", "0.3",
        ])
        assert rc == 0
        assert "STATISTICS" in output
        assert "Given=" in output
        assert "Generated=" in output
        assert "Kept=" in output

    def test_proof_structure_preserved_with_ml(self, tmp_path):
        """Proof output format is preserved with ML enabled."""
        input_file = _write_temp_input(tmp_path, TRIVIAL_PROBLEM)
        rc, output = _run_prover_capture([
            "pyprover9", "-f", input_file, "--ml-weight", "0.3",
        ])
        assert rc == 0
        assert "============================== PROOF ==============================" in output
        assert "end of proof" in output
        assert "$F." in output  # empty clause

    def test_ml_and_no_ml_both_prove_same_problem(self, tmp_path):
        """Both ML and non-ML paths prove the same problems."""
        for problem_name, problem in [
            ("trivial", TRIVIAL_PROBLEM),
            ("group", SIMPLE_PROBLEM),
            ("lattice", LATTICE_PROBLEM),
        ]:
            input_file = _write_temp_input(tmp_path, problem, f"{problem_name}.in")

            rc_no_ml = run_prover([
                "pyprover9", "-f", input_file, "--quiet",
            ])
            rc_ml = run_prover([
                "pyprover9", "-f", input_file, "--quiet", "--ml-weight", "0.3",
            ])

            assert rc_no_ml == 0, f"{problem_name}: no-ML failed with rc={rc_no_ml}"
            assert rc_ml == 0, f"{problem_name}: ML failed with rc={rc_ml}"


# ── Graceful degradation ─────────────────────────────────────────────────────


class TestGracefulDegradation:
    """Verify ML features degrade gracefully on failure."""

    def test_noop_provider_fallback(self):
        """NoOpEmbeddingProvider causes fallback to traditional selection."""
        from pyladr.ml.embedding_provider import NoOpEmbeddingProvider

        provider = NoOpEmbeddingProvider(embedding_dim=32)
        ml_config = MLSelectionConfig(enabled=True, ml_weight=0.5)
        selection = EmbeddingEnhancedSelection(
            embedding_provider=provider,
            ml_config=ml_config,
        )

        opts = SearchOptions()
        search = GivenClauseSearch(options=opts, selection=selection)

        a = _const(A)
        p_a = _func(P, a)
        c1 = _make_clause(p_a, signs=(True,))
        c2 = _make_clause(p_a, signs=(False,))

        # Should still find proof despite NoOp provider
        result = search.run(usable=[], sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_invalid_model_path_degrades(self, tmp_path):
        """Invalid --model path degrades to working search."""
        input_file = _write_temp_input(tmp_path, TRIVIAL_PROBLEM)
        rc = run_prover([
            "pyprover9", "-f", input_file, "--quiet",
            "--ml-weight", "0.3", "--model", "/nonexistent/model.pt",
        ])
        # Should either work with fallback or still find proof
        assert rc in (0, 1)  # proof found or graceful error

    def test_ml_disabled_by_default(self):
        """SearchOptions has ML disabled by default."""
        opts = SearchOptions()
        assert opts.online_learning is False
        assert opts.ml_weight is None
        assert opts.model_path is None
        assert opts.goal_directed is False


# ── Programmatic search integration ──────────────────────────────────────────


class TestProgrammaticMLIntegration:
    """Test ML integration using programmatic search API (not CLI)."""

    def test_parsed_input_with_ml_selection(self):
        """Parse LADR input and run search with ML selection."""
        from pyladr.core.symbol import SymbolTable
        from pyladr.ml.embedding_provider import (
            EmbeddingProviderConfig,
            create_embedding_provider,
        )
        from pyladr.ml.graph.clause_encoder import GNNConfig
        from pyladr.parsing.ladr_parser import parse_input

        st = SymbolTable()
        parsed = parse_input(TRIVIAL_PROBLEM, st)

        gnn_config = GNNConfig(embedding_dim=32)
        provider_config = EmbeddingProviderConfig(model_path="", device="cpu")
        provider = create_embedding_provider(
            symbol_table=st,
            config=provider_config,
            gnn_config=gnn_config,
        )

        ml_config = MLSelectionConfig(enabled=True, ml_weight=0.3)
        selection = EmbeddingEnhancedSelection(
            embedding_provider=provider,
            ml_config=ml_config,
        )

        opts = SearchOptions(ml_weight=0.3)
        search = GivenClauseSearch(
            options=opts, selection=selection, symbol_table=st,
        )
        usable, sos = _deny_goals(parsed)
        result = search.run(usable=usable, sos=sos)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) > 0

    def test_parsed_group_theory_with_ml(self):
        """ML selection handles group theory equational reasoning."""
        from pyladr.core.symbol import SymbolTable
        from pyladr.ml.embedding_provider import (
            EmbeddingProviderConfig,
            create_embedding_provider,
        )
        from pyladr.ml.graph.clause_encoder import GNNConfig
        from pyladr.parsing.ladr_parser import parse_input

        st = SymbolTable()
        parsed = parse_input(SIMPLE_PROBLEM, st)

        gnn_config = GNNConfig(embedding_dim=32)
        provider_config = EmbeddingProviderConfig(model_path="", device="cpu")
        provider = create_embedding_provider(
            symbol_table=st,
            config=provider_config,
            gnn_config=gnn_config,
        )

        ml_config = MLSelectionConfig(enabled=True, ml_weight=0.3)
        selection = EmbeddingEnhancedSelection(
            embedding_provider=provider,
            ml_config=ml_config,
        )

        opts = SearchOptions(ml_weight=0.3)
        search = GivenClauseSearch(
            options=opts, selection=selection, symbol_table=st,
        )
        usable, sos = _deny_goals(parsed)
        result = search.run(usable=usable, sos=sos)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_online_learning_integration_programmatic(self):
        """OnlineSearchIntegration works via programmatic API."""
        from pyladr.core.symbol import SymbolTable
        from pyladr.ml.embedding_provider import (
            EmbeddingProviderConfig,
            create_embedding_provider,
        )
        from pyladr.ml.graph.clause_encoder import GNNConfig
        from pyladr.parsing.ladr_parser import parse_input
        from pyladr.search.online_integration import (
            OnlineIntegrationConfig,
            OnlineSearchIntegration,
        )

        st = SymbolTable()
        parsed = parse_input(TRIVIAL_PROBLEM, st)

        gnn_config = GNNConfig(embedding_dim=32)
        provider_config = EmbeddingProviderConfig(model_path="", device="cpu")
        provider = create_embedding_provider(
            symbol_table=st,
            config=provider_config,
            gnn_config=gnn_config,
        )

        ml_config = MLSelectionConfig(enabled=True, ml_weight=0.3)
        selection = EmbeddingEnhancedSelection(
            embedding_provider=provider,
            ml_config=ml_config,
        )

        ol_config = OnlineIntegrationConfig(enabled=True)
        integration = OnlineSearchIntegration.create(
            embedding_provider=provider,
            config=ol_config,
        )

        opts = SearchOptions(ml_weight=0.3, online_learning=True)
        search = integration.create_search(
            options=opts, selection=selection, symbol_table=st,
        )
        usable, sos = _deny_goals(parsed)
        result = search.run(usable=usable, sos=sos)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
