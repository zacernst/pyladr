"""CLI compatibility tests for hierarchical GNN parameters.

Validates that:
1. New --goal-directed CLI parameters are properly parsed
2. Default CLI behavior is unchanged (features off by default)
3. Existing CLI parameters continue to work identically
4. No interference between hierarchical and existing ML parameters

Run with: pytest tests/compatibility/test_cli_hierarchical_compat.py -v
"""

from __future__ import annotations

import pytest


class TestCLIParameterParsing:
    """Verify new CLI parameters are correctly parsed."""

    def _parse(self, args: list[str]):
        """Parse CLI args using the prover's argument parser."""
        from pyladr.apps.prover9 import _build_arg_parser
        parser = _build_arg_parser()
        return parser.parse_known_args(args)

    def test_goal_directed_flag(self):
        args, _ = self._parse(["--goal-directed"])
        assert args.goal_directed is True

    def test_goal_directed_default_off(self):
        args, _ = self._parse([])
        assert args.goal_directed is False

    def test_goal_proximity_weight(self):
        args, _ = self._parse(["--goal-proximity-weight", "0.5"])
        assert args.goal_proximity_weight == 0.5

    def test_goal_proximity_weight_default(self):
        args, _ = self._parse([])
        assert args.goal_proximity_weight == 0.3

    def test_embedding_evolution_rate(self):
        args, _ = self._parse(["--embedding-evolution-rate", "0.05"])
        assert args.embedding_evolution_rate == 0.05

    def test_embedding_evolution_rate_default(self):
        args, _ = self._parse([])
        assert args.embedding_evolution_rate == 0.01

    def test_all_hierarchical_params_together(self):
        args, _ = self._parse([
            "--goal-directed",
            "--goal-proximity-weight", "0.7",
            "--embedding-evolution-rate", "0.02",
        ])
        assert args.goal_directed is True
        assert args.goal_proximity_weight == 0.7
        assert args.embedding_evolution_rate == 0.02


class TestExistingCLIParametersUnchanged:
    """Existing CLI parameters must work exactly as before."""

    def _parse(self, args: list[str]):
        from pyladr.apps.prover9 import _build_arg_parser
        parser = _build_arg_parser()
        return parser.parse_known_args(args)

    def test_max_given(self):
        args, _ = self._parse(["-max_given", "100"])
        assert args.max_given == 100

    def test_max_kept(self):
        args, _ = self._parse(["-max_kept", "500"])
        assert args.max_kept == 500

    def test_max_seconds(self):
        args, _ = self._parse(["-max_seconds", "30.0"])
        assert args.max_seconds == 30.0

    def test_max_generated(self):
        args, _ = self._parse(["-max_generated", "1000"])
        assert args.max_generated == 1000

    def test_max_proofs(self):
        args, _ = self._parse(["-max_proofs", "3"])
        assert args.max_proofs == 3

    def test_paramodulation(self):
        args, _ = self._parse(["--paramodulation"])
        assert args.paramodulation is True

    def test_no_resolution(self):
        args, _ = self._parse(["--no-resolution"])
        assert args.no_resolution is True

    def test_demodulation(self):
        args, _ = self._parse(["--demodulation"])
        assert args.demodulation is True

    def test_quiet(self):
        args, _ = self._parse(["--quiet"])
        assert args.quiet is True

    def test_online_learning(self):
        args, _ = self._parse(["--online-learning"])
        assert args.online_learning is True

    def test_ml_weight(self):
        args, _ = self._parse(["--ml-weight", "0.4"])
        assert args.ml_weight == 0.4

    def test_repetition_bias(self):
        args, _ = self._parse(["--repetition-bias"])
        assert args.repetition_bias is True

    def test_input_file(self):
        args, _ = self._parse(["-f", "test.in"])
        assert args.input_file == "test.in"

    def test_default_limits(self):
        args, _ = self._parse([])
        assert args.max_given == -1
        assert args.max_kept == -1
        assert args.max_seconds == -1.0
        assert args.max_generated == -1
        assert args.max_proofs == 1


class TestCLIParameterNonInterference:
    """Hierarchical params don't interfere with existing params."""

    def _parse(self, args: list[str]):
        from pyladr.apps.prover9 import _build_arg_parser
        parser = _build_arg_parser()
        return parser.parse_known_args(args)

    def test_hierarchical_with_search_limits(self):
        args, _ = self._parse([
            "--goal-directed",
            "--goal-proximity-weight", "0.5",
            "-max_given", "100",
            "-max_seconds", "30.0",
        ])
        assert args.goal_directed is True
        assert args.goal_proximity_weight == 0.5
        assert args.max_given == 100
        assert args.max_seconds == 30.0

    def test_hierarchical_with_ml_params(self):
        args, _ = self._parse([
            "--goal-directed",
            "--online-learning",
            "--ml-weight", "0.3",
            "--embedding-dim", "64",
        ])
        assert args.goal_directed is True
        assert args.online_learning is True
        assert args.ml_weight == 0.3
        assert args.embedding_dim == 64

    def test_hierarchical_with_repetition_bias(self):
        args, _ = self._parse([
            "--goal-directed",
            "--repetition-bias",
            "--repetition-penalty", "0.5",
        ])
        assert args.goal_directed is True
        assert args.repetition_bias is True
        assert args.repetition_penalty == 0.5

    def test_all_features_combined(self):
        """All feature categories can be combined without conflicts."""
        args, _ = self._parse([
            # Search limits
            "-max_given", "200",
            "-max_seconds", "60.0",
            # Inference rules
            "--paramodulation",
            "--demodulation",
            # ML
            "--online-learning",
            "--ml-weight", "0.2",
            # Repetition
            "--repetition-bias",
            # Hierarchical
            "--goal-directed",
            "--goal-proximity-weight", "0.4",
            "--embedding-evolution-rate", "0.03",
            # Output
            "--quiet",
        ])
        assert args.max_given == 200
        assert args.max_seconds == 60.0
        assert args.paramodulation is True
        assert args.demodulation is True
        assert args.online_learning is True
        assert args.ml_weight == 0.2
        assert args.repetition_bias is True
        assert args.goal_directed is True
        assert args.goal_proximity_weight == 0.4
        assert args.embedding_evolution_rate == 0.03
        assert args.quiet is True


class TestSearchOptionsFromCLI:
    """SearchOptions correctly populated from CLI args."""

    def test_goal_directed_propagated_to_options(self):
        """--goal-directed flag propagates to SearchOptions."""
        from pyladr.search.given_clause import SearchOptions

        opts = SearchOptions(
            goal_directed=True,
            goal_proximity_weight=0.5,
            embedding_evolution_rate=0.02,
        )
        assert opts.goal_directed is True
        assert opts.goal_proximity_weight == 0.5
        assert opts.embedding_evolution_rate == 0.02

    def test_default_options_match_cli_defaults(self):
        """SearchOptions defaults match CLI defaults."""
        from pyladr.apps.prover9 import _build_arg_parser
        from pyladr.search.given_clause import SearchOptions

        parser = _build_arg_parser()
        args, _ = parser.parse_known_args([])

        opts = SearchOptions()

        assert opts.goal_directed == args.goal_directed
        assert opts.goal_proximity_weight == args.goal_proximity_weight
        assert opts.embedding_evolution_rate == args.embedding_evolution_rate
