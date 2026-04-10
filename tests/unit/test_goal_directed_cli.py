"""Tests for goal-directed clause selection CLI arguments (Task #9).

TDD tests for:
- Goal-directed argument group in prover9.py
- Parameters: --goal-directed, --goal-proximity-weight, --embedding-evolution-rate
- Integration with SearchOptions without breaking changes
- Parameter validation and sensible defaults
"""

from __future__ import annotations

import io
import sys
from unittest.mock import patch

import pytest

from pyladr.apps.prover9 import _build_arg_parser
from pyladr.search.given_clause import SearchOptions


# ── Argument parser tests ──────────────────────────────────────────────────


class TestGoalDirectedArgParser:
    """Test that the goal-directed argument group is properly configured."""

    @pytest.fixture
    def parser(self):
        return _build_arg_parser()

    def test_goal_directed_flag_default_off(self, parser):
        """--goal-directed defaults to False (disabled)."""
        args = parser.parse_args([])
        assert args.goal_directed is False

    def test_goal_directed_flag_enabled(self, parser):
        """--goal-directed can be enabled."""
        args = parser.parse_args(["--goal-directed"])
        assert args.goal_directed is True

    def test_goal_proximity_weight_default(self, parser):
        """--goal-proximity-weight defaults to 0.3."""
        args = parser.parse_args([])
        assert args.goal_proximity_weight == 0.3

    def test_goal_proximity_weight_custom(self, parser):
        """--goal-proximity-weight accepts custom float values."""
        args = parser.parse_args(["--goal-proximity-weight", "0.7"])
        assert args.goal_proximity_weight == 0.7

    def test_goal_proximity_weight_zero(self, parser):
        """--goal-proximity-weight accepts 0.0 (no proximity influence)."""
        args = parser.parse_args(["--goal-proximity-weight", "0.0"])
        assert args.goal_proximity_weight == 0.0

    def test_goal_proximity_weight_one(self, parser):
        """--goal-proximity-weight accepts 1.0 (full proximity influence)."""
        args = parser.parse_args(["--goal-proximity-weight", "1.0"])
        assert args.goal_proximity_weight == 1.0

    def test_embedding_evolution_rate_default(self, parser):
        """--embedding-evolution-rate defaults to 0.01."""
        args = parser.parse_args([])
        assert args.embedding_evolution_rate == 0.01

    def test_embedding_evolution_rate_custom(self, parser):
        """--embedding-evolution-rate accepts custom float values."""
        args = parser.parse_args(["--embedding-evolution-rate", "0.05"])
        assert args.embedding_evolution_rate == 0.05

    def test_embedding_evolution_rate_zero(self, parser):
        """--embedding-evolution-rate accepts 0.0 (no evolution)."""
        args = parser.parse_args(["--embedding-evolution-rate", "0.0"])
        assert args.embedding_evolution_rate == 0.0

    def test_all_goal_directed_args_together(self, parser):
        """All goal-directed arguments can be used together."""
        args = parser.parse_args([
            "--goal-directed",
            "--goal-proximity-weight", "0.5",
            "--embedding-evolution-rate", "0.03",
        ])
        assert args.goal_directed is True
        assert args.goal_proximity_weight == 0.5
        assert args.embedding_evolution_rate == 0.03

    def test_goal_directed_args_with_existing_ml_args(self, parser):
        """Goal-directed args work alongside existing ML arguments."""
        args = parser.parse_args([
            "--online-learning",
            "--ml-weight", "0.4",
            "--goal-directed",
            "--goal-proximity-weight", "0.6",
        ])
        assert args.online_learning is True
        assert args.ml_weight == 0.4
        assert args.goal_directed is True
        assert args.goal_proximity_weight == 0.6

    def test_goal_directed_args_with_existing_search_args(self, parser):
        """Goal-directed args don't interfere with search limit arguments."""
        args = parser.parse_args([
            "-max_given", "100",
            "-max_seconds", "30",
            "--goal-directed",
        ])
        assert args.max_given == 100
        assert args.max_seconds == 30.0
        assert args.goal_directed is True


# ── Backward compatibility tests ───────────────────────────────────────────


class TestBackwardCompatibility:
    """Ensure existing CLI behavior is completely preserved."""

    @pytest.fixture
    def parser(self):
        return _build_arg_parser()

    def test_empty_args_unchanged(self, parser):
        """Parsing no arguments still produces the same defaults for all
        existing parameters."""
        args = parser.parse_args([])
        # Search limits
        assert args.max_given == -1
        assert args.max_kept == -1
        assert args.max_seconds == -1.0
        assert args.max_generated == -1
        assert args.max_proofs == 1
        # Inference rules
        assert args.paramodulation is False
        assert args.no_resolution is False
        assert args.no_factoring is False
        assert args.demodulation is False
        assert args.back_demod is False
        # Output
        assert args.quiet is False
        assert args.print_kept is False
        assert args.no_print_given is False
        # ML
        assert args.online_learning is False
        assert args.ml_weight is None
        assert args.embedding_dim == 32
        # Repetition bias
        assert args.repetition_bias is False
        assert args.repetition_penalty == 0.3
        assert args.repetition_decay == 0.02

    def test_existing_ml_args_still_work(self, parser):
        """Existing ML arguments are not broken by new goal-directed group."""
        args = parser.parse_args([
            "--online-learning",
            "--ml-weight", "0.35",
            "--embedding-dim", "64",
        ])
        assert args.online_learning is True
        assert args.ml_weight == 0.35
        assert args.embedding_dim == 64

    def test_existing_repetition_args_still_work(self, parser):
        """Existing repetition bias arguments are not broken."""
        args = parser.parse_args([
            "--repetition-bias",
            "--repetition-penalty", "0.5",
            "--repetition-decay", "0.01",
        ])
        assert args.repetition_bias is True
        assert args.repetition_penalty == 0.5
        assert args.repetition_decay == 0.01


# ── SearchOptions integration tests ───────────────────────────────────────


class TestSearchOptionsGoalDirected:
    """Test goal-directed fields in SearchOptions dataclass."""

    def test_default_search_options_goal_directed_off(self):
        """SearchOptions defaults have goal-directed disabled."""
        opts = SearchOptions()
        assert opts.goal_directed is False
        assert opts.goal_proximity_weight == 0.3
        assert opts.embedding_evolution_rate == 0.01

    def test_search_options_goal_directed_enabled(self):
        """SearchOptions can be created with goal-directed enabled."""
        opts = SearchOptions(
            goal_directed=True,
            goal_proximity_weight=0.5,
            embedding_evolution_rate=0.03,
        )
        assert opts.goal_directed is True
        assert opts.goal_proximity_weight == 0.5
        assert opts.embedding_evolution_rate == 0.03

    def test_search_options_existing_fields_unchanged(self):
        """Adding goal-directed fields doesn't change existing defaults."""
        opts = SearchOptions()
        # Inference rules
        assert opts.binary_resolution is True
        assert opts.paramodulation is False
        assert opts.factoring is True
        # Limits
        assert opts.max_given == -1
        assert opts.max_kept == -1
        assert opts.max_seconds == -1.0
        # ML
        assert opts.online_learning is False
        assert opts.ml_weight is None
        assert opts.embedding_dim == 32
        # Repetition
        assert opts.repetition_bias is False

    def test_search_options_goal_directed_with_ml(self):
        """Goal-directed and ML options can coexist."""
        opts = SearchOptions(
            online_learning=True,
            ml_weight=0.4,
            goal_directed=True,
            goal_proximity_weight=0.6,
        )
        assert opts.online_learning is True
        assert opts.ml_weight == 0.4
        assert opts.goal_directed is True
        assert opts.goal_proximity_weight == 0.6


# ── CLI-to-SearchOptions mapping tests ─────────────────────────────────────


class TestCLIToSearchOptionsMapping:
    """Test that CLI arguments correctly map to SearchOptions fields."""

    @pytest.fixture
    def parser(self):
        return _build_arg_parser()

    def test_goal_directed_args_map_to_search_options(self, parser):
        """CLI goal-directed arguments map correctly to SearchOptions."""
        args = parser.parse_args([
            "--goal-directed",
            "--goal-proximity-weight", "0.5",
            "--embedding-evolution-rate", "0.03",
        ])
        opts = SearchOptions(
            goal_directed=args.goal_directed,
            goal_proximity_weight=args.goal_proximity_weight,
            embedding_evolution_rate=args.embedding_evolution_rate,
        )
        assert opts.goal_directed is True
        assert opts.goal_proximity_weight == 0.5
        assert opts.embedding_evolution_rate == 0.03

    def test_default_args_map_to_default_search_options(self, parser):
        """Default CLI args produce default SearchOptions values."""
        args = parser.parse_args([])
        opts = SearchOptions(
            goal_directed=args.goal_directed,
            goal_proximity_weight=args.goal_proximity_weight,
            embedding_evolution_rate=args.embedding_evolution_rate,
        )
        assert opts.goal_directed is False
        assert opts.goal_proximity_weight == 0.3
        assert opts.embedding_evolution_rate == 0.01


# ── Help text tests ────────────────────────────────────────────────────────


class TestHelpText:
    """Verify help text includes goal-directed section."""

    def test_help_contains_goal_directed_group(self):
        """--help output contains goal-directed selection group."""
        parser = _build_arg_parser()
        help_text = parser.format_help()
        assert "goal-directed" in help_text.lower()

    def test_help_contains_goal_proximity_weight(self):
        """--help output documents --goal-proximity-weight."""
        parser = _build_arg_parser()
        help_text = parser.format_help()
        assert "--goal-proximity-weight" in help_text

    def test_help_contains_embedding_evolution_rate(self):
        """--help output documents --embedding-evolution-rate."""
        parser = _build_arg_parser()
        help_text = parser.format_help()
        assert "--embedding-evolution-rate" in help_text


# ── Parameter validation tests ─────────────────────────────────────────────


class TestParameterValidation:
    """Test that invalid parameter values are handled appropriately."""

    @pytest.fixture
    def parser(self):
        return _build_arg_parser()

    def test_goal_proximity_weight_rejects_non_float(self, parser):
        """--goal-proximity-weight rejects non-numeric values."""
        with pytest.raises(SystemExit):
            parser.parse_args(["--goal-proximity-weight", "abc"])

    def test_embedding_evolution_rate_rejects_non_float(self, parser):
        """--embedding-evolution-rate rejects non-numeric values."""
        with pytest.raises(SystemExit):
            parser.parse_args(["--embedding-evolution-rate", "xyz"])

    def test_goal_proximity_weight_accepts_negative(self, parser):
        """--goal-proximity-weight accepts negative values (argparse level)."""
        # Validation at application level, not argparse level
        args = parser.parse_args(["--goal-proximity-weight", "-0.1"])
        assert args.goal_proximity_weight == -0.1
