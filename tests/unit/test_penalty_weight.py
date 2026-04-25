"""Tests for penalty-based clause weight adjustment system."""

from __future__ import annotations

import pytest

from pyladr.search.penalty_weight import (
    PenaltyWeightConfig,
    PenaltyWeightMode,
    penalty_adjusted_weight,
)


# ── PenaltyWeightMode ───────────────────────────────────────────────────────


class TestPenaltyWeightMode:
    def test_three_modes_exist(self):
        assert len(PenaltyWeightMode) == 3

    def test_modes_distinct(self):
        modes = {PenaltyWeightMode.LINEAR, PenaltyWeightMode.EXPONENTIAL, PenaltyWeightMode.STEP}
        assert len(modes) == 3

    def test_mode_names(self):
        assert PenaltyWeightMode.LINEAR.name == "LINEAR"
        assert PenaltyWeightMode.EXPONENTIAL.name == "EXPONENTIAL"
        assert PenaltyWeightMode.STEP.name == "STEP"


# ── PenaltyWeightConfig ─────────────────────────────────────────────────────


class TestPenaltyWeightConfig:
    def test_defaults(self):
        config = PenaltyWeightConfig()
        assert config.enabled is False
        assert config.threshold == 5.0
        assert config.multiplier == 2.0
        assert config.max_adjusted_weight == 1000.0
        assert config.mode == PenaltyWeightMode.EXPONENTIAL

    def test_disabled_by_default(self):
        """Feature is disabled by default for C Prover9 compatibility."""
        config = PenaltyWeightConfig()
        assert not config.enabled

    def test_frozen(self):
        """Config is immutable."""
        config = PenaltyWeightConfig()
        with pytest.raises(AttributeError):
            config.enabled = True  # type: ignore

    def test_custom_config(self):
        config = PenaltyWeightConfig(
            enabled=True,
            threshold=3.0,
            multiplier=1.5,
            max_adjusted_weight=500.0,
            mode=PenaltyWeightMode.LINEAR,
        )
        assert config.enabled is True
        assert config.threshold == 3.0
        assert config.multiplier == 1.5
        assert config.max_adjusted_weight == 500.0
        assert config.mode == PenaltyWeightMode.LINEAR

    def test_slots(self):
        """Config uses __slots__ for memory efficiency."""
        config = PenaltyWeightConfig()
        # frozen dataclass with slots=True won't allow arbitrary attributes
        with pytest.raises((AttributeError, TypeError)):
            config.nonexistent = 42  # type: ignore


# ── penalty_adjusted_weight — disabled ───────────────────────────────────────


class TestPenaltyAdjustedWeightDisabled:
    """When disabled, weight should always pass through unchanged."""

    def test_disabled_returns_base_weight(self):
        config = PenaltyWeightConfig(enabled=False)
        assert penalty_adjusted_weight(10.0, 100.0, config) == 10.0

    def test_disabled_ignores_high_penalty(self):
        config = PenaltyWeightConfig(enabled=False)
        assert penalty_adjusted_weight(5.0, 999.0, config) == 5.0

    def test_disabled_ignores_zero_penalty(self):
        config = PenaltyWeightConfig(enabled=False)
        assert penalty_adjusted_weight(7.0, 0.0, config) == 7.0

    def test_default_config_passes_through(self):
        """Default PenaltyWeightConfig should not alter weight."""
        config = PenaltyWeightConfig()
        assert penalty_adjusted_weight(42.0, 20.0, config) == 42.0


# ── penalty_adjusted_weight — threshold behavior ────────────────────────────


class TestPenaltyAdjustedWeightThreshold:
    """Penalty below threshold should return base weight unchanged."""

    def test_below_threshold_returns_base(self):
        config = PenaltyWeightConfig(enabled=True, threshold=5.0)
        assert penalty_adjusted_weight(10.0, 3.0, config) == 10.0

    def test_at_threshold_adjusts(self):
        """Penalty exactly at threshold should trigger adjustment."""
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.STEP,
        )
        result = penalty_adjusted_weight(10.0, 5.0, config)
        assert result == 20.0  # 10.0 * 2.0

    def test_just_below_threshold_returns_base(self):
        config = PenaltyWeightConfig(enabled=True, threshold=5.0)
        assert penalty_adjusted_weight(10.0, 4.999, config) == 10.0

    def test_zero_penalty_returns_base(self):
        config = PenaltyWeightConfig(enabled=True, threshold=5.0)
        assert penalty_adjusted_weight(10.0, 0.0, config) == 10.0

    def test_negative_penalty_returns_base(self):
        config = PenaltyWeightConfig(enabled=True, threshold=5.0)
        assert penalty_adjusted_weight(10.0, -3.0, config) == 10.0

    def test_zero_threshold_adjusts_any_positive_penalty(self):
        """With threshold=0, any penalty >= 0 should trigger adjustment."""
        config = PenaltyWeightConfig(
            enabled=True, threshold=0.0, multiplier=2.0,
            mode=PenaltyWeightMode.STEP,
            max_adjusted_weight=1000.0,
        )
        # penalty=0.0 is NOT < threshold=0.0, so it adjusts
        result = penalty_adjusted_weight(10.0, 0.0, config)
        assert result == 20.0


# ── penalty_adjusted_weight — LINEAR mode ────────────────────────────────────


class TestLinearMode:
    """LINEAR: adjusted = base + multiplier * penalty."""

    def test_basic_linear(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.LINEAR, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(10.0, 8.0, config)
        assert result == pytest.approx(10.0 + 2.0 * 8.0)  # 26.0

    def test_linear_at_threshold(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=3.0,
            mode=PenaltyWeightMode.LINEAR, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(10.0, 5.0, config)
        assert result == pytest.approx(10.0 + 3.0 * 5.0)  # 25.0

    def test_linear_high_penalty(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=1.0, multiplier=10.0,
            mode=PenaltyWeightMode.LINEAR, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(5.0, 50.0, config)
        assert result == pytest.approx(5.0 + 10.0 * 50.0)  # 505.0

    def test_linear_multiplier_one(self):
        """Multiplier of 1.0 adds penalty directly to weight."""
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=1.0,
            mode=PenaltyWeightMode.LINEAR, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(10.0, 7.0, config)
        assert result == pytest.approx(17.0)

    def test_linear_zero_base_weight(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.LINEAR, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(0.0, 10.0, config)
        assert result == pytest.approx(0.0 + 2.0 * 10.0)  # 20.0

    def test_linear_cap_enforced(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=1.0, multiplier=100.0,
            mode=PenaltyWeightMode.LINEAR, max_adjusted_weight=500.0,
        )
        result = penalty_adjusted_weight(10.0, 100.0, config)
        assert result == 500.0  # 10 + 100*100 = 10010, capped at 500


# ── penalty_adjusted_weight — EXPONENTIAL mode ──────────────────────────────


class TestExponentialMode:
    """EXPONENTIAL: adjusted = base * multiplier^(penalty / threshold)."""

    def test_basic_exponential(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.EXPONENTIAL, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(10.0, 10.0, config)
        # exponent = 10.0 / 5.0 = 2.0
        # adjusted = 10.0 * 2.0^2 = 40.0
        assert result == pytest.approx(40.0)

    def test_exponential_at_threshold(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.EXPONENTIAL, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(10.0, 5.0, config)
        # exponent = 5.0 / 5.0 = 1.0
        # adjusted = 10.0 * 2.0^1 = 20.0
        assert result == pytest.approx(20.0)

    def test_exponential_double_threshold(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=3.0,
            mode=PenaltyWeightMode.EXPONENTIAL, max_adjusted_weight=10000.0,
        )
        result = penalty_adjusted_weight(10.0, 15.0, config)
        # exponent = 15.0 / 5.0 = 3.0
        # adjusted = 10.0 * 3.0^3 = 270.0
        assert result == pytest.approx(270.0)

    def test_exponential_multiplier_one(self):
        """Multiplier=1.0 means no adjustment regardless of penalty."""
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=1.0,
            mode=PenaltyWeightMode.EXPONENTIAL, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(10.0, 100.0, config)
        assert result == pytest.approx(10.0)  # 1.0^anything = 1.0

    def test_exponential_cap_enforced(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=1.0, multiplier=10.0,
            mode=PenaltyWeightMode.EXPONENTIAL, max_adjusted_weight=500.0,
        )
        result = penalty_adjusted_weight(10.0, 5.0, config)
        # exponent = 5.0 / 1.0 = 5.0
        # adjusted = 10.0 * 10.0^5 = 1_000_000, capped at 500
        assert result == 500.0

    def test_exponential_zero_base_weight(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.EXPONENTIAL, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(0.0, 10.0, config)
        assert result == pytest.approx(0.0)  # 0 * anything = 0

    def test_exponential_fractional_exponent(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=10.0, multiplier=4.0,
            mode=PenaltyWeightMode.EXPONENTIAL, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(10.0, 15.0, config)
        # exponent = 15.0 / 10.0 = 1.5
        # adjusted = 10.0 * 4.0^1.5 = 10 * 8 = 80.0
        assert result == pytest.approx(80.0)


# ── penalty_adjusted_weight — STEP mode ──────────────────────────────────────


class TestStepMode:
    """STEP: adjusted = base * multiplier (flat boost when over threshold)."""

    def test_basic_step(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=3.0,
            mode=PenaltyWeightMode.STEP, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(10.0, 8.0, config)
        assert result == pytest.approx(30.0)  # 10 * 3

    def test_step_at_threshold(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.STEP, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(10.0, 5.0, config)
        assert result == pytest.approx(20.0)

    def test_step_ignores_penalty_magnitude(self):
        """Step mode gives same boost regardless of how far above threshold."""
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.STEP, max_adjusted_weight=1000.0,
        )
        result_low = penalty_adjusted_weight(10.0, 6.0, config)
        result_high = penalty_adjusted_weight(10.0, 100.0, config)
        assert result_low == result_high == pytest.approx(20.0)

    def test_step_cap_enforced(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=1.0, multiplier=100.0,
            mode=PenaltyWeightMode.STEP, max_adjusted_weight=500.0,
        )
        result = penalty_adjusted_weight(10.0, 5.0, config)
        assert result == 500.0  # 10 * 100 = 1000, capped at 500

    def test_step_zero_base_weight(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.STEP, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(0.0, 10.0, config)
        assert result == pytest.approx(0.0)  # 0 * 2 = 0


# ── Cap enforcement ──────────────────────────────────────────────────────────


class TestMaxAdjustedWeightCap:
    """Verify max_adjusted_weight cap across all modes."""

    def test_cap_linear(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=1.0, multiplier=1000.0,
            mode=PenaltyWeightMode.LINEAR, max_adjusted_weight=100.0,
        )
        result = penalty_adjusted_weight(10.0, 50.0, config)
        assert result == 100.0

    def test_cap_exponential(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=1.0, multiplier=10.0,
            mode=PenaltyWeightMode.EXPONENTIAL, max_adjusted_weight=100.0,
        )
        result = penalty_adjusted_weight(10.0, 10.0, config)
        assert result == 100.0

    def test_cap_step(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=1.0, multiplier=50.0,
            mode=PenaltyWeightMode.STEP, max_adjusted_weight=100.0,
        )
        result = penalty_adjusted_weight(10.0, 5.0, config)
        assert result == 100.0

    def test_exactly_at_cap(self):
        """When result equals cap, should return the cap value."""
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.STEP, max_adjusted_weight=20.0,
        )
        result = penalty_adjusted_weight(10.0, 5.0, config)
        assert result == 20.0

    def test_below_cap_not_capped(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.STEP, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(10.0, 5.0, config)
        assert result == 20.0  # Below cap, not capped


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_very_small_base_weight(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=1.0, multiplier=2.0,
            mode=PenaltyWeightMode.EXPONENTIAL, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(0.001, 5.0, config)
        # 0.001 * 2^5 = 0.032
        assert result == pytest.approx(0.032)

    def test_very_large_base_weight_capped(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.STEP, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(999.0, 10.0, config)
        assert result == 1000.0  # 999 * 2 = 1998, capped

    def test_very_large_penalty(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=1.0, multiplier=2.0,
            mode=PenaltyWeightMode.LINEAR, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(5.0, 1_000_000.0, config)
        assert result == 1000.0  # Capped

    def test_very_small_threshold(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=0.001, multiplier=2.0,
            mode=PenaltyWeightMode.EXPONENTIAL, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(1.0, 1.0, config)
        # exponent = 1.0 / 0.001 = 1000 → 2^1000 is huge, capped
        assert result == 1000.0

    def test_multiplier_less_than_one_exponential(self):
        """Multiplier < 1.0 could theoretically reduce weight in exponential mode."""
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=0.5,
            mode=PenaltyWeightMode.EXPONENTIAL, max_adjusted_weight=1000.0,
        )
        result = penalty_adjusted_weight(10.0, 10.0, config)
        # exponent = 10/5 = 2, adjusted = 10 * 0.5^2 = 10 * 0.25 = 2.5
        assert result == pytest.approx(2.5)

    def test_base_weight_already_at_cap(self):
        """Base weight at cap, penalty should not reduce below cap."""
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.STEP, max_adjusted_weight=100.0,
        )
        result = penalty_adjusted_weight(100.0, 10.0, config)
        assert result == 100.0  # min(200, 100) = 100


# ── Composability with penalty computation ───────────────────────────────────


class TestComposability:
    """Verify penalty_adjusted_weight works with realistic penalty values."""

    def test_typical_low_penalty_no_adjustment(self):
        """Clause with low generality penalty passes through."""
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.EXPONENTIAL,
        )
        # Specific clause with penalty ~0.3
        result = penalty_adjusted_weight(6.0, 0.3, config)
        assert result == 6.0  # Below threshold

    def test_typical_general_clause_adjusted(self):
        """Overly general clause (penalty >= 10.0) gets weight increase."""
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.EXPONENTIAL, max_adjusted_weight=1000.0,
        )
        # General clause: P(x) → penalty ~10.0
        result = penalty_adjusted_weight(3.0, 10.0, config)
        # exponent = 10/5 = 2, adjusted = 3 * 2^2 = 12.0
        assert result == pytest.approx(12.0)

    def test_combined_penalty_propagation_and_intrinsic(self):
        """Penalty from both intrinsic generality and inheritance."""
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.LINEAR, max_adjusted_weight=1000.0,
        )
        # Combined penalty: own=6.0 + inherited=4.0 = 10.0
        result = penalty_adjusted_weight(5.0, 10.0, config)
        assert result == pytest.approx(5.0 + 2.0 * 10.0)  # 25.0

    def test_progressive_penalty_increasing_weight(self):
        """Higher penalties should produce higher adjusted weights."""
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.LINEAR, max_adjusted_weight=1000.0,
        )
        base = 10.0
        w5 = penalty_adjusted_weight(base, 5.0, config)
        w10 = penalty_adjusted_weight(base, 10.0, config)
        w15 = penalty_adjusted_weight(base, 15.0, config)
        assert w5 < w10 < w15

    def test_exponential_grows_faster_than_linear(self):
        """Exponential mode should produce higher weights than linear for high penalties."""
        linear_config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.LINEAR, max_adjusted_weight=10000.0,
        )
        exp_config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.EXPONENTIAL, max_adjusted_weight=10000.0,
        )
        base = 10.0
        penalty = 20.0
        linear_result = penalty_adjusted_weight(base, penalty, linear_config)
        exp_result = penalty_adjusted_weight(base, penalty, exp_config)
        # linear: 10 + 2*20 = 50
        # exponential: 10 * 2^(20/5) = 10 * 16 = 160
        assert exp_result > linear_result


# ── Determinism ──────────────────────────────────────────────────────────────


class TestDeterminism:
    """Verify that results are deterministic and reproducible."""

    def test_same_inputs_same_output(self):
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.EXPONENTIAL, max_adjusted_weight=1000.0,
        )
        r1 = penalty_adjusted_weight(10.0, 8.0, config)
        r2 = penalty_adjusted_weight(10.0, 8.0, config)
        assert r1 == r2

    def test_repeated_calls_no_state(self):
        """Function is pure — no hidden state between calls."""
        config = PenaltyWeightConfig(
            enabled=True, threshold=5.0, multiplier=2.0,
            mode=PenaltyWeightMode.LINEAR, max_adjusted_weight=1000.0,
        )
        # Call with high penalty first, then low
        penalty_adjusted_weight(10.0, 100.0, config)
        result = penalty_adjusted_weight(10.0, 6.0, config)
        assert result == pytest.approx(10.0 + 2.0 * 6.0)  # 22.0
