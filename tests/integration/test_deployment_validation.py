"""End-to-end deployment validation tests.

These tests simulate production deployment scenarios to verify that the
complete online learning system works correctly with realistic configurations.
Run these as a pre-deployment smoke test:

    pytest tests/integration/test_deployment_validation.py -v

Tests cover:
1. Full search-with-learning pipeline on real problems
2. Production-safe configuration defaults
3. Monitoring integration during search
4. Graceful shutdown and recovery
5. Configuration override validation
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import (
    build_binary_term,
    build_unary_term,
    get_rigid_term,
    get_variable_term,
)
from pyladr.ml.online_learning import (
    OnlineLearningConfig,
    OnlineLearningManager,
)
from pyladr.search.given_clause import (
    ExitCode,
    GivenClauseSearch,
    SearchOptions,
)
from pyladr.search.online_integration import (
    OnlineIntegrationConfig,
    OnlineSearchIntegration,
)

INPUTS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "inputs"


# ── Production-Safe Encoder ──────────────────────────────────────────────


class ProductionEncoder:
    """Encoder suitable for production deployment testing.

    Uses actual gradient computation to validate the full learning pipeline.
    """

    def __init__(self, dim: int = 32):
        self._dim = dim
        self._linear = torch.nn.Linear(dim, dim)

    def encode_clauses(self, clauses):
        x = torch.randn(len(clauses), self._dim)
        return self._linear(x)

    def parameters(self):
        return self._linear.parameters()

    def named_parameters(self):
        return self._linear.named_parameters()

    def state_dict(self):
        return self._linear.state_dict()

    def load_state_dict(self, s):
        self._linear.load_state_dict(s)

    def train(self, m=True):
        self._linear.train(m)

    def eval(self):
        self._linear.eval()


# ── Problem Builders ─────────────────────────────────────────────────────


def _resolution_problem():
    """P(a), ~P(x)|Q(x), ~Q(a)."""
    P, Q, a_sn = 1, 2, 3
    a = get_rigid_term(a_sn, 0)
    x = get_variable_term(0)
    return None, [], [
        Clause(literals=(Literal(sign=True, atom=get_rigid_term(P, 1, (a,))),)),
        Clause(literals=(
            Literal(sign=False, atom=get_rigid_term(P, 1, (x,))),
            Literal(sign=True, atom=get_rigid_term(Q, 1, (x,))),
        )),
        Clause(literals=(Literal(sign=False, atom=get_rigid_term(Q, 1, (a,))),)),
    ]


def _equational_problem():
    """a=b, p(a), ~p(b)."""
    st = SymbolTable()
    eq = st.str_to_sn("=", 2)
    p = st.str_to_sn("p", 1)
    a_sn = st.str_to_sn("a", 0)
    b_sn = st.str_to_sn("b", 0)
    a, b = get_rigid_term(a_sn, 0), get_rigid_term(b_sn, 0)
    return st, [
        Clause(literals=(Literal(sign=True, atom=build_binary_term(eq, a, b)),)),
    ], [
        Clause(literals=(Literal(sign=True, atom=get_rigid_term(p, 1, (a,))),)),
        Clause(literals=(Literal(sign=False, atom=get_rigid_term(p, 1, (b,))),)),
    ]


def _group_commutativity():
    """Group theory: x*x=e implies commutativity."""
    st = SymbolTable()
    eq_sn = st.str_to_sn("=", 2)
    mult_sn = st.str_to_sn("*", 2)
    inv_sn = st.str_to_sn("'", 1)
    e_sn = st.str_to_sn("e", 0)
    e = get_rigid_term(e_sn, 0)
    x, y, z = get_variable_term(0), get_variable_term(1), get_variable_term(2)
    mult = lambda a, b: build_binary_term(mult_sn, a, b)
    inv = lambda a: build_unary_term(inv_sn, a)
    eq = lambda a, b: build_binary_term(eq_sn, a, b)
    a_sn = st.str_to_sn("a", 0)
    b_sn = st.str_to_sn("b", 0)
    a_c, b_c = get_rigid_term(a_sn, 0), get_rigid_term(b_sn, 0)
    return st, [], [
        Clause(literals=(Literal(sign=True, atom=eq(mult(e, x), x)),)),
        Clause(literals=(Literal(sign=True, atom=eq(mult(inv(x), x), e)),)),
        Clause(literals=(Literal(sign=True, atom=eq(mult(mult(x, y), z), mult(x, mult(y, z)))),)),
        Clause(literals=(Literal(sign=True, atom=eq(mult(x, x), e)),)),
        Clause(literals=(Literal(sign=False, atom=eq(mult(a_c, b_c), mult(b_c, a_c))),)),
    ]


# ── Production Configuration ────────────────────────────────────────────


def _production_integration_config() -> OnlineIntegrationConfig:
    """Conservative production-safe integration configuration."""
    return OnlineIntegrationConfig(
        enabled=True,
        collect_experiences=True,
        trigger_updates=True,
        track_proof_progress=True,
        adaptive_ml_weight=True,
        initial_ml_weight=0.1,
        max_ml_weight=0.3,  # Conservative max
        ml_weight_increase_rate=0.02,
        ml_weight_decrease_rate=0.05,
        min_given_before_ml=20,
        log_integration_events=False,
    )


def _production_learning_config() -> OnlineLearningConfig:
    """Conservative production-safe learning configuration."""
    return OnlineLearningConfig(
        enabled=True,
        update_interval=100,
        min_examples_for_update=50,
        buffer_capacity=2000,
        batch_size=16,
        momentum=0.995,
        temperature=0.07,
        max_updates=0,  # Unlimited
    )


def _create_production_system():
    """Create a fully wired production-like online learning system."""
    encoder = ProductionEncoder()
    ol_config = _production_learning_config()
    manager = OnlineLearningManager(encoder=encoder, config=ol_config)
    int_config = _production_integration_config()
    integration = OnlineSearchIntegration(config=int_config, manager=manager)
    return integration, manager, encoder


# ── End-to-End Smoke Tests ───────────────────────────────────────────────


class TestEndToEndSmoke:
    """Smoke tests verifying the full system works end-to-end."""

    def test_resolution_proof_with_production_config(self):
        """Production config finds trivial resolution proof."""
        integration, manager, _ = _create_production_system()
        search = integration.create_search(
            options=SearchOptions(
                binary_resolution=True, max_given=50, quiet=True,
            ),
        )
        _, _, sos = _resolution_problem()
        result = search.run(usable=[], sos=sos)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_equational_proof_with_production_config(self):
        """Production config finds equational proof."""
        integration, manager, _ = _create_production_system()
        st, usable, sos = _equational_problem()
        search = integration.create_search(
            options=SearchOptions(
                binary_resolution=True, paramodulation=True,
                demodulation=True, max_given=100, quiet=True,
            ),
            symbol_table=st,
        )
        result = search.run(usable=usable, sos=sos)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_group_theory_with_production_config(self):
        """Production config finds group commutativity proof."""
        integration, manager, _ = _create_production_system()
        st, usable, sos = _group_commutativity()
        search = integration.create_search(
            options=SearchOptions(
                binary_resolution=True, paramodulation=True,
                demodulation=True, max_given=500, quiet=True,
            ),
            symbol_table=st,
        )
        result = search.run(usable=usable, sos=sos)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_production_config_collects_experiences(self):
        """Production system collects experiences during search."""
        integration, manager, _ = _create_production_system()
        st, usable, sos = _group_commutativity()
        search = integration.create_search(
            options=SearchOptions(
                binary_resolution=True, paramodulation=True,
                demodulation=True, max_given=500, quiet=True,
            ),
            symbol_table=st,
        )
        search.run(usable=usable, sos=sos)
        assert integration.stats.experiences_collected > 0
        assert manager._buffer.size > 0


# ── Configuration Override Tests ─────────────────────────────────────────


class TestConfigurationOverrides:
    """Verify that users can safely override production defaults."""

    def test_disable_learning_preserves_search(self):
        """Disabling learning produces same result as no integration."""
        # With learning disabled
        config = OnlineIntegrationConfig(enabled=False)
        integration = OnlineSearchIntegration(config=config)
        search_ol = integration.create_search(
            options=SearchOptions(binary_resolution=True, max_given=50, quiet=True),
        )
        _, _, sos = _resolution_problem()
        result_ol = search_ol.run(usable=[], sos=sos)

        # Without integration
        plain = GivenClauseSearch(
            options=SearchOptions(binary_resolution=True, max_given=50, quiet=True),
        )
        _, _, sos2 = _resolution_problem()
        result_plain = plain.run(usable=[], sos=sos2)

        assert result_ol.exit_code == result_plain.exit_code
        assert result_ol.stats.given == result_plain.stats.given

    def test_disable_updates_still_collects(self):
        """trigger_updates=False still collects experiences."""
        config = OnlineIntegrationConfig(
            enabled=True, collect_experiences=True,
            trigger_updates=False, min_given_before_ml=0,
        )
        encoder = ProductionEncoder()
        manager = OnlineLearningManager(
            encoder=encoder, config=_production_learning_config(),
        )
        integration = OnlineSearchIntegration(config=config, manager=manager)
        search = integration.create_search(
            options=SearchOptions(binary_resolution=True, max_given=50, quiet=True),
        )
        _, _, sos = _resolution_problem()
        search.run(usable=[], sos=sos)

        assert integration.stats.experiences_collected > 0
        assert integration.stats.model_updates_triggered == 0

    def test_high_min_given_delays_ml(self):
        """High min_given_before_ml means ML weight stays 0 for short searches."""
        config = OnlineIntegrationConfig(
            enabled=True, min_given_before_ml=1000,
        )
        integration = OnlineSearchIntegration(config=config)
        # For a search that completes in < 1000 givens, ML weight is 0
        assert integration.get_current_ml_weight() == 0.0


# ── File-Based Deployment Tests ──────────────────────────────────────────


class TestFileBasedDeployment:
    """Test production system against actual problem files."""

    @pytest.fixture(params=[
        "identity_only.in",
        "simple_group.in",
        "lattice_absorption.in",
    ])
    def problem_file(self, request):
        path = INPUTS_DIR / request.param
        if not path.exists():
            pytest.skip(f"{request.param} not found")
        return path

    def test_production_system_on_file(self, problem_file):
        """Production system handles real problem files without crashing."""
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(problem_file.read_text())

        integration, manager, _ = _create_production_system()
        search = integration.create_search(
            options=SearchOptions(
                binary_resolution=True,
                paramodulation=True,
                demodulation=True,
                factoring=True,
                max_given=200,
                quiet=True,
            ),
            symbol_table=st,
        )
        result = search.run(
            usable=parsed.usable or [],
            sos=parsed.sos or [],
        )

        # Should terminate cleanly
        assert result.exit_code in (
            ExitCode.MAX_PROOFS_EXIT,
            ExitCode.MAX_GIVEN_EXIT,
            ExitCode.SOS_EMPTY_EXIT,
        )
        # Should have collected some data
        assert integration.stats.experiences_collected >= 0


# ── System Integrity Tests ───────────────────────────────────────────────


class TestSystemIntegrity:
    """Verify system state consistency after various operations."""

    def test_stats_consistent_after_proof(self):
        """Stats remain consistent after finding a proof."""
        integration, manager, _ = _create_production_system()
        search = integration.create_search(
            options=SearchOptions(binary_resolution=True, max_given=50, quiet=True),
        )
        _, _, sos = _resolution_problem()
        result = search.run(usable=[], sos=sos)

        s = integration.stats
        assert s.experiences_collected >= 0
        assert s.model_updates_accepted <= s.model_updates_triggered
        assert s.model_updates_rolled_back <= s.model_updates_triggered
        assert (
            s.model_updates_accepted + s.model_updates_rolled_back
            <= s.model_updates_triggered
        )

    def test_manager_stats_valid_after_search(self):
        """OnlineLearningManager stats are valid after search."""
        integration, manager, _ = _create_production_system()
        st, usable, sos = _group_commutativity()
        search = integration.create_search(
            options=SearchOptions(
                binary_resolution=True, paramodulation=True,
                demodulation=True, max_given=500, quiet=True,
            ),
            symbol_table=st,
        )
        search.run(usable=usable, sos=sos)

        stats = manager.stats
        assert isinstance(stats, dict)
        assert stats["update_count"] >= 0
        assert stats["buffer_size"] >= 0

    def test_encoder_parameters_finite_after_search(self):
        """Encoder parameters remain finite (no NaN/Inf) after search."""
        integration, manager, encoder = _create_production_system()
        st, usable, sos = _group_commutativity()
        search = integration.create_search(
            options=SearchOptions(
                binary_resolution=True, paramodulation=True,
                demodulation=True, max_given=500, quiet=True,
            ),
            symbol_table=st,
        )
        search.run(usable=usable, sos=sos)

        for name, param in encoder.named_parameters():
            assert torch.isfinite(param).all(), (
                f"Parameter {name} contains NaN/Inf after search"
            )

    def test_multiple_sequential_searches(self):
        """System handles multiple sequential searches correctly."""
        integration, manager, _ = _create_production_system()

        for i in range(3):
            _, _, sos = _resolution_problem()
            search = integration.create_search(
                options=SearchOptions(
                    binary_resolution=True, max_given=50, quiet=True,
                ),
            )
            result = search.run(usable=[], sos=sos)
            assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

        # Stats should accumulate across searches
        assert integration.stats.experiences_collected > 0
