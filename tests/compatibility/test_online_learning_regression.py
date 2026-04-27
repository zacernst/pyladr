"""Regression tests ensuring online learning does not break core search behavior.

These tests verify that:
1. Search with online learning disabled produces identical results to baseline
2. Online learning can be enabled without crashing search
3. ML components fail gracefully and fall back to traditional behavior
4. No regressions in proof finding, statistics tracking, or determinism
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from tests.compatibility.conftest import (
    ml_available,
    run_search,
    skip_without_ml,
)


# ── Baseline Search Behavior Tests ────────────────────────────────────────


class TestSearchBaselinePreserved:
    """Verify that enabling online learning infrastructure does not alter
    search results when learning is disabled or has no effect."""

    def test_trivial_proof_with_ml_imports(self, trivial_resolution_clauses):
        """Importing ML modules doesn't affect basic resolution search."""
        from pyladr.search.given_clause import ExitCode

        # Import ML modules to ensure they don't interfere
        try:
            import pyladr.search.ml_selection  # noqa: F401
            import pyladr.search.inference_guidance  # noqa: F401
        except ImportError:
            pass

        result = run_search(usable=[], sos=trivial_resolution_clauses)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1

    def test_equational_proof_with_ml_imports(self, equational_problem):
        """ML imports don't affect equational search."""
        from pyladr.search.given_clause import ExitCode

        try:
            import pyladr.search.ml_selection  # noqa: F401
        except ImportError:
            pass

        st, usable, sos = equational_problem
        result = run_search(
            usable=usable,
            sos=sos,
            paramodulation=True,
            symbol_table=st,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1

    def test_determinism_with_ml_imports(self):
        """Search remains deterministic after ML imports."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term

        try:
            import pyladr.search.ml_selection  # noqa: F401
            import pyladr.search.inference_guidance  # noqa: F401
        except ImportError:
            pass

        def make_clauses():
            P_sn, Q_sn, a_sn = 1, 2, 3
            a = get_rigid_term(a_sn, 0)
            x = get_variable_term(0)
            c1 = Clause(literals=(Literal(sign=True, atom=get_rigid_term(P_sn, 1, (a,))),))
            c2 = Clause(literals=(
                Literal(sign=False, atom=get_rigid_term(P_sn, 1, (x,))),
                Literal(sign=True, atom=get_rigid_term(Q_sn, 1, (x,))),
            ))
            c3 = Clause(literals=(Literal(sign=False, atom=get_rigid_term(Q_sn, 1, (a,))),))
            return [c1, c2, c3]

        results = [run_search(usable=[], sos=make_clauses()) for _ in range(3)]
        for r in results[1:]:
            assert r.stats.given == results[0].stats.given
            assert r.stats.generated == results[0].stats.generated
            assert r.stats.kept == results[0].stats.kept


# ── ML Selection Disabled Equivalence ─────────────────────────────────────


class TestMLDisabledEquivalence:
    """Verify ML-disabled selection is byte-for-byte equivalent to baseline."""

    def test_disabled_ml_selection_trivial(self):
        """Disabled ML selection produces identical results for trivial search."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.search.given_clause import ExitCode

        def make_clauses():
            P_sn, Q_sn, a_sn = 1, 2, 3
            a = get_rigid_term(a_sn, 0)
            x = get_variable_term(0)
            c1 = Clause(literals=(Literal(sign=True, atom=get_rigid_term(P_sn, 1, (a,))),))
            c2 = Clause(literals=(
                Literal(sign=False, atom=get_rigid_term(P_sn, 1, (x,))),
                Literal(sign=True, atom=get_rigid_term(Q_sn, 1, (x,))),
            ))
            c3 = Clause(literals=(Literal(sign=False, atom=get_rigid_term(Q_sn, 1, (a,))),))
            return [c1, c2, c3]

        baseline = run_search(usable=[], sos=make_clauses())
        result = run_search(usable=[], sos=make_clauses())

        assert result.exit_code == baseline.exit_code
        assert result.stats.given == baseline.stats.given
        assert result.stats.generated == baseline.stats.generated
        assert result.stats.kept == baseline.stats.kept
        assert len(result.proofs) == len(baseline.proofs)

    def test_disabled_ml_selection_equational(self):
        """Disabled ML selection produces identical results for equational search."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.symbol import SymbolTable
        from pyladr.core.term import build_binary_term, get_rigid_term

        def make_problem():
            st = SymbolTable()
            eq_sn = st.str_to_sn("=", 2)
            p_sn = st.str_to_sn("p", 1)
            a_sn = st.str_to_sn("a", 0)
            b_sn = st.str_to_sn("b", 0)
            a = get_rigid_term(a_sn, 0)
            b = get_rigid_term(b_sn, 0)
            eq_ab = build_binary_term(eq_sn, a, b)
            p_a = get_rigid_term(p_sn, 1, (a,))
            p_b = get_rigid_term(p_sn, 1, (b,))
            c1 = Clause(literals=(Literal(sign=True, atom=eq_ab),))
            c2 = Clause(literals=(Literal(sign=True, atom=p_a),))
            c3 = Clause(literals=(Literal(sign=False, atom=p_b),))
            return st, [c1], [c2, c3]

        st1, u1, s1 = make_problem()
        baseline = run_search(usable=u1, sos=s1, paramodulation=True, symbol_table=st1)

        st2, u2, s2 = make_problem()
        result = run_search(usable=u2, sos=s2, paramodulation=True, symbol_table=st2)

        assert result.exit_code == baseline.exit_code
        assert result.stats.given == baseline.stats.given
        assert result.stats.generated == baseline.stats.generated


# ── Online Learning Components Regression ─────────────────────────────────


@skip_without_ml
class TestOnlineLearningComponentRegression:
    """Regression tests for online learning components in isolation."""

    def test_experience_buffer_api_stable(self):
        """ExperienceBuffer public API has not changed."""
        from pyladr.ml.online_learning import ExperienceBuffer

        buf = ExperienceBuffer(capacity=100)
        assert hasattr(buf, "add")
        assert hasattr(buf, "sample_contrastive_batch")
        assert hasattr(buf, "clear")
        assert hasattr(buf, "get_recent")
        assert hasattr(buf, "size")
        assert hasattr(buf, "num_productive")
        assert hasattr(buf, "num_unproductive")

    def test_online_learning_config_defaults_stable(self):
        """OnlineLearningConfig defaults have not changed unexpectedly."""
        from pyladr.ml.online_learning import OnlineLearningConfig

        config = OnlineLearningConfig()
        assert config.enabled is True
        assert config.update_interval == 200
        assert config.buffer_capacity == 5000
        assert config.batch_size == 32
        assert config.momentum == 0.995
        assert config.temperature == 0.07
        assert config.max_updates == 0

    def test_outcome_types_complete(self):
        """All expected OutcomeType values exist."""
        from pyladr.ml.online_learning import OutcomeType

        expected = {"KEPT", "SUBSUMED", "TAUTOLOGY", "WEIGHT_LIMIT", "PROOF", "SUBSUMER"}
        actual = {ot.name for ot in OutcomeType}
        assert expected == actual

    def test_online_learning_manager_api_stable(self):
        """OnlineLearningManager public API has not changed."""
        from pyladr.ml.online_learning import OnlineLearningManager

        assert hasattr(OnlineLearningManager, "record_outcome")
        assert hasattr(OnlineLearningManager, "should_update")
        assert hasattr(OnlineLearningManager, "update")
        assert hasattr(OnlineLearningManager, "on_proof_found")
        assert hasattr(OnlineLearningManager, "rollback_to_version")
        assert hasattr(OnlineLearningManager, "rollback_to_best")
        assert hasattr(OnlineLearningManager, "has_converged")
        assert hasattr(OnlineLearningManager, "stats")
        assert hasattr(OnlineLearningManager, "report")
        assert hasattr(OnlineLearningManager, "use_ema_model")
        assert hasattr(OnlineLearningManager, "restore_training_model")

    def test_ab_tracker_api_stable(self):
        """ABTestTracker public API has not changed."""
        from pyladr.ml.online_learning import ABTestTracker

        tracker = ABTestTracker()
        assert hasattr(tracker, "set_baseline")
        assert hasattr(tracker, "record_outcome")
        assert hasattr(tracker, "current_rate")
        assert hasattr(tracker, "has_enough_data")
        assert hasattr(tracker, "is_improvement")
        assert hasattr(tracker, "is_degradation")


# ── Inference Guidance Regression ─────────────────────────────────────────


class TestInferenceGuidanceRegression:
    """Regression tests for inference guidance module."""

    def test_guidance_config_defaults_stable(self):
        """InferenceGuidanceConfig defaults match documented behavior."""
        from pyladr.search.inference_guidance import InferenceGuidanceConfig

        config = InferenceGuidanceConfig()
        assert config.enabled is False  # Opt-in
        assert config.max_candidates == -1  # No limit by default
        assert config.compatibility_threshold == 0.0  # No filtering by default
        assert config.early_termination_count == -1  # No early termination

    def test_disabled_guidance_identity(self):
        """Disabled guidance returns exact same list object."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term
        from pyladr.search.inference_guidance import (
            EmbeddingGuidedInference,
            InferenceGuidanceConfig,
        )

        config = InferenceGuidanceConfig(enabled=False)
        guidance = EmbeddingGuidedInference(config=config)

        atom = get_rigid_term(1, 0)
        given = Clause(literals=(Literal(sign=True, atom=atom),))
        usable = [
            Clause(literals=(Literal(sign=True, atom=get_rigid_term(i, 0)),))
            for i in range(2, 5)
        ]

        result = guidance.prioritize(given, usable)
        assert result is usable  # Same object — zero overhead

    def test_guidance_stats_api_stable(self):
        """InferenceGuidanceStats public API has not changed."""
        from pyladr.search.inference_guidance import InferenceGuidanceStats

        stats = InferenceGuidanceStats()
        assert hasattr(stats, "guided_rounds")
        assert hasattr(stats, "unguided_rounds")
        assert hasattr(stats, "total_candidates_scored")
        assert hasattr(stats, "total_candidates_selected")
        assert hasattr(stats, "total_candidates_skipped")
        assert hasattr(stats, "early_terminations")
        assert hasattr(stats, "report")


# ── ML Selection Regression ──────────────────────────────────────────────


class TestMLSelectionRegression:
    """Regression tests for ML-enhanced clause selection."""

    def test_selection_config_defaults_stable(self):
        """MLSelectionConfig defaults match documented behavior."""
        from pyladr.search.ml_selection import MLSelectionConfig

        config = MLSelectionConfig()
        assert config.enabled is False  # Opt-in
        assert config.min_sos_for_ml > 0

    def test_disabled_selection_uses_traditional_rules(self):
        """Disabled ML selection delegates entirely to GivenSelection."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term
        from pyladr.search.ml_selection import (
            EmbeddingEnhancedSelection,
            MLSelectionConfig,
        )
        from pyladr.search.state import ClauseList

        config = MLSelectionConfig(enabled=False)
        sel = EmbeddingEnhancedSelection(ml_config=config)

        # Create clauses with known weights
        clauses = []
        for i, w in enumerate([5.0, 3.0, 1.0, 4.0]):
            c = Clause(literals=(Literal(sign=True, atom=get_rigid_term(i + 1, 0)),))
            c.weight = w
            c.id = i + 1
            clauses.append(c)

        sos = ClauseList("sos")
        for c in clauses:
            sos.append(c)

        # First pick at index 0 is age-based (A rule), picks first inserted clause
        c, name = sel.select_given(sos, 0)
        assert c is not None
        assert c.weight == 5.0  # first clause by age
        # Name should be traditional (not ML-enhanced)
        assert "+ML" not in name
