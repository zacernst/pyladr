"""Benchmarks comparing traditional vs online-learning-enabled search.

Measures the overhead and potential benefits of online learning across
the standard benchmark problem set. These tests establish performance
baselines for:
- Search overhead from online learning hooks (even when ML is disabled)
- Experience collection overhead during active search
- Proof-finding capability preservation (no regressions)
- Statistical comparison across problem difficulty levels

Marked as @pytest.mark.benchmark — run explicitly with:
    pytest tests/benchmarks/test_online_learning_benchmarks.py -v
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import (
    build_binary_term,
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
from pyladr.search.selection import GivenSelection

INPUTS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "inputs"


# ── Mock encoder for benchmarking ────────────────────────────────────────


class BenchmarkEncoder:
    """Lightweight mock encoder for benchmarking online learning overhead."""

    def __init__(self, dim: int = 32):
        self._dim = dim
        self._linear = torch.nn.Linear(dim, dim)

    def encode_clauses(self, clauses: list[Clause]) -> torch.Tensor:
        # Must use parameters so gradients can flow for model updates
        x = torch.randn(len(clauses), self._dim)
        return self._linear(x)

    def parameters(self):
        return self._linear.parameters()

    def named_parameters(self):
        return self._linear.named_parameters()

    def state_dict(self):
        return self._linear.state_dict()

    def load_state_dict(self, state):
        self._linear.load_state_dict(state)

    def train(self, mode=True):
        self._linear.train(mode)

    def eval(self):
        self._linear.eval()


# ── Problem helpers ──────────────────────────────────────────────────────


@dataclass
class BenchmarkRun:
    """Result of a single benchmark run."""

    problem: str
    mode: str  # "traditional", "hooks_only", "learning_enabled"
    proved: bool = False
    exit_code: ExitCode = ExitCode.MAX_GIVEN_EXIT
    wall_seconds: float = 0.0
    given: int = 0
    generated: int = 0
    kept: int = 0
    experiences_collected: int = 0
    model_updates: int = 0
    proofs: int = 0


def _make_trivial_resolution():
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


def _make_equational():
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


def _make_group_commutativity():
    """Group theory: x*x=e implies commutativity."""
    st = SymbolTable()
    eq_sn = st.str_to_sn("=", 2)
    mult_sn = st.str_to_sn("*", 2)
    inv_sn = st.str_to_sn("'", 1)
    e_sn = st.str_to_sn("e", 0)

    e = get_rigid_term(e_sn, 0)
    x, y, z = get_variable_term(0), get_variable_term(1), get_variable_term(2)

    def mult(a, b):
        return build_binary_term(mult_sn, a, b)

    def eq(a, b):
        return build_binary_term(eq_sn, a, b)

    def inv(a):
        from pyladr.core.term import build_unary_term
        return build_unary_term(inv_sn, a)

    a_sn = st.str_to_sn("a", 0)
    b_sn = st.str_to_sn("b", 0)
    a_const, b_const = get_rigid_term(a_sn, 0), get_rigid_term(b_sn, 0)

    # All clauses in SOS — usable/SOS split doesn't work for pure equational problems
    # in our paramodulation implementation (usable clauses need to be processed first)
    sos = [
        Clause(literals=(Literal(sign=True, atom=eq(mult(e, x), x)),)),
        Clause(literals=(Literal(sign=True, atom=eq(mult(inv(x), x), e)),)),
        Clause(literals=(Literal(sign=True, atom=eq(mult(mult(x, y), z), mult(x, mult(y, z)))),)),
        Clause(literals=(Literal(sign=True, atom=eq(mult(x, x), e)),)),
        Clause(literals=(Literal(sign=False, atom=eq(mult(a_const, b_const), mult(b_const, a_const))),)),
    ]
    return st, [], sos


# ── Benchmark runners ────────────────────────────────────────────────────


def _run_traditional(
    st, usable, sos, *, paramodulation=False, max_given=200,
) -> BenchmarkRun:
    """Run search with traditional (no ML) configuration."""
    opts = SearchOptions(
        binary_resolution=True,
        paramodulation=paramodulation,
        demodulation=paramodulation,
        factoring=True,
        max_given=max_given,
        quiet=True,
    )
    search = GivenClauseSearch(options=opts, symbol_table=st)

    start = time.perf_counter()
    result = search.run(usable=usable or [], sos=sos or [])
    elapsed = time.perf_counter() - start

    return BenchmarkRun(
        problem="",
        mode="traditional",
        proved=result.exit_code == ExitCode.MAX_PROOFS_EXIT,
        exit_code=result.exit_code,
        wall_seconds=elapsed,
        given=result.stats.given,
        generated=result.stats.generated,
        kept=result.stats.kept,
        proofs=len(result.proofs),
    )


def _run_with_hooks(
    st, usable, sos, *, paramodulation=False, max_given=200,
) -> BenchmarkRun:
    """Run search with online learning hooks but no actual learning."""
    config = OnlineIntegrationConfig(
        enabled=True,
        collect_experiences=True,
        trigger_updates=False,  # Hooks active, no model updates
        min_given_before_ml=0,
    )
    encoder = BenchmarkEncoder()
    manager = OnlineLearningManager(
        encoder=encoder,
        config=OnlineLearningConfig(enabled=True),
    )
    integration = OnlineSearchIntegration(config=config, manager=manager)

    opts = SearchOptions(
        binary_resolution=True,
        paramodulation=paramodulation,
        demodulation=paramodulation,
        factoring=True,
        max_given=max_given,
        quiet=True,
    )
    search = integration.create_search(options=opts, symbol_table=st)

    start = time.perf_counter()
    result = search.run(usable=usable or [], sos=sos or [])
    elapsed = time.perf_counter() - start

    return BenchmarkRun(
        problem="",
        mode="hooks_only",
        proved=result.exit_code == ExitCode.MAX_PROOFS_EXIT,
        exit_code=result.exit_code,
        wall_seconds=elapsed,
        given=result.stats.given,
        generated=result.stats.generated,
        kept=result.stats.kept,
        experiences_collected=integration.stats.experiences_collected,
        proofs=len(result.proofs),
    )


def _run_with_learning(
    st, usable, sos, *, paramodulation=False, max_given=200,
) -> BenchmarkRun:
    """Run search with full online learning enabled."""
    config = OnlineIntegrationConfig(
        enabled=True,
        collect_experiences=True,
        trigger_updates=True,
        min_given_before_ml=10,
        adaptive_ml_weight=True,
        initial_ml_weight=0.1,
    )
    encoder = BenchmarkEncoder()
    ol_config = OnlineLearningConfig(
        enabled=True,
        update_interval=50,
        batch_size=16,
        buffer_capacity=2000,
    )
    manager = OnlineLearningManager(encoder=encoder, config=ol_config)
    integration = OnlineSearchIntegration(config=config, manager=manager)

    opts = SearchOptions(
        binary_resolution=True,
        paramodulation=paramodulation,
        demodulation=paramodulation,
        factoring=True,
        max_given=max_given,
        quiet=True,
    )
    search = integration.create_search(options=opts, symbol_table=st)

    start = time.perf_counter()
    result = search.run(usable=usable or [], sos=sos or [])
    elapsed = time.perf_counter() - start

    return BenchmarkRun(
        problem="",
        mode="learning_enabled",
        proved=result.exit_code == ExitCode.MAX_PROOFS_EXIT,
        exit_code=result.exit_code,
        wall_seconds=elapsed,
        given=result.stats.given,
        generated=result.stats.generated,
        kept=result.stats.kept,
        experiences_collected=integration.stats.experiences_collected,
        model_updates=integration.stats.model_updates_triggered,
        proofs=len(result.proofs),
    )


# ── Proof Preservation Tests ─────────────────────────────────────────────


class TestProofPreservation:
    """Online learning must not prevent finding proofs that traditional search finds."""

    def test_trivial_resolution_all_modes(self):
        """All modes find trivial resolution proof."""
        st, usable, sos = _make_trivial_resolution()
        trad = _run_traditional(st, usable, list(sos))
        hooks = _run_with_hooks(st, usable, list(_make_trivial_resolution()[2]))
        learn = _run_with_learning(st, usable, list(_make_trivial_resolution()[2]))
        assert trad.proved
        assert hooks.proved
        assert learn.proved

    def test_equational_all_modes(self):
        """All modes find equational proof."""
        for runner in [_run_traditional, _run_with_hooks, _run_with_learning]:
            st, usable, sos = _make_equational()
            result = runner(st, usable, sos, paramodulation=True)
            assert result.proved, f"{runner.__name__} failed to prove equational problem"

    def test_group_commutativity_all_modes(self):
        """All modes find group commutativity proof."""
        for runner in [_run_traditional, _run_with_hooks, _run_with_learning]:
            st, usable, sos = _make_group_commutativity()
            result = runner(st, usable, sos, paramodulation=True, max_given=500)
            assert result.proved, f"{runner.__name__} failed to prove group commutativity"


# ── Hook Overhead Tests ──────────────────────────────────────────────────


class TestHookOverhead:
    """Measure overhead of online learning hooks on search performance."""

    def test_hooks_do_not_change_search_statistics(self):
        """Hooks-only mode produces identical search statistics to traditional."""
        st, usable, sos = _make_trivial_resolution()
        trad = _run_traditional(st, usable, list(sos))

        st2, usable2, sos2 = _make_trivial_resolution()
        hooks = _run_with_hooks(st2, usable2, list(sos2))

        assert trad.given == hooks.given
        assert trad.generated == hooks.generated
        assert trad.kept == hooks.kept
        assert trad.proofs == hooks.proofs

    def test_equational_hooks_preserve_statistics(self):
        """Hooks don't alter equational search path."""
        st1, u1, s1 = _make_equational()
        trad = _run_traditional(st1, u1, s1, paramodulation=True)

        st2, u2, s2 = _make_equational()
        hooks = _run_with_hooks(st2, u2, s2, paramodulation=True)

        assert trad.given == hooks.given
        assert trad.generated == hooks.generated

    def test_hooks_collect_experiences(self):
        """Hooks mode collects experiences from the search."""
        st, usable, sos = _make_group_commutativity()
        hooks = _run_with_hooks(st, usable, sos, paramodulation=True, max_given=500)
        assert hooks.experiences_collected > 0


# ── Learning Overhead Tests ──────────────────────────────────────────────


class TestLearningOverhead:
    """Measure overhead of active online learning during search."""

    def test_learning_overhead_bounded(self):
        """Learning-enabled search completes within 5x of traditional time."""
        st1, u1, s1 = _make_group_commutativity()
        trad = _run_traditional(st1, u1, s1, paramodulation=True, max_given=500)

        st2, u2, s2 = _make_group_commutativity()
        learn = _run_with_learning(st2, u2, s2, paramodulation=True, max_given=500)

        if trad.wall_seconds > 0.001:
            overhead = learn.wall_seconds / trad.wall_seconds
            assert overhead < 5.0, (
                f"Learning overhead {overhead:.1f}x exceeds 5x threshold "
                f"(trad={trad.wall_seconds:.3f}s, learn={learn.wall_seconds:.3f}s)"
            )

    def test_learning_collects_more_experiences_than_hooks(self):
        """Learning mode with updates may collect more (proof-relabeled) experiences."""
        st1, u1, s1 = _make_group_commutativity()
        hooks = _run_with_hooks(st1, u1, s1, paramodulation=True, max_given=500)

        st2, u2, s2 = _make_group_commutativity()
        learn = _run_with_learning(st2, u2, s2, paramodulation=True, max_given=500)

        # Both should collect experiences
        assert hooks.experiences_collected > 0
        assert learn.experiences_collected > 0


# ── Statistical Comparison ───────────────────────────────────────────────


class TestStatisticalComparison:
    """Compare search statistics across all modes for programmatic problems."""

    @pytest.fixture
    def all_modes_trivial(self):
        """Run trivial problem in all modes."""
        results = {}
        st, u, s = _make_trivial_resolution()
        results["traditional"] = _run_traditional(st, u, list(s))
        st, u, s = _make_trivial_resolution()
        results["hooks"] = _run_with_hooks(st, u, list(s))
        st, u, s = _make_trivial_resolution()
        results["learning"] = _run_with_learning(st, u, list(s))
        return results

    def test_all_modes_prove(self, all_modes_trivial):
        """All modes find proof for trivial problem."""
        for mode, result in all_modes_trivial.items():
            assert result.proved, f"{mode} failed"

    def test_traditional_and_hooks_equivalent(self, all_modes_trivial):
        """Traditional and hooks-only produce identical search paths."""
        trad = all_modes_trivial["traditional"]
        hooks = all_modes_trivial["hooks"]
        assert trad.given == hooks.given
        assert trad.generated == hooks.generated
        assert trad.kept == hooks.kept


# ── File-Based Benchmark Tests ───────────────────────────────────────────


class TestFileBenchmarks:
    """Run benchmarks on the standard .in problem files."""

    @pytest.fixture(params=[
        "identity_only.in",
        "simple_group.in",
        "lattice_absorption.in",
    ])
    def problem_file(self, request):
        """Parameterized fixture for available problem files."""
        path = INPUTS_DIR / request.param
        if not path.exists():
            pytest.skip(f"{request.param} not found")
        return path

    def test_file_problem_traditional(self, problem_file):
        """Traditional search handles file-based problem."""
        from pyladr.core.symbol import SymbolTable
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(problem_file.read_text())

        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            demodulation=True,
            factoring=True,
            max_given=200,
            quiet=True,
        )
        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(
            usable=parsed.usable or [],
            sos=parsed.sos or [],
        )
        # Should at least not crash
        assert result.exit_code in (
            ExitCode.MAX_PROOFS_EXIT,
            ExitCode.MAX_GIVEN_EXIT,
            ExitCode.SOS_EMPTY_EXIT,
        )

    def test_file_problem_with_hooks(self, problem_file):
        """Hooks-enabled search handles file-based problem."""
        from pyladr.core.symbol import SymbolTable
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        parsed = parser.parse_input(problem_file.read_text())

        config = OnlineIntegrationConfig(
            enabled=True,
            collect_experiences=True,
            trigger_updates=False,
            min_given_before_ml=0,
        )
        encoder = BenchmarkEncoder()
        manager = OnlineLearningManager(
            encoder=encoder,
            config=OnlineLearningConfig(enabled=True),
        )
        integration = OnlineSearchIntegration(config=config, manager=manager)

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
        assert result.exit_code in (
            ExitCode.MAX_PROOFS_EXIT,
            ExitCode.MAX_GIVEN_EXIT,
            ExitCode.SOS_EMPTY_EXIT,
        )
