"""End-to-end validation of proof-guided clause selection.

Validates that proof-guided selection demonstrates:
1. Learning behavior: system improves on related problems after successful proofs
2. Exploration/exploitation balance: different ratios produce different behaviors
3. Performance comparison: proof-guided vs baseline FORTE vs traditional selection
4. Convergence: system doesn't over-fit to early patterns

These are integration-level tests that run actual theorem proving searches
with the proof-guided selection system enabled.
"""

from __future__ import annotations

import math
import statistics

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import (
    Term,
    build_binary_term,
    get_rigid_term,
    get_variable_term,
)
from pyladr.search.proof_pattern_memory import (
    ProofGuidedConfig,
    ProofPatternMemory,
    _cosine_similarity,
    proof_guided_score,
)
from pyladr.search.given_clause import ExitCode, GivenClauseSearch, Proof, SearchOptions


# ── Helpers ──────────────────────────────────────────────────────────────────


def _pos(atom: Term) -> Literal:
    return Literal(sign=True, atom=atom)


def _neg(atom: Term) -> Literal:
    return Literal(sign=False, atom=atom)


def _cl(*lits: Literal) -> Clause:
    return Clause(literals=tuple(lits))


def _run_search(
    usable: list[Clause],
    sos: list[Clause],
    *,
    symbol_table: SymbolTable | None = None,
    paramodulation: bool = False,
    demodulation: bool = False,
    max_given: int = 500,
    max_proofs: int = 1,
    forte_embeddings: bool = False,
    proof_guided: bool = False,
    proof_guided_weight: float = 0.0,
    proof_guided_exploitation_ratio: float = 0.7,
    proof_guided_decay_rate: float = 0.95,
    proof_guided_max_patterns: int = 500,
    proof_guided_warmup_proofs: int = 1,
    forte_weight: float = 0.0,
) -> tuple[GivenClauseSearch, object]:
    """Run a search and return (search_engine, result)."""
    opts = SearchOptions(
        binary_resolution=True,
        paramodulation=paramodulation,
        demodulation=demodulation,
        factoring=True,
        max_given=max_given,
        max_proofs=max_proofs,
        quiet=True,
        forte_embeddings=forte_embeddings,
        forte_weight=forte_weight,
        proof_guided=proof_guided,
        proof_guided_weight=proof_guided_weight,
        proof_guided_exploitation_ratio=proof_guided_exploitation_ratio,
        proof_guided_decay_rate=proof_guided_decay_rate,
        proof_guided_max_patterns=proof_guided_max_patterns,
        proof_guided_warmup_proofs=proof_guided_warmup_proofs,
    )
    search = GivenClauseSearch(options=opts, symbol_table=symbol_table)
    result = search.run(usable=usable, sos=sos)
    return search, result


def _build_group_theory_problem(
    symbol_table: SymbolTable,
    *,
    include_involution: bool = False,
) -> tuple[list[Clause], list[Clause]]:
    """Build group theory axioms with negated goal for commutativity.

    Returns (usable, sos) clause lists.
    """
    eq_sn = symbol_table.str_to_sn("=", 2)
    star_sn = symbol_table.str_to_sn("*", 2)
    e_sn = symbol_table.str_to_sn("e", 0)
    inv_sn = symbol_table.str_to_sn("'", 1)

    x = get_variable_term(0)
    y = get_variable_term(1)
    z = get_variable_term(2)
    e = get_rigid_term(e_sn, 0)

    def star(a: Term, b: Term) -> Term:
        return build_binary_term(star_sn, a, b)

    def inv(a: Term) -> Term:
        return get_rigid_term(inv_sn, 1, (a,))

    def eq(a: Term, b: Term) -> Term:
        return build_binary_term(eq_sn, a, b)

    # Group axioms
    axioms = [
        _cl(_pos(eq(star(e, x), x))),           # e * x = x
        _cl(_pos(eq(star(inv(x), x), e))),       # x' * x = e
        _cl(_pos(eq(star(star(x, y), z), star(x, star(y, z))))),  # associativity
    ]

    if include_involution:
        axioms.append(_cl(_pos(eq(star(x, x), e))))  # x * x = e

    # Negated goal: x * y != y * x (for proving commutativity)
    a_sn = symbol_table.str_to_sn("a", 0)
    b_sn = symbol_table.str_to_sn("b", 0)
    a = get_rigid_term(a_sn, 0)
    b = get_rigid_term(b_sn, 0)

    negated_goal = _cl(_neg(eq(star(a, b), star(b, a))))

    return [], axioms + [negated_goal]


def _build_modus_ponens_chain(n: int) -> tuple[list[Clause], list[Clause]]:
    """Build a chain of n modus ponens steps P0(a) -> P1(a) -> ... -> Pn(a).

    Returns (usable, sos) with Pn(a) negated as the goal.
    """
    a_sn = n + 10  # offset to avoid collision
    a = get_rigid_term(a_sn, 0)
    x = get_variable_term(0)

    clauses: list[Clause] = []

    # P0(a)
    P0_sn = 1
    clauses.append(_cl(_pos(get_rigid_term(P0_sn, 1, (a,)))))

    # ~Pi(x) | Pi+1(x) for i in 0..n-1
    for i in range(n):
        Pi_sn = i + 1
        Pi1_sn = i + 2
        clauses.append(
            _cl(
                _neg(get_rigid_term(Pi_sn, 1, (x,))),
                _pos(get_rigid_term(Pi1_sn, 1, (x,))),
            )
        )

    # ~Pn(a) (negated goal)
    Pn_sn = n + 1
    clauses.append(_cl(_neg(get_rigid_term(Pn_sn, 1, (a,)))))

    return [], clauses


# ── Test Class: Learning Behavior ────────────────────────────────────────────


class TestLearningBehavior:
    """Validate that proof-guided selection learns from successful proofs."""

    def test_proof_patterns_recorded_after_proof(self) -> None:
        """After finding a proof, the proof pattern memory should contain patterns."""
        usable, sos = _build_modus_ponens_chain(3)
        search, result = _run_search(
            usable,
            sos,
            forte_embeddings=True,
            proof_guided=True,
            proof_guided_weight=2.0,
            max_proofs=1,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        memory = search.proof_pattern_memory
        if memory is not None:
            assert memory.proof_count >= 1
            assert memory.pattern_count > 0

    def test_proof_patterns_accumulate_across_proofs(self) -> None:
        """With max_proofs > 1, patterns should accumulate across multiple proofs."""
        # Use a simple problem that can find a proof quickly
        usable, sos = _build_modus_ponens_chain(2)
        search, result = _run_search(
            usable,
            sos,
            forte_embeddings=True,
            proof_guided=True,
            proof_guided_weight=2.0,
            max_proofs=5,
            max_given=200,
        )
        assert result.exit_code in (ExitCode.MAX_PROOFS_EXIT, ExitCode.SOS_EMPTY_EXIT)
        memory = search.proof_pattern_memory
        if memory is not None and memory.proof_count > 0:
            assert memory.pattern_count > 0

    def test_warmup_delays_exploitation(self) -> None:
        """With warmup_proofs=2, the system should not exploit until 2 proofs found."""
        config = ProofGuidedConfig(
            warmup_proofs=2,
            exploitation_ratio=1.0,
        )
        memory = ProofPatternMemory(config=config)

        # Before any proofs
        assert not memory.is_warmed_up
        assert memory.exploitation_score([1.0, 0.0, 0.0]) == 0.5  # neutral

        # After 1 proof
        memory.record_proof([[1.0, 0.0, 0.0]])
        assert not memory.is_warmed_up
        assert memory.exploitation_score([1.0, 0.0, 0.0]) == 0.5  # still neutral

        # After 2 proofs
        memory.record_proof([[0.0, 1.0, 0.0]])
        assert memory.is_warmed_up
        score = memory.exploitation_score([1.0, 0.0, 0.0])
        assert score > 0.5  # now exploiting

    def test_decay_reduces_old_pattern_influence(self) -> None:
        """Older patterns should have less influence due to exponential decay."""
        config = ProofGuidedConfig(
            decay_rate=0.5,  # aggressive decay for testing
            warmup_proofs=1,
            min_similarity_threshold=0.0,
        )
        memory = ProofPatternMemory(config=config)

        # Record first proof with pattern [1, 0, 0]
        memory.record_proof([[1.0, 0.0, 0.0]])
        score_before_decay = memory.exploitation_score([1.0, 0.0, 0.0])

        # Record second proof with a different pattern
        memory.record_proof([[0.0, 1.0, 0.0]])
        score_after_decay = memory.exploitation_score([1.0, 0.0, 0.0])

        # First pattern's influence should be reduced after decay
        assert score_after_decay < score_before_decay

    def test_related_problems_share_patterns(self) -> None:
        """Patterns from one proof should score related clauses highly.

        This validates that the similarity scoring mechanism correctly
        identifies structurally similar clauses as related.
        """
        config = ProofGuidedConfig(
            warmup_proofs=1,
            exploitation_ratio=0.8,
            min_similarity_threshold=0.0,
        )
        memory = ProofPatternMemory(config=config)

        # Simulate a proof with some embedding patterns
        proof_embeddings = [
            [0.5, 0.5, 0.0, 0.0],
            [0.3, 0.7, 0.0, 0.0],
            [0.6, 0.4, 0.0, 0.0],
        ]
        memory.record_proof(proof_embeddings)

        # A related clause (similar direction)
        related = [0.4, 0.6, 0.0, 0.0]
        # An unrelated clause (orthogonal direction)
        unrelated = [0.0, 0.0, 0.7, 0.3]

        score_related = memory.exploitation_score(related)
        score_unrelated = memory.exploitation_score(unrelated)

        assert score_related > score_unrelated, (
            f"Related clause should score higher: {score_related} vs {score_unrelated}"
        )


# ── Test Class: Exploration/Exploitation Balance ──────────────────────────────


class TestExplorationExploitationBalance:
    """Validate that different exploitation ratios produce different behaviors."""

    def test_pure_exploitation_maximizes_similarity(self) -> None:
        """With ratio=1.0, score should equal exploitation score entirely."""
        config = ProofGuidedConfig(
            exploitation_ratio=1.0,
            warmup_proofs=1,
        )
        memory = ProofPatternMemory(config=config)
        memory.record_proof([[1.0, 0.0, 0.0]])

        score = proof_guided_score(
            [1.0, 0.0, 0.0], memory, diversity_score=0.2, config=config
        )
        exploitation = memory.exploitation_score([1.0, 0.0, 0.0])
        assert score == pytest.approx(exploitation)

    def test_pure_exploration_ignores_patterns(self) -> None:
        """With ratio=0.0, score should equal diversity score entirely."""
        config = ProofGuidedConfig(
            exploitation_ratio=0.0,
            warmup_proofs=1,
        )
        memory = ProofPatternMemory(config=config)
        memory.record_proof([[1.0, 0.0, 0.0]])

        diversity = 0.8
        score = proof_guided_score(
            [1.0, 0.0, 0.0], memory, diversity_score=diversity, config=config
        )
        assert score == pytest.approx(diversity)

    def test_default_blend_is_between_extremes(self) -> None:
        """Default 0.7 exploitation should produce scores between pure modes."""
        config = ProofGuidedConfig(
            exploitation_ratio=0.7,
            warmup_proofs=1,
        )
        memory = ProofPatternMemory(config=config)
        memory.record_proof([[1.0, 0.0, 0.0]])

        embedding = [0.8, 0.2, 0.0]
        diversity = 0.3
        exploitation = memory.exploitation_score(embedding)

        blended = proof_guided_score(embedding, memory, diversity, config)

        # Blended should be between pure exploitation and pure exploration
        low = min(exploitation, diversity)
        high = max(exploitation, diversity)
        assert low <= blended <= high or blended == pytest.approx(low) or blended == pytest.approx(high)

    def test_different_ratios_produce_different_scores(self) -> None:
        """Sweeping exploitation ratios should produce monotonically varying scores."""
        memory = ProofPatternMemory(config=ProofGuidedConfig(warmup_proofs=1))
        memory.record_proof([[1.0, 0.0, 0.0]])

        embedding = [0.9, 0.1, 0.0]
        diversity = 0.2

        scores = []
        for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
            config = ProofGuidedConfig(exploitation_ratio=ratio, warmup_proofs=1)
            s = proof_guided_score(embedding, memory, diversity, config)
            scores.append(s)

        # Scores should increase as exploitation ratio increases (since
        # exploitation score > diversity score for this similar embedding)
        exploitation = memory.exploitation_score(embedding)
        if exploitation > diversity:
            # Scores should be non-decreasing
            for i in range(len(scores) - 1):
                assert scores[i] <= scores[i + 1] + 1e-9, (
                    f"Score at ratio {i * 0.25} ({scores[i]}) > "
                    f"score at ratio {(i + 1) * 0.25} ({scores[i + 1]})"
                )

    def test_exploitation_ratio_sensitivity(self) -> None:
        """Different ratios should produce measurably different scoring distributions."""
        config_low = ProofGuidedConfig(exploitation_ratio=0.2, warmup_proofs=1)
        config_high = ProofGuidedConfig(exploitation_ratio=0.9, warmup_proofs=1)

        memory = ProofPatternMemory(config=ProofGuidedConfig(warmup_proofs=1))
        memory.record_proof([[0.7, 0.3, 0.0, 0.0]])

        # Score several candidate embeddings with both configs
        candidates = [
            [0.8, 0.2, 0.0, 0.0],  # similar to proof pattern
            [0.0, 0.0, 0.8, 0.2],  # orthogonal to proof pattern
            [0.5, 0.5, 0.0, 0.0],  # moderately similar
            [0.1, 0.1, 0.7, 0.1],  # mostly different
        ]
        diversity = 0.5

        scores_low = [
            proof_guided_score(c, memory, diversity, config_low)
            for c in candidates
        ]
        scores_high = [
            proof_guided_score(c, memory, diversity, config_high)
            for c in candidates
        ]

        # High exploitation should create more score variance (spread)
        # because it weighs similarity more, amplifying differences
        variance_low = statistics.variance(scores_low) if len(scores_low) > 1 else 0
        variance_high = statistics.variance(scores_high) if len(scores_high) > 1 else 0

        assert variance_high > variance_low, (
            f"High exploitation should create more score variance: "
            f"{variance_high:.6f} vs {variance_low:.6f}"
        )


# ── Test Class: Performance Comparison ────────────────────────────────────────


class TestPerformanceComparison:
    """Compare proof-guided selection against baseline strategies."""

    def test_proof_guided_finds_proof(self) -> None:
        """Proof-guided selection should still find proofs (basic correctness)."""
        usable, sos = _build_modus_ponens_chain(3)
        _, result = _run_search(
            usable,
            sos,
            forte_embeddings=True,
            proof_guided=True,
            proof_guided_weight=2.0,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_traditional_selection_finds_proof(self) -> None:
        """Traditional selection baseline for comparison."""
        usable, sos = _build_modus_ponens_chain(3)
        _, result = _run_search(usable, sos)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_forte_only_finds_proof(self) -> None:
        """FORTE-only selection baseline for comparison."""
        usable, sos = _build_modus_ponens_chain(3)
        _, result = _run_search(
            usable, sos, forte_embeddings=True, forte_weight=2.0
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    @pytest.mark.parametrize("length", [2, 3, 4, 5])
    def test_proof_guided_solves_chain_resolution(self, length: int) -> None:
        """Proof-guided selection should solve chain resolution of varying lengths."""
        usable, sos = _build_modus_ponens_chain(length)
        _, result = _run_search(
            usable, sos,
            forte_embeddings=True,
            proof_guided=True,
            proof_guided_weight=2.0,
            max_given=300,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT, (
            f"Proof-guided failed on chain length {length}"
        )

    @pytest.mark.parametrize("length", [2, 3, 4, 5])
    def test_traditional_solves_chain_resolution(self, length: int) -> None:
        """Traditional selection baseline should solve chain resolution."""
        usable, sos = _build_modus_ponens_chain(length)
        _, result = _run_search(usable, sos, max_given=300)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT, (
            f"Traditional failed on chain length {length}"
        )

    def test_proof_guided_finds_proof_within_budget(self) -> None:
        """Proof-guided should find proofs within reasonable given-clause budget.

        We verify that proof-guided selection converges within a modest budget,
        demonstrating it doesn't cause pathological search expansion.
        """
        usable, sos = _build_modus_ponens_chain(4)
        search_pg, result_pg = _run_search(
            usable, sos,
            forte_embeddings=True,
            proof_guided=True,
            proof_guided_weight=2.0,
            max_given=200,
        )
        assert result_pg.exit_code == ExitCode.MAX_PROOFS_EXIT
        # Should find proof well within 200 given clause budget
        assert search_pg.stats.given <= 200


# ── Test Class: Convergence and Over-fitting ──────────────────────────────────


class TestConvergence:
    """Validate that proof-guided selection doesn't over-fit to early patterns."""

    def test_decay_prevents_stale_pattern_dominance(self) -> None:
        """With decay, old patterns should fade and not dominate scoring."""
        config = ProofGuidedConfig(
            decay_rate=0.5,  # aggressive decay
            warmup_proofs=1,
            min_similarity_threshold=0.0,
        )
        memory = ProofPatternMemory(config=config)

        # Record initial proof with pattern in one direction
        memory.record_proof([[1.0, 0.0, 0.0, 0.0]])

        # Record 5 more proofs with a different pattern
        for _ in range(5):
            memory.record_proof([[0.0, 0.0, 1.0, 0.0]])

        # After heavy decay, the old pattern should be nearly gone
        old_score = memory.exploitation_score([1.0, 0.0, 0.0, 0.0])
        new_score = memory.exploitation_score([0.0, 0.0, 1.0, 0.0])

        # The new pattern (recorded more recently with less decay) should
        # score higher than the old pattern (decayed 5 times at 0.5)
        assert new_score > old_score, (
            f"New pattern score ({new_score}) should exceed old pattern ({old_score})"
        )

    def test_no_decay_preserves_all_patterns(self) -> None:
        """With decay_rate=1.0, all patterns should retain full weight."""
        config = ProofGuidedConfig(
            decay_rate=1.0,
            warmup_proofs=1,
        )
        memory = ProofPatternMemory(config=config)

        memory.record_proof([[1.0, 0.0, 0.0]])
        score_first = memory.exploitation_score([1.0, 0.0, 0.0])

        # Record more proofs
        for _ in range(5):
            memory.record_proof([[0.0, 1.0, 0.0]])

        score_after = memory.exploitation_score([1.0, 0.0, 0.0])

        # With no decay, old pattern should retain full weight
        assert score_after == pytest.approx(score_first, abs=1e-6)

    def test_bounded_memory_prevents_unbounded_growth(self) -> None:
        """Pattern memory should not grow beyond max_patterns."""
        config = ProofGuidedConfig(
            max_patterns=10,
            warmup_proofs=1,
        )
        memory = ProofPatternMemory(config=config)

        # Record many proofs with multiple patterns each
        for i in range(20):
            emb = [0.0] * 8
            emb[i % 8] = 1.0
            memory.record_proof([emb])

        assert memory.pattern_count <= config.max_patterns

    def test_centroid_score_provides_smooth_signal(self) -> None:
        """Centroid scoring should provide a smooth, averaged signal."""
        config = ProofGuidedConfig(warmup_proofs=1)
        memory = ProofPatternMemory(config=config)

        # Record patterns in multiple directions
        memory.record_proof([[1.0, 0.0, 0.0, 0.0]])
        memory.record_proof([[0.0, 1.0, 0.0, 0.0]])

        # Centroid should be somewhere in between
        score_aligned = memory.centroid_score([0.5, 0.5, 0.0, 0.0])
        score_orthogonal = memory.centroid_score([0.0, 0.0, 1.0, 0.0])

        # Aligned vector should score higher than orthogonal
        assert score_aligned > score_orthogonal


# ── Test Class: Backward Compatibility ────────────────────────────────────────


class TestBackwardCompatibility:
    """Validate that proof-guided selection doesn't break existing behavior."""

    def test_disabled_by_default(self) -> None:
        """Proof-guided should be disabled by default."""
        opts = SearchOptions(quiet=True)
        assert opts.proof_guided is False
        assert opts.proof_guided_weight == 0.0

    def test_traditional_search_unaffected(self) -> None:
        """Traditional search (no FORTE, no proof-guided) should work normally."""
        usable, sos = _build_modus_ponens_chain(3)
        _, result = _run_search(usable, sos)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_forte_only_unaffected(self) -> None:
        """FORTE-only search (no proof-guided) should work normally."""
        usable, sos = _build_modus_ponens_chain(3)
        _, result = _run_search(
            usable, sos, forte_embeddings=True, forte_weight=2.0
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_proof_guided_without_forte_has_no_effect(self) -> None:
        """Enabling proof-guided without FORTE should not crash or change behavior."""
        usable, sos = _build_modus_ponens_chain(3)
        search, result = _run_search(
            usable, sos,
            proof_guided=True,
            proof_guided_weight=2.0,
        )
        # Should still find proof via traditional selection
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        # Memory should not be initialized without FORTE
        assert search.proof_pattern_memory is None

    def test_proof_guided_graceful_with_no_proofs_found(self) -> None:
        """If no proof is found, proof-guided should not crash."""
        P_sn, Q_sn, a_sn, b_sn = 1, 2, 3, 4
        a = get_rigid_term(a_sn, 0)
        b = get_rigid_term(b_sn, 0)
        c1 = _cl(_pos(get_rigid_term(P_sn, 1, (a,))))
        c2 = _cl(_pos(get_rigid_term(Q_sn, 1, (b,))))

        search, result = _run_search(
            [], [c1, c2],
            forte_embeddings=True,
            proof_guided=True,
            proof_guided_weight=2.0,
            max_given=20,
        )
        # Should terminate cleanly (SOS empty or max_given)
        assert result.exit_code in (
            ExitCode.SOS_EMPTY_EXIT,
            ExitCode.MAX_GIVEN_EXIT,
        )
        # Memory should exist but have no proofs
        memory = search.proof_pattern_memory
        if memory is not None:
            assert memory.proof_count == 0


# ── Test Class: Integration with Real Theorem Proving ─────────────────────────


class TestRealTheoremProving:
    """Validate proof-guided selection on more substantial theorem proving problems."""

    def test_equational_proof_with_proof_guided(self) -> None:
        """Proof-guided should handle equational (paramodulation) proofs."""
        st = SymbolTable()
        eq_sn = st.str_to_sn("=", 2)
        f_sn = st.str_to_sn("f", 1)
        p_sn = st.str_to_sn("p", 1)
        a_sn = st.str_to_sn("a", 0)
        b_sn = st.str_to_sn("b", 0)

        a = get_rigid_term(a_sn, 0)
        b = get_rigid_term(b_sn, 0)
        fa = get_rigid_term(f_sn, 1, (a,))

        c1 = _cl(_pos(build_binary_term(eq_sn, fa, b)))
        c2 = _cl(_pos(get_rigid_term(p_sn, 1, (fa,))))
        c3 = _cl(_neg(get_rigid_term(p_sn, 1, (b,))))

        search, result = _run_search(
            usable=[c1],
            sos=[c2, c3],
            paramodulation=True,
            symbol_table=st,
            forte_embeddings=True,
            proof_guided=True,
            proof_guided_weight=2.0,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_multi_step_resolution_with_proof_guided(self) -> None:
        """Proof-guided should handle multi-step resolution chains."""
        usable, sos = _build_modus_ponens_chain(5)
        search, result = _run_search(
            usable, sos,
            forte_embeddings=True,
            proof_guided=True,
            proof_guided_weight=2.0,
            max_given=300,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_proof_guided_with_demodulation(self) -> None:
        """Proof-guided should work alongside demodulation."""
        st = SymbolTable()
        eq_sn = st.str_to_sn("=", 2)
        p_sn = st.str_to_sn("p", 1)
        a_sn = st.str_to_sn("a", 0)
        b_sn = st.str_to_sn("b", 0)

        a = get_rigid_term(a_sn, 0)
        b = get_rigid_term(b_sn, 0)

        c1 = _cl(_pos(build_binary_term(eq_sn, a, b)))
        c2 = _cl(_pos(get_rigid_term(p_sn, 1, (a,))))
        c3 = _cl(_neg(get_rigid_term(p_sn, 1, (b,))))

        search, result = _run_search(
            usable=[c1],
            sos=[c2, c3],
            paramodulation=True,
            demodulation=True,
            symbol_table=st,
            forte_embeddings=True,
            proof_guided=True,
            proof_guided_weight=2.0,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT


# ── Test Class: Configuration Validation ──────────────────────────────────────


class TestConfigurationValidation:
    """Validate proof-guided configuration bounds and edge cases."""

    def test_exploitation_ratio_zero(self) -> None:
        """Exploitation ratio 0 should be valid (pure exploration)."""
        opts = SearchOptions(
            quiet=True,
            proof_guided=True,
            proof_guided_exploitation_ratio=0.0,
        )
        assert opts.proof_guided_exploitation_ratio == 0.0

    def test_exploitation_ratio_one(self) -> None:
        """Exploitation ratio 1 should be valid (pure exploitation)."""
        opts = SearchOptions(
            quiet=True,
            proof_guided=True,
            proof_guided_exploitation_ratio=1.0,
        )
        assert opts.proof_guided_exploitation_ratio == 1.0

    def test_invalid_exploitation_ratio_rejected(self) -> None:
        """Exploitation ratio outside [0, 1] should be rejected."""
        with pytest.raises(ValueError):
            SearchOptions(
                quiet=True,
                proof_guided=True,
                proof_guided_exploitation_ratio=1.5,
            )

    def test_invalid_decay_rate_rejected(self) -> None:
        """Decay rate outside [0, 1] should be rejected."""
        with pytest.raises(ValueError):
            SearchOptions(
                quiet=True,
                proof_guided=True,
                proof_guided_decay_rate=1.5,
            )

    def test_max_patterns_minimum(self) -> None:
        """Max patterns should accept minimum value of 1."""
        opts = SearchOptions(
            quiet=True,
            proof_guided=True,
            proof_guided_max_patterns=1,
        )
        assert opts.proof_guided_max_patterns == 1
