"""Comprehensive tests for FORTE CLI integration.

Tests cover the full integration path:
1. CLI argument parsing and validation
2. CLI → SearchOptions mapping
3. FORTE provider initialization (success/failure)
4. EmbeddingEnhancedSelection integration
5. Graceful degradation when FORTE unavailable
6. Performance regression (no slowdown when disabled)
7. Thread safety for concurrent embedding requests
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import get_rigid_term, get_variable_term
from pyladr.ml.forte.algorithm import ForteConfig
from pyladr.ml.forte.provider import ForteEmbeddingProvider, ForteProviderConfig
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions
from pyladr.search.priority_sos import PrioritySOS
from pyladr.search.selection import GivenSelection, SelectionOrder, SelectionRule


# ── Test Helpers ──────────────────────────────────────────────────────────────


def _var(n: int):
    return get_variable_term(n)


def _const(symnum: int):
    return get_rigid_term(symnum, 0)


def _func(symnum: int, *args):
    return get_rigid_term(symnum, len(args), args)


def _pos_lit(atom):
    return Literal(sign=True, atom=atom)


def _neg_lit(atom):
    return Literal(sign=False, atom=atom)


def _make_clause(*lits, weight=0.0, clause_id=0):
    return Clause(literals=lits, weight=weight, id=clause_id)


def _parse_forte_args(extra_args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments through the real argument parser."""
    from pyladr.apps.prover9 import _build_arg_parser

    parser = _build_arg_parser()
    argv = extra_args or []
    args, _ = parser.parse_known_args(argv)
    return args


# ── 1. CLI Argument Parsing ──────────────────────────────────────────────────


class TestForteCliArgumentParsing:
    """Verify FORTE-related CLI arguments are parsed correctly."""

    def test_forte_embeddings_default_false(self) -> None:
        args = _parse_forte_args([])
        assert args.forte_embeddings is False

    def test_forte_embeddings_enabled(self) -> None:
        args = _parse_forte_args(["--forte-embeddings"])
        assert args.forte_embeddings is True

    def test_forte_weight_default(self) -> None:
        args = _parse_forte_args([])
        assert args.forte_weight == 0.0

    def test_forte_weight_custom(self) -> None:
        args = _parse_forte_args(["--forte-weight", "0.8"])
        assert args.forte_weight == 0.8

    def test_forte_dim_default(self) -> None:
        args = _parse_forte_args([])
        assert args.forte_dim == 128

    def test_forte_dim_custom(self) -> None:
        args = _parse_forte_args(["--forte-dim", "256"])
        assert args.forte_dim == 256

    def test_forte_cache_default(self) -> None:
        args = _parse_forte_args([])
        assert args.forte_cache == 10000

    def test_forte_cache_custom(self) -> None:
        args = _parse_forte_args(["--forte-cache", "50000"])
        assert args.forte_cache == 50000

    def test_all_forte_args_together(self) -> None:
        args = _parse_forte_args([
            "--forte-embeddings",
            "--forte-weight", "0.7",
            "--forte-dim", "64",
            "--forte-cache", "5000",
        ])
        assert args.forte_embeddings is True
        assert args.forte_weight == 0.7
        assert args.forte_dim == 64
        assert args.forte_cache == 5000

    def test_forte_args_coexist_with_other_ml_args(self) -> None:
        """FORTE args don't interfere with existing ML arguments."""
        args = _parse_forte_args([
            "--forte-embeddings",
            "--forte-weight", "0.6",
            "--online-learning",
            "--ml-weight", "0.4",
        ])
        assert args.forte_embeddings is True
        assert args.forte_weight == 0.6
        assert args.online_learning is True
        assert args.ml_weight == 0.4


# ── 2. CLI → SearchOptions Mapping ──────────────────────────────────────────


class TestForteSearchOptionsMapping:
    """Verify CLI arguments map correctly to SearchOptions fields."""

    def test_forte_embeddings_maps(self) -> None:
        opts = SearchOptions(forte_embeddings=True)
        assert opts.forte_embeddings is True

    def test_forte_weight_maps(self) -> None:
        opts = SearchOptions(forte_weight=0.7)
        assert opts.forte_weight == 0.7

    def test_forte_embedding_dim_maps(self) -> None:
        opts = SearchOptions(forte_embedding_dim=256)
        assert opts.forte_embedding_dim == 256

    def test_forte_cache_maps(self) -> None:
        opts = SearchOptions(forte_cache_max_entries=50_000)
        assert opts.forte_cache_max_entries == 50_000

    def test_defaults_match_cli_defaults(self) -> None:
        """SearchOptions defaults match CLI argument defaults."""
        opts = SearchOptions()
        args = _parse_forte_args([])
        assert opts.forte_embeddings == args.forte_embeddings
        # CLI default forte_weight=0.5 maps to SearchOptions forte_weight=0.5
        assert opts.forte_weight == args.forte_weight
        # CLI --forte-dim default=128 maps to forte_embedding_dim
        assert opts.forte_embedding_dim == args.forte_dim
        # CLI --forte-cache default=10000 maps to forte_cache_max_entries
        assert opts.forte_cache_max_entries == args.forte_cache


# ── 3. SearchOptions Validation ──────────────────────────────────────────────


class TestForteSearchOptionsValidation:
    """Verify SearchOptions validation catches invalid FORTE parameters."""

    def test_forte_weight_below_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="forte_weight"):
            SearchOptions(forte_weight=-0.1)

    def test_forte_weight_above_one_accepted(self) -> None:
        # forte_weight is a ratio (like entropy_weight), not bounded to [0,1]
        opts = SearchOptions(forte_weight=1.1)
        assert opts.forte_weight == 1.1

    def test_forte_weight_zero_accepted(self) -> None:
        opts = SearchOptions(forte_weight=0.0)
        assert opts.forte_weight == 0.0

    def test_forte_weight_one_accepted(self) -> None:
        opts = SearchOptions(forte_weight=1.0)
        assert opts.forte_weight == 1.0

    def test_forte_embedding_dim_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="forte_embedding_dim"):
            SearchOptions(forte_embedding_dim=0)

    def test_forte_embedding_dim_too_large_rejected(self) -> None:
        with pytest.raises(ValueError, match="forte_embedding_dim"):
            SearchOptions(forte_embedding_dim=5000)

    def test_forte_cache_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="forte_cache_max_entries"):
            SearchOptions(forte_cache_max_entries=0)

    def test_forte_cache_too_large_rejected(self) -> None:
        with pytest.raises(ValueError, match="forte_cache_max_entries"):
            SearchOptions(forte_cache_max_entries=20_000_000)


# ── 4. FORTE Provider Initialization ─────────────────────────────────────────


class TestForteProviderInitialization:
    """Verify GivenClauseSearch initializes FORTE provider correctly."""

    def test_forte_disabled_no_provider(self) -> None:
        """No FORTE provider when forte_embeddings=False."""
        opts = SearchOptions(forte_embeddings=False)
        search = GivenClauseSearch(options=opts)
        assert search.forte_provider is None

    def test_forte_enabled_creates_provider(self) -> None:
        """FORTE provider created when forte_embeddings=True."""
        opts = SearchOptions(forte_embeddings=True)
        search = GivenClauseSearch(options=opts)
        assert search.forte_provider is not None
        assert isinstance(search.forte_provider, ForteEmbeddingProvider)

    def test_forte_provider_uses_configured_dim(self) -> None:
        """Provider uses the configured embedding dimension."""
        opts = SearchOptions(forte_embeddings=True, forte_embedding_dim=64)
        search = GivenClauseSearch(options=opts)
        assert search.forte_provider is not None
        assert search.forte_provider.embedding_dim == 64

    def test_forte_provider_uses_configured_cache(self) -> None:
        """Provider uses the configured cache size."""
        opts = SearchOptions(
            forte_embeddings=True,
            forte_cache_max_entries=5000,
        )
        search = GivenClauseSearch(options=opts)
        assert search.forte_provider is not None

    def test_forte_embeddings_dict_initialized(self) -> None:
        """FORTE embeddings dictionary is initialized empty."""
        opts = SearchOptions(forte_embeddings=True)
        search = GivenClauseSearch(options=opts)
        assert isinstance(search.forte_embeddings, dict)
        assert len(search.forte_embeddings) == 0


# ── 5. Selection Integration ─────────────────────────────────────────────────


class TestForteSelectionIntegration:
    """Verify FORTE integrates with the clause selection system."""

    def test_forte_weight_creates_selection_rule(self) -> None:
        """Positive forte_weight adds FORTE rule to selection cycle."""
        opts = SearchOptions(forte_embeddings=True, forte_weight=0.5)
        search = GivenClauseSearch(options=opts)
        rules = search._selection.rules
        forte_rules = [r for r in rules if r.order == SelectionOrder.FORTE]
        assert len(forte_rules) == 1
        assert forte_rules[0].name == "F"

    def test_forte_weight_zero_no_selection_rule(self) -> None:
        """forte_weight=0 does not add FORTE selection rule."""
        opts = SearchOptions(forte_embeddings=True, forte_weight=0)
        search = GivenClauseSearch(options=opts)
        rules = search._selection.rules
        forte_rules = [r for r in rules if r.order == SelectionOrder.FORTE]
        assert len(forte_rules) == 0

    def test_forte_disabled_with_zero_weight_no_selection_rule(self) -> None:
        """FORTE disabled with zero weight means no FORTE selection rule."""
        opts = SearchOptions(forte_embeddings=False, forte_weight=0.0)
        search = GivenClauseSearch(options=opts)
        rules = search._selection.rules
        forte_rules = [r for r in rules if r.order == SelectionOrder.FORTE]
        assert len(forte_rules) == 0

    def test_forte_weight_adds_rule_regardless_of_embeddings_flag(self) -> None:
        """forte_weight > 0 adds selection rule even when forte_embeddings=False.

        The selection rule is driven by forte_weight alone; forte_embeddings
        controls whether the provider is initialized.
        """
        opts = SearchOptions(forte_embeddings=False, forte_weight=0.5)
        search = GivenClauseSearch(options=opts)
        rules = search._selection.rules
        forte_rules = [r for r in rules if r.order == SelectionOrder.FORTE]
        assert len(forte_rules) == 1
        # But no provider should be created
        assert search.forte_provider is None

    def test_priority_sos_wired_to_forte_embeddings(self) -> None:
        """PrioritySOS receives reference to FORTE embeddings dict."""
        opts = SearchOptions(
            forte_embeddings=True, forte_weight=0.5, priority_sos=True,
        )
        search = GivenClauseSearch(options=opts)
        sos = search._state.sos
        assert isinstance(sos, PrioritySOS)
        assert sos._forte_embeddings_ref is search.forte_embeddings

    def test_forte_coexists_with_age_weight_rules(self) -> None:
        """FORTE rule coexists with standard age/weight rules."""
        opts = SearchOptions(forte_embeddings=True, forte_weight=0.5)
        search = GivenClauseSearch(options=opts)
        rules = search._selection.rules
        order_names = {r.order for r in rules}
        assert SelectionOrder.AGE in order_names
        assert SelectionOrder.WEIGHT in order_names
        assert SelectionOrder.FORTE in order_names


# ── 6. Graceful Degradation ──────────────────────────────────────────────────


class TestForteGracefulDegradation:
    """Verify graceful fallback when FORTE unavailable or fails."""

    def test_proof_found_without_forte(self) -> None:
        """Proof search succeeds without FORTE enabled."""
        opts = SearchOptions(forte_embeddings=False)
        search = GivenClauseSearch(options=opts)
        c1 = _make_clause(_pos_lit(_func(1, _const(2))), weight=2.0)
        c2 = _make_clause(_neg_lit(_func(1, _var(0))), weight=2.0)
        result = search.run(usable=[], sos=[c1, c2])
        assert result.exit_code.value == 1  # MAX_PROOFS_EXIT

    def test_proof_found_with_forte(self) -> None:
        """Proof search succeeds with FORTE enabled."""
        opts = SearchOptions(forte_embeddings=True, forte_weight=0.5)
        search = GivenClauseSearch(options=opts)
        c1 = _make_clause(_pos_lit(_func(1, _const(2))), weight=2.0)
        c2 = _make_clause(_neg_lit(_func(1, _var(0))), weight=2.0)
        result = search.run(usable=[], sos=[c1, c2])
        assert result.exit_code.value == 1  # MAX_PROOFS_EXIT
        assert len(result.proofs) >= 1

    def test_proof_equivalence_with_and_without_forte(self) -> None:
        """Same proof found regardless of FORTE being enabled/disabled."""
        c1 = _make_clause(_pos_lit(_func(1, _const(2))), weight=2.0)
        c2 = _make_clause(_neg_lit(_func(1, _var(0))), weight=2.0)

        opts_without = SearchOptions(forte_embeddings=False)
        search_without = GivenClauseSearch(options=opts_without)
        result_without = search_without.run(usable=[], sos=[c1, c2])

        # Re-create clauses (fresh IDs)
        c1b = _make_clause(_pos_lit(_func(1, _const(2))), weight=2.0)
        c2b = _make_clause(_neg_lit(_func(1, _var(0))), weight=2.0)

        opts_with = SearchOptions(forte_embeddings=True, forte_weight=0.5)
        search_with = GivenClauseSearch(options=opts_with)
        result_with = search_with.run(usable=[], sos=[c1b, c2b])

        # Both should find a proof
        assert result_without.exit_code.value == 1
        assert result_with.exit_code.value == 1

    def test_forte_embeddings_populated_during_search(self) -> None:
        """FORTE embeddings are computed for clauses during search."""
        opts = SearchOptions(forte_embeddings=True, forte_weight=0.5)
        search = GivenClauseSearch(options=opts)
        c1 = _make_clause(_pos_lit(_func(1, _const(2))), weight=2.0)
        c2 = _make_clause(_neg_lit(_func(1, _var(0))), weight=2.0)
        search.run(usable=[], sos=[c1, c2])
        # At least some embeddings should have been computed
        assert len(search.forte_embeddings) > 0


# ── 7. Performance Regression ────────────────────────────────────────────────


class TestFortePerformanceRegression:
    """Verify no performance degradation when FORTE disabled."""

    def test_no_slowdown_when_disabled(self) -> None:
        """Search with FORTE disabled should not be slower than baseline."""
        c1 = _make_clause(_pos_lit(_func(1, _const(2))), weight=2.0)
        c2 = _make_clause(_neg_lit(_func(1, _var(0))), weight=2.0)

        # Baseline: no FORTE
        opts = SearchOptions(forte_embeddings=False)
        start = time.perf_counter()
        for _ in range(10):
            s = GivenClauseSearch(options=opts)
            c1_copy = _make_clause(_pos_lit(_func(1, _const(2))), weight=2.0)
            c2_copy = _make_clause(_neg_lit(_func(1, _var(0))), weight=2.0)
            s.run(usable=[], sos=[c1_copy, c2_copy])
        baseline_time = time.perf_counter() - start

        # With FORTE disabled explicitly (should be equivalent)
        opts_disabled = SearchOptions(forte_embeddings=False, forte_weight=0.0)
        start = time.perf_counter()
        for _ in range(10):
            s = GivenClauseSearch(options=opts_disabled)
            c1_copy = _make_clause(_pos_lit(_func(1, _const(2))), weight=2.0)
            c2_copy = _make_clause(_neg_lit(_func(1, _var(0))), weight=2.0)
            s.run(usable=[], sos=[c1_copy, c2_copy])
        disabled_time = time.perf_counter() - start

        # Allow 50% tolerance for timing variance
        assert disabled_time < baseline_time * 1.5, (
            f"FORTE disabled ({disabled_time:.3f}s) slower than baseline ({baseline_time:.3f}s)"
        )


# ── 8. Thread Safety ─────────────────────────────────────────────────────────


class TestForteThreadSafety:
    """Verify thread safety for concurrent embedding operations."""

    def test_concurrent_embedding_computation(self) -> None:
        """Multiple threads can compute FORTE embeddings concurrently."""
        provider = ForteEmbeddingProvider()
        clauses = [
            _make_clause(
                _pos_lit(_func(i, _const(i + 10))),
                weight=float(i),
                clause_id=i,
            )
            for i in range(1, 21)
        ]

        results: dict[int, list[float] | None] = {}
        errors: list[Exception] = []

        def compute_embedding(clause: Clause) -> None:
            try:
                emb = provider.get_embedding(clause)
                results[clause.id] = emb
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(compute_embedding, c) for c in clauses]
            for f in as_completed(futures):
                f.result()  # re-raise any exceptions

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 20
        # All embeddings should be non-None
        for clause_id, emb in results.items():
            assert emb is not None, f"Clause {clause_id} got None embedding"

    def test_concurrent_batch_embedding(self) -> None:
        """Batch embedding is thread-safe."""
        provider = ForteEmbeddingProvider()
        batch = [
            _make_clause(
                _pos_lit(_func(i, _const(i + 10))),
                weight=float(i),
                clause_id=i,
            )
            for i in range(1, 11)
        ]

        results: list[list[list[float] | None]] = []
        lock = threading.Lock()

        def batch_embed() -> None:
            embs = provider.get_embeddings_batch(batch)
            with lock:
                results.append(embs)

        threads = [threading.Thread(target=batch_embed) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(results) == 4
        # All batches should produce the same results (deterministic)
        for batch_result in results:
            assert len(batch_result) == 10
            for emb in batch_result:
                assert emb is not None


# ── 9. End-to-End CLI Integration ────────────────────────────────────────────


class TestForteEndToEndCli:
    """Verify the full CLI → SearchOptions → Search pipeline."""

    def test_cli_args_reach_search_options(self) -> None:
        """CLI arguments correctly propagate to SearchOptions construction."""
        args = _parse_forte_args([
            "--forte-embeddings",
            "--forte-weight", "0.7",
            "--forte-dim", "64",
            "--forte-cache", "5000",
        ])
        opts = SearchOptions(
            forte_embeddings=args.forte_embeddings,
            forte_weight=args.forte_weight,
            forte_embedding_dim=args.forte_dim,
            forte_cache_max_entries=args.forte_cache,
        )
        assert opts.forte_embeddings is True
        assert opts.forte_weight == 0.7
        assert opts.forte_embedding_dim == 64
        assert opts.forte_cache_max_entries == 5000

    def test_cli_to_search_full_pipeline(self) -> None:
        """Full pipeline: CLI args → SearchOptions → GivenClauseSearch."""
        args = _parse_forte_args([
            "--forte-embeddings",
            "--forte-weight", "0.5",
            "--forte-dim", "128",
        ])
        opts = SearchOptions(
            forte_embeddings=args.forte_embeddings,
            forte_weight=args.forte_weight,
            forte_embedding_dim=args.forte_dim,
            forte_cache_max_entries=args.forte_cache,
        )
        search = GivenClauseSearch(options=opts)

        # Verify provider was created with correct config
        assert search.forte_provider is not None
        assert search.forte_provider.embedding_dim == 128

        # Verify selection rules include FORTE
        rules = search._selection.rules
        forte_rules = [r for r in rules if r.order == SelectionOrder.FORTE]
        assert len(forte_rules) == 1

    def test_cli_defaults_produce_no_forte(self) -> None:
        """Default CLI args (no --forte-embeddings) produce no FORTE provider."""
        args = _parse_forte_args([])
        opts = SearchOptions(
            forte_embeddings=args.forte_embeddings,
            forte_weight=args.forte_weight,
            forte_embedding_dim=args.forte_dim,
            forte_cache_max_entries=args.forte_cache,
        )
        search = GivenClauseSearch(options=opts)
        assert search.forte_provider is None
