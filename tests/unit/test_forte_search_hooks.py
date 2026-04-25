"""Tests for FORTE search integration hooks (Phase 2B).

Verifies that FORTE embeddings are correctly computed, stored, and
cleaned up during the search lifecycle.
"""

from __future__ import annotations

import pytest

from pyladr.search.given_clause import GivenClauseSearch, SearchOptions
from tests.factories import make_clause, make_const, make_func, make_neg_lit, make_pos_lit, make_var


# ── FORTE Disabled (Default) ─────────────────────────────────────────────────


class TestForteDisabledByDefault:
    """Verify zero impact when FORTE is disabled."""

    def test_default_options_no_forte(self) -> None:
        opts = SearchOptions()
        assert opts.forte_embeddings is False
        assert opts.forte_weight == 0.0

    def test_search_without_forte(self) -> None:
        """Search should work identically without FORTE."""
        opts = SearchOptions()
        search = GivenClauseSearch(options=opts)
        assert search.forte_provider is None
        assert search.forte_embeddings == {}

    def test_no_forte_embeddings_after_search(self) -> None:
        """No FORTE embeddings stored when disabled."""
        opts = SearchOptions()
        search = GivenClauseSearch(options=opts)

        # Simple resolution proof: P(a) and -P(x) → empty clause
        c1 = make_clause(make_pos_lit(make_func(1, make_const(2))), weight=2.0)
        c2 = make_clause(make_neg_lit(make_func(1, make_var(0))), weight=2.0)

        search.run(usable=[], sos=[c1, c2])
        assert search.forte_embeddings == {}


# ── FORTE Enabled ────────────────────────────────────────────────────────────


class TestForteEnabled:
    """Verify FORTE hooks activate when enabled."""

    def test_forte_provider_created(self) -> None:
        opts = SearchOptions(forte_embeddings=True)
        search = GivenClauseSearch(options=opts)
        assert search.forte_provider is not None

    def test_forte_config_custom_dim(self) -> None:
        opts = SearchOptions(
            forte_embeddings=True,
            forte_embedding_dim=128,
        )
        search = GivenClauseSearch(options=opts)
        provider = search.forte_provider
        assert provider is not None
        assert provider.embedding_dim == 128  # type: ignore[union-attr]

    def test_forte_config_custom_cache(self) -> None:
        opts = SearchOptions(
            forte_embeddings=True,
            forte_cache_max_entries=500,
        )
        search = GivenClauseSearch(options=opts)
        assert search.forte_provider is not None


# ── Init Clauses Batch Hook ──────────────────────────────────────────────────


class TestInitClausesBatchHook:
    """Verify _init_clauses batch pre-computation."""

    def test_initial_clauses_get_embeddings(self) -> None:
        """Initial usable + SOS clauses should get FORTE embeddings."""
        opts = SearchOptions(forte_embeddings=True)
        search = GivenClauseSearch(options=opts)

        usable = [
            make_clause(make_pos_lit(make_func(1, make_const(2))), weight=2.0),
        ]
        sos = [
            make_clause(make_pos_lit(make_func(3, make_var(0))), weight=2.0),
            make_clause(make_neg_lit(make_func(1, make_var(0))), weight=2.0),
        ]

        search._init_clauses(usable, sos)

        # All 3 clauses should have embeddings
        assert len(search.forte_embeddings) == 3
        for c in [*usable, *sos]:
            assert c.id in search.forte_embeddings
            emb = search.forte_embeddings[c.id]
            assert len(emb) == 128

    def test_empty_initial_clauses(self) -> None:
        """Empty initial clause lists should not crash."""
        opts = SearchOptions(forte_embeddings=True)
        search = GivenClauseSearch(options=opts)
        search._init_clauses([], [])
        assert search.forte_embeddings == {}


# ── Keep Clause Hook ─────────────────────────────────────────────────────────


class TestKeepClauseHook:
    """Verify _keep_clause computes FORTE embedding for new clauses."""

    def test_kept_clause_gets_embedding(self) -> None:
        """A kept clause should get a FORTE embedding."""
        opts = SearchOptions(forte_embeddings=True)
        search = GivenClauseSearch(options=opts)

        c = make_clause(make_pos_lit(make_func(1, make_var(0))), weight=2.0)
        search._keep_clause(c)

        assert c.id > 0  # ID was assigned
        assert c.id in search.forte_embeddings
        emb = search.forte_embeddings[c.id]
        assert len(emb) == 128

    def test_kept_clause_no_embedding_when_disabled(self) -> None:
        """No embedding stored when FORTE disabled."""
        opts = SearchOptions(forte_embeddings=False)
        search = GivenClauseSearch(options=opts)

        c = make_clause(make_pos_lit(make_func(1, make_var(0))), weight=2.0)
        search._keep_clause(c)

        assert c.id not in search.forte_embeddings

    def test_multiple_kept_clauses(self) -> None:
        """Multiple kept clauses each get their own embedding."""
        opts = SearchOptions(forte_embeddings=True)
        search = GivenClauseSearch(options=opts)

        clauses = [
            make_clause(make_pos_lit(make_func(i, make_var(0))), weight=float(i))
            for i in range(1, 6)
        ]
        for c in clauses:
            search._keep_clause(c)

        assert len(search.forte_embeddings) == 5
        for c in clauses:
            assert c.id in search.forte_embeddings


# ── End-to-End Search with FORTE ─────────────────────────────────────────────


class TestForteEndToEnd:
    """End-to-end search with FORTE enabled."""

    def test_simple_proof_with_forte(self) -> None:
        """Simple resolution proof should work with FORTE enabled."""
        opts = SearchOptions(forte_embeddings=True)
        search = GivenClauseSearch(options=opts)

        # P(a) and -P(x) → empty clause
        c1 = make_clause(make_pos_lit(make_func(1, make_const(2))), weight=2.0)
        c2 = make_clause(make_neg_lit(make_func(1, make_var(0))), weight=2.0)

        result = search.run(usable=[], sos=[c1, c2])

        # Should find proof
        assert result.exit_code.value == 1  # MAX_PROOFS_EXIT
        assert len(result.proofs) >= 1

        # FORTE embeddings should have been computed for kept clauses
        assert len(search.forte_embeddings) > 0

    def test_sos_exhausted_with_forte(self) -> None:
        """SOS exhaustion should work normally with FORTE enabled."""
        opts = SearchOptions(forte_embeddings=True, max_given=10)
        search = GivenClauseSearch(options=opts)

        # Single clause, no inferences possible
        c1 = make_clause(make_pos_lit(make_func(1, make_const(2))), weight=2.0)
        result = search.run(usable=[], sos=[c1])

        # SOS should be exhausted
        assert result.exit_code.value in (2, 3)  # SOS_EMPTY or MAX_GIVEN

        # Initial clause should have embedding
        assert len(search.forte_embeddings) >= 1

    def test_forte_embeddings_are_deterministic(self) -> None:
        """Two searches with same input produce same FORTE embeddings."""
        opts = SearchOptions(forte_embeddings=True)

        c1 = make_clause(make_pos_lit(make_func(1, make_const(2))), weight=2.0)
        c2 = make_clause(make_neg_lit(make_func(1, make_var(0))), weight=2.0)

        search1 = GivenClauseSearch(options=opts)
        search1.run(usable=[], sos=[c1, c2])

        # Re-create clauses (fresh objects) for second run
        c1b = make_clause(make_pos_lit(make_func(1, make_const(2))), weight=2.0)
        c2b = make_clause(make_neg_lit(make_func(1, make_var(0))), weight=2.0)

        search2 = GivenClauseSearch(options=opts)
        search2.run(usable=[], sos=[c1b, c2b])

        # Both should have embeddings for initial clauses
        # IDs should match since both start from 0
        for cid in search1.forte_embeddings:
            if cid in search2.forte_embeddings:
                assert search1.forte_embeddings[cid] == search2.forte_embeddings[cid]


# ── Configuration Options ────────────────────────────────────────────────────


class TestForteSearchOptions:
    """Verify SearchOptions FORTE fields."""

    def test_defaults(self) -> None:
        opts = SearchOptions()
        assert opts.forte_embeddings is False
        assert opts.forte_weight == 0.0
        assert opts.forte_embedding_dim == 128
        assert opts.forte_cache_max_entries == 10_000

    def test_custom_values(self) -> None:
        opts = SearchOptions(
            forte_embeddings=True,
            forte_weight=0.8,
            forte_embedding_dim=256,
            forte_cache_max_entries=50_000,
        )
        assert opts.forte_embeddings is True
        assert opts.forte_weight == 0.8
        assert opts.forte_embedding_dim == 256
        assert opts.forte_cache_max_entries == 50_000
