"""Unit tests for R2VSearchSubsystem.

These tests exercise the subsystem's contract boundaries — what happens
when R2V is disabled, when the provider fails to build, when the engine
calls into methods in states that were previously triggered by inline
blocks in GivenClauseSearch. Each test asserts *behavior* (what the
engine would observe), not internal shape.
"""
from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType
from pyladr.core.symbol import SymbolTable
from pyladr.search.options import SearchOptions
from pyladr.search.r2v_subsystem import R2VSearchSubsystem
from pyladr.search.selection import SelectionOrder
from pyladr.search.state import ClauseList


def _make_subsystem(**opt_overrides) -> R2VSearchSubsystem:
    opts = SearchOptions(**opt_overrides)
    return R2VSearchSubsystem(opts, state_fn=None)


# ── Behavioral: disabled-R2V paths must be no-ops ────────────────────


def test_default_options_produces_inert_subsystem():
    """With rnn2vec_embeddings=False, no provider is built and no R2V
    state is populated. This is the 'R2V disabled' contract."""
    sub = _make_subsystem()
    assert sub._provider is None
    # process_completions must not crash with no bg_updater:
    sub.process_completions()
    sub.shutdown()


def test_maybe_select_given_returns_none_when_disabled():
    """Engine fallback path: subsystem returns (None, None) so the
    engine uses standard selection. If this ever returns a clause
    when disabled, engine selection is silently bypassed."""
    sub = _make_subsystem()
    sos = ClauseList("sos")
    # Even with a non-RNN2Vec order, subsystem must return sentinel.
    result = sub.maybe_select_given(sos, SelectionOrder.AGE)
    assert result == (None, None)


def test_maybe_select_given_returns_none_for_non_r2v_order():
    """Even with embeddings populated, non-R2V/RGP orders must pass through."""
    sub = _make_subsystem()
    sub._embeddings[1] = [0.1, 0.2, 0.3]
    sos = ClauseList("sos")
    sos.append(Clause(literals=(), justification=()))
    assert sub.maybe_select_given(sos, SelectionOrder.AGE) == (None, None)
    assert sub.maybe_select_given(sos, SelectionOrder.WEIGHT) == (None, None)


def test_maybe_select_given_returns_none_with_empty_sos():
    """Empty SOS should not trigger R2V selection regardless of order."""
    sub = _make_subsystem()
    sub._embeddings[1] = [0.1, 0.2, 0.3]
    sos = ClauseList("sos")
    assert sub.maybe_select_given(sos, SelectionOrder.RNN2VEC) == (None, None)


def test_maybe_select_given_returns_none_with_empty_embeddings():
    """No embeddings → no R2V decision possible, caller must fall back."""
    sub = _make_subsystem()
    sos = ClauseList("sos")
    sos.append(Clause(literals=(), justification=()))
    assert sub.maybe_select_given(sos, SelectionOrder.RNN2VEC) == (None, None)


# ── Behavioral: hot-path calls must be safe when disabled ─────────────


def test_on_clause_kept_noop_without_provider():
    """Engine calls this for every kept clause. Must not crash or
    allocate when provider is None."""
    sub = _make_subsystem()
    c = Clause(literals=(), justification=())
    sub.on_clause_kept(c, all_clauses={})
    assert sub._embeddings == {}
    assert sub._online_batch == []


def test_on_clause_evicted_is_idempotent():
    """Engine calls on_clause_evicted from back-subsumption/back-demod
    for any disabled clause. Must handle missing-id and repeated calls."""
    sub = _make_subsystem()
    sub._embeddings[42] = [1.0]
    sub.on_clause_evicted(42)
    assert 42 not in sub._embeddings
    # Idempotent: second call is a no-op, not a KeyError.
    sub.on_clause_evicted(42)
    # Unknown id is a no-op.
    sub.on_clause_evicted(999)


def test_record_given_distance_noop_without_goal_provider():
    """Engine calls this for every given-clause selection. Must short-circuit
    when goal_provider is None (which is the case for most configs)."""
    sub = _make_subsystem()
    c = Clause(literals=(), justification=())
    sub.record_given_distance(c)
    assert sub._all_given_distances == {}


def test_format_extras_returns_empty_string_when_disabled():
    """_format_selection_extras appends this to a parts list. An empty
    string append is harmless; a None append or an exception breaks
    clause display output. Contract: always returns str."""
    sub = _make_subsystem()
    c = Clause(literals=(), justification=())
    assert sub.format_extras(c) == ""


def test_on_proof_found_noop_without_goal_provider():
    """Engine calls on every proof. Must skip histogram when goal
    tracking not configured (goal_provider is None)."""
    sub = _make_subsystem()
    # Fabricate a minimal proof-like object: only needs to be passable.
    class _FakeProof:
        clauses = ()
    sub.on_proof_found(_FakeProof(), all_proofs=[_FakeProof()], quiet=True)
    # No crash = pass. Nothing to assert about state; this guards the
    # nested-attribute access path.


# ── Contract: shutdown is safe even without background updater ────────


def test_shutdown_safe_without_bg_updater():
    """run() finally: calls shutdown even when bg_updater never started.
    Must be a no-op in that case, not an AttributeError."""
    sub = _make_subsystem()
    assert sub._bg_updater is None
    sub.shutdown()  # must not raise


# ── Structural: subsystem attributes are covered by __slots__ ─────────


def test_subsystem_uses_slots():
    """__slots__ discipline: any untracked attribute assignment should
    fail. This prevents future drift where someone adds state without
    updating __slots__."""
    sub = _make_subsystem()
    with pytest.raises(AttributeError):
        sub.some_new_attribute = 1  # type: ignore[attr-defined]
