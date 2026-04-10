"""Object pools for reducing allocation pressure in hot paths.

The search loop creates millions of Context and Trail objects during
unification, matching, and subsumption. Each Context allocates two
100-element lists. This module provides thread-local pools that reuse
these objects instead of allocating new ones.

Usage:
    from pyladr.core.object_pool import get_context, release_context
    from pyladr.core.object_pool import get_trail, release_trail

    ctx = get_context()
    try:
        # ... use ctx for matching/unification ...
        pass
    finally:
        release_context(ctx)

Or as context managers:
    with pooled_context() as ctx:
        ...
    with pooled_trail() as trail:
        ...
"""

from __future__ import annotations

import threading
from collections import deque
from contextlib import contextmanager
from typing import Iterator

from pyladr.core.substitution import Context, Trail
from pyladr.core.term import MAX_VARS


# ── Pool configuration ─────────────────────────────────────────────────────

# Maximum idle objects per thread. Beyond this, objects are GC'd.
_MAX_POOL_SIZE = 32


# ── Thread-local storage ──────────────────────────────────────────────────

class _PoolLocal(threading.local):
    """Thread-local pool storage."""

    def __init__(self) -> None:
        super().__init__()
        self.contexts: deque[Context] = deque()
        self.trails: deque[Trail] = deque()


_pool = _PoolLocal()

# Pre-built None template for fast context reset
_NONE_TEMPLATE = [None] * MAX_VARS


# ── Context pool ──────────────────────────────────────────────────────────


def get_context() -> Context:
    """Get a Context from the thread-local pool, or create a new one.

    The returned Context has all bindings cleared and a fresh multiplier.
    """
    pool = _pool.contexts
    if pool:
        ctx = pool.pop()
        # Reset: clear all bindings in-place (faster than creating new lists)
        ctx.terms[:] = _NONE_TEMPLATE
        ctx.contexts[:] = _NONE_TEMPLATE
        # Fresh multiplier for this usage
        from pyladr.core.substitution import _get_next_multiplier
        ctx.multiplier = _get_next_multiplier()
        return ctx
    return Context()


def release_context(ctx: Context) -> None:
    """Return a Context to the thread-local pool for reuse."""
    pool = _pool.contexts
    if len(pool) < _MAX_POOL_SIZE:
        pool.append(ctx)


@contextmanager
def pooled_context() -> Iterator[Context]:
    """Context manager that borrows a Context from the pool."""
    ctx = get_context()
    try:
        yield ctx
    finally:
        release_context(ctx)


# ── Trail pool ────────────────────────────────────────────────────────────


def get_trail() -> Trail:
    """Get a Trail from the thread-local pool, or create a new one.

    The returned Trail has all entries cleared.
    """
    pool = _pool.trails
    if pool:
        trail = pool.pop()
        # Ensure it's clean
        if trail._entries:
            trail._entries.clear()
        return trail
    return Trail()


def release_trail(trail: Trail) -> None:
    """Return a Trail to the thread-local pool for reuse."""
    pool = _pool.trails
    if len(pool) < _MAX_POOL_SIZE:
        # Undo any remaining bindings before pooling
        if trail._entries:
            trail.undo()
        pool.append(trail)


@contextmanager
def pooled_trail() -> Iterator[Trail]:
    """Context manager that borrows a Trail from the pool."""
    trail = get_trail()
    try:
        yield trail
    finally:
        release_trail(trail)


# ── Pool statistics ───────────────────────────────────────────────────────


def pool_stats() -> dict[str, int]:
    """Return current pool sizes for monitoring."""
    return {
        "contexts_pooled": len(_pool.contexts),
        "trails_pooled": len(_pool.trails),
    }


def clear_pools() -> None:
    """Clear all pooled objects (for testing/cleanup)."""
    _pool.contexts.clear()
    _pool.trails.clear()
