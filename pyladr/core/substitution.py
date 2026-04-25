"""Substitution / unification context matching C LADR unify.h/unify.c.

The C implementation uses:
- Context: array of MAX_VARS (100) slots, each holding a (term, context) pair
- Trail: linked list recording which bindings were made (for undo)
- Multiplier: distinguishes variables from different clauses

We implement the same semantics with Python data structures.

Convention note: The C occur_check() returns TRUE when the variable does NOT
occur (safe to bind), FALSE when it does. Our Python occur_check() uses the
opposite convention — returns True when the variable DOES occur — but the
unify/match logic accounts for this consistently.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from pyladr.core.term import MAX_VARS, Term, copy_term, get_rigid_term, get_variable_term


# ── Global multiplier counter (C static int Next_multiplier) ──────────────────
# Use itertools.count() for lock-free atomic incrementing (GIL-safe).

_multiplier_counter = itertools.count()


def _get_next_multiplier() -> int:
    return next(_multiplier_counter)


def reset_multiplier() -> None:
    """Reset the multiplier counter (for testing)."""
    global _multiplier_counter
    _multiplier_counter = itertools.count()


# ── Trail entry ───────────────────────────────────────────────────────────────


@dataclass(slots=True)
class TrailEntry:
    """Single binding record for undo (C struct trail).

    Records that variable `varnum` in `context` was bound.
    """

    varnum: int
    context: Context


# ── Context ───────────────────────────────────────────────────────────────────

# Pre-built template for fast Context initialization.
_NONE_TEMPLATE = [None] * MAX_VARS


class Context:
    """Variable binding context matching C struct context.

    Holds substitutions for variables 0..MAX_VARS-1.
    Each slot stores (term, context) or (None, None) if unbound.

    The multiplier distinguishes variables from different clauses
    when applying substitutions.
    """

    __slots__ = ("terms", "contexts", "multiplier")

    def __init__(self) -> None:
        self.terms: list[Term | None] = _NONE_TEMPLATE[:]
        self.contexts: list[Context | None] = _NONE_TEMPLATE[:]
        self.multiplier: int = _get_next_multiplier()

    def is_bound(self, varnum: int) -> bool:
        """Check if variable is bound in this context."""
        return self.terms[varnum] is not None

    def bind(self, varnum: int, term: Term, context: Context | None) -> None:
        """Bind variable to (term, context)."""
        self.terms[varnum] = term
        self.contexts[varnum] = context

    def unbind(self, varnum: int) -> None:
        """Clear binding for variable."""
        self.terms[varnum] = None
        self.contexts[varnum] = None

    def clear(self) -> None:
        """Clear all bindings."""
        for i in range(MAX_VARS):
            self.terms[i] = None
            self.contexts[i] = None

    def __repr__(self) -> str:
        bindings = []
        for i in range(MAX_VARS):
            if self.terms[i] is not None:
                bindings.append(f"v{i}={self.terms[i]!r}")
        return f"Context(m={self.multiplier}, {{{', '.join(bindings)}}})"


# ── Trail-based binding/undo ──────────────────────────────────────────────────


class Trail:
    """Trail of bindings for backtracking (C Trail linked list).

    Supports bind-with-trail and undo operations.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: list[TrailEntry] = []

    def bind(self, varnum: int, ctx: Context, term: Term, term_ctx: Context | None) -> None:
        """Bind variable in context and record on trail.

        Matches C BIND_TR macro:
            c1->terms[i] = t2; c1->contexts[i] = c2;
            record (i, c1) on trail
        """
        ctx.terms[varnum] = term
        ctx.contexts[varnum] = term_ctx
        self._entries.append(TrailEntry(varnum=varnum, context=ctx))

    def undo(self) -> None:
        """Undo all bindings on this trail (C undo_subst).

        Walks the trail in reverse, clearing each binding.
        """
        for entry in reversed(self._entries):
            entry.context.unbind(entry.varnum)
        self._entries.clear()

    def undo_to(self, position: int) -> None:
        """Undo bindings back to a saved position (C partial trail rollback).

        Used by unify() to restore state when complex term arg unification
        fails partway through.
        """
        while len(self._entries) > position:
            entry = self._entries.pop()
            entry.context.unbind(entry.varnum)

    @property
    def position(self) -> int:
        """Current trail position (for saving/restoring)."""
        return len(self._entries)

    @property
    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def vars_in_trail(self) -> list[int]:
        """Return list of variable numbers in trail order (C vars_in_trail)."""
        return [entry.varnum for entry in self._entries]

    def __len__(self) -> int:
        return len(self._entries)


# ── Dereference ───────────────────────────────────────────────────────────────


def dereference(t: Term, c: Context | None) -> tuple[Term, Context | None]:
    """Follow variable bindings to their final value.

    Matches C DEREFERENCE macro:
        while (c != NULL && VARIABLE(t) && c->terms[VARNUM(t)]) {
            t = c->terms[VARNUM(t)]; c = c->contexts[VARNUM(t)];
        }

    Returns (final_term, final_context).
    """
    # Inline t.is_variable as (t.private_symbol >= 0) to avoid property overhead
    while c is not None and t.private_symbol >= 0:
        vn = t.private_symbol  # == t.varnum for variables
        if vn >= MAX_VARS or c.terms[vn] is None:
            break
        t = c.terms[vn]
        c = c.contexts[vn]
    return t, c


# ── Apply substitution ───────────────────────────────────────────────────────


def apply_substitution(t: Term, c: Context | None) -> Term:
    """Build instantiated term from term + context.

    Matches C apply(Term t, Context c).

    A NULL context is ok — it happens when c is built by match.
    If the context is NULL, apply just copies the term.

    Uninstantiated variables get new IDs:
        new_varnum = c->multiplier * MAX_VARS + VARNUM(t)
    """
    t, c = dereference(t, c)
    ps = t.private_symbol
    if ps >= 0:  # is_variable
        if c is None:
            return get_variable_term(ps)
        return get_variable_term(c.multiplier * MAX_VARS + ps)

    if t.arity == 0:  # is_constant
        return Term(private_symbol=ps)

    # Complex term — recursively apply to arguments
    new_args = tuple(apply_substitution(a, c) for a in t.args)
    return Term(private_symbol=ps, arity=t.arity, args=new_args)


def apply_substitute(
    t: Term,
    beta: Term,
    c_from: Context,
    into_term: Term,
    c_into: Context,
) -> Term:
    """Apply substitution with replacement at a specific subterm.

    Matches C apply_substitute(). Used for paramodulation:
    when paramodulating alpha=beta into p[into_term], where alpha
    unifies with into_term, construct the instance of p[beta] in one step.

    When reaching into_term, applies beta with c_from instead.
    All other variables are applied with c_into.
    """
    if t is into_term:
        return apply_substitution(beta, c_from)
    if t.private_symbol >= 0:  # is_variable
        return apply_substitution(t, c_into)
    new_args = tuple(
        apply_substitute(a, beta, c_from, into_term, c_into) for a in t.args
    )
    return Term(private_symbol=t.private_symbol, arity=t.arity, args=new_args)


def apply_substitute_at_pos(
    t: Term,
    beta: Term,
    c_from: Context,
    into_pos: tuple[int, ...],
    c_into: Context,
) -> Term:
    """Apply substitution with replacement at a position vector.

    Matches C apply_substitute2(). Like apply_substitute but the target
    subterm is specified by a position vector (1-indexed, matching C convention)
    instead of by identity. Needed when the target is a variable (shared).
    """
    if len(into_pos) == 0:
        return apply_substitution(beta, c_from)
    if t.private_symbol >= 0:  # is_variable
        return apply_substitution(t, c_into)
    arg_pos = into_pos[0] - 1  # Position vectors count from 1 in C
    new_args = tuple(
        apply_substitute_at_pos(a, beta, c_from, into_pos[1:], c_into)
        if i == arg_pos
        else apply_substitution(a, c_into)
        for i, a in enumerate(t.args)
    )
    return Term(private_symbol=t.private_symbol, arity=t.arity, args=new_args)


def apply_demod(t: Term, c: Context) -> Term:
    """Apply substitution for demodulation.

    Matches C apply_demod(). Assumes every variable in t is instantiated
    by the substitution. Copies the instantiated terms directly.

    Note: The C version sets a term flag on results; we skip that since
    our terms are immutable. The caller can track demod status externally.
    """
    ps = t.private_symbol
    if ps >= 0:  # is_variable; ps == varnum
        bound = c.terms[ps]
        if bound is None:
            raise ValueError(f"Variable v{ps} not instantiated in demod context")
        return copy_term(bound)
    if t.arity == 0:  # is_constant
        return Term(private_symbol=ps)
    new_args = tuple(apply_demod(a, c) for a in t.args)
    return Term(private_symbol=ps, arity=t.arity, args=new_args)


# ── Occur check ───────────────────────────────────────────────────────────────


def occur_check(varnum: int, var_ctx: Context, t: Term, t_ctx: Context | None) -> bool:
    """Check if variable occurs in term (prevents infinite structures).

    Returns True if variable DOES occur in the term.

    Note: C occur_check uses opposite convention (returns TRUE when safe,
    FALSE when variable occurs). Our unify/match handle this consistently.
    """
    t, t_ctx = dereference(t, t_ctx)
    if t.private_symbol >= 0:  # is_variable
        return t.private_symbol == varnum and t_ctx is var_ctx
    return any(occur_check(varnum, var_ctx, a, t_ctx) for a in t.args)


# ── Unification ───────────────────────────────────────────────────────────────


def unify(
    t1: Term,
    c1: Context,
    t2: Term,
    c2: Context,
    trail: Trail,
) -> bool:
    """Two-way unification matching C unify().

    Returns True if t1 in c1 and t2 in c2 can be unified.
    Bindings are recorded on trail for undo.

    Matches C behavior: when complex term unification fails partway through
    arguments, automatically rolls back partial bindings from that call.
    """
    t1, c1_deref = dereference(t1, c1)
    c1 = c1_deref if c1_deref is not None else c1

    t2, c2_deref = dereference(t2, c2)
    c2 = c2_deref if c2_deref is not None else c2

    ps1 = t1.private_symbol
    if ps1 >= 0:  # t1 is_variable
        vn1 = ps1
        ps2 = t2.private_symbol
        if ps2 >= 0 and vn1 == ps2 and c1 is c2:
            return True  # Same variable in same context
        if ps2 >= 0:  # t2 is_variable
            # Both variables, different — no occur check needed (C behavior)
            trail.bind(vn1, c1, t2, c2)
            return True
        # t1 variable, t2 not variable — need occur check
        if occur_check(vn1, c1, t2, c2):
            return False
        trail.bind(vn1, c1, t2, c2)
        return True

    if t2.private_symbol >= 0:  # t2 is_variable
        vn2 = t2.private_symbol
        if occur_check(vn2, c2, t1, c1):
            return False
        trail.bind(vn2, c2, t1, c1)
        return True

    # Both rigid — must have same symbol
    if t1.private_symbol != t2.private_symbol:
        return False

    # Constants with same symbol
    if t1.arity == 0:
        return True

    # Complex terms with same symbol — unify arguments with rollback
    saved_pos = trail.position
    for i in range(t1.arity):
        if not unify(t1.args[i], c1, t2.args[i], c2, trail):
            # Roll back partial bindings from this complex unification
            trail.undo_to(saved_pos)
            return False
    return True


# ── One-way matching ──────────────────────────────────────────────────────────


def match(
    pattern: Term,
    p_ctx: Context,
    target: Term,
    trail: Trail,
) -> bool:
    """One-way matching: target is an instance of pattern.

    Matches C match(). Only pattern variables get bound;
    target is treated as ground (its context is None).
    """
    pattern, p_ctx_deref = dereference(pattern, p_ctx)

    # C behavior: if (c1 == NULL) return term_ident(t1, t2);
    # After dereference, NULL context means the term came from a previous
    # binding to the target side (stored with NULL context). The only valid
    # operation is structural comparison — we cannot look up variables in
    # a NULL context or bind into it.
    if p_ctx_deref is None:
        return pattern.term_ident(target)

    p_ctx = p_ctx_deref

    p_ps = pattern.private_symbol
    if p_ps >= 0:  # pattern is_variable
        vn = p_ps  # == varnum
        if p_ctx.terms[vn] is None:
            # Unbound — bind and succeed
            trail.bind(vn, p_ctx, target, None)
            return True
        # Already bound — check consistency (C: term_ident(c1->terms[vn], t2))
        return p_ctx.terms[vn].term_ident(target)

    if target.private_symbol >= 0:  # target is_variable
        return False  # Can't match rigid pattern against variable target

    if p_ps != target.private_symbol:
        return False
    if pattern.arity != target.arity:
        return False

    # Unrolled loop instead of all(genexpr) to avoid generator overhead
    p_args = pattern.args
    t_args = target.args
    for i in range(len(p_args)):
        if not match(p_args[i], p_ctx, t_args[i], trail):
            return False
    return True


# ── Variant checking ─────────────────────────────────────────────────────────


def variant(
    t1: Term,
    c1: Context,
    t2: Term,
    trail: Trail,
) -> bool:
    """Check if t1 and t2 are variants (each is an instance of the other).

    Matches C variant(). If successful, the matching substitution from
    t1 to t2 is recorded in c1 and on trail.
    """
    # First check: can t2 match t1? (t1 is instance of t2)
    c2 = Context()
    tr_temp = Trail()
    if not match(t2, c2, t1, tr_temp):
        return False
    tr_temp.undo()
    # Second check: can t1 match t2? (t2 is instance of t1)
    return match(t1, c1, t2, trail)


# ── Context utility functions ─────────────────────────────────────────────────


def empty_substitution(ctx: Context) -> bool:
    """Check if context has no bindings (C empty_substitution)."""
    return all(t is None for t in ctx.terms)


def variable_substitution(ctx: Context) -> bool:
    """Check if all bindings resolve to variables (C variable_substitution).

    Returns True if the substitution is a pure variable renaming.
    """
    for i in range(MAX_VARS):
        if ctx.terms[i] is not None:
            t = ctx.terms[i]
            c: Context | None = ctx.contexts[i]
            assert t is not None
            t, c = dereference(t, c)
            if not t.is_variable:
                return False
    return True


def subst_changes_term(t: Term, c: Context) -> bool:
    """Check if applying the context would change the term (C subst_changes_term).

    Returns True if at least one variable in t is bound in c.
    """
    ps = t.private_symbol
    if ps >= 0:  # is_variable
        return c.terms[ps] is not None
    return any(subst_changes_term(a, c) for a in t.args)


def context_to_pairs(
    varnums: set[int],
    c: Context,
) -> list[tuple[Term, Term]]:
    """Convert context bindings to (variable, instantiation) pairs.

    Matches C context_to_pairs(). Only includes variables in varnums
    whose instantiation differs from the variable itself.
    """
    pairs: list[tuple[Term, Term]] = []
    for i in range(MAX_VARS):
        if i in varnums:
            var = get_variable_term(i)
            instantiated = apply_substitution(var, c)
            if not var.term_ident(instantiated):
                pairs.append((var, instantiated))
    return pairs
