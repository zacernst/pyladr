"""Demodulation (term rewriting) matching C LADR demod.c / flatdemod.c / backdemod.c.

Implements:
- demodulator_type: classify whether a clause can serve as a demodulator
- demodulate_term: rewrite a single term using indexed demodulators
- demodulate_clause: rewrite all terms in a clause
- back_demodulatable: find clauses that can be rewritten by a new demodulator

Demodulation rewrites terms using oriented equations as rewrite rules:
  Given demodulator alpha=beta where alpha > beta, replace any instance
  of alpha in a clause with the corresponding instance of beta.

Demodulator types (matching C enum):
- NOT_DEMODULATOR: can't be used for rewriting
- ORIENTED: left > right by term ordering, always rewrite left-to-right
- LEX_DEP_LR: rewrite left-to-right, checking ordering at each use
- LEX_DEP_RL: rewrite right-to-left, checking ordering at each use
- LEX_DEP_BOTH: rewrite in either direction, checking ordering at each use
"""

from __future__ import annotations

from enum import IntEnum, auto
from typing import Any

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.substitution import (
    Context,
    Trail,
    apply_demod,
    match,
)
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term
from pyladr.indexing import IndexType, Mindex
from pyladr.inference.paramodulation import is_eq_atom, is_oriented_eq, is_renamable_flip
from pyladr.ordering.termorder import term_greater


# ── Demodulator type enum ────────────────────────────────────────────────────


class DemodType(IntEnum):
    """Demodulator classification matching C demod.c enum."""

    NOT_DEMODULATOR = 0
    ORIENTED = auto()
    LEX_DEP_LR = auto()
    LEX_DEP_RL = auto()
    LEX_DEP_BOTH = auto()


# ── Demodulator classification ───────────────────────────────────────────────


def demodulator_type(
    clause: Clause,
    symbol_table: SymbolTable,
    lex_dep_demod_lim: int = 0,
) -> DemodType:
    """Classify a clause as a demodulator. Matches C demodulator_type().

    A clause is a demodulator if it is a positive unit equality (one literal,
    positive sign, equality predicate).

    Args:
        clause: The clause to classify.
        symbol_table: Symbol table for equality detection.
        lex_dep_demod_lim: Max combined variable count for lex-dependent
                          demodulators. 0 = disabled (only oriented allowed).

    Returns:
        DemodType classification.
    """
    # Must be a positive unit equality
    if len(clause.literals) != 1:
        return DemodType.NOT_DEMODULATOR

    lit = clause.literals[0]
    if not lit.sign:
        return DemodType.NOT_DEMODULATOR

    atom = lit.atom
    if not is_eq_atom(atom, symbol_table):
        return DemodType.NOT_DEMODULATOR

    alpha = atom.args[0]  # left side
    beta = atom.args[1]   # right side

    # If the equality is oriented by term ordering, it's a simple demodulator
    if is_oriented_eq(atom):
        return DemodType.ORIENTED

    # Check lex-dependent demodulation eligibility
    if lex_dep_demod_lim <= 0:
        return DemodType.NOT_DEMODULATOR

    alpha_vars = alpha.variables()
    beta_vars = beta.variables()

    n_vars = len(alpha_vars | beta_vars)
    if n_vars > lex_dep_demod_lim:
        return DemodType.NOT_DEMODULATOR

    # Left-to-right: beta_vars ⊆ alpha_vars AND alpha is not a variable
    lr = beta_vars <= alpha_vars and not alpha.is_variable

    # Right-to-left: alpha_vars ⊆ beta_vars AND beta is not a variable
    # AND not a renamable flip (to prevent infinite loops)
    rl = (
        not is_renamable_flip(atom)
        and alpha_vars <= beta_vars
        and not beta.is_variable
    )

    if lr and rl:
        return DemodType.LEX_DEP_BOTH
    if lr:
        return DemodType.LEX_DEP_LR
    if rl:
        return DemodType.LEX_DEP_RL
    return DemodType.NOT_DEMODULATOR


# ── Demodulator storage ──────────────────────────────────────────────────────


class DemodulatorIndex:
    """Index of demodulators using discrimination tree for efficient matching.

    Uses a DiscrimWild index on demodulator LHS (and RHS for bidirectional)
    terms for O(log n) candidate retrieval instead of O(n) linear scanning.
    retrieve_generalizations returns terms that generalize the query, which
    is exactly what we need for one-way matching (demodulator LHS matches
    against subterms of the clause being simplified).
    """

    __slots__ = ("_demods", "_lhs_idx", "_rhs_idx")

    def __init__(self) -> None:
        self._demods: dict[int, tuple[Clause, DemodType]] = {}
        # Index LHS terms for left-to-right matching
        self._lhs_idx: Mindex = Mindex(IndexType.DISCRIM_WILD)
        # Index RHS terms for right-to-left matching (LEX_DEP_RL, LEX_DEP_BOTH)
        self._rhs_idx: Mindex = Mindex(IndexType.DISCRIM_WILD)

    def insert(self, clause: Clause, dtype: DemodType) -> None:
        """Add a demodulator to the index."""
        cid = clause.id
        self._demods[cid] = (clause, dtype)
        atom = clause.literals[0].atom
        alpha = atom.args[0]  # LHS
        beta = atom.args[1]   # RHS

        # Index LHS for left-to-right matching
        if dtype in (DemodType.ORIENTED, DemodType.LEX_DEP_LR, DemodType.LEX_DEP_BOTH):
            self._lhs_idx.insert(alpha, cid)

        # Index RHS for right-to-left matching
        if dtype in (DemodType.LEX_DEP_RL, DemodType.LEX_DEP_BOTH):
            self._rhs_idx.insert(beta, cid)

    def remove(self, clause: Clause) -> None:
        """Remove a demodulator from the index."""
        cid = clause.id
        entry = self._demods.pop(cid, None)
        if entry is None:
            return
        _, dtype = entry
        atom = clause.literals[0].atom
        alpha = atom.args[0]
        beta = atom.args[1]

        if dtype in (DemodType.ORIENTED, DemodType.LEX_DEP_LR, DemodType.LEX_DEP_BOTH):
            self._lhs_idx.delete(alpha, cid)
        if dtype in (DemodType.LEX_DEP_RL, DemodType.LEX_DEP_BOTH):
            self._rhs_idx.delete(beta, cid)

    @property
    def is_empty(self) -> bool:
        return len(self._demods) == 0

    def __len__(self) -> int:
        return len(self._demods)

    def __iter__(self):
        """Iterate all demodulators (for back-demodulation compatibility)."""
        return iter(self._demods.values())

    def find_lhs_candidates(self, query: Term) -> list[tuple[Clause, DemodType]]:
        """Find demodulators whose LHS might match the query term."""
        cids = self._lhs_idx.retrieve_generalizations(query)
        return [self._demods[cid] for cid in cids if cid in self._demods]

    def find_rhs_candidates(self, query: Term) -> list[tuple[Clause, DemodType]]:
        """Find demodulators whose RHS might match the query term."""
        cids = self._rhs_idx.retrieve_generalizations(query)
        return [self._demods[cid] for cid in cids if cid in self._demods]


# ── Core demodulation ────────────────────────────────────────────────────────


def _try_demod(
    t: Term,
    clause: Clause,
    dtype: DemodType,
    symbol_table: SymbolTable,
    lex_order_vars: bool,
) -> Term | None:
    """Try to rewrite term t using a single demodulator (both directions).

    Returns the rewritten term if successful, None otherwise.
    Used by back_demodulatable which doesn't use indexed lookup.
    """
    result = _try_demod_lhs(t, clause, dtype, symbol_table, lex_order_vars)
    if result is not None:
        return result
    return _try_demod_rhs(t, clause, dtype, symbol_table, lex_order_vars)


def _try_demod_lhs(
    t: Term,
    clause: Clause,
    dtype: DemodType,
    symbol_table: SymbolTable,
    lex_order_vars: bool,
    ctx: Context | None = None,
    trail: Trail | None = None,
) -> Term | None:
    """Try left-to-right rewrite: match alpha against t, replace with beta.

    Optionally accepts pre-allocated Context/Trail for reuse (avoids allocation
    in hot loops). Caller must ensure trail has been undo()'d between calls.
    """
    atom = clause.literals[0].atom
    alpha = atom.args[0]
    beta = atom.args[1]

    if ctx is None:
        ctx = Context()
    if trail is None:
        trail = Trail()
    if match(alpha, ctx, t, trail):
        contractum = apply_demod(beta, ctx)
        trail.undo()
        if dtype == DemodType.ORIENTED:
            return contractum
        if term_greater(t, contractum, lex_order_vars, symbol_table):
            return contractum
    else:
        trail.undo()
    return None


def _try_demod_rhs(
    t: Term,
    clause: Clause,
    dtype: DemodType,
    symbol_table: SymbolTable,
    lex_order_vars: bool,
    ctx: Context | None = None,
    trail: Trail | None = None,
) -> Term | None:
    """Try right-to-left rewrite: match beta against t, replace with alpha.

    Optionally accepts pre-allocated Context/Trail for reuse.
    """
    if dtype not in (DemodType.LEX_DEP_RL, DemodType.LEX_DEP_BOTH):
        return None

    atom = clause.literals[0].atom
    alpha = atom.args[0]
    beta = atom.args[1]

    if ctx is None:
        ctx = Context()
    if trail is None:
        trail = Trail()
    if match(beta, ctx, t, trail):
        contractum = apply_demod(alpha, ctx)
        trail.undo()
        if term_greater(t, contractum, lex_order_vars, symbol_table):
            return contractum
    else:
        trail.undo()
    return None


def demodulate_term(
    t: Term,
    demod_index: DemodulatorIndex,
    symbol_table: SymbolTable,
    lex_order_vars: bool = False,
    step_limit: int = 1000,
) -> tuple[Term, list[tuple[int, int, int]]]:
    """Rewrite a term using demodulators. Matches C demod() bottom-up rewriting.

    Recursively demodulates subterms first (bottom-up), then tries to
    rewrite the term itself. Repeats until no more rewrites are possible.

    Args:
        t: The term to demodulate.
        demod_index: Index of available demodulators.
        symbol_table: Symbol table.
        lex_order_vars: Whether to use lex ordering for variables.
        step_limit: Maximum number of rewrite steps (prevents infinite loops).

    Returns:
        Tuple of (rewritten_term, demod_steps) where demod_steps is a list
        of (demod_id, position, direction) triples.
    """
    steps: list[tuple[int, int, int]] = []
    result = _demod_term_iterative(
        t, demod_index, symbol_table, lex_order_vars, steps, step_limit,
    )
    return result, steps


def _demod_term_iterative(
    root: Term,
    demod_index: DemodulatorIndex,
    symbol_table: SymbolTable,
    lex_order_vars: bool,
    steps: list[tuple[int, int, int]],
    remaining_steps: int,
) -> Term:
    """Iteratively demodulate a term bottom-up using an explicit stack.

    Semantically equivalent to the original recursive algorithm:
    1. Demodulate subterms (post-order traversal)
    2. Try matching against demodulators at current node
    3. If rewritten, re-demodulate the result bottom-up

    Uses an explicit work stack instead of Python call stack, eliminating
    RecursionError on deep terms and reducing function call overhead.
    """
    if remaining_steps <= 0 or root.is_variable:
        return root

    return _demod_bottom_up(
        root, demod_index, symbol_table, lex_order_vars, steps, remaining_steps,
    )


def _demod_bottom_up(
    root: Term,
    demod_index: DemodulatorIndex,
    symbol_table: SymbolTable,
    lex_order_vars: bool,
    steps: list[tuple[int, int, int]],
    remaining_steps: int,
) -> Term:
    """Bottom-up demodulation using explicit post-order stack.

    Two-phase iterative algorithm (avoids RecursionError on deep terms):

      work_stack entries are (term, phase) pairs:
        Phase 0 — EXPAND: push children right-to-left onto work_stack,
                  then re-push self with phase=1.
        Phase 1 — PROCESS: pop N children results from result_stack,
                  rebuild the node if any child changed, then try to
                  rewrite at this node using the demodulator index.
                  If rewriting succeeds, push the result back as phase 0
                  for full re-demodulation.

      Example for f(g(a), b):
        work:   [(f(g(a),b), 0)]
        → push  [(f(g(a),b), 1), (b, 0), (g(a), 0)]
        → b is constant → result_stack=[b], work=[(f,..,1), (g(a),0)]
        → expand g(a) → [(f,..,1), (g(a),1), (a,0)]
        → a → result=[b, a], process g(a) → result=[b, g(a')]
        → process f → pop [b, g(a')], rebuild, try rewrite → result=[f(g(a'),b)]

    Allocates a single Context/Trail pair and reuses it across all rewrite
    attempts to minimize allocation overhead in the hot loop.
    """
    # Pre-allocate match context/trail for reuse across all rewrite attempts
    reuse_ctx = Context()
    reuse_trail = Trail()

    # Post-order stack: (term, phase)
    #   phase 0: push children right-to-left, then re-push self with phase 1
    #   phase 1: children are done (results on result_stack), process this node
    work_stack: list[tuple[Term, int]] = [(root, 0)]
    result_stack: list[Term] = []

    while work_stack:
        if remaining_steps <= 0:
            # Exhaust remaining stack, returning terms as-is
            for term, phase in reversed(work_stack):
                if phase == 0:
                    result_stack.append(term)
            break

        term, phase = work_stack.pop()

        if term.is_variable:
            result_stack.append(term)
            continue

        if phase == 0:
            if not term.is_complex:
                # Constant: no children, try demodulation directly
                work_stack.append((term, 1))
            else:
                # Push self back for phase 1 processing after children
                work_stack.append((term, 1))
                # Push children right-to-left so left child processed first
                for i in range(term.arity - 1, -1, -1):
                    child = term.args[i]
                    if child.is_variable:
                        result_stack.append(child)
                    else:
                        work_stack.append((child, 0))
            continue

        # Phase 1: all children processed, results on result_stack
        if term.is_complex:
            # Pop children results (they were pushed left-to-right onto result_stack)
            n = term.arity
            new_args_list = result_stack[-n:]
            del result_stack[-n:]
            new_args = tuple(new_args_list)

            changed = any(new_args[i] is not term.args[i] for i in range(n))
            if changed:
                term = Term(
                    private_symbol=term.private_symbol,
                    arity=term.arity,
                    args=new_args,
                )

        # Try rewriting at this node (reuses pre-allocated ctx/trail)
        rewritten_term, remaining_steps = _try_rewrite_at_node(
            term, demod_index, symbol_table, lex_order_vars,
            steps, remaining_steps, reuse_ctx, reuse_trail,
        )

        if rewritten_term is not term:
            # Rewrite succeeded — push result for full re-demodulation
            if rewritten_term.is_variable or remaining_steps <= 0:
                result_stack.append(rewritten_term)
            else:
                work_stack.append((rewritten_term, 0))
        else:
            result_stack.append(term)

    return result_stack[0] if result_stack else root


def _try_rewrite_at_node(
    t: Term,
    demod_index: DemodulatorIndex,
    symbol_table: SymbolTable,
    lex_order_vars: bool,
    steps: list[tuple[int, int, int]],
    remaining_steps: int,
    _ctx: Context | None = None,
    _trail: Trail | None = None,
) -> tuple[Term, int]:
    """Try to rewrite a term at the top level using demodulators.

    Tries LHS candidates first, then RHS candidates.
    Returns (rewritten_term, remaining_steps) if rewritten,
    or (original_term, remaining_steps) if no rewrite applies.

    Reuses a single Context/Trail pair across all candidate matches
    to avoid repeated allocation in the hot path.
    """
    if t.is_variable or remaining_steps <= 0:
        return t, remaining_steps

    # Allocate once, reuse across all candidates
    if _ctx is None:
        _ctx = Context()
    if _trail is None:
        _trail = Trail()

    # Try LHS candidates (left-to-right matching)
    for clause, dtype in demod_index.find_lhs_candidates(t):
        result = _try_demod_lhs(t, clause, dtype, symbol_table, lex_order_vars, _ctx, _trail)
        if result is not None:
            steps.append((clause.id, 0, 1))
            remaining_steps -= 1
            return result, remaining_steps

    # Try RHS candidates (right-to-left matching)
    for clause, dtype in demod_index.find_rhs_candidates(t):
        result = _try_demod_rhs(t, clause, dtype, symbol_table, lex_order_vars, _ctx, _trail)
        if result is not None:
            steps.append((clause.id, 0, 2))
            remaining_steps -= 1
            return result, remaining_steps

    return t, remaining_steps


def demodulate_clause(
    clause: Clause,
    demod_index: DemodulatorIndex,
    symbol_table: SymbolTable,
    lex_order_vars: bool = False,
    step_limit: int = 1000,
) -> tuple[Clause, list[tuple[int, int, int]]]:
    """Demodulate all terms in a clause. Matches C demodulate_clause().

    Rewrites the atom of each literal using the demodulator index.

    Args:
        clause: The clause to demodulate.
        demod_index: Index of available demodulators.
        symbol_table: Symbol table.
        lex_order_vars: Whether to use lex ordering for variables.
        step_limit: Maximum rewrite steps per term.

    Returns:
        Tuple of (demodulated_clause, all_demod_steps).
    """
    if demod_index.is_empty:
        return clause, []

    all_steps: list[tuple[int, int, int]] = []
    new_lits: list[Literal] = []
    changed = False

    for lit in clause.literals:
        new_atom, steps = demodulate_term(
            lit.atom, demod_index, symbol_table, lex_order_vars, step_limit,
        )
        all_steps.extend(steps)
        if new_atom is not lit.atom:
            new_lits.append(Literal(sign=lit.sign, atom=new_atom))
            changed = True
        else:
            new_lits.append(lit)

    if not changed:
        return clause, []

    # Build demodulation justification
    demod_just = Justification(
        just_type=JustType.DEMOD,
        clause_id=clause.id,
        demod_steps=tuple(all_steps),
    )

    new_clause = Clause(
        literals=tuple(new_lits),
        id=clause.id,
        weight=clause.weight,
        justification=clause.justification + (demod_just,),
        is_formula=clause.is_formula,
    )
    return new_clause, all_steps


# ── Back-demodulation ────────────────────────────────────────────────────────


def back_demodulatable(
    new_demod: Clause,
    dtype: DemodType,
    clauses,
    symbol_table: SymbolTable,
    lex_order_vars: bool = False,
) -> list[Clause]:
    """Find clauses that can be rewritten by a new demodulator.

    Matches C back_demodulatable(). Returns list of clauses that contain
    a subterm matching the demodulator's LHS (or RHS for bidirectional).

    Args:
        new_demod: The new demodulator clause.
        dtype: Type of the demodulator.
        clauses: Iterable of kept clauses to check.
        symbol_table: Symbol table.
        lex_order_vars: Whether to use lex ordering for variables.

    Returns:
        List of clauses that can be rewritten.
    """
    rewritable: list[Clause] = []
    atom = new_demod.literals[0].atom
    alpha = atom.args[0]
    beta = atom.args[1]

    for clause in clauses:
        if clause is new_demod:
            continue

        can_rewrite = False
        for lit in clause.literals:
            if _subterm_matches_demod(
                lit.atom, alpha, beta, dtype, symbol_table, lex_order_vars,
            ):
                can_rewrite = True
                break

        if can_rewrite:
            rewritable.append(clause)

    return rewritable


def _subterm_matches_demod(
    t: Term,
    alpha: Term,
    beta: Term,
    dtype: DemodType,
    symbol_table: SymbolTable,
    lex_order_vars: bool,
) -> bool:
    """Check if any subterm of t matches the demodulator.

    Uses iterative traversal to avoid RecursionError on deep terms.
    """
    work: list[Term] = [t]
    while work:
        node = work.pop()
        if node.is_variable:
            continue
        if _matches_demod_at(node, alpha, beta, dtype, symbol_table, lex_order_vars):
            return True
        work.extend(node.args)
    return False


def _matches_demod_at(
    t: Term,
    alpha: Term,
    beta: Term,
    dtype: DemodType,
    symbol_table: SymbolTable,
    lex_order_vars: bool,
) -> bool:
    """Check if term t matches the demodulator at the top level."""
    # Try left-to-right
    if dtype in (DemodType.ORIENTED, DemodType.LEX_DEP_LR, DemodType.LEX_DEP_BOTH):
        ctx = Context()
        trail = Trail()
        if match(alpha, ctx, t, trail):
            if dtype == DemodType.ORIENTED:
                trail.undo()
                return True
            contractum = apply_demod(beta, ctx)
            trail.undo()
            if term_greater(t, contractum, lex_order_vars, symbol_table):
                return True
        else:
            trail.undo()

    # Try right-to-left
    if dtype in (DemodType.LEX_DEP_RL, DemodType.LEX_DEP_BOTH):
        ctx = Context()
        trail = Trail()
        if match(beta, ctx, t, trail):
            contractum = apply_demod(alpha, ctx)
            trail.undo()
            if term_greater(t, contractum, lex_order_vars, symbol_table):
                return True
        else:
            trail.undo()

    return False
