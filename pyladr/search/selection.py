"""Given clause selection strategies matching C giv_select.c.

Implements clause selection from the SOS list using weight-based,
age-based, and ratio-based strategies as in the C Prover9 implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Callable, NamedTuple

from pyladr.core.clause import Clause
from pyladr.search.priority_sos import PrioritySOS
from pyladr.search.state import ClauseList


class SelectionOrder(IntEnum):
    """Selection order types matching C GS_ORDER_* constants."""

    WEIGHT = 0   # Select lightest clause (C: GS_ORDER_WEIGHT)
    AGE = 1      # Select oldest clause (C: GS_ORDER_AGE)
    RANDOM = 2   # Select random clause (C: GS_ORDER_RANDOM)
    ENTROPY = 3  # Select highest-entropy clause (structural diversity)
    UNIFICATION_PENALTY = 4  # Select lowest-penalty clause (most specific preferred)
    FORTE = 5  # Select by FORTE embedding diversity score
    PROOF_GUIDED = 6  # Select by proof-guided exploitation/exploration blend
    TREE2VEC = 7  # Select by Tree2Vec structural embedding diversity
    TREE2VEC_MAXIMIN = 8  # Select by Tree2Vec maximin: highest floor similarity across all goals
    RNN2VEC = 9  # Select by RNN2Vec structural embedding diversity
    RNN2VEC_RANDOM_GOAL = 10  # Select SOS clause nearest to a randomly-chosen unproven goal


@dataclass(slots=True)
class SelectionRule:
    """A single selection rule matching C struct giv_select.

    Each rule has:
    - order: how to select (by weight, age, etc.)
    - part: ratio weight for this rule in the cycle
    - selected: count of clauses selected by this rule
    """

    name: str
    order: SelectionOrder
    part: int = 1
    selected: int = 0


@dataclass(slots=True)
class GivenSelection:
    """Clause selection manager matching C given_selection state.

    Implements the ratio-based selection strategy: cycle through
    selectors, picking from each according to its ratio weight.

    C uses two lists (High, Low) of selectors. For the initial
    implementation we use a simplified single-list approach matching
    the default Prover9 behavior: ratio of weight:age selection.
    """

    # Selection rules
    rules: list[SelectionRule] = field(default_factory=list)

    # Cycle state (C: current, count, cycle_size)
    _current_idx: int = 0
    _count: int = 0
    _cycle_size: int = 0

    def __post_init__(self) -> None:
        if not self.rules:
            # Default: Match C Prover9 ratio=5 behavior.
            # C Prover9 uses age_factor=5 meaning: select by age first,
            # then by weight (age_factor - 1) = 4 times, total cycle of 5.
            # Output pattern: (A), (T), (T), (T), (T), (A), ...
            # where T = Theme (weight-based selection).
            self.rules = [
                SelectionRule("A", SelectionOrder.AGE, part=1),
                SelectionRule("W", SelectionOrder.WEIGHT, part=4),
            ]
        self._cycle_size = sum(r.part for r in self.rules)
        self._count = 0
        self._current_idx = 0

    def add_clause_to_selectors(self, c: Clause) -> None:
        """Register a clause for future selection.

        Currently a no-op since select_given scans the SOS list directly.
        When heap-based selection is added, this will push to the heaps.
        """
        pass

    def select_given(
        self,
        sos: ClauseList,
        given_count: int,
    ) -> tuple[Clause | None, str]:
        """Select the next given clause from the SOS list.

        Matches C get_given_clause2() behavior:
        1. Determine which selector to use based on ratio cycle
        2. Select clause according to that rule's ordering
        3. Remove selected clause from SOS

        Args:
            sos: The set-of-support clause list.
            given_count: Current number of given clauses (for cycle tracking).

        Returns:
            (clause, selection_type) or (None, "") if SOS is empty.
        """
        if sos.is_empty:
            return None, ""

        # Find current selector in the ratio cycle
        rule = self._get_current_rule()
        self._advance_cycle()

        # PrioritySOS: pop methods handle removal internally
        if isinstance(sos, PrioritySOS):
            selected = self._pop_from_priority_sos(sos, rule.order)
        else:
            selected = self._select_by_order(sos, rule.order)
            if selected is not None:
                sos.remove(selected)

        if selected is None:
            return None, ""

        rule.selected += 1
        return selected, rule.name

    def _get_current_rule(self) -> SelectionRule:
        """Get the current selector based on ratio cycle position.

        Matches C ratio-based cycling through selectors.
        """
        pos = self._count % self._cycle_size
        cumulative = 0
        for rule in self.rules:
            cumulative += rule.part
            if pos < cumulative:
                return rule
        return self.rules[-1]  # fallback

    def _advance_cycle(self) -> None:
        """Advance the cycle counter."""
        self._count += 1

    @staticmethod
    def _pop_from_priority_sos(
        sos: PrioritySOS, order: SelectionOrder
    ) -> Clause | None:
        """Select and remove a clause from PrioritySOS. O(log n) / O(1)."""
        if order == SelectionOrder.AGE:
            return sos.pop_first()
        if order == SelectionOrder.WEIGHT:
            return sos.pop_lightest()
        if order == SelectionOrder.ENTROPY:
            return sos.pop_highest_entropy()
        if order == SelectionOrder.UNIFICATION_PENALTY:
            return sos.pop_lowest_penalty()
        if order == SelectionOrder.FORTE:
            return sos.pop_best_forte()
        if order == SelectionOrder.PROOF_GUIDED:
            return sos.pop_best_proof_guided()
        if order in (SelectionOrder.TREE2VEC, SelectionOrder.TREE2VEC_MAXIMIN):
            # Both T2V variants reuse the FORTE diversity heap for fallback.
            # The actual per-clause scoring is handled upstream in
            # GivenClauseSearch._make_inferences() before select_given() is
            # called, so this path is only reached when that scoring returns
            # None (no embeddings available yet).
            result = sos.pop_best_forte()
            return result if result is not None else sos.pop_first()
        if order in (SelectionOrder.RNN2VEC, SelectionOrder.RNN2VEC_RANDOM_GOAL):
            # Both RNN2Vec selection modes are handled upstream by GivenClauseSearch.
            # This fallback is reached when no embeddings are available yet.
            result = sos.pop_best_forte()
            return result if result is not None else sos.pop_first()
        # RANDOM: fall back to age
        return sos.pop_first()

    @staticmethod
    def _select_by_order(
        sos: ClauseList, order: SelectionOrder
    ) -> Clause | None:
        """Select a clause from SOS according to the given ordering.

        C uses AVL trees for efficient min-extraction.
        We use linear scan for correctness-first (optimize later).
        """
        if sos.is_empty:
            return None

        if order == SelectionOrder.AGE:
            # Oldest = first in list (FIFO order, matching C age-based)
            return sos.first

        if order == SelectionOrder.WEIGHT:
            # Lightest clause (C: AVL tree ordered by weight, then ID)
            best: Clause | None = None
            for c in sos:
                if best is None or _weight_compare(c, best) < 0:
                    best = c
            return best

        if order == SelectionOrder.ENTROPY:
            # Highest entropy clause (most structurally diverse)
            best_c: Clause | None = None
            best_entropy = -1.0
            for c in sos:
                e = _clause_entropy(c)
                if e > best_entropy or (e == best_entropy and (best_c is None or c.id < best_c.id)):
                    best_entropy = e
                    best_c = c
            return best_c

        if order == SelectionOrder.UNIFICATION_PENALTY:
            # Lowest penalty clause (most specific preferred)
            best_p: Clause | None = None
            best_penalty = float("inf")
            for c in sos:
                p = _clause_generality_penalty(c)
                if p < best_penalty or (p == best_penalty and (best_p is None or c.id < best_p.id)):
                    best_penalty = p
                    best_p = c
            return best_p

        if PrioritySOS.supports_order(order):
            # ML-based orders require heap-backed extraction that only
            # PrioritySOS provides.  This code path should never be
            # reached in practice because these orders are only added
            # when the corresponding flags force priority_sos=True.
            raise ValueError(
                f"SelectionOrder.{order.name} requires PrioritySOS. "
                f"Enable priority_sos or remove the {order.name.lower()} "
                f"selection rule."
            )

        # RANDOM: not implemented yet, fall back to age
        return sos.first


class NodeCounts(NamedTuple):
    """Counts of each node type in a clause tree."""

    clause: int = 0
    literal: int = 0
    predicate: int = 0
    function: int = 0
    variable: int = 0
    constant: int = 0


def _clause_entropy(clause: Clause) -> float:
    """Calculate Shannon entropy of a clause's node-type distribution.

    Classifies each node in the clause tree into one of 6 types:
    clause, literal, predicate, function, variable, constant.
    Returns H = -sum(p * log2(p)) over the type distribution.

    Performance: O(n) where n = total term nodes. Uses a flat array
    for counts to minimize overhead (no dict allocation per call).
    """
    # Mutable list for accumulation: [clause, literal, predicate, function, variable, constant]
    counts = [1, len(clause.literals), 0, 0, 0, 0]

    for lit in clause.literals:
        _count_nodes_flat(lit.atom, counts, True)

    total = counts[0] + counts[1] + counts[2] + counts[3] + counts[4] + counts[5]
    if total <= 1:
        return 0.0

    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def _count_nodes_flat(term, counts: list, is_predicate: bool) -> None:
    """Count term nodes into flat array.

    Indices: 0=clause, 1=literal, 2=predicate, 3=function, 4=variable, 5=constant.
    """
    if term.is_variable:
        counts[4] += 1
    elif term.is_constant:
        counts[5] += 1
    else:
        if is_predicate:
            counts[2] += 1
        else:
            counts[3] += 1
        for arg in term.args:
            _count_nodes_flat(arg, counts, False)


def _clause_generality_penalty(clause: Clause) -> float:
    """Calculate generality penalty for a clause.

    Penalizes overly general clauses that unify with too many others.
    Uses variable ratio (distinct_vars / total_nodes) as the base metric,
    with heavy penalties for single-literal all-variable patterns like P(x,y,z).

    Returns a penalty score >= 0.0. Lower = more specific (preferred).

    Performance: O(n) where n = total term nodes. Uses a flat set for
    distinct variable tracking and integer counter for total nodes.
    """
    num_literals = len(clause.literals)
    if num_literals == 0:
        return 0.0

    # Count total nodes and collect distinct variables in one pass
    total_nodes = 1 + num_literals  # clause node + literal nodes
    distinct_vars: set[int] = set()
    total_vars = 0

    for lit in clause.literals:
        _collect_var_stats(lit.atom, distinct_vars)
        total_nodes += lit.atom.symbol_count

    total_vars = len(distinct_vars)

    if total_nodes == 0:
        return 0.0

    # Base penalty: variable ratio
    var_ratio = total_vars / total_nodes

    # Heavy penalty for single-literal all-variable clauses: P(x, y, z)
    # These unify with everything and cause unification explosion
    if num_literals == 1:
        atom = clause.literals[0].atom
        if atom.arity > 0 and all(a.is_variable for a in atom.args):
            # All args are variables: maximum generality
            return 10.0 + var_ratio

    # Graduated penalty based on variable ratio
    # var_ratio near 1.0 = very general (many variables, few constants)
    # var_ratio near 0.0 = very specific (ground or mostly constants)
    return var_ratio


def _collect_var_stats(term, distinct_vars: set[int]) -> None:
    """Collect distinct variable numbers from a term tree. O(n)."""
    if term.is_variable:
        distinct_vars.add(term.varnum)
    else:
        for arg in term.args:
            _collect_var_stats(arg, distinct_vars)


def _weight_compare(a: Clause, b: Clause) -> int:
    """Compare clauses by weight, then by ID (tiebreaker).

    Matches C clause_compare_m4() ordering:
    lighter weight first, then smaller ID (older) first.
    """
    if a.weight != b.weight:
        return -1 if a.weight < b.weight else 1
    if a.id != b.id:
        return -1 if a.id < b.id else 1
    return 0


def default_clause_weight(c: Clause) -> float:
    """Default clause weight: number of symbols in all literals.

    Matches C clause_wt() default behavior.
    """
    total = 0
    for lit in c.literals:
        total += _term_symbol_count(lit.atom)
    return float(total)


def _term_symbol_count(t) -> int:
    """Count total symbols (non-variable nodes) in a term."""
    if t.is_variable:
        return 1
    count = 1  # this node
    for a in t.args:
        count += _term_symbol_count(a)
    return count
