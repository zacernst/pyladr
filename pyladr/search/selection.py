"""Given clause selection strategies matching C giv_select.c.

Implements clause selection from the SOS list using weight-based,
age-based, and ratio-based strategies as in the C Prover9 implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Callable

from pyladr.core.clause import Clause
from pyladr.search.priority_sos import PrioritySOS
from pyladr.search.state import ClauseList


class SelectionOrder(IntEnum):
    """Selection order types matching C GS_ORDER_* constants."""

    WEIGHT = 0   # Select lightest clause (C: GS_ORDER_WEIGHT)
    AGE = 1      # Select oldest clause (C: GS_ORDER_AGE)
    RANDOM = 2   # Select random clause (C: GS_ORDER_RANDOM)


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
            # Default: Prover9 default is ratio=5:1 (weight:age)
            self.rules = [
                SelectionRule("W", SelectionOrder.WEIGHT, part=5),
                SelectionRule("A", SelectionOrder.AGE, part=1),
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

        # RANDOM: not implemented yet, fall back to age
        return sos.first


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
