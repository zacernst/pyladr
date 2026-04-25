"""Multiset extension for term ordering.

Matches C multiset.c — greater_multiset() function used by LRPO/RPO
for comparing argument multisets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pyladr.core.term import Term


class TermComparator(Protocol):
    """Protocol for term comparison functions (like lrpo, kbo)."""

    def __call__(self, s: Term, t: Term, lex_order_vars: bool) -> bool: ...


def _set_of_more_occurrences(
    a: tuple[Term, ...], b: tuple[Term, ...]
) -> list[Term]:
    """Find elements with more occurrences in a than in b.

    Matches C set_of_more_occurrences(). For each distinct term in a,
    if count(term, a) > count(term, b), include it in the result.

    Uses hash-based counting for O(n) instead of O(n²).
    """
    # Build counts using Term objects directly as keys (frozen dataclasses)
    counts_a: dict[Term, int] = {}
    for t in a:
        counts_a[t] = counts_a.get(t, 0) + 1

    counts_b: dict[Term, int] = {}
    for t in b:
        counts_b[t] = counts_b.get(t, 0) + 1

    # Collect terms where count in a exceeds count in b
    result: list[Term] = []
    for t, count_a in counts_a.items():
        if count_a > counts_b.get(t, 0):
            result.append(t)
    return result


def greater_multiset(
    a: tuple[Term, ...],
    b: tuple[Term, ...],
    comp: TermComparator,
    lex_order_vars: bool,
) -> bool:
    """Multiset extension comparison. Matches C greater_multiset().

    Algorithm:
    1. s1 = elements with more occurrences in a than b
    2. s2 = elements with more occurrences in b than a
    3. Return: s1 is non-empty AND for every p2 in s2,
       there exists p1 in s1 such that comp(p1, p2)
    """
    s1 = _set_of_more_occurrences(a, b)
    s2 = _set_of_more_occurrences(b, a)

    if not s1:
        return False

    for p2 in s2:
        if not any(comp(p1, p2, lex_order_vars) for p1 in s1):
            return False
    return True
