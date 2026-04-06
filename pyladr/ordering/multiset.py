"""Multiset extension for term ordering.

Matches C multiset.c — greater_multiset() function used by LRPO/RPO
for comparing argument multisets.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pyladr.core.term import Term


class TermComparator(Protocol):
    """Protocol for term comparison functions (like lrpo, kbo)."""

    def __call__(self, s: Term, t: Term, lex_order_vars: bool) -> bool: ...


def _multiset_counts(terms: tuple[Term, ...]) -> Counter[int]:
    """Build a multiset (Counter) keyed by term identity.

    Uses id() as key since we need object-level grouping.
    We actually need structural identity, so we group by hash.
    """
    # Use a more robust approach: count by structural identity
    counts: dict[int, int] = {}
    for t in terms:
        # Group by (private_symbol, arity, args structure)
        # Use Python's default hash since Terms are frozen dataclasses
        h = hash(t)
        counts[h] = counts.get(h, 0) + 1
    return Counter(counts)


def _set_of_more_occurrences(
    a: tuple[Term, ...], b: tuple[Term, ...]
) -> list[Term]:
    """Find elements with more occurrences in a than in b.

    Matches C set_of_more_occurrences(). For each distinct term in a,
    if count(term, a) > count(term, b), include it in the result.
    """
    result: list[Term] = []
    seen: set[int] = set()
    for t in a:
        tid = id(t)
        if tid in seen:
            continue
        count_a = sum(1 for x in a if x.term_ident(t))
        count_b = sum(1 for x in b if x.term_ident(t))
        if count_a > count_b:
            result.append(t)
            seen.add(tid)
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
