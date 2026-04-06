"""Knuth-Bendix Ordering (KBO) matching C termorder.c.

KBO compares terms based on:
1. Weight: sum of symbol weights (variables have weight 1)
2. Variable condition: V(beta) must be multisubset of V(alpha)
3. Precedence: symbol ordering as tiebreaker
4. Lexicographic: first differing argument when symbols are the same

KBO is a simplification ordering (well-founded and compatible with
the subterm property), which guarantees termination of rewriting.
"""

from __future__ import annotations

from collections import Counter

from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term
from pyladr.ordering.base import Ordertype
from pyladr.ordering.lrpo import _sym_precedence


def _var_multiset(t: Term) -> Counter[int]:
    """Collect multiset of variable occurrences in a term.

    Matches C multiset_vars() / multiset_of_vars().
    """
    counts: Counter[int] = Counter()
    if t.is_variable:
        counts[t.varnum] += 1
    else:
        for a in t.args:
            counts += _var_multiset(a)
    return counts


def _variables_multisubset(a: Term, b: Term) -> bool:
    """Check if V(a) is a multisubset of V(b).

    Matches C variables_multisubset(). Returns True if for every variable v,
    the number of occurrences of v in a is <= the number in b.
    """
    a_vars = _var_multiset(a)
    b_vars = _var_multiset(b)
    for var, count in a_vars.items():
        if b_vars.get(var, 0) < count:
            return False
    return True


def kbo_weight(t: Term, st: SymbolTable) -> int:
    """Calculate KBO weight of a term. Matches C kbo_weight().

    Variables have weight 1. Function symbols use their kb_weight
    from the symbol table (default 1). Total weight is sum of all nodes.
    """
    if t.is_variable:
        return 1
    wt = st.get_symbol(t.symnum).kb_weight
    for a in t.args:
        wt += kbo_weight(a, st)
    return wt


def kbo(alpha: Term, beta: Term, lex_order_vars: bool, st: SymbolTable) -> bool:
    """KBO comparison: return True if alpha > beta.

    Matches C kbo() from termorder.c exactly.

    Algorithm:
    1. Variable cases (same as LRPO)
    2. Same unary symbol: recurse on arguments
    3. Variable multisubset check: V(beta) ⊆ V(alpha)
    4. Weight comparison
    5. Equal weights: check V(alpha) = V(beta), then precedence, then args
    """
    # Variable cases
    if alpha.is_variable:
        if lex_order_vars:
            return beta.is_variable and alpha.varnum > beta.varnum
        return False

    if beta.is_variable:
        if lex_order_vars:
            return True
        return beta.occurs_in(alpha)  # alpha > var iff alpha contains that var

    # Same unary symbol: skip to argument (C optimization)
    if (
        alpha.arity == 1
        and beta.arity == 1
        and alpha.symnum == beta.symnum
    ):
        return kbo(alpha.args[0], beta.args[0], lex_order_vars, st)

    # Variable multisubset check: V(beta) must be multisubset of V(alpha)
    if not _variables_multisubset(beta, alpha):
        return False

    wa = kbo_weight(alpha, st)
    wb = kbo_weight(beta, st)

    if wa > wb:
        return True
    if wa < wb:
        return False

    # Equal weights — additional checks
    # V(alpha) must equal V(beta) (both multisubsets of each other)
    if not _variables_multisubset(alpha, beta):
        return False

    # Precedence comparison
    p = _sym_precedence(st, alpha.symnum, beta.symnum)
    if p == Ordertype.GREATER_THAN:
        return True
    if alpha.symnum != beta.symnum:
        return False

    # Same symbol, same weight — compare first differing argument
    i = 0
    while i < alpha.arity and alpha.args[i].term_ident(beta.args[i]):
        i += 1
    if i == alpha.arity:
        return False  # All arguments identical
    return kbo(alpha.args[i], beta.args[i], lex_order_vars, st)
