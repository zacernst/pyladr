"""LRPO (Lexicographic Recursive Path Ordering) matching C termorder.c.

The C implementation uses the term lrpo() for all three RPO variants:
- LRPO: default, symbols can have LR (lexicographic) or multiset status
- LPO: all symbols use lexicographic status
- RPO: all symbols use multiset status

The algorithm is:
1. Variable handling (special case for lex_order_vars)
2. Same symbol with LR status: lexicographic comparison
3. Different symbols: precedence-based comparison
4. Same precedence: multiset comparison
"""

from __future__ import annotations

from pyladr.core.symbol import LrpoStatus, SymbolTable
from pyladr.core.term import Term
from pyladr.ordering.base import Ordertype
from pyladr.ordering.multiset import greater_multiset


def _sym_precedence(st: SymbolTable, sn1: int, sn2: int) -> Ordertype:
    """Compare symbol precedence. Matches C sym_precedence().

    Uses lex_val from symbol table. If either has unset precedence
    (represented as 0 here, INT_MAX in C), returns NOT_COMPARABLE.
    """
    if sn1 == sn2:
        return Ordertype.SAME_AS
    s1 = st.get_symbol(sn1)
    s2 = st.get_symbol(sn2)
    p1 = s1.lex_val
    p2 = s2.lex_val
    if p1 == 0 or p2 == 0:
        return Ordertype.NOT_COMPARABLE
    if p1 > p2:
        return Ordertype.GREATER_THAN
    if p1 < p2:
        return Ordertype.LESS_THAN
    return Ordertype.SAME_AS


def _lrpo_status(st: SymbolTable, symnum: int) -> LrpoStatus:
    """Get LRPO status for a symbol."""
    return st.get_symbol(symnum).lrpo_status


def lrpo(s: Term, t: Term, lex_order_vars: bool, st: SymbolTable) -> bool:
    """LRPO comparison: return True if s > t.

    Matches C lrpo() from termorder.c exactly.
    """
    if s.is_variable:
        if lex_order_vars:
            return t.is_variable and s.varnum > t.varnum
        return False

    if t.is_variable:
        if lex_order_vars:
            return True
        return t.occurs_in(s)  # s > var iff s properly contains that var

    if s.symnum == t.symnum and _lrpo_status(st, s.symnum) == LrpoStatus.LR_STATUS:
        # Same symbol with LR status — lexicographic comparison
        return _lrpo_lex(s, t, lex_order_vars, st)

    p = _sym_precedence(st, s.symnum, t.symnum)

    if p == Ordertype.SAME_AS:
        # Same precedence — multiset comparison
        return _lrpo_multiset(s, t, lex_order_vars, st)
    elif p == Ordertype.GREATER_THAN:
        # s has higher precedence — s must be greater than each arg of t
        return all(lrpo(s, t.args[i], lex_order_vars, st) for i in range(t.arity))
    else:
        # LESS_THAN or NOT_COMPARABLE — check if some arg of s >= t
        return any(
            s.args[i].term_ident(t) or lrpo(s.args[i], t, lex_order_vars, st)
            for i in range(s.arity)
        )


def _lrpo_lex(s: Term, t: Term, lex_order_vars: bool, st: SymbolTable) -> bool:
    """Lexicographic LRPO comparison for terms with same LR-status symbol.

    Matches C lrpo_lex() from termorder.c.
    """
    arity = s.arity
    # Skip identical leading arguments
    i = 0
    while i < arity and s.args[i].term_ident(t.args[i]):
        i += 1

    if i == arity:
        return False  # s and t are identical

    if lrpo(s.args[i], t.args[i], lex_order_vars, st):
        # s[i] > t[i] — check s > each remaining arg of t
        return all(
            lrpo(s, t.args[j], lex_order_vars, st) for j in range(i + 1, arity)
        )
    else:
        # s[i] <= t[i] — check if some remaining arg of s >= t
        return any(
            s.args[j].term_ident(t) or lrpo(s.args[j], t, lex_order_vars, st)
            for j in range(i + 1, arity)
        )


def _lrpo_multiset(s: Term, t: Term, lex_order_vars: bool, st: SymbolTable) -> bool:
    """Multiset LRPO comparison. Matches C lrpo_multiset()."""

    def comp(a: Term, b: Term, lov: bool) -> bool:
        return lrpo(a, b, lov, st)

    return greater_multiset(s.args, t.args, comp, lex_order_vars)
