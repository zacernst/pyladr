"""Term ordering dispatch matching C termorder.c.

Provides term_greater() and term_order() which dispatch to the
currently selected ordering method (KBO or LRPO).
"""

from __future__ import annotations

from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term
from pyladr.ordering.base import OrderMethod, Ordertype
from pyladr.ordering.kbo import kbo
from pyladr.ordering.lrpo import lrpo

# Global ordering method (C: Order_method Ordering_method = LRPO_METHOD)
_ordering_method: OrderMethod = OrderMethod.LRPO


def assign_order_method(method: OrderMethod) -> None:
    """Set the global ordering method. Matches C assign_order_method()."""
    global _ordering_method
    _ordering_method = method


def get_order_method() -> OrderMethod:
    """Get the current ordering method."""
    return _ordering_method


def term_greater(alpha: Term, beta: Term, lex_order_vars: bool, st: SymbolTable) -> bool:
    """Compare two terms: return True if alpha > beta.

    Matches C term_greater(). Dispatches to KBO or LRPO based on
    the current ordering method.
    """
    if _ordering_method == OrderMethod.KBO:
        return kbo(alpha, beta, lex_order_vars, st)
    return lrpo(alpha, beta, lex_order_vars, st)


def term_order(alpha: Term, beta: Term, st: SymbolTable) -> Ordertype:
    """Full four-way comparison. Matches C term_order().

    Returns GREATER_THAN, LESS_THAN, SAME_AS, or NOT_COMPARABLE.
    """
    if term_greater(alpha, beta, False, st):
        return Ordertype.GREATER_THAN
    if term_greater(beta, alpha, False, st):
        return Ordertype.LESS_THAN
    if alpha.term_ident(beta):
        return Ordertype.SAME_AS
    return Ordertype.NOT_COMPARABLE
