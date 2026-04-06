"""Base types and utilities for term ordering.

Matches C order.h (Ordertype enum) and termorder.h (Order_method enum).
"""

from __future__ import annotations

from enum import IntEnum, auto


class Ordertype(IntEnum):
    """Result of comparing two terms. Matches C Ordertype."""

    NOT_COMPARABLE = 0
    SAME_AS = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    LESS_THAN_OR_SAME_AS = auto()
    GREATER_THAN_OR_SAME_AS = auto()
    NOT_LESS_THAN = auto()
    NOT_GREATER_THAN = auto()


class OrderMethod(IntEnum):
    """Ordering method selection. Matches C Order_method."""

    LRPO = 0
    LPO = auto()
    RPO = auto()
    KBO = auto()
