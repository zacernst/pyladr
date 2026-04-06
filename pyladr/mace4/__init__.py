"""Mace4 finite model finder matching C mace4.src/.

Implements finite model construction via systematic search:
- Domain size iteration (start small, increase)
- Cell-based interpretation tables (function/predicate truth tables)
- Backtracking search with constraint propagation
- Ground clause generation and evaluation
"""

from pyladr.mace4.bridge import (
    finitemodel_to_interpretation,
    interpretation_to_finitemodel,
)
from pyladr.mace4.model import (
    Cell,
    FiniteModel,
    ModelResult,
    SymbolInfo,
)
from pyladr.mace4.search import ModelSearcher, SearchOptions

__all__ = [
    "Cell",
    "FiniteModel",
    "ModelResult",
    "ModelSearcher",
    "SearchOptions",
    "SymbolInfo",
    "finitemodel_to_interpretation",
    "interpretation_to_finitemodel",
]
