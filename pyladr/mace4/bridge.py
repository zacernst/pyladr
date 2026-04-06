"""Bridge between FiniteModel (Mace4 search) and Interpretation (evaluation).

Provides direct conversion functions so Mace4 search results can be
immediately evaluated, filtered, and formatted using the 15+ auxiliary
apps without going through text serialization.
"""

from __future__ import annotations

from pyladr.core.interpretation import (
    Interpretation,
    OperationTable,
    TableType,
)
from pyladr.mace4.model import FiniteModel, SymbolType


def finitemodel_to_interpretation(model: FiniteModel) -> Interpretation:
    """Convert a FiniteModel (cell-based) to an Interpretation (table-based).

    Maps each symbol's cell values into a flat operation table suitable
    for clause evaluation, isomorphism checking, and formatting.

    Skips the built-in equality symbol "=" (handled internally by eval).
    Raises ValueError if the model has unassigned cells.
    """
    interp = Interpretation(size=model.domain_size)

    for sym in model.symbols:
        if sym.name == "=":
            continue

        # Map SymbolType → TableType
        if sym.stype == SymbolType.FUNCTION:
            ttype = TableType.FUNCTION
        else:
            ttype = TableType.RELATION

        # Extract values in flat order (row-major, matching Interpretation convention)
        values: list[int] = []
        for flat_idx in range(sym.num_cells):
            indices = model._flat_to_indices(flat_idx, sym.arity)
            cell = model.get_cell(sym.name, indices)
            if cell is None or cell.value is None:
                raise ValueError(
                    f"Unassigned cell: {sym.name}{indices}"
                )
            values.append(cell.value)

        op = OperationTable(
            name=sym.name,
            arity=sym.arity,
            table_type=ttype,
            values=values,
        )
        interp.add_operation(op)

    return interp


def interpretation_to_finitemodel(interp: Interpretation) -> FiniteModel:
    """Convert an Interpretation (table-based) to a FiniteModel (cell-based).

    Useful for feeding known algebraic structures back into Mace4's
    constraint propagation or for comparison with search results.
    """
    model = FiniteModel(domain_size=interp.size)

    for name, op in interp.operations.items():
        if op.table_type == TableType.FUNCTION:
            stype = SymbolType.FUNCTION
        else:
            stype = SymbolType.RELATION
        model.add_symbol(name, op.arity, stype)

    model.initialize_cells()

    for name, op in interp.operations.items():
        sym = model.get_symbol(name)
        if sym is None:
            continue
        for flat_idx, val in enumerate(op.values):
            indices = model._flat_to_indices(flat_idx, sym.arity)
            model.set_value(name, indices, val)

    return model
