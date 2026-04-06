"""Finite model representation matching C mace4.src/msearch.h.

A finite model assigns values to function/predicate cells over a domain
{0, 1, ..., n-1}. Each function/predicate symbol gets a block of cells
indexed by domain element tuples.

Cell ID calculation for symbol with base b and domain size d:
- Constant (arity 0): cell = b
- Unary f(i): cell = b + i
- Binary f(i,j): cell = b + i*d + j
- n-ary f(i1,...,in): cell = b + i1*d^(n-1) + ... + in
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto


class SymbolType(IntEnum):
    """Symbol type in the model."""
    FUNCTION = 0
    RELATION = auto()


@dataclass(slots=True)
class SymbolInfo:
    """Information about a symbol in the model. Matches C Symbol_data.

    Attributes:
        name: Symbol name.
        arity: Number of arguments.
        stype: FUNCTION or RELATION.
        base: Starting cell ID for this symbol's table.
        num_cells: Total cells for this symbol (domain_size^arity).
    """
    name: str
    arity: int
    stype: SymbolType
    base: int = 0
    num_cells: int = 0


@dataclass(slots=True)
class Cell:
    """Single cell in the interpretation table. Matches C struct cell.

    Each cell represents one entry in a function/predicate table:
    e.g., f(2,3) or p(1).

    Attributes:
        cell_id: Unique identifier (index into cells array).
        symbol: The symbol this cell belongs to.
        indices: Domain element indices (e.g., (2,3) for f(2,3)).
        value: Assigned value (None if unassigned). For functions: domain element.
               For relations: 0 (false) or 1 (true).
        possible: Set of still-possible values (None = use full range).
    """
    cell_id: int
    symbol: SymbolInfo
    indices: tuple[int, ...]
    value: int | None = None
    possible: set[int] | None = None

    @property
    def is_assigned(self) -> bool:
        return self.value is not None

    def max_index(self) -> int:
        """Maximum domain element index in this cell's arguments.
        Used for the Least Number Heuristic.
        """
        if not self.indices:
            return 0
        return max(self.indices)


@dataclass(slots=True)
class ModelResult:
    """Result of a model search attempt.

    Attributes:
        found: Whether a model was found.
        domain_size: Size of the domain.
        model: The model if found, None otherwise.
        assignments: Number of cell assignments tried.
        propagations: Number of constraint propagations.
        backtracks: Number of backtracks.
    """
    found: bool = False
    domain_size: int = 0
    model: FiniteModel | None = None
    assignments: int = 0
    propagations: int = 0
    backtracks: int = 0


class FiniteModel:
    """A finite model (interpretation) over domain {0, ..., n-1}.

    Matches C Mace4's model representation using cell tables.
    Each function/predicate symbol has a contiguous block of cells.

    Usage:
        model = FiniteModel(domain_size=3)
        model.add_symbol("f", arity=1, stype=SymbolType.FUNCTION)
        model.add_symbol("p", arity=1, stype=SymbolType.RELATION)
        model.initialize_cells()
        model.set_value("f", (0,), 1)  # f(0) = 1
        model.set_value("p", (2,), 1)  # p(2) = true
    """

    __slots__ = ("domain_size", "symbols", "cells", "_sym_by_name")

    def __init__(self, domain_size: int) -> None:
        self.domain_size = domain_size
        self.symbols: list[SymbolInfo] = []
        self.cells: list[Cell] = []
        self._sym_by_name: dict[str, SymbolInfo] = {}

    def add_symbol(self, name: str, arity: int, stype: SymbolType) -> SymbolInfo:
        """Add a symbol to the model."""
        sym = SymbolInfo(name=name, arity=arity, stype=stype)
        sym.num_cells = self.domain_size ** arity if arity > 0 else 1
        self.symbols.append(sym)
        self._sym_by_name[name] = sym
        return sym

    def get_symbol(self, name: str) -> SymbolInfo | None:
        """Look up a symbol by name."""
        return self._sym_by_name.get(name)

    def initialize_cells(self) -> None:
        """Allocate cells for all symbols. Assigns base offsets and creates Cell objects."""
        self.cells.clear()
        next_base = 0

        for sym in self.symbols:
            sym.base = next_base
            # Create cells for all index combinations
            for flat_idx in range(sym.num_cells):
                indices = self._flat_to_indices(flat_idx, sym.arity)
                max_val = self.domain_size if sym.stype == SymbolType.FUNCTION else 2
                cell = Cell(
                    cell_id=next_base + flat_idx,
                    symbol=sym,
                    indices=indices,
                    possible=set(range(max_val)),
                )
                self.cells.append(cell)
            next_base += sym.num_cells

    def _flat_to_indices(self, flat_idx: int, arity: int) -> tuple[int, ...]:
        """Convert a flat index to a tuple of domain indices.

        E.g., for arity=2, domain_size=3: flat_idx=5 → (1, 2)
        since 5 = 1*3 + 2.
        """
        if arity == 0:
            return ()
        indices: list[int] = []
        remaining = flat_idx
        for _ in range(arity):
            indices.append(remaining % self.domain_size)
            remaining //= self.domain_size
        return tuple(reversed(indices))

    def _indices_to_flat(self, indices: tuple[int, ...]) -> int:
        """Convert domain indices to a flat index."""
        flat = 0
        for idx in indices:
            flat = flat * self.domain_size + idx
        return flat

    def get_cell(self, sym_name: str, indices: tuple[int, ...]) -> Cell | None:
        """Get the cell for symbol(indices)."""
        sym = self._sym_by_name.get(sym_name)
        if sym is None:
            return None
        flat = self._indices_to_flat(indices)
        cell_id = sym.base + flat
        if 0 <= cell_id < len(self.cells):
            return self.cells[cell_id]
        return None

    def set_value(self, sym_name: str, indices: tuple[int, ...], value: int) -> bool:
        """Set a cell's value. Returns False if contradicts existing assignment."""
        cell = self.get_cell(sym_name, indices)
        if cell is None:
            return False
        if cell.value is not None:
            return cell.value == value
        cell.value = value
        return True

    def get_value(self, sym_name: str, indices: tuple[int, ...]) -> int | None:
        """Get a cell's value (None if unassigned)."""
        cell = self.get_cell(sym_name, indices)
        if cell is None:
            return None
        return cell.value

    def is_complete(self) -> bool:
        """Check if all cells have been assigned."""
        return all(c.is_assigned for c in self.cells)

    def unassigned_cells(self) -> list[Cell]:
        """Return list of cells without values."""
        return [c for c in self.cells if not c.is_assigned]

    def setup_equality(self) -> None:
        """Set up the equality predicate (= is true iff args are equal).

        Matches C built_in_assignments() for the equality symbol.
        """
        eq_sym = self._sym_by_name.get("=")
        if eq_sym is None:
            return
        for i in range(self.domain_size):
            for j in range(self.domain_size):
                val = 1 if i == j else 0
                self.set_value("=", (i, j), val)

    def format_model(self) -> str:
        """Format the model for output. Matches C print_model_standard()."""
        lines = [f"interpretation( {self.domain_size}, ["]
        entries = []
        for sym in self.symbols:
            if sym.name == "=":
                continue  # Skip built-in equality
            if sym.arity == 0:
                val = self.get_value(sym.name, ())
                if val is not None:
                    entries.append(f"    {sym.name} = {val}")
            else:
                for flat_idx in range(sym.num_cells):
                    indices = self._flat_to_indices(flat_idx, sym.arity)
                    val = self.get_value(sym.name, indices)
                    if val is not None:
                        args = ",".join(str(i) for i in indices)
                        if sym.stype == SymbolType.RELATION:
                            val_str = "true" if val == 1 else "false"
                            entries.append(f"    {sym.name}({args}) = {val_str}")
                        else:
                            entries.append(f"    {sym.name}({args}) = {val}")
        lines.append(",\n".join(entries))
        lines.append("]).")
        return "\n".join(lines)

    def copy(self) -> FiniteModel:
        """Create a deep copy of this model (for saving found models)."""
        new = FiniteModel(self.domain_size)
        for sym in self.symbols:
            new.add_symbol(sym.name, sym.arity, sym.stype)
        new.initialize_cells()
        for i, cell in enumerate(self.cells):
            new.cells[i].value = cell.value
            new.cells[i].possible = cell.possible.copy() if cell.possible else None
        return new
