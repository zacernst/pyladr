"""Mace4 model search algorithm matching C mace4.src/msearch.c.

Implements the backtracking search with constraint propagation:
1. Collect symbols from clauses
2. For each domain size: initialize cells, generate ground clauses, search
3. Search: select cell → try values → propagate → recurse → backtrack

Ground clauses are represented as lists of (positive, sym_name, indices, value)
tuples, where positive indicates the polarity.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import EQ_SYM, SymbolTable
from pyladr.core.term import Term
from pyladr.mace4.model import (
    Cell,
    FiniteModel,
    ModelResult,
    SymbolInfo,
    SymbolType,
)


# ── Ground literal representation ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GroundLiteral:
    """A ground literal in the model.

    Represents sym(i1,...,in) = value (positive) or sym(i1,...,in) != value (negative).
    For relations: value is 0 or 1. For equality: compare two cell lookups.

    When negated=False (default): literal true iff cell.value == value.
    When negated=True: literal true iff cell.value != value.
    """
    positive: bool
    cell_id: int        # Which cell this refers to
    value: int          # Expected value
    negated: bool = False  # If True, literal is true when cell.value != value


@dataclass(slots=True)
class GroundClause:
    """A ground clause (disjunction of ground literals)."""
    literals: list[GroundLiteral]
    satisfied: bool = False  # True if at least one literal is true
    active_count: int = 0    # Number of not-yet-false literals

    def __post_init__(self) -> None:
        self.active_count = len(self.literals)


# ── Search options ───────────────────────────────────────────────────────────


@dataclass(slots=True)
class SearchOptions:
    """Mace4 search options."""
    start_size: int = 2
    end_size: int = 10
    max_models: int = 1
    max_seconds: float = 60.0
    increment: int = 1
    print_models: bool = True


# ── Core search engine ───────────────────────────────────────────────────────


class ModelSearcher:
    """Mace4 model finder matching C mace4.src/msearch.c.

    Searches for finite models that satisfy a set of clauses.
    Iterates over domain sizes, generating ground instances of each clause
    and using backtracking search with unit propagation.

    Usage:
        searcher = ModelSearcher(symbol_table)
        result = searcher.search(clauses, options)
    """

    __slots__ = (
        "_symbol_table", "_model", "_ground_clauses",
        "_cell_watchers",
        "_stats_assignments", "_stats_propagations", "_stats_backtracks",
        "_start_time", "_max_seconds",
    )

    def __init__(self, symbol_table: SymbolTable) -> None:
        self._symbol_table = symbol_table
        self._model: FiniteModel | None = None
        self._ground_clauses: list[GroundClause] = []
        self._cell_watchers: dict[int, list[int]] = {}
        self._stats_assignments = 0
        self._stats_propagations = 0
        self._stats_backtracks = 0
        self._start_time = 0.0
        self._max_seconds = 60.0

    def search(
        self,
        clauses: list[Clause],
        options: SearchOptions | None = None,
    ) -> list[ModelResult]:
        """Search for finite models satisfying the clauses.

        Matches C mace4() main loop: iterates domain sizes from
        start_size to end_size.

        Args:
            clauses: Input clauses to satisfy.
            options: Search options.

        Returns:
            List of ModelResult objects (one per found model).
        """
        opts = options or SearchOptions()
        self._start_time = time.monotonic()
        self._max_seconds = opts.max_seconds
        results: list[ModelResult] = []

        # Collect symbols from clauses
        symbols = self._collect_symbols(clauses)

        # Iterate over domain sizes
        n = opts.start_size
        while n <= opts.end_size:
            if time.monotonic() - self._start_time > self._max_seconds:
                break

            result = self._search_domain_size(clauses, symbols, n)
            if result.found:
                results.append(result)
                if len(results) >= opts.max_models:
                    break

            n += opts.increment

        return results

    def _collect_symbols(self, clauses: list[Clause]) -> list[tuple[str, int, SymbolType]]:
        """Collect function/predicate symbols from clauses.

        Matches C collect_mace4_syms(): finds all rigid symbols
        and classifies them as functions or relations.
        """
        seen: dict[tuple[str, int], SymbolType] = {}

        for clause in clauses:
            for lit in clause.literals:
                self._collect_from_term(lit.atom, seen, is_atom=True)

        # Always include equality
        eq_key = (EQ_SYM, 2)
        if eq_key not in seen:
            seen[eq_key] = SymbolType.RELATION

        return [(name, arity, stype) for (name, arity), stype in seen.items()]

    def _collect_from_term(
        self,
        t: Term,
        seen: dict[tuple[str, int], SymbolType],
        is_atom: bool = False,
    ) -> None:
        """Recursively collect symbols from a term."""
        if t.is_variable:
            return

        try:
            name = self._symbol_table.sn_to_str(t.symnum)
        except KeyError:
            return

        key = (name, t.arity)

        if key not in seen:
            # Top-level of an atom → relation; subterm → function
            if is_atom:
                seen[key] = SymbolType.RELATION
            else:
                seen[key] = SymbolType.FUNCTION

        # Recurse into arguments (as function terms)
        for arg in t.args:
            self._collect_from_term(arg, seen, is_atom=False)

    def _search_domain_size(
        self,
        clauses: list[Clause],
        symbols: list[tuple[str, int, SymbolType]],
        domain_size: int,
    ) -> ModelResult:
        """Search for a model of a specific domain size.

        Matches C mace4n().
        """
        self._stats_assignments = 0
        self._stats_propagations = 0
        self._stats_backtracks = 0

        # Build model structure
        model = FiniteModel(domain_size)
        for name, arity, stype in symbols:
            model.add_symbol(name, arity, stype)
        model.initialize_cells()

        # Built-in assignments (equality table)
        model.setup_equality()

        self._model = model
        self._ground_clauses = []

        # Generate ground clauses
        for clause in clauses:
            self._generate_ground_clauses(clause, domain_size)

        # Build cell→clause watch lists for fast propagation
        self._build_watchers()

        # Initial unit propagation
        if not self._initial_propagate():
            return ModelResult(
                found=False, domain_size=domain_size,
                assignments=self._stats_assignments,
                propagations=self._stats_propagations,
                backtracks=self._stats_backtracks,
            )

        # Backtracking search
        found = self._backtrack_search()

        if found:
            return ModelResult(
                found=True,
                domain_size=domain_size,
                model=model.copy(),
                assignments=self._stats_assignments,
                propagations=self._stats_propagations,
                backtracks=self._stats_backtracks,
            )
        return ModelResult(
            found=False, domain_size=domain_size,
            assignments=self._stats_assignments,
            propagations=self._stats_propagations,
            backtracks=self._stats_backtracks,
        )

    # ── Ground clause generation ─────────────────────────────────────────

    def _generate_ground_clauses(self, clause: Clause, domain_size: int) -> None:
        """Generate all ground instances of a clause. Matches C generate_ground_clauses().

        Also enumerates over unassigned function cells that appear in each
        ground instance, temporarily assigning them to allow full evaluation
        of nested terms like *(e, x) or *(i(0), 0).
        """
        assert self._model is not None

        # Find variables
        var_set = clause.variables()
        var_list = sorted(var_set) if var_set else []
        domain = range(domain_size)

        # Enumerate over all variable assignments
        var_assignments = (
            itertools.product(domain, repeat=len(var_list))
            if var_list
            else [()]
        )

        for var_assignment in var_assignments:
            var_map = dict(zip(var_list, var_assignment)) if var_list else {}
            self._ground_with_cell_enum(clause, var_map, domain_size)

    def _ground_with_cell_enum(
        self,
        clause: Clause,
        var_map: dict[int, int],
        domain_size: int,
    ) -> None:
        """Ground a clause instance, enumerating over unresolved cells.

        Iteratively discovers and enumerates unresolved cells. Each level
        of enumeration may reveal new resolvable cells at the next level
        (for nested function terms).
        """
        assert self._model is not None
        domain = range(domain_size)

        # Iteratively find unresolved cells — assigning leaf cells may
        # make deeper cells resolvable
        unresolved = self._find_unresolved_cells(clause, var_map)

        if not unresolved:
            # All cells resolved — straightforward instantiation
            clauses = self._instantiate_clause(clause, var_map)
            self._ground_clauses.extend(clauses)
            return

        # Enumerate over unresolved cells, potentially multiple levels deep
        self._enum_cells_recursive(
            clause, var_map, domain_size, unresolved, [], []
        )

    def _enum_cells_recursive(
        self,
        clause: Clause,
        var_map: dict[int, int],
        domain_size: int,
        cells: list[Cell],
        assigned_cells: list[Cell],
        assigned_vals: list[int],
    ) -> None:
        """Recursively enumerate cell values and discover new unresolved cells."""
        assert self._model is not None
        domain = range(domain_size)

        for assignment in itertools.product(domain, repeat=len(cells)):
            # Temporarily assign these cells
            for cell, val in zip(cells, assignment):
                cell.value = val

            # Check if more unresolved cells are now discoverable
            new_unresolved = self._find_unresolved_cells(clause, var_map)

            all_assigned_cells = assigned_cells + list(cells)
            all_assigned_vals = assigned_vals + list(assignment)

            if new_unresolved:
                # Recurse to enumerate newly discovered cells
                self._enum_cells_recursive(
                    clause, var_map, domain_size,
                    new_unresolved, all_assigned_cells, all_assigned_vals,
                )
            else:
                # All cells resolved — instantiate
                clauses = self._instantiate_clause(clause, var_map)
                for gc in clauses:
                    # Add constraints linking cells to their temp values
                    for cell, val in zip(all_assigned_cells, all_assigned_vals):
                        gc.literals.insert(
                            0,
                            GroundLiteral(
                                positive=False,
                                cell_id=cell.cell_id,
                                value=val,
                                negated=True,
                            ),
                        )
                    gc.active_count = len(gc.literals)
                    self._ground_clauses.append(gc)

            # Restore cells
            for cell in cells:
                cell.value = None

    def _find_unresolved_cells(
        self, clause: Clause, var_map: dict[int, int]
    ) -> list[Cell]:
        """Find unassigned function cells in a ground clause instance.

        Collects cells bottom-up: constants first, then function applications
        whose arguments would be resolvable if we knew the inner cells.
        This identifies the minimal set of cells to enumerate over.
        """
        assert self._model is not None
        seen: set[int] = set()
        result: list[Cell] = []

        for lit in clause.literals:
            self._collect_unresolved_cells(lit.atom, var_map, seen, result)

        return result

    def _collect_unresolved_cells(
        self,
        t: Term,
        var_map: dict[int, int],
        seen: set[int],
        result: list[Cell],
    ) -> int | None:
        """Recursively find unresolved cells, bottom-up.

        Returns the resolved value of the term, or None if unresolvable.
        Adds unassigned cells to result as it encounters them.
        """
        assert self._model is not None
        if t.is_variable:
            return var_map.get(t.varnum)

        try:
            name = self._symbol_table.sn_to_str(t.symnum)
        except KeyError:
            return None

        if name == EQ_SYM:
            for arg in t.args:
                self._collect_unresolved_cells(arg, var_map, seen, result)
            return None

        if t.arity == 0:
            cell = self._model.get_cell(name, ())
            if cell is None:
                return None
            if cell.is_assigned:
                return cell.value
            if cell.cell_id not in seen:
                seen.add(cell.cell_id)
                result.append(cell)
            return None

        # Function with args: resolve args first
        arg_vals: list[int | None] = []
        for arg in t.args:
            v = self._collect_unresolved_cells(arg, var_map, seen, result)
            arg_vals.append(v)

        if any(v is None for v in arg_vals):
            # Can't determine which cell this is without knowing args
            return None

        indices = tuple(v for v in arg_vals if v is not None)
        cell = self._model.get_cell(name, indices)
        if cell is None:
            return None
        if cell.is_assigned:
            return cell.value
        if cell.cell_id not in seen:
            seen.add(cell.cell_id)
            result.append(cell)
        return None

    def _instantiate_clause(
        self,
        clause: Clause,
        var_map: dict[int, int],
    ) -> list[GroundClause]:
        """Instantiate a clause with ground values. Returns list of ground clauses.

        May return multiple clauses when equality between unresolved cells
        requires CNF encoding.
        """
        assert self._model is not None
        literals: list[GroundLiteral] = []
        deferred_eq: list[tuple[bool, int, int]] = []
        deferred_pred: list[tuple] = []

        for lit in clause.literals:
            gl = self._eval_literal(lit, var_map)
            if gl is None:
                # Literal evaluates to True → clause is tautology
                return []
            if gl is False:
                # Literal evaluates to False → skip it
                continue
            if isinstance(gl, tuple):
                if gl[0] == "pred":
                    deferred_pred.append(gl)
                else:
                    # Deferred equality: (positive, left_cell_id, right_cell_id)
                    deferred_eq.append(gl)  # type: ignore[arg-type]
            else:
                literals.append(gl)

        if not literals and not deferred_eq and not deferred_pred:
            # All literals false → unsatisfiable clause
            return [GroundClause(literals=[], satisfied=False, active_count=0)]

        if not deferred_eq and not deferred_pred:
            return [GroundClause(literals=literals)]

        # Handle deferred equalities by generating CNF constraints.
        result: list[GroundClause] = []
        for positive, left_cell, right_cell in deferred_eq:
            result.extend(
                self._equality_cnf(positive, left_cell, right_cell, literals)
            )

        # Handle deferred predicates
        for dp in deferred_pred:
            result.extend(
                self._predicate_cnf(dp, literals)
            )

        # If we only had deferred items with no extra literals,
        # make sure we generated something
        if not result and not deferred_eq and not deferred_pred:
            return [GroundClause(literals=literals)]

        return result

    def _equality_cnf(
        self,
        positive: bool,
        left_cell_id: int,
        right_cell_id: int,
        extra_lits: list[GroundLiteral],
    ) -> list[GroundClause]:
        """Generate CNF clauses for equality between two unassigned cells.

        For positive equality (a = b):
          For each pair (i, j) with i != j:
            clause: (a != i) ∨ (b != j) ∨ <extra_lits>
          This forbids a=i, b=j when i != j.

        For negative equality (a != b):
          For each value v:
            clause: (a != v) ∨ (b != v) ∨ <extra_lits>
          This forbids a=v, b=v for any v.
        """
        assert self._model is not None
        d = self._model.domain_size
        clauses: list[GroundClause] = []

        if positive:
            # a = b: forbid all (i, j) pairs where i != j
            for i in range(d):
                for j in range(d):
                    if i != j:
                        lits = [
                            GroundLiteral(positive=False, cell_id=left_cell_id, value=i, negated=True),
                            GroundLiteral(positive=False, cell_id=right_cell_id, value=j, negated=True),
                        ] + list(extra_lits)
                        clauses.append(GroundClause(literals=lits))
        else:
            # a != b: forbid all same-value pairs
            for v in range(d):
                lits = [
                    GroundLiteral(positive=False, cell_id=left_cell_id, value=v, negated=True),
                    GroundLiteral(positive=False, cell_id=right_cell_id, value=v, negated=True),
                ] + list(extra_lits)
                clauses.append(GroundClause(literals=lits))

        return clauses

    def _predicate_cnf(
        self,
        deferred: tuple,
        extra_lits: list[GroundLiteral],
    ) -> list[GroundClause]:
        """Generate CNF clauses for a predicate with one unresolved arg.

        For p(a) (positive) where a is an unresolved constant with cell ca:
          For each domain value v:
            clause: (ca != v) ∨ (p_cell(v) == expected)
          meaning: if a=v then p(v) must have the expected value.
        """
        assert self._model is not None
        d = self._model.domain_size
        _, sign, pred_name, resolved, pos, arg_cell_id = deferred
        expected = 1 if sign else 0
        clauses: list[GroundClause] = []

        for v in range(d):
            # Build indices with v substituted at the unresolved position
            indices = list(resolved)
            indices[pos] = v
            int_indices = tuple(i for i in indices if i is not None)
            pred_cell = self._model.get_cell(pred_name, int_indices)
            if pred_cell is None:
                continue

            if pred_cell.is_assigned:
                if pred_cell.value == expected:
                    # p(v) already satisfies — no constraint needed for this v
                    continue
                else:
                    # p(v) falsifies — ca must not be v
                    lits = [
                        GroundLiteral(positive=False, cell_id=arg_cell_id, value=v, negated=True),
                    ] + list(extra_lits)
                    clauses.append(GroundClause(literals=lits))
            else:
                # p(v) unassigned — generate: (ca != v) ∨ (pred_cell == expected)
                lits = [
                    GroundLiteral(positive=False, cell_id=arg_cell_id, value=v, negated=True),
                    GroundLiteral(positive=True, cell_id=pred_cell.cell_id, value=expected),
                ] + list(extra_lits)
                clauses.append(GroundClause(literals=lits))

        return clauses

    def _eval_literal(
        self,
        lit: Literal,
        var_map: dict[int, int],
    ) -> GroundLiteral | None | bool:
        """Evaluate a literal with ground substitution.

        Returns:
            GroundLiteral: if the literal can't be evaluated yet
            None: if the literal is already TRUE (clause satisfied)
            False: if the literal is already FALSE (skip)
        """
        assert self._model is not None
        atom = lit.atom

        try:
            name = self._symbol_table.sn_to_str(atom.symnum)
        except KeyError:
            return False

        if name == EQ_SYM:
            return self._eval_equality(lit.sign, atom, var_map)

        # Non-equality predicate/function
        indices = tuple(self._eval_term(arg, var_map) for arg in atom.args)

        # Check if any index is None (term can't be evaluated yet)
        if any(i is None for i in indices):
            # Try to generate constraints for predicates with constant args.
            # E.g., p(a) where a is an unresolved constant: expand over domain.
            return self._eval_predicate_with_unresolved_args(
                lit, name, atom, var_map
            )

        int_indices = tuple(i for i in indices if i is not None)
        cell = self._model.get_cell(name, int_indices)
        if cell is None:
            return False

        expected = 1 if lit.sign else 0

        if cell.is_assigned:
            if cell.value == expected:
                return None  # TRUE
            return False  # FALSE

        return GroundLiteral(positive=lit.sign, cell_id=cell.cell_id, value=expected)

    def _eval_equality(
        self,
        positive: bool,
        atom: Term,
        var_map: dict[int, int],
    ) -> GroundLiteral | tuple[bool, int, int] | None | bool:
        """Evaluate an equality literal.

        Returns:
            None: literal is TRUE (clause satisfied)
            False: literal is FALSE (skip)
            GroundLiteral: deferred single-cell constraint
            tuple[bool, int, int]: deferred equality (positive, left_cell, right_cell)
        """
        assert self._model is not None
        left_val = self._eval_term(atom.args[0], var_map)
        right_val = self._eval_term(atom.args[1], var_map)

        if left_val is not None and right_val is not None:
            # Both sides fully evaluated
            equal = (left_val == right_val)
            if positive == equal:
                return None  # TRUE
            return False  # FALSE

        # One or both sides unresolved — try to get cell IDs
        left_cell = self._get_term_cell_id(atom.args[0], var_map)
        right_cell = self._get_term_cell_id(atom.args[1], var_map)

        if left_val is not None and right_cell is not None:
            # Left resolved, right is a cell: cell must equal/not-equal left_val
            cell = self._model.cells[right_cell]
            if cell.is_assigned:
                equal = (left_val == cell.value)
                if positive == equal:
                    return None
                return False
            if positive:
                return GroundLiteral(positive=True, cell_id=right_cell, value=left_val)
            else:
                return GroundLiteral(positive=False, cell_id=right_cell, value=left_val, negated=True)

        if right_val is not None and left_cell is not None:
            # Right resolved, left is a cell
            cell = self._model.cells[left_cell]
            if cell.is_assigned:
                equal = (right_val == cell.value)
                if positive == equal:
                    return None
                return False
            if positive:
                return GroundLiteral(positive=True, cell_id=left_cell, value=right_val)
            else:
                return GroundLiteral(positive=False, cell_id=left_cell, value=right_val, negated=True)

        if left_cell is not None and right_cell is not None:
            # Both sides are unresolved cells — defer as equality constraint
            return (positive, left_cell, right_cell)

        return False  # Can't evaluate or get cell references

    def _get_term_cell_id(self, t: Term, var_map: dict[int, int]) -> int | None:
        """Get the cell ID for an unresolved term.

        Returns the cell_id if the term maps to a single model cell
        (constants, or functions with all args resolved). Returns None otherwise.
        """
        assert self._model is not None

        if t.is_variable:
            return None  # Variables should be resolved via var_map

        try:
            name = self._symbol_table.sn_to_str(t.symnum)
        except KeyError:
            return None

        if t.arity == 0:
            cell = self._model.get_cell(name, ())
            return cell.cell_id if cell is not None else None

        # Function with args: evaluate args to get indices
        indices: list[int] = []
        for arg in t.args:
            val = self._eval_term(arg, var_map)
            if val is None:
                return None
            indices.append(val)

        cell = self._model.get_cell(name, tuple(indices))
        return cell.cell_id if cell is not None else None

    def _eval_predicate_with_unresolved_args(
        self,
        lit: Literal,
        pred_name: str,
        atom: Term,
        var_map: dict[int, int],
    ) -> tuple | bool | None:
        """Handle predicates where some arguments are unresolved cells.

        For p(a) where a is an unresolved constant with cell ca:
        Returns a deferred predicate descriptor for _instantiate_clause
        to expand into CNF constraints.

        Returns False if unable to handle.
        """
        assert self._model is not None

        # Identify which args are resolved vs. unresolved cells
        resolved: list[int | None] = []
        unresolved_positions: list[int] = []
        unresolved_cells: list[int] = []

        for i, arg in enumerate(atom.args):
            val = self._eval_term(arg, var_map)
            if val is not None:
                resolved.append(val)
            else:
                cell_id = self._get_term_cell_id(arg, var_map)
                if cell_id is None:
                    return False
                resolved.append(None)
                unresolved_positions.append(i)
                unresolved_cells.append(cell_id)

        # Handle one unresolved arg
        if len(unresolved_positions) != 1:
            return False

        return ("pred", lit.sign, pred_name, resolved,
                unresolved_positions[0], unresolved_cells[0])

    def _eval_term(self, t: Term, var_map: dict[int, int]) -> int | None:
        """Evaluate a ground term to a domain element.

        Returns None if the term can't be fully evaluated (cell unassigned).
        """
        assert self._model is not None

        if t.is_variable:
            return var_map.get(t.varnum)

        try:
            name = self._symbol_table.sn_to_str(t.symnum)
        except KeyError:
            return None

        if t.arity == 0:
            # Constant: look up its cell
            cell = self._model.get_cell(name, ())
            if cell is not None and cell.is_assigned:
                return cell.value
            return None

        # Function application: evaluate arguments first
        arg_vals = []
        for arg in t.args:
            val = self._eval_term(arg, var_map)
            if val is None:
                return None
            arg_vals.append(val)

        cell = self._model.get_cell(name, tuple(arg_vals))
        if cell is not None and cell.is_assigned:
            return cell.value
        return None

    # ── Constraint propagation ───────────────────────────────────────────

    def _initial_propagate(self) -> bool:
        """Process initial unit ground clauses. Returns False if contradiction."""
        assert self._model is not None
        # First full evaluation
        self._update_clauses()

        changed = True
        while changed:
            if self._timed_out():
                return False
            changed = False
            for gc in self._ground_clauses:
                if gc.satisfied:
                    continue
                if gc.active_count == 0:
                    return False
                if gc.active_count == 1:
                    for gl in gc.literals:
                        cell = self._model.cells[gl.cell_id]
                        if not cell.is_assigned:
                            if gl.negated:
                                if not self._eliminate_value(cell, gl.value):
                                    return False
                            else:
                                if not self._assign_cell(cell, gl.value):
                                    return False
                            # _assign_cell already does targeted update
                            changed = True
                            break

        return True

    def _assign_cell(self, cell: Cell, value: int) -> bool:
        """Assign a value to a cell. Returns False if contradiction."""
        assert self._model is not None
        self._stats_assignments += 1

        if cell.is_assigned:
            return cell.value == value

        if cell.possible is not None and value not in cell.possible:
            return False

        cell.value = value
        self._stats_propagations += 1
        # Targeted clause update
        self._update_clauses_for_cell(cell.cell_id)
        return True

    def _eliminate_value(self, cell: Cell, value: int) -> bool:
        """Remove a value from a cell's possible set. Returns False if contradiction.

        If only one value remains after elimination, assigns the cell.
        Returns True even if value was already absent (but signals no-op via
        not changing anything).
        """
        if cell.is_assigned:
            return cell.value != value

        if cell.possible is None:
            if cell.symbol.stype == SymbolType.RELATION:
                cell.possible = {0, 1}
            else:
                assert self._model is not None
                cell.possible = set(range(self._model.domain_size))

        if value not in cell.possible:
            return True  # Already eliminated, no-op

        cell.possible.discard(value)

        if not cell.possible:
            return False

        if len(cell.possible) == 1:
            only_val = next(iter(cell.possible))
            return self._assign_cell(cell, only_val)

        # Value eliminated but not yet assigned — update clauses referencing this cell
        # so negated literals can be recognized as satisfied
        self._update_clauses_for_cell(cell.cell_id)
        return True

    def _build_watchers(self) -> None:
        """Build cell→clause watch lists for fast targeted updates."""
        self._cell_watchers = {}
        for gc_idx, gc in enumerate(self._ground_clauses):
            for gl in gc.literals:
                if gl.cell_id not in self._cell_watchers:
                    self._cell_watchers[gl.cell_id] = []
                self._cell_watchers[gl.cell_id].append(gc_idx)

    def _eval_gc(self, gc: GroundClause) -> None:
        """Re-evaluate a single ground clause."""
        assert self._model is not None
        if gc.satisfied:
            return
        active = 0
        satisfied = False
        for gl in gc.literals:
            cell = self._model.cells[gl.cell_id]
            if cell.is_assigned:
                if gl.negated:
                    if cell.value != gl.value:
                        satisfied = True
                        break
                else:
                    if cell.value == gl.value:
                        satisfied = True
                        break
            else:
                # For negated literals: if value already eliminated from
                # possible, the literal is effectively TRUE
                if gl.negated and cell.possible is not None and gl.value not in cell.possible:
                    satisfied = True
                    break
                active += 1
        gc.satisfied = satisfied
        gc.active_count = active if not satisfied else len(gc.literals)

    def _update_clauses_for_cell(self, cell_id: int) -> None:
        """Re-evaluate only clauses that reference the given cell."""
        for gc_idx in self._cell_watchers.get(cell_id, []):
            self._eval_gc(self._ground_clauses[gc_idx])

    def _update_clauses(self) -> None:
        """Re-evaluate all ground clauses after assignments."""
        for gc in self._ground_clauses:
            self._eval_gc(gc)

    # ── Backtracking search ──────────────────────────────────────────────

    def _timed_out(self) -> bool:
        """Check if the time limit has been exceeded."""
        return time.monotonic() - self._start_time > self._max_seconds

    def _backtrack_search(self) -> bool:
        """Main backtracking search. Matches C search() in msearch.c."""
        assert self._model is not None

        # Check time limit
        if self._timed_out():
            return False

        # Select next unassigned cell
        cell = self._select_cell()
        if cell is None:
            # All cells assigned → check if model is valid
            return self._verify_model()

        # Determine possible values
        if cell.possible is not None:
            values = sorted(cell.possible)
        elif cell.symbol.stype == SymbolType.RELATION:
            values = [0, 1]
        else:
            values = list(range(self._model.domain_size))

        # Apply Least Number Heuristic for functions:
        # A new domain element n can only appear if n-1 is already used.
        # This exploits symmetry: we only need to consider values up to
        # max_used + 1 (introducing at most one new element per assignment).
        if cell.symbol.stype == SymbolType.FUNCTION:
            max_used = self._max_used_element()
            limit = min(max_used + 2, self._model.domain_size)
            values = [v for v in values if v < limit]

        # Try each value
        for value in values:
            # Save state for backtracking
            saved = self._save_state()

            if self._assign_cell(cell, value):
                # _assign_cell already does targeted clause update
                if self._check_consistency():
                    if self._propagate():
                        if self._backtrack_search():
                            return True

            # Backtrack
            self._stats_backtracks += 1
            self._restore_state(saved)

        return False

    def _max_used_element(self) -> int:
        """Return the maximum domain element currently assigned to any function cell.

        Returns -1 if no function cells are assigned (allowing value 0 to be first).
        """
        assert self._model is not None
        max_val = -1
        for cell in self._model.cells:
            if cell.symbol.stype == SymbolType.FUNCTION and cell.is_assigned:
                if cell.value is not None and cell.value > max_val:
                    max_val = cell.value
        return max_val

    def _select_cell(self) -> Cell | None:
        """Select the next cell to assign. Matches C select_cell().

        Uses a simple strategy: pick the unassigned cell with the
        fewest remaining possible values (most constrained first).
        """
        assert self._model is not None
        best: Cell | None = None
        best_count = float("inf")

        for cell in self._model.cells:
            if cell.is_assigned:
                continue
            count = len(cell.possible) if cell.possible is not None else (
                2 if cell.symbol.stype == SymbolType.RELATION
                else self._model.domain_size
            )
            if count < best_count:
                best = cell
                best_count = count

        return best

    def _check_consistency(self) -> bool:
        """Check if any ground clause has become empty (contradiction)."""
        for gc in self._ground_clauses:
            if not gc.satisfied and gc.active_count == 0:
                return False
        return True

    def _propagate(self) -> bool:
        """Propagate unit clauses. Returns False if contradiction."""
        assert self._model is not None
        changed = True
        while changed:
            if self._timed_out():
                return False
            changed = False
            for gc in self._ground_clauses:
                if gc.satisfied:
                    continue
                if gc.active_count == 0:
                    return False
                if gc.active_count == 1:
                    for gl in gc.literals:
                        cell = self._model.cells[gl.cell_id]
                        if not cell.is_assigned:
                            if gl.negated:
                                if not self._eliminate_value(cell, gl.value):
                                    return False
                            else:
                                if not self._assign_cell(cell, gl.value):
                                    return False
                            # _assign_cell does targeted update
                            changed = True
                            break
        return True

    def _verify_model(self) -> bool:
        """Verify that all ground clauses are satisfied."""
        self._update_clauses()
        return all(gc.satisfied for gc in self._ground_clauses)

    # ── State save/restore for backtracking ──────────────────────────────

    def _save_state(self) -> tuple[list[tuple[int, int | None, set[int] | None]], list[tuple[int, bool, int]]]:
        """Save current model state for backtracking.

        Returns snapshots of cell values and clause states.
        """
        assert self._model is not None
        cell_state = [
            (c.cell_id, c.value, c.possible.copy() if c.possible else None)
            for c in self._model.cells
        ]
        clause_state = [
            (i, gc.satisfied, gc.active_count)
            for i, gc in enumerate(self._ground_clauses)
        ]
        return cell_state, clause_state

    def _restore_state(
        self,
        saved: tuple[list[tuple[int, int | None, set[int] | None]], list[tuple[int, bool, int]]],
    ) -> None:
        """Restore model state from a saved snapshot."""
        assert self._model is not None
        cell_state, clause_state = saved

        for cell_id, value, possible in cell_state:
            cell = self._model.cells[cell_id]
            cell.value = value
            cell.possible = possible

        for idx, satisfied, active_count in clause_state:
            gc = self._ground_clauses[idx]
            gc.satisfied = satisfied
            gc.active_count = active_count
