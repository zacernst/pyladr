"""Interpretation evaluation system matching C LADR interp.h/interp.c.

Provides compiled finite model interpretations for evaluating clauses
and formulas. This is the key infrastructure enabling 15+ auxiliary
applications (clausefilter, isofilter, interpformat, etc.).

Key operations:
- compile_interp(): Parse interpretation from term/text representation
- eval_clause(): Evaluate clause in interpretation (all instances)
- eval_formula_text(): Evaluate formula text in interpretation
- permute_interp(): Apply domain permutation
- isomorphic_interps(): Test isomorphism up to permutation
- Various output formatters (standard, portable, tabular, cooked, tex, xml, raw)
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TextIO

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term


# ── Constants matching C interp.c ────────────────────────────────────────────

MAX_VARS_EVAL = 100

class TableType(IntEnum):
    UNDEFINED = 0
    FUNCTION = 1
    RELATION = 2


class SemanticsResult(IntEnum):
    NOT_EVALUATED = 0
    NOT_EVALUABLE = auto()
    TRUE = auto()
    FALSE = auto()


# ── Table indexing macros (matching C I2, I3) ────────────────────────────────

def _i2(n: int, i: int, j: int) -> int:
    """2D to 1D index: i*n + j."""
    return i * n + j


def _i3(n: int, i: int, j: int, k: int) -> int:
    """3D to 1D index: i*n*n + j*n + k."""
    return i * n * n + j * n + k


# ── Interpretation ───────────────────────────────────────────────────────────


@dataclass
class OperationTable:
    """A single function or relation table in an interpretation."""

    name: str
    arity: int
    table_type: TableType
    values: list[int]  # flat array, indexed by domain^arity


@dataclass
class Interpretation:
    """Compiled finite model interpretation matching C struct interp.

    Domain is {0, 1, ..., size-1}. Operations are stored as flat
    arrays indexed by mixed-radix encoding of arguments.
    """

    size: int
    operations: dict[str, OperationTable] = field(default_factory=dict)
    # Keyed by (name, arity) for unique lookup
    _by_name_arity: dict[tuple[str, int], OperationTable] = field(
        default_factory=dict, repr=False
    )
    # Occurrence counts for isomorphism optimization
    occurrences: list[int] = field(default_factory=list)
    # Isomorphism optimization
    blocks: list[int] = field(default_factory=list)
    profile: list[list[int]] = field(default_factory=list)
    num_profile_components: int = 0
    discriminator_counts: list[int] = field(default_factory=list)
    incomplete: bool = False
    comments: str = ""

    def __post_init__(self) -> None:
        if not self.occurrences:
            self.occurrences = [0] * self.size
        if not self.blocks:
            self.blocks = [-1] * self.size
        if not self.profile:
            self.profile = [[] for _ in range(self.size)]

    def add_operation(self, op: OperationTable) -> None:
        """Add an operation table."""
        self.operations[op.name] = op
        self._by_name_arity[(op.name, op.arity)] = op
        # Count occurrences for functions
        if op.table_type == TableType.FUNCTION:
            for val in op.values:
                if 0 <= val < self.size:
                    self.occurrences[val] += 1

    def get_table(self, name: str, arity: int) -> OperationTable | None:
        """Look up an operation table by name and arity."""
        return self._by_name_arity.get((name, arity))

    def get_table_by_name(self, name: str) -> OperationTable | None:
        """Look up an operation table by name only."""
        return self.operations.get(name)

    def table_value(self, name: str, arity: int, *args: int) -> int:
        """Evaluate an operation on given arguments."""
        op = self._by_name_arity.get((name, arity))
        if op is None:
            raise ValueError(f"Symbol {name}/{arity} not in interpretation")
        idx = 0
        mult = 1
        for i in range(arity - 1, -1, -1):
            idx += args[i] * mult
            mult *= self.size
        return op.values[idx]


# ── Compilation ──────────────────────────────────────────────────────────────


def compile_interp_from_text(text: str, allow_incomplete: bool = False) -> Interpretation:
    """Parse an interpretation from text format.

    Supports the standard LADR interpretation format:
        interpretation(SIZE, [COMMENT], [
            function(SYMBOL(_,...), [VALUES...]),
            relation(SYMBOL(_,...), [VALUES...]),
            ...
        ])

    Also supports the simpler Mace4 output format.
    """
    text = text.strip()

    # Extract size
    size_match = re.match(r"interpretation\(\s*(\d+)", text)
    if not size_match:
        raise ValueError("Cannot parse interpretation: missing 'interpretation(SIZE, ...)'")

    domain_size = int(size_match.group(1))
    if domain_size < 1:
        raise ValueError(f"Domain size must be >= 1, got {domain_size}")

    interp = Interpretation(size=domain_size)

    # Extract operations
    # Find function(...) and relation(...) declarations
    op_pattern = re.compile(
        r"(function|relation)\s*\(\s*"
        r"(\w+)"                       # symbol name
        r"(\([^)]*\))?"                # optional args like (_,_)
        r"\s*,\s*\[([^\]]*)\]"         # value list [v1,v2,...]
        r"\s*\)"
    )

    for match in op_pattern.finditer(text):
        op_type_str = match.group(1)
        name = match.group(2)
        args_str = match.group(3) or ""
        values_str = match.group(4)

        # Determine arity from args
        if args_str:
            arity = args_str.count("_")
        else:
            arity = 0

        # Determine table type
        table_type = TableType.FUNCTION if op_type_str == "function" else TableType.RELATION

        # Parse values
        values: list[int] = []
        for v_str in values_str.split(","):
            v_str = v_str.strip()
            if not v_str:
                continue
            if v_str == "-":
                if allow_incomplete:
                    values.append(-1)
                    interp.incomplete = True
                else:
                    raise ValueError(f"Undefined value '-' in {name}")
            else:
                val = int(v_str)
                if table_type == TableType.FUNCTION and (val < 0 or val >= domain_size):
                    raise ValueError(
                        f"Function {name}: value {val} out of range [0, {domain_size - 1}]"
                    )
                if table_type == TableType.RELATION and val not in (0, 1):
                    raise ValueError(f"Relation {name}: value {val} not in {{0, 1}}")
                values.append(val)

        # Validate length
        expected_len = domain_size ** arity
        if len(values) != expected_len:
            raise ValueError(
                f"{name}: expected {expected_len} values for arity {arity} "
                f"and domain size {domain_size}, got {len(values)}"
            )

        op = OperationTable(name=name, arity=arity, table_type=table_type, values=values)
        interp.add_operation(op)

    return interp


# ── Evaluation ───────────────────────────────────────────────────────────────


def eval_term_ground(
    t: Term,
    interp: Interpretation,
    vals: list[int],
    symbol_table: SymbolTable | None = None,
) -> int:
    """Evaluate a ground term in an interpretation.

    Matches C eval_term_ground(): variables look up vals[], constants
    check domain membership, functions do table lookup.
    """
    if t.is_variable:
        return vals[t.varnum]

    # Get symbol name
    if symbol_table:
        name = symbol_table.sn_to_str(t.symnum)
    else:
        name = str(t.symnum)

    # Check if it's a numeric constant (domain element)
    if t.is_constant:
        try:
            domain_element = int(name)
            if 0 <= domain_element < interp.size:
                return domain_element
        except ValueError:
            pass

    # Table lookup
    op = interp.get_table_by_name(name)
    if op is None:
        # Try by symnum
        op = interp.operations.get(name)
        if op is None:
            raise ValueError(f"Symbol '{name}' not in interpretation")

    idx = 0
    mult = 1
    for i in range(t.arity - 1, -1, -1):
        v = eval_term_ground(t.args[i], interp, vals, symbol_table)
        idx += v * mult
        mult *= interp.size

    return op.values[idx]


def _eval_literals_ground(
    clause: Clause,
    interp: Interpretation,
    vals: list[int],
    symbol_table: SymbolTable | None = None,
    eq_symnum: int | None = None,
) -> bool:
    """Evaluate a ground clause: TRUE if at least one literal is true.

    Matches C eval_literals_ground().
    """
    for lit in clause.literals:
        atom = lit.atom

        # Check for equality
        is_eq = False
        if eq_symnum is not None and atom.is_complex and atom.symnum == eq_symnum:
            is_eq = True
        elif atom.is_complex and atom.arity == 2:
            # Heuristic: check if symbol name is "="
            if symbol_table:
                try:
                    name = symbol_table.sn_to_str(atom.symnum)
                    is_eq = name == "="
                except KeyError:
                    pass

        if is_eq:
            lhs = eval_term_ground(atom.args[0], interp, vals, symbol_table)
            rhs = eval_term_ground(atom.args[1], interp, vals, symbol_table)
            atom_val = lhs == rhs
        else:
            atom_val = bool(eval_term_ground(atom, interp, vals, symbol_table))

        literal_true = atom_val if lit.sign else not atom_val
        if literal_true:
            return True

    return False


def _all_recurse(
    clause: Clause,
    interp: Interpretation,
    vals: list[int],
    nextvar: int,
    nvars: int,
    symbol_table: SymbolTable | None = None,
    eq_symnum: int | None = None,
) -> bool:
    """Enumerate all instantiations and check clause is true in all.

    Matches C all_recurse().
    """
    if nextvar == nvars:
        return _eval_literals_ground(clause, interp, vals, symbol_table, eq_symnum)
    elif vals[nextvar] >= 0:
        return _all_recurse(clause, interp, vals, nextvar + 1, nvars, symbol_table, eq_symnum)
    else:
        for i in range(interp.size):
            vals[nextvar] = i
            if not _all_recurse(clause, interp, vals, nextvar + 1, nvars, symbol_table, eq_symnum):
                vals[nextvar] = -1
                return False
        vals[nextvar] = -1
        return True


def eval_clause(
    clause: Clause,
    interp: Interpretation,
    symbol_table: SymbolTable | None = None,
    eq_symnum: int | None = None,
) -> bool:
    """Evaluate a clause in an interpretation (all instances must be true).

    Matches C eval_literals(): enumerates all domain^nvars instantiations.
    Returns True if the clause is true in all instances.
    """
    # Find the maximum variable number
    max_var = -1
    for lit in clause.literals:
        for t in lit.atom.subterms():
            if t.is_variable and t.varnum > max_var:
                max_var = t.varnum

    nvars = max_var + 1
    if nvars > MAX_VARS_EVAL:
        raise ValueError(f"eval_clause: too many variables ({nvars})")

    vals = [-1] * max(nvars, 1)
    return _all_recurse(clause, interp, vals, 0, nvars, symbol_table, eq_symnum)


def _all_recurse_count(
    clause: Clause,
    interp: Interpretation,
    vals: list[int],
    nextvar: int,
    nvars: int,
    symbol_table: SymbolTable | None = None,
    eq_symnum: int | None = None,
) -> int:
    """Count true instances of a clause. Matches C all_recurse2()."""
    if nextvar == nvars:
        return 1 if _eval_literals_ground(clause, interp, vals, symbol_table, eq_symnum) else 0
    elif vals[nextvar] >= 0:
        return _all_recurse_count(clause, interp, vals, nextvar + 1, nvars, symbol_table, eq_symnum)
    else:
        count = 0
        for i in range(interp.size):
            vals[nextvar] = i
            count += _all_recurse_count(clause, interp, vals, nextvar + 1, nvars, symbol_table, eq_symnum)
        vals[nextvar] = -1
        return count


def eval_clause_true_instances(
    clause: Clause,
    interp: Interpretation,
    symbol_table: SymbolTable | None = None,
    eq_symnum: int | None = None,
) -> int:
    """Count true instances of a clause in an interpretation."""
    max_var = -1
    for lit in clause.literals:
        for t in lit.atom.subterms():
            if t.is_variable and t.varnum > max_var:
                max_var = t.varnum

    nvars = max_var + 1
    vals = [-1] * max(nvars, 1)
    return _all_recurse_count(clause, interp, vals, 0, nvars, symbol_table, eq_symnum)


def eval_clause_false_instances(
    clause: Clause,
    interp: Interpretation,
    symbol_table: SymbolTable | None = None,
    eq_symnum: int | None = None,
) -> int:
    """Count false instances of a clause in an interpretation."""
    max_var = -1
    for lit in clause.literals:
        for t in lit.atom.subterms():
            if t.is_variable and t.varnum > max_var:
                max_var = t.varnum

    nvars = max_var + 1
    total = interp.size ** nvars if nvars > 0 else 1
    true_count = eval_clause_true_instances(clause, interp, symbol_table, eq_symnum)
    return total - true_count


# ── Isomorphism ──────────────────────────────────────────────────────────────


def copy_interp(interp: Interpretation) -> Interpretation:
    """Deep copy an interpretation. Matches C copy_interp()."""
    return deepcopy(interp)


def permute_interp(source: Interpretation, perm: list[int]) -> Interpretation:
    """Apply a domain permutation to an interpretation.

    Returns a new interpretation where domain element i maps to perm[i].
    Matches C permute_interp().
    """
    n = source.size
    dest = Interpretation(size=n)
    dest.comments = source.comments

    for op in source.operations.values():
        st = op.values
        dt = [0] * len(st)
        is_function = op.table_type == TableType.FUNCTION

        if op.arity == 0:
            dt[0] = perm[st[0]] if is_function else st[0]
        elif op.arity == 1:
            for i in range(n):
                dt[perm[i]] = perm[st[i]] if is_function else st[i]
        elif op.arity == 2:
            for i in range(n):
                for j in range(n):
                    src_idx = _i2(n, i, j)
                    dst_idx = _i2(n, perm[i], perm[j])
                    dt[dst_idx] = perm[st[src_idx]] if is_function else st[src_idx]
        elif op.arity == 3:
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        src_idx = _i3(n, i, j, k)
                        dst_idx = _i3(n, perm[i], perm[j], perm[k])
                        dt[dst_idx] = perm[st[src_idx]] if is_function else st[src_idx]
        else:
            raise ValueError(f"permute_interp: arity {op.arity} > 3 not supported")

        new_op = OperationTable(
            name=op.name, arity=op.arity, table_type=op.table_type, values=dt
        )
        dest.operations[op.name] = new_op
        dest._by_name_arity[(op.name, op.arity)] = new_op

    # Permute occurrences
    dest.occurrences = [0] * n
    for i in range(n):
        dest.occurrences[perm[i]] = source.occurrences[i]

    # Permute blocks
    dest.blocks = [-1] * n
    for i in range(n):
        dest.blocks[perm[i]] = source.blocks[i]

    # Permute profile
    dest.num_profile_components = source.num_profile_components
    dest.profile = [[] for _ in range(n)]
    for i in range(n):
        dest.profile[perm[i]] = list(source.profile[i])

    return dest


def ident_interp_perm(a: Interpretation, b: Interpretation, perm: list[int]) -> bool:
    """Test if B equals A permuted by perm. Matches C ident_interp_perm().

    More efficient than permuting then comparing — checks in-place.
    """
    n = a.size
    for name, a_op in a.operations.items():
        b_op = b.operations.get(name)
        if b_op is None:
            return False

        at = a_op.values
        bt = b_op.values
        is_function = a_op.table_type == TableType.FUNCTION

        if a_op.arity == 0:
            expected = perm[at[0]] if is_function else at[0]
            if bt[0] != expected:
                return False
        elif a_op.arity == 1:
            for i in range(n):
                expected = perm[at[i]] if is_function else at[i]
                if bt[perm[i]] != expected:
                    return False
        elif a_op.arity == 2:
            for i in range(n):
                for j in range(n):
                    src_idx = _i2(n, i, j)
                    dst_idx = _i2(n, perm[i], perm[j])
                    expected = perm[at[src_idx]] if is_function else at[src_idx]
                    if bt[dst_idx] != expected:
                        return False
        elif a_op.arity == 3:
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        src_idx = _i3(n, i, j, k)
                        dst_idx = _i3(n, perm[i], perm[j], perm[k])
                        expected = perm[at[src_idx]] if is_function else at[src_idx]
                        if bt[dst_idx] != expected:
                            return False
        else:
            raise ValueError(f"ident_interp_perm: arity {a_op.arity} > 3")

    return True


def normal_interp(a: Interpretation) -> Interpretation:
    """Normalize interpretation by sorting elements by occurrence count.

    Matches C normal_interp(): permutes so occurrence counts are non-increasing.
    """
    n = a.size
    occ = list(a.occurrences)
    perm = [0] * n

    # Greedy: map most-occurring elements to lowest indices
    for i in range(n):
        max_val = -1
        max_idx = -1
        for j in range(n):
            if occ[j] > max_val:
                max_val = occ[j]
                max_idx = j
        perm[max_idx] = i
        occ[max_idx] = -1

    return permute_interp(a, perm)


def _iso_recurse(
    perm: list[int],
    k: int,
    n: int,
    a: Interpretation,
    b: Interpretation,
    use_blocks: bool,
) -> bool:
    """Recursive permutation enumeration for isomorphism checking.

    Matches C iso_interp_recurse(). Uses blocks to prune permutations.
    """
    if k == n:
        return ident_interp_perm(a, b, perm)

    # Try identity first
    if _iso_recurse(perm, k + 1, n, a, b, use_blocks):
        return True

    for i in range(k + 1, n):
        # If using blocks, only swap within same block
        if use_blocks and a.blocks[i] != a.blocks[k]:
            continue
        # Swap
        perm[k], perm[i] = perm[i], perm[k]
        if _iso_recurse(perm, k + 1, n, a, b, use_blocks):
            return True
        # Undo swap
        perm[k], perm[i] = perm[i], perm[k]

    return False


def isomorphic_interps(
    a: Interpretation,
    b: Interpretation,
    use_normal: bool = True,
) -> bool:
    """Test if two interpretations are isomorphic.

    Matches C isomorphic_interps(). When use_normal=True, uses occurrence-based
    blocks to prune the permutation search.
    """
    if a.size != b.size:
        return False

    # Quick check: same discriminator counts
    if a.discriminator_counts and b.discriminator_counts:
        if sorted(a.discriminator_counts) != sorted(b.discriminator_counts):
            return False

    # Quick check: same occurrence profiles
    if use_normal:
        if sorted(a.occurrences) != sorted(b.occurrences):
            return False

    n = a.size
    perm = list(range(n))
    return _iso_recurse(perm, 0, n, a, b, use_normal)


def ident_interp(a: Interpretation, b: Interpretation) -> bool:
    """Test if two interpretations are identical (not just isomorphic)."""
    if a.size != b.size:
        return False
    perm = list(range(a.size))
    return ident_interp_perm(a, b, perm)


def compare_interp(a: Interpretation, b: Interpretation) -> int:
    """Lexicographic comparison. Returns -1, 0, or 1."""
    if a.size != b.size:
        return -1 if a.size < b.size else 1

    for name in sorted(a.operations.keys()):
        a_op = a.operations.get(name)
        b_op = b.operations.get(name)
        if a_op is None or b_op is None:
            continue
        for av, bv in zip(a_op.values, b_op.values, strict=True):
            if av != bv:
                return -1 if av < bv else 1
    return 0


# ── Formatting ───────────────────────────────────────────────────────────────


def format_interp_standard(interp: Interpretation) -> str:
    """Format interpretation in standard LADR form."""
    lines: list[str] = []
    lines.append(f"interpretation( {interp.size}, [number=1, seconds=0], [")

    ops = sorted(interp.operations.values(), key=lambda o: (o.arity, o.name))
    for i, op in enumerate(ops):
        comma = "," if i < len(ops) - 1 else ""
        type_str = "function" if op.table_type == TableType.FUNCTION else "relation"
        if op.arity == 0:
            args_str = op.name
        else:
            args_str = f"{op.name}({','.join(['_'] * op.arity)})"
        values_str = ",".join(str(v) for v in op.values)
        lines.append(f"  {type_str}({args_str}, [{values_str}]){comma}")

    lines.append("]).")
    return "\n".join(lines)


def format_interp_standard2(interp: Interpretation) -> str:
    """Format with binary operations as squares. Matches C fprint_interp_standard2."""
    lines: list[str] = []
    lines.append(f"interpretation( {interp.size}, [number=1, seconds=0], [")

    ops = sorted(interp.operations.values(), key=lambda o: (o.arity, o.name))
    for idx, op in enumerate(ops):
        comma = "," if idx < len(ops) - 1 else ""
        type_str = "function" if op.table_type == TableType.FUNCTION else "relation"
        if op.arity == 0:
            args_str = op.name
        else:
            args_str = f"{op.name}({','.join(['_'] * op.arity)})"

        if op.arity <= 1:
            values_str = ",".join(str(v) for v in op.values)
            lines.append(f"  {type_str}({args_str}, [{values_str}]){comma}")
        elif op.arity == 2:
            n = interp.size
            lines.append(f"  {type_str}({args_str}, [")
            for i in range(n):
                row = op.values[i * n : (i + 1) * n]
                row_str = ",".join(str(v) for v in row)
                row_comma = "," if i < n - 1 else ""
                indent = "      " if i > 0 else "      "
                lines.append(f"{indent}{row_str}{row_comma}")
            lines.append(f"  ]){comma}")
        else:
            values_str = ",".join(str(v) for v in op.values)
            lines.append(f"  {type_str}({args_str}, [{values_str}]){comma}")

    lines.append("]).")
    return "\n".join(lines)


def format_interp_portable(interp: Interpretation) -> str:
    """Format as portable list-of-lists (Python/GAP parseable)."""
    parts: list[str] = []
    n = interp.size
    ops = sorted(interp.operations.values(), key=lambda o: (o.arity, o.name))

    for op in ops:
        if op.arity == 0:
            parts.append(f"  [\"{op.name}\", {op.values[0]}]")
        elif op.arity == 1:
            parts.append(f"  [\"{op.name}\", {op.values}]")
        elif op.arity == 2:
            rows = [op.values[i * n : (i + 1) * n] for i in range(n)]
            parts.append(f"  [\"{op.name}\", {rows}]")
        else:
            parts.append(f"  [\"{op.name}\", {op.values}]")

    return "[\n" + ",\n".join(parts) + "\n]"


def format_interp_tabular(interp: Interpretation) -> str:
    """Format as nice tables (arity <= 2 only)."""
    lines: list[str] = []
    n = interp.size

    for op in sorted(interp.operations.values(), key=lambda o: (o.arity, o.name)):
        type_str = "function" if op.table_type == TableType.FUNCTION else "relation"
        lines.append(f"\n{type_str}({op.name}/{op.arity}):")

        if op.arity == 0:
            lines.append(f"  {op.values[0]}")
        elif op.arity == 1:
            for i in range(n):
                lines.append(f"  {op.name}({i}) = {op.values[i]}")
        elif op.arity == 2:
            # Header row
            header = "    | " + " ".join(f"{j:2d}" for j in range(n))
            lines.append(header)
            lines.append("  --+" + "---" * n)
            for i in range(n):
                row = op.values[i * n : (i + 1) * n]
                row_str = " ".join(f"{v:2d}" for v in row)
                lines.append(f"  {i:2d}| {row_str}")

    return "\n".join(lines)


def format_interp_cooked(interp: Interpretation) -> str:
    """Format as human-readable equations: f(0,1)=2. Matches C fprint_interp_cooked."""
    lines: list[str] = []
    n = interp.size

    for op in sorted(interp.operations.values(), key=lambda o: (o.arity, o.name)):
        if op.table_type == TableType.FUNCTION:
            if op.arity == 0:
                lines.append(f"{op.name} = {op.values[0]}.")
            elif op.arity == 1:
                for i in range(n):
                    lines.append(f"{op.name}({i}) = {op.values[i]}.")
            elif op.arity == 2:
                for i in range(n):
                    for j in range(n):
                        val = op.values[_i2(n, i, j)]
                        lines.append(f"{op.name}({i},{j}) = {val}.")
            elif op.arity == 3:
                for i in range(n):
                    for j in range(n):
                        for k in range(n):
                            val = op.values[_i3(n, i, j, k)]
                            lines.append(f"{op.name}({i},{j},{k}) = {val}.")
        else:  # RELATION
            if op.arity == 0:
                if op.values[0]:
                    lines.append(f"{op.name}.")
            elif op.arity == 1:
                for i in range(n):
                    if op.values[i]:
                        lines.append(f"{op.name}({i}).")
            elif op.arity == 2:
                for i in range(n):
                    for j in range(n):
                        if op.values[_i2(n, i, j)]:
                            lines.append(f"{op.name}({i},{j}).")

    return "\n".join(lines)


def format_interp_raw(interp: Interpretation) -> str:
    """Raw table dump. Matches C fprint_interp_raw."""
    lines: list[str] = []

    for op in sorted(interp.operations.values(), key=lambda o: (o.arity, o.name)):
        type_str = "function" if op.table_type == TableType.FUNCTION else "relation"
        lines.append(f"{type_str} {op.name} {op.arity}")
        lines.append(" ".join(str(v) for v in op.values))

    return "\n".join(lines)


def format_interp_tex(interp: Interpretation) -> str:
    """Format as LaTeX tables. Matches C fprint_interp_tex."""
    lines: list[str] = []
    n = interp.size

    for op in sorted(interp.operations.values(), key=lambda o: (o.arity, o.name)):
        if op.arity == 2:
            lines.append(f"% {op.name}")
            cols = "|".join(["c"] * (n + 1))
            lines.append(f"\\begin{{tabular}}{{|{cols}|}}")
            lines.append("\\hline")
            header = f"${op.name}$" + "".join(f" & {j}" for j in range(n))
            lines.append(f"{header} \\\\")
            lines.append("\\hline")
            for i in range(n):
                row = op.values[i * n : (i + 1) * n]
                row_str = " & ".join(str(v) for v in row)
                lines.append(f"{i} & {row_str} \\\\")
            lines.append("\\hline")
            lines.append("\\end{tabular}")
            lines.append("")

    return "\n".join(lines)


def format_interp_xml(interp: Interpretation) -> str:
    """Format as XML. Matches C fprint_interp_xml."""
    lines: list[str] = []
    n = interp.size
    lines.append(f'<interp size="{n}">')

    for op in sorted(interp.operations.values(), key=lambda o: (o.arity, o.name)):
        tag = f"op{op.arity}" if op.arity <= 2 else "opn"
        type_str = "function" if op.table_type == TableType.FUNCTION else "relation"
        values_str = " ".join(str(v) for v in op.values)
        lines.append(f'  <{tag} type="{type_str}" name="{op.name}" '
                     f'arity="{op.arity}">{values_str}</{tag}>')

    lines.append("</interp>")
    return "\n".join(lines)


# ── Utility ──────────────────────────────────────────────────────────────────


def int_power(n: int, exp: int) -> int:
    """Compute n^exp, matching C int_power()."""
    return n ** exp


def factorial(n: int) -> int:
    """Compute n!, matching C factorial()."""
    return math.factorial(n)


def perms_required(interp: Interpretation) -> int:
    """Estimate permutations to check based on block structure.

    Matches C perms_required().
    """
    if all(b == -1 for b in interp.blocks):
        return factorial(interp.size)

    # Count block sizes
    block_counts: dict[int, int] = {}
    for b in interp.blocks:
        if b >= 0:
            block_counts[b] = block_counts.get(b, 0) + 1

    result = 1
    for count in block_counts.values():
        result *= factorial(count)
    return result
