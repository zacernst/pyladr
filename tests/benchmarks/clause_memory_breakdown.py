"""Per-clause memory breakdown profiler (REQ-P002 follow-up).

Measures the sys.getsizeof breakdown of Clauses to explain the ~3.6x
per-clause memory gap vs C Prover9 (5.8-6.0 KB PyLADR vs ~1.6 KB C).

Two modes:
  1. Parse a real input and profile initial/kept-ish clauses directly
     from the ParsedInput.
  2. Synthetic: construct representative unit equality clauses with
     f(g(x),c) = f(g(y),c) style atoms and measure.

Usage:
    python -m tests.benchmarks.clause_memory_breakdown <input.in>
    python -m tests.benchmarks.clause_memory_breakdown --synthetic
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pyladr.core.clause import Clause, Justification, Literal
from pyladr.core.term import Term, get_rigid_term, get_variable_term

# On CPython 64-bit, every GC-tracked object has a 16-byte PyGC_Head that
# sys.getsizeof() does NOT include. Glibc malloc also rounds allocations up
# to the next 16 bytes. We approximate the true heap footprint as:
#     getsizeof(obj) + GC_HEAD_SIZE  (if gc-tracked)
# then round up to 16-byte boundary.
GC_HEAD_SIZE = 16  # CPython 64-bit PyGC_Head


def _real_size(obj: object) -> int:
    import gc

    s = sys.getsizeof(obj)
    if gc.is_tracked(obj):
        s += GC_HEAD_SIZE
    # malloc 16-byte alignment rounding
    s = (s + 15) & ~15
    return s


@dataclass(slots=True)
class Breakdown:
    clause_count: int = 0
    clause_header: int = 0
    literals_tuple: int = 0
    literal_instances: int = 0
    term_instances: int = 0
    term_args_tuples: int = 0
    justification_tuple: int = 0
    justification_instances: int = 0
    justification_inner_tuples: int = 0
    given_selection_str: int = 0

    def per_clause_kb(self) -> dict[str, float]:
        n = max(self.clause_count, 1)
        return {
            "clause_header": self.clause_header / n / 1024,
            "literals_tuple": self.literals_tuple / n / 1024,
            "literal_instances": self.literal_instances / n / 1024,
            "term_instances": self.term_instances / n / 1024,
            "term_args_tuples": self.term_args_tuples / n / 1024,
            "justification_tuple": self.justification_tuple / n / 1024,
            "justification_instances": self.justification_instances / n / 1024,
            "justification_inner_tuples": self.justification_inner_tuples / n / 1024,
            "given_selection_str": self.given_selection_str / n / 1024,
        }

    def total_per_clause_bytes(self) -> float:
        n = max(self.clause_count, 1)
        return (
            self.clause_header
            + self.literals_tuple
            + self.literal_instances
            + self.term_instances
            + self.term_args_tuples
            + self.justification_tuple
            + self.justification_instances
            + self.justification_inner_tuples
            + self.given_selection_str
        ) / n


def _size_term_tree(t: Term, seen: set[int]) -> tuple[int, int]:
    term_bytes = 0
    tuple_bytes = 0
    stack: list[Term] = [t]
    while stack:
        node = stack.pop()
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)
        term_bytes += _real_size(node)
        if node.arity > 0:
            tuple_bytes += _real_size(node.args)
            stack.extend(node.args)
    return term_bytes, tuple_bytes


def _size_justification(j: Justification, seen: set[int]) -> tuple[int, int]:
    jid = id(j)
    if jid in seen:
        return 0, 0
    seen.add(jid)
    inst = _real_size(j)
    inner = 0
    if j.clause_ids:
        inner += _real_size(j.clause_ids)
    if j.demod_steps:
        inner += _real_size(j.demod_steps)
        for step in j.demod_steps:
            inner += _real_size(step)
    if j.para is not None and id(j.para) not in seen:
        seen.add(id(j.para))
        inst += _real_size(j.para)
        inner += _real_size(j.para.from_pos) + _real_size(j.para.into_pos)
    return inst, inner


def measure_clauses(clauses: Iterable[Clause]) -> Breakdown:
    """Accumulate size breakdown, deduplicating by id() for all shared objects.

    CPython interns `""` and short strings as singletons — a thousand Clauses
    whose given_selection is `""` share ONE string object, so counting it
    per-clause overstates the per-clause cost. We dedup strings and tuples
    by `id()` the same way we do Terms and Justifications.
    """
    b = Breakdown()
    term_seen: set[int] = set()
    just_seen: set[int] = set()
    str_seen: set[int] = set()
    tup_seen: set[int] = set()
    for c in clauses:
        b.clause_count += 1
        b.clause_header += _real_size(c)
        if id(c.literals) not in tup_seen:
            tup_seen.add(id(c.literals))
            b.literals_tuple += _real_size(c.literals)
        for lit in c.literals:
            b.literal_instances += _real_size(lit)
            t_bytes, args_bytes = _size_term_tree(lit.atom, term_seen)
            b.term_instances += t_bytes
            b.term_args_tuples += args_bytes
        if id(c.justification) not in tup_seen:
            tup_seen.add(id(c.justification))
            b.justification_tuple += _real_size(c.justification)
        for j in c.justification:
            inst, inner = _size_justification(j, just_seen)
            b.justification_instances += inst
            b.justification_inner_tuples += inner
        if id(c.given_selection) not in str_seen:
            str_seen.add(id(c.given_selection))
            b.given_selection_str += _real_size(c.given_selection)
    return b


def build_synthetic_clauses(n: int = 1000, intern: bool = True) -> list[Clause]:
    """Generate representative equality clauses like f(g(x,y), c1) = f(g(y,x), c2).

    Approx profile of a "typical" paramod-derived unit:
      - 1 literal (equality predicate, symnum=-1 arbitrarily)
      - atom: complex term of arity 2 (equality)
      - each side: complex term of arity 2 with one variable and one constant
      - depth 2-3, 4-6 symbols per side

    If intern=True, constants are built via get_rigid_term so they share
    one Term object across clauses (cycle 6 T8 optimization). If False,
    constants are built via direct Term() constructor — used to measure
    the interning delta.
    """
    EQ = 1
    F = 2
    G = 3
    C1 = 4
    C2 = 5
    clauses: list[Clause] = []
    for i in range(n):
        x = get_variable_term(0)
        y = get_variable_term(1)
        if intern:
            c1 = get_rigid_term(C1, 0)
            c2 = get_rigid_term(C2, 0)
        else:
            c1 = Term(private_symbol=-C1)
            c2 = Term(private_symbol=-C2)
        g_xy = Term(private_symbol=-G, arity=2, args=(x, y))
        g_yx = Term(private_symbol=-G, arity=2, args=(y, x))
        f_lhs = Term(private_symbol=-F, arity=2, args=(g_xy, c1))
        f_rhs = Term(private_symbol=-F, arity=2, args=(g_yx, c2))
        atom = Term(private_symbol=-EQ, arity=2, args=(f_lhs, f_rhs))
        lit = Literal(sign=True, atom=atom)
        # Representative justification: PARA with parent IDs
        from pyladr.core.clause import JustType, ParaJust
        para = ParaJust(
            from_id=i * 2 + 1,
            into_id=i * 2 + 2,
            from_pos=(1, 0),
            into_pos=(2, 1, 0),
        )
        j = Justification(just_type=JustType.PARA, para=para)
        # Add a secondary DEMOD step (common)
        d = Justification(
            just_type=JustType.DEMOD,
            demod_steps=((i + 100, 1, 0),),
        )
        c = Clause(
            literals=(lit,),
            id=i + 1000,
            weight=float(8 + (i % 10)),
            justification=(j, d),
        )
        clauses.append(c)
    return clauses


def collect_from_parsed_input(input_path: Path) -> list[Clause]:
    """Parse an input file and return all contained clauses (no search)."""
    from pyladr.parsing.ladr_parser import parse_input

    text = input_path.read_text()
    parsed = parse_input(text)
    clauses: list[Clause] = []
    for attr in ("sos", "usable", "goals", "formulas", "clauses"):
        lst = getattr(parsed, attr, None)
        if lst:
            for item in lst:
                if isinstance(item, Clause):
                    clauses.append(item)
    return clauses


def print_report(b: Breakdown, label: str) -> None:
    print(f"\n=== {label} (n={b.clause_count} clauses) ===")
    pc = b.per_clause_kb()
    total_bytes = b.total_per_clause_bytes()
    items = [
        ("Term instances (atoms+subterms)", pc["term_instances"]),
        ("Term args tuples", pc["term_args_tuples"]),
        ("Literal instances", pc["literal_instances"]),
        ("Literals tuple", pc["literals_tuple"]),
        ("Clause header (slots)", pc["clause_header"]),
        ("Justification instances", pc["justification_instances"]),
        ("Justification tuple", pc["justification_tuple"]),
        ("Justification inner tuples", pc["justification_inner_tuples"]),
        ("given_selection str (empty)", pc["given_selection_str"]),
    ]
    items.sort(key=lambda kv: kv[1], reverse=True)
    print(f"{'Component':<40}{'B/clause':>12}{'% of total':>12}")
    print("-" * 64)
    total_b = total_bytes
    for name, kb in items:
        bytes_pc = kb * 1024
        pct = (bytes_pc / total_b * 100) if total_b > 0 else 0
        print(f"{name:<40}{bytes_pc:>12.1f}{pct:>11.1f}%")
    print("-" * 64)
    print(f"{'TOTAL sys.getsizeof per-clause':<40}{total_b:>12.1f}{100.0:>11.1f}%")
    print(f"{'TOTAL KB':<40}{total_b/1024:>12.3f}")
    print(
        "\nNote: sys.getsizeof() reports only the immediate bytes of each\n"
        "object, not Python heap overhead (GC head, malloc rounding, hash\n"
        "table slots for unslotted objects). Real RSS cost per clause is\n"
        "higher than this baseline by ~30-50% — which is why total RSS growth\n"
        "(~5.8-6.0 KB/clause measured in REQ-P002) exceeds this sum."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="?", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true",
                        help="Profile synthetic representative clauses")
    parser.add_argument("-n", type=int, default=1000,
                        help="Synthetic: number of clauses to build")
    parser.add_argument("--no-intern", action="store_true",
                        help="Synthetic: disable rigid-constant interning "
                             "(to measure the interning delta)")
    parser.add_argument("--compare-intern", action="store_true",
                        help="Synthetic: run both intern=True and intern=False "
                             "and report both")
    args = parser.parse_args(argv)

    if args.compare_intern:
        # Must clear caches between runs to get honest per-clause cost
        from pyladr.core import term as _term_mod
        _term_mod._rigid_constant_cache.clear()
        b_intern = measure_clauses(build_synthetic_clauses(args.n, intern=True))
        print_report(b_intern, f"synthetic intern=True (n={args.n})")
        b_plain = measure_clauses(build_synthetic_clauses(args.n, intern=False))
        print_report(b_plain, f"synthetic intern=False (n={args.n})")
        delta = b_plain.total_per_clause_bytes() - b_intern.total_per_clause_bytes()
        print(f"\nInterning delta: {delta:+.1f} B/clause "
              f"({delta/1024:+.3f} KB/clause)")
        return

    if args.synthetic or not args.input_file:
        clauses = build_synthetic_clauses(args.n, intern=not args.no_intern)
        label = f"synthetic (intern={not args.no_intern}, n={args.n})"
    else:
        clauses = collect_from_parsed_input(Path(args.input_file))
        label = args.input_file

    b = measure_clauses(clauses)
    print_report(b, label)


if __name__ == "__main__":
    main()
