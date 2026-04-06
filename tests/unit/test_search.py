"""Unit tests for given-clause search components.

Tests behavioral equivalence with C search.c / giv_select.c:
- ClauseList operations (usable, sos, limbo, disabled management)
- Given clause selection (weight-based, age-based, ratio cycling)
- SearchState management (ID assignment, indexing, clause lifecycle)
- SearchStatistics tracking
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.search.selection import (
    GivenSelection,
    SelectionOrder,
    SelectionRule,
    default_clause_weight,
)
from pyladr.search.state import ClauseList, SearchState
from pyladr.search.statistics import SearchStatistics


# ── Helpers ──────────────────────────────────────────────────────────────────


def _const(symnum: int) -> Term:
    return get_rigid_term(symnum, 0)


def _func(symnum: int, *args: Term) -> Term:
    return get_rigid_term(symnum, len(args), args)


def _make_clause(*atoms: Term, signs: tuple[bool, ...] | None = None) -> Clause:
    """Create a clause from atom terms with optional signs."""
    if signs is None:
        signs = (True,) * len(atoms)
    lits = tuple(Literal(sign=s, atom=a) for s, a in zip(signs, atoms))
    return Clause(literals=lits)


def _make_weighted_clause(weight: float, symnum: int = 1) -> Clause:
    """Create a clause and set its weight."""
    c = _make_clause(_const(symnum))
    c.weight = weight
    return c


# Symbol IDs
A, B, C_SYM = 1, 2, 3
F, G = 10, 11
P, Q = 20, 21


# ── ClauseList Tests ─────────────────────────────────────────────────────────


class TestClauseList:
    """Test ClauseList operations matching C clist behavior."""

    def test_empty_list(self) -> None:
        """New list is empty."""
        cl = ClauseList("test")
        assert cl.is_empty
        assert cl.length == 0
        assert cl.first is None

    def test_append_and_length(self) -> None:
        """Append increases length."""
        cl = ClauseList("test")
        c1 = _make_clause(_const(A))
        c2 = _make_clause(_const(B))
        cl.append(c1)
        cl.append(c2)
        assert cl.length == 2
        assert cl.first is c1

    def test_remove(self) -> None:
        """Remove by identity."""
        cl = ClauseList("test")
        c1 = _make_clause(_const(A))
        c2 = _make_clause(_const(B))
        cl.append(c1)
        cl.append(c2)
        assert cl.remove(c1) is True
        assert cl.length == 1
        assert cl.first is c2

    def test_remove_not_found(self) -> None:
        """Remove returns False when clause not in list."""
        cl = ClauseList("test")
        c = _make_clause(_const(A))
        assert cl.remove(c) is False

    def test_contains(self) -> None:
        """Contains checks membership."""
        cl = ClauseList("test")
        c1 = _make_clause(_const(A))
        c2 = _make_clause(_const(B))
        cl.append(c1)
        assert cl.contains(c1)
        assert not cl.contains(c2)

    def test_pop_first(self) -> None:
        """Pop first returns and removes first clause."""
        cl = ClauseList("test")
        c1 = _make_clause(_const(A))
        c2 = _make_clause(_const(B))
        cl.append(c1)
        cl.append(c2)
        popped = cl.pop_first()
        assert popped is c1
        assert cl.length == 1
        assert cl.first is c2

    def test_pop_first_empty(self) -> None:
        """Pop first on empty list returns None."""
        cl = ClauseList("test")
        assert cl.pop_first() is None

    def test_iter(self) -> None:
        """Iteration yields clauses in order."""
        cl = ClauseList("test")
        c1 = _make_clause(_const(A))
        c2 = _make_clause(_const(B))
        cl.append(c1)
        cl.append(c2)
        items = list(cl)
        assert items == [c1, c2]

    def test_len(self) -> None:
        """len() works on ClauseList."""
        cl = ClauseList("test")
        assert len(cl) == 0
        cl.append(_make_clause(_const(A)))
        assert len(cl) == 1


# ── GivenSelection Tests ────────────────────────────────────────────────────


class TestGivenSelection:
    """Test clause selection strategy matching C giv_select.c."""

    def test_default_rules(self) -> None:
        """Default rules are weight:5, age:1 ratio."""
        gs = GivenSelection()
        assert len(gs.rules) == 2
        assert gs.rules[0].order == SelectionOrder.WEIGHT
        assert gs.rules[0].part == 5
        assert gs.rules[1].order == SelectionOrder.AGE
        assert gs.rules[1].part == 1

    def test_select_by_weight(self) -> None:
        """Weight-based selection picks lightest clause."""
        sos = ClauseList("sos")
        heavy = _make_weighted_clause(10.0, A)
        heavy.id = 1
        light = _make_weighted_clause(3.0, B)
        light.id = 2
        medium = _make_weighted_clause(7.0, C_SYM)
        medium.id = 3
        sos.append(heavy)
        sos.append(light)
        sos.append(medium)

        # Force weight selection
        gs = GivenSelection(rules=[SelectionRule("W", SelectionOrder.WEIGHT)])
        selected, name = gs.select_given(sos, 0)
        assert selected is light
        assert name == "W"
        assert sos.length == 2  # removed from sos

    def test_select_by_age(self) -> None:
        """Age-based selection picks oldest (first) clause."""
        sos = ClauseList("sos")
        c1 = _make_weighted_clause(10.0, A)
        c1.id = 1
        c2 = _make_weighted_clause(3.0, B)
        c2.id = 2
        sos.append(c1)
        sos.append(c2)

        gs = GivenSelection(rules=[SelectionRule("A", SelectionOrder.AGE)])
        selected, name = gs.select_given(sos, 0)
        assert selected is c1  # oldest = first in list
        assert name == "A"

    def test_weight_tiebreak_by_id(self) -> None:
        """When weights are equal, older clause (lower ID) wins."""
        sos = ClauseList("sos")
        c1 = _make_weighted_clause(5.0, A)
        c1.id = 10
        c2 = _make_weighted_clause(5.0, B)
        c2.id = 5
        sos.append(c1)
        sos.append(c2)

        gs = GivenSelection(rules=[SelectionRule("W", SelectionOrder.WEIGHT)])
        selected, _ = gs.select_given(sos, 0)
        assert selected is c2  # lower ID wins tiebreak

    def test_ratio_cycling(self) -> None:
        """Default 5:1 ratio cycles weight 5x then age 1x."""
        gs = GivenSelection()  # default 5:1
        sos = ClauseList("sos")

        # Add enough clauses for cycling — all same weight, different IDs
        for i in range(10):
            c = _make_weighted_clause(5.0, A)
            c.id = i + 1
            sos.append(c)

        selections = []
        for i in range(6):
            _, name = gs.select_given(sos, i)
            selections.append(name)

        # First 5 should be weight ("W"), 6th should be age ("A")
        assert selections == ["W", "W", "W", "W", "W", "A"]

    def test_select_empty_sos(self) -> None:
        """Selecting from empty SOS returns None."""
        gs = GivenSelection()
        sos = ClauseList("sos")
        selected, name = gs.select_given(sos, 0)
        assert selected is None
        assert name == ""

    def test_custom_ratio(self) -> None:
        """Custom 2:1 ratio (weight:age)."""
        gs = GivenSelection(
            rules=[
                SelectionRule("W", SelectionOrder.WEIGHT, part=2),
                SelectionRule("A", SelectionOrder.AGE, part=1),
            ]
        )
        sos = ClauseList("sos")
        for i in range(6):
            c = _make_weighted_clause(5.0)
            c.id = i + 1
            sos.append(c)

        selections = []
        for i in range(6):
            _, name = gs.select_given(sos, i)
            selections.append(name)

        assert selections == ["W", "W", "A", "W", "W", "A"]


# ── SearchState Tests ────────────────────────────────────────────────────────


class TestSearchState:
    """Test SearchState management matching C Glob struct."""

    def test_initial_state(self) -> None:
        """Fresh state has empty lists."""
        state = SearchState()
        assert state.usable.is_empty
        assert state.sos.is_empty
        assert state.limbo.is_empty
        assert state.disabled.is_empty

    def test_assign_clause_id(self) -> None:
        """IDs assigned sequentially starting from 1."""
        state = SearchState()
        c1 = _make_clause(_const(A))
        c2 = _make_clause(_const(B))
        state.assign_clause_id(c1)
        state.assign_clause_id(c2)
        assert c1.id == 1
        assert c2.id == 2
        assert state.clause_ids_assigned() == 2

    def test_disable_clause(self) -> None:
        """Disabling moves clause from its list to disabled."""
        state = SearchState()
        c = _make_clause(_const(A))
        state.sos.append(c)
        assert state.sos.contains(c)

        state.disable_clause(c)
        assert not state.sos.contains(c)
        assert state.disabled.contains(c)

    def test_disable_from_usable(self) -> None:
        """Disabling from usable list works."""
        state = SearchState()
        c = _make_clause(_const(A))
        state.usable.append(c)
        state.disable_clause(c)
        assert not state.usable.contains(c)
        assert state.disabled.contains(c)


# ── SearchStatistics Tests ───────────────────────────────────────────────────


class TestSearchStatistics:
    """Test SearchStatistics counters matching C Stats struct."""

    def test_initial_zeros(self) -> None:
        """All counters start at zero."""
        stats = SearchStatistics()
        assert stats.given == 0
        assert stats.generated == 0
        assert stats.kept == 0
        assert stats.subsumed == 0
        assert stats.proofs == 0

    def test_counter_increment(self) -> None:
        """Counters can be incremented."""
        stats = SearchStatistics()
        stats.given += 1
        stats.generated += 10
        stats.kept += 5
        assert stats.given == 1
        assert stats.generated == 10
        assert stats.kept == 5

    def test_report_format(self) -> None:
        """Report generates expected format."""
        stats = SearchStatistics()
        stats.given = 12
        stats.generated = 118
        stats.kept = 23
        stats.proofs = 1
        report = stats.report()
        assert "given=12" in report
        assert "generated=118" in report
        assert "kept=23" in report
        assert "proofs=1" in report

    def test_timing(self) -> None:
        """Start/elapsed timing works."""
        stats = SearchStatistics()
        stats.start()
        assert stats.elapsed_seconds() >= 0.0
        assert stats.search_seconds() >= 0.0


# ── Clause Weight Tests ─────────────────────────────────────────────────────


class TestClauseWeight:
    """Test default clause weight matching C clause_wt()."""

    def test_constant_weight(self) -> None:
        """Single-constant clause has weight 1."""
        c = _make_clause(_const(A))
        w = default_clause_weight(c)
        assert w == 1.0

    def test_function_weight(self) -> None:
        """f(a, b) has weight 3 (f + a + b)."""
        c = _make_clause(_func(F, _const(A), _const(B)))
        w = default_clause_weight(c)
        assert w == 3.0

    def test_variable_weight(self) -> None:
        """Variables count as 1 symbol each."""
        x = get_variable_term(0)
        c = _make_clause(_func(F, x, _const(A)))
        w = default_clause_weight(c)
        assert w == 3.0  # f + x + a

    def test_multi_literal_weight(self) -> None:
        """Weight is sum across all literals."""
        c = _make_clause(_const(A), _func(F, _const(B)))
        w = default_clause_weight(c)
        assert w == 3.0  # a(1) + f(b)(2)

    def test_nested_function_weight(self) -> None:
        """f(g(a)) has weight 3."""
        c = _make_clause(_func(F, _func(G, _const(A))))
        w = default_clause_weight(c)
        assert w == 3.0  # f + g + a


# ── GivenClauseSearch Integration Tests ─────────────────────────────────────


from pyladr.search.given_clause import (
    ExitCode,
    GivenClauseSearch,
    Proof,
    SearchOptions,
    SearchResult,
)


def _pos_lit(atom: Term) -> tuple[bool, Term]:
    return (True, atom)


def _neg_lit(atom: Term) -> tuple[bool, Term]:
    return (False, atom)


def _make_clause_from_lits(*lits: tuple[bool, Term], id: int = 0) -> Clause:
    """Build a clause from (sign, atom) pairs with justification."""
    return Clause(
        literals=tuple(Literal(sign=s, atom=a) for s, a in lits),
        id=id,
        justification=(Justification(just_type=JustType.INPUT),),
    )


class TestGivenClauseBasic:
    """Test basic given-clause search scenarios."""

    def test_sos_empty_no_proof(self) -> None:
        """Empty SOS → SOS_EMPTY exit."""
        search = GivenClauseSearch()
        result = search.run(usable=[], sos=[])
        assert result.exit_code == ExitCode.SOS_EMPTY_EXIT
        assert len(result.proofs) == 0

    def test_immediate_empty_clause(self) -> None:
        """Empty clause in SOS → immediate proof."""
        empty = Clause(
            literals=(),
            justification=(Justification(just_type=JustType.INPUT),),
        )
        search = GivenClauseSearch()
        result = search.run(sos=[empty])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1

    def test_simple_resolution_proof(self) -> None:
        """P(a) and -P(a) → empty clause via resolution.

        Simplest theorem proving problem.
        """
        a = _const(A)
        pa = _func(P, a)

        c1 = _make_clause_from_lits(_pos_lit(pa))   # P(a)
        c2 = _make_clause_from_lits(_neg_lit(pa))    # -P(a)

        opts = SearchOptions(print_given=False, print_kept=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1, c2])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1
        assert result.stats.proofs == 1

    def test_resolution_with_variables(self) -> None:
        """P(x) and -P(a) resolve to empty clause."""
        a = _const(A)
        x = get_variable_term(0)
        px = _func(P, x)
        pa = _func(P, a)

        c1 = _make_clause_from_lits(_pos_lit(px))   # P(x)
        c2 = _make_clause_from_lits(_neg_lit(pa))    # -P(a)

        opts = SearchOptions(print_given=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1, c2])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1

    def test_two_step_proof(self) -> None:
        """Multi-step: P(a), -P(x)|Q(x), -Q(a) → proof.

        1. Resolve -P(x)|Q(x) with P(a): get Q(a)
        2. Resolve Q(a) with -Q(a): empty clause
        """
        a = _const(A)
        x = get_variable_term(0)
        p_x = _func(P, x)
        p_a = _func(P, a)
        q_x = _func(Q, x)
        q_a = _func(Q, a)

        c1 = _make_clause_from_lits(_pos_lit(p_a))                    # P(a)
        c2 = _make_clause_from_lits(_neg_lit(p_x), _pos_lit(q_x))     # -P(x)|Q(x)
        c3 = _make_clause_from_lits(_neg_lit(q_a))                     # -Q(a)

        opts = SearchOptions(print_given=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1, c2, c3])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert result.stats.proofs == 1


class TestGivenClauseLimits:
    """Test search limit enforcement."""

    def test_max_given_limit(self) -> None:
        """Search stops at max_given limit."""
        a = _const(A)
        b = _const(B)
        pa = _func(P, a)
        pb = _func(P, b)

        c1 = _make_clause_from_lits(_pos_lit(pa))
        c2 = _make_clause_from_lits(_pos_lit(pb))

        opts = SearchOptions(max_given=2, print_given=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1, c2])

        assert result.exit_code in (ExitCode.MAX_GIVEN_EXIT, ExitCode.SOS_EMPTY_EXIT)
        assert result.stats.given <= 2


class TestGivenClauseSimplification:
    """Test clause simplification during search."""

    def test_tautology_deleted(self) -> None:
        """Tautologies are detected and deleted during processing."""
        a = _const(A)
        pa = _func(P, a)

        c1 = _make_clause_from_lits(_pos_lit(pa))

        opts = SearchOptions(print_given=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1])

        assert result.exit_code == ExitCode.SOS_EMPTY_EXIT


class TestProofTracing:
    """Test proof tracing through justifications."""

    def test_proof_contains_initial_clauses(self) -> None:
        """Proof trace includes the initial clauses used."""
        a = _const(A)
        pa = _func(P, a)

        c1 = _make_clause_from_lits(_pos_lit(pa))
        c2 = _make_clause_from_lits(_neg_lit(pa))

        opts = SearchOptions(print_given=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1, c2])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        proof = result.proofs[0]

        proof_ids = {c.id for c in proof.clauses}
        assert c1.id in proof_ids
        assert c2.id in proof_ids

    def test_proof_sorted_by_id(self) -> None:
        """Proof clauses are sorted by ID."""
        a = _const(A)
        pa = _func(P, a)

        c1 = _make_clause_from_lits(_pos_lit(pa))
        c2 = _make_clause_from_lits(_neg_lit(pa))

        opts = SearchOptions(print_given=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1, c2])

        proof = result.proofs[0]
        ids = [c.id for c in proof.clauses]
        assert ids == sorted(ids)


class TestGivenClauseFactoring:
    """Test factoring integration in search."""

    def test_factoring_finds_proof(self) -> None:
        """P(x)|P(y) factors to P(x), resolves with -P(a)."""
        x = get_variable_term(0)
        y = get_variable_term(1)
        a = _const(A)
        p_x = _func(P, x)
        p_y = _func(P, y)
        p_a = _func(P, a)

        c1 = _make_clause_from_lits(_pos_lit(p_x), _pos_lit(p_y))
        c2 = _make_clause_from_lits(_neg_lit(p_a))

        opts = SearchOptions(factoring=True, print_given=False)
        search = GivenClauseSearch(options=opts)
        result = search.run(sos=[c1, c2])

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
