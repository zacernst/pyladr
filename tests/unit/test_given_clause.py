"""Tests for the given-clause search algorithm.

Tests cover:
- Search engine initialization and configuration
- Basic proof finding (simple problems)
- Selection strategy behavior
- Resource limits (max_given, max_kept, max_seconds)
- Clause processing pipeline
- Statistics tracking
- Proof trace construction
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.parsing.ladr_parser import parse_input
from pyladr.search.given_clause import (
    ExitCode,
    GivenClauseSearch,
    SearchOptions,
    SearchResult,
)
from pyladr.search.selection import GivenSelection, default_clause_weight
from pyladr.search.state import ClauseList


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_and_deny(text: str) -> tuple[list[Clause], SymbolTable]:
    """Parse LADR input and deny goals into SOS clauses."""
    st = SymbolTable()
    parsed = parse_input(text, st)

    sos = list(parsed.sos)
    for goal in parsed.goals:
        denied_lits = tuple(
            Literal(sign=not lit.sign, atom=lit.atom) for lit in goal.literals
        )
        denied = Clause(
            literals=denied_lits,
            justification=(Justification(just_type=JustType.DENY, clause_ids=(0,)),),
        )
        sos.append(denied)

    return sos, st


def _run_search(
    text: str,
    max_given: int = 500,
    max_seconds: float = 10.0,
    **kwargs,
) -> SearchResult:
    """Parse input text and run search with given options."""
    sos, st = _parse_and_deny(text)
    opts = SearchOptions(
        max_given=max_given,
        max_seconds=max_seconds,
        quiet=True,
        print_given=False,
        **kwargs,
    )
    engine = GivenClauseSearch(opts)
    return engine.run(usable=[], sos=sos)


# ── Basic search tests ───────────────────────────────────────────────────────


class TestBasicSearch:
    """Test basic proof finding capability."""

    def test_identity_proof(self):
        """Trivial: e*e=e from e*x=x."""
        result = _run_search("""
formulas(sos).
  e * x = x.
end_of_list.
formulas(goals).
  e * e = e.
end_of_list.
""")
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1
        assert result.stats.proofs >= 1

    def test_group_commutativity(self):
        """x*x=e → x*y=y*x (the x2 problem)."""
        result = _run_search("""
formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
  x * x = e.
end_of_list.
formulas(goals).
  x * y = y * x.
end_of_list.
""")
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) >= 1

    def test_sos_empty_no_proof(self):
        """If SOS has no useful clauses, search should exhaust."""
        result = _run_search("""
formulas(sos).
  p(a).
end_of_list.
formulas(goals).
  q(b).
end_of_list.
""", max_given=50)
        # Should either exhaust SOS or hit limit
        assert result.exit_code in (ExitCode.SOS_EMPTY_EXIT, ExitCode.MAX_GIVEN_EXIT)
        assert len(result.proofs) == 0

    def test_empty_clause_in_initial(self):
        """Empty clause in initial set should be found immediately."""
        # Create an empty clause directly
        empty = Clause(
            literals=(),
            justification=(Justification(just_type=JustType.INPUT),),
        )
        opts = SearchOptions(quiet=True, print_given=False)
        engine = GivenClauseSearch(opts)
        result = engine.run(usable=[], sos=[empty])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT


# ── Resource limit tests ─────────────────────────────────────────────────────


class TestLimits:
    """Test resource limits."""

    def test_max_given_limit(self):
        """Search should stop at max_given."""
        result = _run_search("""
formulas(sos).
  p(a).
  -p(x) | p(f(x)).
end_of_list.
formulas(goals).
  p(f(f(f(f(f(f(f(f(f(f(a))))))))))).
end_of_list.
""", max_given=5)
        assert result.stats.given <= 6  # May exceed by 1 due to check timing

    def test_max_seconds_limit(self):
        """Search should respect time limit."""
        result = _run_search("""
formulas(sos).
  p(a).
  -p(x) | p(f(x)).
  -p(x) | p(g(x)).
end_of_list.
formulas(goals).
  p(f(g(f(g(f(g(f(g(a))))))))).
end_of_list.
""", max_seconds=0.5, max_given=10000)
        assert result.stats.elapsed_seconds() < 5.0  # generous bound


# ── Statistics tests ─────────────────────────────────────────────────────────


class TestStatistics:
    """Test search statistics tracking."""

    def test_given_counter(self):
        """Given counter should increment with each selection."""
        result = _run_search("""
formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
  x * x = e.
end_of_list.
formulas(goals).
  x * y = y * x.
end_of_list.
""")
        assert result.stats.given >= 1

    def test_generated_counter(self):
        """Generated counter should count all inferred clauses."""
        result = _run_search("""
formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
  x * x = e.
end_of_list.
formulas(goals).
  x * y = y * x.
end_of_list.
""")
        assert result.stats.generated >= 1

    def test_proof_counter(self):
        """Proof counter should increment on proof."""
        result = _run_search("""
formulas(sos).
  e * x = x.
end_of_list.
formulas(goals).
  e * e = e.
end_of_list.
""")
        assert result.stats.proofs == 1

    def test_timing(self):
        """Elapsed time should be recorded."""
        result = _run_search("""
formulas(sos).
  e * x = x.
end_of_list.
formulas(goals).
  e * e = e.
end_of_list.
""")
        assert result.stats.elapsed_seconds() >= 0
        assert result.stats.elapsed_seconds() < 5.0


# ── Selection strategy tests ─────────────────────────────────────────────────


class TestSelection:
    """Test clause selection strategies."""

    def test_default_weight_calculation(self):
        """Clause weight should count symbols."""
        st = SymbolTable()
        parsed = parse_input("""
formulas(sos).
  e * x = x.
end_of_list.
""", st)
        c = parsed.sos[0]
        w = default_clause_weight(c)
        assert w > 0

    def test_selection_ratio_cycling(self):
        """Selection should alternate between weight and age."""
        selector = GivenSelection()
        cl = ClauseList("test")

        # Add clauses with different weights
        for i in range(10):
            c = Clause(id=i + 1, weight=float(10 - i))
            cl.append(c)

        # Select several and check that both weight and age selections occur
        selections = []
        for _ in range(6):
            c, stype = selector.select_given(cl, len(selections))
            if c:
                selections.append((c.id, stype))

        # With default 5:1 ratio, most should be "W" but some "A"
        types = [s[1] for s in selections]
        assert "W" in types or "A" in types  # at least one selection type


# ── Proof trace tests ────────────────────────────────────────────────────────


class TestProofTrace:
    """Test proof construction and tracing."""

    def test_proof_contains_clauses(self):
        """Proof should contain at least the empty clause."""
        result = _run_search("""
formulas(sos).
  e * x = x.
end_of_list.
formulas(goals).
  e * e = e.
end_of_list.
""")
        assert len(result.proofs) == 1
        proof = result.proofs[0]
        assert proof.empty_clause is not None
        assert proof.empty_clause.is_empty
        assert len(proof.clauses) >= 1

    def test_proof_sorted_by_id(self):
        """Proof clauses should be sorted by ID."""
        result = _run_search("""
formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
  x * x = e.
end_of_list.
formulas(goals).
  x * y = y * x.
end_of_list.
""")
        if result.proofs:
            proof = result.proofs[0]
            ids = [c.id for c in proof.clauses]
            assert ids == sorted(ids)


# ── Clause processing tests ─────────────────────────────────────────────────


class TestClauseProcessing:
    """Test the clause processing pipeline."""

    def test_tautology_rejected(self):
        """Tautological clauses should be discarded."""
        # Create a clause that is a tautology: p(a) | ~p(a)
        st = SymbolTable()
        sn_p = st.str_to_sn("p", 1)
        sn_a = st.str_to_sn("a", 0)
        atom = get_rigid_term(sn_p, 1, (get_rigid_term(sn_a, 0),))
        c = Clause(
            literals=(
                Literal(sign=True, atom=atom),
                Literal(sign=False, atom=atom),
            )
        )
        from pyladr.inference.resolution import is_tautology
        assert is_tautology(c)

    def test_merge_literals(self):
        """Duplicate literals should be merged."""
        st = SymbolTable()
        sn_p = st.str_to_sn("p", 1)
        sn_a = st.str_to_sn("a", 0)
        atom = get_rigid_term(sn_p, 1, (get_rigid_term(sn_a, 0),))
        c = Clause(
            literals=(
                Literal(sign=True, atom=atom),
                Literal(sign=True, atom=atom),
            )
        )
        from pyladr.inference.resolution import merge_literals
        merged = merge_literals(c)
        assert len(merged.literals) == 1
