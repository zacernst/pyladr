"""Tests for the paramodulation inference rule.

Tests cover:
- Equality detection (is_eq_atom, pos_eq, neg_eq)
- Equation orientation (orient_equalities, flip_eq)
- Core paramodulation (paramodulate function)
- Recursive descent (para_from_into generating all paramodulants)
- Position vector tracking for justifications
- Integration with the search loop
- Edge cases: variables, constants, nested equalities
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, JustType, Literal, ParaJust
from pyladr.core.substitution import Context, Trail, unify
from pyladr.core.symbol import EQ_SYM, SymbolTable
from pyladr.core.term import Term, get_variable_term
from pyladr.inference.paramodulation import (
    _oriented_eqs,
    _renamable_flips,
    flip_eq,
    is_eq_atom,
    is_oriented_eq,
    is_renamable_flip,
    mark_oriented_eq,
    mark_renamable_flip,
    neg_eq,
    orient_equalities,
    para_from_into,
    para_from_right,
    paramodulate,
    pos_eq,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_orientation_state():
    """Clear orientation flags between tests."""
    _oriented_eqs.clear()
    _renamable_flips.clear()
    yield
    _oriented_eqs.clear()
    _renamable_flips.clear()


@pytest.fixture
def st() -> SymbolTable:
    """Create a symbol table with common symbols."""
    s = SymbolTable()
    # Register equality
    s.str_to_sn("=", 2)
    # Register some function/predicate symbols
    s.str_to_sn("f", 1)
    s.str_to_sn("g", 1)
    s.str_to_sn("h", 2)
    s.str_to_sn("a", 0)
    s.str_to_sn("b", 0)
    s.str_to_sn("c", 0)
    s.str_to_sn("p", 1)
    s.str_to_sn("q", 1)
    s.str_to_sn("r", 2)
    return s


def _make_term(st: SymbolTable, name: str, *args: Term) -> Term:
    """Build a rigid term from symbol name and arguments."""
    arity = len(args)
    sn = st.str_to_sn(name, arity)
    if arity == 0:
        return Term(private_symbol=-sn)
    return Term(private_symbol=-sn, arity=arity, args=tuple(args))


def _var(n: int) -> Term:
    """Create a variable term."""
    return get_variable_term(n)


def _eq_atom(st: SymbolTable, left: Term, right: Term) -> Term:
    """Build an equality atom: left = right."""
    return _make_term(st, "=", left, right)


def _clause(*lits: Literal, clause_id: int = 0) -> Clause:
    """Build a clause from literals."""
    return Clause(literals=tuple(lits), id=clause_id)


# ── Equality detection tests ────────────────────────────────────────────────


class TestEqualityDetection:
    def test_is_eq_atom_true(self, st: SymbolTable):
        a = _make_term(st, "a")
        b = _make_term(st, "b")
        atom = _eq_atom(st, a, b)
        assert is_eq_atom(atom, st)

    def test_is_eq_atom_false_wrong_symbol(self, st: SymbolTable):
        a = _make_term(st, "a")
        b = _make_term(st, "b")
        atom = _make_term(st, "r", a, b)  # r(a,b) not equality
        assert not is_eq_atom(atom, st)

    def test_is_eq_atom_false_wrong_arity(self, st: SymbolTable):
        a = _make_term(st, "a")
        atom = _make_term(st, "f", a)  # f(a) is unary
        assert not is_eq_atom(atom, st)

    def test_is_eq_atom_variable(self, st: SymbolTable):
        assert not is_eq_atom(_var(0), st)

    def test_is_eq_atom_constant(self, st: SymbolTable):
        assert not is_eq_atom(_make_term(st, "a"), st)

    def test_pos_eq_positive(self, st: SymbolTable):
        atom = _eq_atom(st, _make_term(st, "a"), _make_term(st, "b"))
        lit = Literal(sign=True, atom=atom)
        assert pos_eq(lit, st)

    def test_pos_eq_negative(self, st: SymbolTable):
        atom = _eq_atom(st, _make_term(st, "a"), _make_term(st, "b"))
        lit = Literal(sign=False, atom=atom)
        assert not pos_eq(lit, st)

    def test_neg_eq(self, st: SymbolTable):
        atom = _eq_atom(st, _make_term(st, "a"), _make_term(st, "b"))
        lit = Literal(sign=False, atom=atom)
        assert neg_eq(lit, st)

    def test_neg_eq_positive_fails(self, st: SymbolTable):
        atom = _eq_atom(st, _make_term(st, "a"), _make_term(st, "b"))
        lit = Literal(sign=True, atom=atom)
        assert not neg_eq(lit, st)


# ── Equation orientation tests ───────────────────────────────────────────────


class TestEquationOrientation:
    def test_flip_eq(self, st: SymbolTable):
        a = _make_term(st, "a")
        b = _make_term(st, "b")
        atom = _eq_atom(st, a, b)
        flipped = flip_eq(atom)
        assert flipped.args[0] is b
        assert flipped.args[1] is a

    def test_mark_oriented_eq(self, st: SymbolTable):
        atom = _eq_atom(st, _make_term(st, "a"), _make_term(st, "b"))
        assert not is_oriented_eq(atom)
        mark_oriented_eq(atom)
        assert is_oriented_eq(atom)

    def test_mark_renamable_flip(self, st: SymbolTable):
        atom = _eq_atom(st, _var(0), _var(1))
        assert not is_renamable_flip(atom)
        mark_renamable_flip(atom)
        assert is_renamable_flip(atom)

    def test_para_from_right_oriented(self, st: SymbolTable):
        atom = _eq_atom(st, _make_term(st, "a"), _make_term(st, "b"))
        mark_oriented_eq(atom)
        assert not para_from_right(atom)

    def test_para_from_right_renamable(self, st: SymbolTable):
        atom = _eq_atom(st, _var(0), _var(1))
        mark_renamable_flip(atom)
        assert not para_from_right(atom)

    def test_para_from_right_unoriented(self, st: SymbolTable):
        atom = _eq_atom(st, _make_term(st, "a"), _make_term(st, "b"))
        # Neither oriented nor renamable
        assert para_from_right(atom)


# ── Core paramodulation tests ────────────────────────────────────────────────


class TestParamodulate:
    def test_simple_constant_replacement(self, st: SymbolTable):
        """a=b, p(a) => p(b)"""
        a = _make_term(st, "a")
        b = _make_term(st, "b")

        # from_clause: a = b (clause 1)
        eq_lit = Literal(sign=True, atom=_eq_atom(st, a, b))
        from_clause = _clause(eq_lit, clause_id=1)

        # into_clause: p(a) (clause 2)
        pa = _make_term(st, "p", a)
        p_lit = Literal(sign=True, atom=pa)
        into_clause = _clause(p_lit, clause_id=2)

        # Set up unification: alpha=a unifies with a in p(a)
        from_subst = Context()
        into_subst = Context()

        result = paramodulate(
            from_clause, eq_lit, 0,  # from left side (a)
            from_subst,
            into_clause, p_lit,
            (1,),  # position: first arg of p
            into_subst,
            st,
        )

        # Result should be p(b)
        assert len(result.literals) == 1
        assert result.literals[0].sign is True
        # The atom should be p(b)
        result_atom = result.literals[0].atom
        assert result_atom.arity == 1
        assert result_atom.args[0].term_ident(b)

    def test_paramodulation_justification(self, st: SymbolTable):
        """Check that justification is properly recorded."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")

        eq_lit = Literal(sign=True, atom=_eq_atom(st, a, b))
        from_clause = _clause(eq_lit, clause_id=10)

        pa = _make_term(st, "p", a)
        p_lit = Literal(sign=True, atom=pa)
        into_clause = _clause(p_lit, clause_id=20)

        from_subst = Context()
        into_subst = Context()

        result = paramodulate(
            from_clause, eq_lit, 0, from_subst,
            into_clause, p_lit, (1,), into_subst, st,
        )

        assert len(result.justification) == 1
        just = result.justification[0]
        assert just.just_type == JustType.PARA
        assert just.para is not None
        assert just.para.from_id == 10
        assert just.para.into_id == 20
        assert just.para.from_pos == (1, 1)  # lit 1, arg 1 (left side)
        assert just.para.into_pos == (1, 1)  # lit 1, arg 1

    def test_remaining_literals_included(self, st: SymbolTable):
        """Multi-literal clauses: remaining lits from both parents included."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")
        c = _make_term(st, "c")

        # from_clause: a=b | p(c) (clause 1)
        eq_lit = Literal(sign=True, atom=_eq_atom(st, a, b))
        pc_lit = Literal(sign=True, atom=_make_term(st, "p", c))
        from_clause = _clause(eq_lit, pc_lit, clause_id=1)

        # into_clause: q(a) | -r(a, c) (clause 2)
        qa_lit = Literal(sign=True, atom=_make_term(st, "q", a))
        rac_lit = Literal(sign=False, atom=_make_term(st, "r", a, c))
        into_clause = _clause(qa_lit, rac_lit, clause_id=2)

        from_subst = Context()
        into_subst = Context()

        # Paramodulate into first literal of into_clause, arg 1
        result = paramodulate(
            from_clause, eq_lit, 0, from_subst,
            into_clause, qa_lit, (1,), into_subst, st,
        )

        # Should have: p(c) (from from_clause) + q(b) + -r(a,c) (from into_clause)
        assert len(result.literals) == 3

    def test_paramodulate_from_right_side(self, st: SymbolTable):
        """b=a, p(a) => paramodulating from right side gives p(b)."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")

        # from_clause: b = a (clause 1)
        eq_lit = Literal(sign=True, atom=_eq_atom(st, b, a))
        from_clause = _clause(eq_lit, clause_id=1)

        # into_clause: p(a)
        pa = _make_term(st, "p", a)
        p_lit = Literal(sign=True, atom=pa)
        into_clause = _clause(p_lit, clause_id=2)

        from_subst = Context()
        into_subst = Context()

        # Paramodulate from right side (a is arg 1)
        result = paramodulate(
            from_clause, eq_lit, 1, from_subst,
            into_clause, p_lit, (1,), into_subst, st,
        )

        # beta is left side (b), so result should be p(b)
        assert len(result.literals) == 1
        assert result.literals[0].atom.args[0].term_ident(b)


# ── para_from_into tests ────────────────────────────────────────────────────


class TestParaFromInto:
    def test_simple_ground_para(self, st: SymbolTable):
        """a=b, p(a) => p(b)"""
        a = _make_term(st, "a")
        b = _make_term(st, "b")

        from_clause = _clause(
            Literal(sign=True, atom=_eq_atom(st, a, b)),
            clause_id=1,
        )
        into_clause = _clause(
            Literal(sign=True, atom=_make_term(st, "p", a)),
            clause_id=2,
        )

        results = para_from_into(from_clause, into_clause, False, st)
        assert len(results) >= 1

        # At least one result should have p(b)
        found_pb = False
        for r in results:
            for lit in r.literals:
                if lit.atom.arity == 1:
                    # Check if this is p(b) - the arg should be b
                    if lit.atom.args[0].term_ident(b):
                        found_pb = True
        assert found_pb, f"Expected p(b) in results, got {results}"

    def test_no_para_from_non_equality(self, st: SymbolTable):
        """Non-equality literals should not be used as from_lit."""
        a = _make_term(st, "a")

        # from_clause: p(a) (not an equality)
        from_clause = _clause(
            Literal(sign=True, atom=_make_term(st, "p", a)),
            clause_id=1,
        )
        into_clause = _clause(
            Literal(sign=True, atom=_make_term(st, "q", a)),
            clause_id=2,
        )

        results = para_from_into(from_clause, into_clause, False, st)
        assert len(results) == 0

    def test_no_para_from_negative_eq(self, st: SymbolTable):
        """Negative equalities should not be used as from_lit."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")

        from_clause = _clause(
            Literal(sign=False, atom=_eq_atom(st, a, b)),  # -(a=b)
            clause_id=1,
        )
        into_clause = _clause(
            Literal(sign=True, atom=_make_term(st, "p", a)),
            clause_id=2,
        )

        results = para_from_into(from_clause, into_clause, False, st)
        assert len(results) == 0

    def test_para_with_variables(self, st: SymbolTable):
        """f(x) = g(x), p(f(a)) => p(g(a))"""
        x = _var(0)
        a = _make_term(st, "a")
        fx = _make_term(st, "f", x)
        gx = _make_term(st, "g", x)
        fa = _make_term(st, "f", a)

        from_clause = _clause(
            Literal(sign=True, atom=_eq_atom(st, fx, gx)),
            clause_id=1,
        )
        into_clause = _clause(
            Literal(sign=True, atom=_make_term(st, "p", fa)),
            clause_id=2,
        )

        results = para_from_into(from_clause, into_clause, False, st)
        assert len(results) >= 1

    def test_para_into_nested_subterm(self, st: SymbolTable):
        """a=b, p(f(a)) => p(f(b))"""
        a = _make_term(st, "a")
        b = _make_term(st, "b")
        fa = _make_term(st, "f", a)

        from_clause = _clause(
            Literal(sign=True, atom=_eq_atom(st, a, b)),
            clause_id=1,
        )
        into_clause = _clause(
            Literal(sign=True, atom=_make_term(st, "p", fa)),
            clause_id=2,
        )

        results = para_from_into(from_clause, into_clause, False, st)
        # Should find paramodulant: p(f(b))
        assert len(results) >= 1

    def test_para_into_negative_literal(self, st: SymbolTable):
        """a=b, -p(a) => -p(b) (can paramodulate into negative literals)"""
        a = _make_term(st, "a")
        b = _make_term(st, "b")

        from_clause = _clause(
            Literal(sign=True, atom=_eq_atom(st, a, b)),
            clause_id=1,
        )
        into_clause = _clause(
            Literal(sign=False, atom=_make_term(st, "p", a)),
            clause_id=2,
        )

        results = para_from_into(from_clause, into_clause, False, st)
        assert len(results) >= 1

        # Check that result preserves negative sign
        for r in results:
            for lit in r.literals:
                if lit.atom.arity == 1:  # p(...)
                    assert lit.sign is False

    def test_no_para_into_variables_by_default(self, st: SymbolTable):
        """By default, should not paramodulate into variable positions."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")

        # from_clause: a=b
        from_clause = _clause(
            Literal(sign=True, atom=_eq_atom(st, a, b)),
            clause_id=1,
        )
        # into_clause: p(x) - x is a variable
        into_clause = _clause(
            Literal(sign=True, atom=_make_term(st, "p", _var(0))),
            clause_id=2,
        )

        results = para_from_into(from_clause, into_clause, False, st, para_into_vars=False)
        # Should not paramodulate into variable x
        assert len(results) == 0

    def test_para_into_variables_when_enabled(self, st: SymbolTable):
        """When para_into_vars=True, should paramodulate into variables."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")

        from_clause = _clause(
            Literal(sign=True, atom=_eq_atom(st, a, b)),
            clause_id=1,
        )
        into_clause = _clause(
            Literal(sign=True, atom=_make_term(st, "p", _var(0))),
            clause_id=2,
        )

        results = para_from_into(from_clause, into_clause, False, st, para_into_vars=True)
        assert len(results) >= 1

    def test_para_both_directions(self, st: SymbolTable):
        """Unoriented equality: paramodulate from both sides."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")

        # a=b (unoriented)
        from_clause = _clause(
            Literal(sign=True, atom=_eq_atom(st, a, b)),
            clause_id=1,
        )
        # p(a) | p(b) - both a and b appear
        into_clause = _clause(
            Literal(sign=True, atom=_make_term(st, "p", a)),
            Literal(sign=True, atom=_make_term(st, "p", b)),
            clause_id=2,
        )

        results = para_from_into(from_clause, into_clause, False, st)
        # From left: a->b into p(a) gives p(b), a->b into p(b) gives nothing (no match)
        # From right: b->a into p(a) gives nothing, b->a into p(b) gives p(a)
        assert len(results) >= 2

    def test_oriented_eq_only_from_left(self, st: SymbolTable):
        """Oriented equality: only paramodulate from the left side."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")

        eq_atom = _eq_atom(st, a, b)
        mark_oriented_eq(eq_atom)

        from_clause = _clause(
            Literal(sign=True, atom=eq_atom),
            clause_id=1,
        )

        # into_clause has both a and b
        into_clause = _clause(
            Literal(sign=True, atom=_make_term(st, "p", b)),
            clause_id=2,
        )

        results = para_from_into(from_clause, into_clause, False, st)
        # a is left side. b doesn't unify with a, so no results from left.
        # Right side is disabled (oriented), so no results from right either.
        assert len(results) == 0

        # Now try with p(a) - should work from left side
        into_clause2 = _clause(
            Literal(sign=True, atom=_make_term(st, "p", a)),
            clause_id=3,
        )
        results2 = para_from_into(from_clause, into_clause2, False, st)
        assert len(results2) >= 1


# ── Search integration tests ────────────────────────────────────────────────


class TestSearchIntegration:
    def test_paramodulation_option_exists(self):
        """SearchOptions should have paramodulation flag."""
        from pyladr.search.given_clause import SearchOptions

        opts = SearchOptions()
        assert hasattr(opts, "paramodulation")
        assert opts.paramodulation is False  # default off

    def test_search_with_paramodulation(self, st: SymbolTable):
        """Test that the search engine can use paramodulation to find a proof.

        Problem: a=b, p(a), -p(b) => contradiction
        """
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions, ExitCode

        a = _make_term(st, "a")
        b = _make_term(st, "b")

        # a = b
        eq_clause = _clause(Literal(sign=True, atom=_eq_atom(st, a, b)))
        # p(a)
        pa_clause = _clause(Literal(sign=True, atom=_make_term(st, "p", a)))
        # -p(b)
        npb_clause = _clause(Literal(sign=False, atom=_make_term(st, "p", b)))

        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            factoring=False,
            max_given=50,
        )

        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(
            usable=[eq_clause],
            sos=[pa_clause, npb_clause],
        )

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1

    def test_search_without_paramodulation_fails(self, st: SymbolTable):
        """Same problem without paramodulation should NOT find a proof easily.

        a=b, p(a), -p(b) => resolution alone can't solve this.
        """
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions, ExitCode

        a = _make_term(st, "a")
        b = _make_term(st, "b")

        eq_clause = _clause(Literal(sign=True, atom=_eq_atom(st, a, b)))
        pa_clause = _clause(Literal(sign=True, atom=_make_term(st, "p", a)))
        npb_clause = _clause(Literal(sign=False, atom=_make_term(st, "p", b)))

        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=False,
            factoring=False,
            max_given=20,
        )

        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(
            usable=[eq_clause],
            sos=[pa_clause, npb_clause],
        )

        # Without paramodulation, resolution alone can't handle equality
        assert result.exit_code != ExitCode.MAX_PROOFS_EXIT


# ── Position vector tests ────────────────────────────────────────────────────


class TestPositionVectors:
    def test_position_in_justification(self, st: SymbolTable):
        """Verify position vectors are correctly recorded."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")
        fa = _make_term(st, "f", a)

        from_clause = _clause(
            Literal(sign=True, atom=_eq_atom(st, a, b)),
            clause_id=1,
        )
        into_clause = _clause(
            Literal(sign=True, atom=_make_term(st, "p", fa)),
            clause_id=2,
        )

        results = para_from_into(from_clause, into_clause, False, st)
        assert len(results) >= 1

        # Find the result that paramodulates into the nested a
        for r in results:
            just = r.justification[0]
            assert just.just_type == JustType.PARA
            assert just.para is not None
            # from_pos should be (1, 1) - literal 1, left side
            assert just.para.from_pos[0] == 1  # first literal
            assert just.para.from_pos[1] == 1  # left side (arg 0 → C 1-indexed)
