"""Tests for the demodulation (term rewriting) system.

Tests cover:
- Demodulator type classification
- Forward demodulation (term and clause rewriting)
- DemodulatorIndex management
- Back-demodulation (finding rewritable clauses)
- Integration with the search loop
- Edge cases: step limits, lex-dependent demodulators
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, get_variable_term
from pyladr.inference.demodulation import (
    DemodType,
    DemodulatorIndex,
    back_demodulatable,
    demodulate_clause,
    demodulate_term,
    demodulator_type,
)
from pyladr.inference.paramodulation import (
    _oriented_eqs,
    _renamable_flips,
    is_eq_atom,
    mark_oriented_eq,
    mark_renamable_flip,
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
    s = SymbolTable()
    s.str_to_sn("=", 2)
    s.str_to_sn("f", 1)
    s.str_to_sn("g", 1)
    s.str_to_sn("h", 2)
    s.str_to_sn("a", 0)
    s.str_to_sn("b", 0)
    s.str_to_sn("c", 0)
    s.str_to_sn("d", 0)
    s.str_to_sn("p", 1)
    s.str_to_sn("q", 2)
    return s


def _make_term(st: SymbolTable, name: str, *args: Term) -> Term:
    arity = len(args)
    sn = st.str_to_sn(name, arity)
    if arity == 0:
        return Term(private_symbol=-sn)
    return Term(private_symbol=-sn, arity=arity, args=tuple(args))


def _var(n: int) -> Term:
    return get_variable_term(n)


def _eq_atom(st: SymbolTable, left: Term, right: Term) -> Term:
    return _make_term(st, "=", left, right)


def _unit_eq_clause(st: SymbolTable, left: Term, right: Term, clause_id: int = 0) -> Clause:
    atom = _eq_atom(st, left, right)
    return Clause(
        literals=(Literal(sign=True, atom=atom),),
        id=clause_id,
    )


# ── Demodulator type classification ─────────────────────────────────────────


class TestDemodulatorType:
    def test_non_unit_clause(self, st: SymbolTable):
        """Multi-literal clause is not a demodulator."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")
        c = Clause(literals=(
            Literal(sign=True, atom=_eq_atom(st, a, b)),
            Literal(sign=True, atom=_make_term(st, "p", a)),
        ))
        assert demodulator_type(c, st) == DemodType.NOT_DEMODULATOR

    def test_negative_eq(self, st: SymbolTable):
        """Negative equality is not a demodulator."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")
        c = Clause(literals=(Literal(sign=False, atom=_eq_atom(st, a, b)),))
        assert demodulator_type(c, st) == DemodType.NOT_DEMODULATOR

    def test_non_equality(self, st: SymbolTable):
        """Non-equality predicate is not a demodulator."""
        a = _make_term(st, "a")
        c = Clause(literals=(Literal(sign=True, atom=_make_term(st, "p", a)),))
        assert demodulator_type(c, st) == DemodType.NOT_DEMODULATOR

    def test_oriented_demodulator(self, st: SymbolTable):
        """Oriented equality (left > right) becomes ORIENTED demodulator."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")
        atom = _eq_atom(st, a, b)
        mark_oriented_eq(atom)
        c = Clause(literals=(Literal(sign=True, atom=atom),), id=1)
        assert demodulator_type(c, st) == DemodType.ORIENTED

    def test_unoriented_no_lex(self, st: SymbolTable):
        """Unoriented equality with lex_dep_demod_lim=0 is not a demodulator."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")
        c = _unit_eq_clause(st, a, b)
        assert demodulator_type(c, st, lex_dep_demod_lim=0) == DemodType.NOT_DEMODULATOR

    def test_lex_dep_lr(self, st: SymbolTable):
        """f(x) = a with lex dep: LR (a's vars ⊆ f(x)'s vars)."""
        x = _var(0)
        fx = _make_term(st, "f", x)
        a = _make_term(st, "a")
        c = _unit_eq_clause(st, fx, a, clause_id=1)
        # beta=a has no vars, alpha=f(x) has {0}. beta_vars ⊆ alpha_vars: True
        # alpha_vars ⊆ beta_vars: False (x not in {})
        result = demodulator_type(c, st, lex_dep_demod_lim=10)
        assert result == DemodType.LEX_DEP_LR

    def test_lex_dep_both(self, st: SymbolTable):
        """f(x) = g(x) with lex dep: BOTH (same variable set)."""
        x = _var(0)
        fx = _make_term(st, "f", x)
        gx = _make_term(st, "g", x)
        c = _unit_eq_clause(st, fx, gx, clause_id=1)
        result = demodulator_type(c, st, lex_dep_demod_lim=10)
        assert result == DemodType.LEX_DEP_BOTH

    def test_variable_lhs_not_demodulator(self, st: SymbolTable):
        """x = a: variable LHS can't be a demodulator (would match everything)."""
        x = _var(0)
        a = _make_term(st, "a")
        c = _unit_eq_clause(st, x, a, clause_id=1)
        result = demodulator_type(c, st, lex_dep_demod_lim=10)
        # LR requires !VARIABLE(alpha): x is variable → LR=False
        # RL: alpha_vars={0} ⊆ beta_vars={} → False
        assert result == DemodType.NOT_DEMODULATOR


# ── Forward demodulation ─────────────────────────────────────────────────────


class TestDemodulation:
    def test_simple_constant_rewrite(self, st: SymbolTable):
        """a = b (oriented): rewrite p(a) → p(b)."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")

        atom = _eq_atom(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        idx = DemodulatorIndex()
        idx.insert(demod, DemodType.ORIENTED)

        # Term to demodulate: p(a)
        pa = _make_term(st, "p", a)
        result, steps = demodulate_term(pa, idx, st)

        # Should rewrite to p(b)
        assert result.args[0].term_ident(b)
        assert len(steps) >= 1

    def test_nested_rewrite(self, st: SymbolTable):
        """a = b (oriented): rewrite f(a) → f(b) inside h(f(a), c)."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")
        c = _make_term(st, "c")

        atom = _eq_atom(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        idx = DemodulatorIndex()
        idx.insert(demod, DemodType.ORIENTED)

        fa = _make_term(st, "f", a)
        t = _make_term(st, "h", fa, c)
        result, steps = demodulate_term(t, idx, st)

        # h(f(b), c)
        assert result.args[0].args[0].term_ident(b)
        assert result.args[1].term_ident(c)

    def test_variable_rewrite(self, st: SymbolTable):
        """f(x) = g(x) (oriented): rewrite p(f(a)) → p(g(a))."""
        x = _var(0)
        fx = _make_term(st, "f", x)
        gx = _make_term(st, "g", x)

        atom = _eq_atom(st, fx, gx)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        idx = DemodulatorIndex()
        idx.insert(demod, DemodType.ORIENTED)

        a = _make_term(st, "a")
        fa = _make_term(st, "f", a)
        pa = _make_term(st, "p", fa)
        result, steps = demodulate_term(pa, idx, st)

        # p(g(a))
        ga = _make_term(st, "g", a)
        assert result.args[0].term_ident(ga)

    def test_no_rewrite_when_no_match(self, st: SymbolTable):
        """a = b (oriented): p(c) should not be rewritten."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")
        c = _make_term(st, "c")

        atom = _eq_atom(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        idx = DemodulatorIndex()
        idx.insert(demod, DemodType.ORIENTED)

        pc = _make_term(st, "p", c)
        result, steps = demodulate_term(pc, idx, st)
        assert result is pc  # unchanged
        assert len(steps) == 0

    def test_demodulate_clause(self, st: SymbolTable):
        """Demodulate all literals in a clause."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")

        atom = _eq_atom(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        idx = DemodulatorIndex()
        idx.insert(demod, DemodType.ORIENTED)

        # Clause: p(a) | -q(a, a)
        clause = Clause(
            literals=(
                Literal(sign=True, atom=_make_term(st, "p", a)),
                Literal(sign=False, atom=_make_term(st, "q", a, a)),
            ),
            id=5,
        )

        result, steps = demodulate_clause(clause, idx, st)
        assert len(steps) >= 1

        # Both occurrences of a should be rewritten to b
        for lit in result.literals:
            for arg in lit.atom.args:
                assert arg.term_ident(b), f"Expected b, got {arg}"

    def test_step_limit(self, st: SymbolTable):
        """Step limit prevents infinite loops."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")

        atom = _eq_atom(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        idx = DemodulatorIndex()
        idx.insert(demod, DemodType.ORIENTED)

        pa = _make_term(st, "p", a)
        # Even with step_limit=1, should still rewrite at least once
        result, steps = demodulate_term(pa, idx, st, step_limit=1)
        assert len(steps) <= 1

    def test_empty_demod_index(self, st: SymbolTable):
        """Empty demodulator index returns clause unchanged."""
        a = _make_term(st, "a")
        clause = Clause(
            literals=(Literal(sign=True, atom=_make_term(st, "p", a)),),
            id=5,
        )
        idx = DemodulatorIndex()
        result, steps = demodulate_clause(clause, idx, st)
        assert result is clause
        assert len(steps) == 0


# ── DemodulatorIndex ─────────────────────────────────────────────────────────


class TestDemodulatorIndex:
    def test_insert_and_len(self, st: SymbolTable):
        idx = DemodulatorIndex()
        assert len(idx) == 0
        assert idx.is_empty

        a = _make_term(st, "a")
        b = _make_term(st, "b")
        demod = _unit_eq_clause(st, a, b, clause_id=1)
        idx.insert(demod, DemodType.ORIENTED)

        assert len(idx) == 1
        assert not idx.is_empty

    def test_remove(self, st: SymbolTable):
        idx = DemodulatorIndex()
        a = _make_term(st, "a")
        b = _make_term(st, "b")
        demod = _unit_eq_clause(st, a, b, clause_id=1)
        idx.insert(demod, DemodType.ORIENTED)
        idx.remove(demod)
        assert idx.is_empty


# ── Back-demodulation ────────────────────────────────────────────────────────


class TestBackDemodulation:
    def test_find_rewritable(self, st: SymbolTable):
        """back_demodulatable finds clauses containing matching subterms."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")
        c = _make_term(st, "c")

        atom = _eq_atom(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        # Clause with a in it
        c1 = Clause(
            literals=(Literal(sign=True, atom=_make_term(st, "p", a)),),
            id=2,
        )
        # Clause without a
        c2 = Clause(
            literals=(Literal(sign=True, atom=_make_term(st, "p", c)),),
            id=3,
        )

        result = back_demodulatable(demod, DemodType.ORIENTED, [c1, c2], st)
        assert c1 in result
        assert c2 not in result

    def test_skip_self(self, st: SymbolTable):
        """back_demodulatable skips the demodulator itself."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")

        atom = _eq_atom(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        result = back_demodulatable(demod, DemodType.ORIENTED, [demod], st)
        assert len(result) == 0

    def test_nested_match(self, st: SymbolTable):
        """back_demodulatable finds nested subterms."""
        a = _make_term(st, "a")
        b = _make_term(st, "b")

        atom = _eq_atom(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        # f(a) contains a as a nested subterm
        fa = _make_term(st, "f", a)
        c1 = Clause(
            literals=(Literal(sign=True, atom=_make_term(st, "p", fa)),),
            id=2,
        )

        result = back_demodulatable(demod, DemodType.ORIENTED, [c1], st)
        assert c1 in result


# ── Search integration ───────────────────────────────────────────────────────


class TestSearchIntegration:
    def test_demod_options_exist(self):
        """SearchOptions should have demodulation flags."""
        from pyladr.search.given_clause import SearchOptions

        opts = SearchOptions()
        assert hasattr(opts, "demodulation")
        assert opts.demodulation is False

    def test_search_with_demodulation(self, st: SymbolTable):
        """Test search using demodulation to simplify clauses.

        Problem: f(a)=b (oriented demod), p(f(a)), -p(b) => contradiction
        The demodulator f(a)=b should rewrite p(f(a)) to p(b),
        which then conflicts with -p(b).
        """
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions, ExitCode

        a = _make_term(st, "a")
        b = _make_term(st, "b")
        fa = _make_term(st, "f", a)

        # f(a) = b (oriented demodulator)
        eq_atom = _eq_atom(st, fa, b)
        mark_oriented_eq(eq_atom)
        demod_clause = Clause(literals=(Literal(sign=True, atom=eq_atom),))

        # p(f(a))
        pfa = Clause(literals=(Literal(sign=True, atom=_make_term(st, "p", fa)),))

        # -p(b)
        npb = Clause(literals=(Literal(sign=False, atom=_make_term(st, "p", b)),))

        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            demodulation=True,
            factoring=False,
            max_given=50,
        )

        search = GivenClauseSearch(options=opts, symbol_table=st)
        result = search.run(
            usable=[demod_clause],
            sos=[pfa, npb],
        )

        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1
