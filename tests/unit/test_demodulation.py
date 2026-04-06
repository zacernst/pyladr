"""Comprehensive unit tests for the demodulation (term rewriting) system.

Supplements tests/test_demodulation.py with additional coverage for:
- LEX_DEP_RL classification and rewriting
- Multiple demodulators and chain rewrites
- Lex-dependent rewriting at application time
- Demod justification structure
- Back-demodulation with variable demodulators
- Renamable flip handling
- Edge cases: constants, deeply nested terms, step limit exhaustion
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


def _make(st: SymbolTable, name: str, *args: Term) -> Term:
    arity = len(args)
    sn = st.str_to_sn(name, arity)
    if arity == 0:
        return Term(private_symbol=-sn)
    return Term(private_symbol=-sn, arity=arity, args=tuple(args))


def _var(n: int) -> Term:
    return get_variable_term(n)


def _eq(st: SymbolTable, left: Term, right: Term) -> Term:
    return _make(st, "=", left, right)


def _unit_eq(st: SymbolTable, left: Term, right: Term, cid: int = 0) -> Clause:
    atom = _eq(st, left, right)
    return Clause(literals=(Literal(sign=True, atom=atom),), id=cid)


# ── Demodulator type: LEX_DEP_RL ────────────────────────────────────────────


class TestDemodulatorTypeLexDepRL:
    """Test LEX_DEP_RL classification — not covered in test_demodulation.py."""

    def test_lex_dep_rl_classification(self, st: SymbolTable):
        """a = f(x): alpha_vars={} ⊆ beta_vars={0}, beta not variable → RL."""
        a = _make(st, "a")
        x = _var(0)
        fx = _make(st, "f", x)
        c = _unit_eq(st, a, fx, cid=1)
        result = demodulator_type(c, st, lex_dep_demod_lim=10)
        assert result == DemodType.LEX_DEP_RL

    def test_rl_requires_not_renamable_flip(self, st: SymbolTable):
        """A renamable flip blocks RL; if LR still valid, result is LR."""
        x = _var(0)
        fx = _make(st, "f", x)
        gx = _make(st, "g", x)
        atom = _eq(st, fx, gx)
        mark_renamable_flip(atom)
        c = Clause(literals=(Literal(sign=True, atom=atom),), id=1)
        result = demodulator_type(c, st, lex_dep_demod_lim=10)
        # With renamable flip, RL blocked; LR still valid (same var sets, alpha not variable)
        assert result == DemodType.LEX_DEP_LR

    def test_lex_dep_limit_exceeded(self, st: SymbolTable):
        """Variable count exceeding limit disables lex-dep demodulation."""
        x, y, z = _var(0), _var(1), _var(2)
        hxy = _make(st, "h", x, y)
        fz = _make(st, "f", z)
        c = _unit_eq(st, hxy, fz, cid=1)
        # 3 variables total, limit is 2
        result = demodulator_type(c, st, lex_dep_demod_lim=2)
        assert result == DemodType.NOT_DEMODULATOR


# ── Multiple demodulators and chain rewrites ────────────────────────────────


class TestMultipleDemodulators:
    def test_two_demodulators_both_apply(self, st: SymbolTable):
        """Two oriented demodulators: a→b and c→d, rewrite h(a,c) → h(b,d)."""
        a, b = _make(st, "a"), _make(st, "b")
        c, d = _make(st, "c"), _make(st, "d")

        atom1 = _eq(st, a, b)
        mark_oriented_eq(atom1)
        d1 = Clause(literals=(Literal(sign=True, atom=atom1),), id=1)

        atom2 = _eq(st, c, d)
        mark_oriented_eq(atom2)
        d2 = Clause(literals=(Literal(sign=True, atom=atom2),), id=2)

        idx = DemodulatorIndex()
        idx.insert(d1, DemodType.ORIENTED)
        idx.insert(d2, DemodType.ORIENTED)

        t = _make(st, "h", a, c)
        result, steps = demodulate_term(t, idx, st)
        assert result.args[0].term_ident(b)
        assert result.args[1].term_ident(d)
        assert len(steps) >= 2

    def test_chain_rewrite(self, st: SymbolTable):
        """Chain: a→b, b→c. Rewriting p(a) should yield p(c)."""
        a, b, c = _make(st, "a"), _make(st, "b"), _make(st, "c")

        atom1 = _eq(st, a, b)
        mark_oriented_eq(atom1)
        d1 = Clause(literals=(Literal(sign=True, atom=atom1),), id=1)

        atom2 = _eq(st, b, c)
        mark_oriented_eq(atom2)
        d2 = Clause(literals=(Literal(sign=True, atom=atom2),), id=2)

        idx = DemodulatorIndex()
        idx.insert(d1, DemodType.ORIENTED)
        idx.insert(d2, DemodType.ORIENTED)

        pa = _make(st, "p", a)
        result, steps = demodulate_term(pa, idx, st)
        # a → b → c
        assert result.args[0].term_ident(c)
        assert len(steps) >= 2

    def test_no_infinite_loop_with_step_limit(self, st: SymbolTable):
        """Step limit prevents infinite chaining even with cyclic demods."""
        a, b = _make(st, "a"), _make(st, "b")

        atom1 = _eq(st, a, b)
        mark_oriented_eq(atom1)
        d1 = Clause(literals=(Literal(sign=True, atom=atom1),), id=1)

        atom2 = _eq(st, b, a)
        mark_oriented_eq(atom2)
        d2 = Clause(literals=(Literal(sign=True, atom=atom2),), id=2)

        idx = DemodulatorIndex()
        idx.insert(d1, DemodType.ORIENTED)
        idx.insert(d2, DemodType.ORIENTED)

        pa = _make(st, "p", a)
        result, steps = demodulate_term(pa, idx, st, step_limit=5)
        # Step limit prevents unbounded rewriting; exact count depends on
        # recursive tracking but must terminate (not hang)
        assert len(steps) < 100  # bounded, not infinite


# ── Demod justification structure ───────────────────────────────────────────


class TestDemodJustification:
    def test_justification_records_demod_id(self, st: SymbolTable):
        """Demodulation steps record the demodulator clause ID."""
        a, b = _make(st, "a"), _make(st, "b")
        atom = _eq(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=42)

        idx = DemodulatorIndex()
        idx.insert(demod, DemodType.ORIENTED)

        pa = _make(st, "p", a)
        _, steps = demodulate_term(pa, idx, st)
        assert len(steps) >= 1
        assert steps[0][0] == 42  # demod clause ID

    def test_clause_demod_justification_type(self, st: SymbolTable):
        """Demodulated clause gets DEMOD justification."""
        a, b = _make(st, "a"), _make(st, "b")
        atom = _eq(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        idx = DemodulatorIndex()
        idx.insert(demod, DemodType.ORIENTED)

        clause = Clause(
            literals=(Literal(sign=True, atom=_make(st, "p", a)),),
            id=5,
        )
        result, steps = demodulate_clause(clause, idx, st)
        assert len(result.justification) >= 1
        demod_just = result.justification[-1]
        assert demod_just.just_type == JustType.DEMOD
        assert demod_just.clause_id == 5

    def test_demod_steps_in_justification(self, st: SymbolTable):
        """Justification contains demod_steps with (id, position, direction)."""
        a, b = _make(st, "a"), _make(st, "b")
        atom = _eq(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=7)

        idx = DemodulatorIndex()
        idx.insert(demod, DemodType.ORIENTED)

        clause = Clause(
            literals=(Literal(sign=True, atom=_make(st, "p", a)),),
            id=10,
        )
        result, _ = demodulate_clause(clause, idx, st)
        just = result.justification[-1]
        assert len(just.demod_steps) >= 1
        step = just.demod_steps[0]
        assert step[0] == 7  # demod ID
        assert step[2] == 1  # direction: left-to-right


# ── Back-demodulation with variable demodulators ────────────────────────────


class TestBackDemodVariable:
    def test_variable_demod_finds_instances(self, st: SymbolTable):
        """f(x)=g(x) oriented: back-demod finds clause with f(a)."""
        x = _var(0)
        fx, gx = _make(st, "f", x), _make(st, "g", x)
        atom = _eq(st, fx, gx)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        a = _make(st, "a")
        fa = _make(st, "f", a)
        c1 = Clause(
            literals=(Literal(sign=True, atom=_make(st, "p", fa)),),
            id=2,
        )
        # Clause without f: should not match
        c2 = Clause(
            literals=(Literal(sign=True, atom=_make(st, "p", a)),),
            id=3,
        )

        result = back_demodulatable(demod, DemodType.ORIENTED, [c1, c2], st)
        assert c1 in result
        assert c2 not in result

    def test_back_demod_deeply_nested(self, st: SymbolTable):
        """Demodulator matches in deeply nested subterm."""
        a, b = _make(st, "a"), _make(st, "b")
        atom = _eq(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        # h(f(g(a)), c) — a is 3 levels deep
        c_const = _make(st, "c")
        deep = _make(st, "h", _make(st, "f", _make(st, "g", a)), c_const)
        c1 = Clause(literals=(Literal(sign=True, atom=deep),), id=2)

        result = back_demodulatable(demod, DemodType.ORIENTED, [c1], st)
        assert c1 in result

    def test_back_demod_multiple_literals(self, st: SymbolTable):
        """Back-demod checks all literals in a clause."""
        a, b = _make(st, "a"), _make(st, "b")
        atom = _eq(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        c_const = _make(st, "c")
        # a appears only in second literal
        clause = Clause(
            literals=(
                Literal(sign=True, atom=_make(st, "p", c_const)),
                Literal(sign=False, atom=_make(st, "q", c_const, a)),
            ),
            id=2,
        )

        result = back_demodulatable(demod, DemodType.ORIENTED, [clause], st)
        assert clause in result


# ── DemodulatorIndex additional tests ───────────────────────────────────────


class TestDemodulatorIndexExtended:
    def test_iteration(self, st: SymbolTable):
        """Index iterates over all stored demodulators."""
        a, b = _make(st, "a"), _make(st, "b")
        c, d = _make(st, "c"), _make(st, "d")

        d1 = _unit_eq(st, a, b, cid=1)
        d2 = _unit_eq(st, c, d, cid=2)

        idx = DemodulatorIndex()
        idx.insert(d1, DemodType.ORIENTED)
        idx.insert(d2, DemodType.LEX_DEP_LR)

        items = list(idx)
        assert len(items) == 2
        assert items[0] == (d1, DemodType.ORIENTED)
        assert items[1] == (d2, DemodType.LEX_DEP_LR)

    def test_remove_nonexistent_is_safe(self, st: SymbolTable):
        """Removing a clause not in the index doesn't error."""
        a, b = _make(st, "a"), _make(st, "b")
        d1 = _unit_eq(st, a, b, cid=1)
        d2 = _unit_eq(st, a, b, cid=2)

        idx = DemodulatorIndex()
        idx.insert(d1, DemodType.ORIENTED)
        idx.remove(d2)  # d2 was never inserted
        assert len(idx) == 1

    def test_remove_preserves_others(self, st: SymbolTable):
        """Removing one demodulator leaves others intact."""
        a, b = _make(st, "a"), _make(st, "b")
        c, d = _make(st, "c"), _make(st, "d")

        d1 = _unit_eq(st, a, b, cid=1)
        d2 = _unit_eq(st, c, d, cid=2)

        idx = DemodulatorIndex()
        idx.insert(d1, DemodType.ORIENTED)
        idx.insert(d2, DemodType.ORIENTED)
        idx.remove(d1)

        assert len(idx) == 1
        items = list(idx)
        assert items[0][0] is d2


# ── Demodulation of constants and variables ─────────────────────────────────


class TestDemodEdgeCases:
    def test_demod_constant_term(self, st: SymbolTable):
        """Demodulating a bare constant with matching demod."""
        a, b = _make(st, "a"), _make(st, "b")
        atom = _eq(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        idx = DemodulatorIndex()
        idx.insert(demod, DemodType.ORIENTED)

        result, steps = demodulate_term(a, idx, st)
        assert result.term_ident(b)
        assert len(steps) >= 1

    def test_demod_variable_unchanged(self, st: SymbolTable):
        """Variables are never rewritten by demodulation."""
        a, b = _make(st, "a"), _make(st, "b")
        atom = _eq(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        idx = DemodulatorIndex()
        idx.insert(demod, DemodType.ORIENTED)

        x = _var(0)
        result, steps = demodulate_term(x, idx, st)
        assert result is x
        assert len(steps) == 0

    def test_demod_clause_unchanged_returns_same_object(self, st: SymbolTable):
        """When no rewriting occurs, demodulate_clause returns the same clause."""
        a, b = _make(st, "a"), _make(st, "b")
        c_const = _make(st, "c")
        atom = _eq(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        idx = DemodulatorIndex()
        idx.insert(demod, DemodType.ORIENTED)

        # Clause with c, no a's to rewrite
        clause = Clause(
            literals=(Literal(sign=True, atom=_make(st, "p", c_const)),),
            id=5,
        )
        result, steps = demodulate_clause(clause, idx, st)
        assert result is clause
        assert len(steps) == 0

    def test_demod_multiple_occurrences(self, st: SymbolTable):
        """All occurrences of matching subterm are rewritten."""
        a, b = _make(st, "a"), _make(st, "b")
        atom = _eq(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        idx = DemodulatorIndex()
        idx.insert(demod, DemodType.ORIENTED)

        # h(a, a) → h(b, b)
        t = _make(st, "h", a, a)
        result, steps = demodulate_term(t, idx, st)
        assert result.args[0].term_ident(b)
        assert result.args[1].term_ident(b)

    def test_demod_preserves_non_matching_structure(self, st: SymbolTable):
        """Non-matching subterms are preserved exactly."""
        a, b = _make(st, "a"), _make(st, "b")
        c_const = _make(st, "c")
        atom = _eq(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        idx = DemodulatorIndex()
        idx.insert(demod, DemodType.ORIENTED)

        # h(a, c) → h(b, c): c preserved
        t = _make(st, "h", a, c_const)
        result, steps = demodulate_term(t, idx, st)
        assert result.args[0].term_ident(b)
        assert result.args[1].term_ident(c_const)


# ── Demodulator type: oriented detection ────────────────────────────────────


class TestDemodTypeOrientedEdgeCases:
    def test_oriented_ignores_lex_dep_lim(self, st: SymbolTable):
        """Oriented demodulators don't need lex_dep_demod_lim."""
        a, b = _make(st, "a"), _make(st, "b")
        atom = _eq(st, a, b)
        mark_oriented_eq(atom)
        c = Clause(literals=(Literal(sign=True, atom=atom),), id=1)
        # Even with limit=0, oriented still detected
        assert demodulator_type(c, st, lex_dep_demod_lim=0) == DemodType.ORIENTED

    def test_empty_clause_not_demod(self, st: SymbolTable):
        """Empty clause (no literals) is not a demodulator."""
        c = Clause(literals=(), id=1)
        assert demodulator_type(c, st) == DemodType.NOT_DEMODULATOR

    def test_is_eq_atom_required(self, st: SymbolTable):
        """Only equality atoms qualify as demodulators."""
        a = _make(st, "a")
        # p(a) is not an equality
        c = Clause(literals=(Literal(sign=True, atom=_make(st, "p", a)),), id=1)
        assert demodulator_type(c, st) == DemodType.NOT_DEMODULATOR


# ── Back-demodulation: variable term skipping ───────────────────────────────


class TestBackDemodEdgeCases:
    def test_variable_subterm_not_matched(self, st: SymbolTable):
        """Back-demod does not match inside variable subterms (variables are leaves)."""
        x = _var(0)
        a, b = _make(st, "a"), _make(st, "b")
        atom = _eq(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        # p(x) — no rigid subterms to match
        clause = Clause(
            literals=(Literal(sign=True, atom=_make(st, "p", x)),),
            id=2,
        )
        result = back_demodulatable(demod, DemodType.ORIENTED, [clause], st)
        assert len(result) == 0

    def test_empty_clause_list(self, st: SymbolTable):
        """Back-demod with empty clause list returns empty."""
        a, b = _make(st, "a"), _make(st, "b")
        atom = _eq(st, a, b)
        mark_oriented_eq(atom)
        demod = Clause(literals=(Literal(sign=True, atom=atom),), id=1)

        result = back_demodulatable(demod, DemodType.ORIENTED, [], st)
        assert len(result) == 0
