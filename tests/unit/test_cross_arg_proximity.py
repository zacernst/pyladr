"""Unit tests for cross-argument distance scoring in Tree2Vec goal selection.

Tests the _get_antecedent_term helper, _t2v_cosine similarity, cross-arg
distance computation, SearchOptions defaults, and CLI flag behavior.
"""

from __future__ import annotations

import math

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.search.given_clause import (
    SearchOptions,
    _get_antecedent_term,
    _t2v_cosine,
)


# ── Helpers: vampire.in domain terms ──────────────────────────────────────

SYM_P = 1  # P: unary predicate
SYM_I = 2  # i: binary function
SYM_N = 3  # n: unary function


def var(n: int) -> Term:
    return get_variable_term(n)


def n(arg: Term) -> Term:
    return get_rigid_term(SYM_N, 1, (arg,))


def i(left: Term, right: Term) -> Term:
    return get_rigid_term(SYM_I, 2, (left, right))


def P(arg: Term) -> Term:
    return get_rigid_term(SYM_P, 1, (arg,))


def make_clause(*lits: Literal) -> Clause:
    return Clause(literals=tuple(lits))


def pos_lit(atom: Term) -> Literal:
    return Literal(sign=True, atom=atom)


# ── _get_antecedent_term tests ────────────────────────────────────────────


class TestGetAntecedentTerm:
    def test_standard_structure_returns_first_arg(self):
        """P(i(x, y)) → returns x."""
        x, y = var(0), var(1)
        clause = make_clause(pos_lit(P(i(x, y))))
        ant = _get_antecedent_term(clause)
        assert ant is x

    def test_nested_antecedent(self):
        """P(i(n(x), y)) → returns n(x)."""
        x, y = var(0), var(1)
        nx = n(x)
        clause = make_clause(pos_lit(P(i(nx, y))))
        ant = _get_antecedent_term(clause)
        assert ant is nx

    def test_unary_inner_returns_none(self):
        """P(n(x)) → inner has arity 1 but n has only 1 arg; returns first arg."""
        x = var(0)
        clause = make_clause(pos_lit(P(n(x))))
        ant = _get_antecedent_term(clause)
        # n(x) has arity 1, args[0] = x, so antecedent is x
        assert ant is x

    def test_propositional_atom_returns_none(self):
        """P() (arity 0) → returns None."""
        P0 = get_rigid_term(SYM_P, 0, ())
        clause = make_clause(pos_lit(P0))
        ant = _get_antecedent_term(clause)
        assert ant is None

    def test_variable_arg_returns_none(self):
        """P(x) where x is a variable (arity 0) → inner.arity < 1, returns None."""
        x = var(0)
        # P(x): atom.args[0] = x, x.arity = 0
        clause = make_clause(pos_lit(P(x)))
        ant = _get_antecedent_term(clause)
        assert ant is None

    def test_empty_clause_returns_none(self):
        clause = Clause(literals=())
        assert _get_antecedent_term(clause) is None


# ── _t2v_cosine tests ────────────────────────────────────────────────────


class TestCosine:
    def test_identical_vectors(self):
        assert _t2v_cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _t2v_cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _t2v_cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert _t2v_cosine([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_arbitrary_vectors(self):
        a = [3.0, 4.0]
        b = [4.0, 3.0]
        expected = 24.0 / 25.0  # dot=24, |a|=5, |b|=5
        assert _t2v_cosine(a, b) == pytest.approx(expected)


# ── SearchOptions defaults ────────────────────────────────────────────────


class TestSearchOptionsDefaults:
    def test_cross_arg_proximity_default_true(self):
        opts = SearchOptions()
        assert opts.tree2vec_cross_arg_proximity is True

    def test_cross_arg_can_be_disabled(self):
        opts = SearchOptions(tree2vec_cross_arg_proximity=False)
        assert opts.tree2vec_cross_arg_proximity is False


# ── CLI flag tests ────────────────────────────────────────────────────────


class TestCLIFlag:
    def test_default_is_true(self):
        from pyladr.apps.prover9 import _build_arg_parser
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.tree2vec_cross_arg_proximity is True

    def test_no_flag_disables(self):
        from pyladr.apps.prover9 import _build_arg_parser
        parser = _build_arg_parser()
        args = parser.parse_args(["--no-tree2vec-cross-arg"])
        assert args.tree2vec_cross_arg_proximity is False


# ── Cross-arg score computation tests ─────────────────────────────────────


class TestCrossArgScore:
    """Tests for the cross-arg distance formula: (1 - max_sim) / 2."""

    def test_identical_embeddings_give_zero_distance(self):
        """When antecedent matches goal arg exactly, distance should be 0.0."""
        # dist = (1 - cosine(identical)) / 2 = (1 - 1.0) / 2 = 0.0
        sim = _t2v_cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        dist = (1.0 - sim) / 2.0
        assert dist == pytest.approx(0.0)

    def test_orthogonal_embeddings_give_half_distance(self):
        """Orthogonal vectors → cosine=0 → distance=0.5."""
        sim = _t2v_cosine([1.0, 0.0], [0.0, 1.0])
        dist = (1.0 - sim) / 2.0
        assert dist == pytest.approx(0.5)

    def test_opposite_embeddings_give_max_distance(self):
        """Opposite vectors → cosine=-1 → distance=1.0."""
        sim = _t2v_cosine([1.0, 0.0], [-1.0, 0.0])
        dist = (1.0 - sim) / 2.0
        assert dist == pytest.approx(1.0)

    def test_cross_arg_takes_max_sim_of_both_directions(self):
        """The cross-arg formula uses max similarity then converts to distance."""
        # Simulate: ant1 vs goal_arg, and goal_ant vs full1
        ant1 = [1.0, 0.0]
        goal_arg = [0.8, 0.6]  # somewhat similar to ant1
        goal_ant = [0.0, 1.0]
        full1 = [0.6, 0.8]    # somewhat similar to goal_ant

        sim1 = _t2v_cosine(ant1, goal_arg)  # ant vs goal_arg
        sim2 = _t2v_cosine(goal_ant, full1)  # goal_ant vs full
        best = max(sim1, sim2)
        dist = (1.0 - best) / 2.0

        # sim1 = 0.8/1.0 = 0.8, sim2 = 0.8/1.0 = 0.8 → dist = 0.1
        assert dist == pytest.approx(0.1)


# ── Fallback behavior tests ──────────────────────────────────────────────


class TestFallbackBehavior:
    """Tests that non-standard clause structures fall back gracefully."""

    def test_propositional_clause_has_no_antecedent(self):
        """Propositional atoms cannot have antecedent terms."""
        P0 = get_rigid_term(SYM_P, 0, ())
        clause = make_clause(pos_lit(P0))
        assert _get_antecedent_term(clause) is None

    def test_variable_arg_clause_has_no_antecedent(self):
        """P(x) where x is a variable cannot have antecedent."""
        clause = make_clause(pos_lit(P(var(0))))
        assert _get_antecedent_term(clause) is None
