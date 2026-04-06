"""Unit tests for proof justification tracking.

Tests behavioral equivalence with C just.h/just.c:
- Justification construction for each inference type
- Parent extraction from justification chains
- Proof structure validation
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal, ParaJust
from pyladr.core.term import get_rigid_term, get_variable_term


# ── Helpers ──────────────────────────────────────────────────────────────────


P_SN, Q_SN = 1, 2
A_SN, B_SN = 3, 4

a = get_rigid_term(A_SN, 0)
b = get_rigid_term(B_SN, 0)
x = get_variable_term(0)


def _atom(sn: int, *args):
    return get_rigid_term(sn, len(args), tuple(args))


def _clause(*lits, clause_id=0, justification=()):
    return Clause(
        literals=tuple(Literal(sign=s, atom=at) for s, at in lits),
        id=clause_id,
        justification=justification,
    )


# ── Justification types ─────────────────────────────────────────────────────


class TestJustificationTypes:
    """Test creating justifications for each inference rule."""

    def test_input_justification(self) -> None:
        """Input clause has INPUT type."""
        just = Justification(just_type=JustType.INPUT)
        assert just.just_type == JustType.INPUT
        assert just.clause_ids == ()

    def test_goal_justification(self) -> None:
        """Goal clause has GOAL type."""
        just = Justification(just_type=JustType.GOAL)
        assert just.just_type == JustType.GOAL

    def test_resolution_justification(self) -> None:
        """Binary resolution records parent clause IDs."""
        just = Justification(
            just_type=JustType.BINARY_RES,
            clause_ids=(5, 8),
        )
        assert just.just_type == JustType.BINARY_RES
        assert just.clause_ids == (5, 8)
        assert 5 in just.clause_ids
        assert 8 in just.clause_ids

    def test_paramodulation_justification(self) -> None:
        """Paramodulation records from/into clause IDs and positions."""
        pj = ParaJust(from_id=1, into_id=2, from_pos=(1, 1), into_pos=(1, 1))
        just = Justification(just_type=JustType.PARA, para=pj)
        assert just.just_type == JustType.PARA
        assert just.para is not None
        assert just.para.from_id == 1
        assert just.para.into_id == 2
        assert just.para.from_pos == (1, 1)
        assert just.para.into_pos == (1, 1)

    def test_demodulation_justification(self) -> None:
        """Demodulation records the demodulator IDs used."""
        just = Justification(
            just_type=JustType.DEMOD,
            demod_steps=((10, 1, 1),),
        )
        assert just.just_type == JustType.DEMOD
        assert just.demod_steps == ((10, 1, 1),)

    def test_factor_justification(self) -> None:
        """Factoring records the parent clause."""
        just = Justification(just_type=JustType.FACTOR, clause_ids=(3,))
        assert just.just_type == JustType.FACTOR
        assert just.clause_ids == (3,)

    def test_copy_justification(self) -> None:
        """Copy records original clause ID."""
        just = Justification(just_type=JustType.COPY, clause_id=7)
        assert just.just_type == JustType.COPY
        assert just.clause_id == 7

    def test_deny_justification(self) -> None:
        """Deny (goal denial) records the original goal."""
        just = Justification(just_type=JustType.DENY, clause_id=1)
        assert just.just_type == JustType.DENY

    def test_clausify_justification(self) -> None:
        """Clausification from formula."""
        just = Justification(just_type=JustType.CLAUSIFY, clause_id=2)
        assert just.just_type == JustType.CLAUSIFY

    def test_flip_justification(self) -> None:
        """Flip records reversing equality orientation."""
        just = Justification(just_type=JustType.FLIP, clause_id=5)
        assert just.just_type == JustType.FLIP


# ── Justification on clauses ────────────────────────────────────────────────


class TestJustificationOnClauses:
    """Test justification attached to actual clauses."""

    def test_clause_has_empty_justification_by_default(self) -> None:
        """New clause has no justification steps."""
        c = _clause((True, _atom(P_SN, a)))
        assert c.justification == ()

    def test_clause_with_input_justification(self) -> None:
        """Input clause carries input justification."""
        just = Justification(just_type=JustType.INPUT)
        c = _clause(
            (True, _atom(P_SN, a)),
            clause_id=1,
            justification=(just,),
        )
        assert len(c.justification) == 1
        assert c.justification[0].just_type == JustType.INPUT

    def test_clause_with_multi_step_justification(self) -> None:
        """Clause can have primary + secondary justifications.

        E.g., a clause derived by paramodulation then demodulated:
        [para(1,2), demod(3)]
        """
        j1 = Justification(
            just_type=JustType.PARA,
            para=ParaJust(from_id=1, into_id=2, from_pos=(1, 1), into_pos=(1, 1)),
        )
        j2 = Justification(
            just_type=JustType.DEMOD,
            demod_steps=((3, 1, 1),),
        )
        c = _clause(
            (True, _atom(P_SN, a)),
            clause_id=10,
            justification=(j1, j2),
        )
        assert len(c.justification) == 2
        assert c.justification[0].just_type == JustType.PARA
        assert c.justification[1].just_type == JustType.DEMOD


# ── Proof extraction ────────────────────────────────────────────────────────


class TestProofExtraction:
    """Test extracting proof information from justifications."""

    def test_get_parent_ids_from_resolution(self) -> None:
        """Extract parent clause IDs from resolution justification."""
        just = Justification(just_type=JustType.BINARY_RES, clause_ids=(5, 8))
        parents = just.clause_ids
        assert 5 in parents
        assert 8 in parents

    def test_get_parent_ids_from_para(self) -> None:
        """Extract parent IDs from paramodulation justification."""
        pj = ParaJust(from_id=10, into_id=20, from_pos=(1, 1), into_pos=(1, 1))
        just = Justification(just_type=JustType.PARA, para=pj)
        assert just.para.from_id == 10
        assert just.para.into_id == 20

    def test_paramodulation_position_vectors(self) -> None:
        """ParaJust stores from_pos and into_pos as tuples."""
        pj = ParaJust(
            from_id=1,
            into_id=2,
            from_pos=(1, 2),
            into_pos=(1, 1, 1),
        )
        assert pj.from_pos == (1, 2)
        assert pj.into_pos == (1, 1, 1)
        # All positions are positive (1-indexed, matching C convention)
        assert all(p > 0 for p in pj.from_pos)
        assert all(p > 0 for p in pj.into_pos)

    def test_justification_immutable(self) -> None:
        """Justification is frozen/immutable."""
        just = Justification(just_type=JustType.INPUT)
        with pytest.raises(AttributeError):
            just.just_type = JustType.GOAL  # type: ignore

    def test_parajust_immutable(self) -> None:
        """ParaJust is frozen/immutable."""
        pj = ParaJust(from_id=1, into_id=2, from_pos=(1,), into_pos=(1,))
        with pytest.raises(AttributeError):
            pj.from_id = 99  # type: ignore


# ── JustType enum coverage ──────────────────────────────────────────────────


class TestJustTypeEnum:
    """Verify JustType enum has all C Prover9 justification types."""

    def test_primary_justification_types(self) -> None:
        """All primary justification types exist."""
        primary = [
            JustType.INPUT, JustType.GOAL, JustType.DENY,
            JustType.CLAUSIFY, JustType.COPY, JustType.BINARY_RES,
            JustType.HYPER_RES, JustType.UR_RES, JustType.FACTOR,
            JustType.PARA,
        ]
        assert len(primary) == 10

    def test_secondary_justification_types(self) -> None:
        """All secondary (simplification) justification types exist."""
        secondary = [
            JustType.DEMOD, JustType.UNIT_DEL, JustType.FLIP,
            JustType.BACK_DEMOD, JustType.BACK_UNIT_DEL,
            JustType.NEW_SYMBOL, JustType.EXPAND_DEF, JustType.FOLD_DEF,
            JustType.RENUMBER, JustType.PROPOSITIONAL,
            JustType.INSTANTIATE, JustType.IVY,
        ]
        assert len(secondary) == 12

    def test_total_justification_types(self) -> None:
        """JustType has 22 total values (matching C)."""
        assert len(JustType) == 22
