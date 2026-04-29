"""Unit tests for trace_proof: pure proof-trace reconstruction.

Exercises trace_proof() directly with hand-constructed justification chains —
no search engine required. Verifies ancestor collection across clause_id,
clause_ids, and para links; deduplication; deterministic ordering.
"""

from __future__ import annotations

from pyladr.core.clause import Clause, Justification, JustType, Literal, ParaJust
from pyladr.core.term import get_rigid_term
from pyladr.search.proof_tracing import trace_proof


SYM_P = 1
SYM_A = 2


def _atom_p(arg_sym: int) -> Literal:
    return Literal(
        sign=True,
        atom=get_rigid_term(SYM_P, 1, (get_rigid_term(arg_sym, 0),)),
    )


def _input_clause(cid: int) -> Clause:
    return Clause(
        literals=(_atom_p(SYM_A),),
        id=cid,
        justification=(
            Justification(just_type=JustType.INPUT, clause_id=0, clause_ids=()),
        ),
    )


def _derived_clause(
    cid: int,
    parent_ids: tuple[int, ...],
    just_type: JustType = JustType.BINARY_RES,
) -> Clause:
    return Clause(
        literals=(_atom_p(SYM_A),),
        id=cid,
        justification=(
            Justification(just_type=just_type, clause_id=0, clause_ids=parent_ids),
        ),
    )


class TestTraceProofPure:
    def test_empty_clause_alone(self) -> None:
        """Input empty clause has no parents → returns just itself."""
        empty = _input_clause(cid=1)
        result = trace_proof(empty, {1: empty})
        assert [c.id for c in result] == [1]

    def test_collects_linear_chain_via_clause_ids(self) -> None:
        """A → B → C chain via clause_ids traversal."""
        a = _input_clause(cid=1)
        b = _derived_clause(cid=2, parent_ids=(1,))
        c = _derived_clause(cid=3, parent_ids=(2,))
        all_clauses = {1: a, 2: b, 3: c}
        result = trace_proof(c, all_clauses)
        assert [cl.id for cl in result] == [1, 2, 3]

    def test_collects_branch_via_multiple_parents(self) -> None:
        """Empty clause with two parents returns all ancestors."""
        a = _input_clause(cid=1)
        b = _input_clause(cid=2)
        empty = _derived_clause(cid=3, parent_ids=(1, 2))
        result = trace_proof(empty, {1: a, 2: b, 3: empty})
        assert sorted(cl.id for cl in result) == [1, 2, 3]

    def test_deduplicates_shared_ancestors(self) -> None:
        """Diamond DAG: shared ancestor appears exactly once."""
        a = _input_clause(cid=1)
        b = _derived_clause(cid=2, parent_ids=(1,))
        c = _derived_clause(cid=3, parent_ids=(1,))
        empty = _derived_clause(cid=4, parent_ids=(2, 3))
        result = trace_proof(empty, {1: a, 2: b, 3: c, 4: empty})
        ids = [cl.id for cl in result]
        assert ids == [1, 2, 3, 4]  # deduplicated and sorted

    def test_output_sorted_by_id_deterministic(self) -> None:
        """Result always sorted by id — independent of DFS traversal order."""
        a = _input_clause(cid=10)
        b = _input_clause(cid=5)
        empty = _derived_clause(cid=20, parent_ids=(10, 5))
        result = trace_proof(empty, {5: b, 10: a, 20: empty})
        assert [cl.id for cl in result] == [5, 10, 20]

    def test_missing_parent_id_is_skipped(self) -> None:
        """Parent id absent from all_clauses_by_id is skipped gracefully."""
        empty = _derived_clause(cid=2, parent_ids=(1, 999))  # 999 missing
        # Only 2 is known; 1 and 999 missing
        result = trace_proof(empty, {2: empty})
        assert [cl.id for cl in result] == [2]

    def test_follows_single_clause_id_field(self) -> None:
        """Justification.clause_id (singular, > 0) is also followed."""
        a = _input_clause(cid=1)
        # Justification with clause_id=1 (singular, not in clause_ids)
        b = Clause(
            literals=(_atom_p(SYM_A),),
            id=2,
            justification=(
                Justification(
                    just_type=JustType.FACTOR, clause_id=1, clause_ids=()
                ),
            ),
        )
        result = trace_proof(b, {1: a, 2: b})
        assert [cl.id for cl in result] == [1, 2]

    def test_ignores_zero_clause_id(self) -> None:
        """clause_id == 0 is the INPUT sentinel and must not be followed."""
        a = _input_clause(cid=1)
        result = trace_proof(a, {1: a})
        # INPUT justification has clause_id=0 — must not try to look up id 0.
        assert [cl.id for cl in result] == [1]

    def test_follows_para_from_into(self) -> None:
        """ParaJust's from_id and into_id both become ancestors."""
        a = _input_clause(cid=1)
        b = _input_clause(cid=2)
        para = ParaJust(
            from_id=1, into_id=2,
            from_pos=(), into_pos=(),
        )
        empty = Clause(
            literals=(_atom_p(SYM_A),),
            id=3,
            justification=(
                Justification(
                    just_type=JustType.PARA, clause_id=0, clause_ids=(), para=para,
                ),
            ),
        )
        result = trace_proof(empty, {1: a, 2: b, 3: empty})
        assert sorted(cl.id for cl in result) == [1, 2, 3]

    def test_handles_cycle_in_justification_graph(self) -> None:
        """Pathological cycle: termination guaranteed by visited set."""
        # Manually construct a cycle (should never happen in practice, but
        # trace_proof must not infinite-loop if it does).
        a = Clause(
            literals=(_atom_p(SYM_A),),
            id=1,
            justification=(
                Justification(just_type=JustType.BINARY_RES, clause_id=0, clause_ids=(2,)),
            ),
        )
        b = Clause(
            literals=(_atom_p(SYM_A),),
            id=2,
            justification=(
                Justification(just_type=JustType.BINARY_RES, clause_id=0, clause_ids=(1,)),
            ),
        )
        result = trace_proof(a, {1: a, 2: b})
        assert sorted(cl.id for cl in result) == [1, 2]
