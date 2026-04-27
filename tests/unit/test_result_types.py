"""Unit tests for pyladr.search.result_types — ExitCode, Proof, SearchResult."""

from __future__ import annotations

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import get_rigid_term
from pyladr.search.result_types import ExitCode, Proof, SearchResult
from pyladr.search.statistics import SearchStatistics


def _unit_clause(clause_id: int = 1) -> Clause:
    atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
    return Clause(literals=(Literal(sign=True, atom=atom),), id=clause_id)


def _empty_clause(clause_id: int = 99) -> Clause:
    return Clause(literals=(), id=clause_id)


# ── ExitCode tests ───────────────────────────────────────────────────────────


class TestExitCode:
    def test_values_match_c_exit_codes(self):
        """Exit codes match C search.h values 1-7."""
        assert ExitCode.MAX_PROOFS_EXIT == 1
        assert ExitCode.SOS_EMPTY_EXIT == 2
        assert ExitCode.MAX_GIVEN_EXIT == 3
        assert ExitCode.MAX_KEPT_EXIT == 4
        assert ExitCode.MAX_SECONDS_EXIT == 5
        assert ExitCode.MAX_GENERATED_EXIT == 6
        assert ExitCode.FATAL_EXIT == 7

    def test_is_int_enum(self):
        """ExitCode values are usable as plain ints."""
        assert ExitCode.MAX_PROOFS_EXIT + 1 == 2
        assert int(ExitCode.FATAL_EXIT) == 7


# ── Proof tests ──────────────────────────────────────────────────────────────


class TestProof:
    def test_constructable_with_clause_tuple(self):
        """Proof holds an empty clause and a tuple of proof clauses."""
        empty = _empty_clause()
        c1 = _unit_clause(1)
        c2 = _unit_clause(2)
        proof = Proof(empty_clause=empty, clauses=(c1, c2))
        assert proof.empty_clause is empty
        assert len(proof.clauses) == 2
        assert proof.clauses[0].id == 1
        assert proof.clauses[1].id == 2

    def test_frozen(self):
        """Proof is immutable (frozen dataclass)."""
        proof = Proof(empty_clause=_empty_clause(), clauses=())
        import pytest
        with pytest.raises(AttributeError):
            proof.empty_clause = _empty_clause(100)  # type: ignore[misc]

    def test_repr(self):
        """Proof repr shows empty clause id and length."""
        proof = Proof(empty_clause=_empty_clause(42), clauses=(_unit_clause(1),))
        r = repr(proof)
        assert "42" in r
        assert "length=1" in r


# ── SearchResult tests ───────────────────────────────────────────────────────


class TestSearchResult:
    def test_carries_exit_code_and_proofs(self):
        """SearchResult holds exit code, proofs tuple, and stats."""
        stats = SearchStatistics()
        proof = Proof(empty_clause=_empty_clause(), clauses=(_unit_clause(),))
        result = SearchResult(
            exit_code=ExitCode.MAX_PROOFS_EXIT,
            proofs=(proof,),
            stats=stats,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1
        assert result.stats is stats

    def test_no_proofs(self):
        """SearchResult with empty proofs tuple (e.g., SOS_EMPTY)."""
        stats = SearchStatistics()
        result = SearchResult(
            exit_code=ExitCode.SOS_EMPTY_EXIT,
            proofs=(),
            stats=stats,
        )
        assert result.exit_code == ExitCode.SOS_EMPTY_EXIT
        assert len(result.proofs) == 0


# ── Re-export compatibility ──────────────────────────────────────────────────


class TestReExportCompatibility:
    def test_given_clause_re_exports_exit_code(self):
        """ExitCode is importable from given_clause (backward compat)."""
        from pyladr.search.given_clause import ExitCode as GCExitCode
        assert GCExitCode is ExitCode

    def test_given_clause_re_exports_search_result(self):
        """SearchResult is importable from given_clause (backward compat)."""
        from pyladr.search.given_clause import SearchResult as GCSearchResult
        assert GCSearchResult is SearchResult
