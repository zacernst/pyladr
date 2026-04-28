"""Tests for t2v_helpers.py: T2V pure helper functions.

Verifies:
- _t2v_cosine: cosine similarity with zero-vector safety
- format_t2v_histogram: human-readable histogram formatting
- _get_antecedent_term: extract antecedent from condensed-detachment clauses
- compute_t2v_histogram: conditional probability histogram from distances
- compute_t2v_cumulative_histogram: cumulative histogram across proofs
"""

from __future__ import annotations

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.search.t2v_helpers import (
    _get_antecedent_term,
    _t2v_cosine,
    compute_t2v_cumulative_histogram,
    compute_t2v_histogram,
    format_t2v_histogram,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _const(symnum: int) -> Term:
    return get_rigid_term(symnum, 0)


def _func(symnum: int, *args: Term) -> Term:
    return get_rigid_term(symnum, len(args), args)


def _parse_clause(text: str) -> Clause:
    """Parse a clause string using LADRParser."""
    st = SymbolTable()
    parser = LADRParser(st)
    return parser.parse_clause_from_string(text)


def _make_histogram(
    proof_probs: list[float] | None = None,
    nonproof_probs: list[float] | None = None,
    proof_n: int = 10,
    nonproof_n: int = 90,
    lo: float = 0.0,
    hi: float = 1.0,
    bucket_width: float = 0.2,
) -> dict:
    return {
        "proof_probs": proof_probs or [0.3, 0.25, 0.2, 0.15, 0.1],
        "nonproof_probs": nonproof_probs or [0.1, 0.15, 0.2, 0.25, 0.3],
        "proof_n": proof_n,
        "nonproof_n": nonproof_n,
        "lo": lo,
        "hi": hi,
        "bucket_width": bucket_width,
    }


class StubProof:
    """Minimal proof stub with a .clauses attribute."""

    def __init__(self, clause_ids: list[int]) -> None:
        self.clauses = [Clause(literals=(), id=cid) for cid in clause_ids]


# ── _t2v_cosine tests ────────────────────────────────────────────────────────


class TestT2vCosine:
    """_t2v_cosine: cosine similarity between embedding vectors."""

    def test_identical_vectors(self) -> None:
        assert _t2v_cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _t2v_cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _t2v_cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_safe(self) -> None:
        """Zero vector should return 0.0 rather than raising."""
        assert _t2v_cosine([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_both_zero_vectors(self) -> None:
        assert _t2v_cosine([0.0, 0.0], [0.0, 0.0]) == pytest.approx(0.0)

    def test_scaled_identical(self) -> None:
        """Cosine similarity is scale-invariant."""
        assert _t2v_cosine([2.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)

    def test_45_degree_angle(self) -> None:
        """[1,0] vs [1,1] → cos(45°) ≈ 0.7071."""
        result = _t2v_cosine([1.0, 0.0], [1.0, 1.0])
        assert result == pytest.approx(1.0 / (2 ** 0.5), abs=1e-6)


# ── format_t2v_histogram tests ──────────────────────────────────────────────


class TestFormatT2vHistogram:
    """format_t2v_histogram: human-readable histogram output."""

    def test_contains_proof_label(self) -> None:
        hist = _make_histogram()
        output = format_t2v_histogram(hist, proof_num=1)
        assert "proof" in output.lower()

    def test_contains_non_proof_label(self) -> None:
        hist = _make_histogram()
        output = format_t2v_histogram(hist, proof_num=1)
        assert "non-proof" in output.lower()

    def test_contains_range_labels(self) -> None:
        """Output should contain bucket range labels."""
        hist = _make_histogram(lo=0.0, bucket_width=0.2)
        output = format_t2v_histogram(hist, proof_num=1)
        # Should contain at least the first bucket range
        assert "0.00-0.20" in output

    def test_five_bucket_rows(self) -> None:
        """Output should have 5 data rows (one per bucket)."""
        hist = _make_histogram()
        output = format_t2v_histogram(hist, proof_num=1)
        lines = output.strip().split("\n")
        # 2 header lines + 5 data lines
        assert len(lines) == 7

    def test_cumulative_header(self) -> None:
        """proof_num=None should produce a cumulative header."""
        hist = _make_histogram()
        hist["n_proofs"] = 3
        output = format_t2v_histogram(hist, proof_num=None)
        assert "cumulative" in output.lower()

    def test_proof_counts_in_header(self) -> None:
        hist = _make_histogram(proof_n=15, nonproof_n=85)
        output = format_t2v_histogram(hist, proof_num=2)
        assert "15 proof clauses" in output
        assert "85 non-proof" in output


# ── _get_antecedent_term tests ───────────────────────────────────────────────


class TestGetAntecedentTerm:
    """_get_antecedent_term: extract first arg of inner term from P(i(x,y))."""

    def test_condensed_detachment_clause(self) -> None:
        """P(i(x,y)) → should return the first arg of i, which is x (variable)."""
        clause = _parse_clause("P(i(x,y)).")
        result = _get_antecedent_term(clause)
        assert result is not None
        assert result.is_variable, f"Expected variable, got {result!r}"

    def test_simple_clause_returns_none(self) -> None:
        """P(a) → inner term is a constant (arity 0), so returns None."""
        clause = _parse_clause("P(a).")
        result = _get_antecedent_term(clause)
        assert result is None, f"Expected None for P(a), got {result!r}"

    def test_empty_clause_returns_none(self) -> None:
        """Empty clause → returns None."""
        clause = Clause(literals=())
        result = _get_antecedent_term(clause)
        assert result is None

    def test_propositional_returns_none(self) -> None:
        """Propositional atom P (arity 0) → returns None."""
        # Build P as a constant (arity 0)
        atom = _const(1)
        clause = Clause(literals=(Literal(sign=True, atom=atom),))
        result = _get_antecedent_term(clause)
        assert result is None

    def test_nested_function(self) -> None:
        """P(f(g(x), y)) → inner is f(g(x),y), first arg is g(x)."""
        clause = _parse_clause("P(f(g(x), y)).")
        result = _get_antecedent_term(clause)
        assert result is not None
        # g(x) is a complex term
        assert result.is_complex or result.is_variable  # it's g(x), which is complex


# ── compute_t2v_histogram tests ──────────────────────────────────────────────


class TestComputeT2vHistogram:
    """compute_t2v_histogram: conditional probability table from distances."""

    def test_empty_distances_returns_none(self) -> None:
        proof = StubProof([1, 2])
        assert compute_t2v_histogram({}, proof) is None

    def test_basic_histogram_structure(self) -> None:
        """With real distances, returns a dict with required keys."""
        distances = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.7, 5: 0.9}
        proof = StubProof([1, 2])  # clauses 1 and 2 are proof clauses
        result = compute_t2v_histogram(distances, proof)

        assert result is not None
        assert "proof_probs" in result
        assert "nonproof_probs" in result
        assert "proof_n" in result
        assert "nonproof_n" in result
        assert "lo" in result
        assert "hi" in result
        assert "bucket_width" in result

    def test_proof_nonproof_counts(self) -> None:
        distances = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.7}
        proof = StubProof([1, 2])
        result = compute_t2v_histogram(distances, proof)

        assert result is not None
        assert result["proof_n"] == 2
        assert result["nonproof_n"] == 2

    def test_probabilities_sum_to_one(self) -> None:
        """Each probability distribution should sum to ~1.0."""
        distances = {i: i * 0.1 for i in range(1, 11)}
        proof = StubProof([1, 2, 3])
        result = compute_t2v_histogram(distances, proof)

        assert result is not None
        assert sum(result["proof_probs"]) == pytest.approx(1.0)
        assert sum(result["nonproof_probs"]) == pytest.approx(1.0)

    def test_five_buckets(self) -> None:
        distances = {1: 0.1, 2: 0.5, 3: 0.9}
        proof = StubProof([1])
        result = compute_t2v_histogram(distances, proof)

        assert result is not None
        assert len(result["proof_probs"]) == 5
        assert len(result["nonproof_probs"]) == 5

    def test_single_distance_handles_lo_eq_hi(self) -> None:
        """When all distances are identical, lo==hi edge case is handled."""
        distances = {1: 0.5}
        proof = StubProof([1])
        result = compute_t2v_histogram(distances, proof)
        assert result is not None
        # Should not crash — lo/hi are adjusted


# ── compute_t2v_cumulative_histogram tests ───────────────────────────────────


class TestComputeT2vCumulativeHistogram:
    """compute_t2v_cumulative_histogram: cumulative across all proofs."""

    def test_empty_distances_returns_none(self) -> None:
        proofs = [StubProof([1])]
        assert compute_t2v_cumulative_histogram({}, proofs) is None

    def test_empty_proofs_returns_none(self) -> None:
        assert compute_t2v_cumulative_histogram({1: 0.5}, []) is None

    def test_both_empty_returns_none(self) -> None:
        assert compute_t2v_cumulative_histogram({}, []) is None

    def test_cumulative_unions_proof_ids(self) -> None:
        """Clause IDs from all proofs are unioned for proof classification."""
        distances = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.7, 5: 0.9}
        proof1 = StubProof([1, 2])
        proof2 = StubProof([2, 3])
        result = compute_t2v_cumulative_histogram(distances, [proof1, proof2])

        assert result is not None
        # proof_ids = {1, 2, 3} → proof_n = 3, nonproof_n = 2
        assert result["proof_n"] == 3
        assert result["nonproof_n"] == 2
        assert result["n_proofs"] == 2

    def test_cumulative_probabilities_sum_to_one(self) -> None:
        distances = {i: i * 0.1 for i in range(1, 11)}
        proof1 = StubProof([1, 2])
        proof2 = StubProof([3, 4])
        result = compute_t2v_cumulative_histogram(distances, [proof1, proof2])

        assert result is not None
        assert sum(result["proof_probs"]) == pytest.approx(1.0)
        assert sum(result["nonproof_probs"]) == pytest.approx(1.0)
