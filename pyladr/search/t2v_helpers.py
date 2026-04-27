"""T2V (Tree2Vec) helper functions for goal-distance analysis.

Extracted from GivenClauseSearch — these are pure functions with no
search-loop coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyladr.core.clause import Clause
    from pyladr.core.term import Term


def format_t2v_histogram(histogram: dict, proof_num: int | None = 1) -> str:
    """Format a T2V goal-distance histogram as a conditional probability table.

    Distances are in [0, 1]: 0.0 = identical to goal, 1.0 = maximally distant.
    When *proof_num* is ``None`` the header reads "cumulative" instead of
    referencing a single proof number.
    """
    proof_probs = histogram["proof_probs"]
    nonproof_probs = histogram["nonproof_probs"]
    proof_n = histogram["proof_n"]
    nonproof_n = histogram["nonproof_n"]
    lo = histogram["lo"]
    bw = histogram["bucket_width"]
    if proof_num is None:
        n_proofs = histogram.get("n_proofs", "?")
        header = (
            f"T2V goal distance (cumulative, {n_proofs} proofs,"
            f" {proof_n} proof clauses, {nonproof_n} non-proof):"
        )
    else:
        header = f"T2V goal distance at proof {proof_num} ({proof_n} proof clauses, {nonproof_n} non-proof):"
    lines = [
        header,
        "  range            P(range|proof)  P(range|non-proof)",
    ]
    for i in range(5):
        lo_edge = lo + i * bw
        hi_edge = lo + (i + 1) * bw
        label = f"{lo_edge:.2f}-{hi_edge:.2f}"
        lines.append(
            f"  {label}:  {proof_probs[i]:>16.4f}  {nonproof_probs[i]:>18.4f}"
        )
    return "\n".join(lines)


# ── Cross-argument distance helpers ─────────────────────────────────────────

def _get_antecedent_term(clause: "Clause") -> "Term | None":
    """Extract the antecedent term from a clause of the form P(i(x,y)).

    Returns the first arg of the inner term (x), or None if the clause
    doesn't have the expected structure.
    """
    if not clause.literals:
        return None
    atom = clause.literals[0].atom  # P(...)
    if atom.arity < 1:
        return None
    inner = atom.args[0]  # i(x,y)
    if inner.arity < 1:
        return None
    return inner.args[0]  # x


def _t2v_cosine(a: "list[float]", b: "list[float]") -> float:
    """Cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na * nb > 1e-12 else 0.0


# ── T2V histogram computation ───────────────────────────────────────────────


def compute_t2v_histogram(
    all_given_distances: dict[int, float], proof: object
) -> dict | None:
    """Compute conditional probability histogram: P(range|proof) vs P(range|non-proof).

    Splits all given clause goal-distances into proof vs non-proof populations,
    buckets both, and normalizes to probabilities.
    """
    if not all_given_distances:
        return None

    proof_ids = {c.id for c in proof.clauses}
    proof_scores: list[float] = []
    nonproof_scores: list[float] = []
    all_scores: list[float] = []

    for cid, score in all_given_distances.items():
        all_scores.append(score)
        if cid in proof_ids:
            proof_scores.append(score)
        else:
            nonproof_scores.append(score)

    if not all_scores:
        return None

    lo = min(all_scores)
    hi = max(all_scores)
    if hi == lo:
        lo = max(0.0, lo - 0.05)
        hi = min(1.0, hi + 0.05)
    bucket_width = (hi - lo) / 5

    def _bucket_and_normalize(scores: list[float]) -> list[float]:
        counts = [0, 0, 0, 0, 0]
        for s in scores:
            counts[min(4, int((s - lo) / bucket_width))] += 1
        n = len(scores)
        return [c / n if n > 0 else 0.0 for c in counts]

    return {
        "proof_probs": _bucket_and_normalize(proof_scores),
        "nonproof_probs": _bucket_and_normalize(nonproof_scores),
        "proof_n": len(proof_scores),
        "nonproof_n": len(nonproof_scores),
        "lo": lo,
        "hi": hi,
        "bucket_width": bucket_width,
    }


def compute_t2v_cumulative_histogram(
    all_given_distances: dict[int, float], proofs: list
) -> dict | None:
    """Compute cumulative goal-distance histogram across all proofs found so far.

    Same logic as ``compute_t2v_histogram`` but unions clause IDs from
    every proof in *proofs*.
    """
    if not all_given_distances or not proofs:
        return None

    proof_ids: set[int] = set()
    for p in proofs:
        for c in p.clauses:
            proof_ids.add(c.id)

    proof_scores: list[float] = []
    nonproof_scores: list[float] = []
    all_scores: list[float] = []

    for cid, score in all_given_distances.items():
        all_scores.append(score)
        if cid in proof_ids:
            proof_scores.append(score)
        else:
            nonproof_scores.append(score)

    if not all_scores:
        return None

    lo = min(all_scores)
    hi = max(all_scores)
    if hi == lo:
        lo = max(0.0, lo - 0.05)
        hi = min(1.0, hi + 0.05)
    bucket_width = (hi - lo) / 5

    def _bucket_and_normalize(scores: list[float]) -> list[float]:
        counts = [0, 0, 0, 0, 0]
        for s in scores:
            counts[min(4, int((s - lo) / bucket_width))] += 1
        n = len(scores)
        return [c / n if n > 0 else 0.0 for c in counts]

    return {
        "proof_probs": _bucket_and_normalize(proof_scores),
        "nonproof_probs": _bucket_and_normalize(nonproof_scores),
        "proof_n": len(proof_scores),
        "nonproof_n": len(nonproof_scores),
        "n_proofs": len(proofs),
        "lo": lo,
        "hi": hi,
        "bucket_width": bucket_width,
    }
