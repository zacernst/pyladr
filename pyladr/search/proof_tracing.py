"""Proof-trace reconstruction from an empty clause's justification chain.

Given an empty clause and the map of all clauses by ID, walk the justification
DAG backwards (via clause_ids, clause_id, and para refs) to collect every
ancestor clause in the proof. Matches C prover9's proof_id_set() /
get_clause_by_id() pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyladr.core.clause import Clause


def trace_proof(
    empty: "Clause",
    all_clauses_by_id: dict[int, "Clause"],
) -> list["Clause"]:
    """Collect every clause reachable from `empty` via justification links.

    Args:
        empty: The derived empty clause that anchors the proof.
        all_clauses_by_id: Map from clause ID to clause for all clauses seen
            during the search (search engine's `_all_clauses`).

    Returns:
        Deduplicated list of ancestor clauses, sorted by ID for determinism.
    """
    visited: set[int] = set()
    proof_clauses: list["Clause"] = []
    stack: list["Clause"] = [empty]

    while stack:
        c = stack.pop()
        if c.id in visited:
            continue
        visited.add(c.id)
        proof_clauses.append(c)

        for just in c.justification:
            for cid in just.clause_ids:
                if cid in all_clauses_by_id and cid not in visited:
                    stack.append(all_clauses_by_id[cid])
            if just.clause_id > 0 and just.clause_id in all_clauses_by_id:
                if just.clause_id not in visited:
                    stack.append(all_clauses_by_id[just.clause_id])
            if just.para is not None:
                for pid in (just.para.from_id, just.para.into_id):
                    if pid in all_clauses_by_id and pid not in visited:
                        stack.append(all_clauses_by_id[pid])

    proof_clauses.sort(key=lambda c: c.id)
    return proof_clauses
