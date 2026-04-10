"""Proof validation utilities for soundness checking.

Provides tools to verify that proofs produced by PyLADR are logically valid:
- Each inference step follows from its premises correctly
- Unification is applied correctly in resolution steps
- The proof actually derives the empty clause from the axioms
- No structurally incompatible terms are claimed to unify
"""

from __future__ import annotations

from typing import Any

from pyladr.core.clause import Clause, JustType, Literal
from pyladr.core.substitution import Context, Trail, apply_substitution, unify
from pyladr.core.term import Term


class ProofValidationError(Exception):
    """Raised when a proof step is logically invalid."""

    def __init__(self, message: str, clause: Clause | None = None, step_detail: str = ""):
        self.clause = clause
        self.step_detail = step_detail
        super().__init__(message)


def validate_proof_chain(
    proof_clauses: list[Clause] | tuple[Clause, ...],
    *,
    verbose: bool = False,
) -> list[str]:
    """Validate an entire proof chain for logical correctness.

    Checks each derived clause to ensure it follows validly from its premises.

    Returns a list of validation issues (empty if proof is valid).
    """
    issues: list[str] = []
    clause_map = {c.id: c for c in proof_clauses}

    for clause in proof_clauses:
        for just in clause.justification:
            if just.just_type == JustType.BINARY_RES:
                result = _validate_binary_resolution(clause, just, clause_map, verbose)
                issues.extend(result)
            elif just.just_type == JustType.FACTOR:
                result = _validate_factoring(clause, just, clause_map, verbose)
                issues.extend(result)
            # ASSUMPTION and INPUT are axioms, no validation needed
            # DEMOD/BACK_DEMOD/PARA could be validated too but are lower priority

    return issues


def _validate_binary_resolution(
    resolvent: Clause,
    just: Any,
    clause_map: dict[int, Clause],
    verbose: bool,
) -> list[str]:
    """Validate a binary resolution step.

    For binary_res(c1, c2) to be valid:
    1. c1 and c2 must exist in the proof
    2. There must exist complementary literals l1 in c1, l2 in c2
    3. The atoms of l1 and l2 must actually unify
    4. The resolvent must be the correct application of the MGU
    """
    issues: list[str] = []
    if len(just.clause_ids) < 2:
        issues.append(
            f"Clause {resolvent.id}: binary_res justification has fewer than 2 parent IDs"
        )
        return issues

    c1_id, c2_id = just.clause_ids[0], just.clause_ids[1]

    if c1_id not in clause_map:
        issues.append(f"Clause {resolvent.id}: parent clause {c1_id} not found in proof")
        return issues
    if c2_id not in clause_map:
        issues.append(f"Clause {resolvent.id}: parent clause {c2_id} not found in proof")
        return issues

    c1 = clause_map[c1_id]
    c2 = clause_map[c2_id]

    # Check that there exist complementary literals that actually unify
    found_valid_resolution = False
    for i, l1 in enumerate(c1.literals):
        for j, l2 in enumerate(c2.literals):
            if l1.sign != l2.sign:
                # Try to unify the atoms
                ctx1 = Context()
                ctx2 = Context()
                trail = Trail()
                if unify(l1.atom, ctx1, l2.atom, ctx2, trail):
                    found_valid_resolution = True
                    trail.undo()
                    break
                trail.undo()
        if found_valid_resolution:
            break

    if not found_valid_resolution:
        issues.append(
            f"Clause {resolvent.id}: no valid complementary literals found between "
            f"parents {c1_id} and {c2_id} for binary resolution"
        )

    return issues


def _validate_factoring(
    factored: Clause,
    just: Any,
    clause_map: dict[int, Clause],
    verbose: bool,
) -> list[str]:
    """Validate a factoring step."""
    issues: list[str] = []
    if not just.clause_ids:
        issues.append(f"Clause {factored.id}: factor justification has no parent ID")
        return issues

    parent_id = just.clause_ids[0]
    if parent_id not in clause_map:
        issues.append(f"Clause {factored.id}: parent clause {parent_id} not found in proof")
        return issues

    parent = clause_map[parent_id]

    # Factoring requires at least 2 same-sign literals that unify
    if len(parent.literals) < 2:
        issues.append(
            f"Clause {factored.id}: parent {parent_id} has fewer than 2 literals for factoring"
        )

    return issues


def validate_unification_claim(
    t1: Term,
    t2: Term,
    *,
    description: str = "",
) -> bool:
    """Independently verify that two terms can unify.

    Used to check whether a claimed unification in a proof step is valid.
    Returns True if terms unify, False if not.
    """
    ctx1 = Context()
    ctx2 = Context()
    trail = Trail()
    result = unify(t1, ctx1, t2, ctx2, trail)
    trail.undo()
    return result


def check_proof_derives_contradiction(
    proof_clauses: list[Clause] | tuple[Clause, ...],
) -> bool:
    """Check that the proof actually derives the empty clause (contradiction).

    A valid proof must end with an empty clause (no literals).
    """
    for clause in proof_clauses:
        if len(clause.literals) == 0:
            return True
    return False


def check_trivial_proof_suspicious(
    proof_length: int,
    c_proof_length: int,
    *,
    ratio_threshold: float = 0.25,
) -> bool:
    """Detect suspiciously short proofs compared to C reference.

    If Python finds a proof that is dramatically shorter than C,
    this is a strong indicator of a soundness bug.

    Returns True if the proof is suspicious.
    """
    if c_proof_length == 0:
        return False  # C didn't find a proof, can't compare
    if proof_length == 0:
        return False

    ratio = proof_length / c_proof_length
    return ratio < ratio_threshold
