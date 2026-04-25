"""Subformula repetition penalty for clauses with repeated structural patterns.

Clauses containing repeated subformula patterns (e.g., multiple occurrences
of i(x,x) or f(g(a),g(a))) are penalized during selection to deprioritize
structurally redundant clauses that are less likely to contribute to proof search.

Two matching modes:
  - Exact matching (Phase 1): identical Term objects (same variables, same structure)
  - Normalized matching (Phase 2): variable-agnostic matching where i(x,x) ≡ i(y,y)

The penalty is computed as: base_penalty * sum(count - 1) for each repeated pattern,
capped at max_penalty. This is added to the clause's penalty_override in PrioritySOS.

Design:
    - Disabled by default (opt-in via SearchOptions.repetition_penalty)
    - Composable with existing penalty_propagation system (additive combination)
    - O(n) per clause where n = total subterms
    - Uses Term's frozen dataclass hash/eq for O(1) matching

Performance optimizations:
    - Ground term fast path in normalization (avoids DFS when no variables present)
    - Per-clause normalization cache (avoids re-normalizing shared subterms)
    - Module-level imports (avoids per-call import overhead)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyladr.core.term import Term as _Term, get_variable_term as _get_variable_term

if TYPE_CHECKING:
    from pyladr.core.clause import Clause


@dataclass(frozen=True, slots=True)
class RepetitionPenaltyConfig:
    """Configuration for subformula repetition penalty.

    All fields have safe defaults that disable or minimize the feature
    when not explicitly configured.
    """

    enabled: bool = False
    base_penalty: float = 2.0       # Penalty per extra occurrence of a repeated subterm
    min_subterm_size: int = 2        # Minimum symbol_count to consider (skip vars/constants)
    max_penalty: float = 15.0       # Cap on total repetition penalty
    normalize_variables: bool = False  # Phase 2: variable-agnostic matching


def compute_repetition_penalty(clause: Clause, config: RepetitionPenaltyConfig) -> float:
    """Compute penalty for repeated subformula patterns in a clause.

    Traverses all subterms across all literals, counts occurrences of each
    subterm (filtered by min_subterm_size), and penalizes each extra
    occurrence beyond the first.

    Args:
        clause: The clause to analyze.
        config: Repetition penalty configuration.

    Returns:
        Penalty score >= 0.0. Higher = more repetition (deprioritized).
    """
    num_lits = len(clause.literals)
    if num_lits == 0:
        return 0.0

    # Early termination: small clauses can't have meaningful repetition
    total_symbols = sum(lit.atom.symbol_count for lit in clause.literals)
    if total_symbols < config.min_subterm_size * 2:
        return 0.0

    if config.normalize_variables:
        return _penalty_normalized(clause, config)
    return _penalty_exact(clause, config)


def _penalty_exact(clause: Clause, config: RepetitionPenaltyConfig) -> float:
    """Compute repetition penalty using exact structural matching.

    Uses Term's frozen dataclass __hash__/__eq__ for O(1) matching.
    """
    min_size = config.min_subterm_size
    counts: dict[_Term, int] = {}

    for lit in clause.literals:
        for subterm in lit.atom.subterms():
            if subterm._symbol_count >= min_size:
                counts[subterm] = counts.get(subterm, 0) + 1

    # Sum (count - 1) for each repeated pattern
    total_extra = 0
    for count in counts.values():
        if count > 1:
            total_extra += count - 1

    if total_extra == 0:
        return 0.0

    return min(config.base_penalty * total_extra, config.max_penalty)


def _penalty_normalized(clause: Clause, config: RepetitionPenaltyConfig) -> float:
    """Compute repetition penalty using variable-normalized matching.

    Normalizes variables to canonical DFS ordering before counting,
    so i(x,x) and i(y,y) are treated as identical patterns.

    Optimization: uses a per-clause normalization cache keyed by term id()
    to avoid re-normalizing shared subterms that appear at multiple positions.
    """
    min_size = config.min_subterm_size
    counts: dict[_Term, int] = {}
    # Per-clause cache: original_term -> normalized_term
    # Term is a frozen dataclass with __hash__/__eq__, so it's a valid dict key.
    # Avoids redundant DFS for subterms shared across literals or
    # repeated within a single term tree.
    norm_cache: dict[_Term, _Term] = {}

    for lit in clause.literals:
        for subterm in lit.atom.subterms():
            if subterm._symbol_count >= min_size:
                normalized = norm_cache.get(subterm)
                if normalized is None:
                    normalized = _normalize_variables(subterm)
                    norm_cache[subterm] = normalized
                counts[normalized] = counts.get(normalized, 0) + 1

    total_extra = 0
    for count in counts.values():
        if count > 1:
            total_extra += count - 1

    if total_extra == 0:
        return 0.0

    return min(config.base_penalty * total_extra, config.max_penalty)


def _normalize_variables(term: _Term) -> _Term:
    """Replace all variables with canonical numbering (DFS order).

    i(x,x) and i(y,y) both normalize to i(v0,v0).
    f(x,g(y)) and f(z,g(w)) both normalize to f(v0,g(v1)).

    Optimizations:
    - Ground terms (no variables) are returned immediately without traversal.
    - Uses module-level imports to avoid per-call import overhead.

    Args:
        term: The term to normalize.

    Returns:
        A new term with canonically numbered variables.
    """
    # Fast path: ground terms have no variables to normalize
    if term.is_ground:
        return term

    mapping: dict[int, int] = {}
    counter = 0

    def _normalize(t: _Term) -> _Term:
        nonlocal counter
        if t.is_variable:
            vn = t.varnum
            if vn not in mapping:
                mapping[vn] = counter
                counter += 1
            return _get_variable_term(mapping[vn])
        if t.is_constant:
            return t
        new_args = tuple(_normalize(a) for a in t.args)
        if all(new_args[i] is t.args[i] for i in range(len(new_args))):
            return t  # No variables changed — return original for hash reuse
        return _Term(private_symbol=t.private_symbol, arity=t.arity, args=new_args)

    return _normalize(term)
