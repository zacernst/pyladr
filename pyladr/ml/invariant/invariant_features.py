"""Property-invariant feature extraction for clause graph nodes.

Replaces raw symnum-based features with structural properties that are
independent of symbol naming. The key insight: instead of encoding symbol
identity (which is arbitrary), we encode the symbol's *role* — arity,
whether it's a predicate or function, Skolem status, and occurrence patterns.

This produces embeddings that are invariant under symbol renaming:
  P(f(x), g(y)) ≡ Q(h(x), k(y))  (same structure, different names)

Feature vectors have the same dimensionality as the original symbol features
(6 floats) for drop-in compatibility with the existing GNN architecture.

Invariant symbol features:
  [canonical_id, arity, is_predicate, is_skolem, occurrence_count, distinct_arg_arities]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyladr.core.clause import Clause, Literal
    from pyladr.core.symbol import SymbolTable
    from pyladr.core.term import Term

from pyladr.ml.invariant.canonicalization import (
    CanonicalMapping,
    canonicalize_clause_with_symbol_table,
)


class InvariantFeatureExtractor:
    """Extracts symbol-independent features for graph construction.

    Wraps a CanonicalMapping and produces feature vectors that encode
    structural role rather than symbol identity. Compatible with the
    existing 6-dimensional symbol feature format.

    Usage:
        extractor = InvariantFeatureExtractor(symbol_table)
        extractor.prepare(clause)  # builds canonical mapping
        features = extractor.symbol_features(symnum)  # invariant features
    """

    __slots__ = ("_symbol_table", "_mapping", "_occurrence_counts", "_arg_arity_sets")

    def __init__(self, symbol_table: object | None = None) -> None:
        self._symbol_table = symbol_table
        self._mapping = CanonicalMapping()
        self._occurrence_counts: dict[int, int] = {}
        self._arg_arity_sets: dict[int, set[int]] = {}

    def prepare(self, clause: Clause) -> None:
        """Build canonical mapping and occurrence stats in a single pass.

        Fuses canonicalization and occurrence counting to avoid traversing
        the clause structure twice.
        """
        mapping = self._mapping
        mapping.reset()
        occ = self._occurrence_counts
        occ.clear()
        aas = self._arg_arity_sets
        aas.clear()
        st = self._symbol_table

        for literal in clause.literals:
            atom = literal.atom
            if not atom.is_variable:
                mapping.get_or_assign_fast(atom.symnum, atom.arity, True, False)
                sn = atom.symnum
                occ[sn] = occ.get(sn, 0) + 1
                if sn not in aas:
                    aas[sn] = set()
                for arg in atom.args:
                    if not arg.is_variable:
                        aas[sn].add(arg.arity)
                for arg in atom.args:
                    self._prepare_term(arg, st, mapping, occ, aas)

    def _prepare_term(
        self,
        term: Term,
        st: object | None,
        mapping: CanonicalMapping,
        occ: dict[int, int],
        aas: dict[int, set[int]],
    ) -> None:
        """Single-pass: canonicalize + count occurrences for a term."""
        if term.is_variable:
            return

        sn = term.symnum
        is_skolem = False
        if st is not None:
            try:
                sym = st.get_symbol(sn)
                is_skolem = bool(sym.skolem)
            except (KeyError, IndexError, AttributeError):
                pass

        mapping.get_or_assign_fast(sn, term.arity, False, is_skolem)
        occ[sn] = occ.get(sn, 0) + 1

        if sn not in aas:
            aas[sn] = set()
        for arg in term.args:
            if not arg.is_variable:
                aas[sn].add(arg.arity)

        for arg in term.args:
            self._prepare_term(arg, st, mapping, occ, aas)

    def reset(self) -> None:
        """Reset state for reuse on a new clause."""
        self._mapping.reset()
        self._occurrence_counts.clear()
        self._arg_arity_sets.clear()

    def symbol_features(self, symnum: int) -> list[float]:
        """Extract invariant feature vector for a symbol.

        Returns 6 floats: [canonical_id, arity, is_predicate, is_skolem,
                           occurrence_count, distinct_arg_arities]

        These replace the original [symnum, arity, sym_type, is_skolem,
        kb_weight, occurrences] with structurally meaningful equivalents.
        """
        canonical_id = self._mapping.sym_to_canonical.get(symnum, 0)
        role = self._mapping.canonical_to_role.get(canonical_id)

        if role is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        occ = float(self._occurrence_counts.get(symnum, 0))
        distinct_arities = float(len(self._arg_arity_sets.get(symnum, set())))

        return [
            float(canonical_id),
            float(role.arity),
            float(role.is_predicate),
            float(role.is_skolem),
            occ,
            distinct_arities,
        ]

    def get_canonical_id(self, symnum: int) -> int:
        """Get the canonical ID for a symbol (for embedding table lookup)."""
        return self._mapping.sym_to_canonical.get(symnum, 0)


import hashlib as _hashlib


def _literal_invariant_key(lit: object) -> str:
    """Produce a canonical key for a single literal.

    Both variables and symbols are renumbered in first-occurrence order
    within this literal, making the key independent of original naming.
    """
    var_map: dict[int, int] = {}
    sym_map: dict[int, int] = {}
    vc = 0
    sc = 0
    parts: list[str] = []

    def _term_key(t: object) -> None:
        nonlocal vc, sc
        if t.is_variable:  # type: ignore[union-attr]
            vn = t.varnum  # type: ignore[union-attr]
            if vn not in var_map:
                var_map[vn] = vc
                vc += 1
            parts.append("v")
            parts.append(str(var_map[vn]))
            return
        sn = t.symnum  # type: ignore[union-attr]
        if sn not in sym_map:
            sym_map[sn] = sc
            sc += 1
        parts.append("s(")
        parts.append(str(sym_map[sn]))
        parts.append("/")
        parts.append(str(t.arity))  # type: ignore[union-attr]
        for a in t.args:  # type: ignore[union-attr]
            parts.append(",")
            _term_key(a)
        parts.append(")")

    parts.append("+" if lit.sign else "-")  # type: ignore[union-attr]
    _term_key(lit.atom)  # type: ignore[union-attr]
    return "".join(parts)


def invariant_clause_structural_hash(clause: Clause) -> str:
    """Compute a structural hash that is invariant to symbol renaming.

    Each literal is canonicalized independently (both variables and symbols
    renumbered in first-occurrence order) and the resulting keys are sorted,
    ensuring invariance under symbol renaming and literal reordering.
    """
    lit_keys = sorted(_literal_invariant_key(lit) for lit in clause.literals)
    raw = "|".join(lit_keys)
    return _hashlib.blake2b(raw.encode(), digest_size=16).hexdigest()
