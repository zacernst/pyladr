"""Nucleus pattern extraction and caching for hyperresolution guidance.

Extracts structural patterns from negative literals of nucleus clauses
during hyperresolution, caches them with LRU eviction, and provides
O(1) lookup by predicate symbol for nucleus unification penalty scoring.

A "nucleus pattern" captures the structure of a negative literal that
must be resolved during hyperresolution. Clauses whose positive literals
match many nucleus patterns are more likely to participate as satellites,
making them more valuable for selection.

Thread safety: uses ReadWriteLock from pyladr.threading_guide for
concurrent read access during clause selection with exclusive writes
during pattern insertion/eviction.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term
from pyladr.threading_guide import make_rw_lock


# ---------------------------------------------------------------------------
# Pattern data structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NucleusPattern:
    """A structural pattern extracted from a negative literal in a nucleus clause.

    Attributes
    ----------
    predicate_symbol:
        The symbol number of the predicate (negative of private_symbol).
        Used as the primary index key for O(1) lookup.
    arity:
        Arity of the predicate — quick filter before deeper matching.
    literal_hash:
        Structural hash of the literal (alpha-equivalent literals share
        the same hash). Used for deduplication in the cache.
    arg_complexity:
        Sum of symbol counts across all arguments. Higher complexity
        means more specific patterns that are harder to match.
    subsumption_template:
        Tuple of argument "shapes" for fast subsumption pre-filtering.
        Each entry is: 'v' for variable, 'c' for constant, 'f<arity>'
        for complex terms. E.g., ('v', 'f2', 'c') for P(x, f(a,b), c).
    source_clause_id:
        ID of the clause this pattern was extracted from.
    """

    predicate_symbol: int
    arity: int
    literal_hash: str
    arg_complexity: int
    subsumption_template: tuple[str, ...]
    source_clause_id: int


# ---------------------------------------------------------------------------
# Structural hashing with alpha-equivalence
# ---------------------------------------------------------------------------

def _alpha_normalize_literal(lit: Literal) -> str:
    """Produce a canonical string for a literal's structure.

    Variable identities are preserved within the literal but concrete
    variable numbers are normalized to first-occurrence order, so
    alpha-equivalent literals produce the same key.
    """
    var_map: dict[int, int] = {}
    counter = 0
    parts: list[str] = []

    def _term_key(t: Term) -> None:
        nonlocal counter
        if t.is_variable:
            vn = t.varnum
            if vn not in var_map:
                var_map[vn] = counter
                counter += 1
            parts.append("v")
            parts.append(str(var_map[vn]))
            return
        parts.append("f(")
        parts.append(str(t.private_symbol))
        for a in t.args:
            parts.append(",")
            _term_key(a)
        parts.append(")")

    parts.append("-" if not lit.sign else "+")
    _term_key(lit.atom)
    return "".join(parts)


def literal_structural_hash(lit: Literal) -> str:
    """Return a deterministic structural hash for a literal.

    Alpha-equivalent literals (differing only in variable numbering)
    produce the same hash. Uses BLAKE2b for fast, collision-resistant
    hashing.
    """
    raw = _alpha_normalize_literal(lit)
    return hashlib.blake2b(raw.encode(), digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# Pattern extraction
# ---------------------------------------------------------------------------

def _arg_shape(t: Term) -> str:
    """Classify a term argument for the subsumption template."""
    if t.is_variable:
        return "v"
    if t.is_constant:
        return "c"
    return f"f{t.arity}"


def extract_pattern(lit: Literal, source_clause_id: int) -> NucleusPattern | None:
    """Extract a nucleus pattern from a negative literal.

    Returns None if the literal is positive (not a nucleus pattern)
    or if the atom is a variable (degenerate case).
    """
    if lit.sign:
        return None  # Only negative literals form nucleus patterns

    atom = lit.atom
    if atom.is_variable:
        return None  # Variables as atoms are degenerate

    predicate_symbol = atom.symnum
    arity = atom.arity
    lit_hash = literal_structural_hash(lit)

    # Compute argument complexity: sum of symbol counts
    arg_complexity = sum(a.symbol_count for a in atom.args)

    # Build subsumption template
    subsumption_template = tuple(_arg_shape(a) for a in atom.args)

    return NucleusPattern(
        predicate_symbol=predicate_symbol,
        arity=arity,
        literal_hash=lit_hash,
        arg_complexity=arg_complexity,
        subsumption_template=subsumption_template,
        source_clause_id=source_clause_id,
    )


def extract_patterns(clause: Clause) -> list[NucleusPattern]:
    """Extract all nucleus patterns from a clause's negative literals.

    A clause serves as a nucleus in hyperresolution when it has negative
    literals. Each negative literal yields one pattern describing what
    kind of satellite is needed to resolve it.

    Args:
        clause: A clause that may serve as a nucleus.

    Returns:
        List of patterns, one per negative literal. Empty if the clause
        has no negative literals.
    """
    patterns: list[NucleusPattern] = []
    for lit in clause.literals:
        pat = extract_pattern(lit, clause.id)
        if pat is not None:
            patterns.append(pat)
    return patterns


# ---------------------------------------------------------------------------
# Cache statistics
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PatternCacheStats:
    """Live counters for cache monitoring."""

    hits: int = 0
    misses: int = 0
    insertions: int = 0
    evictions: int = 0
    dedup_skips: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_lookups(self) -> int:
        return self.hits + self.misses

    def snapshot(self) -> dict[str, int | float]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "insertions": self.insertions,
            "evictions": self.evictions,
            "dedup_skips": self.dedup_skips,
            "hit_rate": self.hit_rate,
            "total_lookups": self.total_lookups,
        }


# ---------------------------------------------------------------------------
# Nucleus pattern cache
# ---------------------------------------------------------------------------

class NucleusPatternCache:
    """Thread-safe LRU cache for nucleus patterns with predicate-indexed lookup.

    Provides two indexing dimensions:
    1. **By literal hash** — for deduplication (alpha-equivalent patterns
       are stored once).
    2. **By predicate symbol** — for O(1) retrieval of all patterns
       matching a given predicate, used by the nucleus penalty scorer.

    LRU eviction is based on insertion/access order of individual patterns
    (tracked via an OrderedDict keyed by literal_hash). When the cache
    exceeds ``max_size``, the least-recently-used patterns are evicted.

    Thread safety follows the same ReadWriteLock pattern as
    ``pyladr.ml.embeddings.cache.EmbeddingCache``: concurrent reads
    during clause selection, exclusive writes during insertion/eviction.

    Parameters
    ----------
    max_size:
        Maximum number of unique patterns (by literal_hash) in the cache.
    """

    __slots__ = (
        "_max_size",
        "_patterns",
        "_by_predicate",
        "_rw_lock",
        "_stats",
    )

    def __init__(self, max_size: int = 50_000) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._max_size = max_size
        # Primary store: literal_hash → NucleusPattern, ordered for LRU
        self._patterns: OrderedDict[str, NucleusPattern] = OrderedDict()
        # Secondary index: predicate_symbol → set of literal_hashes
        self._by_predicate: dict[int, set[str]] = {}
        self._rw_lock = make_rw_lock()
        self._stats = PatternCacheStats()

    # -- Public properties ---------------------------------------------------

    @property
    def stats(self) -> PatternCacheStats:
        return self._stats

    @property
    def max_size(self) -> int:
        return self._max_size

    # -- Size / containment --------------------------------------------------

    def __len__(self) -> int:
        with self._rw_lock.read_lock():
            return len(self._patterns)

    def __contains__(self, literal_hash: str) -> bool:
        with self._rw_lock.read_lock():
            return literal_hash in self._patterns

    # -- Insertion -----------------------------------------------------------

    def add(self, pattern: NucleusPattern) -> bool:
        """Insert a pattern into the cache.

        Returns True if the pattern was newly inserted, False if it was
        a duplicate (same literal_hash already cached). Duplicates are
        still touched for LRU freshness.
        """
        key = pattern.literal_hash
        with self._rw_lock.write_lock():
            if key in self._patterns:
                # Deduplicate: touch for LRU freshness
                self._patterns.move_to_end(key)
                self._stats.dedup_skips += 1
                return False

            # Insert new pattern
            self._patterns[key] = pattern
            self._patterns.move_to_end(key)

            # Update predicate index
            pred = pattern.predicate_symbol
            if pred not in self._by_predicate:
                self._by_predicate[pred] = set()
            self._by_predicate[pred].add(key)

            self._stats.insertions += 1

            # Evict if over capacity
            self._maybe_evict()

            return True

    def add_batch(self, patterns: Sequence[NucleusPattern]) -> int:
        """Insert multiple patterns. Returns count of newly inserted patterns."""
        if not patterns:
            return 0

        inserted = 0
        with self._rw_lock.write_lock():
            for pattern in patterns:
                key = pattern.literal_hash
                if key in self._patterns:
                    self._patterns.move_to_end(key)
                    self._stats.dedup_skips += 1
                    continue

                self._patterns[key] = pattern
                self._patterns.move_to_end(key)

                pred = pattern.predicate_symbol
                if pred not in self._by_predicate:
                    self._by_predicate[pred] = set()
                self._by_predicate[pred].add(key)

                self._stats.insertions += 1
                inserted += 1

            self._maybe_evict()

        return inserted

    # -- Lookup --------------------------------------------------------------

    def get_by_predicate(self, predicate_symbol: int) -> list[NucleusPattern]:
        """Return all cached patterns for the given predicate symbol.

        O(1) index lookup + O(k) pattern retrieval where k is the number
        of patterns for this predicate. Touches patterns for LRU freshness.
        """
        with self._rw_lock.read_lock():
            hashes = self._by_predicate.get(predicate_symbol)
            if not hashes:
                self._stats.misses += 1
                return []

            result: list[NucleusPattern] = []
            for h in hashes:
                pat = self._patterns.get(h)
                if pat is not None:
                    result.append(pat)

        if result:
            self._stats.hits += 1
            # Touch for LRU freshness (needs write lock)
            with self._rw_lock.write_lock():
                for pat in result:
                    key = pat.literal_hash
                    if key in self._patterns:
                        self._patterns.move_to_end(key)
        else:
            self._stats.misses += 1

        return result

    def get_by_hash(self, literal_hash: str) -> NucleusPattern | None:
        """Look up a specific pattern by its literal hash."""
        with self._rw_lock.read_lock():
            pat = self._patterns.get(literal_hash)
            if pat is None:
                self._stats.misses += 1
                return None

        self._stats.hits += 1
        # Touch for LRU
        with self._rw_lock.write_lock():
            if literal_hash in self._patterns:
                self._patterns.move_to_end(literal_hash)
        return pat

    def has_predicate(self, predicate_symbol: int) -> bool:
        """Check if any patterns exist for the given predicate. O(1)."""
        with self._rw_lock.read_lock():
            hashes = self._by_predicate.get(predicate_symbol)
            return bool(hashes)

    def predicate_count(self, predicate_symbol: int) -> int:
        """Return the number of cached patterns for a predicate. O(1)."""
        with self._rw_lock.read_lock():
            hashes = self._by_predicate.get(predicate_symbol)
            return len(hashes) if hashes else 0

    def all_predicates(self) -> list[int]:
        """Return all predicate symbols that have cached patterns."""
        with self._rw_lock.read_lock():
            return [p for p, hashes in self._by_predicate.items() if hashes]

    # -- Cache management ----------------------------------------------------

    def clear(self) -> int:
        """Remove all cached patterns. Returns count of evicted entries."""
        with self._rw_lock.write_lock():
            count = len(self._patterns)
            self._patterns.clear()
            self._by_predicate.clear()
            self._stats.evictions += count
            return count

    def remove_by_clause(self, clause_id: int) -> int:
        """Remove all patterns originating from a specific clause.

        Useful when a clause is back-subsumed or otherwise invalidated.
        Returns the number of patterns removed.
        """
        with self._rw_lock.write_lock():
            to_remove: list[str] = []
            for key, pat in self._patterns.items():
                if pat.source_clause_id == clause_id:
                    to_remove.append(key)

            for key in to_remove:
                self._remove_entry(key)

            return len(to_remove)

    # -- Internal helpers ----------------------------------------------------

    def _maybe_evict(self) -> None:
        """Evict LRU entries if over capacity. Caller must hold write lock."""
        overflow = len(self._patterns) - self._max_size
        if overflow > 0:
            self._evict_n(overflow)

    def _evict_n(self, count: int) -> None:
        """Evict the N oldest entries. Caller must hold write lock."""
        actual = min(count, len(self._patterns))
        for _ in range(actual):
            # Pop oldest (LRU)
            key, pat = self._patterns.popitem(last=False)
            # Clean up predicate index
            pred_set = self._by_predicate.get(pat.predicate_symbol)
            if pred_set is not None:
                pred_set.discard(key)
                if not pred_set:
                    del self._by_predicate[pat.predicate_symbol]
        self._stats.evictions += actual

    def _remove_entry(self, key: str) -> None:
        """Remove a single entry by hash. Caller must hold write lock."""
        pat = self._patterns.pop(key, None)
        if pat is not None:
            pred_set = self._by_predicate.get(pat.predicate_symbol)
            if pred_set is not None:
                pred_set.discard(key)
                if not pred_set:
                    del self._by_predicate[pat.predicate_symbol]
            self._stats.evictions += 1


# ---------------------------------------------------------------------------
# Convenience: extract and cache patterns from a clause in one step
# ---------------------------------------------------------------------------

def cache_nucleus_patterns(
    clause: Clause,
    cache: NucleusPatternCache,
) -> int:
    """Extract nucleus patterns from a clause and add them to the cache.

    Called when a clause with negative literals enters the usable set
    or is selected as a given clause with nucleus potential.

    Returns the number of newly cached (non-duplicate) patterns.
    """
    patterns = extract_patterns(clause)
    if not patterns:
        return 0
    return cache.add_batch(patterns)
