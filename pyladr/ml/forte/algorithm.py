"""FORTE core algorithm: deterministic clause → vector embedding.

The FORTE algorithm extracts structural features from first-order logic
clauses and projects them into a fixed-dimensional vector space using
sparse feature hashing (SimHash). This achieves microsecond-level
performance by avoiding neural network forward passes entirely.

Algorithm overview:
  1. Extract 24 structural features from the clause (single traversal)
  2. Extract symbol distribution features via hash bucketing
  3. Extract variable binding pattern features
  4. Project all features into 64-dim space via sparse random hashing
  5. L2-normalize the result

Each feature hashes to K random dimensions (default K=6) with random
signs, providing O(features × K) projection instead of O(features × dim).
This achieves ~15-20 μs per clause in pure Python.

The output is fully deterministic: identical clauses always produce
identical embeddings, enabling structural caching.

Thread-safety: ForteAlgorithm is immutable after construction and safe
for concurrent use from multiple threads without synchronization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyladr.core.clause import Clause

# ── Configuration ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ForteConfig:
    """Configuration for FORTE embedding generation.

    All fields are immutable after construction for thread safety.
    """

    embedding_dim: int = 64
    symbol_buckets: int = 16
    arity_buckets: int = 8
    depth_buckets: int = 8
    hash_k: int = 6  # each feature hashes to K random dimensions
    seed: int = 42


# ── Precomputed sparse hash tables ──────────────────────────────────────────

def _make_sparse_hash(
    num_features: int, embedding_dim: int, k: int, seed: int,
) -> tuple[list[int], list[float]]:
    """Generate sparse hash mapping: each feature → K (dimension, sign) pairs.

    Uses a LCG for deterministic cross-platform reproducibility.

    Returns:
        hash_dims: flat list of length num_features * k, dimension indices
        hash_signs: flat list of length num_features * k, signs (+1.0/-1.0)
    """
    a, c, m = 1664525, 1013904223, 2**32
    state = seed & 0xFFFFFFFF

    hash_dims: list[int] = []
    hash_signs: list[float] = []
    for _ in range(num_features):
        for _ in range(k):
            state = (a * state + c) & (m - 1)
            hash_dims.append(state % embedding_dim)
            state = (a * state + c) & (m - 1)
            hash_signs.append(1.0 if (state >> 31) else -1.0)
    return hash_dims, hash_signs


def _make_projection_matrix(
    num_features: int, embedding_dim: int, seed: int,
) -> list[list[float]]:
    """Generate a deterministic random projection matrix.

    Uses a simple LCG (linear congruential generator) for reproducibility
    across platforms without depending on numpy or random module state.

    Returns matrix of shape (num_features, embedding_dim) with entries
    drawn from {-1/sqrt(embedding_dim), +1/sqrt(embedding_dim)} for
    variance-preserving projection (sparse random projection).
    """
    scale = 1.0 / math.sqrt(embedding_dim)
    a, c, m = 1664525, 1013904223, 2**32

    state = seed & 0xFFFFFFFF
    matrix: list[list[float]] = []
    for _ in range(num_features):
        row: list[float] = []
        for _ in range(embedding_dim):
            state = (a * state + c) & (m - 1)
            row.append(scale if (state >> 31) else -scale)
        matrix.append(row)
    return matrix


# ── Feature extraction constants ─────────────────────────────────────────────

# Base structural features extracted from each clause.
# Indices 0–15: clause-level structural features
# Indices 16–23: derived/distributional features
# Total: 24 base features (before symbol/arity/depth buckets)
_BASE_FEATURES = 24


# ── Core Algorithm ───────────────────────────────────────────────────────────


class ForteAlgorithm:
    """FORTE: Feature-Oriented Representation for Theorem-proving Embeddings.

    Generates deterministic fixed-dimensional vector embeddings from
    first-order logic clauses using sparse feature hashing. The algorithm
    performs a single traversal of the clause structure to extract
    structural, distributional, and relational features, then projects
    them into the embedding space via sparse random hashing.

    Thread-safe: this object is effectively immutable after __init__
    and can be shared across threads without synchronization.

    Performance target: 15-25 μs per clause (pure Python).
    """

    __slots__ = (
        "_config", "_num_features", "_embedding_dim",
        "_hash_dims", "_hash_signs", "_hash_k",
        "_sym_buckets", "_arity_buckets", "_depth_buckets",
        "_sym_offset", "_arity_offset", "_depth_offset",
    )

    def __init__(self, config: ForteConfig | None = None) -> None:
        if config is None:
            config = ForteConfig()
        self._config = config
        self._embedding_dim = config.embedding_dim
        self._hash_k = config.hash_k
        self._sym_buckets = config.symbol_buckets
        self._arity_buckets = config.arity_buckets
        self._depth_buckets = config.depth_buckets
        self._sym_offset = _BASE_FEATURES
        self._arity_offset = _BASE_FEATURES + config.symbol_buckets
        self._depth_offset = _BASE_FEATURES + config.symbol_buckets + config.arity_buckets
        self._num_features = (
            _BASE_FEATURES
            + config.symbol_buckets
            + config.arity_buckets
            + config.depth_buckets
        )
        self._hash_dims, self._hash_signs = _make_sparse_hash(
            self._num_features, config.embedding_dim, config.hash_k, config.seed,
        )

    @property
    def config(self) -> ForteConfig:
        return self._config

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed_clause(self, clause: Clause) -> list[float]:
        """Generate a 64-dimensional embedding for a clause.

        Args:
            clause: A PyLADR Clause object with immutable literals.

        Returns:
            A list of floats of length embedding_dim, L2-normalized.
            Returns a zero vector for empty clauses.
        """
        features = self._extract_features(clause)
        return self._project_and_normalize(features)

    def embed_clauses_batch(self, clauses: list[Clause]) -> list[list[float]]:
        """Batch embedding for multiple clauses.

        Args:
            clauses: List of Clause objects.

        Returns:
            List of embedding vectors, one per clause.
        """
        extract = self._extract_features
        project = self._project_and_normalize
        return [project(extract(c)) for c in clauses]

    def _extract_features(self, clause: Clause) -> list[float]:
        """Extract structural features from a clause in a single pass.

        Optimized for minimal Python overhead: avoids property lookups,
        uses local variable caching, and minimizes allocations.
        """
        num_features = self._num_features
        features: list[float] = [0.0] * num_features

        literals = clause.literals
        num_lits = len(literals)

        if num_lits == 0:
            return features

        # Cache offsets locally
        sym_offset = self._sym_offset
        sym_buckets = self._sym_buckets
        arity_offset = self._arity_offset
        arity_buckets_m1 = self._arity_buckets - 1
        depth_offset = self._depth_offset
        depth_buckets_m1 = self._depth_buckets - 1

        # Clause-level accumulators
        num_pos = 0
        is_ground = True
        has_equality = False
        num_eq_lits = 0
        total_sym_count = 0
        num_vars_total = 0
        num_constants = 0
        num_complex = 0
        max_depth = 0
        max_arity = 0
        max_var_idx = -1
        ground_lits = 0
        lit_size_sum_sq = 0

        distinct_symbols: set[int] = set()
        distinct_vars: set[int] = set()

        # Variable occurrence tracking for shared variable detection
        var_lit_count: dict[int, int] | None = None

        for lit in literals:
            if lit.sign:
                num_pos += 1

            atom = lit.atom
            atom_ps = atom.private_symbol
            atom_arity = atom.arity

            if atom_ps < 0 and atom_arity == 2:
                has_equality = True
                num_eq_lits += 1

            atom_sc = atom._symbol_count
            total_sym_count += atom_sc
            lit_size_sum_sq += atom_sc * atom_sc

            # Iterative term traversal with parallel stacks
            term_stack: list[object] = [atom]
            depth_stack: list[int] = [0]
            lit_is_ground = True
            lit_vars: set[int] | None = None

            while term_stack:
                term = term_stack.pop()
                d = depth_stack.pop()
                ps = term.private_symbol  # type: ignore[union-attr]
                ar = term.arity  # type: ignore[union-attr]

                if ps >= 0:
                    num_vars_total += 1
                    lit_is_ground = False
                    distinct_vars.add(ps)
                    if ps > max_var_idx:
                        max_var_idx = ps
                    if lit_vars is None:
                        lit_vars = {ps}
                    else:
                        lit_vars.add(ps)
                else:
                    symnum = -ps
                    distinct_symbols.add(symnum)
                    features[sym_offset + (symnum % sym_buckets)] += 1.0

                    if ar == 0:
                        num_constants += 1
                    else:
                        num_complex += 1
                        if ar > max_arity:
                            max_arity = ar
                        features[arity_offset + min(ar, arity_buckets_m1)] += 1.0

                    features[depth_offset + min(d, depth_buckets_m1)] += 1.0

                    args = term.args  # type: ignore[union-attr]
                    d1 = d + 1
                    for i in range(ar - 1, -1, -1):
                        term_stack.append(args[i])
                        depth_stack.append(d1)

                if d > max_depth:
                    max_depth = d

            if lit_is_ground:
                ground_lits += 1
            else:
                is_ground = False

            if lit_vars is not None:
                if var_lit_count is None:
                    var_lit_count = {}
                for v in lit_vars:
                    var_lit_count[v] = var_lit_count.get(v, 0) + 1

        # Shared variables
        num_shared_vars = 0
        if var_lit_count is not None:
            for cnt in var_lit_count.values():
                if cnt > 1:
                    num_shared_vars += 1

        # Derived values
        num_neg = num_lits - num_pos
        avg_size = total_sym_count / num_lits
        variance = (lit_size_sum_sq / num_lits) - avg_size * avg_size
        total_nodes = num_vars_total + total_sym_count

        # Pack base features — 24 contiguous slots
        # [0]  num_literals        [1]  num_positive         [2]  num_negative
        # [3]  is_unit             [4]  is_horn              [5]  is_ground
        # [6]  clause_weight       [7]  max_depth            [8]  total_symbols
        # [9]  total_variables     [10] num_constants        [11] num_complex_terms
        # [12] num_eq_lits         [13] max_arity            [14] avg_literal_size
        # [15] variable_ratio      [16] distinct_symbols     [17] distinct_variables
        # [18] max_variable_index  [19] has_equality         [20] shared_variables
        # [21] literal_size_var    [22] positive_ratio       [23] ground_literal_ratio
        features[0] = float(num_lits)
        features[1] = float(num_pos)
        features[2] = float(num_neg)
        features[3] = 1.0 if num_lits == 1 else 0.0
        features[4] = 1.0 if num_pos <= 1 else 0.0
        features[5] = 1.0 if is_ground else 0.0
        features[6] = clause.weight
        features[7] = float(max_depth)
        features[8] = float(total_sym_count)
        features[9] = float(num_vars_total)
        features[10] = float(num_constants)
        features[11] = float(num_complex)
        features[12] = float(num_eq_lits)  # count of equational (binary predicate) literals
        features[13] = float(max_arity)
        features[14] = avg_size
        features[15] = float(num_vars_total) / total_nodes if total_nodes > 0 else 0.0
        features[16] = float(len(distinct_symbols))
        features[17] = float(len(distinct_vars))
        features[18] = float(max_var_idx) if max_var_idx >= 0 else 0.0
        features[19] = 1.0 if has_equality else 0.0
        features[20] = float(num_shared_vars)
        features[21] = variance if variance > 0.0 else 0.0
        features[22] = float(num_pos) / num_lits
        features[23] = float(ground_lits) / num_lits

        return features

    def _project_and_normalize(self, features: list[float]) -> list[float]:
        """Project features to embedding space via sparse hashing + L2 normalize.

        Each non-zero feature contributes to K random dimensions with random
        signs. This is O(non_zero_features × K) instead of O(features × dim).
        """
        dim = self._embedding_dim
        k = self._hash_k
        hash_dims = self._hash_dims
        hash_signs = self._hash_signs
        result: list[float] = [0.0] * dim

        for i, f in enumerate(features):
            if f != 0.0:
                base = i * k
                for ki in range(k):
                    idx = base + ki
                    result[hash_dims[idx]] += f * hash_signs[idx]

        # L2 normalization
        norm_sq = 0.0
        for v in result:
            norm_sq += v * v
        if norm_sq > 0.0:
            inv_norm = 1.0 / math.sqrt(norm_sq)
            for j in range(dim):
                result[j] *= inv_norm

        return result
