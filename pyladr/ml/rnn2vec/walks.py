"""Tree walk generation for RNN2Vec embedding training.

Generates sequences of node labels from term/clause trees using multiple
traversal strategies. Each walk captures different structural patterns:

- Depth-first: Captures parent-child and sibling relationships via pre-order traversal
- Breadth-first: Captures level-wise structural similarity
- Random walks: Stochastic traversals for diverse context sampling
- Path walks: Root-to-leaf paths capturing compositional structure

All walks produce sequences of string tokens representing node types,
suitable for RNN encoder training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term


class WalkType(Enum):
    """Tree walk strategy types."""

    DEPTH_FIRST = auto()
    BREADTH_FIRST = auto()
    RANDOM = auto()
    PATH = auto()


@dataclass(frozen=True, slots=True)
class WalkConfig:
    """Configuration for tree walk generation.

    Attributes:
        walk_types: Which walk strategies to use.
        num_random_walks: Number of random walks per tree (for RANDOM type).
        max_walk_length: Maximum tokens per walk (0 = unlimited).
        include_position: Whether to encode argument position in tokens.
        include_depth: Whether to encode tree depth in tokens.
        include_var_identity: Whether to encode De Bruijn-style variable identity.
            When True, variables are emitted as VAR_1, VAR_2, ... in order of
            first appearance within each walk. Two occurrences of the same
            variable within a walk receive the same index, capturing sharing
            across subtrees (e.g. both sides of an i/2 term). Alpha-equivalent
            terms receive identical token sequences.
        seed: Random seed for reproducibility.
    """

    walk_types: tuple[WalkType, ...] = (
        WalkType.DEPTH_FIRST,
        WalkType.BREADTH_FIRST,
        # WalkType.RANDOM,
        WalkType.PATH,
    )
    num_random_walks: int = 10
    max_walk_length: int = 0
    include_position: bool = False
    include_depth: bool = False
    include_path_length: bool = False
    include_var_identity: bool = False
    skip_predicate_wrapper: bool = False
    seed: int = 42


def _node_token(
    term: Term,
    position: int = -1,
    depth: int = -1,
    include_position: bool = False,
    include_depth: bool = False,
    var_id_map: dict[int, int] | None = None,
) -> str:
    """Convert a term node to a string token.

    Token format encodes the structural role:
    - Variables: "VAR" by default, or "VAR_<n>" when var_id_map is provided.
      The index n is assigned in order of first appearance within the walk,
      so repeated occurrences of the same variable receive the same index.
      This captures variable sharing across subtrees (De Bruijn-style identity).
    - Constants: "CONST:<symnum>"
    - Complex: "FUNC:<symnum>/<arity>"

    Optional suffixes for positional encoding:
    - "@<position>" if include_position is True
    - "#<depth>" if include_depth is True (applies to all node types including VAR)
    """
    if term.is_variable:
        if var_id_map is not None:
            sym = term.private_symbol  # >= 0 for variables; unique per variable
            if sym not in var_id_map:
                var_id_map[sym] = len(var_id_map) + 1
            token = f"VAR_{var_id_map[sym]}"
        else:
            token = "VAR"
    elif term.is_constant:
        token = f"CONST:{term.symnum}"
    else:
        token = f"FUNC:{term.symnum}/{term.arity}"

    if include_position and position >= 0:
        token += f"@{position}"
    if include_depth and depth >= 0:
        token += f"#{depth}"
    return token


def _literal_token(lit: Literal) -> str:
    """Token for a literal node (sign + predicate info)."""
    sign = "+" if lit.sign else "-"
    atom = lit.atom
    if atom.is_complex:
        return f"LIT:{sign}FUNC:{atom.symnum}/{atom.arity}"
    if atom.is_constant:
        return f"LIT:{sign}CONST:{atom.symnum}"
    return f"LIT:{sign}VAR"


class TreeWalker:
    """Generates tree walks from Term and Clause structures.

    Produces sequences of string tokens by traversing formula trees
    using configurable walk strategies. All methods are deterministic
    given the same seed.
    """

    def __init__(self, config: WalkConfig | None = None) -> None:
        self.config = config or WalkConfig()
        self._rng = random.Random(self.config.seed)

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset the random number generator."""
        self._rng = random.Random(seed if seed is not None else self.config.seed)

    # ── Public API ─────────────────────────────────────────────────────

    def walks_from_term(self, term: Term) -> list[list[str]]:
        """Generate all configured walk types from a single term tree."""
        walks: list[list[str]] = []
        for wt in self.config.walk_types:
            if wt == WalkType.DEPTH_FIRST:
                walks.append(self._depth_first(term))
            elif wt == WalkType.BREADTH_FIRST:
                walks.append(self._breadth_first(term))
            elif wt == WalkType.RANDOM:
                for _ in range(self.config.num_random_walks):
                    walks.append(self._random_walk(term))
            elif wt == WalkType.PATH:
                walks.extend(self._path_walks(term))
        return walks

    def walks_from_literal(self, lit: Literal) -> list[list[str]]:
        """Generate walks from a literal, prepending the literal token."""
        lit_tok = _literal_token(lit)
        term_walks = self.walks_from_term(lit.atom)
        return [[lit_tok] + walk for walk in term_walks]

    def walks_from_clause(self, clause: Clause) -> list[list[str]]:
        """Generate walks from all literals in a clause.

        Each walk is prefixed with a CLAUSE token followed by the
        literal-level walks.

        When skip_predicate_wrapper is enabled, walks are generated from
        predicate arguments directly instead of the predicate atom, with a
        sign token prepended. This removes the predicate wrapper from the
        walk vocabulary, focusing embeddings on term structure. Falls back
        to normal behavior for propositional atoms (arity 0).
        """
        all_walks: list[list[str]] = []
        clause_tok = f"CLAUSE:{clause.num_literals}"

        if self.config.skip_predicate_wrapper:
            for lit in clause.literals:
                sign_tok = "SIGN:+" if lit.sign else "SIGN:-"
                if lit.atom.arity == 0:
                    # Propositional atom — fall back to normal literal walk
                    lit_walks = self.walks_from_literal(lit)
                    for walk in lit_walks:
                        all_walks.append([clause_tok] + walk)
                else:
                    # Generate walks from each argument of the predicate
                    for arg in lit.atom.args:
                        arg_walks = self.walks_from_term(arg)
                        for walk in arg_walks:
                            all_walks.append([clause_tok, sign_tok] + walk)
        else:
            for lit in clause.literals:
                lit_walks = self.walks_from_literal(lit)
                for walk in lit_walks:
                    all_walks.append([clause_tok] + walk)

        return all_walks

    def walks_from_clauses(self, clauses: Sequence[Clause]) -> list[list[str]]:
        """Generate walks from multiple clauses."""
        all_walks: list[list[str]] = []
        for clause in clauses:
            all_walks.extend(self.walks_from_clause(clause))
        return all_walks

    # ── Walk strategies ────────────────────────────────────────────────

    def _depth_first(
        self,
        term: Term,
        depth: int = 0,
        position: int = 0,
        var_id_map: dict[int, int] | None = None,
    ) -> list[str]:
        """Pre-order depth-first traversal producing token sequence.

        var_id_map is created fresh at the top-level call when
        include_var_identity is True, then threaded through recursion so
        all nodes in the same walk share the same variable numbering.
        """
        if var_id_map is None and self.config.include_var_identity:
            var_id_map = {}
        tok = _node_token(
            term,
            position,
            depth,
            self.config.include_position,
            self.config.include_depth,
            var_id_map,
        )
        result = [tok]
        if 0 < self.config.max_walk_length <= len(result):
            return result
        for i, arg in enumerate(term.args):
            child_tokens = self._depth_first(arg, depth + 1, i, var_id_map)
            result.extend(child_tokens)
            if 0 < self.config.max_walk_length <= len(result):
                return result[: self.config.max_walk_length]
        return result

    def _breadth_first(self, term: Term) -> list[str]:
        """Level-order breadth-first traversal producing token sequence."""
        var_id_map: dict[int, int] | None = {} if self.config.include_var_identity else None
        result: list[str] = []
        # Queue entries: (term, depth, position)
        queue: list[tuple[Term, int, int]] = [(term, 0, 0)]
        head = 0
        while head < len(queue):
            node, depth, position = queue[head]
            head += 1
            tok = _node_token(
                node,
                position,
                depth,
                self.config.include_position,
                self.config.include_depth,
                var_id_map,
            )
            result.append(tok)
            if 0 < self.config.max_walk_length <= len(result):
                return result
            for i, arg in enumerate(node.args):
                queue.append((arg, depth + 1, i))
        return result

    def _random_walk(self, term: Term) -> list[str]:
        """Random walk from root, choosing a random child at each step."""
        var_id_map: dict[int, int] | None = {} if self.config.include_var_identity else None
        result: list[str] = []
        current = term
        depth = 0
        position = 0
        while True:
            tok = _node_token(
                current,
                position,
                depth,
                self.config.include_position,
                self.config.include_depth,
                var_id_map,
            )
            result.append(tok)
            if 0 < self.config.max_walk_length <= len(result):
                return result
            if current.arity == 0:
                break
            position = self._rng.randrange(current.arity)
            current = current.args[position]
            depth += 1
        return result

    def _path_walks(self, term: Term) -> list[list[str]]:
        """Generate all root-to-leaf paths as walks.

        Each path captures the full compositional structure from
        root to a leaf node. When include_path_length is enabled,
        a ``PATHLEN:<n>`` token is prepended to each path.

        When include_var_identity is True, each path carries its own
        var_id_map that is copied at branch points, so variable indices
        within a path reflect order-of-first-appearance along that
        specific root-to-leaf path.
        """
        paths: list[list[str]] = []
        init_map: dict[int, int] | None = {} if self.config.include_var_identity else None
        # Stack entries: (term, current_path, depth, position, var_id_map)
        stack: list[tuple[Term, list[str], int, int, dict[int, int] | None]] = [
            (term, [], 0, 0, init_map)
        ]
        while stack:
            node, path, depth, position, var_id_map = stack.pop()
            tok = _node_token(
                node,
                position,
                depth,
                self.config.include_position,
                self.config.include_depth,
                var_id_map,
            )
            new_path = path + [tok]
            if node.arity == 0:
                # Leaf node - emit the complete path
                if self.config.max_walk_length > 0:
                    new_path = new_path[: self.config.max_walk_length]
                if self.config.include_path_length:
                    new_path = [f"PATHLEN:{len(new_path)}"] + new_path
                paths.append(new_path)
            else:
                # Push children in reverse order for consistent left-to-right.
                # Copy var_id_map at each branch so sibling paths have
                # independent variable numbering from the branch point onward.
                for i in range(node.arity - 1, -1, -1):
                    child_map = dict(var_id_map) if var_id_map is not None else None
                    stack.append((node.args[i], new_path, depth + 1, i, child_map))
        return (
            paths
            if paths
            else [
                [
                    _node_token(
                        term,
                        0,
                        0,
                        self.config.include_position,
                        self.config.include_depth,
                        {} if self.config.include_var_identity else None,
                    )
                ]
            ]
        )
