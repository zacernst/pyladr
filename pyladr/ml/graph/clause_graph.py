"""Graph construction utilities for converting PyLADR clauses to PyTorch Geometric graphs.

Converts Clause/Literal/Term structures into heterogeneous graphs suitable for
graph neural network processing. The graph captures the full logical structure:
clause → literal → term hierarchy, symbol identity, and variable sharing.

No modifications to core PyLADR data structures are made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import torch
from torch_geometric.data import HeteroData

from pyladr.inference.paramodulation import is_eq_atom

if TYPE_CHECKING:
    from pyladr.core.clause import Clause, Literal
    from pyladr.core.symbol import SymbolTable
    from pyladr.core.term import Term


# ── Node & edge type enumerations ──────────────────────────────────────────


class NodeType(str, Enum):
    """Node types in the heterogeneous clause graph."""

    CLAUSE = "clause"
    LITERAL = "literal"
    TERM = "term"
    SYMBOL = "symbol"
    VARIABLE = "variable"


class EdgeType(str, Enum):
    """Edge types connecting nodes in the clause graph."""

    CONTAINS_LITERAL = "contains_literal"  # clause → literal
    HAS_ATOM = "has_atom"  # literal → term
    HAS_ARG = "has_arg"  # term → term (function arguments)
    SYMBOL_OF = "symbol_of"  # term → symbol
    VAR_OCCURRENCE = "var_occurrence"  # variable → term
    SHARED_VARIABLE = "shared_variable"  # variable ↔ variable (across literals)


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ClauseGraphConfig:
    """Configuration for clause-to-graph conversion.

    Attributes:
        max_term_depth: Maximum recursion depth for term traversal.
            Terms deeper than this are truncated. 0 means unlimited.
        include_variable_sharing: Whether to add SHARED_VARIABLE edges
            between variable occurrences across different literals.
        include_symbol_features: Whether to extract symbol metadata
            (type, arity, skolem, weight) as node features.
        symbol_vocab_size: Maximum number of distinct symbols to track.
            Symbols beyond this are mapped to an <UNK> embedding.
    """

    max_term_depth: int = 0
    include_variable_sharing: bool = True
    include_symbol_features: bool = True
    symbol_vocab_size: int = 10000


_DEFAULT_CONFIG = ClauseGraphConfig()


# ── Internal graph builder ─────────────────────────────────────────────────


class _GraphBuilder:
    """Accumulates nodes and edges while traversing a clause structure.

    Tracks node counts per type, feature tensors, and edge index lists.
    Each node is assigned a unique integer index within its type.
    """

    def __init__(self, config: ClauseGraphConfig, symbol_table: SymbolTable | None):
        self.config = config
        self.symbol_table = symbol_table

        # Node counters
        self._node_counts: dict[str, int] = {nt.value: 0 for nt in NodeType}

        # Feature accumulation (list of feature vectors per node type)
        self._features: dict[str, list[list[float]]] = {
            nt.value: [] for nt in NodeType
        }

        # Edge index accumulation: (src_type, edge_type, dst_type) → ([src], [dst])
        self._edges: dict[tuple[str, str, str], tuple[list[int], list[int]]] = {}

        # De-duplication maps
        self._symbol_nodes: dict[int, int] = {}  # symnum → node index
        self._variable_nodes: dict[int, int] = {}  # varnum → node index

        # Track variable occurrences per literal for shared-variable edges
        # Maps varnum → list of (literal_idx, term_node_idx)
        self._var_occurrences: dict[int, list[tuple[int, int]]] = {}

    def reset(self) -> None:
        """Reset builder state for reuse on a new clause.

        Reuses the existing dict/list objects by clearing them in-place,
        which avoids dict allocation overhead in tight batch loops.
        """
        for k in self._node_counts:
            self._node_counts[k] = 0
        for k in self._features:
            self._features[k].clear()
        self._edges.clear()
        self._symbol_nodes.clear()
        self._variable_nodes.clear()
        self._var_occurrences.clear()

    # ── Node creation ──────────────────────────────────────────────────

    def _add_node(self, node_type: str, features: list[float]) -> int:
        idx = self._node_counts[node_type]
        self._node_counts[node_type] += 1
        self._features[node_type].append(features)
        return idx

    def _add_edge(
        self,
        src_type: str,
        edge_type: str,
        dst_type: str,
        src_idx: int,
        dst_idx: int,
    ) -> None:
        key = (src_type, edge_type, dst_type)
        if key not in self._edges:
            self._edges[key] = ([], [])
        self._edges[key][0].append(src_idx)
        self._edges[key][1].append(dst_idx)

    # ── Clause processing ──────────────────────────────────────────────

    def add_clause(self, clause: Clause) -> int:
        """Add a clause node and recursively process its literals."""
        features = self._clause_features(clause)
        clause_idx = self._add_node(NodeType.CLAUSE.value, features)

        for lit_pos, literal in enumerate(clause.literals):
            lit_idx = self._add_literal(literal, lit_pos)
            self._add_edge(
                NodeType.CLAUSE.value,
                EdgeType.CONTAINS_LITERAL.value,
                NodeType.LITERAL.value,
                clause_idx,
                lit_idx,
            )

        # Add shared-variable edges if configured
        if self.config.include_variable_sharing:
            self._add_shared_variable_edges()

        return clause_idx

    def _add_literal(self, literal: Literal, position: int) -> int:
        """Add a literal node and process its atom term."""
        features = self._literal_features(literal, position)
        lit_idx = self._add_node(NodeType.LITERAL.value, features)

        term_idx = self._add_term(literal.atom, depth=0, literal_idx=lit_idx)
        self._add_edge(
            NodeType.LITERAL.value,
            EdgeType.HAS_ATOM.value,
            NodeType.TERM.value,
            lit_idx,
            term_idx,
        )

        return lit_idx

    def _add_term(self, term: Term, depth: int, literal_idx: int) -> int:
        """Recursively add a term node and its substructure."""
        if self.config.max_term_depth > 0 and depth >= self.config.max_term_depth:
            # Truncated term — add as leaf with truncation marker
            features = self._term_features(term, depth, truncated=True)
            return self._add_node(NodeType.TERM.value, features)

        features = self._term_features(term, depth)
        term_idx = self._add_node(NodeType.TERM.value, features)

        if term.is_variable:
            var_idx = self._get_or_create_variable(term.varnum)
            self._add_edge(
                NodeType.VARIABLE.value,
                EdgeType.VAR_OCCURRENCE.value,
                NodeType.TERM.value,
                var_idx,
                term_idx,
            )
            # Track for shared-variable edges
            if self.config.include_variable_sharing:
                if term.varnum not in self._var_occurrences:
                    self._var_occurrences[term.varnum] = []
                self._var_occurrences[term.varnum].append(
                    (literal_idx, term_idx)
                )
        else:
            # Constant or complex term — link to symbol
            sym_idx = self._get_or_create_symbol(term.symnum)
            self._add_edge(
                NodeType.TERM.value,
                EdgeType.SYMBOL_OF.value,
                NodeType.SYMBOL.value,
                term_idx,
                sym_idx,
            )

            # Recurse into arguments
            for arg_pos in range(term.arity):
                arg_idx = self._add_term(
                    term.args[arg_pos], depth + 1, literal_idx
                )
                self._add_edge(
                    NodeType.TERM.value,
                    EdgeType.HAS_ARG.value,
                    NodeType.TERM.value,
                    term_idx,
                    arg_idx,
                )

        return term_idx

    def _get_or_create_variable(self, varnum: int) -> int:
        """Get existing variable node or create a new one."""
        if varnum in self._variable_nodes:
            return self._variable_nodes[varnum]
        features = [float(varnum)]
        idx = self._add_node(NodeType.VARIABLE.value, features)
        self._variable_nodes[varnum] = idx
        return idx

    def _get_or_create_symbol(self, symnum: int) -> int:
        """Get existing symbol node or create a new one."""
        if symnum in self._symbol_nodes:
            return self._symbol_nodes[symnum]

        features = self._symbol_features(symnum)
        idx = self._add_node(NodeType.SYMBOL.value, features)
        self._symbol_nodes[symnum] = idx
        return idx

    def _add_shared_variable_edges(self) -> None:
        """Connect variable nodes that appear in different literals."""
        for varnum, occurrences in self._var_occurrences.items():
            if len(occurrences) < 2:
                continue
            # Get unique literal indices where this variable appears
            literal_indices = set()
            for lit_idx, _ in occurrences:
                literal_indices.add(lit_idx)
            if len(literal_indices) < 2:
                continue
            # Add bidirectional shared-variable edges
            var_idx = self._variable_nodes[varnum]
            self._add_edge(
                NodeType.VARIABLE.value,
                EdgeType.SHARED_VARIABLE.value,
                NodeType.VARIABLE.value,
                var_idx,
                var_idx,
            )

    # ── Feature extraction ─────────────────────────────────────────────

    def _clause_features(self, clause: Clause) -> list[float]:
        """Extract feature vector for a clause node.

        Features: [num_literals, is_unit, is_horn, is_positive, is_negative,
                   is_ground, weight]
        """
        return [
            float(clause.num_literals),
            float(clause.is_unit),
            float(clause.is_horn),
            float(clause.is_positive),
            float(clause.is_negative),
            float(clause.is_ground),
            float(clause.weight),
        ]

    def _literal_features(self, literal: Literal, position: int) -> list[float]:
        """Extract feature vector for a literal node.

        Features: [sign, position, is_eq_literal]
        """
        is_eq = (
            is_eq_atom(literal.atom, self.symbol_table)
            if self.symbol_table is not None
            else literal.is_eq_literal
        )
        return [
            float(literal.sign),
            float(position),
            float(is_eq),
        ]

    def _term_features(
        self, term: Term, depth: int, truncated: bool = False
    ) -> list[float]:
        """Extract feature vector for a term node.

        Features: [is_variable, is_constant, is_complex, arity, depth,
                   symbol_count, is_ground, truncated]
        """
        return [
            float(term.is_variable),
            float(term.is_constant),
            float(term.is_complex),
            float(term.arity),
            float(depth),
            float(term.symbol_count),
            float(term.is_ground),
            float(truncated),
        ]

    def _symbol_features(self, symnum: int) -> list[float]:
        """Extract feature vector for a symbol node.

        Features: [symnum, arity, sym_type, is_skolem, kb_weight, occurrences]
        If no symbol_table is provided, uses minimal features.
        """
        if self.symbol_table is None or not self.config.include_symbol_features:
            capped = min(symnum, self.config.symbol_vocab_size - 1)
            return [float(capped), 0.0, 0.0, 0.0, 1.0, 0.0]

        try:
            sym = self.symbol_table.get_symbol(symnum)
        except (KeyError, IndexError):
            capped = min(symnum, self.config.symbol_vocab_size - 1)
            return [float(capped), 0.0, 0.0, 0.0, 1.0, 0.0]

        capped = min(symnum, self.config.symbol_vocab_size - 1)
        return [
            float(capped),
            float(sym.arity),
            float(sym.sym_type),
            float(sym.skolem),
            float(sym.kb_weight),
            float(sym.occurrences),
        ]

    # ── Build final HeteroData ─────────────────────────────────────────

    def build(self) -> HeteroData:
        """Construct a PyTorch Geometric HeteroData from accumulated graph."""
        data = HeteroData()

        # Add node features
        for node_type in NodeType:
            nt = node_type.value
            if self._node_counts[nt] > 0:
                data[nt].x = torch.tensor(
                    self._features[nt], dtype=torch.float32
                )
                data[nt].num_nodes = self._node_counts[nt]
            else:
                # Empty node type — still register it with zero nodes
                data[nt].num_nodes = 0

        # Add edge indices
        for (src_type, edge_type, dst_type), (src_list, dst_list) in self._edges.items():
            edge_index = torch.tensor(
                [src_list, dst_list], dtype=torch.long
            )
            data[src_type, edge_type, dst_type].edge_index = edge_index

        return data


# ── Public API ─────────────────────────────────────────────────────────────


def clause_to_heterograph(
    clause: Clause,
    symbol_table: SymbolTable | None = None,
    config: ClauseGraphConfig | None = None,
) -> HeteroData:
    """Convert a single Clause to a PyTorch Geometric HeteroData graph.

    Args:
        clause: The PyLADR Clause to convert.
        symbol_table: Optional SymbolTable for extracting symbol metadata.
        config: Graph construction configuration. Uses defaults if None.

    Returns:
        A HeteroData object with heterogeneous node types and edge types
        representing the clause's logical structure.
    """
    cfg = config or _DEFAULT_CONFIG
    builder = _GraphBuilder(cfg, symbol_table)
    builder.add_clause(clause)
    return builder.build()


def batch_clauses_to_heterograph(
    clauses: list[Clause],
    symbol_table: SymbolTable | None = None,
    config: ClauseGraphConfig | None = None,
) -> list[HeteroData]:
    """Convert multiple clauses to individual HeteroData graphs.

    Each clause gets its own graph. Use PyG's Batch.from_data_list()
    to combine them into a batched graph for efficient GNN processing.

    Args:
        clauses: List of PyLADR Clauses.
        symbol_table: Optional SymbolTable for symbol metadata.
        config: Graph construction configuration.

    Returns:
        List of HeteroData graphs, one per clause.
    """
    cfg = config or _DEFAULT_CONFIG
    builder = _GraphBuilder(cfg, symbol_table)
    results = []
    for clause in clauses:
        builder.reset()
        builder.add_clause(clause)
        results.append(builder.build())
    return results
