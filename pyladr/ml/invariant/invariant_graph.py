"""Invariant graph construction for symbol-independent clause embeddings.

Extends the existing clause graph builder with property-invariant features.
Produces HeteroData graphs where symbol node features encode structural role
(arity, predicate/function, Skolem status) rather than raw symbol identity.

This is a drop-in replacement for clause_to_heterograph / batch_clauses_to_heterograph
when symbol-independent embeddings are desired. The graph topology is identical —
only the symbol features differ.

Design: Creates an InvariantGraphBuilder alongside (not replacing) _GraphBuilder.
The builder delegates clause/literal/term traversal to the same logic but
overrides _symbol_features and _get_or_create_symbol to use canonical IDs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch_geometric.data import HeteroData

from pyladr.ml.graph.clause_graph import (
    ClauseGraphConfig,
    EdgeType,
    NodeType,
    _GraphBuilder,
)
from pyladr.ml.invariant.invariant_features import InvariantFeatureExtractor

if TYPE_CHECKING:
    from pyladr.core.clause import Clause
    from pyladr.core.symbol import SymbolTable


class _InvariantGraphBuilder(_GraphBuilder):
    """Graph builder that uses invariant symbol features.

    Overrides symbol feature extraction to use canonical IDs and
    structural role properties instead of raw symnums. All other
    graph construction (clause/literal/term topology, variable handling,
    shared variable edges) is inherited from _GraphBuilder.
    """

    def __init__(
        self,
        config: ClauseGraphConfig,
        symbol_table: SymbolTable | None,
    ) -> None:
        super().__init__(config, symbol_table)
        self._feature_extractor = InvariantFeatureExtractor(symbol_table)

    def add_clause(self, clause: Clause) -> int:
        """Add a clause, first preparing the canonical mapping."""
        self._feature_extractor.prepare(clause)
        return super().add_clause(clause)

    def reset(self) -> None:
        """Reset builder state for reuse."""
        super().reset()
        self._feature_extractor.reset()

    def _symbol_features(self, symnum: int) -> list[float]:
        """Extract invariant symbol features using canonical mapping.

        Returns 6 floats matching the original feature dimensionality:
        [canonical_id, arity, is_predicate, is_skolem, occurrence_count,
         distinct_arg_arities]
        """
        return self._feature_extractor.symbol_features(symnum)


def invariant_clause_to_heterograph(
    clause: Clause,
    symbol_table: SymbolTable | None = None,
    config: ClauseGraphConfig | None = None,
) -> HeteroData:
    """Convert a clause to a HeteroData graph with invariant symbol features.

    Drop-in replacement for clause_to_heterograph that produces
    symbol-independent graphs. Graph topology is identical; only
    symbol node features differ.

    Args:
        clause: The PyLADR Clause to convert.
        symbol_table: Optional SymbolTable for Skolem detection.
        config: Graph construction configuration.

    Returns:
        A HeteroData with invariant symbol features.
    """
    cfg = config or ClauseGraphConfig()
    builder = _InvariantGraphBuilder(cfg, symbol_table)
    builder.add_clause(clause)
    return builder.build()


def batch_invariant_clauses_to_heterograph(
    clauses: list[Clause],
    symbol_table: SymbolTable | None = None,
    config: ClauseGraphConfig | None = None,
) -> list[HeteroData]:
    """Convert multiple clauses to invariant HeteroData graphs.

    Drop-in replacement for batch_clauses_to_heterograph.

    Args:
        clauses: List of PyLADR Clauses.
        symbol_table: Optional SymbolTable for Skolem detection.
        config: Graph construction configuration.

    Returns:
        List of HeteroData graphs with invariant symbol features.
    """
    cfg = config or ClauseGraphConfig()
    builder = _InvariantGraphBuilder(cfg, symbol_table)
    results = []
    for clause in clauses:
        builder.reset()
        builder.add_clause(clause)
        results.append(builder.build())
    return results
