"""Hierarchical message passing networks for goal-directed clause selection.

Implements three levels of the message passing hierarchy:
  1. SymbolLevelMPN: Type-aware attention between predicate, function, and constant symbols
  2. TermLevelMPN: Tree-structured attention following the term tree hierarchy
  3. LiteralLevelMPN: Polarity-aware messaging between literals in a clause

Each level operates on a heterogeneous graph (HeteroData) from clause_graph.py
and transforms the node feature dict (x_dict) for its specific level, passing
through features of other node types unchanged.

These compose bottom-up: symbol → term → literal → clause.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv

from pyladr.ml.graph.clause_graph import EdgeType, NodeType


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HierarchicalMPNConfig:
    """Configuration for hierarchical message passing networks.

    Attributes:
        hidden_dim: Hidden dimension for all MPN layers.
        num_attention_heads: Number of attention heads for tree-structured attention.
        dropout: Dropout probability.
        symbol_type_count: Number of distinct symbol categories (predicate, function, constant).
    """

    hidden_dim: int = 256
    num_attention_heads: int = 4
    dropout: float = 0.1
    symbol_type_count: int = 3  # predicate, function, constant


# ── Symbol-level message passing ──────────────────────────────────────────


class SymbolLevelMPN(nn.Module):
    """Type-aware message passing between symbol nodes.

    Enriches symbol node representations by:
    1. Adding learned type embeddings (predicate/function/constant)
    2. Running message passing between symbols connected via shared terms
       (symbol_of and rev_symbol_of edges route through term nodes)
    3. Applying attention-weighted aggregation from neighboring symbols

    This level captures relationships like "predicate P always appears with
    function f" or "constants a and b co-occur in similar contexts".
    """

    def __init__(self, config: HierarchicalMPNConfig | None = None):
        super().__init__()
        self.config = config or HierarchicalMPNConfig()
        c = self.config

        # Type embedding: 3 categories (predicate, function, constant)
        self.type_embedding = nn.Embedding(c.symbol_type_count, c.hidden_dim)

        # Combine original features + type embedding
        self.combine = nn.Linear(c.hidden_dim * 2, c.hidden_dim)

        # Message passing: symbol ← term (via rev_symbol_of)
        # and term ← symbol (via symbol_of) for bidirectional flow
        self.conv = HeteroConv({
            (NodeType.SYMBOL.value, "rev_symbol_of", NodeType.TERM.value): SAGEConv(
                (c.hidden_dim, c.hidden_dim), c.hidden_dim
            ),
            (NodeType.TERM.value, EdgeType.SYMBOL_OF.value, NodeType.SYMBOL.value): SAGEConv(
                (c.hidden_dim, c.hidden_dim), c.hidden_dim
            ),
        }, aggr="sum")

        self.norm = nn.LayerNorm(c.hidden_dim)
        self.dropout = nn.Dropout(c.dropout)

    def forward(
        self, x_dict: dict[str, torch.Tensor], data: HeteroData
    ) -> dict[str, torch.Tensor]:
        """Process symbol-level messages.

        Args:
            x_dict: Node features dict {node_type: (N, hidden_dim)}.
            data: HeteroData with edge indices.

        Returns:
            Updated x_dict with transformed symbol features.
        """
        result = dict(x_dict)  # shallow copy — pass through other types

        sym_key = NodeType.SYMBOL.value
        if sym_key not in x_dict or x_dict[sym_key] is None:
            return result

        sym_x = x_dict[sym_key]
        num_symbols = sym_x.shape[0]

        if num_symbols == 0:
            return result

        # Step 1: Classify symbols by type using arity from raw features
        # Raw features: [symnum, arity, sym_type, is_skolem, kb_weight, occurrences]
        if hasattr(data[sym_key], 'x') and data[sym_key].x is not None:
            raw = data[sym_key].x
            arity = raw[:, 1]  # arity feature
            # Heuristic: arity 0 = constant (type 2), arity > 0 = function/predicate (type 1/0)
            # We'll use arity > 0 with sym_type to distinguish pred vs func
            sym_type_raw = raw[:, 2] if raw.shape[1] > 2 else torch.zeros(num_symbols)
            type_ids = torch.where(
                arity == 0,
                torch.full((num_symbols,), 2, dtype=torch.long, device=sym_x.device),  # constant
                torch.where(
                    sym_type_raw > 0,
                    torch.zeros(num_symbols, dtype=torch.long, device=sym_x.device),  # predicate
                    torch.ones(num_symbols, dtype=torch.long, device=sym_x.device),  # function
                ),
            )
        else:
            type_ids = torch.zeros(num_symbols, dtype=torch.long, device=sym_x.device)

        # Step 2: Add type embeddings
        type_emb = self.type_embedding(type_ids)
        sym_enriched = torch.relu(self.combine(torch.cat([sym_x, type_emb], dim=-1)))

        # Step 3: Message passing (if edges exist)
        edge_index_dict = self._collect_edges(data)

        if edge_index_dict:
            msg_x_dict = dict(x_dict)
            msg_x_dict[sym_key] = sym_enriched

            out_dict = self.conv(msg_x_dict, edge_index_dict)

            if sym_key in out_dict:
                # Residual + norm + dropout
                sym_out = self.norm(out_dict[sym_key] + sym_enriched)
                sym_out = self.dropout(sym_out)
                result[sym_key] = sym_out
            else:
                result[sym_key] = sym_enriched
        else:
            result[sym_key] = sym_enriched

        return result

    def _collect_edges(self, data: HeteroData) -> dict[tuple[str, str, str], torch.Tensor]:
        """Collect relevant edge indices for symbol-level message passing."""
        edge_dict: dict[tuple[str, str, str], torch.Tensor] = {}

        # Forward: term → symbol (symbol_of)
        fwd_key = (NodeType.TERM.value, EdgeType.SYMBOL_OF.value, NodeType.SYMBOL.value)
        for et in data.edge_types:
            if tuple(et) == fwd_key:
                edge_dict[fwd_key] = data[et].edge_index
                # Also add reverse: symbol → term
                rev_key = (NodeType.SYMBOL.value, "rev_symbol_of", NodeType.TERM.value)
                edge_dict[rev_key] = data[et].edge_index.flip(0)
                break

        return edge_dict


# ── Term-level message passing ────────────────────────────────────────────


class TermLevelMPN(nn.Module):
    """Tree-structured attention for term nodes.

    Processes the term tree hierarchy with:
    1. Bottom-up aggregation: child terms → parent terms via HAS_ARG edges
    2. Symbol enrichment: symbol features flow to terms via rev_symbol_of
    3. Variable propagation: variable features flow to terms via VAR_OCCURRENCE
    4. Attention-weighted aggregation of child messages

    This captures structural patterns like "terms with deeply nested functions"
    or "terms that share variables at the leaves".
    """

    def __init__(self, config: HierarchicalMPNConfig | None = None):
        super().__init__()
        self.config = config or HierarchicalMPNConfig()
        c = self.config

        # Tree attention: query/key/value projections for parent-child attention
        self.tree_attention = nn.MultiheadAttention(
            embed_dim=c.hidden_dim,
            num_heads=c.num_attention_heads,
            dropout=c.dropout,
            batch_first=True,
        )

        # Message passing conv for term-level edges
        # All source node types also appear as destinations for bidirectional flow
        self.conv = HeteroConv({
            # Child → parent (bottom-up): reverse of HAS_ARG
            (NodeType.TERM.value, "rev_has_arg", NodeType.TERM.value): SAGEConv(
                (c.hidden_dim, c.hidden_dim), c.hidden_dim
            ),
            # Parent → child (top-down): HAS_ARG
            (NodeType.TERM.value, EdgeType.HAS_ARG.value, NodeType.TERM.value): SAGEConv(
                (c.hidden_dim, c.hidden_dim), c.hidden_dim
            ),
            # Symbol → term (symbol info flows to term)
            (NodeType.SYMBOL.value, "rev_symbol_of", NodeType.TERM.value): SAGEConv(
                (c.hidden_dim, c.hidden_dim), c.hidden_dim
            ),
            # Term → symbol (term context flows back to symbol)
            (NodeType.TERM.value, EdgeType.SYMBOL_OF.value, NodeType.SYMBOL.value): SAGEConv(
                (c.hidden_dim, c.hidden_dim), c.hidden_dim
            ),
            # Variable → term (variable info flows to term)
            (NodeType.VARIABLE.value, EdgeType.VAR_OCCURRENCE.value, NodeType.TERM.value): SAGEConv(
                (c.hidden_dim, c.hidden_dim), c.hidden_dim
            ),
            # Term → variable (term context flows back to variable)
            (NodeType.TERM.value, "rev_var_occurrence", NodeType.VARIABLE.value): SAGEConv(
                (c.hidden_dim, c.hidden_dim), c.hidden_dim
            ),
        }, aggr="sum")

        self.norm = nn.LayerNorm(c.hidden_dim)
        self.dropout = nn.Dropout(c.dropout)

        # Post-attention projection
        self.post_attn = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.ReLU(),
            nn.Dropout(c.dropout),
        )

    def forward(
        self, x_dict: dict[str, torch.Tensor], data: HeteroData
    ) -> dict[str, torch.Tensor]:
        """Process term-level messages with tree-structured attention.

        Args:
            x_dict: Node features dict {node_type: (N, hidden_dim)}.
            data: HeteroData with edge indices.

        Returns:
            Updated x_dict with transformed term (and possibly variable) features.
        """
        result = dict(x_dict)

        term_key = NodeType.TERM.value
        if term_key not in x_dict or x_dict[term_key] is None:
            return result

        term_x = x_dict[term_key]
        if term_x.shape[0] == 0:
            return result

        # Step 1: Heterogeneous message passing
        edge_index_dict = self._collect_edges(data)

        if edge_index_dict:
            out_dict = self.conv(x_dict, edge_index_dict)

            if term_key in out_dict:
                # Residual + norm
                term_out = self.norm(out_dict[term_key] + term_x)
                term_out = self.dropout(term_out)
            else:
                term_out = term_x
        else:
            term_out = term_x

        # Step 2: Self-attention over all terms for global context
        # Reshape for multihead attention: (1, N, hidden_dim)
        term_unsq = term_out.unsqueeze(0)
        attn_out, _ = self.tree_attention(term_unsq, term_unsq, term_unsq)
        attn_out = attn_out.squeeze(0)

        # Post-attention with residual
        term_final = self.post_attn(attn_out) + term_out

        result[term_key] = term_final

        return result

    def _collect_edges(self, data: HeteroData) -> dict[tuple[str, str, str], torch.Tensor]:
        """Collect relevant edge indices for term-level message passing."""
        edge_dict: dict[tuple[str, str, str], torch.Tensor] = {}

        edge_types_set = {tuple(et) for et in data.edge_types}

        # HAS_ARG: term → term (parent → child)
        has_arg_key = (NodeType.TERM.value, EdgeType.HAS_ARG.value, NodeType.TERM.value)
        if has_arg_key in edge_types_set:
            ei = data[has_arg_key].edge_index
            edge_dict[has_arg_key] = ei
            # Reverse: child → parent (bottom-up)
            rev_key = (NodeType.TERM.value, "rev_has_arg", NodeType.TERM.value)
            edge_dict[rev_key] = ei.flip(0)

        # SYMBOL_OF: term → symbol, and reverse: symbol → term
        sym_of_key = (NodeType.TERM.value, EdgeType.SYMBOL_OF.value, NodeType.SYMBOL.value)
        if sym_of_key in edge_types_set:
            ei = data[sym_of_key].edge_index
            edge_dict[sym_of_key] = ei
            rev_key = (NodeType.SYMBOL.value, "rev_symbol_of", NodeType.TERM.value)
            edge_dict[rev_key] = ei.flip(0)

        # VAR_OCCURRENCE: variable → term, and reverse: term → variable
        var_key = (NodeType.VARIABLE.value, EdgeType.VAR_OCCURRENCE.value, NodeType.TERM.value)
        if var_key in edge_types_set:
            ei = data[var_key].edge_index
            edge_dict[var_key] = ei
            rev_var_key = (NodeType.TERM.value, "rev_var_occurrence", NodeType.VARIABLE.value)
            edge_dict[rev_var_key] = ei.flip(0)

        return edge_dict


# ── Literal-level message passing ─────────────────────────────────────────


class LiteralLevelMPN(nn.Module):
    """Polarity-aware message passing for literal nodes.

    Processes literal nodes with:
    1. Polarity embedding: learned positive/negative embeddings
    2. Term aggregation: atom term features flow to literals via rev_has_atom
    3. Inter-literal attention: literals attend to each other within a clause

    This captures patterns like "negative literals tend to be resolved against"
    or "equality literals in the head of a Horn clause".
    """

    def __init__(self, config: HierarchicalMPNConfig | None = None):
        super().__init__()
        self.config = config or HierarchicalMPNConfig()
        c = self.config

        # Polarity embedding: 2 types (positive=0, negative=1)
        self.polarity_embedding = nn.Embedding(2, c.hidden_dim)

        # Combine original features + polarity
        self.combine = nn.Linear(c.hidden_dim * 2, c.hidden_dim)

        # Message passing: term ↔ literal (bidirectional for HAS_ATOM)
        self.conv = HeteroConv({
            # Term → literal (atom info flows to literal)
            (NodeType.TERM.value, "rev_has_atom", NodeType.LITERAL.value): SAGEConv(
                (c.hidden_dim, c.hidden_dim), c.hidden_dim
            ),
            # Literal → term (literal context flows back to term)
            (NodeType.LITERAL.value, EdgeType.HAS_ATOM.value, NodeType.TERM.value): SAGEConv(
                (c.hidden_dim, c.hidden_dim), c.hidden_dim
            ),
        }, aggr="sum")

        # Inter-literal attention
        self.literal_attention = nn.MultiheadAttention(
            embed_dim=c.hidden_dim,
            num_heads=c.num_attention_heads,
            dropout=c.dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(c.hidden_dim)
        self.dropout = nn.Dropout(c.dropout)

        # Post-attention projection
        self.post_attn = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.ReLU(),
            nn.Dropout(c.dropout),
        )

    def forward(
        self, x_dict: dict[str, torch.Tensor], data: HeteroData
    ) -> dict[str, torch.Tensor]:
        """Process literal-level messages with polarity awareness.

        Args:
            x_dict: Node features dict {node_type: (N, hidden_dim)}.
            data: HeteroData with edge indices.

        Returns:
            Updated x_dict with transformed literal features.
        """
        result = dict(x_dict)

        lit_key = NodeType.LITERAL.value
        if lit_key not in x_dict or x_dict[lit_key] is None:
            return result

        lit_x = x_dict[lit_key]
        num_lits = lit_x.shape[0]

        if num_lits == 0:
            return result

        # Step 1: Extract polarity from raw literal features
        # Raw features: [sign, position, is_eq_literal]
        if hasattr(data[lit_key], 'x') and data[lit_key].x is not None:
            raw = data[lit_key].x
            # sign feature: 1.0 = positive, 0.0 = negative
            polarity_ids = (1.0 - raw[:, 0]).long()  # 0 = positive, 1 = negative
        else:
            polarity_ids = torch.zeros(num_lits, dtype=torch.long, device=lit_x.device)

        # Step 2: Add polarity embeddings
        pol_emb = self.polarity_embedding(polarity_ids)
        lit_enriched = torch.relu(self.combine(torch.cat([lit_x, pol_emb], dim=-1)))

        # Step 3: Message passing from terms (via rev_has_atom)
        edge_index_dict = self._collect_edges(data)

        if edge_index_dict:
            msg_x_dict = dict(x_dict)
            msg_x_dict[lit_key] = lit_enriched

            out_dict = self.conv(msg_x_dict, edge_index_dict)

            if lit_key in out_dict:
                lit_out = self.norm(out_dict[lit_key] + lit_enriched)
                lit_out = self.dropout(lit_out)
            else:
                lit_out = lit_enriched
        else:
            lit_out = lit_enriched

        # Step 4: Inter-literal self-attention
        lit_unsq = lit_out.unsqueeze(0)
        attn_out, _ = self.literal_attention(lit_unsq, lit_unsq, lit_unsq)
        attn_out = attn_out.squeeze(0)

        lit_final = self.post_attn(attn_out) + lit_out

        result[lit_key] = lit_final

        return result

    def _collect_edges(self, data: HeteroData) -> dict[tuple[str, str, str], torch.Tensor]:
        """Collect relevant edge indices for literal-level message passing."""
        edge_dict: dict[tuple[str, str, str], torch.Tensor] = {}

        edge_types_set = {tuple(et) for et in data.edge_types}

        # HAS_ATOM: literal → term (forward), and reverse: term → literal
        has_atom_key = (NodeType.LITERAL.value, EdgeType.HAS_ATOM.value, NodeType.TERM.value)
        if has_atom_key in edge_types_set:
            ei = data[has_atom_key].edge_index
            edge_dict[has_atom_key] = ei
            rev_key = (NodeType.TERM.value, "rev_has_atom", NodeType.LITERAL.value)
            edge_dict[rev_key] = ei.flip(0)

        return edge_dict
