"""Proof-level message passing network for hierarchical clause embeddings.

Implements the proof level of the hierarchical message passing architecture:
  1. Derivation message passing — propagates information along the proof DAG
     (parent clauses → child clauses via justification edges)
  2. Temporal position encoding — encodes when each clause was generated
     during the proof search
  3. Goal-directed messaging — computes proximity of each clause to the
     goal clauses, enabling goal-directed clause selection
  4. Clause-to-proof composition — aggregates clause representations into
     an overall proof state representation

Designed to receive clause-level representations from ClauseLevelMPN and
produce proof-context-aware embeddings for clause selection.

No modifications to core PyLADR data structures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from pyladr.core.clause import Clause


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ProofLevelConfig:
    """Hyperparameters for the proof-level message passing network.

    Attributes:
        hidden_dim: Hidden dimension for representations.
        num_layers: Number of derivation message passing layers.
        dropout: Dropout probability.
        max_proof_depth: Maximum proof depth for position encoding.
        temporal_dim: Dimension of temporal sinusoidal encoding.
        num_derivation_types: Number of distinct derivation types (JustType enum).
    """

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    max_proof_depth: int = 100
    temporal_dim: int = 32
    num_derivation_types: int = 22  # Number of JustType values


# ── Temporal Position Encoder ─────────────────────────────────────────────


class TemporalPositionEncoder(nn.Module):
    """Encodes the temporal position of clauses in the proof search.

    Uses sinusoidal positional encoding (like Transformer PE) to represent
    when a clause was generated during search, then projects to hidden_dim.
    """

    def __init__(self, temporal_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.projection = nn.Linear(temporal_dim, hidden_dim)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps into temporal representations.

        Args:
            timestamps: (N,) float tensor of clause generation steps.

        Returns:
            (N, hidden_dim) temporal encodings.
        """
        # Sinusoidal encoding
        half_dim = self.temporal_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, dtype=torch.float32, device=timestamps.device)
            / half_dim
        )
        angles = timestamps.unsqueeze(-1) * freqs.unsqueeze(0)  # (N, half_dim)
        pe = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (N, temporal_dim)

        return self.projection(pe)


# ── Derivation Encoder ───────────────────────────────────────────────────


class DerivationEncoder(nn.Module):
    """Encodes the derivation type (justification) of clauses.

    Maps JustType enum values to learned embeddings.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_types: int = 22,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_types, hidden_dim)

    def forward(self, deriv_types: torch.Tensor) -> torch.Tensor:
        """Encode derivation types.

        Args:
            deriv_types: (N,) long tensor of JustType values.

        Returns:
            (N, hidden_dim) derivation encodings.
        """
        return self.embedding(deriv_types)

    @staticmethod
    def extract_derivation_types(clauses: list[Clause]) -> torch.Tensor:
        """Extract derivation type indices from Clause objects.

        Args:
            clauses: List of PyLADR Clause objects.

        Returns:
            (N,) long tensor of JustType values.
        """
        from pyladr.core.clause import JustType

        types = []
        for c in clauses:
            if c.justification:
                types.append(int(c.justification[0].just_type))
            else:
                types.append(int(JustType.INPUT))  # default
        return torch.tensor(types, dtype=torch.long)


# ── Goal Proximity Computer ──────────────────────────────────────────────


class GoalProximityComputer(nn.Module):
    """Computes goal-directed proximity scores for clauses.

    Given clause representations and goal clause representations, computes
    a proximity score in [0, 1] indicating how close each clause is to
    contributing to the proof goal.

    Uses a bilinear attention mechanism with learned projection.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.clause_proj = nn.Linear(hidden_dim, hidden_dim)
        self.goal_proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        clause_reprs: torch.Tensor,
        goal_reprs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute goal proximity for each clause.

        Args:
            clause_reprs: (N, hidden_dim) clause representations.
            goal_reprs: (G, hidden_dim) goal clause representations.
                Multiple goals are mean-pooled.

        Returns:
            (N,) proximity scores in [0, 1].
        """
        # Pool multiple goals into one representation
        if goal_reprs.shape[0] > 1:
            goal_repr = goal_reprs.mean(dim=0, keepdim=True)  # (1, H)
        else:
            goal_repr = goal_reprs  # (1, H)

        # Project both
        c_proj = self.clause_proj(clause_reprs)  # (N, H)
        g_proj = self.goal_proj(goal_repr)  # (1, H)

        # Expand goal to match clause batch
        g_proj = g_proj.expand_as(c_proj)  # (N, H)

        # Score each clause against goal
        combined = torch.cat([c_proj, g_proj], dim=-1)  # (N, 2H)
        return self.score(combined).squeeze(-1)  # (N,)


# ── Proof-Level Message Passing Network ──────────────────────────────────


class ProofLevelMPN(nn.Module):
    """Proof-level message passing network.

    Enriches clause representations with proof search context by:
      1. Adding temporal position encoding (when clause was generated)
      2. Adding derivation type encoding (how clause was derived)
      3. Passing messages along derivation DAG edges (parent → child)
      4. Computing goal proximity scores (optional)

    Architecture:
      1. Temporal + derivation encoding fused with clause representations
      2. N layers of derivation message passing along proof DAG
      3. Optional goal proximity computation
      4. Layer norm + residual connections throughout

    Input:
      - clause_reprs: (N, hidden_dim) — from ClauseLevelMPN
      - timestamps: (N,) — proof search step for each clause
      - deriv_types: (N,) — JustType values for each clause
      - derivation_edges: (2, E) — optional parent→child edge index
      - goal_reprs: (G, hidden_dim) — optional goal clause representations

    Output:
      - (N, hidden_dim) — proof-context-enriched clause representations
      - Optionally: (N,) goal proximity scores
    """

    def __init__(self, config: ProofLevelConfig | None = None):
        super().__init__()
        self.config = config or ProofLevelConfig()
        c = self.config

        # Temporal position encoder
        self.temporal_encoder = TemporalPositionEncoder(
            temporal_dim=c.temporal_dim,
            hidden_dim=c.hidden_dim,
        )

        # Derivation type encoder
        self.derivation_encoder = DerivationEncoder(
            hidden_dim=c.hidden_dim,
            num_types=c.num_derivation_types,
        )

        # Fusion of clause repr + temporal + derivation
        self.input_fusion = nn.Sequential(
            nn.Linear(c.hidden_dim * 3, c.hidden_dim),
            nn.ReLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
        )
        self.input_norm = nn.LayerNorm(c.hidden_dim)

        # Derivation message passing layers
        self.deriv_layers = nn.ModuleList()
        self.deriv_norms = nn.ModuleList()
        for _ in range(c.num_layers):
            self.deriv_layers.append(
                nn.Linear(c.hidden_dim * 2, c.hidden_dim)
            )
            self.deriv_norms.append(nn.LayerNorm(c.hidden_dim))

        # Goal proximity computer
        self.goal_proximity = GoalProximityComputer(hidden_dim=c.hidden_dim)

        self.dropout = nn.Dropout(c.dropout)

    def _derivation_message_pass(
        self,
        x: torch.Tensor,
        derivation_edges: torch.Tensor,
    ) -> torch.Tensor:
        """Message passing along derivation DAG.

        Parent clause representations are aggregated and used to update
        child clause representations.

        Args:
            x: (N, hidden_dim) clause representations.
            derivation_edges: (2, E) edge index (parent → child).

        Returns:
            (N, hidden_dim) updated representations.
        """
        src, dst = derivation_edges[0], derivation_edges[1]

        for layer, norm in zip(self.deriv_layers, self.deriv_norms):
            # Gather parent messages
            parent_msgs = x[src]  # (E, H)

            # Aggregate messages per child (mean)
            num_nodes = x.shape[0]
            agg = torch.zeros_like(x)  # (N, H)
            count = torch.zeros(num_nodes, 1, device=x.device)
            agg.index_add_(0, dst, parent_msgs)
            count.index_add_(0, dst, torch.ones(src.shape[0], 1, device=x.device))
            count = count.clamp(min=1)
            agg = agg / count

            # Combine self with parent aggregate
            combined = torch.cat([x, agg], dim=-1)  # (N, 2H)
            update = F.relu(layer(combined))
            update = self.dropout(update)

            # Residual + layer norm
            x = norm(x + update)

        return x

    def aggregate_proof_state(self, clause_reprs: torch.Tensor) -> torch.Tensor:
        """Aggregate clause representations into an overall proof state.

        Uses mean pooling over all clause representations.

        Args:
            clause_reprs: (N, hidden_dim) proof-enriched clause representations.

        Returns:
            (1, hidden_dim) aggregate proof state.
        """
        return clause_reprs.mean(dim=0, keepdim=True)

    def forward(
        self,
        clause_reprs: torch.Tensor,
        timestamps: torch.Tensor,
        deriv_types: torch.Tensor,
        derivation_edges: torch.Tensor | None = None,
        goal_reprs: torch.Tensor | None = None,
        return_proximity: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Full proof-level forward pass.

        Args:
            clause_reprs: (N, hidden_dim) from clause level.
            timestamps: (N,) proof search step for each clause.
            deriv_types: (N,) JustType values.
            derivation_edges: (2, E) optional parent→child edges.
            goal_reprs: (G, hidden_dim) optional goal representations.
            return_proximity: If True, return (output, proximity) tuple.

        Returns:
            If return_proximity is False: (N, hidden_dim) enriched representations.
            If return_proximity is True: tuple of (N, hidden_dim) and (N,) or None.
        """
        # Step 1: Encode temporal position and derivation type
        temporal_enc = self.temporal_encoder(timestamps)  # (N, H)
        deriv_enc = self.derivation_encoder(deriv_types)  # (N, H)

        # Step 2: Fuse clause repr + temporal + derivation
        fused = self.input_fusion(
            torch.cat([clause_reprs, temporal_enc, deriv_enc], dim=-1)
        )
        x = self.input_norm(fused + clause_reprs)  # residual from clause level

        # Step 3: Derivation message passing (if edges provided)
        if derivation_edges is not None and derivation_edges.shape[1] > 0:
            x = self._derivation_message_pass(x, derivation_edges)

        # Step 4: Goal proximity (optional)
        if return_proximity:
            proximity = None
            if goal_reprs is not None:
                proximity = self.goal_proximity(x, goal_reprs)
            return x, proximity

        return x
