"""Cross-clause attention for relational clause scoring.

Implements multi-head attention over clause embeddings in the SOS (set of support)
to compute relational scores that capture inter-clause dependencies. This enables
the selection mechanism to prefer clauses that are complementary to each other
and to the current proof state, rather than scoring each clause in isolation.

Architecture:
  1. Project clause embeddings to query/key/value spaces
  2. Multi-head attention computes pairwise clause relationships
  3. Relational context is aggregated per clause
  4. A scoring head produces relational scores from the enriched representations

The module is designed to be used alongside (not replacing) the existing
EmbeddingEnhancedSelection scoring pipeline.

Thread Safety:
  All components are stateless nn.Modules — thread-safe for concurrent forward
  passes under torch.no_grad().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class CrossClauseAttentionConfig:
    """Configuration for cross-clause attention mechanisms.

    Attributes:
        enabled: Master switch for cross-clause attention.
        embedding_dim: Input clause embedding dimension (must match provider).
        num_heads: Number of attention heads. Each head captures a different
            relational aspect (complementarity, subsumption, diversity, etc.).
        head_dim: Dimension per attention head. If 0, computed as
            embedding_dim // num_heads.
        dropout: Attention dropout probability (applied during training only).
        use_relative_position: Encode clause age ordering as relative position
            bias in attention scores.
        max_clauses: Maximum number of clauses to attend over. Beyond this,
            the oldest clauses are dropped from the attention window.
        temperature: Softmax temperature for attention weights. Lower values
            produce sharper attention; higher values spread attention.
        use_goal_conditioning: If True, condition attention on goal context
            so that relational scoring is goal-directed.
        goal_dim: Dimension of goal context vector (for conditioning).
        scoring_hidden_dim: Hidden dimension in the relational scoring head.
    """

    enabled: bool = False
    embedding_dim: int = 512
    num_heads: int = 8
    head_dim: int = 0  # 0 = auto (embedding_dim // num_heads)
    dropout: float = 0.1
    use_relative_position: bool = True
    max_clauses: int = 512
    temperature: float = 1.0
    use_goal_conditioning: bool = False
    goal_dim: int = 128
    scoring_hidden_dim: int = 256

    @property
    def effective_head_dim(self) -> int:
        if self.head_dim > 0:
            return self.head_dim
        return self.embedding_dim // self.num_heads


class MultiHeadClauseAttention(nn.Module):
    """Multi-head attention over a set of clause embeddings.

    Given N clause embeddings, computes attention-enriched representations
    where each clause's representation incorporates relational context from
    all other clauses in the set.

    This is a standard scaled dot-product multi-head attention with optional
    relative position bias for clause age ordering.
    """

    def __init__(self, config: CrossClauseAttentionConfig):
        super().__init__()
        self.config = config
        d_model = config.embedding_dim
        n_heads = config.num_heads
        d_head = config.effective_head_dim
        d_inner = n_heads * d_head

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_inner, bias=False)
        self.W_k = nn.Linear(d_model, d_inner, bias=False)
        self.W_v = nn.Linear(d_model, d_inner, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_inner, d_model, bias=False)

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Relative position bias (learned per head)
        if config.use_relative_position:
            # Learnable bias for relative positions [-max, ..., 0, ..., +max]
            self.max_relative_position = config.max_clauses
            self.relative_position_bias = nn.Embedding(
                2 * config.max_clauses + 1, n_heads
            )
        else:
            self.relative_position_bias = None

        # Goal conditioning gate
        if config.use_goal_conditioning:
            self.goal_gate = nn.Sequential(
                nn.Linear(config.goal_dim, d_model),
                nn.Sigmoid(),
            )
        else:
            self.goal_gate = None

        self._scale = 1.0 / math.sqrt(d_head)
        self._n_heads = n_heads
        self._d_head = d_head

    def forward(
        self,
        clause_embeddings: torch.Tensor,
        goal_context: Optional[torch.Tensor] = None,
        clause_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention-enriched clause representations.

        Args:
            clause_embeddings: (N, D) clause embedding matrix where N is the
                number of clauses and D is embedding_dim.
            goal_context: Optional (D_goal,) goal context vector for conditioning.
            clause_ids: Optional (N,) integer clause IDs for relative position
                bias computation. If None, positions 0..N-1 are used.

        Returns:
            (N, D) attention-enriched clause embeddings.
        """
        N, D = clause_embeddings.shape
        residual = clause_embeddings

        # Apply goal conditioning if available
        if self.goal_gate is not None and goal_context is not None:
            gate = self.goal_gate(goal_context)  # (D,)
            clause_embeddings = clause_embeddings * gate.unsqueeze(0)  # (N, D)

        # Project to Q, K, V — shape: (N, n_heads, d_head)
        Q = self.W_q(clause_embeddings).view(N, self._n_heads, self._d_head)
        K = self.W_k(clause_embeddings).view(N, self._n_heads, self._d_head)
        V = self.W_v(clause_embeddings).view(N, self._n_heads, self._d_head)

        # Transpose for batched matmul: (n_heads, N, d_head)
        Q = Q.permute(1, 0, 2)
        K = K.permute(1, 0, 2)
        V = V.permute(1, 0, 2)

        # Scaled dot-product attention: (n_heads, N, N)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self._scale

        # Apply temperature
        if self.config.temperature != 1.0:
            attn_scores = attn_scores / self.config.temperature

        # Add relative position bias
        if self.relative_position_bias is not None:
            rel_bias = self._compute_relative_position_bias(N, clause_ids)
            attn_scores = attn_scores + rel_bias

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted aggregation: (n_heads, N, d_head)
        context = torch.bmm(attn_weights, V)

        # Reshape back: (N, n_heads * d_head)
        context = context.permute(1, 0, 2).contiguous().view(N, -1)

        # Output projection
        output = self.W_o(context)
        output = self.dropout(output)

        # Residual connection + layer norm
        return self.layer_norm(residual + output)

    def _compute_relative_position_bias(
        self, N: int, clause_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute relative position bias for attention scores.

        Uses clause IDs (or sequential positions) to compute relative distances.
        Older clauses (lower IDs) get a mild bias to attend to newer ones,
        modeling the temporal flow of the proof search.

        Returns:
            (n_heads, N, N) bias tensor to add to attention scores.
        """
        if clause_ids is not None:
            positions = clause_ids
        else:
            positions = torch.arange(N, device=self.W_q.weight.device)

        # Pairwise relative positions: (N, N)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)

        # Clamp to valid range
        max_rp = self.max_relative_position
        rel_pos = rel_pos.clamp(-max_rp, max_rp) + max_rp  # shift to [0, 2*max]

        # Lookup bias: (N, N, n_heads) → (n_heads, N, N)
        bias = self.relative_position_bias(rel_pos)  # (N, N, n_heads)
        return bias.permute(2, 0, 1)


class RelationalScoringHead(nn.Module):
    """Scoring head that produces relational clause scores.

    Takes attention-enriched clause embeddings and produces a scalar score
    per clause that reflects both individual quality and relational context
    (complementarity with other clauses, diversity, proof path potential).

    The scoring head combines:
    - Individual clause quality (from the enriched embedding)
    - Relational novelty (how different this clause is from the attended context)
    - Proof potential (predicted usefulness for closing the proof)
    """

    def __init__(self, config: CrossClauseAttentionConfig):
        super().__init__()
        d = config.embedding_dim
        h = config.scoring_hidden_dim

        # Quality score: how individually promising is this clause
        self.quality_net = nn.Sequential(
            nn.Linear(d, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

        # Novelty score: how much new information does this clause add
        # Input: concatenation of enriched embedding and original embedding
        self.novelty_net = nn.Sequential(
            nn.Linear(d * 2, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

        # Final blending weights (learned)
        self.blend = nn.Linear(2, 1, bias=True)

    def forward(
        self,
        enriched_embeddings: torch.Tensor,
        original_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score clauses using enriched relational context.

        Args:
            enriched_embeddings: (N, D) attention-enriched clause embeddings.
            original_embeddings: (N, D) original clause embeddings (pre-attention).

        Returns:
            (N,) relational scores, higher = more promising for selection.
        """
        # Individual quality from enriched representation
        quality = self.quality_net(enriched_embeddings)  # (N, 1)

        # Novelty: how much did attention change the representation?
        combined = torch.cat([enriched_embeddings, original_embeddings], dim=-1)
        novelty = self.novelty_net(combined)  # (N, 1)

        # Blend quality and novelty
        scores = self.blend(torch.cat([quality, novelty], dim=-1))  # (N, 1)
        return scores.squeeze(-1)  # (N,)


class CrossClauseAttentionScorer(nn.Module):
    """Complete cross-clause attention scoring pipeline.

    Combines MultiHeadClauseAttention with RelationalScoringHead into a
    single module that takes raw clause embeddings and produces relational
    scores suitable for blending with the existing selection system.

    This is the main entry point for the selection integration.
    """

    def __init__(self, config: CrossClauseAttentionConfig):
        super().__init__()
        self.config = config
        self.attention = MultiHeadClauseAttention(config)
        self.scorer = RelationalScoringHead(config)

    def forward(
        self,
        clause_embeddings: torch.Tensor,
        goal_context: Optional[torch.Tensor] = None,
        clause_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute relational scores for a set of clause embeddings.

        Args:
            clause_embeddings: (N, D) clause embedding matrix.
            goal_context: Optional goal context for goal-directed attention.
            clause_ids: Optional clause IDs for relative position bias.

        Returns:
            (N,) relational scores, higher = more promising.
        """
        # Truncate to max_clauses if needed
        N = clause_embeddings.shape[0]
        if N > self.config.max_clauses:
            clause_embeddings = clause_embeddings[-self.config.max_clauses:]
            if clause_ids is not None:
                clause_ids = clause_ids[-self.config.max_clauses:]

        # Compute attention-enriched representations
        enriched = self.attention(clause_embeddings, goal_context, clause_ids)

        # Score using relational context
        return self.scorer(enriched, clause_embeddings[-enriched.shape[0]:])

    @torch.no_grad()
    def score_clauses(
        self,
        clause_embeddings: torch.Tensor,
        goal_context: Optional[torch.Tensor] = None,
        clause_ids: Optional[torch.Tensor] = None,
    ) -> list[float]:
        """Inference-mode scoring (no gradients, returns Python list).

        Convenience method for integration with the selection system which
        operates on Python lists rather than tensors.

        Args:
            clause_embeddings: (N, D) clause embedding matrix.
            goal_context: Optional goal context vector.
            clause_ids: Optional clause IDs.

        Returns:
            List of N float scores.
        """
        was_training = self.training
        self.eval()
        scores = self.forward(clause_embeddings, goal_context, clause_ids)
        if was_training:
            self.train()
        return scores.tolist()
