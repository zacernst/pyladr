"""Shared protocols for cross-package abstraction boundaries.

Protocols defined here are the canonical definitions. They are re-exported
from their original locations (e.g. ``search.ml_selection.EmbeddingProvider``,
``ml.training.contrastive.ClauseEncoder``) for full backward compatibility —
existing imports continue to work unchanged.

New code SHOULD import from here to avoid coupling to implementation packages.

Why this module exists:
    ``pyladr.search`` and ``pyladr.ml`` have a bidirectional dependency
    through protocols that cross the package boundary. Extracting protocol
    definitions to a neutral location breaks the architectural coupling and
    enables cleaner layering without any consumer-visible changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

    from pyladr.core.clause import Clause


# ── EmbeddingProvider (was search.ml_selection) ──────────────────────────────


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for clause embedding providers.

    Implementations supply fixed-dimensional vector embeddings for clauses.
    The provider is responsible for caching and batching internally.
    """

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of produced embeddings."""
        ...

    def get_embedding(self, clause: Clause) -> list[float] | None:
        """Return embedding vector for a clause, or None if unavailable.

        Returns None if the provider is not ready (not trained, no model
        loaded, or an error occurred).  Callers must handle None gracefully
        — typically by falling back to traditional (weight/age) scoring.

        Implementations should handle their own caching.
        """
        ...

    def get_embeddings_batch(
        self, clauses: list[Clause],
    ) -> list[list[float] | None]:
        """Batch embedding retrieval.  Default loops over get_embedding.

        The returned list has the same length as *clauses*.  Individual
        entries may be None (same semantics as :meth:`get_embedding`).
        """
        ...


# ── ClauseEncoder (was ml.training.contrastive) ─────────────────────────────


@runtime_checkable
class ClauseEncoder(Protocol):
    """Protocol for any model that encodes clauses into embedding vectors.

    The contrastive trainer works with any encoder satisfying this interface,
    decoupling training logic from the specific GNN architecture.
    """

    def encode_clauses(self, clauses: list[Clause]) -> "torch.Tensor":
        """Encode a batch of clauses into embedding vectors.

        Args:
            clauses: List of PyLADR Clause objects.

        Returns:
            Tensor of shape (len(clauses), embedding_dim).
        """
        ...

    def parameters(self) -> Iterator[Any]:
        """Return model parameters for optimizer."""
        ...

    def train(self, mode: bool = True) -> None:
        """Set training/eval mode."""
        ...

    def eval(self) -> None:
        """Set eval mode."""
        ...
