"""Token vocabulary manager for RNN2Vec.

Handles token-to-integer mapping with PAD/UNK support for batched
RNN sequence encoding. Vocabulary is built from tree walk token
sequences and sorted by frequency descending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass
class TokenVocab:
    """Vocabulary mapping tokens to integer IDs.

    Invariants:
        - PAD_TOKEN always maps to id=0
        - UNK_TOKEN always maps to id=1
        - Real tokens start at id=2
        - Real tokens sorted by frequency descending (ties broken alphabetically)
    """

    PAD_ID: int = 0
    UNK_ID: int = 1
    _token_to_id: dict[str, int] = field(default_factory=dict)
    _id_to_token: list[str] = field(default_factory=list)

    @classmethod
    def from_walks(cls, walks: Sequence[Sequence[str]]) -> TokenVocab:
        """Build vocab from walk corpus.

        PAD=0, UNK=1, then real tokens sorted by frequency descending
        (ties broken alphabetically for determinism).
        """
        counts: dict[str, int] = {}
        for walk in walks:
            for token in walk:
                counts[token] = counts.get(token, 0) + 1

        token_to_id: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        id_to_token: list[str] = [PAD_TOKEN, UNK_TOKEN]

        for token, _count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            if token in token_to_id:
                continue
            tid = len(id_to_token)
            token_to_id[token] = tid
            id_to_token.append(token)

        return cls(_token_to_id=token_to_id, _id_to_token=id_to_token)

    def encode_walk(self, walk: Sequence[str]) -> list[int]:
        """Convert token sequence to int IDs. Unknown tokens -> UNK_ID."""
        return [self._token_to_id.get(t, self.UNK_ID) for t in walk]

    def extend(self, token: str) -> int:
        """Add new token to vocab.

        Returns the new token's ID. Caller is responsible for extending
        the embedding matrix (e.g. mean-initialization).

        Raises:
            ValueError: If token already exists in vocabulary.
        """
        if token in self._token_to_id:
            raise ValueError(f"Token {token!r} already in vocabulary")
        tid = len(self._id_to_token)
        self._token_to_id[token] = tid
        self._id_to_token.append(token)
        return tid

    @property
    def size(self) -> int:
        """Total vocabulary size including PAD and UNK."""
        return len(self._id_to_token)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "tokens": list(self._id_to_token),
        }

    @classmethod
    def from_dict(cls, d: dict) -> TokenVocab:
        """Deserialize from dict produced by to_dict()."""
        tokens: list[str] = d["tokens"]
        token_to_id = {t: i for i, t in enumerate(tokens)}
        return cls(_token_to_id=token_to_id, _id_to_token=list(tokens))
