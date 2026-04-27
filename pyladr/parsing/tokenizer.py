"""Tokenizer for LADR syntax, matching C parse.c tokenize().

Character classification and tokenization rules replicate the C behavior exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto


class TokenType(IntEnum):
    """Token types matching C parse.c enum."""

    ORDINARY = 0  # alphanumeric, _, $
    SPECIAL = auto()  # operator chars (+, -, *, etc.)
    STRING = auto()  # quoted string
    PUNC = auto()  # punctuation: ( ) [ ] { } , .
    UNKNOWN = auto()
    EOF = auto()


@dataclass(slots=True)
class Token:
    """Single token from LADR input."""

    type: TokenType
    value: str
    pos: int = 0  # position in source

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r})"


# ── Character classification (matching C parse.c exactly) ────────────────────

COMMENT_CHAR = "%"
QUOTE_CHAR = '"'
END_CHAR = "."
CONS_CHAR = ":"  # list cons, as in [x:y]

_PUNCTUATION = frozenset(",()" + "[]" + "{}" + END_CHAR)

# Single-character special tokens that do NOT aggregate with adjacent special chars.
# The apostrophe/prime must be solo so that x'' → two separate ' tokens (not one '' token).
_SOLO_SPECIAL = frozenset("'")

_SPECIAL = frozenset(
    "+-*/\\^<>=`~?@&|:!#%'\".;"
)

_ORDINARY_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "_$"
)


def is_ordinary_char(c: str) -> bool:
    """C ordinary_char(): alphanumeric, _, $."""
    return c in _ORDINARY_CHARS


def is_special_char(c: str) -> bool:
    """C special_char(): special operator chars, excluding quote/end/comment/punctuation."""
    if c in (QUOTE_CHAR, COMMENT_CHAR) or c in _PUNCTUATION:
        return False
    return c in _SPECIAL


def is_punctuation_char(c: str) -> bool:
    """C punctuation_char(): structural characters."""
    return c in _PUNCTUATION


def is_white_char(c: str) -> bool:
    """C white_char(): whitespace."""
    return c in (" ", "\t", "\n", "\r", "\f", "\v")


# ── Tokenizer ───────────────────────────────────────────────────────────────


def tokenize(source: str) -> list[Token]:
    """Tokenize LADR input string, matching C tokenize() behavior.

    The C parser reads one term at a time (up to '.'), strips comments,
    then tokenizes the buffer. We replicate that: input should already
    have comments stripped or be a single term string.

    Args:
        source: Input string (comments should be stripped beforehand).

    Returns:
        List of tokens.
    """
    tokens: list[Token] = []
    i = 0
    n = len(source)

    while i < n:
        c = source[i]

        if is_white_char(c):
            i += 1
            continue

        if is_punctuation_char(c):
            tokens.append(Token(type=TokenType.PUNC, value=c, pos=i))
            i += 1

        elif is_ordinary_char(c):
            start = i
            while i < n and is_ordinary_char(source[i]):
                i += 1
            tokens.append(
                Token(type=TokenType.ORDINARY, value=source[start:i], pos=start)
            )

        elif is_special_char(c):
            start = i
            if c in _SOLO_SPECIAL:
                # Solo special chars emit a single-char token (do not aggregate).
                # This ensures x'' → two separate ' tokens, not one '' token.
                i += 1
            else:
                i += 1
                while i < n and is_special_char(source[i]) and source[i] not in _SOLO_SPECIAL:
                    i += 1
            tokens.append(
                Token(type=TokenType.SPECIAL, value=source[start:i], pos=start)
            )

        elif c == QUOTE_CHAR:
            start = i
            i += 1  # skip opening quote
            while i < n and source[i] != QUOTE_CHAR:
                i += 1
            if i < n:
                i += 1  # skip closing quote
            tokens.append(
                Token(type=TokenType.STRING, value=source[start:i], pos=start)
            )

        else:
            tokens.append(Token(type=TokenType.UNKNOWN, value=c, pos=i))
            i += 1

    return tokens


def strip_comments(source: str) -> str:
    """Remove LADR comments (% to end of line), preserving quoted strings.

    Matches C read_buf()/finish_comment() behavior.
    """
    result: list[str] = []
    i = 0
    n = len(source)
    in_quote = False

    while i < n:
        c = source[i]

        if in_quote:
            result.append(c)
            if c == QUOTE_CHAR:
                in_quote = False
            i += 1
        elif c == QUOTE_CHAR:
            result.append(c)
            in_quote = True
            i += 1
        elif c == COMMENT_CHAR:
            # Skip to end of line
            while i < n and source[i] != "\n":
                i += 1
        else:
            result.append(c)
            i += 1

    return "".join(result)
