"""A character-ratio token estimator, standing in until M4.

Why an estimator rather than a real tokenizer
---------------------------------------------
The tokenizer that matters is BGE-M3's (XLM-RoBERTa SentencePiece), and it lives
with the model service that M4 introduces. Two alternatives were rejected:

*`tiktoken`* is OpenAI's BPE. It would give a precise count for a model we do
not use — systematically wrong for ours, and wrong in a way that looks
authoritative. An estimate labelled as an estimate is more honest and no less
accurate here.

*Vendoring `transformers` + the tokenizer weights* means a large dependency and
a model download in M3, to compute a number that M4 will compute properly
anyway.

What the number is used for
---------------------------
Chunk sizing, and `chunks.token_count`. Both tolerate being approximate: being
10% out moves a chunk boundary slightly. What would *not* tolerate it is
budgeting a model's context window — nothing does that until M4, by which point
this is replaced.

The ratio
---------
Roughly four characters per token for English prose, which is the widely
observed figure for subword vocabularies. CJK text is far denser — closer to one
token per character — so it is counted separately rather than being underestimated
by a factor of four, which would produce chunks that silently overflow the
model's window.
"""

from __future__ import annotations

import unicodedata

__all__ = ["HeuristicTokenCounter"]

#: Latin-script prose, subword vocabulary.
_CHARS_PER_TOKEN = 4.0

#: Unicode ranges where one character is approximately one token.
_DENSE_RANGES: tuple[tuple[int, int], ...] = (
    (0x3040, 0x30FF),  # Hiragana, Katakana
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
)


def _is_dense(character: str) -> bool:
    codepoint = ord(character)
    return any(low <= codepoint <= high for low, high in _DENSE_RANGES)


class HeuristicTokenCounter:
    """Satisfies `rag.domain.ports.TokenCounter`."""

    def __init__(self, *, chars_per_token: float = _CHARS_PER_TOKEN) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self._chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        """Estimated token count. Never negative; zero only for empty input.

        A non-empty string always counts as at least one token, because the
        chunker uses this to decide whether progress was made — a span that
        counted as zero would let it loop.
        """
        if not text:
            return 0

        dense = 0
        sparse = 0
        for character in text:
            if _is_dense(character):
                dense += 1
            elif not unicodedata.category(character).startswith("C"):
                # Control characters are markup for the file, not content.
                sparse += 1

        estimate = dense + round(sparse / self._chars_per_token)
        return max(estimate, 1)
