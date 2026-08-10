"""Split extracted text into retrievable chunks.

Pure and deterministic: same text and same policy give byte-identical chunks
every time. That matters more than it sounds — chunk text is the input to
embedding, so non-determinism here would mean a re-ingest produces different
vectors for an unchanged document, and no diff would explain why.

The approach is **recursive splitting on a separator hierarchy**, then greedy
packing with overlap. Split on paragraph breaks first; if a piece is still too
big, split it on lines, then sentences, then words, then characters. The effect
is that chunk boundaries land on the most semantically meaningful break
available, and the ugly ones are only reached by text that offers nothing better.

Everything below operates on **(start, end) spans into the original string**,
never on substrings. Offsets are therefore exact by construction rather than
reconstructed afterwards, which is what makes `chunks.char_start`/`char_end`
trustworthy enough to highlight a citation in the source document.

Why overlap exists: a passage that straddles a boundary is otherwise present in
neither chunk in full, and retrieval fails on exactly the query that passage
would have answered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.domain.models import NewChunk

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

__all__ = ["SEPARATORS", "chunk_text"]

#: Tried in order, coarsest first. Each entry is a fallback for text that the
#: previous one could not break down small enough.
SEPARATORS: tuple[str, ...] = (
    "\n\n",  # paragraphs
    "\n",  # lines
    ". ",  # sentences — crude, and the alternative is a sentence tokeniser
    " ",  # words
)

_Span = tuple[int, int]


def chunk_text(
    text: str,
    *,
    target_tokens: int,
    overlap_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[NewChunk]:
    """Split `text` into overlapping chunks of roughly `target_tokens` each.

    `count_tokens` is injected rather than imported: M3a passes a character
    ratio estimator and M4 passes the real tokenizer from the model service,
    and this function should not know or care which (see `rag.domain.ports`).
    """
    if overlap_tokens >= target_tokens:
        # Guaranteed non-termination: each chunk would begin with at least as
        # much carried-over text as it is allowed to hold. Config rejects this
        # too; asserted here because this function is called directly by tests
        # and, later, by a re-chunking job.
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be smaller than "
            f"target_tokens ({target_tokens})"
        )

    if not text.strip():
        return []

    leaves = list(_leaf_spans(text, (0, len(text)), SEPARATORS, target_tokens, count_tokens))
    return _pack(text, leaves, target_tokens, overlap_tokens, count_tokens)


# --- splitting -------------------------------------------------------------


def _leaf_spans(
    text: str,
    span: _Span,
    separators: Sequence[str],
    target_tokens: int,
    count_tokens: Callable[[str], int],
) -> Iterator[_Span]:
    """Break `span` down until every piece fits, or nothing splits it further."""
    start, end = span
    if start >= end:
        return

    if count_tokens(text[start:end]) <= target_tokens:
        yield span
        return

    if not separators:
        # No separator left: a single enormous token, a minified file, a
        # language without spaces. Cut it by length rather than refusing.
        yield from _hard_split(text, span, target_tokens, count_tokens)
        return

    parts = _split_on(text, span, separators[0])
    if len(parts) == 1:
        # This separator does not occur here; try the next without re-testing
        # the size, which we already know is too large.
        yield from _leaf_spans(text, span, separators[1:], target_tokens, count_tokens)
        return

    for part in parts:
        yield from _leaf_spans(text, part, separators[1:], target_tokens, count_tokens)


def _split_on(text: str, span: _Span, separator: str) -> list[_Span]:
    """Split a span, keeping each separator attached to the piece before it.

    Attaching rather than discarding is what keeps the pieces contiguous: every
    character of the original appears in exactly one span, so offsets stay exact
    and no text is silently dropped between chunks.
    """
    start, end = span
    spans: list[_Span] = []
    cursor = start

    while cursor < end:
        found = text.find(separator, cursor, end)
        if found == -1:
            break
        boundary = found + len(separator)
        spans.append((cursor, boundary))
        cursor = boundary

    if cursor < end:
        spans.append((cursor, end))

    return spans or [span]


def _hard_split(
    text: str,
    span: _Span,
    target_tokens: int,
    count_tokens: Callable[[str], int],
) -> Iterator[_Span]:
    """Last resort: cut by length, using an observed characters-per-token ratio.

    Growing a window one character at a time and re-counting would be quadratic,
    which on the pathological input that reaches this branch — a megabyte with
    no whitespace — is the difference between a slow job and a hung worker.
    """
    start, end = span
    tokens = max(count_tokens(text[start:end]), 1)
    chars_per_token = (end - start) / tokens
    # At least one character, so the loop always advances.
    width = max(int(target_tokens * chars_per_token), 1)

    cursor = start
    while cursor < end:
        stop = min(cursor + width, end)
        yield (cursor, stop)
        cursor = stop


# --- packing ---------------------------------------------------------------


def _pack(
    text: str,
    leaves: list[_Span],
    target_tokens: int,
    overlap_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[NewChunk]:
    """Greedily fill chunks from consecutive leaves, carrying overlap forward."""
    chunks: list[NewChunk] = []
    current: list[_Span] = []
    current_tokens = 0

    for leaf in leaves:
        leaf_tokens = count_tokens(text[leaf[0] : leaf[1]])

        if current and current_tokens + leaf_tokens > target_tokens:
            _emit(text, current, chunks, count_tokens)
            current = _overlap_tail(text, current, overlap_tokens, count_tokens)
            current_tokens = sum(count_tokens(text[s:e]) for s, e in current)

        current.append(leaf)
        current_tokens += leaf_tokens

    if current:
        _emit(text, current, chunks, count_tokens)

    return chunks


def _emit(
    text: str,
    spans: list[_Span],
    chunks: list[NewChunk],
    count_tokens: Callable[[str], int],
) -> None:
    """Append one chunk covering `spans`, with surrounding whitespace trimmed.

    Trimming moves the *offsets* rather than stripping the string, so
    `text[char_start:char_end]` still reproduces the chunk exactly. Stripping
    the text alone would leave the stored offsets pointing at whitespace and
    quietly break any future attempt to highlight a citation in the source.
    """
    start, end = spans[0][0], spans[-1][1]

    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1

    if start >= end:
        # Whitespace-only run between two real chunks. Dropping it is right:
        # an empty chunk embeds to noise and can only ever pollute retrieval.
        return

    body = text[start:end]
    chunks.append(
        NewChunk(
            ordinal=len(chunks),
            text=body,
            token_count=count_tokens(body),
            char_start=start,
            char_end=end,
        )
    )


def _overlap_tail(
    text: str,
    spans: list[_Span],
    overlap_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[_Span]:
    """The trailing spans to carry into the next chunk.

    Never the whole of `spans`. If it were, the next chunk would start exactly
    where this one did and the packer would not advance — the infinite loop this
    function exists to not have. Dropping the leading span guarantees progress
    even when a single leaf is larger than the overlap budget.
    """
    if overlap_tokens <= 0 or len(spans) <= 1:
        return []

    tail: list[_Span] = []
    accumulated = 0
    for span in reversed(spans):
        span_tokens = count_tokens(text[span[0] : span[1]])
        if accumulated + span_tokens > overlap_tokens and tail:
            break
        tail.insert(0, span)
        accumulated += span_tokens
        if accumulated >= overlap_tokens:
            break

    if len(tail) >= len(spans):
        tail = tail[1:]
    return tail
