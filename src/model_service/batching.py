"""Splitting a request into forward passes.

A pure function, in its own module, because it is the piece most likely to be
wrong in a way that only shows up as an out-of-memory error under load — and an
OOM on a GPU is not a nice exception, it can leave the CUDA context unusable for
the rest of the process's life. Pure means it is exhaustively testable on a CI
runner with no GPU at all.

Why a token budget rather than a fixed batch size
-------------------------------------------------
Activation memory scales with *tokens*, not items, and quadratically in sequence
length within an item. A batch of 64 thirty-token chunks and a batch of 64
thousand-token chunks differ by well over an order of magnitude in peak memory.
A fixed item count must therefore be tuned for the worst case, which wastes the
GPU on the common case; a token budget adapts to what actually arrived.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["plan_batches"]


def plan_batches(
    token_counts: Sequence[int],
    *,
    max_batch_tokens: int,
    max_batch_items: int,
) -> list[list[int]]:
    """Group indices into batches that respect both budgets.

    Returns lists of indices into `token_counts`, in order, so the caller can
    reassemble results positionally. Order is preserved rather than sorted by
    length: length bucketing would pack the GPU slightly better, and would also
    mean every result comes back attached to the wrong input unless the caller
    remembers to unsort. Retrieval still *works* after that mistake, it just
    returns unrelated text, which is why it is not a trade worth making for a
    few percent of throughput.

    An item that exceeds `max_batch_tokens` on its own still gets a batch of its
    own rather than being dropped or splitting the budget. The caller is
    expected to have rejected over-length inputs already — `Settings` enforces
    `max_batch_tokens >= max_sequence_tokens` — so this is the defensive branch
    that guarantees the function always makes progress and never returns an
    empty batch.
    """
    if max_batch_tokens < 1 or max_batch_items < 1:
        raise ValueError("Batch budgets must be positive.")

    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0

    for index, tokens in enumerate(token_counts):
        # An empty string still costs the special tokens, so treat every item as
        # at least one token. Zero-cost items would let an unbounded number of
        # them into one batch under the token budget alone.
        cost = max(tokens, 1)
        would_exceed = current and (
            current_tokens + cost > max_batch_tokens or len(current) >= max_batch_items
        )
        if would_exceed:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(index)
        current_tokens += cost

    if current:
        batches.append(current)
    return batches
