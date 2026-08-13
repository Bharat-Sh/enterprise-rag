"""Splitting a request into forward passes.

Exhaustive because the failure mode is an out-of-memory error on a GPU, which
is not a tidy exception — a CUDA OOM can leave the context unusable for the rest
of the process's life. A pure function is the one part of that story that can be
pinned down completely on a machine with no GPU.
"""

from __future__ import annotations

import pytest

from model_service.batching import plan_batches


def _flatten(batches: list[list[int]]) -> list[int]:
    return [index for batch in batches for index in batch]


class TestBudgets:
    def test_a_batch_stops_at_the_token_budget(self) -> None:
        batches = plan_batches([40, 40, 40], max_batch_tokens=100, max_batch_items=99)

        assert batches == [[0, 1], [2]]

    def test_a_batch_stops_at_the_item_budget(self) -> None:
        # The degenerate case the token budget cannot catch: thousands of
        # near-empty strings never reach the token ceiling but still cost
        # per-item memory.
        batches = plan_batches([1, 1, 1, 1, 1], max_batch_tokens=10_000, max_batch_items=2)

        assert batches == [[0, 1], [2, 3], [4]]

    def test_everything_fits_in_one_batch_when_it_fits(self) -> None:
        batches = plan_batches([10, 10, 10], max_batch_tokens=100, max_batch_items=10)

        assert batches == [[0, 1, 2]]

    def test_empty_input_produces_no_batches(self) -> None:
        assert plan_batches([], max_batch_tokens=100, max_batch_items=10) == []


class TestItAlwaysMakesProgress:
    def test_an_oversized_item_still_gets_its_own_batch(self) -> None:
        # `Settings` forbids this combination, so it is the defensive branch.
        # Dropping the item or splitting the budget would either lose an input
        # or loop forever; a batch of one is the only answer that terminates.
        batches = plan_batches([500], max_batch_tokens=100, max_batch_items=10)

        assert batches == [[0]]

    def test_zero_token_items_do_not_pack_without_bound(self) -> None:
        # An empty string still costs the two special tokens, and more
        # importantly still costs a row in the batch. Treating it as free would
        # let an unbounded number into one pass.
        batches = plan_batches([0] * 10, max_batch_tokens=1_000_000, max_batch_items=3)

        assert [len(batch) for batch in batches] == [3, 3, 3, 1]

    def test_no_batch_is_ever_empty(self) -> None:
        batches = plan_batches([7, 300, 2, 900, 1], max_batch_tokens=100, max_batch_items=2)

        assert all(batches)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_non_positive_budget_is_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            plan_batches([1, 2], max_batch_tokens=bad, max_batch_items=10)
        with pytest.raises(ValueError, match="positive"):
            plan_batches([1, 2], max_batch_tokens=10, max_batch_items=bad)


class TestOrderIsPreserved:
    @pytest.mark.parametrize(
        "counts",
        [
            [5, 500, 5, 500, 5],
            [100] * 7,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
        ],
    )
    def test_indices_come_back_in_order_and_exactly_once(self, counts: list[int]) -> None:
        # This is the property the whole design rests on. Length bucketing would
        # pack the GPU marginally better and would attach every vector to the
        # wrong chunk unless the caller remembers to unsort — after which
        # retrieval still *works*, it just returns unrelated text.
        batches = plan_batches(counts, max_batch_tokens=200, max_batch_items=3)

        assert _flatten(batches) == list(range(len(counts)))
