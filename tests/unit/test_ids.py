"""UUIDv7 generation."""

from __future__ import annotations

import time

import pytest

from rag.core.ids import timestamp_of, uuid7


class TestLayout:
    def test_version_is_seven(self) -> None:
        assert uuid7().version == 7

    def test_variant_is_rfc_9562(self) -> None:
        # byte 8, top two bits == 0b10
        assert uuid7().bytes[8] >> 6 == 0b10

    def test_ids_are_unique(self) -> None:
        assert len({uuid7() for _ in range(10_000)}) == 10_000


class TestTimeOrdering:
    def test_ids_sort_by_creation_time(self) -> None:
        # This is the entire reason for choosing v7 over v4: sequential inserts
        # append to the right-hand edge of the index instead of scattering
        # across every leaf page.
        first = uuid7()
        time.sleep(0.005)
        second = uuid7()

        assert first < second

    def test_a_batch_is_non_decreasing_across_milliseconds(self) -> None:
        batch = []
        for _ in range(5):
            batch.append(uuid7())
            time.sleep(0.002)

        assert batch == sorted(batch)

    def test_embedded_timestamp_is_recoverable(self) -> None:
        before = time.time()
        value = uuid7()
        after = time.time()

        recovered = timestamp_of(value)

        # Millisecond resolution, so allow a small margin either side.
        assert before - 0.002 <= recovered <= after + 0.002

    def test_reading_a_timestamp_from_another_version_is_refused(self) -> None:
        # The first six bytes of a v4 are random and would decode to a
        # plausible-looking but entirely meaningless date.
        from uuid import uuid4

        with pytest.raises(ValueError, match="UUIDv7"):
            timestamp_of(uuid4())
