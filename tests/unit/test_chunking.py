"""Chunking: boundaries, overlap, offsets, and the ways it could fail to terminate.

The offset assertions matter more than they look. `char_start`/`char_end` are
what a future citation feature uses to highlight a passage in the source, so a
chunk whose offsets do not reproduce its own text is a silent corruption that
nothing else in the system would notice.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from rag.domain.chunking import chunk_text


def words(count: int, *, word: str = "alpha") -> str:
    return " ".join(f"{word}{index}" for index in range(count))


def by_characters(text: str) -> int:
    """A token counter with no rounding, so expectations are exact."""
    return len(text)


class TestOffsets:
    def test_offsets_reproduce_the_chunk_text(self) -> None:
        text = "\n\n".join(f"Paragraph {index}. {words(20)}" for index in range(10))

        chunks = chunk_text(text, target_tokens=200, overlap_tokens=40, count_tokens=by_characters)

        assert chunks
        for chunk in chunks:
            assert text[chunk.char_start : chunk.char_end] == chunk.text

    def test_offsets_are_non_decreasing(self) -> None:
        text = "\n\n".join(words(30) for _ in range(20))

        chunks = chunk_text(text, target_tokens=150, overlap_tokens=30, count_tokens=by_characters)

        starts = [chunk.char_start for chunk in chunks]
        assert starts == sorted(starts)

    def test_ordinals_are_dense_and_start_at_zero(self) -> None:
        # Chunk ordinal is part of the `uq_chunks_document_id_ordinal`
        # constraint, so a gap or a repeat is an insert failure at ingest time.
        text = "\n\n".join(words(30) for _ in range(20))

        chunks = chunk_text(text, target_tokens=150, overlap_tokens=30, count_tokens=by_characters)

        assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))

    def test_chunks_carry_no_leading_or_trailing_whitespace(self) -> None:
        text = "\n\n\n".join(f"   {words(20)}   " for _ in range(8))

        chunks = chunk_text(text, target_tokens=120, overlap_tokens=20, count_tokens=by_characters)

        for chunk in chunks:
            assert chunk.text == chunk.text.strip()
            # And the offsets moved with the trim, rather than the text being
            # stripped after the fact and the offsets left pointing at spaces.
            assert text[chunk.char_start : chunk.char_end] == chunk.text


class TestSizing:
    def test_short_text_is_a_single_chunk(self) -> None:
        chunks = chunk_text(
            "A short note.", target_tokens=500, overlap_tokens=50, count_tokens=by_characters
        )

        assert len(chunks) == 1
        assert chunks[0].text == "A short note."
        assert chunks[0].char_start == 0

    def test_long_text_is_split(self) -> None:
        text = "\n\n".join(words(50) for _ in range(20))

        chunks = chunk_text(text, target_tokens=200, overlap_tokens=20, count_tokens=by_characters)

        assert len(chunks) > 1

    def test_chunks_stay_near_the_target(self) -> None:
        """Near, not under.

        A single indivisible leaf can exceed the target — a 300-character word
        cannot be split at a word boundary — so the guarantee is that the packer
        never *combines* beyond the target, not that no chunk ever exceeds it.
        """
        text = "\n\n".join(words(40) for _ in range(30))
        target = 300

        chunks = chunk_text(
            text, target_tokens=target, overlap_tokens=50, count_tokens=by_characters
        )

        # Generous headroom for the overlap carried into each chunk.
        assert all(chunk.token_count <= target * 2 for chunk in chunks)

    def test_no_chunk_is_empty(self) -> None:
        # An empty chunk embeds to noise and can only pollute retrieval.
        text = "one\n\n\n\n\n\ntwo\n\n\n\n\n\nthree"

        chunks = chunk_text(text, target_tokens=4, overlap_tokens=1, count_tokens=by_characters)

        assert all(chunk.text.strip() for chunk in chunks)


class TestOverlap:
    def test_adjacent_chunks_share_text(self) -> None:
        """The reason overlap exists.

        Without it a passage straddling a boundary is wholly present in neither
        chunk, and retrieval fails on exactly the query it would have answered.
        """
        text = "\n".join(f"line {index} {words(10)}" for index in range(40))

        chunks = chunk_text(text, target_tokens=300, overlap_tokens=100, count_tokens=by_characters)

        assert len(chunks) > 1
        overlapping = [
            later for earlier, later in pairwise(chunks) if later.char_start < earlier.char_end
        ]
        assert overlapping, "no adjacent pair overlapped"

    def test_zero_overlap_produces_disjoint_chunks(self) -> None:
        text = "\n".join(f"line {index} {words(10)}" for index in range(40))

        chunks = chunk_text(text, target_tokens=300, overlap_tokens=0, count_tokens=by_characters)

        for earlier, later in pairwise(chunks):
            assert later.char_start >= earlier.char_end

    def test_overlap_at_or_above_the_target_is_rejected(self) -> None:
        # Would guarantee non-termination: every chunk would begin with at least
        # as much carried text as it is allowed to hold.
        with pytest.raises(ValueError, match="must be smaller"):
            chunk_text("text", target_tokens=100, overlap_tokens=100, count_tokens=by_characters)


class TestTermination:
    """Inputs that could make a naive splitter loop or explode."""

    def test_a_single_enormous_word_is_split_by_length(self) -> None:
        text = "x" * 5000

        chunks = chunk_text(text, target_tokens=100, overlap_tokens=10, count_tokens=by_characters)

        assert len(chunks) > 1
        assert "".join(chunk.text for chunk in chunks).count("x") >= 5000 - len(chunks)

    def test_text_with_no_separators_at_all_terminates(self) -> None:
        chunks = chunk_text(
            "a" * 1000, target_tokens=10, overlap_tokens=5, count_tokens=by_characters
        )

        assert len(chunks) > 1

    @pytest.mark.parametrize("text", ["", "   ", "\n\n\n", "\t\n "])
    def test_blank_input_produces_nothing(self, text: str) -> None:
        assert (
            chunk_text(text, target_tokens=100, overlap_tokens=10, count_tokens=by_characters) == []
        )

    def test_a_pathological_target_of_one_still_terminates(self) -> None:
        chunks = chunk_text(
            words(50), target_tokens=1, overlap_tokens=0, count_tokens=by_characters
        )

        assert len(chunks) > 1


class TestDeterminism:
    def test_the_same_input_gives_identical_chunks(self) -> None:
        """Chunk text is the input to embedding.

        Non-determinism here would mean a re-ingest produces different vectors
        for an unchanged document, with no diff to explain it.
        """
        text = "\n\n".join(words(40) for _ in range(15))

        first = chunk_text(text, target_tokens=250, overlap_tokens=50, count_tokens=by_characters)
        second = chunk_text(text, target_tokens=250, overlap_tokens=50, count_tokens=by_characters)

        assert [(c.ordinal, c.char_start, c.char_end, c.text) for c in first] == [
            (c.ordinal, c.char_start, c.char_end, c.text) for c in second
        ]


class TestStructureIsPreferred:
    def test_paragraph_boundaries_are_preferred_over_mid_sentence_cuts(self) -> None:
        # The whole point of the separator hierarchy: break where a human would.
        paragraphs = [f"Paragraph {index}." for index in range(6)]
        text = "\n\n".join(paragraphs)

        chunks = chunk_text(
            text,
            target_tokens=len("Paragraph 0.") + 2,
            overlap_tokens=0,
            count_tokens=by_characters,
        )

        assert [chunk.text for chunk in chunks] == paragraphs
