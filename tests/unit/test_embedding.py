"""Domain embedding types."""

from __future__ import annotations

import pytest

from rag.domain.embedding import (
    Embedding,
    EmbedMode,
    ModelInfo,
    RerankResult,
    SparseVector,
)


class TestSparseVector:
    def test_parallel_arrays_must_correspond(self) -> None:
        # Two arrays of different lengths are not a sparse vector, they are
        # corruption. Rejected at construction, because otherwise it surfaces
        # deep inside the vector store — or worse, gets silently truncated to
        # the shorter of the two.
        with pytest.raises(ValueError, match="must correspond"):
            SparseVector(indices=(1, 2, 3), values=(0.5, 0.25))

    def test_length_is_the_number_of_non_zero_terms(self) -> None:
        vector = SparseVector(indices=(7, 19), values=(0.9, 0.1))

        assert len(vector) == 2

    def test_an_empty_vector_is_representable(self) -> None:
        # A chunk of pure punctuation legitimately produces no weighted terms.
        # It is a state to check, not an error — indexing decides what to do.
        assert SparseVector(indices=(), values=()).is_empty

    def test_it_is_frozen(self) -> None:
        vector = SparseVector(indices=(1,), values=(1.0,))

        with pytest.raises(AttributeError):
            vector.indices = (2,)  # type: ignore[misc]


class TestEmbedding:
    def test_dimensions_come_from_the_dense_vector(self) -> None:
        embedding = Embedding(
            dense=(0.1, 0.2, 0.3),
            sparse=SparseVector(indices=(4,), values=(1.0,)),
        )

        assert embedding.dimensions == 3

    def test_dense_and_sparse_travel_together(self) -> None:
        # The reason they are one type: a chunk indexed densely but not sparsely
        # is half-retrievable and nothing errors. Constructing one without the
        # other must be impossible, not merely discouraged.
        with pytest.raises(TypeError):
            Embedding(dense=(0.1,))  # type: ignore[call-arg]


class TestEmbedMode:
    def test_it_serialises_to_the_wire_value(self) -> None:
        # `str(mode)` is what the client puts in the request body, so a rename
        # of the Python member must not be able to change the wire contract
        # silently.
        assert str(EmbedMode.QUERY) == "query"
        assert str(EmbedMode.PASSAGE) == "passage"


class TestModelInfo:
    def test_the_reranker_is_optional(self) -> None:
        # A service started with the reranker disabled to save VRAM is a
        # supported configuration, not a broken one.
        info = ModelInfo(
            embedding_model="bge-m3",
            embedding_version="abc123",
            dimensions=1024,
            max_sequence_tokens=1024,
            tokenizer_hash="deadbeefdeadbeef",
        )

        assert info.reranker_model is None


class TestRerankResult:
    def test_it_carries_a_position_not_the_passage(self) -> None:
        # Echoing the text back would double the payload and put customer data
        # on the wire a second time for something the caller already holds.
        result = RerankResult(index=3, score=-1.5)

        assert result.index == 3
        assert not hasattr(result, "passage")
