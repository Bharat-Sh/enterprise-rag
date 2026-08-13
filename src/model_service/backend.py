"""The inference seam: what the HTTP layer is allowed to know about a model.

`InferenceBackend` exists so that everything above it — routing, authentication,
validation, batching, error mapping, the limits that keep a 6 GB card alive — is
testable without a GPU, and so that swapping FlagEmbedding for something else is
one class rather than a rewrite of the service.

**Synchronous by design**, exactly like `rag.domain.ports.DocumentParser`.
Inference is a blocking CPU-to-GPU-to-CPU round trip; an `async def` that never
awaits is a lie that invites someone to call it on the event loop, where it
would stall every other in-flight request on the process. The thread hop is the
caller's job, which puts the cost at the call site where it can be seen.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from model_service.settings import Settings

__all__ = [
    "BackendInfo",
    "InferenceBackend",
    "RawEmbedding",
    "StubBackend",
    "build_backend",
    "normalise_l2",
]

#: The stub reports BGE-M3's real width so that anything downstream which
#: validates dimensions — the vector store's collection schema, most obviously —
#: behaves identically against either backend.
STUB_DIMENSIONS = 1024


@dataclass(frozen=True, slots=True)
class RawEmbedding:
    """One text's vectors, before they become JSON.

    Tuples rather than lists: these are handed straight to response
    serialisation and must not be mutated by anything in between.
    """

    dense: tuple[float, ...]
    sparse_indices: tuple[int, ...]
    sparse_values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """Identity of what is loaded. Published by `/v1/info`."""

    embedding_model: str
    embedding_version: str
    dimensions: int
    tokenizer_hash: str
    backend: str
    reranker_model: str | None = None


@runtime_checkable
class InferenceBackend(Protocol):
    """A loaded set of models."""

    def info(self) -> BackendInfo: ...

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Token counts, including special tokens.

        Used to reject over-length input and to size batches, so it must count
        the way the model does. Approximating here would let inputs through that
        the model then truncates.
        """
        ...

    def embed(self, texts: Sequence[str], *, mode: str) -> list[RawEmbedding]:
        """Embed one batch. Dense vectors are L2-normalised."""
        ...

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        """Raw cross-encoder scores, one per passage, in the order given."""
        ...


def normalise_l2(values: list[float]) -> tuple[float, ...]:
    """L2-normalise, so cosine similarity is a dot product.

    Done here, in the service, rather than at each call site. A caller that
    forgets does not get an error — it gets scores that are wrong by a factor
    nobody notices until relevance is measured.
    """
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        # Only reachable for a degenerate all-zero vector. Returning it unchanged
        # keeps the dimension right; dividing would produce NaNs that poison
        # every downstream comparison instead of just this one.
        return tuple(values)
    return tuple(value / norm for value in values)


_WORD = re.compile(r"\w+|\S")


class StubBackend:
    """A deterministic fake with no torch, no weights, and no GPU.

    This is what makes the HTTP layer testable in CI. It is *not* a mock in the
    usual sense — it is a real implementation of the contract with an
    uninteresting model behind it, so the tests that use it exercise the actual
    routing, validation, batching and error paths rather than a stand-in for
    them.

    Two properties it deliberately has:

    *Deterministic.* The same text always yields the same vectors, derived from
    a hash. Tests can assert on equality between calls, and an integration test
    can check that the batching path and the single-pass path agree.

    *Self-identifying.* `/v1/info` reports `stub`, and that string is stamped
    into `chunks.embedding_model` for every row it produces. A corpus embedded
    by accident says so in the database, rather than looking like real vectors
    that merely retrieve badly — which is the kind of thing that costs a week.

    Its reranker scores by token overlap, so results are ordered *plausibly*.
    A constant would have made every ordering assertion in the test suite pass
    for the wrong reason.
    """

    def __init__(self, *, dimensions: int = STUB_DIMENSIONS) -> None:
        self._dimensions = dimensions

    def info(self) -> BackendInfo:
        return BackendInfo(
            embedding_model="stub",
            embedding_version="stub",
            dimensions=self._dimensions,
            tokenizer_hash="stub",
            backend="stub",
            reranker_model="stub",
        )

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        # Words and lone punctuation, plus two for the special tokens a real
        # sentencepiece model would add. Not accurate — it is not pretending to
        # be — but it is monotonic in length, which is all the batching and
        # limit logic under test actually depends on.
        return [len(_WORD.findall(text)) + 2 for text in texts]

    def embed(self, texts: Sequence[str], *, mode: str) -> list[RawEmbedding]:
        # `mode` is accepted and ignored, exactly as BGE-M3 does.
        del mode
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> RawEmbedding:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        # Stretch the digest to the required width deterministically. Values are
        # centred on zero so the vectors are not all crowded into one orthant,
        # which would make every cosine similarity ~1 and hide ordering bugs.
        dense: list[float] = []
        counter = 0
        while len(dense) < self._dimensions:
            block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            dense.extend((byte - 127.5) / 127.5 for byte in block)
            counter += 1
        del dense[self._dimensions :]

        tokens = _WORD.findall(text.lower())
        # Term frequency over a stable pseudo-vocabulary. Sorted and deduplicated
        # because a sparse vector with repeated indices is not well defined and
        # Qdrant would reject it.
        weights: dict[int, float] = {}
        for token in tokens:
            token_id = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big")
            weights[token_id % 250_000] = weights.get(token_id % 250_000, 0.0) + 1.0
        indices = sorted(weights)

        return RawEmbedding(
            dense=normalise_l2(dense),
            sparse_indices=tuple(indices),
            sparse_values=tuple(weights[index] for index in indices),
        )

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        query_tokens = set(_WORD.findall(query.lower()))
        scores: list[float] = []
        for passage in passages:
            passage_tokens = set(_WORD.findall(passage.lower()))
            union = query_tokens | passage_tokens
            overlap = len(query_tokens & passage_tokens) / len(union) if union else 0.0
            # Mapped onto a logit-like range so the shape matches a real
            # cross-encoder's unbounded output rather than a tidy 0..1.
            scores.append(overlap * 10.0 - 5.0)
        return scores


def build_backend(settings: Settings) -> InferenceBackend:
    """Construct the configured backend.

    The FlagEmbedding import is inside the branch on purpose: importing it pulls
    in torch and CUDA initialisation, which must not happen in a process that
    was configured not to use them — including every CI run of this service's
    own tests.
    """
    from model_service.settings import Backend

    if settings.backend is Backend.STUB:
        return StubBackend()

    from model_service.flag import FlagBackend

    return FlagBackend(settings)
