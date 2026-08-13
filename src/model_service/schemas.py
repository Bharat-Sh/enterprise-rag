"""The wire contract, server side.

The client defines its own copy of these in `rag.adapters.models.schemas`. That
duplication is deliberate: the two processes deploy independently, so the client
must tolerate a server that has grown a field it does not know about, and
sharing one class would quietly make every server-side addition a client-side
breaking change at import time.

What stops the two copies drifting is not discipline, it is
`tests/unit/test_model_contract.py`, which round-trips every server response
model through the corresponding client model. That test is the entire reason
both ends live in one repository (docs/adr/0011).

Field naming note: `embedding_model` rather than `model`, because pydantic
reserves the `model_` prefix and a bare `model` field earns a warning in every
process that imports this.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "EmbedRequest",
    "EmbedResponse",
    "EmbeddingPayload",
    "InfoResponse",
    "RerankRequest",
    "RerankResponse",
    "RerankScore",
    "SparsePayload",
]


class EmbedRequest(BaseModel):
    """Texts to embed."""

    texts: list[str]
    #: BGE-M3 ignores this — it uses no asymmetric instruction prefix — but the
    #: contract carries it so a hosted provider that *does* need it can be
    #: substituted without a client change. See `rag.domain.embedding.EmbedMode`.
    mode: Literal["query", "passage"] = "passage"


class SparsePayload(BaseModel):
    """Lexical weights as parallel arrays — Qdrant's sparse-vector shape.

    Not a `{token_id: weight}` object: JSON keys are strings, which roughly
    doubles the payload and forces an int cast on every element read.
    """

    indices: list[int]
    values: list[float]


class EmbeddingPayload(BaseModel):
    """One text's vectors.

    `dense` is L2-normalised **by the service**, so cosine similarity equals a
    dot product and the vector store can use the cheaper metric. Normalising
    here rather than at each call site means no caller can forget and silently
    produce scores that are wrong but plausible.
    """

    dense: list[float]
    sparse: SparsePayload


class EmbedResponse(BaseModel):
    """Vectors, in the order the texts were given.

    Order is contractual — the caller pairs results back to chunk ids
    positionally. Internally the service may split a request into several
    forward passes; it reassembles before responding.

    The model identity rides along on every response rather than requiring a
    second `/v1/info` call, so the values stamped into `chunks.embedding_model`
    describe the pass that actually produced these vectors. Reading them from a
    cached `/v1/info` would attribute them to whatever was loaded at startup,
    which is a different thing after a redeploy.
    """

    embedding_model: str
    embedding_version: str
    dimensions: int
    embeddings: list[EmbeddingPayload]


class RerankRequest(BaseModel):
    """A query and the passages to score against it."""

    query: str
    passages: list[str]
    #: Trims the response only. A cross-encoder must score every candidate to
    #: know which ones win, so this saves bandwidth, never GPU time.
    top_k: int | None = Field(default=None, ge=1)


class RerankScore(BaseModel):
    """One passage's relevance.

    `index` refers to the position in the request's `passages`. The passage text
    is not echoed back: the caller already has it, and returning it would double
    the payload and put customer text on the wire a second time for nothing.
    """

    index: int
    score: float


class RerankResponse(BaseModel):
    """Scores, highest first."""

    reranker_model: str
    results: list[RerankScore]


class InfoResponse(BaseModel):
    """What is loaded, for readiness probes and model-version stamping.

    `tokenizer_hash` is what makes a chunking/embedding tokenizer mismatch
    detectable. If the worker sizes chunks with a different vocabulary than the
    service embeds with, chunks silently exceed the window and lose their tails.
    """

    embedding_model: str
    embedding_version: str
    dimensions: int
    max_sequence_tokens: int
    tokenizer_hash: str
    reranker_model: str | None = None
    #: `flag` or `stub`. Present so an accidental stub deployment is visible in
    #: a probe rather than only in the vectors it produces.
    backend: str
