"""The wire contract, client side.

A separate copy of `model_service.schemas`, deliberately. The two processes
deploy independently, so this end must keep working against a server that has
grown a field it has never heard of — pydantic ignores unknown keys by default,
which is exactly the behaviour we want and exactly what sharing one class would
take away, since then every server-side addition would become a synchronised
deploy.

Drift is caught by `tests/unit/test_model_contract.py`, which feeds each server
response model through its client counterpart. That test is why both ends live
in one repository: two repos would have made the contract unenforceable by
anything except an incident.

Only the *responses* are modelled here. Requests are built as plain dicts at the
call site — there is nothing to validate on the way out that the server will not
validate on the way in, and a second set of request classes would be two more
things to keep in step for no benefit.
"""

from __future__ import annotations

from pydantic import BaseModel

__all__ = [
    "EmbedResponseModel",
    "EmbeddingPayloadModel",
    "InfoResponseModel",
    "RerankResponseModel",
    "RerankScoreModel",
    "SparsePayloadModel",
]


class SparsePayloadModel(BaseModel):
    indices: list[int]
    values: list[float]


class EmbeddingPayloadModel(BaseModel):
    dense: list[float]
    sparse: SparsePayloadModel


class EmbedResponseModel(BaseModel):
    embedding_model: str
    embedding_version: str
    dimensions: int
    embeddings: list[EmbeddingPayloadModel]


class RerankScoreModel(BaseModel):
    index: int
    score: float


class RerankResponseModel(BaseModel):
    reranker_model: str
    results: list[RerankScoreModel]


class InfoResponseModel(BaseModel):
    embedding_model: str
    embedding_version: str
    dimensions: int
    max_sequence_tokens: int
    tokenizer_hash: str
    #: Optional on this side even though the server always sends it, so that an
    #: older service — or one with reranking disabled — parses rather than
    #: failing at the boundary with a validation error that reads like a bug.
    reranker_model: str | None = None
    backend: str = "unknown"
