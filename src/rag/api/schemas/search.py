"""Request and response shapes for `/search`."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from rag.services.retrieval import MAX_TOP_K

__all__ = ["SearchRequest", "SearchResponse", "SearchResult"]


class SearchRequest(BaseModel):
    """A query, and optional narrowing.

    Notably absent: anything identifying the *caller*. Tenant and principals
    come from the verified token and never from the body — non-negotiable #5.
    A `tenant_id` field here would be the single most dangerous thing this
    schema could contain, so it is worth saying why it is missing rather than
    leaving that to be inferred.
    """

    query: str = Field(min_length=1, max_length=4096)
    #: Bounded by `MAX_TOP_K` rather than left open: embedding, filtering,
    #: hydrating and serialising 10,000 results turns one cheap request into a
    #: very expensive one, which is an amplification shape rather than a usage
    #: pattern.
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)

    #: Narrowing only. These can shrink the result set the ACL pre-filter has
    #: already bounded; they can never widen it, which is what makes them safe
    #: to accept from a caller at all.
    collection_id: UUID | None = None
    document_ids: list[UUID] | None = Field(default=None, max_length=100)


class SearchResult(BaseModel):
    """One matching chunk."""

    chunk_id: UUID
    document_id: UUID
    document_title: str
    ordinal: int
    text: str
    #: Comparable only within one response. Cosine similarity today and a fused
    #: rank once M6 lands, so a client that thresholds this against a constant
    #: will silently break when the ranking method changes.
    score: float
    #: Character offsets into the document's extracted text. Exact, which is
    #: what a citation feature will need.
    char_start: int
    char_end: int


class SearchResponse(BaseModel):
    """Results, most relevant first."""

    query: str
    results: list[SearchResult]
    #: How many results came back, so a client need not special-case an empty
    #: list to tell "no matches" from "not permitted to see any". The two are
    #: deliberately indistinguishable — reporting that results were withheld is
    #: itself a disclosure.
    count: int
