"""What goes into the vector index, and what comes back out.

These types are the vocabulary of the retrieval path. Like `rag.domain.embedding`
they know nothing about Qdrant — swapping to pgvector (ADR-0001's stated escape
hatch) changes an adapter, not this module.

The payload is deliberately thin
--------------------------------
`VectorPoint` carries only what a *filter* needs: the tenant, the ACL, and the
identifiers a caller might narrow by. It does **not** carry the chunk's text.

That is a security property, not a storage saving. Results are hydrated from
Postgres by `ChunkRepository.get_many`, which re-applies the access filter — so
if the index ever drifts and a deleted or re-permissioned document's vectors
survive, the match produces no row and the result disappears. An orphaned vector
degrades recall; it cannot disclose anything. Copying text into the payload
would trade that away for one saved query, and would put every customer's
content in a second system that has no row-level security.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from rag.domain.embedding import Embedding

__all__ = ["ScoredChunk", "SearchHit", "VectorPoint"]


@dataclass(frozen=True, slots=True)
class VectorPoint:
    """One chunk, ready to be written to the index.

    `chunk_id` is also the point id in the store. Deterministic on purpose
    (ADR-0001): re-indexing the same chunk overwrites rather than duplicating,
    which is what makes the whole write path safe to retry after a crash without
    any bookkeeping about what got as far as being written.
    """

    chunk_id: UUID
    tenant_id: UUID
    document_id: UUID
    ordinal: int
    embedding: Embedding
    #: Copied from the chunk row, which copied it from the document. This is the
    #: value the pre-filter matches against (docs/adr/0006).
    acl_principals: tuple[str, ...] = ()
    collection_id: UUID | None = None
    #: Stamped so a model migration can find rows still carrying the old one.
    embedding_model: str | None = None
    embedding_version: str | None = None


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A match, as the vector store knows it: an id and a score.

    Nothing else, because nothing else is in the index. `score` is comparable
    only within one result set — cosine similarity here, but a fused rank in M6
    — so it is for ordering and never for thresholding against a constant.
    """

    chunk_id: UUID
    score: float
    document_id: UUID


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A hit after hydration from Postgres: the text, and where it came from.

    Assembled by `rag.services.retrieval`, which is also where the second access
    check happens. Two separate objects rather than one mutable one, so it is
    impossible to hold something that has a score but has not yet been
    authorised.
    """

    chunk_id: UUID
    document_id: UUID
    document_title: str
    ordinal: int
    text: str
    score: float
    char_start: int
    char_end: int
    metadata: dict[str, object] = field(default_factory=dict)
