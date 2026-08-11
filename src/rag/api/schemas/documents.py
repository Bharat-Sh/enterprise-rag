"""Wire models for documents and collections."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from rag.domain.enums import DocumentStatus

__all__ = [
    "AclUpdateRequest",
    "ChunkResponse",
    "CollectionCreateRequest",
    "CollectionResponse",
    "DocumentResponse",
]


class CollectionCreateRequest(BaseModel):
    slug: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)


class CollectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    slug: str
    name: str
    description: str | None = None
    created_at: datetime


class DocumentResponse(BaseModel):
    """A document as the API discloses it.

    `blob_key` is deliberately absent. It is an internal storage location whose
    layout belongs to whichever adapter is configured; publishing it would make
    a storage change a breaking API change, and would hand a caller a path.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    collection_id: UUID
    title: str
    source_uri: str
    content_hash: str
    mime_type: str
    size_bytes: int
    status: DocumentStatus
    #: Populated once ingestion fails, and the only place the reason surfaces.
    status_reason: str | None = None
    acl_principals: tuple[str, ...] = ()
    created_at: datetime
    updated_at: datetime
    indexed_at: datetime | None = None
    page_count: int | None = None
    version: int = 1


class ChunkResponse(BaseModel):
    """A chunk, for inspecting how a document was split.

    Offsets are into the *extracted* text, not the original bytes — the two
    differ for every format that is not plain text.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    ordinal: int
    text: str
    token_count: int
    char_start: int
    char_end: int


class AclUpdateRequest(BaseModel):
    """Replace a document's ACL.

    Replace rather than patch: an add/remove API makes "who can see this?" a
    question you answer by replaying a history, and makes a lost update
    invisible. The caller sends the set it wants.
    """

    principals: list[str] = Field(
        min_length=1,
        max_length=256,
        examples=[["tenant:0195...", "group:0195...", "role:admin"]],
        description=(
            "Principal tokens, `<type>:<id>`. A `tenant:` token naming another tenant is rejected."
        ),
    )
