"""Domain entities.

Plain frozen dataclasses with no ORM, no framework, and no I/O. The persistence
models live in `rag.db.models`; repositories convert at the boundary.

Why pay for two representations instead of passing ORM objects around:

* **Testability.** A service test constructs a `Document(...)` in a line, with
  no session, no event loop, and no database. Handing services ORM instances
  means every service test needs a live session or a mock of one.
* **No lazy-load landmines.** A detached ORM instance raises on attribute access
  when a relationship was not eagerly loaded — a failure that appears far from
  its cause, often only under production access patterns. A dataclass either has
  the field or does not, and you find out at construction.
* **The boundary stays honest.** `rag.domain` cannot import SQLAlchemy (enforced
  by import-linter), so this is not a stylistic preference; it is what makes the
  contract checkable.

These are lean by design: they carry what callers need, not every column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from rag.domain.enums import (
    DocumentStatus,
    JobKind,
    JobStatus,
    Role,
    TenantStatus,
    UserStatus,
)

__all__ = [
    "Chunk",
    "Collection",
    "Document",
    "Group",
    "Job",
    "NewChunk",
    "Tenant",
    "User",
]


@dataclass(frozen=True, slots=True)
class Tenant:
    """An isolated customer organisation. The root of every access decision."""

    id: UUID
    slug: str
    name: str
    status: TenantStatus
    created_at: datetime
    updated_at: datetime
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status is TenantStatus.ACTIVE


@dataclass(frozen=True, slots=True)
class User:
    """A person within a tenant.

    `email` is unique *per tenant*, not globally: the same person may hold
    accounts in two customer organisations, and a global constraint would let
    one tenant discover whether an address exists in another.
    """

    id: UUID
    tenant_id: UUID
    email: str
    full_name: str
    role: Role
    status: UserStatus
    created_at: datetime
    updated_at: datetime
    last_login_at: datetime | None = None

    @property
    def can_authenticate(self) -> bool:
        return self.status is UserStatus.ACTIVE


@dataclass(frozen=True, slots=True)
class Group:
    """A named set of users, used to grant document access in bulk.

    Flat in M1 — no nesting. Nested groups need recursive expansion when
    building a caller's principal set; a Postgres recursive CTE handles it
    cheaply, but it is scope we do not need yet. Recorded in docs/FUTURE.md.
    """

    id: UUID
    tenant_id: UUID
    slug: str
    name: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Collection:
    """A named grouping of documents — the unit users organise and filter by."""

    id: UUID
    tenant_id: UUID
    slug: str
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class Document:
    """An ingested source document and its position in the pipeline."""

    id: UUID
    tenant_id: UUID
    collection_id: UUID
    title: str
    source_uri: str
    content_hash: str
    mime_type: str
    size_bytes: int
    status: DocumentStatus
    acl_principals: tuple[str, ...]
    created_at: datetime
    updated_at: datetime
    uploaded_by: UUID | None = None
    status_reason: str | None = None
    page_count: int | None = None
    indexed_at: datetime | None = None
    #: Bumped on every re-ingest of the same logical document, so chunks from a
    #: superseded version can be identified and purged.
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_queryable(self) -> bool:
        return self.status is DocumentStatus.READY


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of text, and the row a vector points back to.

    `tenant_id` is denormalised onto this table rather than reached through
    `document_id`. That is a deliberate duplication: it lets the RLS policy and
    every access filter apply directly, with no join, on the hottest read path
    in the system.

    `embedding_model` and `embedding_version` are recorded per chunk from the
    very first migration. Without them, changing embedding model means dropping
    and rebuilding the entire index; with them, we re-embed into a new named
    vector, backfill, and cut over with no downtime.
    """

    id: UUID
    tenant_id: UUID
    document_id: UUID
    ordinal: int
    text: str
    token_count: int
    char_start: int
    char_end: int
    created_at: datetime
    embedding_model: str | None = None
    embedding_version: str | None = None
    #: Denormalised from the parent document so retrieval never needs a join.
    acl_principals: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NewChunk:
    """A chunk about to be persisted, before the database assigns it identity.

    Separate from `Chunk` because a chunker produces text and offsets — it has
    no id, no timestamp, and no opinion about tenancy or access. Tenant and ACL
    are inherited from the parent document at insert time, which is exactly the
    invariant we want enforced in one place rather than trusted at every call
    site.
    """

    ordinal: int
    text: str
    token_count: int = 0
    char_start: int = 0
    char_end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Job:
    """A unit of background work.

    Lives in Postgres rather than a broker so that creating a document and
    enqueuing its ingestion commit in one transaction — see docs/adr/0002.
    """

    id: UUID
    tenant_id: UUID
    kind: JobKind
    status: JobStatus
    payload: dict[str, Any]
    priority: int
    attempts: int
    max_attempts: int
    run_after: datetime
    created_at: datetime
    updated_at: datetime
    locked_by: str | None = None
    locked_at: datetime | None = None
    last_error: str | None = None

    @property
    def is_exhausted(self) -> bool:
        """Retries used up — the next failure is terminal."""
        return self.attempts >= self.max_attempts
