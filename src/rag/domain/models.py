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
    "ApiKey",
    "Chunk",
    "Collection",
    "Document",
    "Group",
    "Job",
    "NewChunk",
    "RefreshToken",
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
    #: Access tokens issued before this instant are rejected. Carried on the
    #: entity — unlike `password_hash`, which is not — because it is read on
    #: *every* authenticated request, and a second query per request to fetch one
    #: nullable timestamp is a real cost. It is a timestamp, not a credential.
    tokens_valid_after: datetime | None = None

    @property
    def can_authenticate(self) -> bool:
        return self.status is UserStatus.ACTIVE

    def accepts_token_issued_at(self, issued_at: datetime) -> bool:
        """Whether a token minted at `issued_at` is still honoured.

        This is what makes a stateless token revocable. A password change, a
        forced logout, or a detected refresh-token theft moves the watermark
        forward and every outstanding token falls behind it at once.

        **The watermark is truncated to whole seconds before comparing.** A JWT
        `iat` is a NumericDate (RFC 7519) and therefore has one-second
        resolution, while the watermark comes from `datetime.now()` and has
        microseconds. Comparing them directly rejects the replacement token
        issued *by* the revocation — a password change would hand back a pair
        that was already dead, which is precisely the flow the feature exists to
        support.

        The cost is that the boundary is one second wide: a token minted in the
        same second as the revocation survives. That is inherent to the
        resolution of `iat`, not a choice, and one second is well inside the
        window an attacker would need anyway.
        """
        if self.tokens_valid_after is None:
            return True
        return issued_at >= self.tokens_valid_after.replace(microsecond=0)


@dataclass(frozen=True, slots=True)
class ApiKey:
    """A machine credential belonging to a user (docs/adr/0007).

    The secret itself is absent by construction — the row holds a SHA-256 and a
    display prefix, so this object can be returned from any endpoint without
    thinking about it.

    **A key belongs to a user rather than standing alone.** A key with no user
    could only ever match `role:` and `tenant:` document grants, because
    `group_members` keys on `users.id` and the ACL model (docs/adr/0006) has no
    service principal type. A machine credential that can read everything shared
    organisation-wide but nothing shared with a team is the wrong default. The
    price is that a key stops working when its owner is disabled — an ops
    complaint, and also the correct behaviour.

    `role` is a **ceiling**, never a grant: the request runs as
    `less_privileged_of(user.role, key.role)`.
    """

    id: UUID
    tenant_id: UUID
    user_id: UUID
    name: str
    display_prefix: str
    role: Role
    created_at: datetime
    created_by: UUID | None = None
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None

    def is_usable_at(self, now: datetime) -> bool:
        """Whether the key may authenticate a request at `now`."""
        if self.revoked_at is not None:
            return False
        return self.expires_at is None or self.expires_at > now


@dataclass(frozen=True, slots=True)
class RefreshToken:
    """A single-use handle for obtaining a new access token.

    Opaque and database-backed rather than a JWT. Revoking a JWT refresh token
    needs a denylist lookup, so you pay for statefulness either way — and this
    way the token's validity *is* a row, which makes revocation exact instead of
    eventually consistent.

    `family_id` ties a rotation lineage together. Presenting a token that has
    already been used means two parties hold it, which means one of them stole
    it; the response is to revoke the whole family rather than to guess which.
    """

    id: UUID
    tenant_id: UUID
    user_id: UUID
    family_id: UUID
    issued_at: datetime
    expires_at: datetime
    used_at: datetime | None = None
    revoked_at: datetime | None = None
    replaced_by: UUID | None = None

    def is_usable_at(self, now: datetime) -> bool:
        return self.used_at is None and self.revoked_at is None and self.expires_at > now


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
    #: Provenance: where the bytes came from, for a human reading an audit log.
    source_uri: str
    #: Location in the blob store. Opaque — its layout belongs to the adapter,
    #: and a caller that parses it has taken a dependency on the storage
    #: backend (docs/adr/0009).
    blob_key: str
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
