"""Domain enumerations.

All are `StrEnum` so they serialise to readable strings in JSON, in Postgres
columns, and in log lines. Storing `"embedding"` rather than `3` costs a few
bytes and saves every future operator from looking up what `3` meant.

These are stored as native Postgres enum types. The tradeoff versus a plain
`VARCHAR` + check constraint: native enums give real type safety and a compact
on-disk representation, but adding a value requires `ALTER TYPE ... ADD VALUE`,
which cannot run inside a transaction block in older Postgres and cannot be
reordered. Since these vocabularies are small and change rarely, the safety is
worth the migration friction.
"""

from __future__ import annotations

from enum import StrEnum


class TenantStatus(StrEnum):
    """Lifecycle of a tenant account."""

    ACTIVE = "active"
    SUSPENDED = "suspended"  # billing or policy hold; reads may continue, writes stop
    DELETED = "deleted"  # tombstoned; retained for audit until purge


class UserStatus(StrEnum):
    """Lifecycle of a user account."""

    ACTIVE = "active"
    INVITED = "invited"  # created but has never authenticated
    DISABLED = "disabled"


class Role(StrEnum):
    """Tenant-wide role, ordered from most to least privileged.

    Deliberately separate from document ACLs. Roles govern *actions* ("may this
    user upload?"); ACLs govern *resources* ("may this user read that
    document?"). Collapsing the two produces a model that can express neither
    "an admin who cannot see HR documents" nor "a viewer with access to one
    confidential collection".
    """

    OWNER = "owner"  # billing, tenant deletion, role assignment
    ADMIN = "admin"  # user and collection management, ingestion
    MEMBER = "member"  # upload and query
    VIEWER = "viewer"  # query only


class PrincipalType(StrEnum):
    """The kind of subject a permission can be granted to.

    Rendered as `"<type>:<id>"` tokens (see `rag.domain.access`) so an ACL is a
    flat array of strings — the only shape a vector-store payload filter can
    evaluate without a join.
    """

    USER = "user"
    GROUP = "group"
    ROLE = "role"
    TENANT = "tenant"  # every member of the tenant


class DocumentStatus(StrEnum):
    """Position in the ingestion state machine.

    Explicit intermediate states are not bureaucracy: they make "how many
    documents are stuck in EMBEDDING?" a SQL query, they let a failure at one
    stage retry without discarding the work of earlier stages, and they turn a
    concurrency bug into a loud `InvalidStateTransitionError` instead of a
    silently corrupted index.
    """

    UPLOADED = "uploaded"  # bytes stored, nothing parsed
    QUEUED = "queued"  # job enqueued, awaiting a worker
    PARSING = "parsing"  # extracting text and structure
    CHUNKING = "chunking"  # splitting into retrievable units
    EMBEDDING = "embedding"  # computing dense + sparse vectors
    INDEXING = "indexing"  # upserting into the vector store
    READY = "ready"  # queryable
    FAILED = "failed"  # exhausted retries; `status_reason` explains why
    REINDEXING = "reindexing"  # rebuilding vectors (model or chunker change)
    DELETING = "deleting"  # purging vectors before removing rows
    DELETED = "deleted"  # tombstone

    @property
    def is_terminal(self) -> bool:
        """No further automatic transitions occur from here."""
        return self in {DocumentStatus.READY, DocumentStatus.FAILED, DocumentStatus.DELETED}

    @property
    def is_in_flight(self) -> bool:
        """A worker is, or should be, actively processing this document."""
        return self in {
            DocumentStatus.QUEUED,
            DocumentStatus.PARSING,
            DocumentStatus.CHUNKING,
            DocumentStatus.EMBEDDING,
            DocumentStatus.INDEXING,
            DocumentStatus.REINDEXING,
            DocumentStatus.DELETING,
        }


class JobKind(StrEnum):
    """What a queued job asks a worker to do."""

    INGEST_DOCUMENT = "ingest_document"
    REINDEX_DOCUMENT = "reindex_document"
    DELETE_DOCUMENT = "delete_document"


class JobStatus(StrEnum):
    """Position in the job queue lifecycle."""

    QUEUED = "queued"  # claimable
    RUNNING = "running"  # claimed by a worker
    SUCCEEDED = "succeeded"
    FAILED = "failed"  # retries exhausted — the dead-letter state
    CANCELLED = "cancelled"
