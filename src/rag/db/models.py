"""ORM models.

`rag.db` may import `rag.domain` (the contract only forbids the reverse), so
each model carries a `to_domain()` returning the pure dataclass. Conversion
lives here, next to the columns, rather than in a separate mapper module that
would drift out of sync.

Enum columns use `values_callable` so Postgres stores the enum *value*
(`"embedding"`) rather than the Python member *name* (`"EMBEDDING"`). Without
it, the database contents disagree with every JSON payload and log line the
system emits, and hand-written SQL silently matches nothing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from rag.db.base import Base, TenantScopedMixin, TimestampMixin, UUIDPrimaryKeyMixin
from rag.domain import models as domain
from rag.domain.enums import (
    DocumentStatus,
    JobKind,
    JobStatus,
    PrincipalType,
    Role,
    TenantStatus,
    UserStatus,
)

__all__ = [
    "TENANT_SCOPED_TABLES",
    "ApiKeyORM",
    "ChunkORM",
    "CollectionORM",
    "DocumentORM",
    "DocumentPermissionORM",
    "GroupMemberORM",
    "GroupORM",
    "JobORM",
    "RefreshTokenORM",
    "TenantORM",
    "UserORM",
]


def _pg_enum(enum_type: type, name: str) -> Enum:
    """Native Postgres enum storing member values, not member names."""
    return Enum(
        enum_type,
        name=name,
        native_enum=True,
        values_callable=lambda members: [member.value for member in members],
    )


class TenantORM(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "tenants"

    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[TenantStatus] = mapped_column(
        _pg_enum(TenantStatus, "tenant_status"),
        nullable=False,
        default=TenantStatus.ACTIVE,
    )
    settings: Mapped[dict[str, Any]] = mapped_column(nullable=False, server_default=text("'{}'"))

    def to_domain(self) -> domain.Tenant:
        return domain.Tenant(
            id=self.id,
            slug=self.slug,
            name=self.name,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            settings=dict(self.settings or {}),
        )


class UserORM(UUIDPrimaryKeyMixin, TenantScopedMixin, TimestampMixin, Base):
    __tablename__ = "users"
    __table_args__ = (
        # Per tenant, not global. A global unique constraint would leak whether
        # an address exists in another customer's organisation, and would stop
        # one person holding accounts in two tenants.
        UniqueConstraint("tenant_id", "email", name="uq_users_tenant_id_email"),
    )

    email: Mapped[str] = mapped_column(String(320), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False, server_default=text("''"))
    #: Nullable so SSO-provisioned users (M2) exist without a local password.
    password_hash: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[Role] = mapped_column(
        _pg_enum(Role, "user_role"), nullable=False, default=Role.MEMBER
    )
    status: Mapped[UserStatus] = mapped_column(
        _pg_enum(UserStatus, "user_status"), nullable=False, default=UserStatus.INVITED
    )
    last_login_at: Mapped[datetime | None] = mapped_column()

    #: Access tokens issued before this instant are rejected. Set on password
    #: change, forced logout, and admin lockout. This is how a stateless token
    #: becomes revocable without a denylist: the user row is loaded on every
    #: request anyway, so the check is free (docs/adr/0007).
    tokens_valid_after: Mapped[datetime | None] = mapped_column()

    def to_domain(self) -> domain.User:
        return domain.User(
            id=self.id,
            tenant_id=self.tenant_id,
            email=self.email,
            full_name=self.full_name,
            role=self.role,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            last_login_at=self.last_login_at,
            tokens_valid_after=self.tokens_valid_after,
        )


class ApiKeyORM(UUIDPrimaryKeyMixin, TenantScopedMixin, TimestampMixin, Base):
    """Machine credentials (docs/adr/0007).

    Under row-level security like every other table holding customer data. That
    is possible — despite this being the table the *authentication* lookup reads
    — because the key's own tenant segment binds the scope before the lookup
    runs. Exempting it the way `jobs` is exempt was rejected: unlike `jobs`,
    this table has management endpoints, so it is exactly the code most in need
    of the backstop.
    """

    __tablename__ = "api_keys"
    __table_args__ = (
        # The authentication lookup, in one index hit. Scoped by tenant because
        # RLS has already narrowed the visible rows and a composite index lets
        # the planner use both predicates.
        UniqueConstraint("tenant_id", "secret_hash", name="uq_api_keys_tenant_id_secret_hash"),
        Index("ix_api_keys_user_id", "user_id"),
    )

    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    #: Human label, chosen by whoever created the key ("ci-deploy", "laptop").
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    #: First few characters of the secret, kept in clear so a key is
    #: recognisable in a list. The rest is unrecoverable.
    display_prefix: Mapped[str] = mapped_column(String(16), nullable=False)
    #: Hex SHA-256 of the secret. Not Argon2 — see rag.domain.credentials.
    secret_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    #: A ceiling on the owner's role, never a grant of its own.
    role: Mapped[Role] = mapped_column(
        _pg_enum(Role, "user_role"), nullable=False, default=Role.VIEWER
    )
    created_by: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    expires_at: Mapped[datetime | None] = mapped_column()
    last_used_at: Mapped[datetime | None] = mapped_column()
    revoked_at: Mapped[datetime | None] = mapped_column()

    def to_domain(self) -> domain.ApiKey:
        return domain.ApiKey(
            id=self.id,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            name=self.name,
            display_prefix=self.display_prefix,
            role=self.role,
            created_at=self.created_at,
            created_by=self.created_by,
            expires_at=self.expires_at,
            last_used_at=self.last_used_at,
            revoked_at=self.revoked_at,
        )


class RefreshTokenORM(UUIDPrimaryKeyMixin, TenantScopedMixin, Base):
    """Rotating refresh tokens with family-level reuse detection."""

    __tablename__ = "refresh_tokens"
    __table_args__ = (
        UniqueConstraint("tenant_id", "token_hash", name="uq_refresh_tokens_tenant_id_token_hash"),
        Index("ix_refresh_tokens_family_id", "family_id"),
        Index("ix_refresh_tokens_user_id", "user_id"),
        # Supports the housekeeping sweep that deletes expired rows. Without it
        # this table grows without bound: one row per login, forever.
        Index("ix_refresh_tokens_expires_at", "expires_at"),
    )

    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Shared by every token in one rotation lineage. Replaying a spent token
    #: revokes the family, because two parties holding one single-use token
    #: means one of them stole it.
    family_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)

    issued_at: Mapped[datetime] = mapped_column(server_default=text("now()"), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(nullable=False)
    used_at: Mapped[datetime | None] = mapped_column()
    revoked_at: Mapped[datetime | None] = mapped_column()
    replaced_by: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("refresh_tokens.id", ondelete="SET NULL")
    )

    def to_domain(self) -> domain.RefreshToken:
        return domain.RefreshToken(
            id=self.id,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            family_id=self.family_id,
            issued_at=self.issued_at,
            expires_at=self.expires_at,
            used_at=self.used_at,
            revoked_at=self.revoked_at,
            replaced_by=self.replaced_by,
        )


class GroupORM(UUIDPrimaryKeyMixin, TenantScopedMixin, TimestampMixin, Base):
    __tablename__ = "groups"
    __table_args__ = (UniqueConstraint("tenant_id", "slug", name="uq_groups_tenant_id_slug"),)

    slug: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    def to_domain(self) -> domain.Group:
        return domain.Group(
            id=self.id,
            tenant_id=self.tenant_id,
            slug=self.slug,
            name=self.name,
            created_at=self.created_at,
        )


class GroupMemberORM(Base):
    """Join table. Composite primary key — membership has no identity of its own."""

    __tablename__ = "group_members"

    group_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    tenant_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )

    __table_args__ = (Index("ix_group_members_user_id", "user_id"),)


class CollectionORM(UUIDPrimaryKeyMixin, TenantScopedMixin, TimestampMixin, Base):
    __tablename__ = "collections"
    __table_args__ = (UniqueConstraint("tenant_id", "slug", name="uq_collections_tenant_id_slug"),)

    slug: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    def to_domain(self) -> domain.Collection:
        return domain.Collection(
            id=self.id,
            tenant_id=self.tenant_id,
            slug=self.slug,
            name=self.name,
            description=self.description,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class DocumentORM(UUIDPrimaryKeyMixin, TenantScopedMixin, TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        # Idempotency: re-uploading identical bytes into the same tenant is a
        # no-op rather than a duplicate, which prevents double-billing the
        # embedding cost and duplicate chunks polluting retrieval.
        UniqueConstraint("tenant_id", "content_hash", name="uq_documents_tenant_id_content_hash"),
        CheckConstraint("size_bytes >= 0", name="size_bytes_non_negative"),
        CheckConstraint("version >= 1", name="version_positive"),
        # GIN over the ACL array makes `acl_principals && :caller_principals`
        # an index scan. This is the exact operator the retrieval path uses, and
        # the SQL counterpart of Qdrant's `match_any`.
        Index("ix_documents_acl_principals", "acl_principals", postgresql_using="gin"),
        Index("ix_documents_tenant_id_status", "tenant_id", "status"),
        Index("ix_documents_collection_id", "collection_id"),
    )

    collection_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("collections.id", ondelete="CASCADE"), nullable=False
    )
    uploaded_by: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )

    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    #: SHA-256 of the raw bytes, hex encoded.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)

    status: Mapped[DocumentStatus] = mapped_column(
        _pg_enum(DocumentStatus, "document_status"),
        nullable=False,
        default=DocumentStatus.UPLOADED,
    )
    status_reason: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    indexed_at: Mapped[datetime | None] = mapped_column()
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))

    #: Materialised projection of `document_permissions`. Denormalised on
    #: purpose: this is the form shipped into the vector-store payload, where a
    #: join is not available.
    acl_principals: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'")
    )
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", nullable=False, server_default=text("'{}'")
    )

    def to_domain(self) -> domain.Document:
        return domain.Document(
            id=self.id,
            tenant_id=self.tenant_id,
            collection_id=self.collection_id,
            title=self.title,
            source_uri=self.source_uri,
            content_hash=self.content_hash,
            mime_type=self.mime_type,
            size_bytes=self.size_bytes,
            status=self.status,
            acl_principals=tuple(self.acl_principals or ()),
            created_at=self.created_at,
            updated_at=self.updated_at,
            uploaded_by=self.uploaded_by,
            status_reason=self.status_reason,
            page_count=self.page_count,
            indexed_at=self.indexed_at,
            version=self.version,
            metadata=dict(self.doc_metadata or {}),
        )


class DocumentPermissionORM(Base):
    """Normalised ACL grants — the source of truth `acl_principals` projects from.

    Kept because the array alone cannot answer "which documents can this group
    see?", cannot be audited row by row, and cannot record *who* granted access
    or when. The array exists for the read path; this exists for management.
    """

    __tablename__ = "document_permissions"

    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    principal_type: Mapped[PrincipalType] = mapped_column(
        _pg_enum(PrincipalType, "principal_type"), primary_key=True
    )
    principal_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    granted_by: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )

    __table_args__ = (
        Index("ix_document_permissions_principal", "tenant_id", "principal_type", "principal_id"),
    )


class ChunkORM(UUIDPrimaryKeyMixin, TenantScopedMixin, Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_id_ordinal"),
        CheckConstraint("char_end >= char_start", name="char_range_ordered"),
        CheckConstraint("ordinal >= 0", name="ordinal_non_negative"),
        Index("ix_chunks_acl_principals", "acl_principals", postgresql_using="gin"),
        Index("ix_chunks_document_id", "document_id"),
        # Supports the M4/M5 backfill: "which chunks still carry the old model?"
        Index("ix_chunks_embedding_model", "tenant_id", "embedding_model"),
    )

    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text_content: Mapped[str] = mapped_column("text", Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    char_start: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    char_end: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))

    #: Recorded from the first migration so an embedding-model change becomes a
    #: backfill-and-cut-over rather than a drop-and-rebuild.
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    embedding_version: Mapped[str | None] = mapped_column(String(64))

    #: Copied from the parent document so the retrieval path needs no join.
    acl_principals: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'")
    )
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", nullable=False, server_default=text("'{}'")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=text("now()"), nullable=False)

    def to_domain(self) -> domain.Chunk:
        return domain.Chunk(
            id=self.id,
            tenant_id=self.tenant_id,
            document_id=self.document_id,
            ordinal=self.ordinal,
            text=self.text_content,
            token_count=self.token_count,
            char_start=self.char_start,
            char_end=self.char_end,
            created_at=self.created_at,
            embedding_model=self.embedding_model,
            embedding_version=self.embedding_version,
            acl_principals=tuple(self.acl_principals or ()),
            metadata=dict(self.chunk_metadata or {}),
        )


class JobORM(UUIDPrimaryKeyMixin, TenantScopedMixin, TimestampMixin, Base):
    """The work queue. Deliberately *not* under row-level security.

    Workers poll across tenants by design, then scope themselves to the tenant
    of each claimed job. No HTTP endpoint exposes this table, so the exemption
    does not widen the API's attack surface.
    """

    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint("attempts >= 0", name="attempts_non_negative"),
        CheckConstraint("max_attempts >= 1", name="max_attempts_positive"),
        # Partial index over exactly the claim query's predicate and ordering.
        # A full index would also cover the millions of finished rows nobody
        # queries; this one stays small enough to remain in cache.
        Index(
            "ix_jobs_claimable",
            "priority",
            "run_after",
            postgresql_where=text("status = 'queued'"),
        ),
        # Supports reaping jobs whose worker died mid-flight.
        Index(
            "ix_jobs_running_locked_at",
            "locked_at",
            postgresql_where=text("status = 'running'"),
        ),
    )

    kind: Mapped[JobKind] = mapped_column(_pg_enum(JobKind, "job_kind"), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        _pg_enum(JobStatus, "job_status"), nullable=False, default=JobStatus.QUEUED
    )
    payload: Mapped[dict[str, Any]] = mapped_column(nullable=False, server_default=text("'{}'"))

    #: Higher runs first.
    priority: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("5"))
    #: Backoff and scheduling in one column: a job is claimable once now() passes it.
    run_after: Mapped[datetime] = mapped_column(server_default=text("now()"), nullable=False)

    locked_by: Mapped[str | None] = mapped_column(String(128))
    locked_at: Mapped[datetime | None] = mapped_column()
    last_error: Mapped[str | None] = mapped_column(Text)

    def to_domain(self) -> domain.Job:
        return domain.Job(
            id=self.id,
            tenant_id=self.tenant_id,
            kind=self.kind,
            status=self.status,
            payload=dict(self.payload or {}),
            priority=self.priority,
            attempts=self.attempts,
            max_attempts=self.max_attempts,
            run_after=self.run_after,
            created_at=self.created_at,
            updated_at=self.updated_at,
            locked_by=self.locked_by,
            locked_at=self.locked_at,
            last_error=self.last_error,
        )


#: Tables that hold customer data and therefore carry an RLS policy.
#: `jobs` is intentionally absent — see `JobORM`. Migration 0001 iterates this
#: list, so adding a tenant-scoped table here is what turns its policy on.
TENANT_SCOPED_TABLES: tuple[str, ...] = (
    "users",
    "api_keys",
    "refresh_tokens",
    "groups",
    "group_members",
    "collections",
    "documents",
    "document_permissions",
    "chunks",
)
