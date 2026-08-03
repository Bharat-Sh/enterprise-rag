"""Ports: the interfaces `rag.services` depends on.

`typing.Protocol`, not abstract base classes. Structural typing means an adapter
satisfies a port without importing or subclassing anything from the domain — the
dependency arrow points one way only, and a test fake is a plain class with the
right method names. With ABCs every adapter would have to import the domain and
every fake would have to inherit from it.

All are `runtime_checkable` so `isinstance` works in tests, but note that only
method *names* are checked at runtime, never signatures. mypy checks signatures
statically; that is where the real enforcement lives.

**Why `create(...)` takes fields rather than an entity.** Timestamps, defaults,
and the primary key are the database's business. A `create(entity)` API forces
the caller to invent a `created_at` before the row exists, and that invented
value then disagrees with `now()` on the server by the network round trip.
Passing fields keeps one authority for every column.

Ports added in later milestones — `VectorStore`, `EmbeddingProvider`,
`Reranker`, `LLMClient`, `Cache` — belong in this module too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from types import TracebackType
    from uuid import UUID

    from rag.domain.access import AccessFilter
    from rag.domain.enums import DocumentStatus, JobKind, JobStatus, Role, UserStatus
    from rag.domain.models import (
        Chunk,
        Collection,
        Document,
        Group,
        Job,
        NewChunk,
        Tenant,
        User,
    )

__all__ = [
    "ChunkRepository",
    "CollectionRepository",
    "DocumentRepository",
    "GroupRepository",
    "JobRepository",
    "TenantRepository",
    "UnitOfWork",
    "UserRepository",
]


@runtime_checkable
class TenantRepository(Protocol):
    """Tenant lookup. Not itself tenant-scoped — this establishes the scope."""

    async def get(self, tenant_id: UUID) -> Tenant | None: ...

    async def get_by_slug(self, slug: str) -> Tenant | None: ...

    async def create(
        self, *, slug: str, name: str, settings: dict[str, Any] | None = None
    ) -> Tenant: ...


@runtime_checkable
class UserRepository(Protocol):
    """Users within the currently scoped tenant."""

    async def get(self, user_id: UUID) -> User | None: ...

    async def get_by_email(self, email: str) -> User | None: ...

    async def create(
        self,
        *,
        tenant_id: UUID,
        email: str,
        full_name: str = "",
        role: Role | None = None,
        status: UserStatus | None = None,
        password_hash: str | None = None,
    ) -> User: ...

    async def group_ids_for(self, user_id: UUID) -> tuple[UUID, ...]:
        """Groups the user belongs to, for assembling their `AccessFilter`."""
        ...

    async def access_filter_for(self, user: User) -> AccessFilter:
        """Build the caller's complete principal set.

        Computed per request rather than stored, which is what lets a group
        membership change take effect without re-indexing a single document.
        """
        ...


@runtime_checkable
class GroupRepository(Protocol):
    async def create(self, *, tenant_id: UUID, slug: str, name: str) -> Group: ...

    async def add_member(self, *, group_id: UUID, user_id: UUID, tenant_id: UUID) -> None: ...

    async def remove_member(self, *, group_id: UUID, user_id: UUID) -> None: ...


@runtime_checkable
class CollectionRepository(Protocol):
    async def get(self, collection_id: UUID) -> Collection | None: ...

    async def list_all(self) -> Sequence[Collection]: ...

    async def create(
        self, *, tenant_id: UUID, slug: str, name: str, description: str | None = None
    ) -> Collection: ...


@runtime_checkable
class DocumentRepository(Protocol):
    """Documents within the currently scoped tenant.

    Every read takes an `AccessFilter`. That is structural, not conventional:
    there is no method that returns documents without one.
    """

    async def get(self, document_id: UUID, access: AccessFilter) -> Document | None: ...

    async def get_by_content_hash(self, content_hash: str) -> Document | None:
        """Idempotency probe.

        Takes no `AccessFilter` because it runs *before* a document exists to be
        authorised against; tenant isolation still applies via row-level
        security. Re-uploading identical bytes must be a no-op, not a duplicate.
        """
        ...

    async def list_for(
        self,
        access: AccessFilter,
        *,
        collection_id: UUID | None = None,
        status: DocumentStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Document]: ...

    async def create(
        self,
        *,
        tenant_id: UUID,
        collection_id: UUID,
        title: str,
        source_uri: str,
        content_hash: str,
        mime_type: str,
        size_bytes: int,
        uploaded_by: UUID | None = None,
        acl_principals: Sequence[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> Document: ...

    async def transition_status(
        self,
        document_id: UUID,
        *,
        expected: DocumentStatus,
        target: DocumentStatus,
        reason: str | None = None,
    ) -> Document:
        """Move to `target`, asserting the row is currently `expected`.

        A compare-and-set, not a read-then-write. That is what makes concurrent
        workers safe: the loser of a race finds the row no longer in `expected`
        and raises, instead of overwriting the winner's progress.
        """
        ...

    async def set_acl(self, document_id: UUID, principal_tokens: Sequence[str]) -> None:
        """Replace the ACL and reproject it onto the document's chunks.

        One call, because the array on `chunks` is a denormalisation of the one
        on `documents`; letting them diverge is how a revoked permission keeps
        returning results.
        """
        ...


@runtime_checkable
class ChunkRepository(Protocol):
    async def add_many(self, document_id: UUID, chunks: Sequence[NewChunk]) -> int:
        """Insert chunks, inheriting tenant and ACL from the parent document."""
        ...

    async def list_for_document(self, document_id: UUID) -> Sequence[Chunk]: ...

    async def delete_for_document(self, document_id: UUID) -> int: ...

    async def get_many(self, chunk_ids: Sequence[UUID], access: AccessFilter) -> Sequence[Chunk]:
        """Hydrate chunks by id after a vector search, re-checking access.

        The vector store already pre-filtered, so this is defence in depth: an
        index that has drifted from Postgres must not surface text the caller
        may not read.
        """
        ...

    async def set_embedding_metadata(
        self, document_id: UUID, *, model: str, version: str
    ) -> int: ...


@runtime_checkable
class JobRepository(Protocol):
    """The Postgres-backed work queue (docs/adr/0002)."""

    async def enqueue(
        self,
        *,
        tenant_id: UUID,
        kind: JobKind,
        payload: dict[str, Any] | None = None,
        priority: int = 0,
        max_attempts: int = 5,
        run_after: datetime | None = None,
    ) -> Job: ...

    async def claim(
        self, *, worker_id: str, kinds: Sequence[JobKind] | None = None, limit: int = 1
    ) -> Sequence[Job]:
        """Atomically claim up to `limit` runnable jobs.

        `FOR UPDATE SKIP LOCKED`, so concurrent workers take disjoint sets
        without blocking one another.
        """
        ...

    async def complete(
        self, job_id: UUID, status: JobStatus, *, error: str | None = None
    ) -> None: ...

    async def reschedule(self, job_id: UUID, *, run_after: datetime, error: str) -> None:
        """Return a job to the queue after a retryable failure."""
        ...

    async def reap_stalled(self, *, older_than: datetime) -> int:
        """Requeue jobs whose worker died holding them.

        Without this a crash strands in-flight jobs in RUNNING for ever. The
        visibility timeout is what makes at-least-once delivery real rather
        than aspirational.
        """
        ...


@runtime_checkable
class UnitOfWork(Protocol):
    """A transaction boundary exposing the repositories that share it.

    Services depend on this rather than on a session, so a service decides
    *what* to persist and when it commits, never *how*. One `async with` is one
    transaction.
    """

    tenants: TenantRepository
    users: UserRepository
    groups: GroupRepository
    collections: CollectionRepository
    documents: DocumentRepository
    chunks: ChunkRepository
    jobs: JobRepository

    async def __aenter__(self) -> UnitOfWork: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...

    async def scope_to_tenant(self, tenant_id: UUID | None) -> None:
        """Bind the tenant for row-level security within this transaction.

        Implementations issue `SET LOCAL`, so the scope reverts when the
        transaction ends. With a pooled connection a session-level setting would
        leak one tenant's scope into whichever request borrows it next.
        """
        ...
