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
    from datetime import datetime, timedelta
    from types import TracebackType
    from uuid import UUID

    from rag.domain.access import AccessFilter
    from rag.domain.credentials import AccessToken, TokenClaims
    from rag.domain.enums import DocumentStatus, JobKind, JobStatus, Role, UserStatus
    from rag.domain.models import (
        ApiKey,
        Chunk,
        Collection,
        Document,
        Group,
        Job,
        NewChunk,
        RefreshToken,
        Tenant,
        User,
    )
    from rag.domain.ratelimit import RateLimitDecision, RateLimitPolicy

__all__ = [
    "ApiKeyRepository",
    "ChunkRepository",
    "CollectionRepository",
    "DocumentRepository",
    "GroupRepository",
    "JobRepository",
    "PasswordHasher",
    "RateLimiter",
    "RefreshTokenRepository",
    "TenantRepository",
    "TokenIssuer",
    "TokenVerifier",
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

    async def access_filter_for(self, user: User, *, role: Role | None = None) -> AccessFilter:
        """Build the caller's complete principal set.

        Computed per request rather than stored, which is what lets a group
        membership change take effect without re-indexing a single document.

        `role` overrides `user.role` so an API key's ceiling reaches the filter.
        Without it a key scoped down to viewer would still carry the
        `role:admin` principal and match admin-granted document ACLs.
        """
        ...

    async def get_password_hash(self, user_id: UUID) -> str | None:
        """Read the stored password hash.

        Separate from `get()` so the hash never rides along on the `User` the
        rest of the system passes around, gets logged, or is serialised into a
        response by an over-eager `model_validate`.
        """
        ...

    async def set_password_hash(self, user_id: UUID, password_hash: str) -> None: ...

    async def touch_last_login(self, user_id: UUID, *, at: datetime) -> None: ...

    async def invalidate_tokens_before(self, user_id: UUID, *, at: datetime) -> None:
        """Reject every access token issued before `at`.

        The revocation mechanism for stateless tokens. A denylist was rejected:
        it needs infrastructure we do not have until M9, it is eventually
        consistent, and it is a network hop to answer a question that a column
        on a row we already load answers exactly. The watermark is read back on
        `User` itself, so checking it costs no extra query.
        """
        ...

    async def list_all(self, *, limit: int = 50, offset: int = 0) -> Sequence[User]: ...

    async def set_role(self, user_id: UUID, role: Role) -> User: ...

    async def set_status(self, user_id: UUID, status: UserStatus) -> User: ...


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
class ApiKeyRepository(Protocol):
    """API keys within the currently scoped tenant (docs/adr/0007).

    Every method here runs under row-level security, including the
    authentication lookup — the tenant is bound from the key's own tenant
    segment before `get_by_hash` runs. A key quoting another tenant therefore
    finds nothing rather than finding a row it must then be compared against.
    """

    async def get_by_hash(self, secret_hash: str) -> ApiKey | None: ...

    async def get(self, key_id: UUID) -> ApiKey | None: ...

    async def create(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        name: str,
        display_prefix: str,
        secret_hash: str,
        role: Role,
        created_by: UUID | None = None,
        expires_at: datetime | None = None,
    ) -> ApiKey: ...

    async def list_for_user(self, user_id: UUID) -> Sequence[ApiKey]: ...

    async def revoke(self, key_id: UUID, *, at: datetime) -> bool:
        """Mark a key unusable. Returns False if it was already revoked or absent."""
        ...

    async def touch_last_used(self, key_id: UUID, *, at: datetime, stale_after: timedelta) -> bool:
        """Record use, but only if the stored timestamp is already stale.

        Writing on every request would turn every authenticated GET into an
        update of the busiest row in the tenant: WAL amplification and row
        contention, bought with timestamp precision nobody reads.

        Returns whether a row was actually written, so the caller can skip a
        commit round trip on the overwhelmingly common no-op path.
        """
        ...


@runtime_checkable
class RefreshTokenRepository(Protocol):
    """Rotating refresh tokens, scoped to the current tenant."""

    async def create(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        token_hash: str,
        family_id: UUID,
        expires_at: datetime,
    ) -> RefreshToken: ...

    async def get_by_hash(self, token_hash: str) -> RefreshToken | None: ...

    async def mark_used(self, token_id: UUID, *, at: datetime, replaced_by: UUID) -> bool:
        """Consume a token. Returns False if it was already spent.

        A compare-and-set rather than a read-then-write, so two requests racing
        with the same refresh token cannot both succeed — which is what makes
        reuse detection reliable rather than probabilistic.
        """
        ...

    async def revoke_family(self, family_id: UUID, *, at: datetime) -> int:
        """Kill an entire rotation lineage. The response to a replayed token."""
        ...

    async def revoke_for_user(self, user_id: UUID, *, at: datetime) -> int: ...


@runtime_checkable
class PasswordHasher(Protocol):
    """Password stretching. Async because the work belongs in a thread.

    Argon2 is 50-100 ms of CPU. Running it on the event loop stalls every
    concurrent request on the worker, including open SSE streams — the sixth
    non-negotiable in CLAUDE.md, and one that no linter will catch here because
    the call is not a *known* blocking primitive.
    """

    async def hash(self, password: str) -> str: ...

    async def verify(self, password_hash: str | None, password: str) -> bool:
        """Check a password, tolerating a missing hash.

        `None` must still do the work: `users.password_hash` is nullable for
        SSO-provisioned accounts, and short-circuiting on it would make "this
        account has no password" measurably faster than "wrong password".
        """
        ...

    def needs_rehash(self, password_hash: str) -> bool:
        """Whether the stored hash predates the current cost parameters.

        Synchronous: it parses the hash string and does no work. Raising cost
        parameters is worthless without this — existing users would keep their
        old, cheaper hashes forever.
        """
        ...


@runtime_checkable
class TokenIssuer(Protocol):
    """Mints signed access tokens. Held only where tokens are created."""

    def issue_access_token(self, *, subject: UUID, tenant_id: UUID) -> AccessToken: ...


@runtime_checkable
class TokenVerifier(Protocol):
    """Verifies signed access tokens.

    Split from `TokenIssuer` even though one adapter satisfies both: a verifier
    needs only public keys, and a deployment that verifies without being able to
    sign is a real and desirable shape.
    """

    def verify_access_token(self, token: str) -> TokenClaims:
        """Verify signature, algorithm, `kid`, `typ`, `exp`, `nbf`, `aud`, `iss`.

        Raises `AuthenticationError` for every failure, without distinguishing
        them to the caller.
        """
        ...

    def public_jwks(self) -> dict[str, Any]:
        """The public key set, for `/.well-known/jwks.json`."""
        ...


@runtime_checkable
class RateLimiter(Protocol):
    """Token-bucket accounting (docs/adr/0008)."""

    async def check(self, key: str, policy: RateLimitPolicy, *, cost: int = 1) -> RateLimitDecision:
        """Consume `cost` tokens if available, and report the outcome.

        Implementations **fail open**: a limiter that cannot answer must admit
        the request and log loudly. Losing rate limiting costs fairness; failing
        closed on a limiter outage costs the entire API.
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
    api_keys: ApiKeyRepository
    refresh_tokens: RefreshTokenRepository

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
