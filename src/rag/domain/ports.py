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

Ports added in later milestones — `LLMClient`, `Cache` — belong in this module
too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from datetime import datetime, timedelta
    from types import TracebackType
    from uuid import UUID

    from rag.domain.access import AccessFilter
    from rag.domain.credentials import AccessToken, TokenClaims
    from rag.domain.embedding import Embedding, EmbedMode, ModelInfo, RerankResult
    from rag.domain.enums import DocumentStatus, JobKind, JobStatus, Role, UserStatus
    from rag.domain.ingestion import ParsedDocument
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
    from rag.domain.retrieval import SearchHit, VectorPoint

__all__ = [
    "ApiKeyRepository",
    "BlobStore",
    "ChunkRepository",
    "CollectionRepository",
    "DocumentParser",
    "DocumentRepository",
    "EmbeddingProvider",
    "GroupRepository",
    "JobRepository",
    "PasswordHasher",
    "RateLimiter",
    "RefreshTokenRepository",
    "Reranker",
    "TenantRepository",
    "TokenCounter",
    "TokenIssuer",
    "TokenVerifier",
    "UnitOfWork",
    "UserRepository",
    "VectorStore",
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

    async def get_for_processing(self, document_id: UUID) -> Document | None:
        """Read a document as the *system*, for a background worker.

        The only method here that returns a document without an `AccessFilter`,
        and it is named to be conspicuous. A worker acts on behalf of the
        platform rather than a user: there is no caller whose principals could
        be applied, and applying the uploader's would be wrong the moment their
        access changed.

        Tenant isolation is untouched — row-level security still applies, bound
        from the job's own tenant. This widens *ACL* visibility within one
        tenant, never tenant visibility.
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
        blob_key: str,
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
class BlobStore(Protocol):
    """Where a document's raw bytes live (docs/adr/0009).

    A port because the filesystem adapter shipped in M3 is not what runs in
    production — an S3-compatible store is — and the difference should be one
    adapter and one wiring line, not a change at every call site.

    Keys are opaque to callers. The adapter decides the layout; a caller that
    builds a key by hand has taken a dependency on the storage layout and will
    break when it changes.
    """

    async def put(self, key: str, stream: AsyncIterator[bytes]) -> int:
        """Write a stream, returning the number of bytes written.

        Takes an iterator rather than `bytes` so an upload is never held in
        memory in full. Overwriting an existing key is a no-op in effect —
        keys are content-addressed, so identical keys hold identical bytes.
        """
        ...

    async def open(self, key: str) -> AsyncIterator[bytes]:
        """Stream a blob back. Raises `NotFoundError` if the key is absent."""
        ...

    async def read(self, key: str) -> bytes:
        """Read a whole blob. Only for content already known to be bounded."""
        ...

    async def delete(self, key: str) -> bool:
        """Remove a blob. False if it was not there — deletion is idempotent."""
        ...

    def key_for(self, *, tenant_id: UUID, content_hash: str) -> str:
        """The key for a document's raw bytes.

        Content-addressed *within* a tenant. Identical bytes uploaded by two
        customers are stored twice, deliberately: sharing them would make one
        tenant's deletion affect another, and make storage a cross-tenant
        existence oracle.
        """
        ...

    def derived_key_for(self, *, tenant_id: UUID, content_hash: str, kind: str) -> str:
        """The key for something computed *from* a document, e.g. extracted text.

        Storing extracted text means a change to chunking is a re-chunk rather
        than a re-parse of every PDF ever ingested.
        """
        ...


@runtime_checkable
class DocumentParser(Protocol):
    """Turns bytes of one content type into text.

    Synchronous on purpose. Parsing is blocking CPU work, and an `async def`
    that never awaits is a lie that invites someone to call it on the event
    loop. The caller is responsible for the thread hop, which makes the cost
    visible at the call site.
    """

    def parse(self, data: bytes) -> ParsedDocument: ...


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens the way the embedding model will.

    M3 ships a character-ratio estimator; M4 adds `BgeTokenCounter`, which loads
    BGE-M3's real vocabulary. A port rather than a direct import because chunk
    sizing is the one place where being wrong is invisible — chunks silently
    over the model's window get truncated at embedding time, losing their tail
    with no error anywhere.

    **Synchronous, and that is a constraint on implementations, not an
    oversight.** The recursive splitter calls this once per candidate span,
    which is hundreds of calls per document. An implementation that reached the
    model service over HTTP would therefore make chunking hundreds of round
    trips, and would have to be `async`, which would push the thread hop into
    `rag.domain.chunking`. The tokenizer runs in-process; only inference is
    remote.
    """

    def count(self, text: str) -> int: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors (docs/adr/0004).

    Async because the implementation is a network call to the GPU model service.
    Batched because the GPU is: one forward pass over 32 texts costs barely more
    than one over a single text, so a per-text interface would leave most of the
    hardware idle and multiply the round trips.
    """

    async def embed(self, texts: Sequence[str], *, mode: EmbedMode) -> Sequence[Embedding]:
        """Embed every text, returning results in the order given.

        Order is part of the contract: the caller pairs results back to chunk
        ids positionally, so a provider that reorders — for length bucketing,
        say — must restore the original order before returning. Getting this
        wrong attaches every vector to the wrong chunk, and retrieval still
        *works*, it just returns nonsense.

        Raises `DependencyUnavailableError` when the service cannot be reached
        or fails. Deliberately **not** fail-open, unlike `RateLimiter`: there is
        no degraded embedding, and a document indexed with placeholder vectors
        is unfindable while claiming to be searchable.
        """
        ...

    async def info(self) -> ModelInfo:
        """Identify the models in use, for stamping onto chunk rows."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Re-scores retrieved passages against a query with a cross-encoder.

    Separate from `EmbeddingProvider` even though one adapter satisfies both.
    They are used at different points by different code, and a deployment that
    embeds locally while reranking through a hosted API — or skips reranking
    entirely — is a real shape. One combined port would make that a fork rather
    than a wiring change.
    """

    async def rerank(
        self, query: str, passages: Sequence[str], *, top_k: int | None = None
    ) -> Sequence[RerankResult]:
        """Score passages, returned highest-first.

        Results reference passages by their index in `passages`. `top_k` trims
        the response; scoring cost is unaffected, because a cross-encoder must
        run over every candidate to know which ones win.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """The derived vector index (docs/adr/0001, docs/adr/0006).

    **Every read takes an `AccessFilter`, and there is no method that does not.**
    That is structural, matching `DocumentRepository`, and here it carries more
    weight than it does there. Postgres has row-level security: a query that
    forgets its tenant scope finds zero rows. A vector store has no equivalent —
    a query that forgets its tenant filter returns *every tenant's* vectors, and
    nothing anywhere objects. The filter cannot be optional and cannot be
    supplied by a caller; implementations build it themselves from the
    `AccessFilter` and must never accept a pre-built one.

    Disposable by contract. Nothing here is a source of truth: every point can
    be rebuilt from `chunks` plus the embedding provider, which is what makes
    changing chunking or embedding model survivable rather than terrifying.
    """

    async def ensure_ready(self) -> None:
        """Create the collection and its payload indexes if absent.

        Idempotent, and called at startup rather than lazily on first write: a
        misconfigured vector store should fail where an operator is watching,
        not on the first document a user uploads.
        """
        ...

    async def upsert(self, points: Sequence[VectorPoint]) -> int:
        """Write points, overwriting any with the same id.

        Idempotent because point ids are chunk ids. A worker that dies
        mid-indexing is redelivered and rewrites the same points, rather than
        needing to know which of them already landed.
        """
        ...

    async def search(
        self,
        embedding: Embedding,
        access: AccessFilter,
        *,
        limit: int,
        collection_id: UUID | None = None,
        document_ids: Sequence[UUID] | None = None,
    ) -> Sequence[SearchHit]:
        """Nearest neighbours the caller is permitted to see.

        The access check is a **pre-filter**, evaluated inside the query
        (non-negotiable #4). Post-filtering is wrong twice over: asking for the
        top 50 and discarding 40 leaves 10 results, not the true top 10 — and
        the vectors were already read before the check, so the disclosure has
        already happened.
        """
        ...

    async def set_acl(
        self, tenant_id: UUID, document_id: UUID, acl_principals: Sequence[str]
    ) -> None:
        """Rewrite the ACL on every point of one document.

        An ACL change does not change a single vector — only who is allowed to
        match one — so this updates the payload in place rather than re-embedding
        anything. That is the difference between a permission change costing a
        payload write and costing a GPU pass over the whole document.

        It matters that this exists at all. `documents.set_acl` rewrites three
        representations in Postgres (docs/adr/0006); without a fourth write here
        the index keeps the old ACL, and a *widened* grant would then silently
        fail to return the newly-shared document. A *narrowed* one is caught by
        the re-check at hydration, so the failure is one-directional — but
        "half of revocation works" is not a property worth relying on.
        """
        ...

    async def delete_for_document(self, tenant_id: UUID, document_id: UUID) -> None:
        """Remove every point belonging to one document.

        By filter rather than by id, so purging does not depend on first
        reading the chunk rows it is about to delete.
        """
        ...

    async def count_for_tenant(self, tenant_id: UUID) -> int:
        """Points held for one tenant. For operations and reconciliation."""
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
