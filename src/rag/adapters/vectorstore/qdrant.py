"""The Qdrant adapter. Satisfies `rag.domain.ports.VectorStore`.

One collection, tenant isolation by payload filter
--------------------------------------------------
Not a collection per tenant. Qdrant collections are heavyweight — each carries
its own segments and threads — so a few thousand tenants is untenable, and
creating one becomes part of tenant provisioning with its own half-created
failure states. Instead there is one collection and a `tenant_id` payload index
declared with `is_tenant=True`, which tells Qdrant to co-locate points by tenant
in storage rather than merely filtering over them.

The isolation itself comes from the filter, and the filter comes from exactly
one place (`filters.py`). This class never accepts a caller-built filter, which
is the only structural defence available: unlike Postgres there is no row-level
security here to catch a query that forgets its tenant clause.

Async by adaptation, not by nature
----------------------------------
`AsyncQdrantClient` is used against a server. Local mode has no async client, so
those calls run through `anyio.to_thread` — the port is `async` either way and
callers cannot tell. Doing it the other way round, making the port synchronous
because one backing mode is, would put blocking I/O in an `async def` handler
and stall every concurrent request on the worker (non-negotiable #6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

import anyio
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from rag.adapters.vectorstore.filters import (
    ACL_PRINCIPALS_FIELD,
    COLLECTION_ID_FIELD,
    DOCUMENT_ID_FIELD,
    ORDINAL_FIELD,
    TENANT_ID_FIELD,
    build_search_filter,
    build_tenant_document_filter,
)
from rag.core.errors import DependencyUnavailableError
from rag.core.logging import get_logger
from rag.domain.retrieval import SearchHit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag.core.config import QdrantSettings
    from rag.domain.access import AccessFilter
    from rag.domain.embedding import Embedding
    from rag.domain.retrieval import VectorPoint

__all__ = ["DEPENDENCY_NAME", "QdrantVectorStore"]

_log = get_logger(__name__)

DEPENDENCY_NAME = "qdrant"


class QdrantVectorStore:
    """Satisfies `rag.domain.ports.VectorStore`."""

    def __init__(self, settings: QdrantSettings) -> None:
        self._settings = settings
        self._collection = settings.collection
        self._dense = settings.dense_vector_name
        self._sparse = settings.sparse_vector_name

        if settings.uses_local_mode:
            # Embedded, no server. `_local` is a synchronous client; every call
            # below routes through `_call`, which hops to a thread for it.
            #
            # `:memory:` and a directory path are different constructor
            # arguments, not the same one with a magic value — `path=":memory:"`
            # would try to create a directory with that name, which is not even
            # a legal filename on Windows. In-memory also takes no lock, which
            # matters: a path-backed local client holds an exclusive lock on its
            # directory, so only one client per path per process can exist.
            self._local: QdrantClient | None = (
                QdrantClient(location=":memory:")
                if settings.local_path == ":memory:"
                else QdrantClient(path=settings.local_path)
            )
            self._remote: AsyncQdrantClient | None = None
        else:
            self._local = None
            self._remote = AsyncQdrantClient(
                host=settings.host,
                port=settings.port,
                grpc_port=settings.grpc_port,
                prefer_grpc=settings.prefer_grpc,
                api_key=settings.api_key.get_secret_value() if settings.api_key else None,
                timeout=int(settings.timeout_seconds),
                # The client's compatibility check compares its own version
                # against the server's and warns on any minor-version gap,
                # including the many combinations that work perfectly. Disabled
                # because a warning nobody can act on is a warning everyone
                # learns to ignore. What actually establishes compatibility is
                # the integration suite running against the server image pinned
                # in CI and in docker/compose.yml.
                check_compatibility=False,
            )

    async def aclose(self) -> None:
        if self._remote is not None:
            await self._remote.close()
        if self._local is not None:
            # Local mode holds a file lock on its storage directory. Leaving it
            # open makes the next process to open the same path fail, which in
            # a test suite looks like an unrelated flake two tests later.
            await anyio.to_thread.run_sync(self._local.close)

    # -- VectorStore -------------------------------------------------------

    async def ensure_ready(self) -> None:
        """Create the collection and payload indexes if they are absent.

        Called at startup, not lazily on first write, so a misconfigured vector
        store fails where an operator is watching rather than on the first
        document a user happens to upload.
        """
        if await self._call("collection_exists", collection_name=self._collection):
            _log.info("qdrant.collection_present", collection=self._collection)
        else:
            await self._call(
                "create_collection",
                collection_name=self._collection,
                vectors_config={
                    self._dense: models.VectorParams(
                        size=self._settings.vector_size,
                        # Cosine, because the model service returns L2-normalised
                        # vectors (docs/adr/0011). With unit vectors cosine and
                        # dot product rank identically; naming cosine explicitly
                        # means a provider that ever stops normalising degrades
                        # scores rather than silently inverting them.
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={self._sparse: models.SparseVectorParams()},
            )
            _log.info(
                "qdrant.collection_created",
                collection=self._collection,
                vector_size=self._settings.vector_size,
            )

        await self._ensure_payload_indexes()

    async def _ensure_payload_indexes(self) -> None:
        """Index the two fields every query filters on.

        Without these Qdrant evaluates the filter as a scan over the whole
        collection — correct, and progressively slower as other tenants' data
        grows, which is the worst shape of performance bug because it appears
        only in production and only later.

        `is_tenant=True` on the tenant field is Qdrant's multitenancy hint: it
        co-locates each tenant's points in storage, so a filtered search touches
        one tenant's segments instead of scanning across everyone's.

        Creating an index that already exists is not an error, so this is
        idempotent and safe on every boot. In local mode the client warns that
        payload indexes do nothing; the calls are still made so that the code
        path exercised in local tests is the same one production runs.
        """
        await self._call(
            "create_payload_index",
            collection_name=self._collection,
            field_name=TENANT_ID_FIELD,
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD, is_tenant=True
            ),
        )
        await self._call(
            "create_payload_index",
            collection_name=self._collection,
            field_name=ACL_PRINCIPALS_FIELD,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await self._call(
            "create_payload_index",
            collection_name=self._collection,
            field_name=DOCUMENT_ID_FIELD,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    async def upsert(self, points: Sequence[VectorPoint]) -> int:
        if not points:
            return 0

        batch_size = self._settings.upsert_batch_size
        written = 0
        for start in range(0, len(points), batch_size):
            window = points[start : start + batch_size]
            await self._call(
                "upsert",
                collection_name=self._collection,
                points=[self._to_point(point) for point in window],
                wait=True,
            )
            written += len(window)
        return written

    async def search(
        self,
        embedding: Embedding,
        access: AccessFilter,
        *,
        limit: int,
        collection_id: UUID | None = None,
        document_ids: Sequence[UUID] | None = None,
    ) -> Sequence[SearchHit]:
        """Dense nearest neighbours, pre-filtered by tenant and ACL.

        Dense only in M5. Sparse vectors are written and sit unqueried until M6
        adds fusion — see `QdrantSettings.sparse_vector_name` for why they are
        stored now rather than then.
        """
        response = await self._call(
            "query_points",
            collection_name=self._collection,
            query=list(embedding.dense),
            using=self._dense,
            query_filter=build_search_filter(
                access, collection_id=collection_id, document_ids=document_ids
            ),
            limit=limit,
            # The payload holds no text, so fetching it costs almost nothing and
            # carries the document id needed to group results.
            with_payload=True,
            with_vectors=False,
        )
        return [self._to_hit(point) for point in response.points]

    async def set_acl(
        self, tenant_id: UUID, document_id: UUID, acl_principals: Sequence[str]
    ) -> None:
        """Overwrite `acl_principals` on every point of one document.

        `set_payload` rather than `overwrite_payload`: the former merges the
        given keys and leaves the rest of the payload alone, which is what is
        wanted. `overwrite_payload` would replace the whole thing and silently
        drop `tenant_id` — removing the very field the isolation filter matches
        on, and making those points visible to every tenant at once.
        """
        await self._call(
            "set_payload",
            collection_name=self._collection,
            payload={ACL_PRINCIPALS_FIELD: list(acl_principals)},
            points=models.FilterSelector(
                filter=build_tenant_document_filter(tenant_id, document_id)
            ),
            wait=True,
        )

    async def delete_for_document(self, tenant_id: UUID, document_id: UUID) -> None:
        await self._call(
            "delete",
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=build_tenant_document_filter(tenant_id, document_id)
            ),
            wait=True,
        )

    async def drop(self) -> None:
        """Delete the collection entirely.

        Not part of the `VectorStore` port — no service should be able to do
        this — but the index is *disposable by contract* (ADR-0001), and a
        method that makes that true in code rather than in principle is worth
        having. Used by tests to clean up after themselves, and it is what a
        rebuild-from-Postgres command will call first.
        """
        await self._call("delete_collection", collection_name=self._collection)

    async def count_for_tenant(self, tenant_id: UUID) -> int:
        result = await self._call(
            "count",
            collection_name=self._collection,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=TENANT_ID_FIELD, match=models.MatchValue(value=str(tenant_id))
                    )
                ]
            ),
            exact=True,
        )
        return int(result.count)

    # -- health ------------------------------------------------------------

    async def ping(self) -> None:
        """Readiness probe: the collection exists and is reachable.

        Checks the collection rather than merely the connection. A Qdrant that
        is up but has lost its collection would answer a bare health check and
        fail every search.
        """
        try:
            exists = await self._call("collection_exists", collection_name=self._collection)
        except Exception as exc:
            raise DependencyUnavailableError(DEPENDENCY_NAME, str(exc)) from exc
        if not exists:
            raise DependencyUnavailableError(
                DEPENDENCY_NAME, f"collection {self._collection!r} does not exist"
            )

    # -- internals ---------------------------------------------------------

    def _to_point(self, point: VectorPoint) -> models.PointStruct:
        """Map a domain point onto Qdrant's wire shape.

        The payload deliberately excludes the chunk's text — see
        `rag.domain.retrieval`. Everything here is either a filter key or an
        identifier needed to hydrate from Postgres.
        """
        vector: dict[str, Any] = {self._dense: list(point.embedding.dense)}
        if not point.embedding.sparse.is_empty:
            # An empty sparse vector is legitimate for a chunk of pure
            # punctuation, and Qdrant rejects one with no entries — so it is
            # omitted rather than sent empty. The dense vector still indexes the
            # chunk, so it stays retrievable.
            vector[self._sparse] = models.SparseVector(
                indices=list(point.embedding.sparse.indices),
                values=list(point.embedding.sparse.values),
            )

        return models.PointStruct(
            # The chunk id *is* the point id (docs/adr/0001), which is what makes
            # re-indexing an overwrite rather than a duplicate.
            id=str(point.chunk_id),
            vector=vector,
            payload={
                TENANT_ID_FIELD: str(point.tenant_id),
                ACL_PRINCIPALS_FIELD: list(point.acl_principals),
                DOCUMENT_ID_FIELD: str(point.document_id),
                COLLECTION_ID_FIELD: (
                    str(point.collection_id) if point.collection_id is not None else None
                ),
                ORDINAL_FIELD: point.ordinal,
                "embedding_model": point.embedding_model,
            },
        )

    @staticmethod
    def _to_hit(scored: models.ScoredPoint) -> SearchHit:
        payload = scored.payload or {}
        return SearchHit(
            chunk_id=UUID(str(scored.id)),
            score=float(scored.score),
            document_id=UUID(str(payload[DOCUMENT_ID_FIELD])),
        )

    async def _call(self, method: str, **kwargs: Any) -> Any:
        """Invoke `method` on whichever client backs this store.

        Local mode ships no async client, so its calls take a thread hop. The
        two modes are otherwise API-compatible, which is what lets one adapter
        serve both and lets the local tests exercise the same code production
        runs rather than a parallel implementation of it.
        """
        if self._remote is not None:
            return await getattr(self._remote, method)(**kwargs)

        assert self._local is not None  # noqa: S101 - one of the two is always set
        local = self._local

        def _invoke() -> Any:
            return getattr(local, method)(**kwargs)

        return await anyio.to_thread.run_sync(_invoke)
