"""Document and chunk repositories.

This is where the access-control design becomes concrete. Two properties are
worth reading the code for:

* **No unauthorised read is expressible.** Every method that returns documents
  takes an `AccessFilter` and applies it in the `WHERE` clause. There is no
  "get everything then filter" path to accidentally take.
* **The ACL predicate is the array-overlap operator `&&`.** That is the exact
  semantics of Qdrant's `match_any`, so M5 enforces access with the same
  decision procedure rather than a similar-looking reimplementation. Divergence
  between two access checks is how leaks appear.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError

from rag.db.models import ChunkORM, DocumentORM, DocumentPermissionORM
from rag.db.repositories import affected_rows
from rag.domain.access import Principal
from rag.domain.enums import DocumentStatus
from rag.domain.errors import AlreadyExistsError, ConcurrentModificationError, NotFoundError
from rag.domain.models import Chunk, Document
from rag.domain.state import assert_can_transition

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from rag.domain.access import AccessFilter
    from rag.domain.models import NewChunk

__all__ = ["SqlAlchemyChunkRepository", "SqlAlchemyDocumentRepository"]


class SqlAlchemyDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, document_id: UUID, access: AccessFilter) -> Document | None:
        result = await self._session.execute(
            select(DocumentORM).where(
                DocumentORM.id == document_id,
                DocumentORM.tenant_id == access.tenant_id,
                DocumentORM.acl_principals.overlap(list(access.principal_tokens)),
            )
        )
        orm = result.scalar_one_or_none()
        return orm.to_domain() if orm else None

    async def get_for_processing(self, document_id: UUID) -> Document | None:
        """Read a document as the system, bypassing the ACL but not the tenant.

        Named to be conspicuous: it is the only read here without an
        `AccessFilter`. Row-level security still applies — the tenant scope is
        bound from the job — so this widens visibility *within* one tenant and
        never across them. A worker has no caller whose principals it could use.
        """
        orm = await self._session.get(DocumentORM, document_id)
        return orm.to_domain() if orm else None

    async def get_by_content_hash(self, content_hash: str) -> Document | None:
        """Idempotency probe, scoped to the tenant by row-level security alone.

        No `AccessFilter`: this runs before there is a document to authorise
        against, and its answer ("do we already hold these exact bytes?") is not
        a disclosure within a tenant.
        """
        result = await self._session.execute(
            select(DocumentORM).where(DocumentORM.content_hash == content_hash)
        )
        orm = result.scalar_one_or_none()
        return orm.to_domain() if orm else None

    async def list_for(
        self,
        access: AccessFilter,
        *,
        collection_id: UUID | None = None,
        status: DocumentStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Document]:
        stmt = select(DocumentORM).where(
            DocumentORM.tenant_id == access.tenant_id,
            DocumentORM.acl_principals.overlap(list(access.principal_tokens)),
        )
        if collection_id is not None:
            stmt = stmt.where(DocumentORM.collection_id == collection_id)
        if status is not None:
            stmt = stmt.where(DocumentORM.status == status)

        # Ordering by the UUIDv7 primary key gives newest-first for free: the
        # timestamp lives in the high bits, so the key order *is* creation
        # order. No secondary index on created_at required.
        stmt = stmt.order_by(DocumentORM.id.desc()).limit(limit).offset(offset)

        result = await self._session.execute(stmt)
        return [orm.to_domain() for orm in result.scalars().all()]

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
    ) -> Document:
        orm = DocumentORM(
            tenant_id=tenant_id,
            collection_id=collection_id,
            title=title,
            source_uri=source_uri,
            blob_key=blob_key,
            content_hash=content_hash,
            mime_type=mime_type,
            size_bytes=size_bytes,
            uploaded_by=uploaded_by,
            status=DocumentStatus.UPLOADED,
            acl_principals=list(acl_principals),
            doc_metadata=metadata or {},
        )
        try:
            # SAVEPOINT, so losing the race on `uq_documents_tenant_id_
            # content_hash` does not abort the caller's whole transaction. Two
            # simultaneous uploads of identical bytes both pass the
            # `get_by_content_hash` probe and both reach here; the constraint is
            # the only thing that can adjudicate, and the service turns this
            # into the same idempotent answer the probe would have given.
            async with self._session.begin_nested():
                self._session.add(orm)
                await self._session.flush()
        except IntegrityError as exc:
            raise AlreadyExistsError(
                "A document with these exact contents already exists in this tenant.",
                details={"content_hash": content_hash},
            ) from exc
        await self._session.refresh(orm)
        return orm.to_domain()

    async def transition_status(
        self,
        document_id: UUID,
        *,
        expected: DocumentStatus,
        target: DocumentStatus,
        reason: str | None = None,
    ) -> Document:
        """Compare-and-set the document status.

        Two steps, and both matter:

        1. `assert_can_transition` rejects transitions the state machine
           forbids, before touching the database. A caller trying to jump
           PARSING straight to READY is a bug, not a race.
        2. `WHERE status = :expected` makes the write conditional. Two workers
           racing on the same document means exactly one `UPDATE` matches a row;
           the other gets zero and raises, rather than clobbering the winner.
        """
        assert_can_transition(expected, target, document_id=str(document_id))

        values: dict[str, Any] = {
            "status": target,
            "status_reason": reason,
            "updated_at": func.now(),
        }
        if target is DocumentStatus.READY:
            values["indexed_at"] = func.now()

        result = await self._session.execute(
            update(DocumentORM)
            .where(DocumentORM.id == document_id, DocumentORM.status == expected)
            .values(**values)
            .returning(DocumentORM)
            .execution_options(synchronize_session=False)
        )
        orm = result.scalar_one_or_none()
        if orm is not None:
            return orm.to_domain()

        # Nothing matched, which means one of exactly two things. They need
        # different responses (404 vs 409) and different client behaviour
        # (give up vs re-read and retry), so we distinguish them rather than
        # collapsing both into a generic failure.
        current = await self._session.execute(
            select(DocumentORM.status).where(DocumentORM.id == document_id)
        )
        actual = current.scalar_one_or_none()
        if actual is None:
            raise NotFoundError("Document", str(document_id))

        raise ConcurrentModificationError(
            "Document",
            expected=expected.value,
            actual=actual.value,
            identifier=str(document_id),
        )

    async def set_acl(self, document_id: UUID, principal_tokens: Sequence[str]) -> None:
        """Replace the ACL, rewriting all three places it is represented.

        Normalised grants, the document's denormalised array, and the copy on
        every chunk — updated together in one transaction. Letting them diverge
        is precisely how a revoked permission keeps returning search results.
        """
        tokens = sorted(set(principal_tokens))

        result = await self._session.execute(
            select(DocumentORM.tenant_id).where(DocumentORM.id == document_id)
        )
        tenant_id = result.scalar_one_or_none()
        if tenant_id is None:
            raise NotFoundError("Document", str(document_id))

        await self._session.execute(
            delete(DocumentPermissionORM).where(DocumentPermissionORM.document_id == document_id)
        )
        for token in tokens:
            principal = Principal.parse(token)
            self._session.add(
                DocumentPermissionORM(
                    document_id=document_id,
                    principal_type=principal.type,
                    principal_id=principal.id,
                    tenant_id=tenant_id,
                )
            )

        await self._session.execute(
            update(DocumentORM)
            .where(DocumentORM.id == document_id)
            .values(acl_principals=tokens, updated_at=func.now())
        )
        await self._session.execute(
            update(ChunkORM)
            .where(ChunkORM.document_id == document_id)
            .values(acl_principals=tokens)
        )
        await self._session.flush()


class SqlAlchemyChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_many(self, document_id: UUID, chunks: Sequence[NewChunk]) -> int:
        """Insert chunks, inheriting tenant and ACL from the parent document.

        Inheritance happens here rather than at the call site so it cannot be
        forgotten. A chunker knows about text and offsets; it has no business
        knowing about tenancy, and a chunk written with the wrong ACL is
        directly a data leak.
        """
        if not chunks:
            return 0

        result = await self._session.execute(
            select(DocumentORM.tenant_id, DocumentORM.acl_principals).where(
                DocumentORM.id == document_id
            )
        )
        row = result.one_or_none()
        if row is None:
            raise NotFoundError("Document", str(document_id))
        tenant_id, acl_principals = row

        self._session.add_all(
            [
                ChunkORM(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    ordinal=chunk.ordinal,
                    text_content=chunk.text,
                    token_count=chunk.token_count,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    acl_principals=list(acl_principals or []),
                    chunk_metadata=chunk.metadata,
                )
                for chunk in chunks
            ]
        )
        await self._session.flush()
        return len(chunks)

    async def list_for_document(self, document_id: UUID) -> Sequence[Chunk]:
        result = await self._session.execute(
            select(ChunkORM).where(ChunkORM.document_id == document_id).order_by(ChunkORM.ordinal)
        )
        return [orm.to_domain() for orm in result.scalars().all()]

    async def delete_for_document(self, document_id: UUID) -> int:
        result = await self._session.execute(
            delete(ChunkORM).where(ChunkORM.document_id == document_id)
        )
        return affected_rows(result)

    async def get_many(self, chunk_ids: Sequence[UUID], access: AccessFilter) -> Sequence[Chunk]:
        """Hydrate chunks after a vector search, re-applying the access filter.

        The vector store already pre-filtered on the same principals, so this is
        redundant *when the two stores agree*. It exists for when they do not:
        a stale index, a failed delete, a botched migration. Redundant checks
        on the read path are cheap; a leak is not.
        """
        if not chunk_ids:
            return []

        result = await self._session.execute(
            select(ChunkORM).where(
                ChunkORM.id.in_(list(chunk_ids)),
                ChunkORM.tenant_id == access.tenant_id,
                ChunkORM.acl_principals.overlap(list(access.principal_tokens)),
            )
        )
        found = {orm.id: orm.to_domain() for orm in result.scalars().all()}
        # Preserve the caller's ordering — it is the relevance ranking, and
        # returning it in primary-key order would silently discard the ranking.
        return [found[chunk_id] for chunk_id in chunk_ids if chunk_id in found]

    async def set_embedding_metadata(self, document_id: UUID, *, model: str, version: str) -> int:
        """Stamp which model produced the vectors for this document's chunks.

        Recorded per chunk so that changing embedding model is a backfill —
        re-embed into a new named vector, cut over — instead of dropping and
        rebuilding the whole index.
        """
        result = await self._session.execute(
            update(ChunkORM)
            .where(ChunkORM.document_id == document_id)
            .values(embedding_model=model, embedding_version=version)
        )
        return affected_rows(result)
