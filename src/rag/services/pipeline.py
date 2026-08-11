"""The ingestion pipeline: what a worker actually does with a claimed job.

    QUEUED ──► PARSING ──► CHUNKING ──► READY
                  │            │
              to_thread    to_thread

Every arrow is `transition_status`, a compare-and-set. Two workers racing on the
same document means the loser attempts a move from a state that no longer holds
and raises `InvalidStateTransitionError` rather than overwriting the winner's
progress — which is the difference between a duplicated effort and a corrupted
index.

Blocking work runs in a thread
------------------------------
Parsing and chunking are CPU-bound and synchronous. The worker holds one job at
a time, so nothing else is competing for its event loop, but the hop is still
made explicitly: it keeps the parse under a timeout (a malformed file that sends
a parser into a loop loses its job, not its worker) and it means this code does
not change when the API ever calls into it.

Re-runnable by construction
---------------------------
A job may be delivered more than once — that is what at-least-once delivery
means, and `reap_stalled` guarantees it happens eventually. So the chunking step
deletes existing chunks before inserting: a retry after a crash mid-insert must
converge on the right answer rather than doubling the document.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import anyio

from rag.core.logging import get_logger
from rag.domain.chunking import chunk_text
from rag.domain.enums import DocumentStatus
from rag.domain.errors import InvalidInputError, NotFoundError
from rag.domain.ingestion import ContentType

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from rag.core.config import IngestionSettings
    from rag.domain.ingestion import ParsedDocument
    from rag.domain.models import Document, Job
    from rag.domain.ports import BlobStore, DocumentParser, TokenCounter, UnitOfWork

__all__ = ["EXTRACTED_TEXT_KIND", "IngestionPipeline"]

_log = get_logger(__name__)

#: Suffix for the derived blob holding a document's extracted text. Keeping it
#: means a future change to chunking is a re-chunk rather than a re-parse of
#: every document ever ingested.
EXTRACTED_TEXT_KIND = "text"


class IngestionPipeline:
    """Runs one ingestion job to completion, or raises trying."""

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        blobs: BlobStore,
        tokens: TokenCounter,
        parser_for: Callable[[ContentType], DocumentParser | None],
        settings: IngestionSettings,
    ) -> None:
        self._uow = uow
        self._blobs = blobs
        self._tokens = tokens
        self._parser_for = parser_for
        self._settings = settings

    async def ingest(self, job: Job) -> None:
        """Parse, chunk, and mark a document ready.

        Redelivery after success is a no-op, not a failure. A worker that
        crashes between finishing the work and marking the job complete leaves a
        job that will be reclaimed — and without this guard the retry would
        attempt `QUEUED -> PARSING` on a document that is already `READY`, be
        dead-lettered for an illegal transition, and log a failure for work that
        actually succeeded.
        """
        document_id = _document_id_of(job)
        document = await self._load(document_id)

        if document.status is DocumentStatus.READY:
            _log.info("pipeline.already_ready", document_id=str(document_id))
            return

        parsed = await self._parse(document)
        await self._chunk(document, parsed)

    async def purge(self, job: Job) -> None:
        """Remove a document's derived data and tombstone the row.

        Chunks go first, then the blobs, then the tombstone. If the process dies
        between steps the job is redelivered and every step is idempotent, so it
        converges. The tombstone is written last on purpose: while it is absent
        the document is still visibly `DELETING`, which is a state an operator
        can find, rather than `DELETED` with data still behind it.
        """
        document_id = _document_id_of(job)
        document = await self._load(document_id)

        if document.status is DocumentStatus.DELETED:
            # Already purged on an earlier delivery. Same reasoning as `ingest`.
            _log.info("pipeline.already_deleted", document_id=str(document_id))
            return
        if document.status is not DocumentStatus.DELETING:
            raise InvalidInputError(
                f"Document is {document.status.value}, expected {DocumentStatus.DELETING.value}.",
                details={"document_id": str(document_id)},
            )

        removed = await self._uow.chunks.delete_for_document(document_id)
        await self._uow.commit()

        await self._blobs.delete(document.blob_key)
        await self._blobs.delete(
            self._blobs.derived_key_for(
                tenant_id=document.tenant_id,
                content_hash=document.content_hash,
                kind=EXTRACTED_TEXT_KIND,
            )
        )

        await self._uow.documents.transition_status(
            document_id, expected=DocumentStatus.DELETING, target=DocumentStatus.DELETED
        )
        await self._uow.commit()
        _log.info("pipeline.purged", document_id=str(document_id), chunks=removed)

    # -- stages ------------------------------------------------------------

    async def _parse(self, document: Document) -> ParsedDocument:
        await self._uow.documents.transition_status(
            document.id, expected=DocumentStatus.QUEUED, target=DocumentStatus.PARSING
        )
        await self._uow.commit()

        content_type = ContentType(document.mime_type)
        parser = self._parser_for(content_type)
        if parser is None:
            # Reachable if a document was accepted by an older build that
            # supported a format this one does not. Not retryable.
            raise InvalidInputError(
                f"No parser is registered for {content_type.value}.",
                details={"document_id": str(document.id)},
            )

        data = await self._blobs.read(document.blob_key)

        with anyio.fail_after(self._settings.parse_timeout_seconds):
            parsed = await anyio.to_thread.run_sync(parser.parse, data)

        if parsed.is_empty:
            # A scanned PDF, an empty HTML shell. Loud, because the alternative
            # is a document that reaches READY with zero chunks and is
            # permanently unfindable while appearing to have worked.
            raise InvalidInputError(
                "No text could be extracted from the document.",
                details={"document_id": str(document.id), "content_type": content_type.value},
            )

        await self._blobs.put(
            self._blobs.derived_key_for(
                tenant_id=document.tenant_id,
                content_hash=document.content_hash,
                kind=EXTRACTED_TEXT_KIND,
            ),
            _once(parsed.text.encode("utf-8")),
        )
        _log.info("pipeline.parsed", document_id=str(document.id), characters=len(parsed.text))
        return parsed

    async def _chunk(self, document: Document, parsed: ParsedDocument) -> None:
        await self._uow.documents.transition_status(
            document.id, expected=DocumentStatus.PARSING, target=DocumentStatus.CHUNKING
        )
        await self._uow.commit()

        chunks = await anyio.to_thread.run_sync(
            lambda: chunk_text(
                parsed.text,
                target_tokens=self._settings.chunk_target_tokens,
                overlap_tokens=self._settings.chunk_overlap_tokens,
                count_tokens=self._tokens.count,
            )
        )
        if not chunks:
            raise InvalidInputError(
                "The document produced no chunks.",
                details={"document_id": str(document.id)},
            )

        # Idempotent under redelivery: a retry replaces rather than appends.
        await self._uow.chunks.delete_for_document(document.id)
        await self._uow.chunks.add_many(document.id, chunks)

        # Straight to READY: there is no embedding provider until M4 and no
        # vector index until M5, so chunks are as far as this pipeline goes.
        # See the note on the CHUNKING edge in `rag.domain.state`.
        await self._uow.documents.transition_status(
            document.id, expected=DocumentStatus.CHUNKING, target=DocumentStatus.READY
        )
        await self._uow.commit()

        _log.info(
            "pipeline.ready",
            document_id=str(document.id),
            chunks=len(chunks),
            tokens=sum(chunk.token_count for chunk in chunks),
        )

    # -- internals ---------------------------------------------------------

    async def _load(self, document_id: UUID) -> Document:
        """Read the document the job names.

        Unfiltered by ACL, deliberately: a worker acts on behalf of the system,
        not a user, and the tenant scope is already bound from the job's own
        tenant. There is no caller here whose principals could be applied.
        """
        document = await self._uow.documents.get_for_processing(document_id)
        if document is None:
            raise NotFoundError("Document", str(document_id))
        return document


def _document_id_of(job: Job) -> UUID:
    raw = job.payload.get("document_id")
    if not isinstance(raw, str):
        raise InvalidInputError("Job payload has no document_id.", details={"job_id": str(job.id)})
    try:
        return UUID(raw)
    except ValueError as exc:
        raise InvalidInputError(
            f"Job payload document_id is not a UUID: {raw!r}", details={"job_id": str(job.id)}
        ) from exc


async def _once(data: bytes) -> AsyncIterator[bytes]:
    """Adapt a single in-memory buffer to the streaming `BlobStore.put` API."""
    yield data
