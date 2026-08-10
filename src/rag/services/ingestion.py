"""Accepting a document: hash it, store it, record it, queue it.

Depends on ports only — a `UnitOfWork`, a `BlobStore` — never on FastAPI or the
filesystem. The upload source is described by a local Protocol rather than typed
as Starlette's `UploadFile`, so this module stays testable with a `BytesIO`
wrapper and free of the delivery layer.

The ordering that matters
-------------------------
**Blob first, then one transaction for the row and the job.**

Writing the blob before committing means a failure between the two leaves an
orphan blob: inert, invisible, and collectable by a sweep. Committing first
would mean a worker could claim the job before the bytes existed, turning a
storage hiccup into a user-visible ingestion failure.

Note this inverts ADR-0001's Postgres-then-Qdrant ordering, and correctly so:
Qdrant holds *derived* data that can be rebuilt, a blob holds *source* data that
cannot.

The row and the job then commit together, which is the entire justification for
a database-backed queue (ADR-0002): a job for a document that does not exist is
not a race to handle but a state that cannot occur.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from rag.core.logging import get_logger
from rag.domain.access import Principal
from rag.domain.enums import DocumentStatus, JobKind
from rag.domain.errors import (
    AlreadyExistsError,
    InvalidInputError,
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from rag.domain.ingestion import ContentType
from rag.domain.sniff import SNIFF_BYTES, sniff

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from rag.core.config import IngestionSettings
    from rag.domain.access import AuthenticatedPrincipal
    from rag.domain.models import Document
    from rag.domain.ports import BlobStore, UnitOfWork

__all__ = ["IngestionService", "UploadSource"]

_log = get_logger(__name__)


class UploadSource(Protocol):
    """The parts of an uploaded file this service needs.

    Structural, so Starlette's `UploadFile` satisfies it without this module
    importing anything from the web framework — and so a test can pass a
    thirty-line fake instead of constructing a multipart request.
    """

    async def read(self, size: int = -1) -> bytes: ...

    async def seek(self, offset: int) -> object: ...


class IngestionService:
    """Upload, re-upload, ACL changes, and deletion of documents."""

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        blobs: BlobStore,
        settings: IngestionSettings,
    ) -> None:
        self._uow = uow
        self._blobs = blobs
        self._settings = settings

    async def upload(
        self,
        principal: AuthenticatedPrincipal,
        *,
        collection_id: UUID,
        source: UploadSource,
        filename: str,
        declared_content_type: str | None = None,
        title: str | None = None,
    ) -> tuple[Document, bool]:
        """Ingest a file. Returns the document and whether it is newly created.

        `False` for the second element means the bytes were already held — see
        the idempotency note below. Callers turn that into 200 rather than 201.
        """
        collection = await self._uow.collections.get(collection_id)
        if collection is None:
            # RLS makes another tenant's collection invisible, so this covers
            # "does not exist" and "not yours" identically — which is what we
            # want: 403 here would confirm the collection exists.
            raise NotFoundError("Collection", str(collection_id))

        content_hash, size_bytes, head = await self._digest(source)
        if size_bytes == 0:
            raise InvalidInputError("The uploaded file is empty.", details={"filename": filename})

        content_type = self._resolve_content_type(head, filename, declared_content_type)

        existing = await self._uow.documents.get_by_content_hash(content_hash)
        if existing is not None:
            return await self._handle_duplicate(existing), False

        # Blob before the transaction — see the module docstring.
        blob_key = self._blobs.key_for(tenant_id=principal.tenant_id, content_hash=content_hash)
        await source.seek(0)
        await self._blobs.put(blob_key, self._stream(source))

        try:
            document = await self._uow.documents.create(
                tenant_id=principal.tenant_id,
                collection_id=collection_id,
                title=title or filename,
                source_uri=f"upload://{filename}",
                blob_key=blob_key,
                content_hash=content_hash,
                mime_type=content_type.value,
                size_bytes=size_bytes,
                uploaded_by=principal.user.id,
                acl_principals=self._default_acl(principal),
            )
        except AlreadyExistsError:
            # Lost a race with a simultaneous upload of the same bytes. Both
            # requests passed the probe above; the unique constraint decided.
            # The answer is the same one the probe would have given, so the
            # caller sees idempotency rather than a conflict they cannot act on.
            raced = await self._uow.documents.get_by_content_hash(content_hash)
            if raced is None:  # pragma: no cover - the constraint just fired
                raise
            _log.info("ingestion.duplicate_race", document_id=str(raced.id))
            return await self._handle_duplicate(raced), False
        # Rebind: `transition_status` returns the updated row, and returning the
        # object from `create()` would report `uploaded` to a caller who is
        # about to poll for a status that has already moved past it.
        document = await self._uow.documents.transition_status(
            document.id, expected=DocumentStatus.UPLOADED, target=DocumentStatus.QUEUED
        )
        await self._uow.jobs.enqueue(
            tenant_id=principal.tenant_id,
            kind=JobKind.INGEST_DOCUMENT,
            payload={"document_id": str(document.id)},
        )
        await self._uow.commit()

        _log.info(
            "ingestion.accepted",
            document_id=str(document.id),
            content_type=content_type.value,
            size_bytes=size_bytes,
        )
        return document, True

    async def set_acl(
        self, principal: AuthenticatedPrincipal, document_id: UUID, principals: Sequence[str]
    ) -> Document:
        """Replace a document's ACL, reprojecting it onto every chunk.

        The read is access-filtered, so an admin cannot change the ACL of a
        document they cannot already see — otherwise "manage ACLs" would be a
        way to discover documents by id.
        """
        document = await self._uow.documents.get(document_id, principal.access)
        if document is None:
            raise NotFoundError("Document", str(document_id))

        tokens = self._validated_tokens(principal, principals)
        await self._uow.documents.set_acl(document_id, tokens)
        await self._uow.commit()

        _log.info("ingestion.acl_changed", document_id=str(document_id), principals=len(tokens))
        updated = await self._uow.documents.get(document_id, principal.access)
        # The caller may have just removed their own access, which is a legal
        # thing to do and leaves nothing to return.
        return updated if updated is not None else document

    async def delete(self, principal: AuthenticatedPrincipal, document_id: UUID) -> None:
        """Begin deletion: move to DELETING and queue the purge.

        Deliberately asynchronous. The purge has to remove chunks, the blob and
        — from M5 — vectors from a second datastore, and doing that inside a
        request would make deletion latency depend on how large the document is.
        """
        document = await self._uow.documents.get(document_id, principal.access)
        if document is None:
            raise NotFoundError("Document", str(document_id))

        await self._uow.documents.transition_status(
            document_id, expected=document.status, target=DocumentStatus.DELETING
        )
        await self._uow.jobs.enqueue(
            tenant_id=principal.tenant_id,
            kind=JobKind.DELETE_DOCUMENT,
            payload={"document_id": str(document_id)},
            # Ahead of ingestion: a user who deletes something expects it gone,
            # and finishing the ingest of a doomed document is wasted work.
            priority=10,
        )
        await self._uow.commit()
        _log.info("ingestion.delete_requested", document_id=str(document_id))

    # -- internals ---------------------------------------------------------

    async def _digest(self, source: UploadSource) -> tuple[str, int, bytes]:
        """Stream the upload once to compute its hash, size, and sniff sample.

        The size check here is defence in depth: `BodySizeLimitMiddleware` has
        already capped the request body on the ASGI channel, which is the only
        place it can be enforced before the bytes reach disk. This catches a
        caller that reaches the service by some other route.
        """
        digest = hashlib.sha256()
        size = 0
        head = bytearray()

        while block := await source.read(self._settings.upload_chunk_bytes):
            size += len(block)
            if size > self._settings.max_upload_bytes:
                raise PayloadTooLargeError(limit_bytes=self._settings.max_upload_bytes)
            digest.update(block)
            if len(head) < SNIFF_BYTES:
                head.extend(block[: SNIFF_BYTES - len(head)])

        return digest.hexdigest(), size, bytes(head)

    async def _stream(self, source: UploadSource) -> AsyncIterator[bytes]:
        while block := await source.read(self._settings.upload_chunk_bytes):
            yield block

    def _resolve_content_type(
        self, head: bytes, filename: str, declared: str | None
    ) -> ContentType:
        """Decide the type from the bytes, and refuse what we cannot parse.

        Two distinct refusals, and the difference matters to the caller: "we do
        not recognise this at all" and "we recognise it and cannot parse it
        yet". Collapsing them would tell someone uploading a PDF that their file
        is corrupt.
        """
        content_type = sniff(head, filename=filename, declared=declared)
        if content_type is None:
            raise UnsupportedMediaTypeError(
                "The file is not a text format this system can ingest.",
                details={"filename": filename},
            )

        if not content_type.is_supported_in_m3a:
            raise UnsupportedMediaTypeError(
                f"{content_type.value} files are recognised but not yet supported; "
                f"parsing for them arrives in a later milestone.",
                detected=content_type.value,
                details={"filename": filename},
            )
        return content_type

    async def _handle_duplicate(self, existing: Document) -> Document:
        """Re-upload of bytes we already hold.

        Idempotent by content hash, which is what `uq_documents_tenant_id_
        content_hash` exists for: re-uploading identical bytes must not create a
        second document or pay the embedding cost twice.

        The exception is a document that previously *failed*. Returning it
        unchanged would mean a user who retries after we fixed the bug gets a
        permanent no, so it is requeued instead.
        """
        if existing.status is not DocumentStatus.FAILED:
            _log.info("ingestion.duplicate", document_id=str(existing.id))
            return existing

        await self._uow.documents.transition_status(
            existing.id, expected=DocumentStatus.FAILED, target=DocumentStatus.QUEUED
        )
        await self._uow.jobs.enqueue(
            tenant_id=existing.tenant_id,
            kind=JobKind.INGEST_DOCUMENT,
            payload={"document_id": str(existing.id)},
        )
        await self._uow.commit()
        _log.info("ingestion.requeued_after_failure", document_id=str(existing.id))

        requeued = await self._uow.documents.get_by_content_hash(existing.content_hash)
        return requeued if requeued is not None else existing

    def _default_acl(self, principal: AuthenticatedPrincipal) -> list[str]:
        """Who can read a freshly uploaded document.

        Tenant-wide, plus the uploader explicitly. A shared knowledge base whose
        documents nobody but the uploader can retrieve is not a knowledge base,
        and the tenant boundary is still absolute — `tenant:` grants reach
        exactly the people row-level security already lets in.

        The uploader's own principal is included so that narrowing the ACL later
        cannot accidentally lock out the person who owns the document.
        """
        return sorted(
            {
                Principal.tenant(principal.tenant_id).token,
                Principal.user(principal.user.id).token,
            }
        )

    def _validated_tokens(
        self, principal: AuthenticatedPrincipal, principals: Sequence[str]
    ) -> list[str]:
        """Parse and re-serialise every token, and pin them to this tenant.

        Round-tripping rejects malformed input, and rebuilding the token from
        the parsed parts means nothing reaches the ACL array that did not come
        out of `Principal`. A `tenant:` grant naming another tenant is refused
        outright — the array is matched with `&&` against a caller's principal
        set, so an unchecked foreign tenant token would be a cross-tenant grant
        written by a tenant admin.
        """
        tokens: set[str] = set()
        for raw in principals:
            try:
                parsed = Principal.parse(raw)
            except ValueError as exc:
                raise InvalidInputError(
                    f"Malformed principal token: {raw!r}", details={"principal": raw}
                ) from exc

            if parsed.type.value == "tenant" and parsed.id != str(principal.tenant_id):
                raise InvalidInputError(
                    "A document cannot be shared with another tenant.",
                    details={"principal": raw},
                )
            tokens.add(parsed.token)

        if not tokens:
            raise InvalidInputError("A document must be readable by at least one principal.")
        return sorted(tokens)
