"""Document upload, inspection, ACL management, and deletion.

Upload answers **202 Accepted**, not 201. The document exists but is not
searchable yet, and saying "created" would invite a client to query it
immediately and conclude the system is broken. `status` is the field to poll.

Nothing here parses anything. Parsing is blocking CPU work and belongs to the
worker (`rag-worker`); doing it in a handler would stall every concurrent
request on that process, including open SSE streams from M8.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Path, Query, Response, UploadFile, status

from rag.api.deps import (
    IngestionServiceDep,
    TenantRateLimit,
    UnitOfWorkDep,
    require,
)
from rag.api.schemas.documents import AclUpdateRequest, ChunkResponse, DocumentResponse
from rag.domain.access import AuthenticatedPrincipal
from rag.domain.authz import Permission
from rag.domain.enums import DocumentStatus
from rag.domain.errors import NotFoundError

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[TenantRateLimit])

ReaderDep = Annotated[AuthenticatedPrincipal, Depends(require(Permission.DOCUMENT_READ))]
UploaderDep = Annotated[AuthenticatedPrincipal, Depends(require(Permission.DOCUMENT_UPLOAD))]
DeleterDep = Annotated[AuthenticatedPrincipal, Depends(require(Permission.DOCUMENT_DELETE))]
AclManagerDep = Annotated[AuthenticatedPrincipal, Depends(require(Permission.DOCUMENT_MANAGE_ACL))]

DocumentIdDep = Annotated[UUID, Path(description="Identifier of the document.")]


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for ingestion",
    description=(
        "Accepts the file, queues it, and returns immediately. Poll `status` "
        "until it reaches `ready` or `failed`.\n\n"
        "The content type is determined from the **bytes**, not from the "
        "declared `Content-Type` or the filename — both are caller-supplied. "
        "Re-uploading identical bytes is idempotent and answers 200 with the "
        "existing document, except when that document previously failed, in "
        "which case it is requeued."
    ),
    responses={
        status.HTTP_200_OK: {
            "model": DocumentResponse,
            "description": (
                "These exact bytes were already held; the existing document is returned."
            ),
        },
        status.HTTP_413_CONTENT_TOO_LARGE: {"description": "The file exceeds the limit."},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"description": "The format cannot be ingested."},
    },
)
async def upload_document(
    principal: UploaderDep,
    service: IngestionServiceDep,
    response: Response,
    file: Annotated[UploadFile, File(description="The document to ingest.")],
    collection_id: Annotated[UUID, Form(description="Collection to file it under.")],
    title: Annotated[str | None, Form(description="Defaults to the filename.")] = None,
) -> DocumentResponse:
    document, created = await service.upload(
        principal,
        collection_id=collection_id,
        source=file,
        filename=file.filename or "upload",
        declared_content_type=file.content_type,
        title=title,
    )
    # Set explicitly rather than raising: the body is the same either way, and
    # the distinction ("we made this" vs "we already had it") is worth
    # expressing in the status rather than in a field the client must read.
    response.status_code = status.HTTP_202_ACCEPTED if created else status.HTTP_200_OK
    return DocumentResponse.model_validate(document)


@router.get(
    "",
    response_model=list[DocumentResponse],
    summary="List documents you may read",
    description=(
        "Filtered by your principal set as a SQL pre-filter, not by discarding "
        "rows after the fact — so pagination counts what you can actually see."
    ),
)
async def list_documents(
    principal: ReaderDep,
    uow: UnitOfWorkDep,
    collection_id: Annotated[UUID | None, Query()] = None,
    document_status: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DocumentResponse]:
    documents = await uow.documents.list_for(
        principal.access,
        collection_id=collection_id,
        status=document_status,
        limit=limit,
        offset=offset,
    )
    return [DocumentResponse.model_validate(document) for document in documents]


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Fetch one document",
    description=(
        "A document in another tenant, or one your principals do not reach, "
        "answers 404 rather than 403 — a 403 would confirm it exists."
    ),
)
async def get_document(
    principal: ReaderDep, uow: UnitOfWorkDep, document_id: DocumentIdDep
) -> DocumentResponse:
    document = await uow.documents.get(document_id, principal.access)
    if document is None:
        raise NotFoundError("Document", str(document_id))
    return DocumentResponse.model_validate(document)


@router.get(
    "/{document_id}/chunks",
    response_model=list[ChunkResponse],
    summary="Inspect how a document was split",
    description=(
        "Mostly a debugging surface: chunk boundaries are the single biggest "
        "lever on retrieval quality, and being unable to see them makes tuning "
        "guesswork."
    ),
)
async def list_chunks(
    principal: ReaderDep, uow: UnitOfWorkDep, document_id: DocumentIdDep
) -> list[ChunkResponse]:
    # Authorise against the *document* first. `list_for_document` is not
    # access-filtered, so reaching it without this check would leak chunk text
    # for any document whose id could be guessed.
    if await uow.documents.get(document_id, principal.access) is None:
        raise NotFoundError("Document", str(document_id))

    chunks = await uow.chunks.list_for_document(document_id)
    return [ChunkResponse.model_validate(chunk) for chunk in chunks]


@router.put(
    "/{document_id}/acl",
    response_model=DocumentResponse,
    summary="Replace a document's access list",
    description=(
        "Rewrites the grants, the document's principal array, and the copy on "
        "every chunk in one transaction (docs/adr/0006). Revocation takes "
        "effect on the next request."
    ),
)
async def set_acl(
    body: AclUpdateRequest,
    principal: AclManagerDep,
    service: IngestionServiceDep,
    document_id: DocumentIdDep,
) -> DocumentResponse:
    document = await service.set_acl(principal, document_id, body.principals)
    return DocumentResponse.model_validate(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Delete a document",
    description=(
        "Asynchronous: the document moves to `deleting` and a purge job removes "
        "its chunks and stored bytes. 202 rather than 204 because the work is "
        "not finished when the response is sent."
    ),
)
async def delete_document(
    principal: DeleterDep, service: IngestionServiceDep, document_id: DocumentIdDep
) -> None:
    await service.delete(principal, document_id)


__all__ = ["router"]
