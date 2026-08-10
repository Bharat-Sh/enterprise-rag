"""Collections — the unit documents are filed under.

Minimal on purpose: a document upload needs a collection to belong to, and
without these two endpoints there would be no way to make one through the API.
Renaming, deleting, and per-collection defaults are not needed yet.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from rag.api.deps import TenantRateLimit, UnitOfWorkDep, require
from rag.api.schemas.documents import CollectionCreateRequest, CollectionResponse
from rag.domain.access import AuthenticatedPrincipal
from rag.domain.authz import Permission

router = APIRouter(prefix="/collections", tags=["collections"], dependencies=[TenantRateLimit])

ReaderDep = Annotated[AuthenticatedPrincipal, Depends(require(Permission.COLLECTION_READ))]
CreatorDep = Annotated[AuthenticatedPrincipal, Depends(require(Permission.COLLECTION_CREATE))]


@router.get("", response_model=list[CollectionResponse], summary="List collections")
async def list_collections(principal: ReaderDep, uow: UnitOfWorkDep) -> list[CollectionResponse]:
    # Not access-filtered: collections are tenant-wide organisational units, and
    # row-level security already bounds them. Per-collection ACLs would be a
    # second access model layered on the document one, and documents are where
    # the sensitivity actually lives.
    collections = await uow.collections.list_all()
    return [CollectionResponse.model_validate(collection) for collection in collections]


@router.post(
    "",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a collection",
)
async def create_collection(
    body: CollectionCreateRequest, principal: CreatorDep, uow: UnitOfWorkDep
) -> CollectionResponse:
    collection = await uow.collections.create(
        tenant_id=principal.tenant_id,
        slug=body.slug,
        name=body.name,
        description=body.description,
    )
    await uow.commit()
    return CollectionResponse.model_validate(collection)


__all__ = ["router"]
