"""Self-service API key management.

Keys belong to the calling user and can only ever narrow that user's authority
(docs/adr/0007), which is why issuing one needs no elevated permission — it
grants nothing the caller did not already have. What it *does* need is an
interactive session: an API key may not mint another API key, or a leaked key
becomes a foothold that renews itself and revoking the leaked one achieves
nothing.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Path, status

from rag.api.deps import AuthServiceDep, TenantRateLimit, require
from rag.api.schemas.auth import ApiKeyCreatedResponse, ApiKeyCreateRequest, ApiKeyResponse
from rag.domain.access import AuthenticatedPrincipal
from rag.domain.authz import Permission

router = APIRouter(prefix="/api-keys", tags=["api-keys"], dependencies=[TenantRateLimit])

ManagerDep = Annotated[AuthenticatedPrincipal, Depends(require(Permission.API_KEY_MANAGE))]


@router.post(
    "",
    response_model=ApiKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Issue an API key",
    description=(
        "The secret is returned exactly once. Only a SHA-256 and a short display "
        "prefix are stored, so no code path — including a database dump — can "
        "produce it again. The requested role is a ceiling and may not exceed "
        "your own."
    ),
)
async def create_api_key(
    body: ApiKeyCreateRequest, principal: ManagerDep, service: AuthServiceDep
) -> ApiKeyCreatedResponse:
    key, secret = await service.create_api_key(
        principal, name=body.name, role=body.role, expires_in_days=body.expires_in_days
    )
    return ApiKeyCreatedResponse(key=ApiKeyResponse.model_validate(key), secret=secret)


@router.get(
    "",
    response_model=list[ApiKeyResponse],
    summary="List your API keys",
)
async def list_api_keys(principal: ManagerDep, service: AuthServiceDep) -> list[ApiKeyResponse]:
    keys = await service.list_api_keys(principal)
    return [ApiKeyResponse.model_validate(key) for key in keys]


@router.delete(
    "/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke an API key",
    description=(
        "A key belonging to another user — or another tenant — answers 404, not "
        "403. A 403 would confirm that the key exists."
    ),
)
async def revoke_api_key(
    principal: ManagerDep,
    service: AuthServiceDep,
    key_id: Annotated[UUID, Path(description="Identifier of the key to revoke.")],
) -> None:
    await service.revoke_api_key(principal, key_id)


__all__ = ["router"]
