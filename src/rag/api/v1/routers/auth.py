"""Login, token refresh, logout, and self-service account operations.

Three of these endpoints run *before* a caller is known — login, refresh, and
logout — so they take the deliberately conspicuous `UnscopedAuthServiceDep` and
bind their own tenant scope as their first act. Everything else in the API takes
`UnitOfWorkDep`, which cannot be obtained without a verified tenant.

Rate limiting differs for the same reason. The unauthenticated endpoints are
limited by client address, because there is no tenant yet and they are the
credential-stuffing surface; the authenticated ones are limited per tenant.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, status

from rag.api.deps import (
    AuthServiceDep,
    LoginRateLimit,
    PrincipalDep,
    TenantRateLimit,
    UnscopedAuthServiceDep,
)
from rag.api.schemas.auth import (
    LoginRequest,
    MeResponse,
    PasswordChangeRequest,
    RefreshRequest,
    TokenResponse,
    UserResponse,
)
from rag.domain.credentials import TokenPair

router = APIRouter(prefix="/auth", tags=["auth"])

#: Documented once and reused: every failure on these endpoints is this shape.
_UNAUTHORIZED: dict[int | str, dict[str, Any]] = {
    status.HTTP_401_UNAUTHORIZED: {
        "description": (
            "Authentication failed. Deliberately indistinguishable across unknown "
            "tenant, unknown user, wrong password, expired token, and revoked "
            "credential — each distinction would be an enumeration oracle."
        )
    }
}


def _as_response(pair: TokenPair) -> TokenResponse:
    return TokenResponse(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        token_type=pair.token_type,
        access_expires_at=pair.access_expires_at,
        refresh_expires_at=pair.refresh_expires_at,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    dependencies=[LoginRateLimit],
    responses=_UNAUTHORIZED,
    summary="Exchange a password for a token pair",
)
async def login(body: LoginRequest, service: UnscopedAuthServiceDep) -> TokenResponse:
    return _as_response(
        await service.login(
            tenant_slug=body.tenant_slug, email=str(body.email), password=body.password
        )
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    dependencies=[LoginRateLimit],
    responses=_UNAUTHORIZED,
    summary="Rotate a refresh token",
    description=(
        "Single use. Presenting a token that has already been spent means two "
        "parties hold it, so the entire rotation family is revoked and every "
        "outstanding access token for that user is invalidated."
    ),
)
async def refresh(body: RefreshRequest, service: UnscopedAuthServiceDep) -> TokenResponse:
    return _as_response(await service.refresh(body.refresh_token))


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[LoginRateLimit],
    summary="Revoke one device's refresh-token family",
    description=(
        "Idempotent, and takes no access token: a client whose access token has "
        "already expired must still be able to sign out. The caller's access "
        "token remains valid until it expires — the bounded cost of stateless "
        "tokens. Use a password change to invalidate every session at once."
    ),
)
async def logout(body: RefreshRequest, service: UnscopedAuthServiceDep) -> None:
    await service.logout(body.refresh_token)


@router.post(
    "/logout-all",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[TenantRateLimit],
    responses=_UNAUTHORIZED,
    summary="Sign out of every device",
    description=(
        "The compromised-account button. Revokes every refresh token and moves "
        "the access-token watermark, so live tokens stop being honoured too. "
        "Unlike `/logout` this needs an access token, because it acts on the "
        "whole account rather than on the one credential presented."
    ),
)
async def logout_everywhere(principal: PrincipalDep, service: AuthServiceDep) -> None:
    await service.logout_everywhere(principal.user)


@router.get(
    "/me",
    response_model=MeResponse,
    dependencies=[TenantRateLimit],
    responses=_UNAUTHORIZED,
    summary="The calling principal",
)
async def me(principal: PrincipalDep) -> MeResponse:
    return MeResponse(
        user=UserResponse.model_validate(principal.user),
        tenant_id=principal.tenant_id,
        # The effective role, not `user.role`: a narrowed API key must be able to
        # show the caller the ceiling that is actually in force.
        effective_role=principal.effective_role,
        credential=principal.credential,
    )


@router.post(
    "/password",
    response_model=TokenResponse,
    dependencies=[TenantRateLimit],
    responses=_UNAUTHORIZED,
    summary="Change your password",
    description=(
        "Revokes every other session — the reason to change a password is that it "
        "may be compromised, so leaving sessions alive defeats the act. A fresh "
        "token pair is returned so the client that did the right thing is not "
        "signed out. Not available to API keys."
    ),
)
async def change_password(
    body: PasswordChangeRequest,
    principal: PrincipalDep,
    service: AuthServiceDep,
) -> TokenResponse:
    return _as_response(
        await service.change_password(
            principal,
            current_password=body.current_password,
            new_password=body.new_password,
        )
    )


#: Re-exported for the route-coverage security test, which asserts that every
#: route outside a known-unauthenticated set resolves a principal.
UNAUTHENTICATED_PATHS: frozenset[str] = frozenset(
    {"/api/v1/auth/login", "/api/v1/auth/refresh", "/api/v1/auth/logout"}
)

__all__ = ["UNAUTHENTICATED_PATHS", "router"]
