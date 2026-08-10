"""Tenant user administration.

Small on purpose — the milestone is authentication and RBAC, not user
management — but not omitted, because without an admin-gated surface the role
model is untested at the HTTP boundary and "a viewer cannot reach an admin
route" is an assertion with nothing to point at.

Three privilege levels are exercised here: listing and creating users are
`ADMIN`, and assigning a role is `OWNER`. The last one is not fussiness. An
admin who can grant roles can grant themselves `owner`, which makes the
distinction between the two decorative.

No endpoint accepts a `tenant_id`. It comes from the verified token, and the
row-level-security scope was bound from the same value before this module ran.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Path, Query, status

from rag.api.deps import TenantRateLimit, UnitOfWorkDep, require
from rag.api.schemas.auth import UserCreateRequest, UserResponse, UserRoleUpdateRequest
from rag.domain.access import AuthenticatedPrincipal
from rag.domain.authz import Permission
from rag.domain.enums import UserStatus
from rag.domain.errors import InvalidInputError, NotFoundError

router = APIRouter(prefix="/users", tags=["users"], dependencies=[TenantRateLimit])

ListerDep = Annotated[AuthenticatedPrincipal, Depends(require(Permission.USER_LIST))]
CreatorDep = Annotated[AuthenticatedPrincipal, Depends(require(Permission.USER_CREATE))]
RoleAssignerDep = Annotated[AuthenticatedPrincipal, Depends(require(Permission.USER_ASSIGN_ROLE))]


@router.get("", response_model=list[UserResponse], summary="List users in your tenant")
async def list_users(
    principal: ListerDep,
    uow: UnitOfWorkDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[UserResponse]:
    users = await uow.users.list_all(limit=limit, offset=offset)
    return [UserResponse.model_validate(user) for user in users]


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Invite a user",
    description=(
        "Creates the account in `invited` status with no password. It cannot "
        "authenticate until one is set — there is no code path that logs in a "
        "user whose `password_hash` is NULL."
    ),
)
async def create_user(
    body: UserCreateRequest, principal: CreatorDep, uow: UnitOfWorkDep
) -> UserResponse:
    # No pre-check for an existing address. The repository translates the unique
    # constraint into `AlreadyExistsError` (409), which is the only answer that
    # is also correct when two invites race. A read-then-insert here would add a
    # round trip and a second code path producing the same outcome less
    # reliably.
    #
    # Disclosing that the address is taken is safe: the caller is an admin of
    # this tenant and can list its users anyway, and the constraint is per
    # tenant, so it says nothing about any other customer.
    user = await uow.users.create(
        tenant_id=principal.tenant_id,
        email=str(body.email),
        full_name=body.full_name,
        role=body.role,
        status=UserStatus.INVITED,
    )
    await uow.commit()
    return UserResponse.model_validate(user)


@router.patch(
    "/{user_id}/role",
    response_model=UserResponse,
    summary="Assign a role",
    description="Owner only: an admin who can grant roles can promote themselves.",
)
async def set_user_role(
    body: UserRoleUpdateRequest,
    principal: RoleAssignerDep,
    uow: UnitOfWorkDep,
    user_id: Annotated[UUID, Path(description="Identifier of the user to modify.")],
) -> UserResponse:
    if user_id == principal.user.id:
        # Not paternalism: an owner demoting themselves can leave a tenant with
        # no owner at all, and there is no endpoint that could then restore one.
        raise InvalidInputError(
            "You cannot change your own role. Ask another owner.",
            details={"user_id": str(user_id)},
        )

    target = await uow.users.get(user_id)
    if target is None:
        # A user in another tenant is invisible under RLS and lands here as
        # absent, which is the answer we want: 403 would confirm they exist.
        raise NotFoundError("User", str(user_id))

    updated = await uow.users.set_role(user_id, body.role)
    await uow.commit()
    return UserResponse.model_validate(updated)


__all__ = ["router"]
