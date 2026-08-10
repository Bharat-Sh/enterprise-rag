"""Wire models for authentication and credential management.

Separate from the domain dataclasses on purpose. These describe the *HTTP
contract* — field names clients depend on, validation Pydantic can express, and
what we are willing to disclose. A domain entity serialised directly makes every
future column an accidental part of the public API.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from rag.domain.credentials import CredentialKind
from rag.domain.enums import Role, UserStatus

__all__ = [
    "ApiKeyCreateRequest",
    "ApiKeyCreatedResponse",
    "ApiKeyResponse",
    "LoginRequest",
    "MeResponse",
    "PasswordChangeRequest",
    "RefreshRequest",
    "TokenResponse",
    "UserCreateRequest",
    "UserResponse",
    "UserRoleUpdateRequest",
]


class LoginRequest(BaseModel):
    """Credentials for `POST /api/v1/auth/login`.

    `tenant_slug` is an *addressing* input, not an authorization one: it selects
    whose credential store to check. Naming someone else's tenant still requires
    that tenant's password, and the failure is indistinguishable from any other.
    Resolving the tenant from the email address instead was rejected — email is
    unique per tenant by design, and a global lookup would tell one customer
    whether an address exists in another.
    """

    tenant_slug: str = Field(min_length=1, max_length=64, examples=["acme"])
    email: EmailStr = Field(examples=["ada@acme.test"])
    # Bounded above as well as below. Argon2's cost is essentially flat in input
    # length, but an unbounded field is still an unbounded allocation on an
    # unauthenticated endpoint.
    password: str = Field(min_length=1, max_length=1024)


class RefreshRequest(BaseModel):
    """A refresh token, in the body rather than the `Authorization` header.

    Deliberate: the bearer header carries *access* credentials, and accepting a
    refresh token there would eventually let one authenticate an ordinary
    request. Keeping them in different places makes that mistake impossible
    rather than merely discouraged.
    """

    refresh_token: str = Field(min_length=1, max_length=512)


class TokenResponse(BaseModel):
    """The result of a login, refresh, or password change.

    The refresh token appears here and is never retrievable again — only its
    SHA-256 is stored.
    """

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"  # noqa: S105 - a scheme name, not a credential
    access_expires_at: datetime
    refresh_expires_at: datetime


class PasswordChangeRequest(BaseModel):
    """Self-service password change.

    The current password is required even though the caller is authenticated:
    it is what stops a stolen access token from locking the real owner out.
    """

    current_password: str = Field(min_length=1, max_length=1024)
    new_password: str = Field(min_length=1, max_length=1024)


class UserResponse(BaseModel):
    """A user, as the API discloses them.

    `password_hash` and `tokens_valid_after` are absent by construction rather
    than by exclusion — this model lists what goes out, so a new column is
    private until somebody adds it here on purpose.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    full_name: str
    role: Role
    status: UserStatus
    created_at: datetime
    last_login_at: datetime | None = None


class MeResponse(BaseModel):
    """Who the API thinks the caller is, right now.

    Reports the **effective** role rather than the user's own, so a caller
    holding a narrowed API key can see the ceiling that is actually in force.
    Debugging "why was I denied?" without this means reading the audit log.
    """

    user: UserResponse
    tenant_id: UUID
    effective_role: Role
    credential: CredentialKind


class UserCreateRequest(BaseModel):
    """Invite a user into the calling tenant.

    No `tenant_id`: it comes from the verified token, and accepting one here
    would be a cross-tenant write with a friendly interface.
    """

    email: EmailStr
    full_name: str = Field(default="", max_length=255)
    role: Role = Role.MEMBER


class UserRoleUpdateRequest(BaseModel):
    role: Role


class ApiKeyCreateRequest(BaseModel):
    """Mint a key for the calling user.

    `role` is a **ceiling**, and it may not exceed the caller's own — the
    request is rejected rather than quietly narrowed. Defaults to `viewer`, the
    least privileged option, so a key issued without thought is the safest one.
    """

    name: str = Field(min_length=1, max_length=128, examples=["ci-deploy"])
    role: Role = Role.VIEWER
    expires_in_days: int | None = Field(default=None, ge=1, le=3650)


class ApiKeyResponse(BaseModel):
    """A key's metadata. Contains no secret and cannot be made to."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    display_prefix: str
    role: Role
    created_at: datetime
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None


class ApiKeyCreatedResponse(BaseModel):
    """The one and only response that carries a key's secret."""

    key: ApiKeyResponse
    #: Shown once. We store a SHA-256 and a display prefix, so there is no code
    #: path — not even a database dump — that can produce this value again.
    secret: str
