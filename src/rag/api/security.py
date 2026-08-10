"""Credential extraction and the OpenAPI security schemes.

Split from `rag.api.deps` because this is where "what did the caller send?" is
answered, with no I/O and no database. The dependency *chain* that turns the
answer into a scoped transaction and a principal lives next door.

Why a dependency and not middleware
-----------------------------------
Middleware runs before routing, so it cannot know whether the matched route
needs authentication. Using it would force a hand-maintained path allowlist for
`/health`, `/ready`, `/docs`, `/openapi.json`, `/.well-known/jwks.json` and
`/auth/login` — and an allowlist that drifts fails **open**. It also cannot
express per-route permissions and never appears in the OpenAPI schema.

`auto_error=False` is deliberate. FastAPI's `HTTPBearer` answers a missing
`Authorization` header with **403**, which is the wrong code — 403 means "I know
who you are and you may not", and we do not yet know who they are. Turning its
error handling off lets a missing credential raise the same
`AuthenticationError` as a bad one, so every unauthenticated outcome is a 401
carrying `WWW-Authenticate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rag.domain.credentials import CredentialKind, OpaqueCredential, classify
from rag.domain.errors import AuthenticationError

if TYPE_CHECKING:
    from uuid import UUID

    from rag.adapters.auth.tokens import JwtTokenService
    from rag.domain.credentials import TokenClaims

__all__ = [
    "CredentialDep",
    "VerifiedCredential",
    "VerifiedCredentialDep",
    "bearer_scheme",
    "get_credential",
    "get_verified_credential",
]

bearer_scheme = HTTPBearer(
    scheme_name="Bearer",
    description=(
        "A JWT access token, or an API key beginning `ragk_`. Both are presented "
        "in the same `Authorization: Bearer` header and discriminated on the prefix."
    ),
    auto_error=False,
)


@dataclass(frozen=True, slots=True)
class VerifiedCredential:
    """A credential that has survived every check that needs no database.

    For an access token that means the signature, algorithm, `kid`, `typ`,
    `exp`, `nbf`, `aud`, and `iss`. For an API key it means only that the string
    is well formed — the secret has not been checked yet, and cannot be until a
    tenant scope exists to look it up under.

    Either way `tenant_id` is now known, which is the whole point: it is what
    `get_unit_of_work` binds row-level security to before anything reads a row.
    A tenant claimed here and not backed by the credential simply finds nothing.
    """

    kind: CredentialKind
    tenant_id: UUID
    #: Present for an access token.
    claims: TokenClaims | None = None
    #: Present for an API key, carrying the secret still to be verified.
    opaque: OpaqueCredential | None = None


async def get_credential(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> str:
    """Pull the bearer value out of the request. No I/O."""
    if credentials is None or not credentials.credentials:
        raise AuthenticationError()
    return credentials.credentials


CredentialDep = Annotated[str, Depends(get_credential)]


async def get_verified_credential(
    request: Request, credential: CredentialDep
) -> VerifiedCredential:
    """Verify what can be verified without touching the database.

    A refresh token presented here is refused outright. It is a credential for
    exactly one endpoint, and letting it authenticate ordinary requests would
    hand a long-lived token the authority of a short-lived one.
    """
    kind = classify(credential)

    if kind is CredentialKind.API_KEY:
        opaque = OpaqueCredential.parse(credential, expected=CredentialKind.API_KEY)
        return VerifiedCredential(kind=kind, tenant_id=opaque.tenant_id, opaque=opaque)

    if kind is not CredentialKind.ACCESS_TOKEN:
        raise AuthenticationError()

    tokens: JwtTokenService = request.app.state.token_service
    claims = tokens.verify_access_token(credential)
    return VerifiedCredential(kind=kind, tenant_id=claims.tenant_id, claims=claims)


VerifiedCredentialDep = Annotated[VerifiedCredential, Depends(get_verified_credential)]
