"""FastAPI dependency wiring.

Dependencies are declared as `Annotated` aliases so routers read as
`settings: SettingsDep` rather than repeating `Depends(...)` at every call site.
Overriding one of these in tests (via `app.dependency_overrides`) then swaps the
implementation everywhere at once.

The authentication chain
------------------------
::

    get_credential           header only, no I/O
        v
    get_verified_credential  signature / algorithm / kid / exp / aud / iss,
        |                    or an API key's shape. Yields a tenant id.
        v
    get_unit_of_work         opens the transaction and IMMEDIATELY binds
        |                    row-level security to that tenant
        v
    get_principal            under that scope: user row, statuses, groups,
                             API-key ceiling -> AccessFilter

The ordering is enforced by the dependency graph rather than by convention, and
it produces the property M2 exists for: **a database transaction cannot be
obtained without a verified tenant scope.** A new route that forgets
authentication does not compile against `UnitOfWorkDep`, so the omission is a
type error rather than a silent hole.

The two endpoints that genuinely predate a token — login and refresh — use
`UnscopedUnitOfWorkDep`, which is named to be conspicuous and is asserted by
test to have exactly those call sites.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Request

# Imported at runtime, not under TYPE_CHECKING, and this is load-bearing:
# FastAPI resolves every dependency annotation with `get_type_hints` when a
# route is registered. A name that exists only for the type checker cannot be
# resolved, and the parameter is silently reinterpreted as a *query parameter* —
# which surfaces as a 422 on a route that looked perfectly correct.
from rag.adapters.auth.passwords import Argon2PasswordHasher
from rag.adapters.auth.tokens import JwtTokenService
from rag.adapters.ratelimit.inprocess import InProcessRateLimiter
from rag.api.security import VerifiedCredentialDep
from rag.core.config import Settings, get_settings
from rag.core.context import set_tenant_id
from rag.core.errors import DependencyUnavailableError
from rag.core.health import HealthRegistry
from rag.core.logging import get_logger
from rag.db.uow import SqlAlchemyUnitOfWork
from rag.domain.access import AuthenticatedPrincipal
from rag.domain.authz import Permission, permits
from rag.domain.errors import AuthenticationError, PermissionDeniedError, RateLimitExceededError
from rag.domain.ratelimit import RateLimitDecision, RateLimitPolicy
from rag.services.auth import AuthService

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_log = get_logger(__name__)


def get_app_settings(request: Request) -> Settings:
    """Return the settings bound to this application instance.

    Reads from `app.state` rather than calling `get_settings()` directly, so a
    test that builds an app with custom settings gets those settings rather than
    whatever the cached process-wide singleton holds.
    """
    settings = getattr(request.app.state, "settings", None)
    if isinstance(settings, Settings):
        return settings
    return get_settings()


def get_health_registry(request: Request) -> HealthRegistry:
    """Return the application's health-check registry."""
    registry = getattr(request.app.state, "health", None)
    if not isinstance(registry, HealthRegistry):  # pragma: no cover - lifespan guarantees it
        raise DependencyUnavailableError(
            "health-registry",
            "The health registry was not initialised; the application did not start cleanly.",
        )
    return registry


def utc_now() -> datetime:
    """The application clock. A dependency so tests can move it."""
    return datetime.now(UTC)


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
HealthRegistryDep = Annotated[HealthRegistry, Depends(get_health_registry)]


# --- unit of work ----------------------------------------------------------


def _session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    factory = getattr(request.app.state, "db_session_factory", None)
    if factory is None:  # pragma: no cover - lifespan guarantees it
        raise DependencyUnavailableError(
            "postgres", "The database was not initialised; the application did not start cleanly."
        )
    return factory  # type: ignore[no-any-return]  # set by lifespan, untyped app.state


async def get_unit_of_work(
    request: Request, credential: VerifiedCredentialDep
) -> AsyncIterator[SqlAlchemyUnitOfWork]:
    """Yield a unit of work already scoped to the credential's tenant.

    The `async with` guarantees the session is closed and any uncommitted work
    rolled back even if the handler raises, so no route can leak a connection
    back to the pool mid-transaction.

    The scope is bound here, before a single handler statement runs, and it is
    bound from the *verified* credential — never from a body, query parameter,
    or header. That closes the gap M1 left open, where the row-level-security
    scope was exercised only by tests.
    """
    async with SqlAlchemyUnitOfWork(_session_factory(request)) as uow:
        await uow.scope_to_tenant(credential.tenant_id)
        # Also attach it to the ambient log context, so every line emitted for
        # the rest of this request carries the tenant.
        set_tenant_id(str(credential.tenant_id))
        yield uow


async def get_unscoped_unit_of_work(request: Request) -> AsyncIterator[SqlAlchemyUnitOfWork]:
    """A unit of work with **no tenant bound**. Two legitimate call sites.

    Login and refresh have to read before a credential has been verified, so
    they resolve the tenant themselves — login from the submitted slug, refresh
    from the token's own tenant segment — and scope the transaction as their
    first act. Every other endpoint uses `UnitOfWorkDep`.

    Deliberately conspicuous. An unscoped session sees nothing at all thanks to
    the fail-closed RLS predicate, so the risk is not disclosure but confusion;
    the name is here to make a third call site something a reviewer notices.
    """
    async with SqlAlchemyUnitOfWork(_session_factory(request)) as uow:
        yield uow


UnitOfWorkDep = Annotated[SqlAlchemyUnitOfWork, Depends(get_unit_of_work)]
UnscopedUnitOfWorkDep = Annotated[SqlAlchemyUnitOfWork, Depends(get_unscoped_unit_of_work)]


# --- authentication --------------------------------------------------------


def get_token_service(request: Request) -> JwtTokenService:
    service = getattr(request.app.state, "token_service", None)
    if service is None:  # pragma: no cover - lifespan guarantees it
        raise DependencyUnavailableError("token-service", "Token signing was not initialised.")
    return service  # type: ignore[no-any-return]


def get_password_hasher(request: Request) -> Argon2PasswordHasher:
    hasher = getattr(request.app.state, "password_hasher", None)
    if hasher is None:  # pragma: no cover - lifespan guarantees it
        raise DependencyUnavailableError("password-hasher", "Password hashing was not initialised.")
    return hasher  # type: ignore[no-any-return]


def get_rate_limiter(request: Request) -> InProcessRateLimiter:
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:  # pragma: no cover - lifespan guarantees it
        raise DependencyUnavailableError("rate-limiter", "Rate limiting was not initialised.")
    return limiter  # type: ignore[no-any-return]


TokenServiceDep = Annotated[JwtTokenService, Depends(get_token_service)]
PasswordHasherDep = Annotated[Argon2PasswordHasher, Depends(get_password_hasher)]
RateLimiterDep = Annotated[InProcessRateLimiter, Depends(get_rate_limiter)]


def _auth_service(
    uow: SqlAlchemyUnitOfWork,
    settings: Settings,
    hasher: Argon2PasswordHasher,
    tokens: JwtTokenService,
) -> AuthService:
    return AuthService(uow, hasher=hasher, issuer=tokens, settings=settings.auth, now=utc_now)


def get_auth_service(
    uow: UnitOfWorkDep,
    settings: SettingsDep,
    hasher: PasswordHasherDep,
    tokens: TokenServiceDep,
) -> AuthService:
    """An `AuthService` over the request's *scoped* transaction."""
    return _auth_service(uow, settings, hasher, tokens)


def get_unscoped_auth_service(
    uow: UnscopedUnitOfWorkDep,
    settings: SettingsDep,
    hasher: PasswordHasherDep,
    tokens: TokenServiceDep,
) -> AuthService:
    """An `AuthService` for login and refresh, which bind their own scope."""
    return _auth_service(uow, settings, hasher, tokens)


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
UnscopedAuthServiceDep = Annotated[AuthService, Depends(get_unscoped_auth_service)]


async def get_principal(
    credential: VerifiedCredentialDep, service: AuthServiceDep
) -> AuthenticatedPrincipal:
    """Resolve the caller, under the tenant scope already bound.

    Every authenticated request pays for one read of `users` and one of
    `group_members` here. That is the deliberate consequence of keeping role and
    group membership *out* of the token: a demotion or a group change takes
    effect on the next request instead of at token expiry, and there is one
    source of truth per authorization input rather than two that can disagree.
    It is also what makes a stateless token revocable, since the row we load
    carries the `tokens_valid_after` watermark. M9 caches it.
    """
    if credential.claims is not None:
        return await service.authenticate_access_token(credential.claims)
    if credential.opaque is not None:
        return await service.authenticate_api_key(credential.opaque)
    raise AuthenticationError()  # pragma: no cover - VerifiedCredential always sets one


PrincipalDep = Annotated[AuthenticatedPrincipal, Depends(get_principal)]


# --- authorization ---------------------------------------------------------


def require(
    permission: Permission,
) -> Callable[[AuthenticatedPrincipal], Coroutine[Any, Any, AuthenticatedPrincipal]]:
    """Gate a route on a named action.

    Routes name the *action*, not the role, so "who may do this" changes in one
    row of `rag.domain.authz.MINIMUM_ROLE` rather than at every call site. The
    policy is a table in the domain and is unit-testable with no HTTP at all.

    403 rather than 404 here: the caller is authenticated and the endpoint's
    existence is not a secret. Cross-*tenant* lookups are the opposite case and
    raise `NotFoundError` — a 403 there would confirm the resource exists.
    """

    async def dependency(principal: PrincipalDep) -> AuthenticatedPrincipal:
        if not permits(principal.effective_role, permission):
            raise PermissionDeniedError(
                f"This action requires more privilege than the {principal.effective_role.value!r} "
                f"role holds.",
                details={
                    "permission": permission.value,
                    "your_role": principal.effective_role.value,
                    # Named so an operator can see when an API key's ceiling,
                    # rather than the user's own role, is what denied the call.
                    "credential": principal.credential.value,
                },
            )
        return principal

    return dependency


# --- rate limiting ---------------------------------------------------------


def _client_key(request: Request) -> str:
    """Identify an unauthenticated caller.

    `request.client.host` only. `X-Forwarded-For` is caller-controlled unless a
    trusted proxy overwrites it, so trusting it here would let an attacker reset
    their own bucket by inventing a new value per attempt — a rate limiter that
    is worse than none, because it looks like protection. Proxy-aware client
    resolution belongs at the ingress, via uvicorn's `--proxy-headers` and a
    trusted-host list.
    """
    return request.client.host if request.client else "unknown"


async def enforce_tenant_rate_limit(
    principal: PrincipalDep, limiter: RateLimiterDep, settings: SettingsDep
) -> None:
    """Per-tenant throttling, applied after authentication.

    Keyed on the tenant rather than the client address: the resource being
    protected is per-tenant capacity and cost, and an address is meaningless
    behind NAT and free to rotate. The consequence is that a rejected request
    has already paid for token verification and one database round trip —
    acceptable, because volumetric abuse is the ingress's job and this limiter
    exists for fairness and cost control between tenants.
    """
    if not settings.rate_limit.enabled:
        return

    policy = RateLimitPolicy.per_minute(
        settings.rate_limit.tenant_requests_per_minute,
        burst=settings.rate_limit.tenant_burst,
    )
    decision = await _decide(limiter, f"tenant:{principal.tenant_id}", policy)
    if not decision.allowed:
        raise RateLimitExceededError(
            retry_after_seconds=decision.retry_after_seconds,
            limit=decision.limit,
            scope="tenant",
        )


async def enforce_login_rate_limit(
    request: Request, limiter: RateLimiterDep, settings: SettingsDep
) -> None:
    """Throttling for the endpoints that have no tenant yet.

    A per-tenant limiter cannot protect a pre-authentication endpoint, and login
    is the credential-stuffing surface, so it gets its own much smaller bucket
    keyed on the client address.
    """
    if not settings.rate_limit.enabled:
        return

    policy = RateLimitPolicy.per_minute(
        settings.rate_limit.login_attempts_per_minute,
        burst=settings.rate_limit.login_burst,
    )
    decision = await _decide(limiter, f"login:{_client_key(request)}", policy)
    if not decision.allowed:
        raise RateLimitExceededError(
            retry_after_seconds=decision.retry_after_seconds,
            limit=decision.limit,
            scope="login",
        )


async def _decide(
    limiter: InProcessRateLimiter, key: str, policy: RateLimitPolicy
) -> RateLimitDecision:
    """Consult the limiter, **failing open** on any error.

    Losing rate limiting costs fairness. Failing closed on a limiter outage
    costs the entire API — the same posture as
    `RedisSettings.required_for_readiness`, which is False for exactly this
    reason. The error is logged loudly so the degradation is visible rather than
    silent, and the broad `except` is the point: a limiter must never be the
    reason a legitimate request fails.
    """
    try:
        return await limiter.check(key, policy)
    except Exception:
        _log.error("ratelimit.unavailable", key=key, exc_info=True)
        return RateLimitDecision.allow(remaining=policy.capacity, limit=policy.requests_per_minute)


#: Convenience aliases for router `dependencies=[...]` lists.
TenantRateLimit = Depends(enforce_tenant_rate_limit)
LoginRateLimit = Depends(enforce_login_rate_limit)
