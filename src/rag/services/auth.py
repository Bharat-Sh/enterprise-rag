"""Authentication and credential lifecycle.

Depends on ports only — a `UnitOfWork`, a `PasswordHasher`, a `TokenIssuer` —
never on SQLAlchemy, PyJWT, or argon2. Enforced by the import-linter contract in
`pyproject.toml`, which is what keeps that sentence true rather than aspirational.

Constructed per request around the unit of work the request already holds, so
the transaction boundary stays the request boundary and this class carries no
state between calls.

The invariant every method here preserves
-----------------------------------------
Row-level security is bound *before* any tenant data is read, from the tenant
named by the credential itself. A credential quoting the wrong tenant therefore
finds nothing — the lookup fails rather than a comparison failing — and there is
no code path that reads `users`, `api_keys`, or `refresh_tokens` unscoped. See
docs/adr/0007.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

from rag.core.logging import get_logger
from rag.domain.access import AuthenticatedPrincipal
from rag.domain.authz import less_privileged_of
from rag.domain.credentials import (
    CredentialKind,
    OpaqueCredential,
    TokenPair,
)
from rag.domain.enums import TenantStatus
from rag.domain.errors import (
    AuthenticationError,
    InvalidInputError,
    NotFoundError,
    PermissionDeniedError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import datetime
    from uuid import UUID

    from rag.core.config import AuthSettings
    from rag.domain.credentials import TokenClaims
    from rag.domain.enums import Role
    from rag.domain.models import ApiKey, User
    from rag.domain.ports import PasswordHasher, TokenIssuer, UnitOfWork

__all__ = ["API_KEY_LAST_USED_THROTTLE", "AuthService"]

_log = get_logger(__name__)

#: How stale `api_keys.last_used_at` may get before we write it again. Five
#: minutes turns a write-per-request into a write-per-key-per-five-minutes and
#: costs nothing anyone reads that timestamp for.
API_KEY_LAST_USED_THROTTLE = timedelta(minutes=5)


class AuthService:
    """Login, refresh, credential issue and revocation, request authentication."""

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        hasher: PasswordHasher,
        issuer: TokenIssuer,
        settings: AuthSettings,
        now: Callable[[], datetime],
    ) -> None:
        self._uow = uow
        self._hasher = hasher
        self._issuer = issuer
        self._settings = settings
        self._now = now

    # -- login and refresh -------------------------------------------------

    async def login(self, *, tenant_slug: str, email: str, password: str) -> TokenPair:
        """Exchange a password for a token pair.

        **Why the tenant arrives in the request body, given non-negotiable #5.**
        That rule governs *authenticated* requests: tenant identity comes from
        the verified token. At login there is no token yet, so the tenant has to
        come from somewhere the caller controls. It is safe because the slug is
        an *addressing* input, not an authorization one — it selects whose
        credential store to check, and a caller naming someone else's tenant
        still has to present that tenant's password. A global email lookup was
        rejected instead: it contradicts the deliberately per-tenant uniqueness
        constraint on `users.email`, and would confirm to one customer whether an
        address exists in another.

        **Every failure looks and costs the same.** Unknown tenant, unknown
        email, wrong password, and disabled account all raise the identical
        error *and* all pay for one Argon2 verification. Skipping the hash when
        there is no user would make "no such account" measurably faster than
        "wrong password" — an enumeration oracle a stopwatch can read.
        """
        now = self._now()
        tenant = await self._uow.tenants.get_by_slug(tenant_slug)

        # A suspended tenant may still sign in: per `TenantStatus`, suspension
        # stops writes and leaves reads working. Only a tombstoned tenant is
        # refused outright.
        if tenant is None or tenant.status is TenantStatus.DELETED:
            await self._hasher.verify(None, password)
            raise AuthenticationError()

        await self._uow.scope_to_tenant(tenant.id)

        user = await self._uow.users.get_by_email(email)
        stored = await self._uow.users.get_password_hash(user.id) if user is not None else None
        password_ok = await self._hasher.verify(stored, password)

        if user is None or not password_ok or not user.can_authenticate:
            _log.info("auth.login_failed", tenant_slug=tenant_slug)
            raise AuthenticationError()

        # Raising the cost parameters is worthless without this: existing users
        # would keep their original, cheaper hashes for ever.
        if stored is not None and self._hasher.needs_rehash(stored):
            await self._uow.users.set_password_hash(user.id, await self._hasher.hash(password))

        await self._uow.users.touch_last_login(user.id, at=now)
        pair, _ = await self._issue_pair(user=user, family_id=uuid4(), now=now)
        await self._uow.commit()

        _log.info("auth.login", user_id=str(user.id), tenant_id=str(tenant.id))
        return pair

    async def refresh(self, presented: str) -> TokenPair:
        """Rotate a refresh token, detecting replay.

        A spent token being presented again means two parties hold a single-use
        credential, so one of them stole it — and there is no way to tell which.
        The response is to revoke the whole rotation family *and* move the
        access-token watermark, which logs both out. The legitimate user signs
        in again; the thief cannot.
        """
        credential = OpaqueCredential.parse(presented, expected=CredentialKind.REFRESH_TOKEN)
        await self._uow.scope_to_tenant(credential.tenant_id)

        now = self._now()
        token = await self._uow.refresh_tokens.get_by_hash(credential.lookup_hash)
        if token is None:
            raise AuthenticationError()

        if token.used_at is not None or token.revoked_at is not None:
            await self._uow.refresh_tokens.revoke_family(token.family_id, at=now)
            # Moves the access-token watermark too: the thief may already hold a
            # live access token, and revoking only the refresh lineage would
            # leave it working until it expired. The watermark rejects anything
            # issued in an earlier second — see `User.accepts_token_issued_at`
            # for why the boundary is one second and not exact.
            await self._uow.users.invalidate_tokens_before(token.user_id, at=now)
            await self._uow.commit()
            _log.warning(
                "auth.refresh_token_replayed",
                user_id=str(token.user_id),
                family_id=str(token.family_id),
            )
            raise AuthenticationError()

        if token.expires_at <= now:
            raise AuthenticationError()

        user = await self._uow.users.get(token.user_id)
        if user is None or not user.can_authenticate:
            raise AuthenticationError()

        pair, replacement_id = await self._issue_pair(user=user, family_id=token.family_id, now=now)

        # Consumed *after* the replacement exists, because `replaced_by` needs
        # its id. A compare-and-set, so two requests racing with one token
        # cannot both succeed — the detector must not be the thing with a race.
        consumed = await self._uow.refresh_tokens.mark_used(
            token.id, at=now, replaced_by=replacement_id
        )
        if not consumed:
            await self._uow.rollback()
            raise AuthenticationError()

        await self._uow.commit()
        return pair

    async def logout(self, presented: str) -> None:
        """Revoke one device's rotation family.

        Not a global logout: a family is one client's lineage, and signing out
        of a laptop should not sign out a phone. `logout_everywhere` is the
        blunt instrument, and it is a different action.

        The caller's *access* token stays valid until it expires. That is the
        bounded, deliberate cost of stateless tokens — bounded by
        `access_token_ttl_seconds`, which is why it is fifteen minutes.
        """
        credential = OpaqueCredential.parse(presented, expected=CredentialKind.REFRESH_TOKEN)
        await self._uow.scope_to_tenant(credential.tenant_id)

        token = await self._uow.refresh_tokens.get_by_hash(credential.lookup_hash)
        if token is None:
            # Idempotent: logging out with a token we do not recognise is a
            # success, not a 401. There is nothing to protect and nothing to
            # leak, and clients retry this call on flaky networks.
            return

        await self._uow.refresh_tokens.revoke_family(token.family_id, at=self._now())
        await self._uow.commit()

    async def logout_everywhere(self, user: User) -> None:
        """Invalidate every session for a user, in both directions.

        Both halves are needed and neither is redundant: revoking refresh tokens
        stops new access tokens being minted, and moving the watermark stops the
        ones already issued from being honoured. Doing only the first leaves a
        stolen access token working for up to its full lifetime.
        """
        now = self._now()
        await self._uow.refresh_tokens.revoke_for_user(user.id, at=now)
        await self._uow.users.invalidate_tokens_before(user.id, at=now)
        await self._uow.commit()

    # -- request authentication --------------------------------------------

    async def authenticate_access_token(self, claims: TokenClaims) -> AuthenticatedPrincipal:
        """Resolve a verified token into a principal.

        The unit of work is already scoped to `claims.tenant_id` by the time this
        runs, so a token carrying a forged tenant finds no user and fails here
        rather than being caught by a comparison — see docs/adr/0007.
        """
        user = await self._uow.users.get(claims.subject)
        if user is None or not user.can_authenticate:
            raise AuthenticationError()

        # Row-level security has already guaranteed this. Checked anyway: it
        # costs nothing, and it is the assertion that would catch a policy
        # accidentally disabled by a future migration.
        if user.tenant_id != claims.tenant_id:  # pragma: no cover - RLS makes this unreachable
            _log.error("auth.tenant_mismatch", user_id=str(user.id))
            raise AuthenticationError()

        if not user.accepts_token_issued_at(claims.issued_at):
            raise AuthenticationError()

        return AuthenticatedPrincipal(
            user=user,
            tenant_id=user.tenant_id,
            effective_role=user.role,
            access=await self._uow.users.access_filter_for(user),
            credential=CredentialKind.ACCESS_TOKEN,
        )

    async def authenticate_api_key(self, credential: OpaqueCredential) -> AuthenticatedPrincipal:
        """Resolve a presented API key into a principal.

        The key's role is a **ceiling**, and it is applied before the access
        filter is built. Applying it only to route permissions would narrow what
        the key may *call* while leaving what it may *read* untouched, which is
        the more damaging half.
        """
        now = self._now()
        key = await self._uow.api_keys.get_by_hash(credential.lookup_hash)
        if key is None or not key.is_usable_at(now):
            raise AuthenticationError()

        user = await self._uow.users.get(key.user_id)
        if user is None or not user.can_authenticate:
            # A key outliving its owner's account is exactly the artifact that
            # turns up in breach post-mortems.
            raise AuthenticationError()

        effective_role = less_privileged_of(user.role, key.role)

        wrote = await self._uow.api_keys.touch_last_used(
            key.id, at=now, stale_after=API_KEY_LAST_USED_THROTTLE
        )
        if wrote:
            # Committed here rather than left to the handler: a read-only
            # handler never commits, and the touch would be rolled back on
            # exactly the requests that are most common. Skipped when nothing
            # was written, which is the overwhelming majority of calls.
            await self._uow.commit()

        return AuthenticatedPrincipal(
            user=user,
            tenant_id=user.tenant_id,
            effective_role=effective_role,
            access=await self._uow.users.access_filter_for(user, role=effective_role),
            credential=CredentialKind.API_KEY,
            api_key_id=key.id,
        )

    # -- credential management ---------------------------------------------

    async def create_api_key(
        self,
        principal: AuthenticatedPrincipal,
        *,
        name: str,
        role: Role,
        expires_in_days: int | None = None,
    ) -> tuple[ApiKey, str]:
        """Mint a key for the calling user. Returns the row and the one-time secret.

        **An API key may not mint another API key.** Otherwise a leaked key is a
        foothold that renews itself: the attacker issues a fresh key, and
        revoking the one that leaked achieves nothing.

        A caller cannot request a role above their own. Rejected rather than
        silently narrowed — asking for admin as a member is a mistake, and
        quietly issuing something weaker than requested is how a deploy script
        fails a fortnight later for no visible reason.
        """
        self._require_interactive(principal, action="create an API key")

        if less_privileged_of(principal.effective_role, role) is not role:
            raise PermissionDeniedError(
                f"You cannot issue a key with the {role.value!r} role; "
                f"your own role is {principal.effective_role.value!r}.",
                details={"requested_role": role.value, "your_role": principal.effective_role.value},
            )

        now = self._now()
        ttl_days = (
            expires_in_days
            if expires_in_days is not None
            else (self._settings.api_key_default_ttl_days)
        )
        expires_at = now + timedelta(days=ttl_days) if ttl_days is not None else None

        credential = OpaqueCredential.mint(CredentialKind.API_KEY, principal.tenant_id)
        key = await self._uow.api_keys.create(
            tenant_id=principal.tenant_id,
            user_id=principal.user.id,
            name=name,
            display_prefix=credential.display_prefix,
            secret_hash=credential.lookup_hash,
            role=role,
            created_by=principal.user.id,
            expires_at=expires_at,
        )
        await self._uow.commit()

        _log.info("auth.api_key_created", key_id=str(key.id), user_id=str(principal.user.id))
        return key, credential.token

    async def list_api_keys(self, principal: AuthenticatedPrincipal) -> Sequence[ApiKey]:
        return await self._uow.api_keys.list_for_user(principal.user.id)

    async def revoke_api_key(self, principal: AuthenticatedPrincipal, key_id: UUID) -> None:
        """Revoke one of the caller's own keys.

        A key that belongs to another user, or to another tenant, raises
        `NotFoundError` rather than `PermissionDeniedError`. 403 would confirm
        the key exists, which is the enumeration oracle CLAUDE.md calls out.
        """
        key = await self._uow.api_keys.get(key_id)
        if key is None or key.user_id != principal.user.id:
            raise NotFoundError("API key", str(key_id))

        await self._uow.api_keys.revoke(key_id, at=self._now())
        await self._uow.commit()
        _log.info("auth.api_key_revoked", key_id=str(key_id), user_id=str(principal.user.id))

    async def change_password(
        self,
        principal: AuthenticatedPrincipal,
        *,
        current_password: str,
        new_password: str,
    ) -> TokenPair:
        """Replace a password and re-issue the caller's session.

        Every other session dies: their refresh tokens are revoked outright, and
        the access-token watermark moves so their live tokens stop being honoured
        too. The reason anyone changes a password is that it might be
        compromised, so leaving existing sessions alive defeats the act.

        The caller gets a fresh pair back, so the one client that did the right
        thing is not signed out for its trouble. That is also why the watermark
        comparison is second-resolution rather than exact — see
        `User.accepts_token_issued_at`.
        """
        self._require_interactive(principal, action="change a password")

        stored = await self._uow.users.get_password_hash(principal.user.id)
        if not await self._hasher.verify(stored, current_password):
            raise AuthenticationError()

        minimum = self._settings.password_min_length
        if len(new_password) < minimum:
            # Length, not composition. NIST 800-63B: complexity rules push people
            # towards predictable substitutions and yield less entropy, not more.
            raise InvalidInputError(
                f"A password must be at least {minimum} characters.",
                details={"minimum_length": minimum},
            )

        now = self._now()
        await self._uow.users.set_password_hash(
            principal.user.id, await self._hasher.hash(new_password)
        )
        await self._uow.refresh_tokens.revoke_for_user(principal.user.id, at=now)
        await self._uow.users.invalidate_tokens_before(principal.user.id, at=now)

        pair, _ = await self._issue_pair(user=principal.user, family_id=uuid4(), now=now)
        await self._uow.commit()

        _log.info("auth.password_changed", user_id=str(principal.user.id))
        return pair

    # -- internals ---------------------------------------------------------

    def _require_interactive(self, principal: AuthenticatedPrincipal, *, action: str) -> None:
        """Refuse account-level changes to a machine credential."""
        if principal.is_api_key:
            raise PermissionDeniedError(
                f"An API key may not {action}. Use an interactive session.",
                details={"credential": principal.credential.value},
            )

    async def _issue_pair(
        self, *, user: User, family_id: UUID, now: datetime
    ) -> tuple[TokenPair, UUID]:
        """Mint an access token and a refresh token in one rotation family.

        Returns the pair and the id of the new refresh row, which `refresh()`
        needs for `replaced_by`. Returned rather than stashed on `self`: this
        object is per-request, but hidden state between two method calls is a
        bug waiting for the first caller who invokes them out of order.

        The access token is issued *after* the refresh row exists, so a failure
        writing the row can never leave a usable access token with no way to
        renew it.
        """
        credential = OpaqueCredential.mint(CredentialKind.REFRESH_TOKEN, user.tenant_id)
        refresh_expires_at = now + timedelta(seconds=self._settings.refresh_token_ttl_seconds)

        row = await self._uow.refresh_tokens.create(
            tenant_id=user.tenant_id,
            user_id=user.id,
            token_hash=credential.lookup_hash,
            family_id=family_id,
            expires_at=refresh_expires_at,
        )

        access = self._issuer.issue_access_token(subject=user.id, tenant_id=user.tenant_id)
        pair = TokenPair(
            access_token=access.token,
            access_expires_at=access.expires_at,
            refresh_token=credential.token,
            refresh_expires_at=refresh_expires_at,
        )
        return pair, row.id
