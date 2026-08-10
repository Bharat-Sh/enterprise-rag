"""JWT access tokens: minting and verification.

Shape follows RFC 9068 (*JWT Profile for OAuth 2.0 Access Tokens*): a
`typ: "at+jwt"` header and the registered claim set, plus one private claim,
`tid`, for the tenant. Using the profile rather than inventing a layout means
generic JOSE tooling understands our tokens and a future gateway can validate
them without bespoke code.

What the token deliberately does **not** carry
----------------------------------------------
No role. No groups. No email. The user's row is loaded on every request anyway —
to check `status` and to assemble the caller's principal set — so a `role` claim
would save no work while creating a *second* source of truth for an
authorization input. The failure mode is one-directional and silent: a demoted
user keeps their old role until the token expires. Groups are worse still,
because ADR-0006's entire argument is that a membership change takes effect on
the next request without re-indexing anything.

The three verification details that matter
------------------------------------------
1. **The key selects the algorithm, not the token.** `algorithms=` is built from
   the resolved key's own type. Reading `alg` out of the header and trusting it
   is the classic confusion attack: sign with HS256 using the *public* key as
   the HMAC secret, and a verifier that believes the header validates it.
2. **An unknown `kid` is a rejection, never a search.** Trying every key in turn
   would make `kid` decorative and turn verification into a loop an attacker can
   time.
3. **`aud` and `iss` are checked explicitly.** PyJWT silently skips audience
   validation unless an audience is passed, so "we set an `aud` claim" is not
   the same as "we verify it".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import jwt

from rag.core.logging import get_logger
from rag.domain.credentials import AccessToken, TokenClaims
from rag.domain.errors import AuthenticationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from rag.adapters.auth.keys import Keyring, VerificationKey
    from rag.core.config import AuthSettings

__all__ = ["ACCESS_TOKEN_TYPE", "JwtTokenService"]

_log = get_logger(__name__)

#: RFC 9068. Distinguishes an access token from any other JWT we might ever mint,
#: so one cannot be replayed as another.
ACCESS_TOKEN_TYPE = "at+jwt"  # noqa: S105 - a media type, not a credential

#: Claims whose absence is a rejection rather than a default.
_REQUIRED_CLAIMS = ("exp", "iat", "nbf", "sub", "aud", "iss", "jti")


def _utc_now() -> datetime:
    return datetime.now(UTC)


class JwtTokenService:
    """Satisfies both `TokenIssuer` and `TokenVerifier`.

    One object in this process, two ports. The split is not ceremony: a verifier
    needs only public keys, and a deployment that can verify without being able
    to mint is a shape we want to keep reachable.

    `now` is injected so token lifetimes can be tested without freezing global
    time. A test issues a token from a clock set in the past and PyJWT's own
    real-time check then finds it expired — no `freezegun`, no monkeypatching of
    `time.time`, and the arithmetic under test is our own.
    """

    def __init__(
        self,
        keyring: Keyring,
        settings: AuthSettings,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._keyring = keyring
        self._settings = settings
        self._now = now or _utc_now

    def issue_access_token(self, *, subject: UUID, tenant_id: UUID) -> AccessToken:
        issued_at = self._now()
        expires_at = issued_at + timedelta(seconds=self._settings.access_token_ttl_seconds)
        jwt_id = uuid4().hex

        claims: dict[str, Any] = {
            "iss": self._settings.issuer,
            "sub": str(subject),
            "aud": self._settings.audience,
            "exp": int(expires_at.timestamp()),
            "iat": int(issued_at.timestamp()),
            "nbf": int(issued_at.timestamp()),
            "jti": jwt_id,
            "tid": str(tenant_id),
        }
        token = jwt.encode(
            claims,
            self._keyring.private_key,
            algorithm=self._keyring.algorithm.value,
            headers={"typ": ACCESS_TOKEN_TYPE, "kid": self._keyring.signing_kid},
        )
        return AccessToken(token=token, expires_at=expires_at, jwt_id=jwt_id)

    def verify_access_token(self, token: str) -> TokenClaims:
        """Verify and project. Every failure is the same `AuthenticationError`."""
        key = self._resolve_key(token)

        try:
            claims = jwt.decode(
                token,
                key.public_key,
                # Built from the *key*, never from the token's own header.
                algorithms=[key.algorithm.value],
                audience=self._settings.audience,
                issuer=self._settings.issuer,
                leeway=self._settings.leeway_seconds,
                options={"require": list(_REQUIRED_CLAIMS)},
            )
        except jwt.PyJWTError as exc:
            _log.info("auth.token_rejected", reason=type(exc).__name__)
            raise AuthenticationError() from exc

        return self._project(claims)

    def public_jwks(self) -> dict[str, Any]:
        return self._keyring.public_jwks()

    # -- internals ---------------------------------------------------------

    def _resolve_key(self, token: str) -> VerificationKey:
        """Pick the verification key from the token's `kid`.

        Reading an unverified header is safe here because nothing is *trusted*
        from it — the `kid` only selects which key to try, and a wrong guess
        fails the signature check a moment later.
        """
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise AuthenticationError() from exc

        if header.get("typ") != ACCESS_TOKEN_TYPE:
            raise AuthenticationError()

        kid = header.get("kid")
        key = self._keyring.verification.get(kid) if isinstance(kid, str) else None
        if key is None:
            # Not a search through the other keys: see the module docstring.
            _log.info("auth.token_rejected", reason="unknown_kid")
            raise AuthenticationError()
        return key

    def _project(self, claims: dict[str, Any]) -> TokenClaims:
        """Narrow the verified claim set to the fields we actually use."""
        try:
            return TokenClaims(
                subject=UUID(str(claims["sub"])),
                tenant_id=UUID(str(claims["tid"])),
                jwt_id=str(claims["jti"]),
                issued_at=datetime.fromtimestamp(int(claims["iat"]), tz=UTC),
                expires_at=datetime.fromtimestamp(int(claims["exp"]), tz=UTC),
            )
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            # A signed token with a malformed `tid` means our own issuer is
            # broken, or a key we trust has been compromised. Either way the
            # caller learns nothing beyond "no".
            _log.warning("auth.token_malformed", reason=type(exc).__name__)
            raise AuthenticationError() from exc
