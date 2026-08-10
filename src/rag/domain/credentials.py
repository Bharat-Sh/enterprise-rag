"""Credential formats and claim shapes.

Pure: `hashlib`, `secrets`, `base64`. The libraries that *sign* tokens and
*stretch* passwords are vendor choices and live in `rag.adapters.auth`; the
question of what a credential looks like on the wire is domain logic, and
keeping it here means it is testable with no keys, no database, and no clock.

The opaque credential layout
----------------------------
::

    ragk_ndkbc5jzhffwbcbnasu6qnmhoe_kQ7t...43 chars...
    ^     ^                          ^
    kind  tenant id, base32          secret, 256 bits from a CSPRNG

**Why the tenant travels inside the credential.** Every table holding user data
is under row-level security (docs/adr/0005), so validating a credential means
reading rows that are invisible until a tenant scope is bound — and the
credential is what names the tenant. Carrying it in the credential breaks the
cycle: bind the scope from the credential, then validate the credential *under*
that scope. A tampered tenant segment needs no comparison to catch it, because
the lookup simply returns nothing. See docs/adr/0007.

**Why base32 for the tenant.** It is case-insensitive and strictly alphanumeric,
so it cannot contain the `_` separator. Base64url would be shorter but its
alphabet includes `_`, which is exactly the character the parser splits on. Hex
would be safe but is six characters longer for no benefit.

**Why the secret is the last segment.** `secrets.token_urlsafe` emits `-` and
`_`. Splitting with a bounded `maxsplit` puts every remaining underscore inside
the secret where it is harmless. A secret in the middle would corrupt parsing.

**Why SHA-256 and not Argon2 for the secret.** A password KDF exists to
compensate for low entropy. These secrets are 256 bits from `secrets`; there is
no dictionary to search and no rainbow table to build, so a slow hash buys
nothing and puts 50-100 ms of CPU on the hottest authentication path in the
system. This reasoning depends entirely on the secret being machine-generated:
the day anyone proposes a user-chosen API key, this decision inverts.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Self
from uuid import UUID

from rag.domain.errors import AuthenticationError

__all__ = [
    "DISPLAY_PREFIX_LENGTH",
    "AccessToken",
    "CredentialKind",
    "OpaqueCredential",
    "TokenClaims",
    "TokenPair",
    "classify",
    "hash_secret",
]

_SEPARATOR = "_"
#: 32 bytes of entropy, rendered as 43 URL-safe base64 characters.
_SECRET_BYTES = 32

#: How much of the secret is stored in clear for display ("ragk_…_kQ7tv2Bx…").
#: Eight of 43 base64url characters is 48 bits given away, leaving over 200 —
#: the same trade GitHub and Stripe make so a key is recognisable in a UI.
DISPLAY_PREFIX_LENGTH = 8


class CredentialKind(StrEnum):
    """How a request proved who it is.

    Recorded on the request principal because it drives real decisions: an API
    key carries a role ceiling, and refusing to let a key change its owner's
    password is the difference between a leaked key and a lost account.
    """

    # Every member trips bandit's hardcoded-password heuristic, which reads the
    # *name* and cannot tell a vocabulary from a secret. These are labels.
    PASSWORD = "password"  # noqa: S105 - only ever seen by the login endpoint
    ACCESS_TOKEN = "access_token"  # noqa: S105
    API_KEY = "api_key"
    REFRESH_TOKEN = "refresh_token"  # noqa: S105


#: Wire prefixes. Distinct per kind so a refresh token presented as an API key
#: is rejected by shape rather than by an unlucky lookup miss, and so secret
#: scanners can match on a literal.
_PREFIXES: dict[CredentialKind, str] = {
    CredentialKind.API_KEY: "ragk",
    CredentialKind.REFRESH_TOKEN: "ragr",
}


def hash_secret(secret: str) -> str:
    """Lookup hash for an opaque credential. Hex-encoded SHA-256.

    The database index equality check *is* the comparison, so there is no
    in-process `compare_digest` here. Index-comparison timing is not
    exploitable: an attacker would have to control the digest being compared,
    which needs a SHA-256 preimage.
    """
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _encode_tenant(tenant_id: UUID) -> str:
    return base64.b32encode(tenant_id.bytes).decode("ascii").rstrip("=").lower()


def _decode_tenant(segment: str) -> UUID:
    padded = segment.upper() + "=" * (-len(segment) % 8)
    return UUID(bytes=base64.b32decode(padded))


@dataclass(frozen=True, slots=True)
class OpaqueCredential:
    """A minted API key or refresh token, in its full plaintext form.

    Exists only between minting and handing the string to the caller. Nothing
    persists it: the database stores `lookup_hash` and `display_prefix`, so a
    dump of `api_keys` cannot be replayed against the API.
    """

    kind: CredentialKind
    tenant_id: UUID
    secret: str

    @classmethod
    def mint(cls, kind: CredentialKind, tenant_id: UUID) -> Self:
        if kind not in _PREFIXES:
            raise ValueError(f"{kind} is not an opaque credential kind")
        return cls(kind=kind, tenant_id=tenant_id, secret=secrets.token_urlsafe(_SECRET_BYTES))

    @property
    def token(self) -> str:
        """The string handed to the caller. Shown once and never recoverable."""
        return _SEPARATOR.join((_PREFIXES[self.kind], _encode_tenant(self.tenant_id), self.secret))

    @property
    def lookup_hash(self) -> str:
        return hash_secret(self.secret)

    @property
    def display_prefix(self) -> str:
        return self.secret[:DISPLAY_PREFIX_LENGTH]

    @classmethod
    def parse(cls, token: str, *, expected: CredentialKind) -> Self:
        """Parse a presented credential. Raises `AuthenticationError` on anything odd.

        The tenant id here is **unverified** — it came from the caller. That is
        safe by construction: it only selects which rows are visible, and the
        secret still has to hash to one of them. See the module docstring.
        """
        prefix = _PREFIXES.get(expected)
        if prefix is None:
            raise ValueError(f"{expected} is not an opaque credential kind")

        parts = token.split(_SEPARATOR, 2)
        # A single generic failure for every malformed shape: telling a caller
        # *how* their credential was wrong tells an attacker how to fix it.
        if len(parts) != 3 or parts[0] != prefix or not parts[2]:
            raise AuthenticationError()

        try:
            # `binascii.Error` subclasses ValueError, so one clause covers both
            # a bad base32 alphabet and a segment of the wrong length.
            tenant_id = _decode_tenant(parts[1])
        except (ValueError, TypeError) as exc:
            raise AuthenticationError() from exc

        return cls(kind=expected, tenant_id=tenant_id, secret=parts[2])


def classify(credential: str) -> CredentialKind:
    """Decide how to interpret an `Authorization: Bearer` value.

    One header for both credential types, discriminated on the prefix — the
    shape Stripe uses. A second header would mean two extraction paths and two
    places to get 401 semantics wrong, and clients would lose their libraries'
    built-in bearer support.

    Anything that is not a recognised opaque prefix is treated as a JWT. There
    is deliberately no fallback chain: "try JWT, then try API key" gives a
    malformed token two unrelated failure paths and an error that tells the
    caller nothing.
    """
    if credential.startswith(_PREFIXES[CredentialKind.API_KEY] + _SEPARATOR):
        return CredentialKind.API_KEY
    if credential.startswith(_PREFIXES[CredentialKind.REFRESH_TOKEN] + _SEPARATOR):
        return CredentialKind.REFRESH_TOKEN
    return CredentialKind.ACCESS_TOKEN


@dataclass(frozen=True, slots=True)
class TokenClaims:
    """The claims we require and trust, after signature verification.

    Deliberately narrow. The access token carries *identity* and nothing that
    authorizes anything: no role, no groups. Both are read from the user's row
    on every request, so a demotion or a group change takes effect immediately
    rather than at token expiry, and there is exactly one source of truth for
    each authorization input.
    """

    subject: UUID
    tenant_id: UUID
    jwt_id: str
    issued_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class AccessToken:
    """A freshly minted access token and the metadata a caller needs."""

    token: str
    expires_at: datetime
    jwt_id: str


@dataclass(frozen=True, slots=True)
class TokenPair:
    """What a successful login or refresh hands back.

    The refresh token appears here in plaintext exactly once — this object is
    built, serialised into the response, and dropped. Only its hash is stored.
    """

    access_token: str
    access_expires_at: datetime
    refresh_token: str
    refresh_expires_at: datetime
    #: RFC 6750. Constant, but clients and generated SDKs read it.
    token_type: str = "Bearer"  # noqa: S105 - a scheme name, not a credential
