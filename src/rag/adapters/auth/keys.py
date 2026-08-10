"""Signing keys, key identifiers, and JWKS publication.

Why asymmetric at all
---------------------
An HMAC secret has to be shared with everything that verifies a token, and
anything that can verify can then also mint. With a signature, the private key
stays in the service that issues tokens and everyone else — an API gateway, a
sidecar, a future worker — verifies from a public key they can fetch. That is
what `/.well-known/jwks.json` exists for.

Why Ed25519 by default
----------------------
No parameters to get wrong: no key size, no PKCS#1-v1.5-versus-PSS padding
choice, no exponent. Signatures are deterministic, so it cannot fail the way
ECDSA does when the random number generator is weak, and they are 64 bytes
rather than RSA's 256 in a header sent on every request.

The known risk is interoperability: some older enterprise verifiers do RS256
and nothing else. The mitigation is structural rather than optimistic — keys
are addressed by `kid` and carry their own algorithm, so adding an RS256 key is
one configuration entry and one JWKS row, with no code change and no flag day.

Why `kid` from the first commit
-------------------------------
Without it, rotating a key means invalidating every outstanding token and
restarting every verifier at the same instant. With it, the new key signs while
the old key still verifies, and the changeover is a deploy. It costs almost
nothing now and cannot be retrofitted compatibly later, because tokens already
in the wild would carry no `kid` to match on.

The identifier is an RFC 7638 JWK thumbprint rather than a random string, so it
is derivable by anyone holding the public key and stays stable across restarts,
across replicas, and across serialisation formats.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa

from rag.core.config import SigningAlgorithm
from rag.core.errors import ConfigurationError
from rag.core.logging import get_logger

if TYPE_CHECKING:
    from rag.core.config import AuthSettings

__all__ = ["Keyring", "VerificationKey", "build_keyring"]

_log = get_logger(__name__)

_PrivateKey = ed25519.Ed25519PrivateKey | rsa.RSAPrivateKey
_PublicKey = ed25519.Ed25519PublicKey | rsa.RSAPublicKey

#: Local-development RSA keys only. Below 2048 the key is not merely weak, it is
#: rejected outright by most verifiers.
_RSA_KEY_SIZE = 2048


def _b64u(data: bytes) -> str:
    """base64url without padding, as every JOSE structure requires."""
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _int_to_b64u(value: int) -> str:
    return _b64u(value.to_bytes((value.bit_length() + 7) // 8 or 1, "big"))


def _public_jwk_material(public_key: _PublicKey) -> dict[str, str]:
    """The key's JWK members, without `kid`, `use`, or `alg`.

    Exactly the members RFC 7638 says take part in the thumbprint, which is why
    the metadata is added afterwards rather than here.
    """
    if isinstance(public_key, ed25519.Ed25519PublicKey):
        raw = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return {"crv": "Ed25519", "kty": "OKP", "x": _b64u(raw)}

    numbers = public_key.public_numbers()
    return {"e": _int_to_b64u(numbers.e), "kty": "RSA", "n": _int_to_b64u(numbers.n)}


def _thumbprint(public_key: _PublicKey) -> str:
    """RFC 7638 JWK thumbprint: SHA-256 over the canonical JSON of the key."""
    material = _public_jwk_material(public_key)
    # Lexicographic key order and no whitespace are both required by the spec —
    # they are what make the thumbprint reproducible across implementations.
    canonical = json.dumps(material, separators=(",", ":"), sort_keys=True)
    return _b64u(hashlib.sha256(canonical.encode("utf-8")).digest())


def _algorithm_for(key: _PrivateKey | _PublicKey) -> SigningAlgorithm:
    if isinstance(key, ed25519.Ed25519PrivateKey | ed25519.Ed25519PublicKey):
        return SigningAlgorithm.EDDSA
    return SigningAlgorithm.RS256


@dataclass(frozen=True, slots=True)
class VerificationKey:
    """A public key and everything needed to check a token signed with it."""

    kid: str
    algorithm: SigningAlgorithm
    public_key: _PublicKey

    def to_jwk(self) -> dict[str, Any]:
        """Public JWK. Contains no private material — asserted by test."""
        jwk = dict(_public_jwk_material(self.public_key))
        jwk["kid"] = self.kid
        jwk["use"] = "sig"
        jwk["alg"] = self.algorithm.value
        return jwk


@dataclass(frozen=True, slots=True)
class Keyring:
    """One key to sign with, and every key we still accept signatures from.

    `verification` always contains the signing key. It also contains retired
    keys during a rotation, which is the whole point: tokens minted before the
    changeover keep working until they expire naturally.
    """

    signing_kid: str
    algorithm: SigningAlgorithm
    private_key: _PrivateKey
    verification: dict[str, VerificationKey]
    #: True when the key was invented at boot rather than configured. Tokens do
    #: not survive a restart, and nothing but local development may do this.
    ephemeral: bool = False

    def public_jwks(self) -> dict[str, Any]:
        return {"keys": [key.to_jwk() for key in self.verification.values()]}


def _load_private_key(settings: AuthSettings) -> _PrivateKey | None:
    """Read the configured private key, if there is one.

    PEM text wins over a path: an operator who sets both has almost certainly
    just switched to injecting the key directly, and silently preferring the
    stale file would be the worse surprise.
    """
    material: bytes | None = None
    if settings.private_key_pem is not None:
        material = settings.private_key_pem.get_secret_value().encode("utf-8")
    elif settings.private_key_path is not None:
        path = Path(settings.private_key_path)
        if not path.is_file():
            raise ConfigurationError(f"auth.private_key_path does not exist: {path}")
        material = path.read_bytes()

    if material is None:
        return None

    try:
        key = serialization.load_pem_private_key(material, password=None)
    except (ValueError, TypeError) as exc:
        # The message deliberately says nothing about the key's contents.
        raise ConfigurationError("The configured signing key is not a readable PEM.") from exc

    if not isinstance(key, ed25519.Ed25519PrivateKey | rsa.RSAPrivateKey):
        raise ConfigurationError(
            f"Unsupported signing key type {type(key).__name__}; expected Ed25519 or RSA."
        )
    return key


def _generate_private_key(algorithm: SigningAlgorithm) -> _PrivateKey:
    if algorithm is SigningAlgorithm.EDDSA:
        return ed25519.Ed25519PrivateKey.generate()
    return rsa.generate_private_key(public_exponent=65537, key_size=_RSA_KEY_SIZE)


def _load_public_key(pem: str) -> _PublicKey:
    try:
        key = serialization.load_pem_public_key(pem.encode("utf-8"))
    except (ValueError, TypeError) as exc:
        raise ConfigurationError("A retired public key is not a readable PEM.") from exc
    if not isinstance(key, ed25519.Ed25519PublicKey | rsa.RSAPublicKey):
        raise ConfigurationError(
            f"Unsupported retired key type {type(key).__name__}; expected Ed25519 or RSA."
        )
    return key


def build_keyring(settings: AuthSettings) -> Keyring:
    """Assemble the process's keyring at startup.

    With no key configured this generates an ephemeral one. That is a
    local-development convenience and nothing else — `Settings` refuses to
    construct at all in a production-like environment without a configured key,
    so this branch is unreachable there. Two replicas generating different keys
    would each reject the other's tokens, and every deploy would log everyone
    out.
    """
    private_key = _load_private_key(settings)
    ephemeral = private_key is None

    if private_key is None:
        private_key = _generate_private_key(settings.algorithm)
        _log.warning(
            "auth.ephemeral_signing_key",
            algorithm=settings.algorithm.value,
            detail=(
                "No signing key configured; generated one for this process. "
                "Tokens will not survive a restart and are not valid on any other replica."
            ),
        )

    algorithm = _algorithm_for(private_key)
    if algorithm is not settings.algorithm:
        # A configuration that says EdDSA while holding an RSA key is a
        # contradiction, and guessing which half the operator meant is worse
        # than refusing. The key wins nothing by default; we simply stop.
        raise ConfigurationError(
            f"auth.algorithm is {settings.algorithm.value} but the configured key is "
            f"{algorithm.value}. Set them consistently."
        )

    public_key = private_key.public_key()
    signing = VerificationKey(
        kid=_thumbprint(public_key), algorithm=algorithm, public_key=public_key
    )
    verification = {signing.kid: signing}

    for pem in settings.retired_public_keys_pem:
        retired_key = _load_public_key(pem)
        retired = VerificationKey(
            kid=_thumbprint(retired_key),
            algorithm=_algorithm_for(retired_key),
            public_key=retired_key,
        )
        # A retired key that is also the signing key is a no-op, not an error:
        # it happens naturally when a rotation is rolled back.
        verification.setdefault(retired.kid, retired)

    _log.info(
        "auth.keyring_ready",
        signing_kid=signing.kid,
        algorithm=algorithm.value,
        verification_kids=sorted(verification),
        ephemeral=ephemeral,
    )
    return Keyring(
        signing_kid=signing.kid,
        algorithm=algorithm,
        private_key=private_key,
        verification=verification,
        ephemeral=ephemeral,
    )
