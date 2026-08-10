"""Access-token minting and verification.

The interesting tests here are the forgeries. A token library that verifies
correct tokens is easy; the failures below are the ones that have produced real
CVEs in real systems.

Time is controlled by injecting the *issuer's* clock rather than by freezing the
process. A token minted from a clock set in the past is genuinely expired by the
time PyJWT's own real-time check looks at it, so the arithmetic under test is
ours and nothing global is patched.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import jwt
import pytest

from rag.adapters.auth.keys import build_keyring
from rag.adapters.auth.tokens import ACCESS_TOKEN_TYPE, JwtTokenService
from rag.core.config import AuthSettings, SigningAlgorithm
from rag.domain.errors import AuthenticationError
from tests.support import generate_private_pem, public_pem


def _b64u(raw: bytes) -> str:
    """base64url without padding, as every JOSE segment is encoded."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _settings(**overrides: object) -> AuthSettings:
    base: dict[str, object] = {
        "private_key_pem": generate_private_pem(),
        "issuer": "https://rag.test",
        "audience": "rag-api",
    }
    base.update(overrides)
    return AuthSettings(**base)  # type: ignore[arg-type]


def _service(
    settings: AuthSettings | None = None, *, now: datetime | None = None
) -> JwtTokenService:
    settings = settings or _settings()
    clock = (lambda: now) if now is not None else None
    return JwtTokenService(build_keyring(settings), settings, now=clock)


class TestRoundTrip:
    def test_a_minted_token_verifies(self) -> None:
        service = _service()
        subject, tenant_id = uuid4(), uuid4()

        issued = service.issue_access_token(subject=subject, tenant_id=tenant_id)
        claims = service.verify_access_token(issued.token)

        assert claims.subject == subject
        assert claims.tenant_id == tenant_id
        assert claims.jwt_id == issued.jwt_id

    def test_the_header_declares_the_rfc_9068_type_and_a_kid(self) -> None:
        service = _service()

        token = service.issue_access_token(subject=uuid4(), tenant_id=uuid4()).token
        header = jwt.get_unverified_header(token)

        assert header["typ"] == ACCESS_TOKEN_TYPE
        assert header["kid"]

    def test_the_token_carries_no_role_and_no_groups(self) -> None:
        # Both are read from the user's row on every request instead, so a
        # demotion or a group change takes effect immediately rather than at
        # token expiry — and there is one source of truth per input, not two.
        service = _service()

        token = service.issue_access_token(subject=uuid4(), tenant_id=uuid4()).token
        payload = jwt.decode(token, options={"verify_signature": False})

        assert "role" not in payload
        assert "groups" not in payload
        assert set(payload) == {"iss", "sub", "aud", "exp", "iat", "nbf", "jti", "tid"}

    def test_every_token_gets_a_distinct_jti(self) -> None:
        service = _service()
        subject, tenant_id = uuid4(), uuid4()

        first = service.issue_access_token(subject=subject, tenant_id=tenant_id)
        second = service.issue_access_token(subject=subject, tenant_id=tenant_id)

        assert first.jwt_id != second.jwt_id


class TestForgeries:
    def test_a_tampered_payload_is_rejected(self) -> None:
        service = _service()
        token = service.issue_access_token(subject=uuid4(), tenant_id=uuid4()).token
        header, payload, signature = token.split(".")
        forged = f"{header}.{payload[:-2]}xy.{signature}"

        with pytest.raises(AuthenticationError):
            service.verify_access_token(forged)

    def test_the_alg_none_attack_is_rejected(self) -> None:
        """A token asking to be verified with no algorithm at all."""
        service = _service()
        forged = jwt.encode({"sub": str(uuid4())}, key="", algorithm="none")

        with pytest.raises(AuthenticationError):
            service.verify_access_token(forged)

    def test_the_algorithm_confusion_attack_is_rejected(self) -> None:
        """The classic: sign with HMAC, using the *public* key as the secret.

        A verifier that reads `alg` out of the token's own header will validate
        this happily, because it holds the public key and the public key is the
        shared secret. We build the algorithm allowlist from the *resolved key*
        instead, so HS256 is never a candidate.

        Assembled by hand rather than with `jwt.encode`, which refuses to use an
        asymmetric PEM as an HMAC secret — PyJWT will not help you build the
        attack, so a test written through it would prove nothing about *our*
        verifier.
        """
        settings = _settings()
        service = _service(settings)
        keyring = build_keyring(settings)
        public = public_pem(settings.private_key_pem.get_secret_value())  # type: ignore[union-attr]

        now = datetime.now(UTC)
        header = _b64u(
            json.dumps(
                {"typ": ACCESS_TOKEN_TYPE, "alg": "HS256", "kid": keyring.signing_kid}
            ).encode()
        )
        payload = _b64u(
            json.dumps(
                {
                    "iss": settings.issuer,
                    "sub": str(uuid4()),
                    "aud": settings.audience,
                    "tid": str(uuid4()),
                    "exp": int((now + timedelta(hours=1)).timestamp()),
                    "iat": int(now.timestamp()),
                    "nbf": int(now.timestamp()),
                    "jti": "forged",
                }
            ).encode()
        )
        signing_input = f"{header}.{payload}".encode()
        signature = _b64u(hmac.new(public.encode(), signing_input, hashlib.sha256).digest())

        with pytest.raises(AuthenticationError):
            service.verify_access_token(f"{header}.{payload}.{signature}")

    def test_a_token_signed_by_an_unknown_key_is_rejected(self) -> None:
        stranger = _service()
        ours = _service()

        token = stranger.issue_access_token(subject=uuid4(), tenant_id=uuid4()).token

        with pytest.raises(AuthenticationError):
            ours.verify_access_token(token)

    def test_an_unknown_kid_is_rejected_rather_than_searched(self) -> None:
        settings = _settings()
        service = _service(settings)
        token = service.issue_access_token(subject=uuid4(), tenant_id=uuid4()).token
        _, payload, signature = token.split(".")
        # Re-header the token with a kid we do not hold. If verification looped
        # over every key instead of resolving one, this would still pass.
        rewritten = _b64u(b'{"typ":"at+jwt","alg":"EdDSA","kid":"nope"}')

        with pytest.raises(AuthenticationError):
            service.verify_access_token(f"{rewritten}.{payload}.{signature}")

    def test_a_token_with_the_wrong_typ_is_rejected(self) -> None:
        # Stops any other JWT we might ever mint being replayed as an access
        # token.
        settings = _settings()
        keyring = build_keyring(settings)
        service = JwtTokenService(keyring, settings)
        now = datetime.now(UTC)
        other = jwt.encode(
            {
                "iss": settings.issuer,
                "sub": str(uuid4()),
                "aud": settings.audience,
                "tid": str(uuid4()),
                "exp": int((now + timedelta(hours=1)).timestamp()),
                "iat": int(now.timestamp()),
                "nbf": int(now.timestamp()),
                "jti": "other",
            },
            keyring.private_key,
            algorithm=keyring.algorithm.value,
            headers={"typ": "JWT", "kid": keyring.signing_kid},
        )

        with pytest.raises(AuthenticationError):
            service.verify_access_token(other)

    @pytest.mark.parametrize("garbage", ["", "not-a-token", "a.b", "a.b.c.d"])
    def test_structurally_invalid_tokens_are_rejected(self, garbage: str) -> None:
        with pytest.raises(AuthenticationError):
            _service().verify_access_token(garbage)


class TestClaimValidation:
    def test_an_expired_token_is_rejected(self) -> None:
        settings = _settings(access_token_ttl_seconds=60, leeway_seconds=0)
        past = datetime.now(UTC) - timedelta(hours=1)

        issuer = _service(settings, now=past)
        token = issuer.issue_access_token(subject=uuid4(), tenant_id=uuid4()).token

        verifier = JwtTokenService(build_keyring(settings), settings)
        with pytest.raises(AuthenticationError):
            verifier.verify_access_token(token)

    def test_a_token_from_the_future_is_rejected(self) -> None:
        settings = _settings(leeway_seconds=0)
        future = datetime.now(UTC) + timedelta(hours=1)

        issuer = _service(settings, now=future)
        token = issuer.issue_access_token(subject=uuid4(), tenant_id=uuid4()).token

        verifier = JwtTokenService(build_keyring(settings), settings)
        with pytest.raises(AuthenticationError):
            verifier.verify_access_token(token)

    def test_leeway_absorbs_small_clock_skew(self) -> None:
        # Zero leeway makes token validity depend on NTP being perfect on every
        # machine that verifies one.
        settings = _settings(access_token_ttl_seconds=60, leeway_seconds=120)
        slightly_past = datetime.now(UTC) - timedelta(seconds=90)

        issuer = _service(settings, now=slightly_past)
        token = issuer.issue_access_token(subject=uuid4(), tenant_id=uuid4()).token

        JwtTokenService(build_keyring(settings), settings).verify_access_token(token)

    def test_a_token_for_another_audience_is_rejected(self) -> None:
        private = generate_private_pem()
        theirs = _settings(private_key_pem=private, audience="some-other-service")
        ours = _settings(private_key_pem=private, audience="rag-api")

        token = _service(theirs).issue_access_token(subject=uuid4(), tenant_id=uuid4()).token

        # Same key, so the signature is fine. Only the audience check stops it —
        # PyJWT skips that check entirely unless an audience is passed.
        with pytest.raises(AuthenticationError):
            _service(ours).verify_access_token(token)

    def test_a_token_from_another_issuer_is_rejected(self) -> None:
        private = generate_private_pem()
        theirs = _settings(private_key_pem=private, issuer="https://elsewhere.test")
        ours = _settings(private_key_pem=private, issuer="https://rag.test")

        token = _service(theirs).issue_access_token(subject=uuid4(), tenant_id=uuid4()).token

        with pytest.raises(AuthenticationError):
            _service(ours).verify_access_token(token)


class TestKeyring:
    def test_a_missing_key_yields_an_ephemeral_one(self) -> None:
        keyring = build_keyring(AuthSettings())

        assert keyring.ephemeral is True
        assert keyring.signing_kid in keyring.verification

    def test_the_kid_is_stable_for_a_given_key(self) -> None:
        # An RFC 7638 thumbprint, so it survives restarts and is identical on
        # every replica holding the same key — which is what makes rotation a
        # deploy rather than a flag day.
        settings = _settings()

        assert build_keyring(settings).signing_kid == build_keyring(settings).signing_kid

    def test_different_keys_get_different_kids(self) -> None:
        assert build_keyring(_settings()).signing_kid != build_keyring(_settings()).signing_kid

    def test_a_retired_key_still_verifies_its_old_tokens(self) -> None:
        """The whole reason `kid` exists: rotate without invalidating anything."""
        old_private = generate_private_pem()
        old_settings = _settings(private_key_pem=old_private)
        old_token = (
            _service(old_settings).issue_access_token(subject=uuid4(), tenant_id=uuid4()).token
        )

        rotated = _settings(
            private_key_pem=generate_private_pem(),
            retired_public_keys_pem=(public_pem(old_private),),
        )
        service = _service(rotated)

        # New tokens use the new key...
        service.verify_access_token(
            service.issue_access_token(subject=uuid4(), tenant_id=uuid4()).token
        )
        # ...and tokens minted before the changeover keep working.
        service.verify_access_token(old_token)

    def test_dropping_a_retired_key_invalidates_its_tokens(self) -> None:
        old_settings = _settings()
        old_token = (
            _service(old_settings).issue_access_token(subject=uuid4(), tenant_id=uuid4()).token
        )

        with pytest.raises(AuthenticationError):
            _service(_settings()).verify_access_token(old_token)

    def test_rsa_keys_are_supported(self) -> None:
        # The interoperability escape hatch: a verifier that cannot do EdDSA is
        # a configuration change, not a code change.
        settings = _settings(
            private_key_pem=generate_private_pem(SigningAlgorithm.RS256),
            algorithm=SigningAlgorithm.RS256,
        )
        service = _service(settings)

        token = service.issue_access_token(subject=uuid4(), tenant_id=uuid4())

        assert service.verify_access_token(token.token).jwt_id == token.jwt_id
        assert jwt.get_unverified_header(token.token)["alg"] == "RS256"


class TestJwks:
    def test_the_published_set_contains_no_private_material(self) -> None:
        service = _service()

        jwks = service.public_jwks()

        serialised = str(jwks)
        # `d` is the private scalar for OKP and the private exponent for RSA.
        for key in jwks["keys"]:
            assert "d" not in key
            assert set(key) <= {"kty", "crv", "x", "n", "e", "kid", "use", "alg"}
        assert "PRIVATE" not in serialised

    def test_the_published_kid_matches_the_one_in_tokens(self) -> None:
        service = _service()

        token = service.issue_access_token(subject=uuid4(), tenant_id=uuid4()).token

        published = {key["kid"] for key in service.public_jwks()["keys"]}
        assert jwt.get_unverified_header(token)["kid"] in published
