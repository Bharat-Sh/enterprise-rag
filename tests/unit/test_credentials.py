"""Credential formats: minting, parsing, and the failures that must stay uniform.

No keys, no clock, no database — the point of keeping the wire format in the
domain is that it is testable with nothing at all.
"""

from __future__ import annotations

import hashlib
from uuid import UUID, uuid4

import pytest

from rag.domain.credentials import (
    DISPLAY_PREFIX_LENGTH,
    CredentialKind,
    OpaqueCredential,
    classify,
    hash_secret,
)
from rag.domain.errors import AuthenticationError


class TestMintingAndParsing:
    def test_a_minted_credential_round_trips(self) -> None:
        tenant_id = uuid4()
        minted = OpaqueCredential.mint(CredentialKind.API_KEY, tenant_id)

        parsed = OpaqueCredential.parse(minted.token, expected=CredentialKind.API_KEY)

        assert parsed.tenant_id == tenant_id
        assert parsed.secret == minted.secret
        assert parsed.lookup_hash == minted.lookup_hash

    def test_the_tenant_is_recoverable_without_any_lookup(self) -> None:
        # The property the whole design rests on: the tenant is readable from
        # the credential alone, so row-level security can be bound before
        # anything is read (docs/adr/0007).
        tenant_id = UUID("3f9c8b1a-0000-4000-8000-000000000001")
        token = OpaqueCredential.mint(CredentialKind.API_KEY, tenant_id).token

        assert OpaqueCredential.parse(token, expected=CredentialKind.API_KEY).tenant_id == tenant_id

    def test_secrets_are_unique_across_mints(self) -> None:
        tenant_id = uuid4()
        secrets = {
            OpaqueCredential.mint(CredentialKind.API_KEY, tenant_id).secret for _ in range(50)
        }

        assert len(secrets) == 50

    def test_the_secret_survives_underscores_from_the_url_safe_alphabet(self) -> None:
        # `token_urlsafe` emits `-` and `_`, and `_` is the field separator.
        # Parsing must split with a bound, or a secret containing one is
        # silently truncated and the key stops working at random.
        crafted = OpaqueCredential(
            kind=CredentialKind.API_KEY, tenant_id=uuid4(), secret="ab_cd_ef-gh"
        )

        parsed = OpaqueCredential.parse(crafted.token, expected=CredentialKind.API_KEY)

        assert parsed.secret == "ab_cd_ef-gh"

    def test_display_prefix_is_a_short_slice_of_the_secret(self) -> None:
        minted = OpaqueCredential.mint(CredentialKind.API_KEY, uuid4())

        assert minted.display_prefix == minted.secret[:DISPLAY_PREFIX_LENGTH]
        assert len(minted.display_prefix) == DISPLAY_PREFIX_LENGTH


class TestParsingFailures:
    """Every malformed shape produces the same error, and it discloses nothing."""

    @pytest.mark.parametrize(
        "token",
        [
            "",
            "ragk",
            "ragk_",
            "ragk_abc",  # no secret segment
            "ragk_ndkbc5jzhffwbcbnasu6qnmhoe_",  # empty secret
            "ragr_ndkbc5jzhffwbcbnasu6qnmhoe_secret",  # right shape, wrong kind
            "ragk_not!valid!base32_secret",
            "ragk_aaaa_secret",  # tenant segment too short to be a UUID
            "eyJhbGciOiJFZERTQSJ9.e30.sig",  # a JWT
        ],
    )
    def test_malformed_credentials_raise_the_uniform_error(self, token: str) -> None:
        with pytest.raises(AuthenticationError) as raised:
            OpaqueCredential.parse(token, expected=CredentialKind.API_KEY)

        # Nothing about *why*: an attacker learning that the prefix was right
        # but the tenant was wrong learns how to fix their next attempt.
        assert raised.value.details == {}
        assert raised.value.code == "unauthenticated"

    def test_a_refresh_token_cannot_be_parsed_as_an_api_key(self) -> None:
        refresh = OpaqueCredential.mint(CredentialKind.REFRESH_TOKEN, uuid4())

        with pytest.raises(AuthenticationError):
            OpaqueCredential.parse(refresh.token, expected=CredentialKind.API_KEY)


class TestClassify:
    def test_an_api_key_is_recognised_by_prefix(self) -> None:
        token = OpaqueCredential.mint(CredentialKind.API_KEY, uuid4()).token

        assert classify(token) is CredentialKind.API_KEY

    def test_a_refresh_token_is_recognised_by_prefix(self) -> None:
        token = OpaqueCredential.mint(CredentialKind.REFRESH_TOKEN, uuid4()).token

        assert classify(token) is CredentialKind.REFRESH_TOKEN

    def test_anything_else_is_treated_as_an_access_token(self) -> None:
        # Deliberately not a fallback chain. Something that is not a recognised
        # opaque prefix goes to exactly one place, and fails there.
        assert classify("eyJhbGciOiJFZERTQSJ9.e30.sig") is CredentialKind.ACCESS_TOKEN
        assert classify("garbage") is CredentialKind.ACCESS_TOKEN


class TestLookupHash:
    def test_the_lookup_hash_is_sha256_of_the_secret(self) -> None:
        assert hash_secret("hello") == hashlib.sha256(b"hello").hexdigest()

    def test_the_token_string_never_contains_the_stored_hash(self) -> None:
        # A database dump holds hashes; if a hash were replayable as a token,
        # the dump would be a set of working credentials.
        minted = OpaqueCredential.mint(CredentialKind.API_KEY, uuid4())

        assert minted.lookup_hash not in minted.token
