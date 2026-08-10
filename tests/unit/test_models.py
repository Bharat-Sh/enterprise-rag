"""Behaviour that lives on the domain entities themselves.

These are the exact statements of properties the integration tests can only
assert loosely, because they depend on wall-clock boundaries.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from rag.domain.enums import Role, TenantStatus, UserStatus
from rag.domain.models import ApiKey, RefreshToken, Tenant, User

NOW = datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)


def _user(**overrides: object) -> User:
    fields: dict[str, object] = {
        "id": uuid4(),
        "tenant_id": uuid4(),
        "email": "someone@acme.example",
        "full_name": "Someone",
        "role": Role.MEMBER,
        "status": UserStatus.ACTIVE,
        "created_at": NOW,
        "updated_at": NOW,
    }
    fields.update(overrides)
    return User(**fields)  # type: ignore[arg-type]


class TestTokenWatermark:
    """`tokens_valid_after` is what makes a stateless token revocable."""

    def test_no_watermark_accepts_everything(self) -> None:
        assert _user().accepts_token_issued_at(NOW - timedelta(days=365)) is True

    def test_a_token_from_an_earlier_second_is_rejected(self) -> None:
        user = _user(tokens_valid_after=NOW)

        assert user.accepts_token_issued_at(NOW - timedelta(seconds=1)) is False

    def test_a_token_from_a_later_second_is_accepted(self) -> None:
        user = _user(tokens_valid_after=NOW)

        assert user.accepts_token_issued_at(NOW + timedelta(seconds=1)) is True

    def test_a_token_issued_in_the_same_second_survives(self) -> None:
        """The documented one-second boundary, pinned so it cannot drift silently.

        `iat` is a NumericDate and carries whole seconds only, so a token minted
        in the same second as the revocation is genuinely indistinguishable from
        one minted just before it. Accepting is the only option that keeps the
        replacement token a password change hands back from being born dead.
        """
        user = _user(tokens_valid_after=NOW.replace(microsecond=750_000))

        assert user.accepts_token_issued_at(NOW) is True

    def test_the_replacement_token_from_a_password_change_survives(self) -> None:
        # The concrete flow the truncation exists for: the watermark is stamped
        # with microsecond precision, and the token issued microseconds later
        # has an `iat` truncated to the second below it.
        watermark = NOW.replace(microsecond=123_456)
        user = _user(tokens_valid_after=watermark)

        assert user.accepts_token_issued_at(NOW) is True


class TestUserAuthentication:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [(UserStatus.ACTIVE, True), (UserStatus.INVITED, False), (UserStatus.DISABLED, False)],
    )
    def test_only_active_users_authenticate(self, status: UserStatus, expected: bool) -> None:
        assert _user(status=status).can_authenticate is expected


class TestApiKeyUsability:
    def _key(self, **overrides: object) -> ApiKey:
        fields: dict[str, object] = {
            "id": uuid4(),
            "tenant_id": uuid4(),
            "user_id": uuid4(),
            "name": "ci",
            "display_prefix": "abcd1234",
            "role": Role.VIEWER,
            "created_at": NOW,
        }
        fields.update(overrides)
        return ApiKey(**fields)  # type: ignore[arg-type]

    def test_a_key_with_no_expiry_is_usable(self) -> None:
        assert self._key().is_usable_at(NOW + timedelta(days=3650)) is True

    def test_an_expired_key_is_not(self) -> None:
        key = self._key(expires_at=NOW + timedelta(days=1))

        assert key.is_usable_at(NOW + timedelta(days=2)) is False
        assert key.is_usable_at(NOW) is True

    def test_a_revoked_key_is_never_usable_again(self) -> None:
        # Revocation beats expiry: a key revoked today must not work tomorrow
        # just because its `expires_at` is still in the future.
        key = self._key(revoked_at=NOW, expires_at=NOW + timedelta(days=365))

        assert key.is_usable_at(NOW + timedelta(seconds=1)) is False


class TestRefreshTokenUsability:
    def _token(self, **overrides: object) -> RefreshToken:
        fields: dict[str, object] = {
            "id": uuid4(),
            "tenant_id": uuid4(),
            "user_id": uuid4(),
            "family_id": uuid4(),
            "issued_at": NOW,
            "expires_at": NOW + timedelta(days=30),
        }
        fields.update(overrides)
        return RefreshToken(**fields)  # type: ignore[arg-type]

    def test_a_fresh_token_is_usable(self) -> None:
        assert self._token().is_usable_at(NOW) is True

    def test_a_spent_token_is_not(self) -> None:
        # Single use. The second presentation is the theft signal.
        assert self._token(used_at=NOW).is_usable_at(NOW) is False

    def test_a_revoked_token_is_not(self) -> None:
        assert self._token(revoked_at=NOW).is_usable_at(NOW) is False

    def test_an_expired_token_is_not(self) -> None:
        assert self._token().is_usable_at(NOW + timedelta(days=31)) is False


class TestTenant:
    def test_only_an_active_tenant_is_active(self) -> None:
        def _tenant(status: TenantStatus) -> Tenant:
            return Tenant(
                id=uuid4(),
                slug="acme",
                name="Acme",
                status=status,
                created_at=NOW,
                updated_at=NOW,
            )

        assert _tenant(TenantStatus.ACTIVE).is_active is True
        assert _tenant(TenantStatus.SUSPENDED).is_active is False
        assert _tenant(TenantStatus.DELETED).is_active is False
