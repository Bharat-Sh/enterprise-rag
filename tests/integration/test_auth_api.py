"""Login, refresh, logout, and password change, end to end over HTTP.

Against the real application and the real database, because the properties
being asserted — that a scope is bound before a handler runs, that a rotation
family is revoked in one transaction — are properties of the whole stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rag.domain.enums import UserStatus
from tests.integration.conftest import TEST_PASSWORD, bearer, login, requires_postgres

if TYPE_CHECKING:
    from httpx import AsyncClient

    from rag.db.uow import SqlAlchemyUnitOfWork
    from rag.domain.models import Tenant, User

pytestmark = [pytest.mark.integration, requires_postgres]


class TestLogin:
    async def test_valid_credentials_return_a_token_pair(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        body = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        assert body["token_type"] == "Bearer"
        assert body["access_token"]
        assert body["refresh_token"].startswith("ragr_")
        assert body["access_expires_at"] < body["refresh_expires_at"]

    async def test_the_token_works_against_a_protected_route(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        body = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        response = await api_client.get("/api/v1/auth/me", headers=bearer(body["access_token"]))

        assert response.status_code == 200
        assert response.json()["user"]["email"] == member.email
        assert response.json()["tenant_id"] == str(tenant.id)
        assert response.json()["credential"] == "access_token"

    @pytest.mark.parametrize(
        ("slug", "email", "password", "case"),
        [
            ("acme", "member@acme.example", "wrong-password", "wrong password"),
            ("acme", "nobody@acme.example", TEST_PASSWORD, "unknown email"),
            ("no-such-tenant", "member@acme.example", TEST_PASSWORD, "unknown tenant"),
        ],
    )
    async def test_every_failure_is_the_same_response(
        self,
        api_client: AsyncClient,
        tenant: Tenant,
        member: User,
        slug: str,
        email: str,
        password: str,
        case: str,
    ) -> None:
        """The three cases must be indistinguishable to the caller.

        Any difference between them — status, code, wording, an extra detail
        field — tells an attacker which half of the guess was right, which is
        the whole game in credential stuffing.
        """
        response = await api_client.post(
            "/api/v1/auth/login",
            json={"tenant_slug": slug, "email": email, "password": password},
        )

        assert response.status_code == 401, case
        body = response.json()
        assert body["code"] == "unauthenticated", case
        assert body["detail"] == "Authentication failed.", case
        assert "errors" not in body, case

    async def test_a_401_carries_www_authenticate(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        # RFC 9110 requires it. Without it a client cannot discover the scheme,
        # and generated SDKs will not attempt a refresh.
        response = await api_client.post(
            "/api/v1/auth/login",
            json={"tenant_slug": tenant.slug, "email": member.email, "password": "nope"},
        )

        assert response.headers["www-authenticate"].startswith("Bearer")

    async def test_a_disabled_user_cannot_log_in(
        self, api_client: AsyncClient, uow: SqlAlchemyUnitOfWork, tenant: Tenant, member: User
    ) -> None:
        await uow.users.set_status(member.id, UserStatus.DISABLED)
        await uow.commit()

        response = await api_client.post(
            "/api/v1/auth/login",
            json={
                "tenant_slug": tenant.slug,
                "email": member.email,
                "password": TEST_PASSWORD,
            },
        )

        assert response.status_code == 401

    async def test_an_invited_user_with_no_password_cannot_log_in(
        self, api_client: AsyncClient, uow: SqlAlchemyUnitOfWork, tenant: Tenant
    ) -> None:
        # `password_hash` is nullable for SSO-provisioned accounts. There must be
        # no code path that treats "no password" as "any password".
        await uow.users.create(
            tenant_id=tenant.id, email="invited@acme.example", status=UserStatus.INVITED
        )
        await uow.commit()

        response = await api_client.post(
            "/api/v1/auth/login",
            json={
                "tenant_slug": tenant.slug,
                "email": "invited@acme.example",
                "password": TEST_PASSWORD,
            },
        )

        assert response.status_code == 401

    async def test_login_records_last_login(
        self, api_client: AsyncClient, uow: SqlAlchemyUnitOfWork, tenant: Tenant, member: User
    ) -> None:
        assert member.last_login_at is None

        await login(api_client, tenant_slug=tenant.slug, email=member.email)

        await uow.rollback()  # see the other transaction's committed work
        assert (await uow.users.get(member.id)).last_login_at is not None  # type: ignore[union-attr]


class TestUnauthenticatedAccess:
    async def test_a_protected_route_without_a_token_is_401(self, api_client: AsyncClient) -> None:
        # Not 403, which is FastAPI's `HTTPBearer` default and the wrong code:
        # 403 means "I know who you are and you may not".
        response = await api_client.get("/api/v1/auth/me")

        assert response.status_code == 401
        assert response.json()["code"] == "unauthenticated"

    @pytest.mark.parametrize(
        "credential", ["garbage", "ragk_bad_key", "eyJhbGciOiJub25lIn0.e30.", ""]
    )
    async def test_malformed_credentials_are_401(
        self, api_client: AsyncClient, credential: str
    ) -> None:
        response = await api_client.get("/api/v1/auth/me", headers=bearer(credential))

        assert response.status_code == 401

    async def test_a_refresh_token_cannot_authenticate_an_ordinary_request(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        # Refresh tokens are long lived. Letting one act as an access token
        # would hand it authority it was never meant to carry.
        body = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        response = await api_client.get("/api/v1/auth/me", headers=bearer(body["refresh_token"]))

        assert response.status_code == 401


class TestRefresh:
    async def test_a_refresh_token_yields_a_new_pair(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        first = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        response = await api_client.post(
            "/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}
        )

        assert response.status_code == 200
        second = response.json()
        assert second["refresh_token"] != first["refresh_token"]
        assert (
            await api_client.get("/api/v1/auth/me", headers=bearer(second["access_token"]))
        ).status_code == 200

    async def test_a_spent_refresh_token_cannot_be_used_again(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        first = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        await api_client.post(
            "/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}
        )

        replay = await api_client.post(
            "/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}
        )

        assert replay.status_code == 401

    async def test_replaying_a_spent_token_revokes_the_whole_family(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        """The reuse-detection response.

        Two parties holding a single-use credential means one of them stole it,
        and there is no way to tell which. Revoking the family logs both out;
        the legitimate user signs in again and the thief cannot.
        """
        first = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        second = (
            await api_client.post(
                "/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}
            )
        ).json()

        # The thief replays the old one...
        await api_client.post(
            "/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}
        )

        # ...and the legitimate holder's current token is dead too.
        response = await api_client.post(
            "/api/v1/auth/refresh", json={"refresh_token": second["refresh_token"]}
        )
        assert response.status_code == 401

    async def test_a_replay_also_moves_the_access_token_watermark(
        self, api_client: AsyncClient, uow: SqlAlchemyUnitOfWork, tenant: Tenant, member: User
    ) -> None:
        """Revoking only the refresh lineage would leave a stolen access token alive.

        Asserted on the watermark rather than on a live token, because `iat` has
        one-second resolution: a token minted in the same second as the
        revocation is indistinguishable from one minted just before it, and a
        test that hid that behind a sleep would be asserting the sleep. The
        one-second boundary itself is pinned exactly in
        `tests/unit/test_models.py`.
        """
        first = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        await api_client.post(
            "/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}
        )
        await uow.rollback()
        assert (await uow.users.get(member.id)).tokens_valid_after is None  # type: ignore[union-attr]

        await api_client.post(
            "/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}
        )

        await uow.rollback()
        assert (await uow.users.get(member.id)).tokens_valid_after is not None  # type: ignore[union-attr]

    async def test_an_unknown_refresh_token_is_401(self, api_client: AsyncClient) -> None:
        from uuid import uuid4

        from rag.domain.credentials import CredentialKind, OpaqueCredential

        stranger = OpaqueCredential.mint(CredentialKind.REFRESH_TOKEN, uuid4())

        response = await api_client.post(
            "/api/v1/auth/refresh", json={"refresh_token": stranger.token}
        )

        assert response.status_code == 401


class TestLogout:
    async def test_logout_revokes_the_refresh_family(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        body = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        assert (
            await api_client.post(
                "/api/v1/auth/logout", json={"refresh_token": body["refresh_token"]}
            )
        ).status_code == 204

        response = await api_client.post(
            "/api/v1/auth/refresh", json={"refresh_token": body["refresh_token"]}
        )
        assert response.status_code == 401

    async def test_logout_is_idempotent(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        # Clients retry this on flaky networks, and there is nothing to protect:
        # an unrecognised token has nothing to reveal and nothing to revoke.
        body = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        for _ in range(3):
            response = await api_client.post(
                "/api/v1/auth/logout", json={"refresh_token": body["refresh_token"]}
            )
            assert response.status_code == 204

    async def test_logout_does_not_affect_another_session(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        # A family is one device's lineage. Signing out of a laptop must not
        # sign out a phone.
        laptop = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        phone = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        await api_client.post(
            "/api/v1/auth/logout", json={"refresh_token": laptop["refresh_token"]}
        )

        response = await api_client.post(
            "/api/v1/auth/refresh", json={"refresh_token": phone["refresh_token"]}
        )
        assert response.status_code == 200


class TestPasswordChange:
    async def test_a_password_change_returns_a_working_pair(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        body = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        response = await api_client.post(
            "/api/v1/auth/password",
            headers=bearer(body["access_token"]),
            json={"current_password": TEST_PASSWORD, "new_password": "a-brand-new-passphrase"},
        )

        assert response.status_code == 200
        assert (
            await api_client.get("/api/v1/auth/me", headers=bearer(response.json()["access_token"]))
        ).status_code == 200

    async def test_the_new_password_is_what_works_afterwards(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        body = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        await api_client.post(
            "/api/v1/auth/password",
            headers=bearer(body["access_token"]),
            json={"current_password": TEST_PASSWORD, "new_password": "a-brand-new-passphrase"},
        )

        old = await api_client.post(
            "/api/v1/auth/login",
            json={
                "tenant_slug": tenant.slug,
                "email": member.email,
                "password": TEST_PASSWORD,
            },
        )
        new = await api_client.post(
            "/api/v1/auth/login",
            json={
                "tenant_slug": tenant.slug,
                "email": member.email,
                "password": "a-brand-new-passphrase",
            },
        )

        assert old.status_code == 401
        assert new.status_code == 200

    async def test_a_password_change_revokes_every_other_session(
        self, api_client: AsyncClient, uow: SqlAlchemyUnitOfWork, tenant: Tenant, member: User
    ) -> None:
        """The reason to change a password is that it might be compromised.

        Leaving other sessions alive would defeat the act entirely.
        """
        laptop = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        phone = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        await api_client.post(
            "/api/v1/auth/password",
            headers=bearer(laptop["access_token"]),
            json={"current_password": TEST_PASSWORD, "new_password": "a-brand-new-passphrase"},
        )

        # The other device cannot mint a new access token — exact, because
        # refresh tokens are database rows.
        assert (
            await api_client.post(
                "/api/v1/auth/refresh", json={"refresh_token": phone["refresh_token"]}
            )
        ).status_code == 401
        # And its existing access token is past the watermark, so it dies as
        # soon as the second in which both were issued has elapsed. See
        # `User.accepts_token_issued_at` for why that boundary is one second.
        await uow.rollback()
        assert (await uow.users.get(member.id)).tokens_valid_after is not None  # type: ignore[union-attr]

    async def test_the_current_password_is_required(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        # What stops a stolen access token from locking the real owner out.
        body = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        response = await api_client.post(
            "/api/v1/auth/password",
            headers=bearer(body["access_token"]),
            json={"current_password": "not-it", "new_password": "a-brand-new-passphrase"},
        )

        assert response.status_code == 401

    async def test_a_short_password_is_rejected(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        body = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        response = await api_client.post(
            "/api/v1/auth/password",
            headers=bearer(body["access_token"]),
            json={"current_password": TEST_PASSWORD, "new_password": "short"},
        )

        assert response.status_code == 400
        assert response.json()["code"] == "invalid_input"


class TestJwks:
    async def test_the_key_set_is_public_and_carries_no_private_material(
        self, api_client: AsyncClient
    ) -> None:
        response = await api_client.get("/.well-known/jwks.json")

        assert response.status_code == 200
        keys = response.json()["keys"]
        assert keys
        assert all("d" not in key for key in keys)
        assert "PRIVATE" not in response.text
