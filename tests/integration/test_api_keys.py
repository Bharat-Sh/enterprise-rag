"""API keys: issue, use, narrow, revoke.

The load-bearing assertions here are the ones about the **ceiling**. A key that
narrows what it may call but not what it may read has narrowed the less
important half.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from rag.domain.enums import Role, UserStatus
from tests.integration.conftest import bearer, login, requires_postgres

if TYPE_CHECKING:
    from httpx import AsyncClient

    from rag.db.uow import SqlAlchemyUnitOfWork
    from rag.domain.models import Tenant, User

pytestmark = [pytest.mark.integration, requires_postgres]


async def _issue_key(
    client: AsyncClient, access_token: str, *, name: str = "ci", role: str = "viewer"
) -> tuple[str, dict[str, object]]:
    response = await client.post(
        "/api/v1/api-keys",
        headers=bearer(access_token),
        json={"name": name, "role": role},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    return body["secret"], body["key"]


class TestIssuing:
    async def test_a_key_is_issued_and_works(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        secret, _ = await _issue_key(api_client, session["access_token"])

        assert secret.startswith("ragk_")
        response = await api_client.get("/api/v1/auth/me", headers=bearer(secret))
        assert response.status_code == 200
        assert response.json()["credential"] == "api_key"
        assert response.json()["user"]["email"] == member.email

    async def test_the_secret_is_never_returned_again(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        # Only a SHA-256 and a display prefix are stored, so there is no code
        # path — including a database dump — that can produce it a second time.
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        secret, _ = await _issue_key(api_client, session["access_token"])

        listing = await api_client.get("/api/v1/api-keys", headers=bearer(session["access_token"]))

        assert listing.status_code == 200
        assert secret not in listing.text
        assert listing.json()[0]["display_prefix"] in secret

    async def test_listing_a_key_never_exposes_the_hash(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        await _issue_key(api_client, session["access_token"])

        listing = await api_client.get("/api/v1/api-keys", headers=bearer(session["access_token"]))

        assert "secret_hash" not in listing.text

    async def test_an_expiry_can_be_requested(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        response = await api_client.post(
            "/api/v1/api-keys",
            headers=bearer(session["access_token"]),
            json={"name": "short-lived", "role": "viewer", "expires_in_days": 7},
        )

        assert response.status_code == 201
        assert response.json()["key"]["expires_at"] is not None


class TestRoleCeiling:
    async def test_a_key_cannot_be_issued_above_the_callers_role(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        # Rejected rather than silently narrowed: asking for admin as a member
        # is a mistake, and quietly issuing something weaker is how a deploy
        # script fails mysteriously a fortnight later.
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        response = await api_client.post(
            "/api/v1/api-keys",
            headers=bearer(session["access_token"]),
            json={"name": "too-powerful", "role": "owner"},
        )

        assert response.status_code == 403
        assert response.json()["code"] == "permission_denied"

    async def test_the_ceiling_narrows_the_effective_role(
        self, api_client: AsyncClient, tenant: Tenant, owner: User
    ) -> None:
        session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        secret, _ = await _issue_key(api_client, session["access_token"], role="viewer")

        me = await api_client.get("/api/v1/auth/me", headers=bearer(secret))

        assert me.json()["user"]["role"] == "owner"
        assert me.json()["effective_role"] == "viewer"

    async def test_a_narrowed_key_cannot_reach_an_admin_route(
        self, api_client: AsyncClient, tenant: Tenant, owner: User
    ) -> None:
        """The ceiling is enforced, not merely reported."""
        session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        secret, _ = await _issue_key(api_client, session["access_token"], role="viewer")

        # The owner can list users...
        assert (
            await api_client.get("/api/v1/users", headers=bearer(session["access_token"]))
        ).status_code == 200
        # ...and their own viewer-scoped key cannot.
        denied = await api_client.get("/api/v1/users", headers=bearer(secret))
        assert denied.status_code == 403
        assert denied.json()["errors"]["credential"] == "api_key"

    async def test_the_ceiling_also_narrows_the_access_filter(
        self, api_client: AsyncClient, tenant: Tenant, owner: User
    ) -> None:
        """The half that is easy to forget.

        `AccessFilter` carries a `role:` principal, and document ACLs can be
        granted to it. If the ceiling were applied only to route permissions,
        a key scoped down to viewer would still match `role:owner` grants.
        """
        session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        secret, _ = await _issue_key(api_client, session["access_token"], role="viewer")

        me = await api_client.get("/api/v1/auth/me", headers=bearer(secret))

        assert me.json()["effective_role"] == "viewer"

    async def test_a_key_may_not_issue_another_key(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        # A leaked key must not be a foothold that renews itself; otherwise
        # revoking the one that leaked achieves nothing.
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        secret, _ = await _issue_key(api_client, session["access_token"])

        response = await api_client.post(
            "/api/v1/api-keys", headers=bearer(secret), json={"name": "child", "role": "viewer"}
        )

        assert response.status_code == 403

    async def test_a_key_may_not_change_a_password(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        secret, _ = await _issue_key(api_client, session["access_token"])

        response = await api_client.post(
            "/api/v1/auth/password",
            headers=bearer(secret),
            json={"current_password": "anything", "new_password": "a-brand-new-passphrase"},
        )

        assert response.status_code == 403


class TestRevocation:
    async def test_a_revoked_key_stops_working(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        secret, key = await _issue_key(api_client, session["access_token"])
        assert (await api_client.get("/api/v1/auth/me", headers=bearer(secret))).status_code == 200

        deleted = await api_client.delete(
            f"/api/v1/api-keys/{key['id']}", headers=bearer(session["access_token"])
        )

        assert deleted.status_code == 204
        assert (await api_client.get("/api/v1/auth/me", headers=bearer(secret))).status_code == 401

    async def test_revoking_an_unknown_key_is_404(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        response = await api_client.delete(
            f"/api/v1/api-keys/{uuid4()}", headers=bearer(session["access_token"])
        )

        assert response.status_code == 404

    async def test_a_key_stops_working_when_its_owner_is_disabled(
        self,
        api_client: AsyncClient,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        member: User,
    ) -> None:
        # The deliberate consequence of binding a key to a user: it cannot
        # outlive their access. A key still working for a departed employee is
        # the artifact that turns up in breach post-mortems.
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        secret, _ = await _issue_key(api_client, session["access_token"])

        await uow.users.set_status(member.id, UserStatus.DISABLED)
        await uow.commit()

        assert (await api_client.get("/api/v1/auth/me", headers=bearer(secret))).status_code == 401


class TestLastUsed:
    async def test_first_use_records_a_timestamp(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        secret, key = await _issue_key(api_client, session["access_token"])
        assert key["last_used_at"] is None

        await api_client.get("/api/v1/auth/me", headers=bearer(secret))

        listing = await api_client.get("/api/v1/api-keys", headers=bearer(session["access_token"]))
        assert listing.json()[0]["last_used_at"] is not None

    async def test_subsequent_use_does_not_rewrite_it(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        """Throttled: one write per key per five minutes, not one per request.

        Writing on every request would make each authenticated GET an update of
        the hottest row in the tenant.
        """
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        secret, _ = await _issue_key(api_client, session["access_token"])

        await api_client.get("/api/v1/auth/me", headers=bearer(secret))
        first = (
            await api_client.get("/api/v1/api-keys", headers=bearer(session["access_token"]))
        ).json()[0]["last_used_at"]

        for _ in range(3):
            await api_client.get("/api/v1/auth/me", headers=bearer(secret))
        second = (
            await api_client.get("/api/v1/api-keys", headers=bearer(session["access_token"]))
        ).json()[0]["last_used_at"]

        assert first == second


class TestUserAdministration:
    """RBAC at the HTTP boundary, across three privilege levels."""

    async def test_a_member_cannot_list_users(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        response = await api_client.get("/api/v1/users", headers=bearer(session["access_token"]))

        assert response.status_code == 403
        assert response.json()["errors"]["your_role"] == "member"

    async def test_an_owner_can_list_and_create_users(
        self, api_client: AsyncClient, tenant: Tenant, owner: User
    ) -> None:
        session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)

        created = await api_client.post(
            "/api/v1/users",
            headers=bearer(session["access_token"]),
            json={"email": "newcomer@acme.example", "full_name": "New Comer", "role": "viewer"},
        )

        assert created.status_code == 201
        assert created.json()["status"] == "invited"
        listing = await api_client.get("/api/v1/users", headers=bearer(session["access_token"]))
        assert {user["email"] for user in listing.json()} == {
            owner.email,
            "newcomer@acme.example",
        }

    async def test_a_created_user_carries_no_secret_fields(
        self, api_client: AsyncClient, tenant: Tenant, owner: User
    ) -> None:
        session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)

        created = await api_client.post(
            "/api/v1/users",
            headers=bearer(session["access_token"]),
            json={"email": "newcomer@acme.example"},
        )

        assert "password_hash" not in created.text
        assert "tokens_valid_after" not in created.text

    async def test_role_assignment_is_owner_only(
        self, api_client: AsyncClient, tenant: Tenant, owner: User, member: User
    ) -> None:
        owner_session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        member_session = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        denied = await api_client.patch(
            f"/api/v1/users/{owner.id}/role",
            headers=bearer(member_session["access_token"]),
            json={"role": "owner"},
        )
        allowed = await api_client.patch(
            f"/api/v1/users/{member.id}/role",
            headers=bearer(owner_session["access_token"]),
            json={"role": "admin"},
        )

        assert denied.status_code == 403
        assert allowed.status_code == 200
        assert allowed.json()["role"] == "admin"

    async def test_an_owner_cannot_demote_themselves(
        self, api_client: AsyncClient, tenant: Tenant, owner: User
    ) -> None:
        # Not paternalism: a tenant with no owner has no endpoint that could
        # restore one.
        session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)

        response = await api_client.patch(
            f"/api/v1/users/{owner.id}/role",
            headers=bearer(session["access_token"]),
            json={"role": "member"},
        )

        assert response.status_code == 400

    async def test_a_duplicate_email_is_rejected(
        self, api_client: AsyncClient, tenant: Tenant, owner: User, member: User
    ) -> None:
        session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)

        response = await api_client.post(
            "/api/v1/users",
            headers=bearer(session["access_token"]),
            json={"email": member.email},
        )

        assert response.status_code == 409
        assert response.json()["code"] == "already_exists"

    async def test_concurrent_invites_of_the_same_address_do_not_500(
        self, api_client: AsyncClient, tenant: Tenant, owner: User
    ) -> None:
        """The race a read-then-insert cannot win.

        Both requests can find nothing and both attempt the insert, so the
        unique constraint is the only thing that can adjudicate. It must surface
        as a 409, not as an unhandled `IntegrityError`.
        """
        import asyncio

        session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        body = {"email": "contested@acme.example"}

        responses = await asyncio.gather(
            *(
                api_client.post("/api/v1/users", headers=bearer(session["access_token"]), json=body)
                for _ in range(4)
            )
        )

        statuses = sorted(response.status_code for response in responses)
        assert statuses.count(201) == 1
        assert set(statuses[1:]) == {409}

    async def test_a_role_change_takes_effect_on_the_next_request(
        self, api_client: AsyncClient, tenant: Tenant, owner: User, member: User
    ) -> None:
        """The payoff for keeping `role` out of the token.

        With a `role` claim this would take up to fifteen minutes, and the
        window would be invisible.
        """
        owner_session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        member_session = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        assert (
            await api_client.get("/api/v1/users", headers=bearer(member_session["access_token"]))
        ).status_code == 403

        await api_client.patch(
            f"/api/v1/users/{member.id}/role",
            headers=bearer(owner_session["access_token"]),
            json={"role": Role.ADMIN.value},
        )

        # Same token, new authority — no re-login required.
        assert (
            await api_client.get("/api/v1/users", headers=bearer(member_session["access_token"]))
        ).status_code == 200
