"""Cross-tenant isolation, exercised through HTTP rather than the repositories.

M1 proved that row-level security isolates tenants when a scope is bound. The
question M2 has to answer is whether the scope is *actually bound* on the path a
real request takes, which can only be asked from outside the application.

Every "not yours" case answers **404, never 403**. A 403 confirms the resource
exists, which is an enumeration oracle: an attacker walks ids and learns another
customer's user count from the status codes alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rag.domain.enums import Role, UserStatus
from tests.integration.conftest import TEST_PASSWORD, bearer, login, requires_postgres

if TYPE_CHECKING:
    from httpx import AsyncClient

    from rag.adapters.auth.passwords import Argon2PasswordHasher
    from rag.db.uow import SqlAlchemyUnitOfWork
    from rag.domain.models import Tenant, User

pytestmark = [pytest.mark.security, pytest.mark.integration, requires_postgres]


@pytest.fixture
async def rival_owner(
    uow: SqlAlchemyUnitOfWork,
    tenant: Tenant,
    other_tenant: Tenant,
    hasher: Argon2PasswordHasher,
) -> User:
    """An owner in the *other* tenant, with a working password."""
    await uow.scope_to_tenant(other_tenant.id)
    created = await uow.users.create(
        tenant_id=other_tenant.id,
        email="boss@globex.example",
        full_name="Rival Boss",
        role=Role.OWNER,
        status=UserStatus.ACTIVE,
        password_hash=await hasher.hash(TEST_PASSWORD),
    )
    await uow.commit()
    await uow.scope_to_tenant(tenant.id)
    return created


class TestListingIsScoped:
    async def test_a_tenant_never_sees_another_tenants_users(
        self,
        api_client: AsyncClient,
        tenant: Tenant,
        other_tenant: Tenant,
        owner: User,
        rival_owner: User,
    ) -> None:
        session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)

        listing = await api_client.get("/api/v1/users", headers=bearer(session["access_token"]))

        emails = {user["email"] for user in listing.json()}
        assert owner.email in emails
        assert rival_owner.email not in emails

    async def test_both_tenants_see_only_their_own(
        self,
        api_client: AsyncClient,
        tenant: Tenant,
        other_tenant: Tenant,
        owner: User,
        rival_owner: User,
    ) -> None:
        # Symmetric on purpose: an isolation bug that only leaks one direction
        # is still a breach, and a one-sided test would miss half of them.
        ours = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        theirs = await login(api_client, tenant_slug=other_tenant.slug, email=rival_owner.email)

        our_view = await api_client.get("/api/v1/users", headers=bearer(ours["access_token"]))
        their_view = await api_client.get("/api/v1/users", headers=bearer(theirs["access_token"]))

        assert {user["email"] for user in our_view.json()} == {owner.email}
        assert {user["email"] for user in their_view.json()} == {rival_owner.email}


class TestCrossTenantLookupsAreNotFound:
    async def test_modifying_another_tenants_user_is_404_not_403(
        self,
        api_client: AsyncClient,
        tenant: Tenant,
        other_tenant: Tenant,
        owner: User,
        rival_owner: User,
    ) -> None:
        """The enumeration oracle CLAUDE.md names, tested at the boundary.

        The caller is a genuine owner with every permission this endpoint asks
        for, so authorization passes. It is the *lookup* that must come back
        empty — and 404 is what says "there is nothing here" without confirming
        that there is.
        """
        session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)

        response = await api_client.patch(
            f"/api/v1/users/{rival_owner.id}/role",
            headers=bearer(session["access_token"]),
            json={"role": "viewer"},
        )

        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    async def test_a_cross_tenant_id_is_indistinguishable_from_a_random_one(
        self,
        api_client: AsyncClient,
        tenant: Tenant,
        other_tenant: Tenant,
        owner: User,
        rival_owner: User,
    ) -> None:
        from uuid import uuid4

        session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)

        real_but_foreign = await api_client.patch(
            f"/api/v1/users/{rival_owner.id}/role",
            headers=bearer(session["access_token"]),
            json={"role": "viewer"},
        )
        entirely_invented = await api_client.patch(
            f"/api/v1/users/{uuid4()}/role",
            headers=bearer(session["access_token"]),
            json={"role": "viewer"},
        )

        assert real_but_foreign.status_code == entirely_invented.status_code
        assert real_but_foreign.json()["code"] == entirely_invented.json()["code"]

    async def test_the_write_does_not_happen(
        self,
        api_client: AsyncClient,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        other_tenant: Tenant,
        owner: User,
        rival_owner: User,
    ) -> None:
        # A 404 that still wrote the row would be the worst of both worlds.
        session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)

        await api_client.patch(
            f"/api/v1/users/{rival_owner.id}/role",
            headers=bearer(session["access_token"]),
            json={"role": "viewer"},
        )

        await uow.scope_to_tenant(other_tenant.id)
        unchanged = await uow.users.get(rival_owner.id)
        assert unchanged is not None
        assert unchanged.role is Role.OWNER

    async def test_revoking_another_tenants_api_key_is_404(
        self,
        api_client: AsyncClient,
        tenant: Tenant,
        other_tenant: Tenant,
        owner: User,
        rival_owner: User,
    ) -> None:
        theirs = await login(api_client, tenant_slug=other_tenant.slug, email=rival_owner.email)
        their_key = (
            await api_client.post(
                "/api/v1/api-keys",
                headers=bearer(theirs["access_token"]),
                json={"name": "their-ci", "role": "viewer"},
            )
        ).json()["key"]

        ours = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        response = await api_client.delete(
            f"/api/v1/api-keys/{their_key['id']}", headers=bearer(ours["access_token"])
        )

        assert response.status_code == 404


class TestCreationIsScoped:
    async def test_a_created_user_lands_in_the_callers_tenant(
        self,
        api_client: AsyncClient,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        other_tenant: Tenant,
        owner: User,
    ) -> None:
        """No endpoint accepts a `tenant_id`, and RLS `WITH CHECK` backs that up.

        Even if a handler were changed to stamp a foreign tenant, the insert
        would be refused by the policy rather than quietly corrupting the other
        customer's data.
        """
        session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)

        created = await api_client.post(
            "/api/v1/users",
            headers=bearer(session["access_token"]),
            json={"email": "newcomer@acme.example"},
        )

        assert created.status_code == 201
        await uow.scope_to_tenant(other_tenant.id)
        assert await uow.users.get_by_email("newcomer@acme.example") is None
        await uow.scope_to_tenant(tenant.id)
        assert await uow.users.get_by_email("newcomer@acme.example") is not None

    async def test_the_same_email_may_exist_in_both_tenants(
        self,
        api_client: AsyncClient,
        tenant: Tenant,
        other_tenant: Tenant,
        owner: User,
        rival_owner: User,
    ) -> None:
        """Email uniqueness is per tenant, and that is a security property.

        A global constraint would let one customer discover, by trying to invite
        an address, whether it already exists in another.
        """
        ours = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        theirs = await login(api_client, tenant_slug=other_tenant.slug, email=rival_owner.email)

        first = await api_client.post(
            "/api/v1/users",
            headers=bearer(ours["access_token"]),
            json={"email": "shared@consultant.example"},
        )
        second = await api_client.post(
            "/api/v1/users",
            headers=bearer(theirs["access_token"]),
            json={"email": "shared@consultant.example"},
        )

        assert first.status_code == 201
        assert second.status_code == 201


class TestCredentialsDoNotCrossTenants:
    async def test_a_password_only_works_in_its_own_tenant(
        self,
        api_client: AsyncClient,
        tenant: Tenant,
        other_tenant: Tenant,
        member: User,
        rival_owner: User,
    ) -> None:
        # Naming another tenant at login is allowed — the slug is an addressing
        # input. It simply does not help.
        response = await api_client.post(
            "/api/v1/auth/login",
            json={
                "tenant_slug": other_tenant.slug,
                "email": member.email,
                "password": TEST_PASSWORD,
            },
        )

        assert response.status_code == 401

    async def test_an_api_key_cannot_read_across_tenants(
        self,
        api_client: AsyncClient,
        tenant: Tenant,
        other_tenant: Tenant,
        owner: User,
        rival_owner: User,
    ) -> None:
        theirs = await login(api_client, tenant_slug=other_tenant.slug, email=rival_owner.email)
        their_secret = (
            await api_client.post(
                "/api/v1/api-keys",
                headers=bearer(theirs["access_token"]),
                json={"name": "their-ci", "role": "owner"},
            )
        ).json()["secret"]

        listing = await api_client.get("/api/v1/users", headers=bearer(their_secret))

        assert listing.status_code == 200
        assert {user["email"] for user in listing.json()} == {rival_owner.email}
