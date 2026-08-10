"""Forged credentials, signed with our own key.

A forgery that fails the signature check proves nothing about the design — it
proves PyJWT works. These tests mint **validly signed** tokens carrying wrong
claims, and validly shaped API keys carrying a wrong tenant, using the running
application's own keyring. What must stop them is the mechanism in ADR-0007: the
tenant is bound from the credential before anything is read, so a forged tenant
finds no rows rather than finding a row that then has to be checked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from rag.domain.credentials import CredentialKind, OpaqueCredential
from tests.integration.conftest import bearer, login, requires_postgres

if TYPE_CHECKING:
    from fastapi import FastAPI
    from httpx import AsyncClient

    from rag.db.uow import SqlAlchemyUnitOfWork
    from rag.domain.models import Tenant, User

pytestmark = [pytest.mark.security, pytest.mark.integration, requires_postgres]


class TestForgedTokens:
    async def test_a_validly_signed_token_for_another_tenant_is_rejected(
        self,
        api_app: FastAPI,
        api_client: AsyncClient,
        tenant: Tenant,
        other_tenant: Tenant,
        member: User,
    ) -> None:
        """The core claim of ADR-0007, attacked directly.

        The token is signed by the real key and passes every signature,
        audience, and issuer check. Its `sub` is a real user. Only `tid` is
        wrong — and that is enough, because the scope is bound from `tid`, so
        the user lookup runs inside the wrong tenant and returns nothing.

        No comparison rejects this. The absence of a row does.
        """
        forged = api_app.state.token_service.issue_access_token(
            subject=member.id, tenant_id=other_tenant.id
        )

        response = await api_client.get("/api/v1/auth/me", headers=bearer(forged.token))

        assert response.status_code == 401

    async def test_a_validly_signed_token_for_a_nonexistent_tenant_is_rejected(
        self, api_app: FastAPI, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        forged = api_app.state.token_service.issue_access_token(
            subject=member.id, tenant_id=uuid4()
        )

        response = await api_client.get("/api/v1/auth/me", headers=bearer(forged.token))

        assert response.status_code == 401

    async def test_a_validly_signed_token_for_a_nonexistent_user_is_rejected(
        self, api_app: FastAPI, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        forged = api_app.state.token_service.issue_access_token(
            subject=uuid4(), tenant_id=tenant.id
        )

        response = await api_client.get("/api/v1/auth/me", headers=bearer(forged.token))

        assert response.status_code == 401

    async def test_a_token_naming_another_tenants_user_is_rejected(
        self,
        api_app: FastAPI,
        api_client: AsyncClient,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        other_tenant: Tenant,
    ) -> None:
        """Both halves wrong in a *consistent* way is the interesting case.

        Here `tid` and `sub` agree with each other — they just belong to a
        different customer. This is what a stolen-token replay looks like, and
        it must fail on the credential, not on a downstream check.
        """
        await uow.scope_to_tenant(other_tenant.id)
        stranger = await uow.users.create(tenant_id=other_tenant.id, email="rival@globex.example")
        await uow.commit()
        await uow.scope_to_tenant(tenant.id)

        forged = api_app.state.token_service.issue_access_token(
            subject=stranger.id, tenant_id=tenant.id
        )

        response = await api_client.get("/api/v1/auth/me", headers=bearer(forged.token))

        assert response.status_code == 401


class TestForgedApiKeys:
    async def test_rewriting_the_tenant_segment_breaks_the_key(
        self,
        api_client: AsyncClient,
        tenant: Tenant,
        other_tenant: Tenant,
        member: User,
    ) -> None:
        """The API-key half of the same mechanism.

        The tenant segment is unauthenticated input — the caller can put
        anything there. That is safe by construction: it only chooses which rows
        are visible, and the secret still has to hash to one of them. Point it
        at another tenant and the key's own row becomes invisible.
        """
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        secret = (
            await api_client.post(
                "/api/v1/api-keys",
                headers=bearer(session["access_token"]),
                json={"name": "ci", "role": "viewer"},
            )
        ).json()["secret"]
        assert (await api_client.get("/api/v1/auth/me", headers=bearer(secret))).status_code == 200

        parsed = OpaqueCredential.parse(secret, expected=CredentialKind.API_KEY)
        rewritten = OpaqueCredential(
            kind=CredentialKind.API_KEY, tenant_id=other_tenant.id, secret=parsed.secret
        ).token

        assert (
            await api_client.get("/api/v1/auth/me", headers=bearer(rewritten))
        ).status_code == 401

    async def test_a_well_formed_key_for_no_tenant_at_all_is_rejected(
        self, api_client: AsyncClient
    ) -> None:
        stranger = OpaqueCredential.mint(CredentialKind.API_KEY, uuid4())

        response = await api_client.get("/api/v1/auth/me", headers=bearer(stranger.token))

        assert response.status_code == 401

    async def test_a_refresh_token_reshaped_as_an_api_key_is_rejected(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        # The secret is genuinely ours and genuinely in the database — just in
        # the wrong table. Distinct prefixes make this a shape error rather than
        # a lookup that happens to miss.
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        refresh = OpaqueCredential.parse(
            session["refresh_token"], expected=CredentialKind.REFRESH_TOKEN
        )
        reshaped = OpaqueCredential(
            kind=CredentialKind.API_KEY, tenant_id=refresh.tenant_id, secret=refresh.secret
        ).token

        response = await api_client.get("/api/v1/auth/me", headers=bearer(reshaped))

        assert response.status_code == 401


class TestForgedRefreshTokens:
    async def test_a_refresh_token_pointed_at_another_tenant_is_rejected(
        self, api_client: AsyncClient, tenant: Tenant, other_tenant: Tenant, member: User
    ) -> None:
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        parsed = OpaqueCredential.parse(
            session["refresh_token"], expected=CredentialKind.REFRESH_TOKEN
        )
        rewritten = OpaqueCredential(
            kind=CredentialKind.REFRESH_TOKEN, tenant_id=other_tenant.id, secret=parsed.secret
        ).token

        response = await api_client.post("/api/v1/auth/refresh", json={"refresh_token": rewritten})

        assert response.status_code == 401

    async def test_a_rewritten_refresh_token_does_not_burn_the_real_one(
        self, api_client: AsyncClient, tenant: Tenant, other_tenant: Tenant, member: User
    ) -> None:
        """A failed forgery must not be a denial-of-service on the victim.

        If the rewritten token were found and then rejected by a comparison, a
        careless implementation might still mark it used — letting anyone log a
        user out by guessing at their tenant.
        """
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        parsed = OpaqueCredential.parse(
            session["refresh_token"], expected=CredentialKind.REFRESH_TOKEN
        )
        rewritten = OpaqueCredential(
            kind=CredentialKind.REFRESH_TOKEN, tenant_id=other_tenant.id, secret=parsed.secret
        ).token

        await api_client.post("/api/v1/auth/refresh", json={"refresh_token": rewritten})

        genuine = await api_client.post(
            "/api/v1/auth/refresh", json={"refresh_token": session["refresh_token"]}
        )
        assert genuine.status_code == 200
