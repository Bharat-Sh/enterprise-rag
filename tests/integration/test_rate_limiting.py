"""Per-tenant and per-login throttling, over HTTP.

The two limiters exist for different reasons and are keyed differently. The
tenant bucket protects per-tenant capacity and cost; the login bucket protects
credentials and cannot be keyed on a tenant that has not been verified yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rag.core.config import Environment, LogFormat, Settings
from tests.integration.conftest import TEST_PASSWORD, bearer, login, requires_postgres
from tests.support import generate_private_pem

if TYPE_CHECKING:
    from httpx import AsyncClient

    from rag.domain.models import Tenant, User

pytestmark = [pytest.mark.integration, requires_postgres]


def _settings_with(**rate_limit: object) -> Settings:
    return Settings(
        _env_file=None,
        environment=Environment.LOCAL,
        log_level="CRITICAL",
        log_format=LogFormat.JSON,
        database={
            "host": "127.0.0.1",
            "port": 5432,
            "user": "rag",
            "password": "rag",
            "name": "rag_test",
        },
        auth={
            "private_key_pem": generate_private_pem(),
            "argon2_time_cost": 1,
            "argon2_memory_cost_kib": 8192,
        },
        rate_limit=rate_limit,
    )


class TestLoginLimiter:
    @pytest.fixture
    def api_settings(self) -> Settings:
        # Tight enough to trip in a handful of requests. The behaviour under
        # test is the bucket, not the number.
        return _settings_with(login_attempts_per_minute=6, login_burst=3)

    async def test_repeated_failed_logins_are_throttled(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        """The credential-stuffing surface.

        A per-tenant limiter cannot help here: there is no verified tenant until
        the login succeeds, which is precisely what the attacker is trying to
        make happen.
        """
        body = {"tenant_slug": tenant.slug, "email": member.email, "password": "wrong"}

        statuses = [
            (await api_client.post("/api/v1/auth/login", json=body)).status_code for _ in range(5)
        ]

        assert statuses[:3] == [401, 401, 401]
        assert 429 in statuses

    async def test_a_throttled_response_tells_the_client_when_to_come_back(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        body = {"tenant_slug": tenant.slug, "email": member.email, "password": "wrong"}
        for _ in range(4):
            await api_client.post("/api/v1/auth/login", json=body)

        response = await api_client.post("/api/v1/auth/login", json=body)

        assert response.status_code == 429
        assert response.json()["code"] == "rate_limit_exceeded"
        # Distinct from `quota_exceeded`: "slow down" and "you are out of
        # allowance" need different client behaviour.
        assert int(response.headers["retry-after"]) >= 1
        assert response.headers["x-ratelimit-limit"] == "6"
        assert response.headers["x-ratelimit-remaining"] == "0"

    async def test_throttling_does_not_leak_whether_the_password_was_right(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        # Once the bucket is empty, a correct password must be refused too —
        # otherwise the 429/401 split becomes the oracle the uniform 401 was
        # designed to remove.
        wrong = {"tenant_slug": tenant.slug, "email": member.email, "password": "wrong"}
        for _ in range(4):
            await api_client.post("/api/v1/auth/login", json=wrong)

        response = await api_client.post(
            "/api/v1/auth/login",
            json={
                "tenant_slug": tenant.slug,
                "email": member.email,
                "password": TEST_PASSWORD,
            },
        )

        assert response.status_code == 429


class TestTenantLimiter:
    @pytest.fixture
    def api_settings(self) -> Settings:
        return _settings_with(tenant_requests_per_minute=60, tenant_burst=3)

    async def test_an_authenticated_tenant_is_throttled(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        headers = bearer(session["access_token"])

        statuses = [
            (await api_client.get("/api/v1/auth/me", headers=headers)).status_code for _ in range(5)
        ]

        assert statuses[:3] == [200, 200, 200]
        assert statuses[-1] == 429

    async def test_the_limit_is_per_tenant_not_per_user(
        self, api_client: AsyncClient, tenant: Tenant, member: User, owner: User
    ) -> None:
        """Keyed on the tenant, because that is what the cost accrues to.

        Also the honest consequence: one noisy user can throttle a colleague.
        Per-tenant fairness is the goal; per-user fairness is not something this
        limiter promises.
        """
        theirs = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        ours = await login(api_client, tenant_slug=tenant.slug, email=owner.email)

        for _ in range(4):
            await api_client.get("/api/v1/auth/me", headers=bearer(theirs["access_token"]))

        response = await api_client.get("/api/v1/auth/me", headers=bearer(ours["access_token"]))
        assert response.status_code == 429


class TestDisabled:
    @pytest.fixture
    def api_settings(self) -> Settings:
        return _settings_with(enabled=False, tenant_requests_per_minute=60, tenant_burst=1)

    async def test_limits_can_be_turned_off_entirely(
        self, api_client: AsyncClient, tenant: Tenant, member: User
    ) -> None:
        session = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        headers = bearer(session["access_token"])

        statuses = [
            (await api_client.get("/api/v1/auth/me", headers=headers)).status_code
            for _ in range(10)
        ]

        assert set(statuses) == {200}
