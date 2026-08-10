"""`rag-admin`, the bootstrap path.

A fresh deployment has no tenant and no user and every endpoint needs a
credential, so this is the only way in. If it is broken, nothing else in the
system can be reached — which makes it worth testing despite being three
commands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from cryptography.hazmat.primitives import serialization

from rag.cli import main, run
from rag.core.config import Environment, LogFormat, Settings, SigningAlgorithm
from tests.integration.conftest import bearer, requires_postgres
from tests.support import generate_private_pem

if TYPE_CHECKING:
    from httpx import AsyncClient

    from rag.db.uow import SqlAlchemyUnitOfWork

pytestmark = [pytest.mark.integration, requires_postgres]

BOOTSTRAP_PASSWORD = "a-sufficiently-long-bootstrap-password"


@pytest.fixture
def api_settings() -> Settings:
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
    )


@pytest.fixture
def cli_settings(api_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Point the CLI's `get_settings()` at the same test database."""
    monkeypatch.setattr("rag.cli.get_settings", lambda: api_settings)
    return api_settings


class TestKeyGeneration:
    def test_generate_key_emits_a_usable_ed25519_pem(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Works before anything is configured, which is when it is needed.

        The only subcommand that touches neither the database nor `Settings`.
        """
        assert main(["generate-key"]) == 0

        pem = capsys.readouterr().out
        key = serialization.load_pem_private_key(pem.encode(), password=None)
        assert "BEGIN PRIVATE KEY" in pem
        assert key.public_key()

    def test_generate_key_can_emit_rsa(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["generate-key", "--algorithm", SigningAlgorithm.RS256.value]) == 0

        pem = capsys.readouterr().out
        assert serialization.load_pem_private_key(pem.encode(), password=None).key_size == 2048


class TestBootstrap:
    async def test_a_tenant_and_owner_created_by_cli_can_log_in(
        self,
        cli_settings: Settings,
        api_client: AsyncClient,
        db_engine: object,
    ) -> None:
        """The whole point: from an empty database to a working session."""
        assert await run(["create-tenant", "--slug", "newco", "--name", "New Co"]) == 0
        assert (
            await run(
                [
                    "create-user",
                    "--tenant",
                    "newco",
                    "--email",
                    "founder@newco.example",
                    "--role",
                    "owner",
                    "--password",
                    BOOTSTRAP_PASSWORD,
                ]
            )
            == 0
        )

        response = await api_client.post(
            "/api/v1/auth/login",
            json={
                "tenant_slug": "newco",
                "email": "founder@newco.example",
                "password": BOOTSTRAP_PASSWORD,
            },
        )

        assert response.status_code == 200
        me = await api_client.get(
            "/api/v1/auth/me", headers=bearer(response.json()["access_token"])
        )
        assert me.json()["effective_role"] == "owner"

    async def test_a_bootstrap_user_is_active_not_invited(
        self, cli_settings: Settings, uow: SqlAlchemyUnitOfWork, db_engine: object
    ) -> None:
        # An invited user has no password and cannot log in, which would make
        # the bootstrap command bootstrap nothing.
        await run(["create-tenant", "--slug", "newco", "--name", "New Co"])
        await run(
            [
                "create-user",
                "--tenant",
                "newco",
                "--email",
                "founder@newco.example",
                "--password",
                BOOTSTRAP_PASSWORD,
            ]
        )

        tenant = await uow.tenants.get_by_slug("newco")
        assert tenant is not None
        await uow.scope_to_tenant(tenant.id)
        created = await uow.users.get_by_email("founder@newco.example")
        assert created is not None
        assert created.can_authenticate is True

    async def test_creating_a_duplicate_tenant_fails_loudly(
        self, cli_settings: Settings, db_engine: object
    ) -> None:
        await run(["create-tenant", "--slug", "newco", "--name", "New Co"])

        with pytest.raises(SystemExit) as raised:
            await run(["create-tenant", "--slug", "newco", "--name", "New Co Again"])

        assert raised.value.code == 1

    async def test_creating_a_user_in_a_missing_tenant_fails_loudly(
        self, cli_settings: Settings, db_engine: object
    ) -> None:
        with pytest.raises(SystemExit) as raised:
            await run(
                [
                    "create-user",
                    "--tenant",
                    "nope",
                    "--email",
                    "nobody@nowhere.example",
                    "--password",
                    BOOTSTRAP_PASSWORD,
                ]
            )

        assert raised.value.code == 1


class TestPasswordReset:
    async def test_set_password_replaces_the_password_and_revokes_sessions(
        self,
        cli_settings: Settings,
        api_client: AsyncClient,
        db_engine: object,
    ) -> None:
        await run(["create-tenant", "--slug", "newco", "--name", "New Co"])
        await run(
            [
                "create-user",
                "--tenant",
                "newco",
                "--email",
                "founder@newco.example",
                "--password",
                BOOTSTRAP_PASSWORD,
            ]
        )
        session = (
            await api_client.post(
                "/api/v1/auth/login",
                json={
                    "tenant_slug": "newco",
                    "email": "founder@newco.example",
                    "password": BOOTSTRAP_PASSWORD,
                },
            )
        ).json()

        assert (
            await run(
                [
                    "set-password",
                    "--tenant",
                    "newco",
                    "--email",
                    "founder@newco.example",
                    "--password",
                    "an-entirely-different-password",
                ]
            )
            == 0
        )

        # The old refresh token is dead — exact, because it is a database row.
        assert (
            await api_client.post(
                "/api/v1/auth/refresh", json={"refresh_token": session["refresh_token"]}
            )
        ).status_code == 401
        # And the new password is what works.
        assert (
            await api_client.post(
                "/api/v1/auth/login",
                json={
                    "tenant_slug": "newco",
                    "email": "founder@newco.example",
                    "password": "an-entirely-different-password",
                },
            )
        ).status_code == 200
