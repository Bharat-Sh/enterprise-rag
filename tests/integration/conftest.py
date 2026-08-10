"""Fixtures for tests that need a real PostgreSQL.

**Real database, not a mock.** These tests exist to verify row-level security,
`SKIP LOCKED` under concurrency, unique-constraint behaviour, and array-overlap
operators. Every one of those is a property of Postgres. A mock would assert
that our mock behaves the way we assumed Postgres does, which is precisely the
assumption under test.

**Auto-skip when unavailable.** If nothing is listening on the configured port,
the whole module skips rather than fails. That keeps the suite green on a
machine without a database — including CI jobs that only run unit tests — while
still running for real wherever one exists. A skipped test is honest; a mocked
one is a false pass.

Local development uses a native Postgres install; CI uses a service container.
There is no Docker on the primary dev machine (see CLAUDE.md).
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import pytest
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from rag.adapters.auth.passwords import Argon2PasswordHasher
from rag.api.main import create_app
from rag.core.config import Environment, LogFormat, Settings
from rag.db.session import create_session_factory
from rag.db.uow import SqlAlchemyUnitOfWork
from rag.domain.enums import Role, UserStatus
from tests.support import generate_private_pem

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

    from rag.domain.models import Collection, Tenant, User

# Every test runs against a dedicated database, never the development one.
TEST_HOST = os.environ.get("RAG_TEST_DB_HOST", "127.0.0.1")
TEST_PORT = int(os.environ.get("RAG_TEST_DB_PORT", "5432"))
TEST_USER = os.environ.get("RAG_TEST_DB_USER", "rag")
TEST_PASSWORD_DB = os.environ.get("RAG_TEST_DB_PASSWORD", "rag")
TEST_NAME = os.environ.get("RAG_TEST_DB_NAME", "rag_test")

TEST_DSN = (
    f"postgresql+asyncpg://{TEST_USER}:{TEST_PASSWORD_DB}@{TEST_HOST}:{TEST_PORT}/{TEST_NAME}"
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Truncated between tests. Order is irrelevant given CASCADE, but TRUNCATE is
#: used rather than DELETE precisely because it is not subject to row-level
#: security — a DELETE without a tenant scope bound would match nothing and
#: silently leave the previous test's rows behind.
ALL_TABLES = (
    "chunks",
    "document_permissions",
    "documents",
    "collections",
    "group_members",
    "groups",
    "refresh_tokens",
    "api_keys",
    "users",
    "jobs",
    "tenants",
)

#: The password every seeded user holds. A constant so a test that logs in reads
#: as "log in", not as "manage a credential".
TEST_PASSWORD = "correct-horse-battery-staple"


def _database_reachable() -> bool:
    """Cheap TCP probe. Avoids paying connection setup just to decide to skip."""
    try:
        with socket.create_connection((TEST_HOST, TEST_PORT), timeout=1.0):
            return True
    except OSError:
        return False


requires_postgres = pytest.mark.skipif(
    not _database_reachable(),
    reason=f"No PostgreSQL listening on {TEST_HOST}:{TEST_PORT}",
)


@pytest.fixture(scope="session")
def migrated_database() -> None:
    """Bring the test database to head, once per run.

    Runs the real migrations rather than `Base.metadata.create_all`. That is the
    point: `create_all` would build the tables *and skip every RLS policy*,
    because policies live only in the migration. The suite would then pass
    against a schema the production database never has.
    """
    if not _database_reachable():
        pytest.skip(f"No PostgreSQL listening on {TEST_HOST}:{TEST_PORT}")

    config = Config(os.path.join(PROJECT_ROOT, "alembic.ini"))
    config.set_main_option("script_location", os.path.join(PROJECT_ROOT, "migrations"))

    previous = os.environ.get("RAG_MIGRATION_DSN")
    os.environ["RAG_MIGRATION_DSN"] = TEST_DSN
    try:
        command.upgrade(config, "head")
    finally:
        if previous is None:
            os.environ.pop("RAG_MIGRATION_DSN", None)
        else:
            os.environ["RAG_MIGRATION_DSN"] = previous


@pytest.fixture
async def db_engine(migrated_database: None) -> AsyncIterator[AsyncEngine]:
    """A fresh engine per test, against a truncated database.

    `NullPool` because each test gets its own engine; pooling across engines
    that are immediately disposed only complicates teardown.
    """
    engine = create_async_engine(TEST_DSN, poolclass=NullPool)
    async with engine.begin() as connection:
        await connection.execute(text(f"TRUNCATE {', '.join(ALL_TABLES)} RESTART IDENTITY CASCADE"))
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
def session_factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return create_session_factory(db_engine)


@pytest.fixture
async def uow(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[SqlAlchemyUnitOfWork]:
    """An open unit of work. Tests commit explicitly when they mean to."""
    async with SqlAlchemyUnitOfWork(session_factory) as unit:
        yield unit


@pytest.fixture
async def tenant(uow: SqlAlchemyUnitOfWork) -> Tenant:
    """A committed tenant, with the unit of work scoped to it."""
    created = await uow.tenants.create(slug="acme", name="Acme Corp")
    await uow.commit()
    await uow.scope_to_tenant(created.id)
    return created


@pytest.fixture
async def other_tenant(uow: SqlAlchemyUnitOfWork, tenant: Tenant) -> Tenant:
    """A second tenant, for proving isolation. Never scoped to by default."""
    await uow.scope_to_tenant(None)
    created = await uow.tenants.create(slug="globex", name="Globex Inc")
    await uow.commit()
    await uow.scope_to_tenant(tenant.id)
    return created


@pytest.fixture
async def user(uow: SqlAlchemyUnitOfWork, tenant: Tenant) -> User:
    created = await uow.users.create(
        tenant_id=tenant.id,
        email="ada@acme.example",
        full_name="Ada Lovelace",
        role=Role.MEMBER,
        status=UserStatus.ACTIVE,
    )
    await uow.commit()
    return created


@pytest.fixture
async def collection(uow: SqlAlchemyUnitOfWork, tenant: Tenant) -> Collection:
    created = await uow.collections.create(
        tenant_id=tenant.id, slug="handbook", name="Employee Handbook"
    )
    await uow.commit()
    return created


# --- HTTP-level fixtures ---------------------------------------------------
#
# These drive the real application against the real database, because what they
# assert — that a route binds the row-level-security scope, that a cross-tenant
# id answers 404 — is a property of the whole stack. Testing it below the HTTP
# boundary would verify the layer we are least worried about.


@pytest.fixture
def api_settings() -> Settings:
    """Application settings pointing at the test database.

    Argon2 runs at the configured minimum cost. These tests assert behaviour —
    a wrong password is refused, a changed password revokes sessions — none of
    which depends on how expensive the hash is, and all of which would take a
    minute per run at production parameters.

    The signing key is generated per test, so tokens from one test are worthless
    in another.
    """
    return Settings(
        _env_file=None,
        environment=Environment.LOCAL,
        log_level="WARNING",
        log_format=LogFormat.JSON,
        database={
            "host": TEST_HOST,
            "port": TEST_PORT,
            "user": TEST_USER,
            "password": TEST_PASSWORD_DB,
            "name": TEST_NAME,
        },
        auth={
            "private_key_pem": generate_private_pem(),
            "argon2_time_cost": 1,
            "argon2_memory_cost_kib": 8192,
        },
    )


@pytest.fixture
async def api_app(db_engine: AsyncEngine, api_settings: Settings) -> AsyncIterator[FastAPI]:
    """The real application, with lifespan run, against the truncated test database.

    Depends on `db_engine` for its truncation side effect, so each test starts
    from an empty schema even though the application opens its own pool.

    Exposed separately from `api_client` because the security suite needs
    `app.state.token_service` to mint deliberately malformed tokens *with our
    own key* — a forgery that fails the signature check proves nothing.
    """
    app = create_app(api_settings)
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
async def api_client(api_app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=api_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def hasher(api_settings: Settings) -> Argon2PasswordHasher:
    return Argon2PasswordHasher(api_settings.auth)


@pytest.fixture
async def member(uow: SqlAlchemyUnitOfWork, tenant: Tenant, hasher: Argon2PasswordHasher) -> User:
    """An active member who can actually log in."""
    created = await uow.users.create(
        tenant_id=tenant.id,
        email="member@acme.example",
        full_name="Grace Hopper",
        role=Role.MEMBER,
        status=UserStatus.ACTIVE,
        password_hash=await hasher.hash(TEST_PASSWORD),
    )
    await uow.commit()
    return created


@pytest.fixture
async def owner(uow: SqlAlchemyUnitOfWork, tenant: Tenant, hasher: Argon2PasswordHasher) -> User:
    created = await uow.users.create(
        tenant_id=tenant.id,
        email="owner@acme.example",
        full_name="Ada Lovelace",
        role=Role.OWNER,
        status=UserStatus.ACTIVE,
        password_hash=await hasher.hash(TEST_PASSWORD),
    )
    await uow.commit()
    return created


async def login(client: AsyncClient, *, tenant_slug: str, email: str) -> dict[str, Any]:
    """Log in and return the token payload. Raises loudly if it did not work."""
    response = await client.post(
        "/api/v1/auth/login",
        json={"tenant_slug": tenant_slug, "email": email, "password": TEST_PASSWORD},
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
