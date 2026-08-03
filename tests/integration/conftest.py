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
from typing import TYPE_CHECKING

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from rag.db.session import create_session_factory
from rag.db.uow import SqlAlchemyUnitOfWork
from rag.domain.enums import Role, UserStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

    from rag.domain.models import Collection, Tenant, User

# Every test runs against a dedicated database, never the development one.
TEST_HOST = os.environ.get("RAG_TEST_DB_HOST", "127.0.0.1")
TEST_PORT = int(os.environ.get("RAG_TEST_DB_PORT", "5432"))
TEST_USER = os.environ.get("RAG_TEST_DB_USER", "rag")
TEST_PASSWORD = os.environ.get("RAG_TEST_DB_PASSWORD", "rag")
TEST_NAME = os.environ.get("RAG_TEST_DB_NAME", "rag_test")

TEST_DSN = f"postgresql+asyncpg://{TEST_USER}:{TEST_PASSWORD}@{TEST_HOST}:{TEST_PORT}/{TEST_NAME}"

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
    "users",
    "jobs",
    "tenants",
)


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
        email="ada@acme.test",
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
