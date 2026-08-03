"""Engine, session factory, and tenant scoping.

The critical function here is `set_tenant_scope`. Everything else is plumbing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from rag.core.logging import get_logger

if TYPE_CHECKING:
    from rag.core.config import Settings

__all__ = [
    "TENANT_SETTING",
    "create_engine",
    "create_session_factory",
    "ping",
    "set_tenant_scope",
]

_log = get_logger(__name__)

#: The Postgres run-time parameter the RLS policies read. Custom parameters must
#: contain a dot, otherwise Postgres rejects them as unrecognised.
TENANT_SETTING = "rag.tenant_id"


def create_engine(settings: Settings) -> AsyncEngine:
    """Build the async engine.

    `pool_pre_ping` costs one trivial round trip per checkout and eliminates the
    single most common production error with pooled connections: handing out a
    socket the database closed while it sat idle, which surfaces as a random
    `OperationalError` on an unrelated query.
    """
    return create_async_engine(
        settings.database.dsn,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_pre_ping=True,
        # Recycle below typical proxy and firewall idle timeouts.
        pool_recycle=1800,
        echo=False,
        connect_args={"timeout": settings.database.connect_timeout_seconds},
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory with ORM defaults chosen for async use.

    `expire_on_commit=False` matters here. The default expires every instance on
    commit, so the next attribute access triggers a lazy refresh — which in
    async code raises `MissingGreenlet` rather than quietly issuing a query.
    Since repositories convert to domain dataclasses before returning, nothing
    downstream needs the ORM identity map to stay live.
    """
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autoflush=False,
        class_=AsyncSession,
    )


async def set_tenant_scope(session: AsyncSession, tenant_id: UUID | None) -> None:
    """Bind the tenant for row-level security, for this transaction only.

    Uses `set_config(..., is_local => true)`, the function form of `SET LOCAL`.
    That distinction is the whole ballgame with a connection pool:

    * `SET` persists for the life of the *connection*. Return that connection to
      the pool and the next request to borrow it inherits the previous tenant's
      scope. That is strictly worse than having no RLS, because it looks safe.
    * `SET LOCAL` reverts when the transaction ends, so a pooled connection is
      always handed back clean.

    Passing `None` clears the scope. Because the policies read the setting with
    `missing_ok = true`, an unset value is SQL NULL, `tenant_id = NULL` is NULL,
    and no rows match. Forgetting to scope therefore returns *nothing* rather
    than *everything* — the failure mode that made RLS worth its complexity.
    """
    await session.execute(
        text(f"SELECT set_config('{TENANT_SETTING}', :tenant_id, true)"),
        {"tenant_id": str(tenant_id) if tenant_id is not None else ""},
    )


async def current_tenant_scope(session: AsyncSession) -> UUID | None:
    """Read back the tenant currently bound. Diagnostics and tests."""
    result = await session.execute(
        text(f"SELECT NULLIF(current_setting('{TENANT_SETTING}', true), '')")
    )
    raw = result.scalar_one_or_none()
    return UUID(raw) if raw else None


async def ping(session: AsyncSession) -> None:
    """Readiness check: succeed by returning, fail by raising.

    Deliberately trivial. A readiness probe should answer "is the connection
    usable?", not exercise application tables — a probe that runs a real query
    fails during a long migration and pulls healthy replicas out of rotation.
    """
    await session.execute(text("SELECT 1"))
