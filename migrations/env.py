"""Alembic environment.

Two things worth noting:

* The URL comes from `rag.core.config.Settings`, never from alembic.ini. One
  source of truth means a migration cannot run against a different database
  than the application does.
* Migrations run through the async engine, matching the driver the application
  uses. Running DDL over psycopg while the app runs asyncpg would let a type or
  dialect difference pass here and fail at runtime.
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from rag.core.config import get_settings
from rag.db import models as _models  # noqa: F401  registers tables on Base.metadata
from rag.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

#: Explicit override, checked before application settings.
#:
#: Two real uses: the integration suite migrates `rag_test` without mutating
#: process-wide configuration, and production deployments run migrations as a
#: privileged role distinct from the low-privilege role the application uses.
#: Both are cases where the migration target legitimately differs from the
#: application's own database.
DSN_OVERRIDE_ENV = "RAG_MIGRATION_DSN"

_dsn = os.environ.get(DSN_OVERRIDE_ENV) or get_settings().database.dsn
# configparser treats '%' as interpolation syntax, so a password containing one
# would raise here rather than at connect time. Escape it.
config.set_main_option("sqlalchemy.url", _dsn.replace("%", "%%"))

target_metadata = Base.metadata


def _include_object(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    """Keep autogenerate away from things it does not own."""
    return not (type_ == "table" and name == "alembic_version")


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting — for review or manual application."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Without compare_type, a column changing from VARCHAR(64) to
        # VARCHAR(128) is silently invisible to autogenerate.
        compare_type=True,
        compare_server_default=True,
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Apply migrations against a live database."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
