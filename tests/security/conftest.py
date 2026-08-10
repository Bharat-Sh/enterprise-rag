"""Fixtures for the security suite.

Re-bound from `tests.integration.conftest` rather than duplicated. pytest only
discovers fixtures in the conftest files along a test's own path, so a sibling
package cannot see them — importing the names here makes them fixtures in this
directory too, with one definition and no drift.
"""

from __future__ import annotations

from tests.integration.conftest import (
    api_app,
    api_client,
    api_settings,
    bearer,
    db_engine,
    hasher,
    login,
    member,
    migrated_database,
    other_tenant,
    owner,
    requires_postgres,
    session_factory,
    tenant,
    uow,
    user,
)

__all__ = [
    "api_app",
    "api_client",
    "api_settings",
    "bearer",
    "db_engine",
    "hasher",
    "login",
    "member",
    "migrated_database",
    "other_tenant",
    "owner",
    "requires_postgres",
    "session_factory",
    "tenant",
    "uow",
    "user",
]
