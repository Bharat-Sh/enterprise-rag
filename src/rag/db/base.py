"""Declarative base, naming conventions, and shared column mixins."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, MetaData, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column

from rag.core.ids import uuid7

# Without an explicit convention Postgres invents constraint names, Alembic
# autogenerate cannot reliably reference them, and a migration that needs to
# drop a unique constraint has nothing stable to name. Set this on day one:
# retrofitting it means renaming every constraint in an existing database.
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Root of every ORM model."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    # Maps Python annotations to column types, so `Mapped[dict[str, Any]]`
    # becomes JSONB without repeating the type at every call site. JSONB rather
    # than JSON: it is binary, indexable with GIN, and supports containment
    # operators. Plain JSON stores reformatted text and can only be scanned.
    type_annotation_map = {  # noqa: RUF012 - SQLAlchemy reads this as a class attr
        dict[str, Any]: JSONB,
        datetime: DateTime(timezone=True),
    }

    def __repr__(self) -> str:
        identifier = getattr(self, "id", None)
        return f"<{type(self).__name__} id={identifier}>"


class UUIDPrimaryKeyMixin:
    """Client-generated, time-ordered UUIDv7 primary key (see `rag.core.ids`)."""

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid7)


class TimestampMixin:
    """Creation and modification timestamps, always timezone-aware.

    `TIMESTAMPTZ`, never naive: a naive timestamp is only interpretable if you
    also know which server wrote it, which stops being true the moment you have
    two of them.

    Caveat: `onupdate` fires for ORM-level updates only. Bulk `UPDATE`
    statements bypass it, so those set `updated_at` explicitly. A database
    trigger would close the gap; it is not worth the migration complexity while
    the raw-SQL paths are few and reviewed.
    """

    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now(), nullable=False
    )


class TenantScopedMixin:
    """A `tenant_id` foreign key plus the index every scoped query relies on.

    Present on every table holding customer data. It is both the discriminator
    the application filters on and the column the row-level security policy
    evaluates — see the RLS section of migration 0001.
    """

    @declared_attr
    @classmethod
    def tenant_id(cls) -> Mapped[UUID]:
        return mapped_column(
            PG_UUID(as_uuid=True),
            ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
