"""SQLAlchemy unit of work.

One `async with` is one transaction. It commits on clean exit and rolls back on
any exception, so a service never has to remember either — and a half-applied
multi-table change becomes impossible rather than merely unlikely.

This is what makes docs/adr/0002 work: inserting a document row and enqueuing
its ingestion job happen inside the same transaction, so a job for a document
that does not exist is not a race to be handled but a state that cannot occur.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING
from uuid import UUID

from rag.core.logging import get_logger
from rag.db.repositories.document import (
    SqlAlchemyChunkRepository,
    SqlAlchemyDocumentRepository,
)
from rag.db.repositories.job import SqlAlchemyJobRepository
from rag.db.repositories.tenant import (
    SqlAlchemyCollectionRepository,
    SqlAlchemyGroupRepository,
    SqlAlchemyTenantRepository,
    SqlAlchemyUserRepository,
)
from rag.db.session import set_tenant_scope

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

__all__ = ["SqlAlchemyUnitOfWork"]

_log = get_logger(__name__)


class SqlAlchemyUnitOfWork:
    """Concrete `rag.domain.ports.UnitOfWork`.

    Usage::

        async with SqlAlchemyUnitOfWork(session_factory) as uow:
            await uow.scope_to_tenant(tenant_id)
            document = await uow.documents.add(document)
            await uow.jobs.enqueue(job)
            await uow.commit()

    Without an explicit `commit()` the block rolls back. That is deliberate:
    an implicit commit on exit means a function that raised *after* its last
    write still persists a partial change.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None
        self._tenant_id: UUID | None = None

    async def __aenter__(self) -> SqlAlchemyUnitOfWork:
        self._session = self._session_factory()
        session = self._session

        self.tenants = SqlAlchemyTenantRepository(session)
        self.users = SqlAlchemyUserRepository(session)
        self.groups = SqlAlchemyGroupRepository(session)
        self.collections = SqlAlchemyCollectionRepository(session)
        self.documents = SqlAlchemyDocumentRepository(session)
        self.chunks = SqlAlchemyChunkRepository(session)
        self.jobs = SqlAlchemyJobRepository(session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._session is None:  # pragma: no cover - __aenter__ always sets it
            return
        try:
            if exc is not None:
                await self._session.rollback()
            else:
                # Roll back anything the caller did not explicitly commit.
                # A no-op after commit(); a safety net otherwise.
                await self._session.rollback()
        finally:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> AsyncSession:
        """The underlying session. For migrations, tests, and raw SQL only."""
        if self._session is None:
            raise RuntimeError("UnitOfWork used outside an `async with` block")
        return self._session

    async def commit(self) -> None:
        await self.session.commit()
        # SET LOCAL is transaction-scoped, so committing discards the tenant
        # binding. Reapply it, or every statement after the first commit in a
        # long-lived unit of work would silently see zero rows.
        if self._tenant_id is not None:
            await set_tenant_scope(self.session, self._tenant_id)

    async def rollback(self) -> None:
        await self.session.rollback()
        if self._tenant_id is not None:
            await set_tenant_scope(self.session, self._tenant_id)

    async def scope_to_tenant(self, tenant_id: UUID | None) -> None:
        """Bind row-level security to `tenant_id` for this transaction."""
        self._tenant_id = tenant_id
        await set_tenant_scope(self.session, tenant_id)
