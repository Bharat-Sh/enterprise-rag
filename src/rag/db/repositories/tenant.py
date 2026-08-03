"""Tenant, user, group, and collection repositories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import delete, select

from rag.db.models import CollectionORM, GroupMemberORM, GroupORM, TenantORM, UserORM
from rag.domain.access import AccessFilter
from rag.domain.enums import Role, UserStatus
from rag.domain.models import Collection, Group, Tenant, User

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "SqlAlchemyCollectionRepository",
    "SqlAlchemyGroupRepository",
    "SqlAlchemyTenantRepository",
    "SqlAlchemyUserRepository",
]


class SqlAlchemyTenantRepository:
    """Tenant lookup and creation.

    Deliberately outside row-level security: this repository is how a request
    resolves *which* tenant it is operating as, so it cannot itself require a
    tenant to already be bound. The `tenants` table carries no policy.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, tenant_id: UUID) -> Tenant | None:
        orm = await self._session.get(TenantORM, tenant_id)
        return orm.to_domain() if orm else None

    async def get_by_slug(self, slug: str) -> Tenant | None:
        result = await self._session.execute(select(TenantORM).where(TenantORM.slug == slug))
        orm = result.scalar_one_or_none()
        return orm.to_domain() if orm else None

    async def create(
        self, *, slug: str, name: str, settings: dict[str, Any] | None = None
    ) -> Tenant:
        orm = TenantORM(slug=slug, name=name, settings=settings or {})
        self._session.add(orm)
        # Flush to obtain server-generated defaults (timestamps) without
        # committing — the caller owns the transaction boundary.
        await self._session.flush()
        await self._session.refresh(orm)
        return orm.to_domain()


class SqlAlchemyUserRepository:
    """Users within the currently scoped tenant.

    Every query here is implicitly filtered by the RLS policy on `users`. The
    explicit `tenant_id` predicates elsewhere in this package are belt and
    braces, not the primary control.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, user_id: UUID) -> User | None:
        orm = await self._session.get(UserORM, user_id)
        return orm.to_domain() if orm else None

    async def get_by_email(self, email: str) -> User | None:
        result = await self._session.execute(select(UserORM).where(UserORM.email == email))
        orm = result.scalar_one_or_none()
        return orm.to_domain() if orm else None

    async def create(
        self,
        *,
        tenant_id: UUID,
        email: str,
        full_name: str = "",
        role: Role | None = None,
        status: UserStatus | None = None,
        password_hash: str | None = None,
    ) -> User:
        orm = UserORM(
            tenant_id=tenant_id,
            email=email,
            full_name=full_name,
            role=role or Role.MEMBER,
            status=status or UserStatus.INVITED,
            password_hash=password_hash,
        )
        self._session.add(orm)
        await self._session.flush()
        await self._session.refresh(orm)
        return orm.to_domain()

    async def group_ids_for(self, user_id: UUID) -> tuple[UUID, ...]:
        result = await self._session.execute(
            select(GroupMemberORM.group_id).where(GroupMemberORM.user_id == user_id)
        )
        return tuple(result.scalars().all())

    async def access_filter_for(self, user: User) -> AccessFilter:
        """Assemble the caller's full principal set for this request.

        Computed rather than stored. That is the whole reason a group membership
        change needs no re-indexing: the document's ACL is stable, and only the
        caller's side of the intersection moves.
        """
        return AccessFilter.build(
            tenant_id=user.tenant_id,
            user_id=user.id,
            role=user.role,
            group_ids=await self.group_ids_for(user.id),
        )


class SqlAlchemyGroupRepository:
    """Groups and their membership."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, *, tenant_id: UUID, slug: str, name: str) -> Group:
        orm = GroupORM(tenant_id=tenant_id, slug=slug, name=name)
        self._session.add(orm)
        await self._session.flush()
        await self._session.refresh(orm)
        return orm.to_domain()

    async def add_member(self, *, group_id: UUID, user_id: UUID, tenant_id: UUID) -> None:
        self._session.add(GroupMemberORM(group_id=group_id, user_id=user_id, tenant_id=tenant_id))
        await self._session.flush()

    async def remove_member(self, *, group_id: UUID, user_id: UUID) -> None:
        await self._session.execute(
            delete(GroupMemberORM).where(
                GroupMemberORM.group_id == group_id,
                GroupMemberORM.user_id == user_id,
            )
        )


class SqlAlchemyCollectionRepository:
    """Collections within the currently scoped tenant."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, collection_id: UUID) -> Collection | None:
        orm = await self._session.get(CollectionORM, collection_id)
        return orm.to_domain() if orm else None

    async def list_all(self) -> Sequence[Collection]:
        result = await self._session.execute(select(CollectionORM).order_by(CollectionORM.name))
        return [orm.to_domain() for orm in result.scalars().all()]

    async def create(
        self, *, tenant_id: UUID, slug: str, name: str, description: str | None = None
    ) -> Collection:
        orm = CollectionORM(tenant_id=tenant_id, slug=slug, name=name, description=description)
        self._session.add(orm)
        await self._session.flush()
        await self._session.refresh(orm)
        return orm.to_domain()
