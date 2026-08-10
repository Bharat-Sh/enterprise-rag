"""Tenant, user, group, and collection repositories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError

from rag.db.models import CollectionORM, GroupMemberORM, GroupORM, TenantORM, UserORM
from rag.domain.access import AccessFilter
from rag.domain.enums import Role, UserStatus
from rag.domain.errors import AlreadyExistsError, NotFoundError
from rag.domain.models import Collection, Group, Tenant, User

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

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
        try:
            # A SAVEPOINT, not a bare flush. Postgres aborts the whole
            # transaction on a constraint violation, so without one this method
            # could only report the conflict by destroying work the caller had
            # already done in the same unit of work. `begin_nested` rolls back
            # to the savepoint and leaves the outer transaction usable.
            async with self._session.begin_nested():
                self._session.add(orm)
                await self._session.flush()
        except IntegrityError as exc:
            # Checking `get_by_email` first is not enough: two requests can both
            # find nothing and both insert. The unique constraint is the only
            # thing that can adjudicate, so the violation is translated here
            # rather than surfacing as a 500 on a race.
            raise AlreadyExistsError(
                f"A user with the address {email!r} already exists in this tenant.",
                details={"email": email},
            ) from exc
        await self._session.refresh(orm)
        return orm.to_domain()

    async def group_ids_for(self, user_id: UUID) -> tuple[UUID, ...]:
        result = await self._session.execute(
            select(GroupMemberORM.group_id).where(GroupMemberORM.user_id == user_id)
        )
        return tuple(result.scalars().all())

    async def access_filter_for(self, user: User, *, role: Role | None = None) -> AccessFilter:
        """Assemble the caller's full principal set for this request.

        Computed rather than stored. That is the whole reason a group membership
        change needs no re-indexing: the document's ACL is stable, and only the
        caller's side of the intersection moves.

        `role` exists so an API key's ceiling reaches the filter. Passing
        `user.role` when the request authenticated with a viewer-scoped key
        would put `role:admin` in the principal set and match admin-granted
        document ACLs — the narrowing would apply to route permissions and
        silently not to data.
        """
        return AccessFilter.build(
            tenant_id=user.tenant_id,
            user_id=user.id,
            role=role if role is not None else user.role,
            group_ids=await self.group_ids_for(user.id),
        )

    async def get_password_hash(self, user_id: UUID) -> str | None:
        """Read the stored hash on its own.

        Never carried on the `User` dataclass: a credential that only two call
        sites need should not ride along on the object every handler, log line,
        and response serialiser touches.
        """
        result = await self._session.execute(
            select(UserORM.password_hash).where(UserORM.id == user_id)
        )
        return result.scalar_one_or_none()

    async def set_password_hash(self, user_id: UUID, password_hash: str) -> None:
        await self._session.execute(
            update(UserORM).where(UserORM.id == user_id).values(password_hash=password_hash)
        )

    async def touch_last_login(self, user_id: UUID, *, at: datetime) -> None:
        await self._session.execute(
            update(UserORM).where(UserORM.id == user_id).values(last_login_at=at)
        )

    async def invalidate_tokens_before(self, user_id: UUID, *, at: datetime) -> None:
        """Reject every access token issued before `at`."""
        await self._session.execute(
            update(UserORM).where(UserORM.id == user_id).values(tokens_valid_after=at)
        )

    async def list_all(self, *, limit: int = 50, offset: int = 0) -> Sequence[User]:
        result = await self._session.execute(
            select(UserORM).order_by(UserORM.email).limit(limit).offset(offset)
        )
        return [orm.to_domain() for orm in result.scalars().all()]

    async def set_role(self, user_id: UUID, role: Role) -> User:
        orm = await self._session.get(UserORM, user_id)
        if orm is None:
            raise NotFoundError("User", str(user_id))
        orm.role = role
        await self._session.flush()
        await self._session.refresh(orm)
        return orm.to_domain()

    async def set_status(self, user_id: UUID, status: UserStatus) -> User:
        orm = await self._session.get(UserORM, user_id)
        if orm is None:
            raise NotFoundError("User", str(user_id))
        orm.status = status
        await self._session.flush()
        await self._session.refresh(orm)
        return orm.to_domain()


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
