"""API key and refresh token repositories.

Every query here runs under the row-level security policy on its table,
*including* the two authentication lookups. That is only possible because the
tenant is bound from the credential's own tenant segment before the lookup runs
(docs/adr/0007), which is what lets these tables keep the same protection as the
rest of the customer data rather than being exempted like `jobs`.

The state changes are compare-and-set `UPDATE`s rather than read-then-write.
For `mark_used` that is load-bearing: two requests racing with the same refresh
token must not both succeed, or reuse detection becomes probabilistic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select, update

from rag.db.models import ApiKeyORM, RefreshTokenORM
from rag.db.repositories import affected_rows
from rag.domain.models import ApiKey, RefreshToken

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime, timedelta

    from sqlalchemy.ext.asyncio import AsyncSession

    from rag.domain.enums import Role

__all__ = ["SqlAlchemyApiKeyRepository", "SqlAlchemyRefreshTokenRepository"]


class SqlAlchemyApiKeyRepository:
    """API keys within the currently scoped tenant."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_hash(self, secret_hash: str) -> ApiKey | None:
        """The authentication lookup: one index hit on (tenant_id, secret_hash).

        No constant-time comparison follows, because the index equality check
        *is* the comparison. Timing on it is not exploitable — an attacker would
        need to control the digest being compared, which requires a SHA-256
        preimage.
        """
        result = await self._session.execute(
            select(ApiKeyORM).where(ApiKeyORM.secret_hash == secret_hash)
        )
        orm = result.scalar_one_or_none()
        return orm.to_domain() if orm else None

    async def get(self, key_id: UUID) -> ApiKey | None:
        orm = await self._session.get(ApiKeyORM, key_id)
        return orm.to_domain() if orm else None

    async def create(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        name: str,
        display_prefix: str,
        secret_hash: str,
        role: Role,
        created_by: UUID | None = None,
        expires_at: datetime | None = None,
    ) -> ApiKey:
        orm = ApiKeyORM(
            tenant_id=tenant_id,
            user_id=user_id,
            name=name,
            display_prefix=display_prefix,
            secret_hash=secret_hash,
            role=role,
            created_by=created_by,
            expires_at=expires_at,
        )
        self._session.add(orm)
        await self._session.flush()
        await self._session.refresh(orm)
        return orm.to_domain()

    async def list_for_user(self, user_id: UUID) -> Sequence[ApiKey]:
        result = await self._session.execute(
            select(ApiKeyORM)
            .where(ApiKeyORM.user_id == user_id)
            .order_by(ApiKeyORM.created_at.desc())
        )
        return [orm.to_domain() for orm in result.scalars().all()]

    async def revoke(self, key_id: UUID, *, at: datetime) -> bool:
        """Revoke a key. False if it was already revoked, or is not visible.

        "Not visible" covers both "does not exist" and "belongs to another
        tenant", because RLS makes those the same thing here — which is the
        behaviour we want at the API boundary anyway (404, not 403).
        """
        result = await self._session.execute(
            update(ApiKeyORM)
            .where(ApiKeyORM.id == key_id, ApiKeyORM.revoked_at.is_(None))
            .values(revoked_at=at, updated_at=at)
        )
        return affected_rows(result) > 0

    async def touch_last_used(self, key_id: UUID, *, at: datetime, stale_after: timedelta) -> bool:
        """Record use, but only when the stored value is already stale.

        Writing on every request would make each authenticated GET an update of
        the hottest row in the tenant — WAL amplification and row contention,
        bought with timestamp precision nobody reads. Throttled, it is one write
        per key per `stale_after`, and the return value lets the caller skip the
        commit entirely on the no-op path.

        Deliberately does not touch `updated_at`: last use is not a modification
        of the key, and letting it move would make `updated_at` useless for
        answering "when was this key last *changed*".
        """
        threshold = at - stale_after
        result = await self._session.execute(
            update(ApiKeyORM)
            .where(
                ApiKeyORM.id == key_id,
                (ApiKeyORM.last_used_at.is_(None)) | (ApiKeyORM.last_used_at < threshold),
            )
            .values(last_used_at=at)
        )
        return affected_rows(result) > 0


class SqlAlchemyRefreshTokenRepository:
    """Rotating refresh tokens within the currently scoped tenant."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        token_hash: str,
        family_id: UUID,
        expires_at: datetime,
    ) -> RefreshToken:
        orm = RefreshTokenORM(
            tenant_id=tenant_id,
            user_id=user_id,
            token_hash=token_hash,
            family_id=family_id,
            expires_at=expires_at,
        )
        self._session.add(orm)
        await self._session.flush()
        await self._session.refresh(orm)
        return orm.to_domain()

    async def get_by_hash(self, token_hash: str) -> RefreshToken | None:
        result = await self._session.execute(
            select(RefreshTokenORM).where(RefreshTokenORM.token_hash == token_hash)
        )
        orm = result.scalar_one_or_none()
        return orm.to_domain() if orm else None

    async def mark_used(self, token_id: UUID, *, at: datetime, replaced_by: UUID) -> bool:
        """Consume a token. False means it was already spent or revoked.

        The `used_at IS NULL` predicate is the whole mechanism. A read-then-write
        would let two concurrent refreshes with the same token both observe
        "unused" and both mint a new pair, which is exactly the situation reuse
        detection exists to catch — so the detector must not be the thing with
        the race.
        """
        result = await self._session.execute(
            update(RefreshTokenORM)
            .where(
                RefreshTokenORM.id == token_id,
                RefreshTokenORM.used_at.is_(None),
                RefreshTokenORM.revoked_at.is_(None),
            )
            .values(used_at=at, replaced_by=replaced_by)
        )
        return affected_rows(result) > 0

    async def revoke_family(self, family_id: UUID, *, at: datetime) -> int:
        """Revoke every live token in a rotation lineage.

        The response to a replayed token: two parties hold a single-use
        credential, so one of them stole it, and we cannot tell which. Killing
        the family logs both out, and the legitimate user simply signs in again.
        """
        result = await self._session.execute(
            update(RefreshTokenORM)
            .where(
                RefreshTokenORM.family_id == family_id,
                RefreshTokenORM.revoked_at.is_(None),
            )
            .values(revoked_at=at)
        )
        return affected_rows(result)

    async def revoke_for_user(self, user_id: UUID, *, at: datetime) -> int:
        """Log a user out everywhere. Paired with `invalidate_tokens_before`.

        Both are needed and neither is redundant: this kills the ability to mint
        *new* access tokens, while `users.tokens_valid_after` kills the ones
        already issued.
        """
        result = await self._session.execute(
            update(RefreshTokenORM)
            .where(
                RefreshTokenORM.user_id == user_id,
                RefreshTokenORM.revoked_at.is_(None),
            )
            .values(revoked_at=at)
        )
        return affected_rows(result)
