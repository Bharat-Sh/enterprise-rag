"""The Postgres-backed job queue (docs/adr/0002).

The interesting method is `claim`. Everything else is bookkeeping.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import func, select, text, update

from rag.db.models import JobORM
from rag.db.repositories import affected_rows
from rag.domain.enums import JobKind, JobStatus
from rag.domain.models import Job

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = ["SqlAlchemyJobRepository", "backoff_delay"]

#: Truncated exponential backoff, capped so a poisoned job still retries within
#: an operator's attention span rather than drifting to hours.
_BASE_BACKOFF_SECONDS = 5
_MAX_BACKOFF_SECONDS = 600

#: Fraction of the delay that is randomised. "Full jitter" — sampling uniformly
#: across the whole window — spreads a thundering herd best, but makes the first
#: retry occasionally near-instant, which is the opposite of backing off. Half
#: the window keeps a guaranteed floor and still breaks up lockstep.
_JITTER_FRACTION = 0.5


def backoff_delay(attempts: int, *, jitter: Callable[[], float] | None = None) -> timedelta:
    """Delay before the next attempt: ~5s, 10s, 20s, 40s ... capped at 10 min.

    **Jittered**, which M1 deferred to this milestone. Without it, every job
    that failed on a shared cause — a dependency outage, an exhausted quota —
    retries at the same instant, hammering the service just as it recovers and
    knocking it over again. The synchronised herd is the failure mode; the
    backoff curve alone does nothing about it.

    `jitter` returns a value in [0, 1) and is injected so tests can pin it. The
    default is `random.random()`, which is fine here: this is scheduling, not
    security, and `secrets` would buy nothing for it.
    """
    ceiling = min(_BASE_BACKOFF_SECONDS * (2 ** max(attempts - 1, 0)), _MAX_BACKOFF_SECONDS)
    sample = jitter() if jitter is not None else random.random()  # noqa: S311 - not cryptographic
    seconds = ceiling * (1 - _JITTER_FRACTION) + ceiling * _JITTER_FRACTION * sample
    return timedelta(seconds=seconds)


#: Claim and lock in a single statement.
#:
#: `FOR UPDATE SKIP LOCKED` is the whole trick: rows already locked by another
#: worker are passed over rather than waited on, so N workers claim N disjoint
#: sets with no coordination, no broker, and no blocking. Without SKIP LOCKED
#: every worker would queue behind the same row and throughput would collapse
#: to that of a single worker.
#:
#: The CTE selects ids first so the `LIMIT` applies to the lock, then the UPDATE
#: marks exactly those rows. Doing it as one `UPDATE ... WHERE id IN (SELECT
#: ... LIMIT)` without the lock would let two workers select the same row.
_CLAIM_SQL = text(
    """
    WITH claimable AS (
        SELECT id
          FROM jobs
         WHERE status = 'queued'
           AND run_after <= now()
           AND (CAST(:kinds AS text[]) IS NULL OR kind::text = ANY(CAST(:kinds AS text[])))
         ORDER BY priority DESC, run_after
         FOR UPDATE SKIP LOCKED
         LIMIT :limit
    )
    UPDATE jobs
       SET status     = 'running',
           attempts   = jobs.attempts + 1,
           locked_by  = :worker_id,
           locked_at  = now(),
           updated_at = now()
      FROM claimable
     WHERE jobs.id = claimable.id
    RETURNING jobs.id
    """
)


class SqlAlchemyJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def enqueue(
        self,
        *,
        tenant_id: UUID,
        kind: JobKind,
        payload: dict[str, Any] | None = None,
        priority: int = 0,
        max_attempts: int = 5,
        run_after: datetime | None = None,
    ) -> Job:
        """Add a job to the queue.

        Called inside the same transaction as the write it accompanies, which is
        the entire justification for a database-backed queue: a job for a
        document that does not exist is not a race to handle but a state that
        cannot occur.
        """
        orm = JobORM(
            tenant_id=tenant_id,
            kind=kind,
            status=JobStatus.QUEUED,
            payload=payload or {},
            priority=priority,
            max_attempts=max_attempts,
            run_after=run_after or datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.flush()
        await self._session.refresh(orm)
        return orm.to_domain()

    async def claim(
        self, *, worker_id: str, kinds: Sequence[JobKind] | None = None, limit: int = 1
    ) -> Sequence[Job]:
        """Atomically claim up to `limit` runnable jobs."""
        result = await self._session.execute(
            _CLAIM_SQL,
            {
                "worker_id": worker_id,
                "limit": limit,
                "kinds": [kind.value for kind in kinds] if kinds else None,
            },
        )
        claimed_ids = list(result.scalars().all())
        if not claimed_ids:
            return []

        # Re-read through the ORM rather than parsing the RETURNING rows: with
        # raw SQL the JSONB payload comes back as text, because there is no
        # result-type information for the driver to apply a codec from.
        #
        # `populate_existing` forces a refresh of anything already in the
        # identity map, which the raw UPDATE above changed behind the ORM's back.
        rows = await self._session.execute(
            select(JobORM)
            .where(JobORM.id.in_(claimed_ids))
            .order_by(JobORM.priority.desc(), JobORM.run_after)
            .execution_options(populate_existing=True)
        )
        return [orm.to_domain() for orm in rows.scalars().all()]

    async def complete(self, job_id: UUID, status: JobStatus, *, error: str | None = None) -> None:
        """Mark a claimed job finished, successfully or terminally."""
        await self._session.execute(
            update(JobORM)
            .where(JobORM.id == job_id)
            .values(
                status=status,
                last_error=error,
                locked_by=None,
                locked_at=None,
                updated_at=func.now(),
            )
        )

    async def reschedule(self, job_id: UUID, *, run_after: datetime, error: str) -> None:
        """Return a job to the queue after a retryable failure.

        The lock is released and `run_after` pushed into the future, so the job
        becomes claimable again only once the backoff has elapsed.
        """
        await self._session.execute(
            update(JobORM)
            .where(JobORM.id == job_id)
            .values(
                status=JobStatus.QUEUED,
                run_after=run_after,
                last_error=error,
                locked_by=None,
                locked_at=None,
                updated_at=func.now(),
            )
        )

    async def reap_stalled(self, *, older_than: datetime) -> int:
        """Requeue jobs whose worker died while holding them.

        A worker that is SIGKILLed, OOM-killed, or loses its network never gets
        to call `complete` or `reschedule`, so its jobs sit in RUNNING for ever.
        This visibility timeout is what makes at-least-once delivery real.

        Jobs that have exhausted their attempts go to FAILED instead of being
        requeued, so a job that reliably kills its worker cannot loop for ever.
        """
        exhausted = await self._session.execute(
            update(JobORM)
            .where(
                JobORM.status == JobStatus.RUNNING,
                JobORM.locked_at < older_than,
                JobORM.attempts >= JobORM.max_attempts,
            )
            .values(
                status=JobStatus.FAILED,
                last_error="worker stalled; retries exhausted",
                locked_by=None,
                locked_at=None,
                updated_at=func.now(),
            )
        )
        requeued = await self._session.execute(
            update(JobORM)
            .where(
                JobORM.status == JobStatus.RUNNING,
                JobORM.locked_at < older_than,
                JobORM.attempts < JobORM.max_attempts,
            )
            .values(
                status=JobStatus.QUEUED,
                last_error="worker stalled; requeued",
                locked_by=None,
                locked_at=None,
                updated_at=func.now(),
            )
        )
        return affected_rows(exhausted) + affected_rows(requeued)
