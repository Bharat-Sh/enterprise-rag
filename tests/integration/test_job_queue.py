"""The Postgres job queue: SKIP LOCKED claiming, backoff, and stall recovery."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from rag.db.repositories.job import backoff_delay
from rag.db.uow import SqlAlchemyUnitOfWork
from rag.domain.enums import DocumentStatus, JobKind, JobStatus
from tests.integration.conftest import requires_postgres

if TYPE_CHECKING:
    from rag.domain.models import Collection, Tenant

pytestmark = [requires_postgres, pytest.mark.integration]


class TestClaiming:
    async def test_a_queued_job_can_be_claimed(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant
    ) -> None:
        await uow.jobs.enqueue(tenant_id=tenant.id, kind=JobKind.INGEST_DOCUMENT)
        await uow.commit()

        (claimed,) = await uow.jobs.claim(worker_id="worker-1")

        assert claimed.status is JobStatus.RUNNING
        assert claimed.locked_by == "worker-1"
        assert claimed.attempts == 1

    async def test_claiming_returns_nothing_when_the_queue_is_empty(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant
    ) -> None:
        assert await uow.jobs.claim(worker_id="worker-1") == []

    async def test_higher_priority_runs_first(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant
    ) -> None:
        await uow.jobs.enqueue(tenant_id=tenant.id, kind=JobKind.INGEST_DOCUMENT, priority=0)
        await uow.jobs.enqueue(tenant_id=tenant.id, kind=JobKind.REINDEX_DOCUMENT, priority=10)
        await uow.commit()

        (claimed,) = await uow.jobs.claim(worker_id="worker-1")

        assert claimed.kind is JobKind.REINDEX_DOCUMENT

    async def test_jobs_scheduled_for_the_future_are_not_claimable(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant
    ) -> None:
        # This is what makes `run_after` serve as both scheduling and backoff.
        await uow.jobs.enqueue(
            tenant_id=tenant.id,
            kind=JobKind.INGEST_DOCUMENT,
            run_after=datetime.now(UTC) + timedelta(hours=1),
        )
        await uow.commit()

        assert await uow.jobs.claim(worker_id="worker-1") == []

    async def test_claiming_can_be_filtered_by_kind(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant
    ) -> None:
        # Lets a specialised worker pool take only the work it can do — a GPU
        # worker should not claim a deletion job.
        await uow.jobs.enqueue(tenant_id=tenant.id, kind=JobKind.INGEST_DOCUMENT)
        await uow.jobs.enqueue(tenant_id=tenant.id, kind=JobKind.DELETE_DOCUMENT)
        await uow.commit()

        (claimed,) = await uow.jobs.claim(worker_id="worker-1", kinds=[JobKind.DELETE_DOCUMENT])

        assert claimed.kind is JobKind.DELETE_DOCUMENT

    async def test_the_payload_survives_the_round_trip(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant
    ) -> None:
        # Guards the reason `claim` re-reads through the ORM instead of parsing
        # RETURNING rows: raw SQL has no result-type information, so JSONB comes
        # back as text and the payload would arrive as a string.
        payload = {"document_id": "abc", "attempt_hint": 3, "nested": {"a": [1, 2]}}
        await uow.jobs.enqueue(tenant_id=tenant.id, kind=JobKind.INGEST_DOCUMENT, payload=payload)
        await uow.commit()

        (claimed,) = await uow.jobs.claim(worker_id="worker-1")

        assert claimed.payload == payload


class TestSkipLocked:
    async def test_concurrent_workers_claim_disjoint_sets(
        self, session_factory, tenant: Tenant
    ) -> None:
        """The property the whole design rests on.

        Worker A holds locks on its claimed rows in an open transaction. Worker
        B must skip straight past them and take different work — not block, not
        duplicate. Without `SKIP LOCKED` every worker queues behind the same row
        and throughput collapses to that of a single worker.
        """
        async with SqlAlchemyUnitOfWork(session_factory) as setup:
            await setup.scope_to_tenant(tenant.id)
            for index in range(4):
                await setup.jobs.enqueue(
                    tenant_id=tenant.id,
                    kind=JobKind.INGEST_DOCUMENT,
                    payload={"n": index},
                )
            await setup.commit()

        async with SqlAlchemyUnitOfWork(session_factory) as worker_a:
            claimed_a = await worker_a.jobs.claim(worker_id="worker-a", limit=2)
            assert len(claimed_a) == 2

            # Worker A has NOT committed — its rows are still locked.
            async with SqlAlchemyUnitOfWork(session_factory) as worker_b:
                claimed_b = await worker_b.jobs.claim(worker_id="worker-b", limit=2)
                await worker_b.commit()

            await worker_a.commit()

        assert len(claimed_b) == 2
        ids_a = {job.id for job in claimed_a}
        ids_b = {job.id for job in claimed_b}
        assert ids_a.isdisjoint(ids_b), "two workers claimed the same job"

    async def test_a_claimed_job_is_not_claimed_again(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant
    ) -> None:
        await uow.jobs.enqueue(tenant_id=tenant.id, kind=JobKind.INGEST_DOCUMENT)
        await uow.commit()

        first = await uow.jobs.claim(worker_id="worker-1")
        await uow.commit()
        second = await uow.jobs.claim(worker_id="worker-2")

        assert len(first) == 1
        assert second == []


class TestCompletionAndRetry:
    async def test_a_completed_job_leaves_the_queue(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant
    ) -> None:
        await uow.jobs.enqueue(tenant_id=tenant.id, kind=JobKind.INGEST_DOCUMENT)
        await uow.commit()

        (claimed,) = await uow.jobs.claim(worker_id="worker-1")
        await uow.jobs.complete(claimed.id, JobStatus.SUCCEEDED)
        await uow.commit()

        assert await uow.jobs.claim(worker_id="worker-2") == []

    async def test_a_rescheduled_job_becomes_claimable_again(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant
    ) -> None:
        await uow.jobs.enqueue(tenant_id=tenant.id, kind=JobKind.INGEST_DOCUMENT)
        await uow.commit()

        (claimed,) = await uow.jobs.claim(worker_id="worker-1")
        await uow.jobs.reschedule(
            claimed.id, run_after=datetime.now(UTC) - timedelta(seconds=1), error="boom"
        )
        await uow.commit()

        (again,) = await uow.jobs.claim(worker_id="worker-2")

        assert again.id == claimed.id
        assert again.attempts == 2
        assert again.last_error == "boom"

    async def test_backoff_grows_and_is_capped(self) -> None:
        """Pinned at the top of the jitter window, so the curve is the variable.

        M3 made the delay jittered — without it, every job that failed on a
        shared cause retries at the same instant and knocks the recovering
        dependency over again. The ceiling is what this asserts.
        """
        top = {"jitter": lambda: 1.0}

        assert backoff_delay(1, **top) == timedelta(seconds=5)
        assert backoff_delay(2, **top) == timedelta(seconds=10)
        assert backoff_delay(3, **top) == timedelta(seconds=20)
        # Capped, so a poisoned job still retries within an operator's
        # attention span rather than drifting out to hours.
        assert backoff_delay(50, **top) == timedelta(seconds=600)

    async def test_backoff_never_drops_below_half_the_ceiling(self) -> None:
        # Jitter spreads the herd; it must not turn a backoff into a retry storm.
        assert backoff_delay(3, jitter=lambda: 0.0) == timedelta(seconds=10)

    async def test_attempts_are_tracked_towards_exhaustion(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant
    ) -> None:
        await uow.jobs.enqueue(tenant_id=tenant.id, kind=JobKind.INGEST_DOCUMENT, max_attempts=2)
        await uow.commit()

        (first,) = await uow.jobs.claim(worker_id="w1")
        await uow.jobs.reschedule(
            first.id, run_after=datetime.now(UTC) - timedelta(seconds=1), error="1"
        )
        await uow.commit()

        (second,) = await uow.jobs.claim(worker_id="w2")

        assert second.attempts == 2
        assert second.is_exhausted


class TestStallRecovery:
    async def test_a_stalled_job_is_requeued(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant
    ) -> None:
        """A worker that is SIGKILLed never calls complete or reschedule.

        Without a visibility timeout its jobs sit in RUNNING for ever, and
        at-least-once delivery is aspirational rather than real.
        """
        await uow.jobs.enqueue(tenant_id=tenant.id, kind=JobKind.INGEST_DOCUMENT)
        await uow.commit()

        (claimed,) = await uow.jobs.claim(worker_id="doomed-worker")
        await uow.commit()

        reaped = await uow.jobs.reap_stalled(older_than=datetime.now(UTC) + timedelta(minutes=1))
        await uow.commit()

        assert reaped == 1
        (recovered,) = await uow.jobs.claim(worker_id="healthy-worker")
        assert recovered.id == claimed.id

    async def test_a_healthy_in_flight_job_is_left_alone(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant
    ) -> None:
        await uow.jobs.enqueue(tenant_id=tenant.id, kind=JobKind.INGEST_DOCUMENT)
        await uow.commit()
        await uow.jobs.claim(worker_id="busy-worker")
        await uow.commit()

        reaped = await uow.jobs.reap_stalled(older_than=datetime.now(UTC) - timedelta(minutes=5))

        assert reaped == 0

    async def test_a_job_that_keeps_killing_workers_is_dead_lettered(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant
    ) -> None:
        # Otherwise a job that reliably crashes its worker loops for ever,
        # taking down a worker each time round.
        await uow.jobs.enqueue(tenant_id=tenant.id, kind=JobKind.INGEST_DOCUMENT, max_attempts=1)
        await uow.commit()
        await uow.jobs.claim(worker_id="doomed")
        await uow.commit()

        await uow.jobs.reap_stalled(older_than=datetime.now(UTC) + timedelta(minutes=1))
        await uow.commit()

        assert await uow.jobs.claim(worker_id="next") == []
        result = await uow.session.execute(text("SELECT status FROM jobs"))
        assert result.scalar_one() == JobStatus.FAILED.value


class TestTransactionalEnqueue:
    async def test_document_and_job_commit_together(
        self, session_factory, tenant: Tenant, collection: Collection
    ) -> None:
        """The entire justification for a database-backed queue (ADR-0002).

        If the transaction rolls back, neither the document nor its job exists.
        With an external broker these are two systems and the write is a dual
        write — a job referencing a document that was never persisted becomes a
        race to handle rather than a state that cannot occur.
        """
        async with SqlAlchemyUnitOfWork(session_factory) as unit:
            await unit.scope_to_tenant(tenant.id)
            document = await unit.documents.create(
                tenant_id=tenant.id,
                collection_id=collection.id,
                title="Doomed",
                source_uri="s3://bucket/doomed.pdf",
                blob_key="test-blob-key",
                content_hash="e" * 64,
                mime_type="application/pdf",
                size_bytes=10,
            )
            await unit.jobs.enqueue(
                tenant_id=tenant.id,
                kind=JobKind.INGEST_DOCUMENT,
                payload={"document_id": str(document.id)},
            )
            await unit.rollback()

        async with SqlAlchemyUnitOfWork(session_factory) as check:
            await check.scope_to_tenant(tenant.id)
            assert await check.documents.get_by_content_hash("e" * 64) is None
            assert await check.jobs.claim(worker_id="worker-1") == []

    async def test_both_survive_a_commit(
        self, session_factory, tenant: Tenant, collection: Collection
    ) -> None:
        async with SqlAlchemyUnitOfWork(session_factory) as unit:
            await unit.scope_to_tenant(tenant.id)
            document = await unit.documents.create(
                tenant_id=tenant.id,
                collection_id=collection.id,
                title="Kept",
                source_uri="s3://bucket/kept.pdf",
                blob_key="test-blob-key",
                content_hash="f" * 64,
                mime_type="application/pdf",
                size_bytes=10,
            )
            await unit.documents.transition_status(
                document.id,
                expected=DocumentStatus.UPLOADED,
                target=DocumentStatus.QUEUED,
            )
            await unit.jobs.enqueue(
                tenant_id=tenant.id,
                kind=JobKind.INGEST_DOCUMENT,
                payload={"document_id": str(document.id)},
            )
            await unit.commit()

        async with SqlAlchemyUnitOfWork(session_factory) as check:
            await check.scope_to_tenant(tenant.id)
            stored = await check.documents.get_by_content_hash("f" * 64)
            assert stored is not None
            assert stored.status is DocumentStatus.QUEUED

            (job,) = await check.jobs.claim(worker_id="worker-1")
            assert job.payload["document_id"] == str(stored.id)
