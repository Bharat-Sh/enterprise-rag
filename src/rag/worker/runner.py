"""The worker loop: claim a job, run it, decide what its failure meant.

Three things this file is responsible for getting right.

**One job at a time.** A worker claims a single job, runs it to completion, and
only then polls again. Parsing untrusted files is where runaway memory and
infinite loops live, so a bad document costs one job rather than a batch, and
scaling is by process count — the same story as the API.

**Retryable versus not.** A `DomainError` means the job is wrong and will be
wrong identically forever: an unparseable file does not become parseable on the
fourth attempt, so it is dead-lettered immediately and the document is marked
`FAILED` with the reason. Everything else — a dropped connection, a full disk —
is assumed transient and retried with jittered backoff until `max_attempts`.
Retrying a permanent failure five times just delays the same answer while
burning the queue.

**The tenant scope.** `jobs` is exempt from row-level security precisely so a
worker can poll across tenants (docs/adr/0005). Every claimed job therefore
binds its own tenant before touching anything else, and the unit of work is
opened per job rather than per loop, so one job can never see another's rows.
"""

from __future__ import annotations

import contextlib
import socket
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import anyio

from rag.adapters.blobs.filesystem import FilesystemBlobStore
from rag.adapters.parsers import build_registry
from rag.adapters.tokenize import build_token_counter
from rag.core.logging import get_logger
from rag.db.repositories.job import backoff_delay
from rag.db.uow import SqlAlchemyUnitOfWork
from rag.domain.enums import DocumentStatus, JobKind, JobStatus
from rag.domain.errors import DomainError
from rag.services.pipeline import IngestionPipeline

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from rag.core.config import Settings
    from rag.domain.models import Job

__all__ = ["Worker"]

_log = get_logger(__name__)


class Worker:
    """Polls the queue and runs ingestion jobs until asked to stop."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._name = settings.worker.name or f"{socket.gethostname()}-{id(self):x}"
        self._blobs = FilesystemBlobStore(settings.ingestion.blob_root)
        self._parsers = build_registry(settings.ingestion)
        # Built here, at construction, so a `bge-m3` tokenizer whose vocabulary
        # file is missing kills the worker at startup. Deferring it to first use
        # would mean the worker starts, claims a job, and fails it — repeatedly,
        # across every document in the queue, for a configuration problem.
        self._tokens = build_token_counter(settings.ingestion)
        self._stopping = anyio.Event()
        self._last_reap = datetime.now(UTC) - timedelta(days=1)

    @property
    def name(self) -> str:
        return self._name

    def stop(self) -> None:
        """Ask the loop to exit after the current job.

        Cooperative rather than immediate: killing a worker mid-job leaves it in
        RUNNING until the reaper notices, which is correct but wastes the
        visibility timeout on every deploy.
        """
        self._stopping.set()

    async def run_forever(self) -> None:
        # The tokenizer is logged because it is invisible otherwise and it
        # changes the output: chunks sized by the estimator and chunks sized by
        # BGE-M3's vocabulary have different boundaries, so a corpus ingested
        # across a config change is not internally consistent. This line is what
        # makes that answerable after the fact.
        _log.info(
            "worker.started",
            worker=self._name,
            blob_root=self._settings.ingestion.blob_root,
            tokenizer=str(self._settings.ingestion.tokenizer),
        )
        while not self._stopping.is_set():
            await self._reap_if_due()
            processed = await self.run_once()
            if processed == 0:
                # Nothing to do. Sleep, but wake immediately on shutdown so a
                # SIGTERM is not held up by a full poll interval.
                with contextlib.suppress(TimeoutError):
                    with anyio.fail_after(self._settings.worker.poll_interval_seconds):
                        await self._stopping.wait()
        _log.info("worker.stopped", worker=self._name)

    async def run_once(self) -> int:
        """Claim and run at most one batch. Returns how many jobs ran.

        Public and returning a count so tests can drive the worker
        deterministically instead of starting the loop and sleeping.
        """
        jobs = await self._claim()
        for job in jobs:
            await self._run(job)
        return len(jobs)

    # -- internals ---------------------------------------------------------

    async def _claim(self) -> list[Job]:
        async with SqlAlchemyUnitOfWork(self._session_factory) as uow:
            jobs = await uow.jobs.claim(
                worker_id=self._name,
                kinds=[JobKind.INGEST_DOCUMENT, JobKind.DELETE_DOCUMENT],
                limit=self._settings.worker.batch_size,
            )
            # Committed before running anything: the claim must be durable, or a
            # crash mid-job leaves a row that no reaper can identify as stalled.
            await uow.commit()
            return list(jobs)

    async def _run(self, job: Job) -> None:
        log = _log.bind(job_id=str(job.id), kind=job.kind.value, attempt=job.attempts)

        async with SqlAlchemyUnitOfWork(self._session_factory) as uow:
            # `jobs` is exempt from RLS so the claim above could see every
            # tenant. Everything from here is scoped to this job's tenant.
            await uow.scope_to_tenant(job.tenant_id)
            pipeline = IngestionPipeline(
                uow,
                blobs=self._blobs,
                tokens=self._tokens,
                parser_for=self._parsers.get,
                settings=self._settings.ingestion,
            )

            try:
                if job.kind is JobKind.DELETE_DOCUMENT:
                    await pipeline.purge(job)
                else:
                    await pipeline.ingest(job)
            except DomainError as exc:
                await self._dead_letter(uow, job, exc, log)
            # Broad by design: a worker that dies on an unexpected exception stops
            # processing every other tenant's documents too.
            except Exception as exc:
                await self._retry_or_fail(uow, job, exc, log)
            else:
                await uow.jobs.complete(job.id, JobStatus.SUCCEEDED)
                await uow.commit()
                log.info("worker.job_succeeded")

    async def _dead_letter(
        self, uow: SqlAlchemyUnitOfWork, job: Job, exc: DomainError, log: object
    ) -> None:
        """A permanent failure. Do not retry; record why."""
        await uow.rollback()
        await uow.jobs.complete(job.id, JobStatus.FAILED, error=exc.message)
        await self._mark_document_failed(uow, job, exc.message)
        await uow.commit()
        _log.warning(
            "worker.job_dead_lettered",
            job_id=str(job.id),
            code=exc.code,
            reason=exc.message,
        )

    async def _retry_or_fail(
        self, uow: SqlAlchemyUnitOfWork, job: Job, exc: Exception, log: object
    ) -> None:
        """An unexpected failure. Assume transient until the attempts run out."""
        await uow.rollback()
        reason = f"{type(exc).__name__}: {exc}"

        if job.attempts >= job.max_attempts:
            await uow.jobs.complete(job.id, JobStatus.FAILED, error=reason)
            await self._mark_document_failed(uow, job, reason)
            await uow.commit()
            _log.error("worker.job_exhausted", job_id=str(job.id), reason=reason, exc_info=exc)
            return

        run_after = datetime.now(UTC) + backoff_delay(job.attempts)
        await uow.jobs.reschedule(job.id, run_after=run_after, error=reason)
        await uow.commit()
        _log.warning(
            "worker.job_retrying",
            job_id=str(job.id),
            attempt=job.attempts,
            retry_at=run_after.isoformat(),
            reason=reason,
        )

    async def _mark_document_failed(self, uow: SqlAlchemyUnitOfWork, job: Job, reason: str) -> None:
        """Move the document to FAILED so the failure is visible over the API.

        A job in the dead-letter state that leaves its document sitting in
        `parsing` forever is a support ticket nobody can answer. Best-effort: if
        the document is already terminal, or the transition is illegal, the job
        outcome still stands and forcing it would be worse.
        """
        raw_id = job.payload.get("document_id")
        if not isinstance(raw_id, str):
            return

        from uuid import UUID

        try:
            document_id = UUID(raw_id)
        except ValueError:
            return

        document = await uow.documents.get_for_processing(document_id)
        if document is None or document.status.is_terminal:
            return

        with contextlib.suppress(DomainError):
            await uow.documents.transition_status(
                document_id,
                expected=document.status,
                target=DocumentStatus.FAILED,
                reason=reason[:1000],
            )

    async def _reap_if_due(self) -> None:
        """Requeue jobs whose worker died holding them.

        Every worker runs this, not a dedicated process. It is one indexed
        UPDATE on a partial index, running at most once a minute per worker, and
        a separate reaper process would be another thing to deploy and monitor
        for no benefit.
        """
        now = datetime.now(UTC)
        if (now - self._last_reap).total_seconds() < self._settings.worker.reap_interval_seconds:
            return
        self._last_reap = now

        cutoff = now - timedelta(seconds=self._settings.worker.stalled_after_seconds)
        async with SqlAlchemyUnitOfWork(self._session_factory) as uow:
            reaped = await uow.jobs.reap_stalled(older_than=cutoff)
            await uow.commit()
        if reaped:
            _log.warning("worker.reaped_stalled_jobs", count=reaped)
