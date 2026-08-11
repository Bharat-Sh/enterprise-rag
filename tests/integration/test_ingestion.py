"""Ingestion end to end: upload, worker, chunks, deletion.

Against the real application, the real database, and the real worker. The
transactional guarantee being asserted — that a document and its job commit
together — is a property of Postgres, and the state machine only means anything
when two processes drive it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from rag.domain.enums import DocumentStatus, JobStatus
from tests.integration.conftest import bearer, login, requires_postgres

if TYPE_CHECKING:
    from httpx import AsyncClient

    from rag.adapters.blobs.filesystem import FilesystemBlobStore
    from rag.db.uow import SqlAlchemyUnitOfWork
    from rag.domain.models import Tenant, User
    from rag.worker.runner import Worker

pytestmark = [pytest.mark.integration, requires_postgres]

SAMPLE = (
    "The Employee Handbook\n\n"
    "Annual leave is twenty-five days per year, plus public holidays.\n\n"
    "Expenses must be submitted within thirty days of being incurred.\n\n"
) * 5


async def _session(client: AsyncClient, tenant: Tenant, user: User) -> dict[str, Any]:
    return await login(client, tenant_slug=tenant.slug, email=user.email)


async def _collection(client: AsyncClient, token: str, *, slug: str = "handbook") -> str:
    response = await client.post(
        "/api/v1/collections",
        headers=bearer(token),
        json={"slug": slug, "name": "Handbook"},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def _upload(
    client: AsyncClient,
    token: str,
    collection_id: str,
    *,
    content: bytes = SAMPLE.encode(),
    filename: str = "handbook.txt",
    content_type: str = "text/plain",
) -> Any:
    return await client.post(
        "/api/v1/documents",
        headers=bearer(token),
        files={"file": (filename, content, content_type)},
        data={"collection_id": collection_id},
    )


class TestUpload:
    async def test_a_text_upload_is_accepted_and_queued(
        self, api_client: AsyncClient, tenant: Tenant, owner: User
    ) -> None:
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])

        response = await _upload(api_client, session["access_token"], collection_id)

        assert response.status_code == 202, response.text
        body = response.json()
        assert body["status"] == DocumentStatus.QUEUED.value
        assert body["mime_type"] == "text/plain"
        assert body["size_bytes"] == len(SAMPLE.encode())
        # An internal storage location is not part of the API contract.
        assert "blob_key" not in body

    async def test_the_document_and_its_job_commit_together(
        self,
        api_client: AsyncClient,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        owner: User,
    ) -> None:
        """The entire justification for a database-backed queue (ADR-0002).

        A job for a document that does not exist is not a race to handle but a
        state that cannot occur.
        """
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])

        document_id = (await _upload(api_client, session["access_token"], collection_id)).json()[
            "id"
        ]

        await uow.rollback()
        jobs = await uow.jobs.claim(worker_id="test", limit=10)
        assert [job.payload["document_id"] for job in jobs] == [document_id]

    async def test_the_bytes_reach_the_blob_store(
        self,
        api_client: AsyncClient,
        uow: SqlAlchemyUnitOfWork,
        blob_store: FilesystemBlobStore,
        tenant: Tenant,
        owner: User,
    ) -> None:
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        document_id = (await _upload(api_client, session["access_token"], collection_id)).json()[
            "id"
        ]

        await uow.rollback()
        from uuid import UUID

        document = await uow.documents.get_for_processing(UUID(document_id))
        assert document is not None
        assert await blob_store.read(document.blob_key) == SAMPLE.encode()

    async def test_uploading_to_another_tenants_collection_is_404(
        self, api_client: AsyncClient, tenant: Tenant, other_tenant: Tenant, owner: User
    ) -> None:
        # RLS makes it invisible, so "does not exist" and "not yours" are the
        # same answer — which is what stops the endpoint being a collection
        # enumeration oracle.
        session = await _session(api_client, tenant, owner)

        from uuid import uuid4

        response = await _upload(api_client, session["access_token"], str(uuid4()))

        assert response.status_code == 404


class TestRejections:
    async def test_an_empty_file_is_rejected(
        self, api_client: AsyncClient, tenant: Tenant, owner: User
    ) -> None:
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])

        response = await _upload(api_client, session["access_token"], collection_id, content=b"")

        assert response.status_code == 400
        assert response.json()["code"] == "invalid_input"

    async def test_a_binary_file_is_rejected_as_unsupported(
        self, api_client: AsyncClient, tenant: Tenant, owner: User
    ) -> None:
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])

        response = await _upload(
            api_client,
            session["access_token"],
            collection_id,
            content=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
            filename="image.png",
            content_type="image/png",
        )

        assert response.status_code == 415
        assert response.json()["code"] == "unsupported_media_type"

    async def test_a_lying_content_type_is_ignored_in_favour_of_the_bytes(
        self, api_client: AsyncClient, tenant: Tenant, owner: User
    ) -> None:
        """A PDF renamed and re-declared as plain text.

        Now that PDF is parseable, the interesting assertion is stronger than a
        refusal: the document is accepted and recorded as a PDF, so it reaches
        the PDF parser. Trusting the declaration would have handed a binary to a
        text decoder and produced mojibake chunks that look like success.
        """
        from tests.support import make_pdf

        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])

        response = await _upload(
            api_client,
            session["access_token"],
            collection_id,
            content=make_pdf(["Real PDF content."]),
            filename="notes.txt",
            content_type="text/plain",
        )

        assert response.status_code == 202
        assert response.json()["mime_type"] == "application/pdf"

    async def test_an_oversized_upload_is_rejected(
        self, api_client: AsyncClient, api_settings: object, tenant: Tenant, owner: User
    ) -> None:
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])

        # Comfortably past the default 50 MiB limit would be slow to build; the
        # middleware's Content-Length check rejects before any body is read.
        oversized = b"x" * (50 * 1024 * 1024 + 1)
        response = await _upload(
            api_client, session["access_token"], collection_id, content=oversized
        )

        assert response.status_code == 413
        assert response.json()["code"] == "payload_too_large"

    async def test_a_viewer_cannot_upload(
        self, api_client: AsyncClient, uow: SqlAlchemyUnitOfWork, tenant: Tenant, owner: User
    ) -> None:
        from rag.domain.enums import Role

        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        await uow.users.set_role(owner.id, Role.VIEWER)
        await uow.commit()

        response = await _upload(api_client, session["access_token"], collection_id)

        assert response.status_code == 403


class TestPipeline:
    async def test_the_worker_drives_a_document_to_ready(
        self, api_client: AsyncClient, worker: Worker, tenant: Tenant, owner: User
    ) -> None:
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        document_id = (await _upload(api_client, session["access_token"], collection_id)).json()[
            "id"
        ]

        assert await worker.run_once() == 1

        document = await api_client.get(
            f"/api/v1/documents/{document_id}", headers=bearer(session["access_token"])
        )
        assert document.json()["status"] == DocumentStatus.READY.value

    async def test_chunks_are_produced_with_usable_offsets(
        self, api_client: AsyncClient, worker: Worker, tenant: Tenant, owner: User
    ) -> None:
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        document_id = (await _upload(api_client, session["access_token"], collection_id)).json()[
            "id"
        ]
        await worker.run_once()

        response = await api_client.get(
            f"/api/v1/documents/{document_id}/chunks",
            headers=bearer(session["access_token"]),
        )

        chunks = response.json()
        assert chunks
        assert [chunk["ordinal"] for chunk in chunks] == list(range(len(chunks)))
        for chunk in chunks:
            assert chunk["text"].strip()
            assert chunk["char_end"] > chunk["char_start"]
            assert chunk["token_count"] > 0

    async def test_chunks_inherit_the_documents_acl(
        self,
        api_client: AsyncClient,
        uow: SqlAlchemyUnitOfWork,
        worker: Worker,
        tenant: Tenant,
        owner: User,
    ) -> None:
        """Inheritance happens in the repository so it cannot be forgotten.

        A chunk written with the wrong ACL is directly a data leak, because
        retrieval matches against the chunk array (docs/adr/0006).
        """
        from uuid import UUID

        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        document_id = UUID(
            (await _upload(api_client, session["access_token"], collection_id)).json()["id"]
        )
        await worker.run_once()

        await uow.rollback()
        document = await uow.documents.get_for_processing(document_id)
        chunks = await uow.chunks.list_for_document(document_id)
        assert document is not None
        assert chunks
        for chunk in chunks:
            assert set(chunk.acl_principals) == set(document.acl_principals)

    async def test_the_extracted_text_is_stored_for_re_chunking(
        self,
        api_client: AsyncClient,
        uow: SqlAlchemyUnitOfWork,
        blob_store: FilesystemBlobStore,
        worker: Worker,
        tenant: Tenant,
        owner: User,
    ) -> None:
        # So a future change to chunking is a re-chunk rather than a re-parse of
        # every document ever ingested.
        from uuid import UUID

        from rag.services.pipeline import EXTRACTED_TEXT_KIND

        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        document_id = UUID(
            (await _upload(api_client, session["access_token"], collection_id)).json()["id"]
        )
        await worker.run_once()

        await uow.rollback()
        document = await uow.documents.get_for_processing(document_id)
        assert document is not None
        stored = await blob_store.read(
            blob_store.derived_key_for(
                tenant_id=document.tenant_id,
                content_hash=document.content_hash,
                kind=EXTRACTED_TEXT_KIND,
            )
        )
        assert "Annual leave" in stored.decode()

    async def test_html_is_parsed_to_text(
        self, api_client: AsyncClient, worker: Worker, tenant: Tenant, owner: User
    ) -> None:
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        html = b"<html><body><script>var x=1;</script><p>Visible prose here.</p></body></html>"
        document_id = (
            await _upload(
                api_client,
                session["access_token"],
                collection_id,
                content=html,
                filename="page.html",
                content_type="text/html",
            )
        ).json()["id"]

        await worker.run_once()

        chunks = (
            await api_client.get(
                f"/api/v1/documents/{document_id}/chunks",
                headers=bearer(session["access_token"]),
            )
        ).json()
        combined = " ".join(chunk["text"] for chunk in chunks)
        assert "Visible prose here." in combined
        assert "var x=1" not in combined

    async def test_a_pdf_is_parsed_end_to_end(
        self, api_client: AsyncClient, worker: Worker, tenant: Tenant, owner: User
    ) -> None:
        from tests.support import make_pdf

        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        document_id = (
            await _upload(
                api_client,
                session["access_token"],
                collection_id,
                content=make_pdf(["Annual leave is twenty-five days per year."]),
                filename="handbook.pdf",
                content_type="application/pdf",
            )
        ).json()["id"]

        await worker.run_once()

        document = (
            await api_client.get(
                f"/api/v1/documents/{document_id}", headers=bearer(session["access_token"])
            )
        ).json()
        assert document["status"] == DocumentStatus.READY.value
        assert document["mime_type"] == "application/pdf"

        chunks = (
            await api_client.get(
                f"/api/v1/documents/{document_id}/chunks",
                headers=bearer(session["access_token"]),
            )
        ).json()
        assert "Annual leave" in " ".join(chunk["text"] for chunk in chunks)

    async def test_a_docx_is_parsed_end_to_end(
        self, api_client: AsyncClient, worker: Worker, tenant: Tenant, owner: User
    ) -> None:
        from tests.support import make_docx

        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        document_id = (
            await _upload(
                api_client,
                session["access_token"],
                collection_id,
                content=make_docx(["Expenses must be submitted within thirty days."]),
                filename="policy.docx",
                content_type=(
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ),
            )
        ).json()["id"]

        await worker.run_once()

        chunks = (
            await api_client.get(
                f"/api/v1/documents/{document_id}/chunks",
                headers=bearer(session["access_token"]),
            )
        ).json()
        assert "Expenses must be submitted" in " ".join(chunk["text"] for chunk in chunks)

    async def test_a_malicious_docx_fails_the_document_not_the_worker(
        self, api_client: AsyncClient, worker: Worker, tenant: Tenant, owner: User
    ) -> None:
        """The end-to-end shape of the hostile-input defence.

        An entity-expansion payload is accepted at upload — it is a valid zip
        with a `word/document.xml`, and the boundary cannot tell — and refused
        by the parser. What matters is that the worker survives, the document is
        marked `failed` with a reason, and the queue keeps moving.
        """
        from tests.support import make_docx

        payload = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>&xxe;</w:t></w:r></w:p></w:body></w:document>"
        )
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        document_id = (
            await _upload(
                api_client,
                session["access_token"],
                collection_id,
                content=make_docx(document_xml=payload),
                filename="evil.docx",
                content_type=(
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ),
            )
        ).json()["id"]

        assert await worker.run_once() == 1

        document = (
            await api_client.get(
                f"/api/v1/documents/{document_id}", headers=bearer(session["access_token"])
            )
        ).json()
        assert document["status"] == DocumentStatus.FAILED.value
        assert document["status_reason"]
        # Nothing from the host filesystem became searchable.
        chunks = (
            await api_client.get(
                f"/api/v1/documents/{document_id}/chunks",
                headers=bearer(session["access_token"]),
            )
        ).json()
        assert chunks == []

    async def test_a_document_with_no_extractable_text_fails_loudly(
        self, api_client: AsyncClient, worker: Worker, tenant: Tenant, owner: User
    ) -> None:
        """Better than a READY document with zero chunks.

        That would claim to be searchable while being permanently unfindable,
        and nothing would have errored.
        """
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        document_id = (
            await _upload(
                api_client,
                session["access_token"],
                collection_id,
                content=b"<html><head><title>x</title></head><body></body></html>",
                filename="empty.html",
                content_type="text/html",
            )
        ).json()["id"]

        await worker.run_once()

        document = (
            await api_client.get(
                f"/api/v1/documents/{document_id}", headers=bearer(session["access_token"])
            )
        ).json()
        assert document["status"] == DocumentStatus.FAILED.value
        assert document["status_reason"]

    async def test_a_permanent_failure_is_not_retried(
        self,
        api_client: AsyncClient,
        uow: SqlAlchemyUnitOfWork,
        worker: Worker,
        tenant: Tenant,
        owner: User,
    ) -> None:
        # An unparseable file does not become parseable on the fourth attempt,
        # so retrying just delays the same answer while burning the queue.
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        await _upload(
            api_client,
            session["access_token"],
            collection_id,
            content=b"<html><body></body></html>",
            filename="empty.html",
            content_type="text/html",
        )
        await worker.run_once()

        await uow.rollback()
        assert await worker.run_once() == 0  # nothing left to claim

    async def test_running_the_worker_twice_is_idempotent(
        self, api_client: AsyncClient, worker: Worker, tenant: Tenant, owner: User
    ) -> None:
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        document_id = (await _upload(api_client, session["access_token"], collection_id)).json()[
            "id"
        ]

        await worker.run_once()
        first = (
            await api_client.get(
                f"/api/v1/documents/{document_id}/chunks",
                headers=bearer(session["access_token"]),
            )
        ).json()
        await worker.run_once()
        second = (
            await api_client.get(
                f"/api/v1/documents/{document_id}/chunks",
                headers=bearer(session["access_token"]),
            )
        ).json()

        assert len(first) == len(second)


class TestIdempotency:
    async def test_re_uploading_identical_bytes_returns_the_same_document(
        self, api_client: AsyncClient, tenant: Tenant, owner: User
    ) -> None:
        # `uq_documents_tenant_id_content_hash` exists so this cannot create a
        # second document or pay the embedding cost twice.
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])

        first = await _upload(api_client, session["access_token"], collection_id)
        second = await _upload(api_client, session["access_token"], collection_id)

        assert first.status_code == 202
        assert second.status_code == 200
        assert first.json()["id"] == second.json()["id"]

    async def test_simultaneous_identical_uploads_do_not_500(
        self, api_client: AsyncClient, tenant: Tenant, owner: User
    ) -> None:
        """The race a read-then-insert cannot win.

        Both requests pass the content-hash probe and both reach the insert, so
        `uq_documents_tenant_id_content_hash` is the only thing that can
        adjudicate. The loser must see the same idempotent answer the probe
        would have given, not a conflict it cannot act on.
        """
        import asyncio

        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])

        responses = await asyncio.gather(
            *(_upload(api_client, session["access_token"], collection_id) for _ in range(4))
        )

        statuses = [response.status_code for response in responses]
        assert statuses.count(202) == 1, "exactly one request should have created it"
        assert statuses.count(200) == 3, "the rest should be idempotent, not conflicts"
        assert len({response.json()["id"] for response in responses}) == 1

    async def test_re_uploading_after_a_failure_requeues(
        self, api_client: AsyncClient, worker: Worker, tenant: Tenant, owner: User
    ) -> None:
        """Otherwise a user who retries after we fixed the bug gets a permanent no."""
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        empty_html = b"<html><body></body></html>"
        document_id = (
            await _upload(
                api_client,
                session["access_token"],
                collection_id,
                content=empty_html,
                filename="empty.html",
                content_type="text/html",
            )
        ).json()["id"]
        await worker.run_once()

        again = await _upload(
            api_client,
            session["access_token"],
            collection_id,
            content=empty_html,
            filename="empty.html",
            content_type="text/html",
        )

        assert again.status_code == 200
        assert again.json()["id"] == document_id
        assert again.json()["status"] == DocumentStatus.QUEUED.value


class TestDeletion:
    async def test_deletion_purges_chunks_and_bytes(
        self,
        api_client: AsyncClient,
        uow: SqlAlchemyUnitOfWork,
        blob_store: FilesystemBlobStore,
        worker: Worker,
        tenant: Tenant,
        owner: User,
    ) -> None:
        from uuid import UUID

        from rag.domain.errors import NotFoundError

        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        document_id = UUID(
            (await _upload(api_client, session["access_token"], collection_id)).json()["id"]
        )
        await worker.run_once()

        await uow.rollback()
        document = await uow.documents.get_for_processing(document_id)
        assert document is not None
        blob_key = document.blob_key

        deleted = await api_client.delete(
            f"/api/v1/documents/{document_id}", headers=bearer(session["access_token"])
        )
        assert deleted.status_code == 202
        await worker.run_once()

        await uow.rollback()
        purged = await uow.documents.get_for_processing(document_id)
        assert purged is not None
        assert purged.status is DocumentStatus.DELETED
        assert await uow.chunks.list_for_document(document_id) == []
        with pytest.raises(NotFoundError):
            await blob_store.read(blob_key)

    async def test_a_member_cannot_delete(
        self, api_client: AsyncClient, tenant: Tenant, owner: User, member: User
    ) -> None:
        # DOCUMENT_DELETE is an admin power — the conservative call, with no
        # per-document ownership model to maintain.
        owner_session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, owner_session["access_token"])
        document_id = (
            await _upload(api_client, owner_session["access_token"], collection_id)
        ).json()["id"]
        member_session = await _session(api_client, tenant, member)

        response = await api_client.delete(
            f"/api/v1/documents/{document_id}",
            headers=bearer(member_session["access_token"]),
        )

        assert response.status_code == 403


class TestJobOutcomes:
    async def test_a_successful_job_is_marked_succeeded(
        self,
        api_client: AsyncClient,
        uow: SqlAlchemyUnitOfWork,
        worker: Worker,
        tenant: Tenant,
        owner: User,
    ) -> None:
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        await _upload(api_client, session["access_token"], collection_id)

        await worker.run_once()

        await uow.rollback()
        from sqlalchemy import text as sql

        result = await uow.session.execute(sql("SELECT status FROM jobs"))
        assert [row[0] for row in result] == [JobStatus.SUCCEEDED.value]

    async def test_an_empty_queue_is_a_no_op(self, worker: Worker) -> None:
        assert await worker.run_once() == 0

    async def test_redelivering_a_finished_ingest_job_is_not_a_failure(
        self,
        api_client: AsyncClient,
        uow: SqlAlchemyUnitOfWork,
        worker: Worker,
        tenant: Tenant,
        owner: User,
    ) -> None:
        """At-least-once delivery means this happens for real.

        A worker that dies between finishing the work and marking the job
        complete leaves a job the reaper will requeue. Without a guard the retry
        would attempt `QUEUED -> PARSING` on a `READY` document, be
        dead-lettered, and report a failure for work that succeeded.
        """
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        document_id = (await _upload(api_client, session["access_token"], collection_id)).json()[
            "id"
        ]
        await worker.run_once()

        # Simulate the crash: put the completed job back on the queue.
        await uow.rollback()
        from sqlalchemy import text as sql

        await uow.session.execute(
            sql("UPDATE jobs SET status = 'queued', locked_by = NULL, locked_at = NULL")
        )
        await uow.commit()

        assert await worker.run_once() == 1

        await uow.rollback()
        result = await uow.session.execute(sql("SELECT status FROM jobs"))
        assert [row[0] for row in result] == [JobStatus.SUCCEEDED.value]
        document = (
            await api_client.get(
                f"/api/v1/documents/{document_id}", headers=bearer(session["access_token"])
            )
        ).json()
        assert document["status"] == DocumentStatus.READY.value

    async def test_redelivering_a_finished_purge_job_is_not_a_failure(
        self,
        api_client: AsyncClient,
        uow: SqlAlchemyUnitOfWork,
        worker: Worker,
        tenant: Tenant,
        owner: User,
    ) -> None:
        session = await _session(api_client, tenant, owner)
        collection_id = await _collection(api_client, session["access_token"])
        document_id = (await _upload(api_client, session["access_token"], collection_id)).json()[
            "id"
        ]
        await worker.run_once()
        await api_client.delete(
            f"/api/v1/documents/{document_id}", headers=bearer(session["access_token"])
        )
        await worker.run_once()

        await uow.rollback()
        from sqlalchemy import text as sql

        await uow.session.execute(
            sql("UPDATE jobs SET status = 'queued' WHERE kind = 'delete_document'")
        )
        await uow.commit()

        assert await worker.run_once() == 1

        await uow.rollback()
        result = await uow.session.execute(
            sql("SELECT status FROM jobs WHERE kind = 'delete_document'")
        )
        assert [row[0] for row in result] == [JobStatus.SUCCEEDED.value]
