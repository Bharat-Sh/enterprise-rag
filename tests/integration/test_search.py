"""Upload a document, let the worker index it, then find it again.

This is the first test in the project that exercises the whole platform end to
end: HTTP upload, blob store, parse, chunk, embed, index, and then a search that
crosses Postgres *and* the vector store and reassembles the two. Everything
before M5 could only ever assert about half of it.

Deliberately not mocked at any layer. The database is real, the vector store is
real (embedded locally, a service container in CI), and the embeddings come from
`StubBackend` — the same implementation CI serves over HTTP. A fake vector store
here would assert that our fake behaves the way we assumed Qdrant does, which is
exactly the assumption these tests exist to check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

from rag.domain.enums import DocumentStatus
from tests.integration.conftest import bearer, login, requires_postgres

if TYPE_CHECKING:
    from httpx import AsyncClient

    from rag.adapters.vectorstore import QdrantVectorStore
    from rag.domain.models import Tenant, User
    from rag.worker.runner import Worker

pytestmark = [pytest.mark.integration, requires_postgres]

SEARCH = "/api/v1/search"

HANDBOOK = (
    "Row-level security is enforced in the database rather than in the application. "
    "A policy that lives in code is a policy a new query can forget.\n\n"
    "Expense claims must be submitted within thirty days. Receipts are required "
    "for any claim over fifty pounds.\n\n"
    "The office is closed on public holidays. Remote work is available to every "
    "employee by default."
)


async def _token(client: AsyncClient, tenant: Tenant, user: User) -> str:
    session = await login(client, tenant_slug=tenant.slug, email=user.email)
    return str(session["access_token"])


async def _index(
    client: AsyncClient,
    token: str,
    worker: Worker,
    collection_id: str,
    *,
    text: str = HANDBOOK,
    title: str = "handbook",
) -> str:
    """Upload, drain the queue, and assert the document really reached READY."""
    response = await client.post(
        "/api/v1/documents",
        headers=bearer(token),
        files={"file": (f"{title}.txt", text.encode("utf-8"), "text/plain")},
        data={"collection_id": collection_id, "title": title},
    )
    assert response.status_code == 202, response.text
    document_id = str(response.json()["id"])

    assert await worker.run_once() == 1
    document = await client.get(f"/api/v1/documents/{document_id}", headers=bearer(token))
    assert document.json()["status"] == DocumentStatus.READY.value, document.text
    return document_id


async def _search(client: AsyncClient, token: str, **body: Any) -> dict[str, Any]:
    response = await client.post(SEARCH, headers=bearer(token), json=body)
    assert response.status_code == 200, response.text
    return dict(response.json())


class TestTheDocumentBecomesFindable:
    async def test_a_document_is_searchable_after_ingestion(
        self, api_client, worker, tenant, member, collection
    ) -> None:
        token = await _token(api_client, tenant, member)
        collection_id = str(collection.id)
        await _index(api_client, token, worker, collection_id)

        body = await _search(api_client, token, query="expense claims and receipts", top_k=5)

        assert body["count"] >= 1
        assert body["results"][0]["document_title"] == "handbook"

    async def test_results_carry_the_text_from_postgres(
        self, api_client, worker, tenant, member, collection
    ) -> None:
        # The vector store holds no text at all, so this proves the hydration
        # step really happened rather than the payload quietly carrying content.
        token = await _token(api_client, tenant, member)
        collection_id = str(collection.id)
        await _index(api_client, token, worker, collection_id)

        body = await _search(api_client, token, query="remote work")

        assert body["results"], "nothing came back"
        first = body["results"][0]
        assert first["text"].strip()
        assert first["char_end"] > first["char_start"]

    async def test_nothing_is_searchable_before_the_worker_runs(
        self, api_client, tenant, member, collection
    ) -> None:
        # READY means indexed, now that `CHUNKING -> READY` is gone. A queued
        # document must not be findable, or "ready" would be a claim rather than
        # a fact.
        token = await _token(api_client, tenant, member)
        collection_id = str(collection.id)
        await api_client.post(
            "/api/v1/documents",
            headers=bearer(token),
            files={"file": ("pending.txt", HANDBOOK.encode("utf-8"), "text/plain")},
            data={"collection_id": collection_id, "title": "pending"},
        )

        body = await _search(api_client, token, query="expense claims")

        assert body["count"] == 0

    async def test_an_empty_index_returns_an_empty_result_not_an_error(
        self, api_client, tenant, member, collection
    ) -> None:
        token = await _token(api_client, tenant, member)

        body = await _search(api_client, token, query="anything at all")

        assert body == {"query": "anything at all", "results": [], "count": 0}


class TestNarrowing:
    async def test_document_ids_restrict_the_search(
        self, api_client, worker, tenant, member, collection
    ) -> None:
        token = await _token(api_client, tenant, member)
        collection_id = str(collection.id)
        first = await _index(api_client, token, worker, collection_id, title="handbook")
        await _index(
            api_client,
            token,
            worker,
            collection_id,
            text="Unrelated content about server provisioning and network topology.",
            title="runbook",
        )

        body = await _search(api_client, token, query="policy", document_ids=[first], top_k=10)

        assert body["count"] >= 1
        assert {result["document_id"] for result in body["results"]} == {first}

    async def test_an_unknown_collection_returns_nothing(
        self, api_client, worker, tenant, member, collection
    ) -> None:
        token = await _token(api_client, tenant, member)
        collection_id = str(collection.id)
        await _index(api_client, token, worker, collection_id)

        body = await _search(api_client, token, query="expense claims", collection_id=str(uuid4()))

        assert body["count"] == 0

    async def test_top_k_bounds_the_result_count(
        self, api_client, worker, tenant, member, collection
    ) -> None:
        token = await _token(api_client, tenant, member)
        collection_id = str(collection.id)
        await _index(api_client, token, worker, collection_id, text=HANDBOOK * 4, title="long")

        body = await _search(api_client, token, query="policy", top_k=2)

        assert len(body["results"]) <= 2


class TestDeletionRemovesItFromTheIndex:
    async def test_a_purged_document_is_no_longer_findable(
        self, api_client, worker, tenant, owner, collection
    ) -> None:
        token = await _token(api_client, tenant, owner)
        collection_id = str(collection.id)
        document_id = await _index(api_client, token, worker, collection_id)
        assert (await _search(api_client, token, query="expense claims"))["count"] >= 1

        delete = await api_client.delete(f"/api/v1/documents/{document_id}", headers=bearer(token))
        assert delete.status_code in {202, 204}, delete.text
        await worker.run_once()

        assert (await _search(api_client, token, query="expense claims"))["count"] == 0

    async def test_the_vectors_are_actually_gone(
        self, api_client, worker, tenant, owner, collection, vector_store: QdrantVectorStore
    ) -> None:
        # Asserted against the store directly, not through search. Search alone
        # cannot distinguish "the vectors were deleted" from "the vectors are
        # still there and hydration hid them", and those are very different
        # states to be in — the second one leaves the index growing forever.
        token = await _token(api_client, tenant, owner)
        collection_id = str(collection.id)
        document_id = await _index(api_client, token, worker, collection_id)
        assert await vector_store.count_for_tenant(tenant.id) > 0

        await api_client.delete(f"/api/v1/documents/{document_id}", headers=bearer(token))
        await worker.run_once()

        assert await vector_store.count_for_tenant(tenant.id) == 0


class TestRedeliveryDoesNotInflateTheIndex:
    async def test_draining_the_queue_again_changes_nothing(
        self, api_client, worker, tenant, member, collection, vector_store: QdrantVectorStore
    ) -> None:
        # At-least-once delivery is real (`reap_stalled`), so the index must not
        # grow a little every time a worker crashes and its job comes back.
        # Upsert idempotency itself is asserted at the adapter level in
        # `test_vector_store.py`; this is the end-to-end consequence of it.
        token = await _token(api_client, tenant, member)
        collection_id = str(collection.id)
        await _index(api_client, token, worker, collection_id)
        after_first = await vector_store.count_for_tenant(tenant.id)

        await worker.run_once()

        assert await vector_store.count_for_tenant(tenant.id) == after_first
        assert (await _search(api_client, token, query="expense claims"))["count"] >= 1


class TestValidation:
    async def test_a_blank_query_is_rejected(self, api_client, tenant, member, collection) -> None:
        token = await _token(api_client, tenant, member)

        response = await api_client.post(SEARCH, headers=bearer(token), json={"query": "   "})

        # Pydantic's `min_length` lets whitespace through; the service strips and
        # rejects. Either way it must not reach the embedder, which would happily
        # return a vector for it and produce arbitrary results that look like a
        # working search.
        assert response.status_code in {400, 422}

    async def test_an_oversized_top_k_is_rejected(
        self, api_client, tenant, member, collection
    ) -> None:
        token = await _token(api_client, tenant, member)

        response = await api_client.post(
            SEARCH, headers=bearer(token), json={"query": "x", "top_k": 10_000}
        )

        assert response.status_code == 422

    async def test_search_requires_a_credential(self, api_client) -> None:
        response = await api_client.post(SEARCH, json={"query": "expense claims"})

        assert response.status_code == 401
