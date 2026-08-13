"""The Qdrant adapter against a real store.

Local mode on a machine with no Docker, a service container in CI. The
assertions are identical against both, which is the point: the isolation
properties proved here are properties of Qdrant's filtering, not of a fake.

What differs between the two backings is only performance. Local mode warns
that payload indexes have no effect, so every filter is a full scan — correct
answers, no index. Nothing here asserts anything about speed, so the results
mean exactly the same thing either way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest

from model_service.backend import STUB_DIMENSIONS
from rag.domain.access import AccessFilter
from rag.domain.embedding import Embedding, SparseVector
from rag.domain.enums import Role
from rag.domain.retrieval import VectorPoint

if TYPE_CHECKING:
    from rag.adapters.vectorstore import QdrantVectorStore

pytestmark = pytest.mark.integration

TENANT_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
TENANT_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def _embedding(*, lead: float = 1.0, dimensions: int = STUB_DIMENSIONS) -> Embedding:
    """A unit vector pointing mostly along one axis."""
    dense = [0.0] * dimensions
    dense[0] = lead
    dense[1] = (1.0 - lead**2) ** 0.5
    return Embedding(
        dense=tuple(dense),
        sparse=SparseVector(indices=(1, 42), values=(0.5, 0.9)),
    )


def _point(
    *,
    tenant_id: UUID,
    acl: tuple[str, ...],
    document_id: UUID | None = None,
    chunk_id: UUID | None = None,
    lead: float = 1.0,
    collection_id: UUID | None = None,
) -> VectorPoint:
    return VectorPoint(
        chunk_id=chunk_id or uuid4(),
        tenant_id=tenant_id,
        document_id=document_id or uuid4(),
        ordinal=0,
        embedding=_embedding(lead=lead),
        acl_principals=acl,
        collection_id=collection_id,
    )


def _access(tenant_id: UUID, *, user_id: UUID | None = None) -> AccessFilter:
    return AccessFilter.build(tenant_id=tenant_id, user_id=user_id or uuid4(), role=Role.MEMBER)


class TestTenantScoping:
    """Scoping that is *not* the isolation guarantee itself.

    The cross-tenant leakage assertions live in
    `tests/security/test_vector_isolation.py`, where they belong and where the
    skip guard protects them. What is left here is the ordinary behaviour of
    `count_for_tenant`, which the tests below rely on to mean anything.
    """

    async def test_counting_is_scoped_to_one_tenant(self, vector_store: QdrantVectorStore) -> None:
        await vector_store.upsert(
            [
                _point(tenant_id=TENANT_A, acl=("tenant:a",)),
                _point(tenant_id=TENANT_B, acl=("tenant:b",)),
                _point(tenant_id=TENANT_B, acl=("tenant:b",)),
            ]
        )

        assert await vector_store.count_for_tenant(TENANT_A) == 1
        assert await vector_store.count_for_tenant(TENANT_B) == 2


class TestTheAclPreFilter:
    async def test_a_point_the_caller_does_not_match_is_not_returned(
        self, vector_store: QdrantVectorStore
    ) -> None:
        access = _access(TENANT_A)
        visible = _point(tenant_id=TENANT_A, acl=access.principal_tokens)
        hidden = _point(tenant_id=TENANT_A, acl=("group:finance",))
        await vector_store.upsert([visible, hidden])

        hits = await vector_store.search(_embedding(), access, limit=10)

        assert [hit.chunk_id for hit in hits] == [visible.chunk_id]

    async def test_one_matching_principal_is_enough(self, vector_store: QdrantVectorStore) -> None:
        # `match_any` is an overlap test, the Qdrant twin of Postgres `&&`.
        access = _access(TENANT_A)
        acl = ("group:finance", access.principal_tokens[0], "group:legal")
        point = _point(tenant_id=TENANT_A, acl=acl)
        await vector_store.upsert([point])

        hits = await vector_store.search(_embedding(), access, limit=10)

        assert [hit.chunk_id for hit in hits] == [point.chunk_id]

    async def test_a_point_with_no_acl_is_visible_to_nobody(
        self, vector_store: QdrantVectorStore
    ) -> None:
        # Fails closed. An empty ACL array is what a document gets if its
        # permissions were never projected, and the safe reading of "no
        # principals granted" is "nobody", not "everybody".
        await vector_store.upsert([_point(tenant_id=TENANT_A, acl=())])

        hits = await vector_store.search(_embedding(), _access(TENANT_A), limit=10)

        assert hits == []


class TestUpsertIsIdempotent:
    async def test_the_same_chunk_written_twice_is_one_point(
        self, vector_store: QdrantVectorStore
    ) -> None:
        # Point id *is* chunk id (ADR-0001). This is what makes the whole write
        # path safe to retry after a crash with no bookkeeping about which
        # points already landed — and what stops at-least-once delivery from
        # inflating the index a little on every redelivery.
        chunk_id = uuid4()
        point = _point(tenant_id=TENANT_A, acl=("tenant:a",), chunk_id=chunk_id)

        await vector_store.upsert([point])
        await vector_store.upsert([point])
        await vector_store.upsert([point])

        assert await vector_store.count_for_tenant(TENANT_A) == 1

    async def test_rewriting_a_chunk_updates_its_payload(
        self, vector_store: QdrantVectorStore
    ) -> None:
        # The mechanism an ACL change relies on: re-indexing with a new ACL must
        # replace the old one, not sit alongside it.
        chunk_id = uuid4()
        access = _access(TENANT_A)
        await vector_store.upsert(
            [_point(tenant_id=TENANT_A, acl=("group:finance",), chunk_id=chunk_id)]
        )
        assert await vector_store.search(_embedding(), access, limit=10) == []

        await vector_store.upsert(
            [_point(tenant_id=TENANT_A, acl=access.principal_tokens, chunk_id=chunk_id)]
        )

        hits = await vector_store.search(_embedding(), access, limit=10)
        assert [hit.chunk_id for hit in hits] == [chunk_id]


class TestDeletion:
    async def test_it_removes_only_the_named_document(
        self, vector_store: QdrantVectorStore
    ) -> None:
        keep_document, drop_document = uuid4(), uuid4()
        await vector_store.upsert(
            [
                _point(tenant_id=TENANT_A, acl=("tenant:a",), document_id=keep_document),
                _point(tenant_id=TENANT_A, acl=("tenant:a",), document_id=drop_document),
                _point(tenant_id=TENANT_A, acl=("tenant:a",), document_id=drop_document),
            ]
        )

        await vector_store.delete_for_document(TENANT_A, drop_document)

        assert await vector_store.count_for_tenant(TENANT_A) == 1

    async def test_deleting_is_scoped_to_the_tenant(self, vector_store: QdrantVectorStore) -> None:
        # A document id is a UUID and therefore already unique, so this can only
        # fail if the tenant clause were dropped from the deletion filter. It is
        # asserted anyway: deletion is one copy-paste away from search, and the
        # habit of always carrying the tenant is what is actually being pinned.
        shared_document = uuid4()
        await vector_store.upsert(
            [
                _point(tenant_id=TENANT_A, acl=("tenant:a",), document_id=shared_document),
                _point(tenant_id=TENANT_B, acl=("tenant:b",), document_id=shared_document),
            ]
        )

        await vector_store.delete_for_document(TENANT_A, shared_document)

        assert await vector_store.count_for_tenant(TENANT_A) == 0
        assert await vector_store.count_for_tenant(TENANT_B) == 1

    async def test_deleting_an_unknown_document_is_not_an_error(
        self, vector_store: QdrantVectorStore
    ) -> None:
        # Purge is redelivered under at-least-once, so the second run must be a
        # no-op rather than a failure that dead-letters completed work.
        await vector_store.delete_for_document(TENANT_A, uuid4())


class TestRanking:
    async def test_closer_vectors_rank_higher(self, vector_store: QdrantVectorStore) -> None:
        access = _access(TENANT_A)
        near = _point(tenant_id=TENANT_A, acl=access.principal_tokens, lead=1.0)
        far = _point(tenant_id=TENANT_A, acl=access.principal_tokens, lead=0.2)
        await vector_store.upsert([far, near])

        hits = await vector_store.search(_embedding(lead=1.0), access, limit=10)

        assert [hit.chunk_id for hit in hits] == [near.chunk_id, far.chunk_id]
        assert hits[0].score > hits[1].score

    async def test_limit_is_respected(self, vector_store: QdrantVectorStore) -> None:
        access = _access(TENANT_A)
        await vector_store.upsert(
            [_point(tenant_id=TENANT_A, acl=access.principal_tokens) for _ in range(5)]
        )

        assert len(await vector_store.search(_embedding(), access, limit=2)) == 2

    async def test_the_document_id_comes_back_on_every_hit(
        self, vector_store: QdrantVectorStore
    ) -> None:
        # Search results are grouped and hydrated by document, so a hit without
        # one is useless — and the payload is the only place it can come from.
        access = _access(TENANT_A)
        document_id = uuid4()
        await vector_store.upsert(
            [_point(tenant_id=TENANT_A, acl=access.principal_tokens, document_id=document_id)]
        )

        hits = await vector_store.search(_embedding(), access, limit=10)

        assert [hit.document_id for hit in hits] == [document_id]


class TestSparseVectorsAreStored:
    async def test_a_point_with_an_empty_sparse_vector_is_still_indexed(
        self, vector_store: QdrantVectorStore
    ) -> None:
        # A chunk of pure punctuation legitimately produces no weighted terms,
        # and Qdrant rejects a sparse vector with no entries. The adapter omits
        # it rather than sending an empty one, so the chunk stays retrievable by
        # its dense vector.
        access = _access(TENANT_A)
        chunk_id = uuid4()
        await vector_store.upsert(
            [
                VectorPoint(
                    chunk_id=chunk_id,
                    tenant_id=TENANT_A,
                    document_id=uuid4(),
                    ordinal=0,
                    embedding=Embedding(
                        dense=_embedding().dense,
                        sparse=SparseVector(indices=(), values=()),
                    ),
                    acl_principals=access.principal_tokens,
                )
            ]
        )

        hits = await vector_store.search(_embedding(), access, limit=10)

        assert [hit.chunk_id for hit in hits] == [chunk_id]


class TestReadiness:
    async def test_ping_succeeds_when_the_collection_exists(
        self, vector_store: QdrantVectorStore
    ) -> None:
        await vector_store.ping()

    async def test_ensure_ready_is_idempotent(self, vector_store: QdrantVectorStore) -> None:
        # Called on every boot, so the second call must not fail on an existing
        # collection or a payload index that is already there.
        await vector_store.ensure_ready()
        await vector_store.ensure_ready()

        await vector_store.ping()
