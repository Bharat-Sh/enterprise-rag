"""Cross-tenant isolation *inside the vector store*, asserted at the store.

Why this suite exists separately from `test_search_isolation.py`
---------------------------------------------------------------
Because the HTTP-level suite **cannot prove this property**, and that was
verified rather than assumed: deleting the tenant clause from
`build_search_filter` leaves every test in that file green.

The reason is defence in depth doing its job. A search hydrates its hits from
Postgres, and that read runs under row-level security bound to the caller's
tenant — so another tenant's chunks are dropped there even if the vector query
returned them. From outside, a vector store with no isolation at all is
indistinguishable from one with perfect isolation.

That does not make the tenant clause optional. It is what bounds *what gets
read at all*:

- **Embeddings are derived from the text.** Returning another tenant's vectors
  discloses them to the process, and vectors are not opaque — inversion
  attacks reconstruct approximate source text from embeddings. ADR-0006's
  objection to post-filtering is exactly this: by then the data has been read.
- **It silently destroys recall.** Foreign vectors consume slots in the top-k,
  so a caller asking for ten results gets however many of their own survive.
  That is a correctness bug with no error attached to it.
- **It is the only control if hydration ever changes.** A cache, a payload
  optimisation, or any future path that trusts the index would remove the
  backstop that currently hides the problem.

So the assertion has to be made where it is observable: against the store,
below HTTP and below Postgres.
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

pytestmark = [pytest.mark.integration, pytest.mark.security]

TENANT_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
TENANT_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def _embedding() -> Embedding:
    dense = [0.0] * STUB_DIMENSIONS
    dense[0] = 1.0
    return Embedding(dense=tuple(dense), sparse=SparseVector(indices=(1, 42), values=(0.5, 0.9)))


def _point(
    *, tenant_id: UUID, acl: tuple[str, ...], document_id: UUID | None = None
) -> VectorPoint:
    return VectorPoint(
        chunk_id=uuid4(),
        tenant_id=tenant_id,
        document_id=document_id or uuid4(),
        ordinal=0,
        embedding=_embedding(),
        acl_principals=acl,
    )


def _access(tenant_id: UUID, *, user_id: UUID | None = None, role: Role = Role.MEMBER):
    return AccessFilter.build(tenant_id=tenant_id, user_id=user_id or uuid4(), role=role)


class TestTheTenantClauseIsLoadBearing:
    """Each of these fails if `build_search_filter` stops emitting the clause."""

    async def test_identical_vectors_in_two_tenants_do_not_cross(
        self, vector_store: QdrantVectorStore
    ) -> None:
        # The ACL is deliberately the *same* on both points, so the ACL clause
        # matches each of them and cannot be what separates the two.
        access = _access(TENANT_A)
        mine = _point(tenant_id=TENANT_A, acl=access.principal_tokens)
        theirs = _point(tenant_id=TENANT_B, acl=access.principal_tokens)
        await vector_store.upsert([mine, theirs])

        hits = await vector_store.search(_embedding(), access, limit=10)

        assert [hit.chunk_id for hit in hits] == [mine.chunk_id]

    async def test_a_role_principal_is_the_same_string_in_every_tenant(
        self, vector_store: QdrantVectorStore
    ) -> None:
        # Not a contrived ACL: `role:owner` is a grant the API accepts, and
        # unlike users and groups — whose ids are UUIDs — a role token collides
        # across tenants by construction. A document shared with all owners in
        # one tenant carries a principal every other tenant's owners hold too.
        access = _access(TENANT_A, role=Role.OWNER)
        theirs = _point(tenant_id=TENANT_B, acl=("role:owner",))
        await vector_store.upsert([theirs])

        assert "role:owner" in access.principal_tokens
        hits = await vector_store.search(_embedding(), access, limit=10)

        assert hits == [], "role:owner granted access across a tenant boundary"

    async def test_the_same_user_id_in_two_tenants_does_not_cross(
        self, vector_store: QdrantVectorStore
    ) -> None:
        # A user id is unique in practice, but nothing in the *filter* knows
        # that. This is the shape a `should` instead of a `must` would produce.
        user_id = uuid4()
        access = _access(TENANT_A, user_id=user_id)
        other = _access(TENANT_B, user_id=user_id)
        theirs = _point(tenant_id=TENANT_B, acl=other.principal_tokens)
        await vector_store.upsert([theirs])

        hits = await vector_store.search(_embedding(), access, limit=10)

        assert hits == []

    async def test_foreign_vectors_do_not_consume_result_slots(
        self, vector_store: QdrantVectorStore
    ) -> None:
        # The recall half of the argument. Without the tenant clause the other
        # tenant's points fill the top-k and the caller's own results are pushed
        # out — a correctness failure with no error attached to it, and one that
        # hydration cannot repair because the rows were never fetched.
        access = _access(TENANT_A)
        mine = [_point(tenant_id=TENANT_A, acl=access.principal_tokens) for _ in range(2)]
        theirs = [_point(tenant_id=TENANT_B, acl=access.principal_tokens) for _ in range(20)]
        await vector_store.upsert([*theirs, *mine])

        hits = await vector_store.search(_embedding(), access, limit=5)

        assert {hit.chunk_id for hit in hits} == {point.chunk_id for point in mine}


class TestTheAclClauseIsAlsoLoadBearing:
    async def test_a_point_the_caller_does_not_match_is_never_returned(
        self, vector_store: QdrantVectorStore
    ) -> None:
        # Within one tenant, row-level security permits everything, so the ACL
        # clause is the only control — and here there is no second check that
        # could mask its absence at this level either.
        access = _access(TENANT_A)
        await vector_store.upsert([_point(tenant_id=TENANT_A, acl=("group:finance",))])

        hits = await vector_store.search(_embedding(), access, limit=10)

        assert hits == []

    async def test_an_unprojected_acl_fails_closed(self, vector_store: QdrantVectorStore) -> None:
        # An empty ACL array is what a point carries if its permissions were
        # never projected. "No principals granted" must mean nobody, not
        # everybody.
        await vector_store.upsert([_point(tenant_id=TENANT_A, acl=())])

        hits = await vector_store.search(_embedding(), _access(TENANT_A), limit=10)

        assert hits == []


class TestDeletionIsScoped:
    async def test_deleting_a_document_cannot_reach_another_tenant(
        self, vector_store: QdrantVectorStore
    ) -> None:
        # Deletion is one copy-paste away from search. A missing tenant clause
        # here destroys another tenant's data rather than disclosing it, which
        # is a different failure and not a smaller one.
        shared_document = uuid4()
        await vector_store.upsert(
            [
                _point(tenant_id=TENANT_A, acl=("tenant:a",), document_id=shared_document),
                _point(tenant_id=TENANT_B, acl=("tenant:b",), document_id=shared_document),
            ]
        )

        await vector_store.delete_for_document(TENANT_A, shared_document)

        assert await vector_store.count_for_tenant(TENANT_B) == 1

    async def test_rewriting_an_acl_cannot_reach_another_tenant(
        self, vector_store: QdrantVectorStore
    ) -> None:
        shared_document = uuid4()
        await vector_store.upsert(
            [_point(tenant_id=TENANT_B, acl=("group:finance",), document_id=shared_document)]
        )

        await vector_store.set_acl(TENANT_A, shared_document, ["tenant:a", "role:owner"])

        hits = await vector_store.search(_embedding(), _access(TENANT_B, role=Role.OWNER), limit=10)
        assert hits == [], "an ACL write from tenant A altered tenant B's points"
