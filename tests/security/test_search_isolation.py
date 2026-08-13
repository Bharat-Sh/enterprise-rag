"""Search must not return anything the caller may not read.

This is the highest-stakes suite in the repository, because search is the first
feature where **there is no database backstop**. Everywhere else, a query that
forgets its tenant scope finds zero rows: row-level security is enforced by
Postgres and applies whether or not the application remembers. A vector search
that forgets its tenant clause returns every tenant's vectors, with a 200, and
nothing anywhere objects.

So these tests go through the real HTTP surface, against a real database and a
real vector store, with two tenants whose documents are deliberately
indistinguishable by content. If isolation held only because the corpora
differed, that would prove nothing.

**What this file cannot prove, and where that proof lives.** Deleting the
tenant clause from `build_search_filter` leaves every test here green. That was
checked, not assumed. Hydration reads chunks from Postgres under row-level
security, so another tenant's rows are dropped there even when the vector query
returned them — from outside, an unfiltered index is indistinguishable from a
filtered one.

That is defence in depth working exactly as intended, and it is also why the
vector store's own isolation has to be asserted below HTTP, in
`tests/security/test_vector_isolation.py`. The tests here prove the *observable*
guarantee: nothing crosses the boundary through the API. They do not prove which
of the two layers is holding it.

Per CLAUDE.md these must never be skipped or marked xfail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from rag.domain.enums import DocumentStatus, Role, UserStatus
from tests.integration.conftest import TEST_PASSWORD, bearer, login, requires_postgres

if TYPE_CHECKING:
    from httpx import AsyncClient

pytestmark = [pytest.mark.integration, pytest.mark.security, requires_postgres]

SEARCH = "/api/v1/search"

#: The *same* text in both tenants. Identical content means identical
#: embeddings, so similarity cannot be what separates the two — only the filter
#: can. A test using different text per tenant would pass even with no tenant
#: clause at all, purely because the query happened to match one corpus.
SHARED_TEXT = (
    "Quarterly revenue reached four million pounds. The board approved the "
    "acquisition of a competitor. Severance terms were agreed for the outgoing "
    "chief executive."
)


async def _index_for(
    client: AsyncClient,
    worker: Any,
    *,
    token: str,
    collection_id: str,
    title: str,
    text: str = SHARED_TEXT,
) -> str:
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


@pytest.fixture
async def rival(uow, other_tenant, hasher):
    """A user in the *other* tenant, with their own collection and documents."""
    await uow.scope_to_tenant(other_tenant.id)
    user = await uow.users.create(
        tenant_id=other_tenant.id,
        email="rival@globex.example",
        full_name="Rival User",
        role=Role.OWNER,
        status=UserStatus.ACTIVE,
        password_hash=await hasher.hash(TEST_PASSWORD),
    )
    collection = await uow.collections.create(
        tenant_id=other_tenant.id, slug="secrets", name="Secrets"
    )
    await uow.commit()
    return user, collection


class TestCrossTenantSearch:
    async def test_a_search_never_returns_another_tenants_documents(
        self, api_client, worker, tenant, owner, collection, other_tenant, rival
    ) -> None:
        rival_user, rival_collection = rival

        rival_token = await _token(api_client, other_tenant.slug, rival_user.email)
        await _index_for(
            api_client,
            worker,
            token=rival_token,
            collection_id=str(rival_collection.id),
            title="globex-board-minutes",
        )

        my_token = await _token(api_client, tenant.slug, owner.email)
        body = await _search(api_client, my_token, query="quarterly revenue acquisition")

        assert body["count"] == 0, f"leaked: {body}"

    async def test_identical_content_in_both_tenants_stays_separated(
        self, api_client, worker, tenant, owner, collection, other_tenant, rival
    ) -> None:
        # Both tenants hold byte-identical documents, so both index identical
        # vectors. Each caller must see exactly their own.
        rival_user, rival_collection = rival

        rival_token = await _token(api_client, other_tenant.slug, rival_user.email)
        theirs = await _index_for(
            api_client,
            worker,
            token=rival_token,
            collection_id=str(rival_collection.id),
            title="globex-copy",
        )

        my_token = await _token(api_client, tenant.slug, owner.email)
        mine = await _index_for(
            api_client,
            worker,
            token=my_token,
            collection_id=str(collection.id),
            title="acme-copy",
        )

        my_results = await _search(api_client, my_token, query="severance terms", top_k=50)
        their_results = await _search(api_client, rival_token, query="severance terms", top_k=50)

        assert {r["document_id"] for r in my_results["results"]} == {mine}
        assert {r["document_id"] for r in their_results["results"]} == {theirs}

    async def test_a_role_granted_document_does_not_cross_tenants(
        self, api_client, worker, tenant, owner, collection, other_tenant, rival
    ) -> None:
        """The ACL that genuinely collides across tenants.

        Everywhere else the default ACL contains `tenant:<id>`, which is itself
        a principal — so ACL matching *incidentally* enforces tenant isolation
        and nothing distinguishes the two clauses.

        `role:` tokens are the exception, and they are not hypothetical: the ACL
        endpoint accepts them, and unlike users and groups — whose ids are
        UUIDs — a role token is the *same string* in every tenant. `role:owner`
        here is `role:owner` there. So a document shared with all owners carries
        a principal the other tenant's owners also hold, and the ACL clause
        matches it.

        At this level the result is still empty because Postgres refuses to
        hydrate the row. `test_vector_isolation.py` asserts the same case
        against the store, where the tenant clause is the only thing left.
        """
        rival_user, rival_collection = rival
        rival_token = await _token(api_client, other_tenant.slug, rival_user.email)
        theirs = await _index_for(
            api_client,
            worker,
            token=rival_token,
            collection_id=str(rival_collection.id),
            title="globex-all-owners",
        )

        shared = await api_client.put(
            f"/api/v1/documents/{theirs}/acl",
            headers=bearer(rival_token),
            json={"principals": ["role:owner"]},
        )
        assert shared.status_code == 200, shared.text

        my_token = await _token(api_client, tenant.slug, owner.email)
        # Sanity: this caller really does hold the principal the document grants.
        assert owner.role.value == "owner"

        body = await _search(api_client, my_token, query="severance terms", top_k=50)

        assert body["count"] == 0, f"leaked across tenants via role:owner: {body}"

    async def test_naming_another_tenants_document_id_returns_nothing(
        self, api_client, worker, tenant, owner, collection, other_tenant, rival
    ) -> None:
        # `document_ids` is caller-supplied narrowing. It must not become a way
        # to *reach* a document — the tenant clause is ANDed, so naming a
        # foreign id narrows to the empty set rather than widening.
        rival_user, rival_collection = rival

        rival_token = await _token(api_client, other_tenant.slug, rival_user.email)
        theirs = await _index_for(
            api_client,
            worker,
            token=rival_token,
            collection_id=str(rival_collection.id),
            title="globex-target",
        )

        my_token = await _token(api_client, tenant.slug, owner.email)
        body = await _search(api_client, my_token, query="quarterly revenue", document_ids=[theirs])

        assert body["count"] == 0

    async def test_naming_another_tenants_collection_returns_nothing(
        self, api_client, worker, tenant, owner, collection, other_tenant, rival
    ) -> None:
        rival_user, rival_collection = rival

        rival_token = await _token(api_client, other_tenant.slug, rival_user.email)
        await _index_for(
            api_client,
            worker,
            token=rival_token,
            collection_id=str(rival_collection.id),
            title="globex-target",
        )

        my_token = await _token(api_client, tenant.slug, owner.email)
        body = await _search(
            api_client,
            my_token,
            query="quarterly revenue",
            collection_id=str(rival_collection.id),
        )

        assert body["count"] == 0


class TestAclWithinATenant:
    async def test_revoking_access_removes_a_document_from_search(
        self, api_client, worker, tenant, owner, member, collection
    ) -> None:
        # Same tenant, so row-level security permits both callers. Only the ACL
        # pre-filter separates them — the half of the access model RLS cannot
        # express.
        owner_token = await _token(api_client, tenant.slug, owner.email)
        document_id = await _index_for(
            api_client,
            worker,
            token=owner_token,
            collection_id=str(collection.id),
            title="was-shared",
        )
        member_token = await _token(api_client, tenant.slug, member.email)
        assert (await _search(api_client, member_token, query="severance terms"))["count"] >= 1

        # Through the real endpoint, which rewrites all four representations:
        # the grants, the document array, the chunk arrays, and the vector
        # payload. No re-embedding — an ACL change does not change a vector.
        response = await api_client.put(
            f"/api/v1/documents/{document_id}/acl",
            headers=bearer(owner_token),
            json={"principals": ["group:finance-only"]},
        )
        assert response.status_code == 200, response.text

        body = await _search(api_client, member_token, query="severance terms", top_k=50)

        assert body["count"] == 0

    async def test_granting_access_makes_a_document_findable(
        self, api_client, worker, uow, tenant, owner, member, collection
    ) -> None:
        """The direction the second access check cannot save us in.

        Revocation fails safe even with a stale index, because hydration
        re-checks Postgres and drops the row. **Granting does not**: if the
        vector payload still carries the old ACL, the pre-filter never surfaces
        the chunk in the first place and there is nothing for hydration to
        rescue. A newly-shared document would simply stay invisible.

        So this is the test that proves the vector payload is really being
        rewritten, rather than the suite passing on one-directional luck.
        """
        owner_token = await _token(api_client, tenant.slug, owner.email)
        document_id = await _index_for(
            api_client,
            worker,
            token=owner_token,
            collection_id=str(collection.id),
            title="restricted",
        )

        # The owner keeps their own principal. Dropping it would lock even them
        # out: `set_acl` reads the document through the caller's access filter
        # first, so an admin who removes themselves cannot manage it again —
        # see the note in CLAUDE.md's known limitations.
        restricted = await api_client.put(
            f"/api/v1/documents/{document_id}/acl",
            headers=bearer(owner_token),
            json={"principals": ["group:finance-only", f"user:{owner.id}"]},
        )
        assert restricted.status_code == 200, restricted.text
        member_token = await _token(api_client, tenant.slug, member.email)
        assert (await _search(api_client, member_token, query="severance terms"))["count"] == 0

        granted = await api_client.put(
            f"/api/v1/documents/{document_id}/acl",
            headers=bearer(owner_token),
            json={"principals": [f"user:{member.id}", f"user:{owner.id}"]},
        )
        assert granted.status_code == 200, granted.text

        body = await _search(api_client, member_token, query="severance terms", top_k=50)

        assert body["count"] >= 1

    async def test_a_stale_index_still_cannot_disclose(
        self, api_client, worker, uow, tenant, owner, member, collection
    ) -> None:
        """Defence in depth, with the index deliberately left behind.

        The ACL is narrowed in Postgres *only* — going around the service, so
        the vector payload keeps the old permissive ACL. That is exactly the
        drift a failed reprojection, a partial write or a restored snapshot
        would produce. The pre-filter still matches, and
        `ChunkRepository.get_many` refuses to return the row.

        This is also why the index holds no text: if the payload carried
        content, the pre-filter would be the *only* thing between stale state
        and a disclosure.
        """
        owner_token = await _token(api_client, tenant.slug, owner.email)
        document_id = await _index_for(
            api_client,
            worker,
            token=owner_token,
            collection_id=str(collection.id),
            title="drifted",
        )
        member_token = await _token(api_client, tenant.slug, member.email)
        assert (await _search(api_client, member_token, query="severance terms"))["count"] >= 1

        await uow.scope_to_tenant(tenant.id)
        await uow.documents.set_acl(_uuid(document_id), ["group:finance-only"])
        await uow.commit()

        body = await _search(api_client, member_token, query="severance terms", top_k=50)

        assert body["count"] == 0


class TestTheCredentialDecidesScope:
    async def test_a_body_cannot_choose_a_tenant(
        self, api_client, worker, tenant, owner, collection, other_tenant, rival
    ) -> None:
        # `SearchRequest` has no tenant field, and unknown fields are ignored, so
        # this is asserting that the omission is real rather than a schema that
        # quietly accepts one. Non-negotiable #5.
        rival_user, rival_collection = rival
        rival_token = await _token(api_client, other_tenant.slug, rival_user.email)
        await _index_for(
            api_client,
            worker,
            token=rival_token,
            collection_id=str(rival_collection.id),
            title="globex-target",
        )

        my_token = await _token(api_client, tenant.slug, owner.email)
        response = await api_client.post(
            SEARCH,
            headers=bearer(my_token),
            json={
                "query": "quarterly revenue",
                "tenant_id": str(other_tenant.id),
                "acl_principals": [f"tenant:{other_tenant.id}"],
            },
        )

        assert response.status_code == 200
        assert response.json()["count"] == 0

    async def test_an_expired_or_absent_credential_cannot_search(self, api_client) -> None:
        assert (await api_client.post(SEARCH, json={"query": "x"})).status_code == 401
        assert (
            await api_client.post(SEARCH, headers=bearer("not-a-token"), json={"query": "x"})
        ).status_code == 401


# --- helpers ---------------------------------------------------------------


def _uuid(value: str):
    from uuid import UUID

    return UUID(value)


async def _token(client: AsyncClient, tenant_slug: str, email: str) -> str:
    session = await login(client, tenant_slug=tenant_slug, email=email)
    return str(session["access_token"])


async def _search(client: AsyncClient, token: str, **body: Any) -> dict[str, Any]:
    response = await client.post(SEARCH, headers=bearer(token), json=body)
    assert response.status_code == 200, response.text
    return dict(response.json())
