"""Documents and chunks must not cross a tenant, or an ACL, boundary.

Two distinct controls are under test and they fail differently:

* **Row-level security** bounds the *tenant*. Its failure is a breach.
* **The ACL pre-filter** bounds visibility *within* a tenant. Its failure is a
  colleague reading an HR file.

The chunk tests matter as much as the document ones. Chunks carry a denormalised
copy of the ACL (docs/adr/0006) and hold the actual text; a document that is
correctly hidden while its chunks are readable is a leak with extra steps — and
from M5 the chunk array is what retrieval filters on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

from rag.domain.access import Principal
from rag.domain.enums import Role, UserStatus
from tests.integration.conftest import TEST_PASSWORD, bearer, login, requires_postgres

if TYPE_CHECKING:
    from httpx import AsyncClient

    from rag.adapters.auth.passwords import Argon2PasswordHasher
    from rag.db.uow import SqlAlchemyUnitOfWork
    from rag.domain.models import Tenant, User
    from rag.worker.runner import Worker

pytestmark = [pytest.mark.security, pytest.mark.integration, requires_postgres]

SECRET = "Confidential: the acquisition closes in March.\n\n" * 5


@pytest.fixture
async def rival_owner(
    uow: SqlAlchemyUnitOfWork,
    tenant: Tenant,
    other_tenant: Tenant,
    hasher: Argon2PasswordHasher,
) -> User:
    await uow.scope_to_tenant(other_tenant.id)
    created = await uow.users.create(
        tenant_id=other_tenant.id,
        email="boss@globex.example",
        full_name="Rival Boss",
        role=Role.OWNER,
        status=UserStatus.ACTIVE,
        password_hash=await hasher.hash(TEST_PASSWORD),
    )
    await uow.commit()
    await uow.scope_to_tenant(tenant.id)
    return created


async def _ingest(
    client: AsyncClient, token: str, *, slug: str, content: str = SECRET
) -> dict[str, Any]:
    collection = await client.post(
        "/api/v1/collections", headers=bearer(token), json={"slug": slug, "name": slug}
    )
    assert collection.status_code == 201, collection.text
    uploaded = await client.post(
        "/api/v1/documents",
        headers=bearer(token),
        files={"file": ("secret.txt", content.encode(), "text/plain")},
        data={"collection_id": str(collection.json()["id"])},
    )
    assert uploaded.status_code == 202, uploaded.text
    return dict(uploaded.json())


class TestCrossTenant:
    async def test_a_tenant_cannot_read_another_tenants_document(
        self,
        api_client: AsyncClient,
        tenant: Tenant,
        other_tenant: Tenant,
        owner: User,
        rival_owner: User,
    ) -> None:
        theirs = await login(api_client, tenant_slug=other_tenant.slug, email=rival_owner.email)
        document = await _ingest(api_client, theirs["access_token"], slug="theirs")

        ours = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        response = await api_client.get(
            f"/api/v1/documents/{document['id']}", headers=bearer(ours["access_token"])
        )

        # 404, not 403: a 403 confirms the document exists.
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    async def test_a_cross_tenant_document_is_indistinguishable_from_a_missing_one(
        self,
        api_client: AsyncClient,
        tenant: Tenant,
        other_tenant: Tenant,
        owner: User,
        rival_owner: User,
    ) -> None:
        theirs = await login(api_client, tenant_slug=other_tenant.slug, email=rival_owner.email)
        document = await _ingest(api_client, theirs["access_token"], slug="theirs")
        ours = await login(api_client, tenant_slug=tenant.slug, email=owner.email)

        real = await api_client.get(
            f"/api/v1/documents/{document['id']}", headers=bearer(ours["access_token"])
        )
        invented = await api_client.get(
            f"/api/v1/documents/{uuid4()}", headers=bearer(ours["access_token"])
        )

        assert real.status_code == invented.status_code
        assert real.json()["code"] == invented.json()["code"]

    async def test_listing_never_includes_another_tenants_documents(
        self,
        api_client: AsyncClient,
        tenant: Tenant,
        other_tenant: Tenant,
        owner: User,
        rival_owner: User,
    ) -> None:
        theirs = await login(api_client, tenant_slug=other_tenant.slug, email=rival_owner.email)
        await _ingest(api_client, theirs["access_token"], slug="theirs")
        ours = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        await _ingest(api_client, ours["access_token"], slug="ours", content="Our own notes.\n")

        listing = await api_client.get("/api/v1/documents", headers=bearer(ours["access_token"]))

        titles = {document["title"] for document in listing.json()}
        assert titles == {"secret.txt"}
        assert len(listing.json()) == 1

    async def test_chunk_text_does_not_cross_tenants(
        self,
        api_client: AsyncClient,
        worker: Worker,
        tenant: Tenant,
        other_tenant: Tenant,
        owner: User,
        rival_owner: User,
    ) -> None:
        """Chunks hold the text. Hiding the document and not the chunks is a leak."""
        theirs = await login(api_client, tenant_slug=other_tenant.slug, email=rival_owner.email)
        document = await _ingest(api_client, theirs["access_token"], slug="theirs")
        await worker.run_once()

        ours = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        response = await api_client.get(
            f"/api/v1/documents/{document['id']}/chunks", headers=bearer(ours["access_token"])
        )

        assert response.status_code == 404
        assert "acquisition" not in response.text

    async def test_deleting_another_tenants_document_is_refused_and_does_nothing(
        self,
        api_client: AsyncClient,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        other_tenant: Tenant,
        owner: User,
        rival_owner: User,
    ) -> None:
        from uuid import UUID

        from rag.domain.enums import DocumentStatus

        theirs = await login(api_client, tenant_slug=other_tenant.slug, email=rival_owner.email)
        document = await _ingest(api_client, theirs["access_token"], slug="theirs")
        ours = await login(api_client, tenant_slug=tenant.slug, email=owner.email)

        response = await api_client.delete(
            f"/api/v1/documents/{document['id']}", headers=bearer(ours["access_token"])
        )

        assert response.status_code == 404
        # And the refusal did not half-apply: a 404 that still moved the
        # document to DELETING would be the worst of both.
        await uow.scope_to_tenant(other_tenant.id)
        survivor = await uow.documents.get_for_processing(UUID(document["id"]))
        assert survivor is not None
        assert survivor.status is not DocumentStatus.DELETING


class TestAclWithinATenant:
    async def test_narrowing_the_acl_hides_the_document_from_a_colleague(
        self,
        api_client: AsyncClient,
        tenant: Tenant,
        owner: User,
        member: User,
    ) -> None:
        """The ACL bounds visibility inside a tenant.

        Tenant isolation is untouched here — both users are in the same tenant,
        and it is `acl_principals && caller_principals` doing the work.
        """
        owner_session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        document = await _ingest(api_client, owner_session["access_token"], slug="hr")

        member_session = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        before = await api_client.get(
            f"/api/v1/documents/{document['id']}", headers=bearer(member_session["access_token"])
        )
        assert before.status_code == 200, "default ACL should be tenant-wide"

        narrowed = await api_client.put(
            f"/api/v1/documents/{document['id']}/acl",
            headers=bearer(owner_session["access_token"]),
            json={"principals": [Principal.user(owner.id).token]},
        )
        assert narrowed.status_code == 200

        after = await api_client.get(
            f"/api/v1/documents/{document['id']}", headers=bearer(member_session["access_token"])
        )
        assert after.status_code == 404

    async def test_narrowing_the_acl_hides_the_chunks_too(
        self,
        api_client: AsyncClient,
        worker: Worker,
        tenant: Tenant,
        owner: User,
        member: User,
    ) -> None:
        """ADR-0006's reprojection, tested where it matters.

        The ACL lives in three places — grants, the document array, and a copy
        on every chunk. If `set_acl` failed to rewrite the chunk copies, the
        document would vanish while its text stayed readable.
        """
        owner_session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        document = await _ingest(api_client, owner_session["access_token"], slug="hr")
        await worker.run_once()

        member_session = await login(api_client, tenant_slug=tenant.slug, email=member.email)
        assert (
            await api_client.get(
                f"/api/v1/documents/{document['id']}/chunks",
                headers=bearer(member_session["access_token"]),
            )
        ).status_code == 200

        await api_client.put(
            f"/api/v1/documents/{document['id']}/acl",
            headers=bearer(owner_session["access_token"]),
            json={"principals": [Principal.user(owner.id).token]},
        )

        response = await api_client.get(
            f"/api/v1/documents/{document['id']}/chunks",
            headers=bearer(member_session["access_token"]),
        )
        assert response.status_code == 404
        assert "acquisition" not in response.text

    async def test_the_chunk_acl_array_is_actually_rewritten(
        self,
        api_client: AsyncClient,
        uow: SqlAlchemyUnitOfWork,
        worker: Worker,
        tenant: Tenant,
        owner: User,
    ) -> None:
        # Asserted on the stored array, not just on the endpoint: from M5 this
        # array is what the vector store filters on, and an endpoint check would
        # not notice it going stale.
        from uuid import UUID

        session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        document = await _ingest(api_client, session["access_token"], slug="hr")
        await worker.run_once()

        await api_client.put(
            f"/api/v1/documents/{document['id']}/acl",
            headers=bearer(session["access_token"]),
            json={"principals": [Principal.user(owner.id).token]},
        )

        await uow.rollback()
        chunks = await uow.chunks.list_for_document(UUID(document["id"]))
        assert chunks
        for chunk in chunks:
            assert set(chunk.acl_principals) == {Principal.user(owner.id).token}

    async def test_a_document_cannot_be_shared_with_another_tenant(
        self,
        api_client: AsyncClient,
        tenant: Tenant,
        other_tenant: Tenant,
        owner: User,
    ) -> None:
        """The ACL array is matched with `&&` against a caller's principal set.

        An unchecked `tenant:` token naming somebody else would therefore be a
        cross-tenant grant, writable by a tenant admin, that row-level security
        would not catch — the chunk array is consulted *after* the tenant scope.
        """
        session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        document = await _ingest(api_client, session["access_token"], slug="hr")

        response = await api_client.put(
            f"/api/v1/documents/{document['id']}/acl",
            headers=bearer(session["access_token"]),
            json={"principals": [Principal.tenant(other_tenant.id).token]},
        )

        assert response.status_code == 400
        assert response.json()["code"] == "invalid_input"

    async def test_a_malformed_principal_is_refused(
        self, api_client: AsyncClient, tenant: Tenant, owner: User
    ) -> None:
        session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        document = await _ingest(api_client, session["access_token"], slug="hr")

        response = await api_client.put(
            f"/api/v1/documents/{document['id']}/acl",
            headers=bearer(session["access_token"]),
            json={"principals": ["not-a-principal"]},
        )

        assert response.status_code == 400

    async def test_a_member_cannot_change_an_acl(
        self, api_client: AsyncClient, tenant: Tenant, owner: User, member: User
    ) -> None:
        # Granting access is a different power from uploading.
        owner_session = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        document = await _ingest(api_client, owner_session["access_token"], slug="hr")
        member_session = await login(api_client, tenant_slug=tenant.slug, email=member.email)

        response = await api_client.put(
            f"/api/v1/documents/{document['id']}/acl",
            headers=bearer(member_session["access_token"]),
            json={"principals": [Principal.user(member.id).token]},
        )

        assert response.status_code == 403


class TestBlobIsolation:
    async def test_two_tenants_uploading_identical_bytes_get_separate_blobs(
        self,
        api_client: AsyncClient,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        other_tenant: Tenant,
        owner: User,
        rival_owner: User,
    ) -> None:
        """Deliberate duplication (docs/adr/0009).

        Sharing storage across tenants would make one tenant's deletion destroy
        another's document, and make the store a cross-tenant existence oracle.
        """
        from uuid import UUID

        ours = await login(api_client, tenant_slug=tenant.slug, email=owner.email)
        theirs = await login(api_client, tenant_slug=other_tenant.slug, email=rival_owner.email)

        mine = await _ingest(api_client, ours["access_token"], slug="ours")
        yours = await _ingest(api_client, theirs["access_token"], slug="theirs")

        assert mine["content_hash"] == yours["content_hash"]
        assert mine["id"] != yours["id"]

        await uow.scope_to_tenant(tenant.id)
        first = await uow.documents.get_for_processing(UUID(mine["id"]))
        await uow.scope_to_tenant(other_tenant.id)
        second = await uow.documents.get_for_processing(UUID(yours["id"]))

        assert first is not None
        assert second is not None
        assert first.blob_key != second.blob_key
