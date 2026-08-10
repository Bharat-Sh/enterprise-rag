"""Document lifecycle, idempotency, chunking, and ACL enforcement."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy.exc import IntegrityError

from rag.domain.access import Principal
from rag.domain.enums import DocumentStatus, UserStatus
from rag.domain.errors import (
    AlreadyExistsError,
    ConcurrentModificationError,
    InvalidStateTransitionError,
    NotFoundError,
)
from rag.domain.models import NewChunk
from tests.integration.conftest import requires_postgres

if TYPE_CHECKING:
    from rag.db.uow import SqlAlchemyUnitOfWork
    from rag.domain.models import Collection, Document, Tenant, User

pytestmark = [requires_postgres, pytest.mark.integration]


async def _make_document(
    uow: SqlAlchemyUnitOfWork,
    tenant: Tenant,
    collection: Collection,
    *,
    content_hash: str = "a" * 64,
    acl: tuple[str, ...] = (),
) -> Document:
    return await uow.documents.create(
        tenant_id=tenant.id,
        collection_id=collection.id,
        title="Employee Handbook",
        source_uri="s3://bucket/handbook.pdf",
        blob_key="test-blob-key",
        content_hash=content_hash,
        mime_type="application/pdf",
        size_bytes=1024,
        acl_principals=acl,
    )


class TestIdempotency:
    async def test_identical_content_is_rejected_within_a_tenant(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant, collection: Collection
    ) -> None:
        """Re-uploading the same bytes must not create a second document.

        Without this, one user double-clicking Upload pays the embedding cost
        twice and pollutes retrieval with duplicate chunks that then compete
        with each other for the same top-k slots.
        """
        await _make_document(uow, tenant, collection)
        await uow.commit()

        # A domain error, not the driver's `IntegrityError`. M3 moved the
        # translation into the repository so the ingestion service can turn a
        # lost race into the same idempotent answer the content-hash probe would
        # have given, instead of a 500.
        with pytest.raises(AlreadyExistsError):
            await _make_document(uow, tenant, collection)

    async def test_a_conflict_does_not_destroy_the_callers_transaction(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant, collection: Collection
    ) -> None:
        """The reason the insert runs inside a SAVEPOINT.

        Postgres aborts the whole transaction on a constraint violation, so
        without one the only way to report the conflict would be to destroy
        whatever else the caller had already done.
        """
        await _make_document(uow, tenant, collection)
        await uow.commit()

        with pytest.raises(AlreadyExistsError):
            await _make_document(uow, tenant, collection)

        # The session is still usable, and the first document is still there.
        surviving = await uow.documents.get_by_content_hash("a" * 64)
        assert surviving is not None

    async def test_the_same_content_may_exist_in_two_tenants(
        self,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        other_tenant: Tenant,
        collection: Collection,
    ) -> None:
        # The uniqueness constraint is (tenant_id, content_hash). Two customers
        # uploading the same public PDF is entirely normal and must not collide.
        await _make_document(uow, tenant, collection)
        await uow.commit()

        await uow.scope_to_tenant(other_tenant.id)
        other_collection = await uow.collections.create(
            tenant_id=other_tenant.id, slug="handbook", name="Handbook"
        )
        await uow.documents.create(
            tenant_id=other_tenant.id,
            collection_id=other_collection.id,
            title="Same File",
            source_uri="s3://bucket/handbook.pdf",
            blob_key="test-blob-key",
            content_hash="a" * 64,
            mime_type="application/pdf",
            size_bytes=1024,
        )
        await uow.commit()

    async def test_lookup_by_hash_finds_the_existing_document(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant, collection: Collection
    ) -> None:
        created = await _make_document(uow, tenant, collection)
        await uow.commit()

        found = await uow.documents.get_by_content_hash("a" * 64)

        assert found is not None
        assert found.id == created.id


class TestStatusTransitions:
    async def test_the_pipeline_advances(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant, collection: Collection
    ) -> None:
        document = await _make_document(uow, tenant, collection)
        await uow.commit()

        current = DocumentStatus.UPLOADED
        for target in (
            DocumentStatus.QUEUED,
            DocumentStatus.PARSING,
            DocumentStatus.CHUNKING,
            DocumentStatus.EMBEDDING,
            DocumentStatus.INDEXING,
            DocumentStatus.READY,
        ):
            updated = await uow.documents.transition_status(
                document.id, expected=current, target=target
            )
            assert updated.status is target
            current = target

        assert updated.indexed_at is not None
        assert updated.is_queryable

    async def test_an_illegal_transition_is_refused_before_touching_the_database(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant, collection: Collection
    ) -> None:
        document = await _make_document(uow, tenant, collection)
        await uow.commit()

        with pytest.raises(InvalidStateTransitionError):
            await uow.documents.transition_status(
                document.id,
                expected=DocumentStatus.UPLOADED,
                target=DocumentStatus.READY,
            )

    async def test_losing_a_race_raises_rather_than_clobbering(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant, collection: Collection
    ) -> None:
        """Compare-and-set, not read-then-write.

        Two workers picking up the same document must not both "succeed". The
        second sees the row is no longer in the state it expected and fails,
        instead of overwriting the winner's progress and losing work.
        """
        document = await _make_document(uow, tenant, collection)
        await uow.commit()

        await uow.documents.transition_status(
            document.id, expected=DocumentStatus.UPLOADED, target=DocumentStatus.QUEUED
        )
        await uow.commit()

        # A second worker still believing the document is UPLOADED.
        with pytest.raises(ConcurrentModificationError) as caught:
            await uow.documents.transition_status(
                document.id,
                expected=DocumentStatus.UPLOADED,
                target=DocumentStatus.QUEUED,
            )

        # Distinct from an illegal transition: this move *would* have been legal
        # from UPLOADED, it just arrived second. A client should re-read and
        # retry, which is why it needs its own error code rather than sharing
        # one with "you asked for something impossible".
        assert caught.value.code == "concurrent_modification"
        assert caught.value.expected == "uploaded"
        assert caught.value.actual == "queued"

    async def test_a_missing_document_is_not_found(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant
    ) -> None:
        from rag.core.ids import uuid7

        with pytest.raises(NotFoundError):
            await uow.documents.transition_status(
                uuid7(), expected=DocumentStatus.UPLOADED, target=DocumentStatus.QUEUED
            )

    async def test_a_failure_reason_is_recorded(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant, collection: Collection
    ) -> None:
        document = await _make_document(uow, tenant, collection)
        await uow.commit()

        failed = await uow.documents.transition_status(
            document.id,
            expected=DocumentStatus.UPLOADED,
            target=DocumentStatus.FAILED,
            reason="unsupported PDF encryption",
        )

        assert failed.status_reason == "unsupported PDF encryption"


class TestAccessControl:
    async def test_a_document_you_were_granted_is_visible(
        self,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        collection: Collection,
        user: User,
    ) -> None:
        document = await _make_document(
            uow, tenant, collection, acl=(Principal.user(user.id).token,)
        )
        await uow.commit()

        access = await uow.users.access_filter_for(user)

        assert await uow.documents.get(document.id, access) is not None

    async def test_a_document_you_were_not_granted_is_invisible(
        self,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        collection: Collection,
        user: User,
    ) -> None:
        # Same tenant, so RLS permits the row. Only the ACL stops it — this is
        # the second, independent layer.
        other = await uow.users.create(
            tenant_id=tenant.id, email="bob@acme.example", status=UserStatus.ACTIVE
        )
        document = await _make_document(
            uow, tenant, collection, acl=(Principal.user(other.id).token,)
        )
        await uow.commit()

        access = await uow.users.access_filter_for(user)

        assert await uow.documents.get(document.id, access) is None

    async def test_group_membership_grants_access_without_reindexing(
        self,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        collection: Collection,
        user: User,
    ) -> None:
        """The payoff of the chosen ACL shape.

        The document's ACL never changes. Adding the user to the group changes
        only the *caller's* principal set, computed per request — so a
        membership change costs nothing, where storing expanded user ids on
        documents would trigger a re-index of everything the group can see.
        """
        group = await uow.groups.create(tenant_id=tenant.id, slug="eng", name="Engineering")
        document = await _make_document(
            uow, tenant, collection, acl=(Principal.group(group.id).token,)
        )
        await uow.commit()

        before = await uow.users.access_filter_for(user)
        assert await uow.documents.get(document.id, before) is None

        await uow.groups.add_member(group_id=group.id, user_id=user.id, tenant_id=tenant.id)
        await uow.commit()

        after = await uow.users.access_filter_for(user)
        assert await uow.documents.get(document.id, after) is not None

    async def test_removing_membership_revokes_access(
        self,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        collection: Collection,
        user: User,
    ) -> None:
        group = await uow.groups.create(tenant_id=tenant.id, slug="eng", name="Engineering")
        document = await _make_document(
            uow, tenant, collection, acl=(Principal.group(group.id).token,)
        )
        await uow.groups.add_member(group_id=group.id, user_id=user.id, tenant_id=tenant.id)
        await uow.commit()

        assert (
            await uow.documents.get(document.id, await uow.users.access_filter_for(user))
            is not None
        )

        await uow.groups.remove_member(group_id=group.id, user_id=user.id)
        await uow.commit()

        assert await uow.documents.get(document.id, await uow.users.access_filter_for(user)) is None

    async def test_a_tenant_wide_grant_reaches_everyone(
        self,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        collection: Collection,
        user: User,
    ) -> None:
        document = await _make_document(
            uow, tenant, collection, acl=(Principal.tenant(tenant.id).token,)
        )
        await uow.commit()

        access = await uow.users.access_filter_for(user)

        assert await uow.documents.get(document.id, access) is not None

    async def test_listing_only_returns_permitted_documents(
        self,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        collection: Collection,
        user: User,
    ) -> None:
        await _make_document(
            uow, tenant, collection, content_hash="a" * 64, acl=(Principal.user(user.id).token,)
        )
        await _make_document(
            uow, tenant, collection, content_hash="b" * 64, acl=("user:someone-else",)
        )
        await uow.commit()

        visible = await uow.documents.list_for(await uow.users.access_filter_for(user))

        assert len(visible) == 1

    async def test_setting_an_acl_reprojects_onto_chunks(
        self,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        collection: Collection,
        user: User,
    ) -> None:
        """Chunks must not outlive their document's permissions.

        The array on `chunks` is a denormalisation of the one on `documents`.
        Letting them diverge is exactly how a revoked permission keeps returning
        search results — the document disappears from listings while its text
        keeps surfacing in answers.
        """
        document = await _make_document(uow, tenant, collection, acl=("user:old-owner",))
        await uow.chunks.add_many(
            document.id,
            [NewChunk(ordinal=0, text="first"), NewChunk(ordinal=1, text="second")],
        )
        await uow.commit()

        await uow.documents.set_acl(document.id, [Principal.user(user.id).token])
        await uow.commit()

        chunks = await uow.chunks.list_for_document(document.id)
        assert all(chunk.acl_principals == (Principal.user(user.id).token,) for chunk in chunks)

        access = await uow.users.access_filter_for(user)
        hydrated = await uow.chunks.get_many([chunk.id for chunk in chunks], access)
        assert len(hydrated) == 2

    async def test_chunk_hydration_re_checks_access(
        self,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        collection: Collection,
        user: User,
    ) -> None:
        # Defence in depth: even holding valid chunk ids (as one would after a
        # vector search against a stale index), the text must not come back.
        document = await _make_document(uow, tenant, collection, acl=("user:someone-else",))
        await uow.chunks.add_many(document.id, [NewChunk(ordinal=0, text="secret")])
        await uow.commit()

        chunks = await uow.chunks.list_for_document(document.id)
        access = await uow.users.access_filter_for(user)

        assert await uow.chunks.get_many([chunk.id for chunk in chunks], access) == []


class TestChunks:
    async def test_chunks_inherit_tenant_and_acl_from_their_document(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant, collection: Collection
    ) -> None:
        document = await _make_document(uow, tenant, collection, acl=("tenant:x",))
        await uow.chunks.add_many(document.id, [NewChunk(ordinal=0, text="hello")])
        await uow.commit()

        (chunk,) = await uow.chunks.list_for_document(document.id)

        assert chunk.tenant_id == tenant.id
        assert chunk.acl_principals == ("tenant:x",)

    async def test_ordinals_are_unique_per_document(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant, collection: Collection
    ) -> None:
        document = await _make_document(uow, tenant, collection)
        await uow.chunks.add_many(document.id, [NewChunk(ordinal=0, text="a")])
        await uow.commit()

        # Still the raw driver error here, deliberately. Only the *document*
        # insert translates, because only there is a conflict a legitimate user
        # action (re-uploading the same file). A duplicate chunk ordinal is a
        # bug in the chunker, and a bug should not be dressed up as a domain
        # error a caller might try to handle.
        with pytest.raises(IntegrityError):
            await uow.chunks.add_many(document.id, [NewChunk(ordinal=0, text="duplicate")])

        await uow.rollback()

    async def test_hydration_preserves_the_caller_ordering(
        self,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        collection: Collection,
        user: User,
    ) -> None:
        """The id order is the relevance ranking, and must survive the round trip.

        Returning rows in primary-key order would silently discard the ranking
        the reranker just spent 60ms computing.
        """
        document = await _make_document(
            uow, tenant, collection, acl=(Principal.user(user.id).token,)
        )
        await uow.chunks.add_many(
            document.id, [NewChunk(ordinal=index, text=f"chunk {index}") for index in range(5)]
        )
        await uow.commit()

        chunks = await uow.chunks.list_for_document(document.id)
        shuffled = [chunks[3].id, chunks[0].id, chunks[4].id]

        access = await uow.users.access_filter_for(user)
        hydrated = await uow.chunks.get_many(shuffled, access)

        assert [chunk.id for chunk in hydrated] == shuffled

    async def test_embedding_metadata_is_stamped_per_chunk(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant, collection: Collection
    ) -> None:
        # Recorded from the first migration so a model change becomes a
        # backfill rather than a drop-and-rebuild of the whole index.
        document = await _make_document(uow, tenant, collection)
        await uow.chunks.add_many(
            document.id, [NewChunk(ordinal=index, text="x") for index in range(3)]
        )
        await uow.commit()

        updated = await uow.chunks.set_embedding_metadata(
            document.id, model="BAAI/bge-m3", version="1"
        )
        await uow.commit()

        assert updated == 3
        chunks = await uow.chunks.list_for_document(document.id)
        assert all(chunk.embedding_model == "BAAI/bge-m3" for chunk in chunks)

    async def test_deleting_a_document_removes_its_chunks(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant, collection: Collection
    ) -> None:
        document = await _make_document(uow, tenant, collection)
        await uow.chunks.add_many(
            document.id, [NewChunk(ordinal=index, text="x") for index in range(4)]
        )
        await uow.commit()

        assert await uow.chunks.delete_for_document(document.id) == 4


class TestUnitOfWork:
    async def test_uncommitted_work_is_discarded(
        self, session_factory, tenant: Tenant, collection: Collection
    ) -> None:
        """Leaving the block without committing must not persist anything.

        Deliberately not an implicit commit on exit: a function that raises
        *after* its last write would otherwise still persist a partial change.
        """
        from rag.db.uow import SqlAlchemyUnitOfWork as UoW

        async with UoW(session_factory) as first:
            await first.scope_to_tenant(tenant.id)
            await _make_document(first, tenant, collection, content_hash="c" * 64)
            # no commit

        async with UoW(session_factory) as second:
            await second.scope_to_tenant(tenant.id)
            assert await second.documents.get_by_content_hash("c" * 64) is None

    async def test_an_exception_rolls_everything_back(
        self, session_factory, tenant: Tenant, collection: Collection
    ) -> None:
        from rag.db.uow import SqlAlchemyUnitOfWork as UoW

        with pytest.raises(RuntimeError):
            async with UoW(session_factory) as unit:
                await unit.scope_to_tenant(tenant.id)
                await _make_document(unit, tenant, collection, content_hash="d" * 64)
                raise RuntimeError("something went wrong after the write")

        async with UoW(session_factory) as check:
            await check.scope_to_tenant(tenant.id)
            assert await check.documents.get_by_content_hash("d" * 64) is None

    async def test_using_a_unit_of_work_outside_its_block_is_an_error(
        self, session_factory
    ) -> None:
        from rag.db.uow import SqlAlchemyUnitOfWork as UoW

        unit = UoW(session_factory)

        with pytest.raises(RuntimeError, match="outside an `async with`"):
            _ = unit.session
