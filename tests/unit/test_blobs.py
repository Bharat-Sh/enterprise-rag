"""Filesystem blob store: keys, atomicity, and refusing to leave its root.

The traversal tests are the ones that matter. The failure they guard against is
arbitrary file read or write, and the key is assembled from a content hash and a
tenant id — both of which are ours, right up until somebody adds a code path
that builds a key from a filename.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from rag.adapters.blobs.filesystem import FilesystemBlobStore
from rag.domain.errors import NotFoundError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

TENANT = uuid4()
HASH = "a" * 64


@pytest.fixture
def store(tmp_path: Path) -> FilesystemBlobStore:
    return FilesystemBlobStore(tmp_path / "blobs")


async def stream(*blocks: bytes) -> AsyncIterator[bytes]:
    for block in blocks:
        yield block


class TestKeys:
    def test_a_key_is_sharded_under_its_tenant(self, store: FilesystemBlobStore) -> None:
        key = store.key_for(tenant_id=TENANT, content_hash=HASH)

        assert key.startswith(f"{TENANT}/")
        assert HASH in key

    def test_the_same_bytes_give_the_same_key(self, store: FilesystemBlobStore) -> None:
        assert store.key_for(tenant_id=TENANT, content_hash=HASH) == store.key_for(
            tenant_id=TENANT, content_hash=HASH
        )

    def test_two_tenants_holding_identical_bytes_get_different_keys(
        self, store: FilesystemBlobStore
    ) -> None:
        """Deliberate duplication.

        Sharing storage across tenants would make one tenant's deletion affect
        another's document, and make the store a cross-tenant existence oracle.
        """
        assert store.key_for(tenant_id=uuid4(), content_hash=HASH) != store.key_for(
            tenant_id=uuid4(), content_hash=HASH
        )

    def test_derived_keys_are_distinct_from_the_source(self, store: FilesystemBlobStore) -> None:
        source = store.key_for(tenant_id=TENANT, content_hash=HASH)
        derived = store.derived_key_for(tenant_id=TENANT, content_hash=HASH, kind="text")

        assert derived != source


class TestRoundTrip:
    async def test_written_bytes_come_back(self, store: FilesystemBlobStore) -> None:
        key = store.key_for(tenant_id=TENANT, content_hash=HASH)

        written = await store.put(key, stream(b"hello ", b"world"))

        assert written == 11
        assert await store.read(key) == b"hello world"

    async def test_reading_streams_in_blocks(self, store: FilesystemBlobStore) -> None:
        key = store.key_for(tenant_id=TENANT, content_hash=HASH)
        await store.put(key, stream(b"x" * 100))

        collected = b"".join([block async for block in await store.open(key)])

        assert collected == b"x" * 100

    async def test_overwriting_a_key_is_safe(self, store: FilesystemBlobStore) -> None:
        # Keys are content-addressed, so this only happens on a retry — and it
        # must not leave a corrupted blob behind.
        key = store.key_for(tenant_id=TENANT, content_hash=HASH)
        await store.put(key, stream(b"first"))

        await store.put(key, stream(b"second"))

        assert await store.read(key) == b"second"

    async def test_an_empty_blob_round_trips(self, store: FilesystemBlobStore) -> None:
        key = store.key_for(tenant_id=TENANT, content_hash=HASH)

        assert await store.put(key, stream()) == 0
        assert await store.read(key) == b""


class TestMissing:
    async def test_reading_an_absent_key_raises_not_found(self, store: FilesystemBlobStore) -> None:
        with pytest.raises(NotFoundError):
            await store.read(store.key_for(tenant_id=TENANT, content_hash=HASH))

    async def test_opening_an_absent_key_raises_not_found(self, store: FilesystemBlobStore) -> None:
        with pytest.raises(NotFoundError):
            await store.open(store.key_for(tenant_id=TENANT, content_hash=HASH))

    async def test_deleting_is_idempotent(self, store: FilesystemBlobStore) -> None:
        # The purge job may be redelivered; a second delete is not an error.
        key = store.key_for(tenant_id=TENANT, content_hash=HASH)
        await store.put(key, stream(b"data"))

        assert await store.delete(key) is True
        assert await store.delete(key) is False


class TestAtomicity:
    async def test_a_failed_write_leaves_nothing_behind(
        self, store: FilesystemBlobStore, tmp_path: Path
    ) -> None:
        """A reader must never observe a half-written blob.

        The upload streams for seconds and the job it enqueues can be claimed
        the instant it commits, so a torn read is a real race rather than a
        theoretical one.
        """

        async def failing() -> AsyncIterator[bytes]:
            yield b"partial"
            raise RuntimeError("connection dropped")

        key = store.key_for(tenant_id=TENANT, content_hash=HASH)

        with pytest.raises(RuntimeError, match="connection dropped"):
            await store.put(key, failing())

        with pytest.raises(NotFoundError):
            await store.read(key)
        # And no temporary file was orphaned for every retry to accumulate.
        assert not list((tmp_path / "blobs").rglob("*.part"))


class TestPathTraversal:
    """The key must not be able to escape the store root.

    Two independent defences: a charset that cannot express traversal, and a
    containment check on the resolved path for whatever the charset missed.
    """

    @pytest.mark.parametrize(
        "key",
        [
            "../escape",
            "../../etc/passwd",
            "tenant/../../escape",
            "/absolute/path",
            "tenant/..%2fescape",
            "tenant/\\windows\\path",
            "tenant/nul\x00byte",
            "",
            "x" * 300,
        ],
    )
    async def test_unsafe_keys_are_refused(self, store: FilesystemBlobStore, key: str) -> None:
        with pytest.raises(ValueError, match=r"[Uu]nsafe|escapes"):
            await store.read(key)

    async def test_an_unsafe_key_is_refused_on_write_too(self, store: FilesystemBlobStore) -> None:
        # Reads are not the only direction; a traversal on write is arbitrary
        # file *creation*.
        with pytest.raises(ValueError, match=r"[Uu]nsafe|escapes"):
            await store.put("../../evil", stream(b"payload"))

    async def test_a_legitimate_key_is_not_refused(self, store: FilesystemBlobStore) -> None:
        # The guard must not be so strict that real keys fail.
        key = store.key_for(tenant_id=TENANT, content_hash=HASH)

        await store.put(key, stream(b"fine"))

        assert await store.read(key) == b"fine"
