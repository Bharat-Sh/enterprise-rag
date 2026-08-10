"""Filesystem `BlobStore` (docs/adr/0009).

Correct for a single node and for the local development loop, which is what M3
needs. It is explicitly *not* the production answer: two API replicas do not
share a disk, so an upload handled by one and parsed by a worker on another
would not find its bytes. The S3 adapter arrives with deployment, and this file
exists partly to prove the port is the right shape before that.

Two properties worth reading for:

**Keys are constructed here and nowhere else.** `key_for` is on the port so
callers never build a path, which is what keeps the layout an implementation
detail — and, more importantly, keeps caller-supplied strings out of the
filesystem path entirely.

**Writes are atomic.** Content goes to a temporary file and is renamed into
place. A worker reading a key can therefore never observe a half-written blob,
which is otherwise a real race: the upload streams for seconds and the job it
enqueues may be claimed the instant it commits.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import anyio

from rag.core.logging import get_logger
from rag.domain.errors import NotFoundError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["FilesystemBlobStore"]

_log = get_logger(__name__)

#: Read size when streaming a blob back out.
_READ_CHUNK = 1024 * 1024

#: Keys we are willing to touch. Content hashes are hex and tenants are UUIDs,
#: so this is generous — its job is to make a traversal attempt impossible to
#: express rather than to validate a well-formed key.
_SAFE_KEY = re.compile(r"\A[A-Za-z0-9/_.-]{1,256}\Z")


class FilesystemBlobStore:
    """Satisfies `rag.domain.ports.BlobStore`."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    # -- key construction --------------------------------------------------

    def key_for(self, *, tenant_id: UUID, content_hash: str) -> str:
        # The two-character shard keeps directory sizes sane: some filesystems
        # degrade badly past a few tens of thousands of entries in one
        # directory, and a busy tenant reaches that.
        return f"{tenant_id}/{content_hash[:2]}/{content_hash}"

    def derived_key_for(self, *, tenant_id: UUID, content_hash: str, kind: str) -> str:
        return f"{tenant_id}/{content_hash[:2]}/{content_hash}.{kind}"

    # -- io ----------------------------------------------------------------

    async def put(self, key: str, stream: AsyncIterator[bytes]) -> int:
        """Write atomically, returning the byte count.

        The temporary file is created in the destination directory rather than
        the system temp dir, because `os.replace` is only atomic within a
        filesystem — across one it silently degrades to copy-then-delete, which
        reintroduces the torn-read this method exists to prevent.
        """
        path = self._path_for(key)
        await anyio.to_thread.run_sync(lambda: path.parent.mkdir(parents=True, exist_ok=True))

        handle, raw_temp = await anyio.to_thread.run_sync(
            lambda: tempfile.mkstemp(dir=path.parent, suffix=".part")
        )
        os.close(handle)
        temp = Path(raw_temp)

        written = 0
        try:
            async with await anyio.open_file(temp, "wb") as file:
                async for block in stream:
                    await file.write(block)
                    written += len(block)
            await anyio.to_thread.run_sync(lambda: os.replace(temp, path))
        except BaseException:
            # Includes cancellation: an aborted upload must not leave a `.part`
            # file behind for every retry.
            await anyio.to_thread.run_sync(lambda: temp.unlink(missing_ok=True))
            raise

        return written

    async def open(self, key: str) -> AsyncIterator[bytes]:
        path = self._path_for(key)
        if not await anyio.to_thread.run_sync(path.is_file):
            raise NotFoundError("Blob", key)
        return self._iterate(path)

    async def read(self, key: str) -> bytes:
        path = self._path_for(key)
        if not await anyio.to_thread.run_sync(path.is_file):
            raise NotFoundError("Blob", key)
        return await anyio.to_thread.run_sync(path.read_bytes)

    async def delete(self, key: str) -> bool:
        path = self._path_for(key)

        def _unlink() -> bool:
            try:
                path.unlink()
            except FileNotFoundError:
                return False
            return True

        removed = await anyio.to_thread.run_sync(_unlink)
        if removed:
            _log.info("blob.deleted", key=key)
        return removed

    # -- internals ---------------------------------------------------------

    async def _iterate(self, path: Path) -> AsyncIterator[bytes]:
        async with await anyio.open_file(path, "rb") as file:
            while block := await file.read(_READ_CHUNK):
                yield block

    def _path_for(self, key: str) -> Path:
        """Resolve a key to a path, refusing anything that escapes the root.

        Two independent checks. The charset rejects `..` and absolute paths
        before they reach the filesystem; the containment check catches whatever
        the charset did not think of — symlinks, Windows short names, and the
        next encoding trick nobody has thought of yet. Belt and braces is
        appropriate here: the failure is arbitrary file read or write.
        """
        if not _SAFE_KEY.match(key) or ".." in key:
            raise ValueError(f"Unsafe blob key: {key!r}")

        candidate = (self._root / key).resolve()
        if not candidate.is_relative_to(self._root):
            raise ValueError(f"Blob key escapes the store root: {key!r}")
        return candidate
