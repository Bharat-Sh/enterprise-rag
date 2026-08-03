"""Identifier generation.

We use **UUIDv7** (RFC 9562) for primary keys rather than UUIDv4.

Why it matters: v4 is uniformly random, so consecutive inserts land in random
leaves of the B-tree index. Every insert dirties a different page, write
amplification climbs, and the index fragments. v7 puts a 48-bit millisecond
timestamp in the high bits, so ids generated near each other in time sort near
each other — inserts append to the right-hand edge of the index like a sequence,
while remaining globally unique and unguessable.

Why not `BIGSERIAL`: sequences require a database round trip before the object
exists, leak volume through guessable ids, and make cross-shard merges painful.
Client-generated ids let us build an entire object graph in memory and insert it
in one statement.

Python 3.12 has no `uuid.uuid7()`, so we implement the layout directly. The
extra ordering benefit of a monotonic counter within a millisecond is not worth
the shared mutable state; ties inside one millisecond are broken randomly, which
is harmless for index locality.
"""

from __future__ import annotations

import secrets
import time
from uuid import UUID

__all__ = ["timestamp_of", "uuid7"]


def uuid7() -> UUID:
    """Generate a time-ordered UUIDv7."""
    unix_ms = int(time.time() * 1000)

    # Layout: 48 bits timestamp | 4 bits version | 12 bits rand | 2 bits variant
    #         | 62 bits rand
    raw = bytearray(unix_ms.to_bytes(6, "big") + secrets.token_bytes(10))
    raw[6] = (raw[6] & 0x0F) | 0x70  # version 7
    raw[8] = (raw[8] & 0x3F) | 0x80  # RFC 9562 variant
    return UUID(bytes=bytes(raw))


def timestamp_of(value: UUID) -> float:
    """Extract the creation time (Unix seconds) embedded in a UUIDv7.

    Useful for debugging and for coarse time-range scans without a separate
    index. Raises for any other UUID version, because reading the first six
    bytes of a v4 yields a plausible-looking but entirely meaningless timestamp.
    """
    if value.version != 7:
        raise ValueError(f"Expected a UUIDv7, got version {value.version}")
    return int.from_bytes(value.bytes[:6], "big") / 1000.0
