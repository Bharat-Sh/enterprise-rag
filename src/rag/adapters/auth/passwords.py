"""Argon2id password hashing.

`argon2-cffi` directly rather than `passlib`. Passlib has been unmaintained
since 2020 and its bcrypt backend broke outright against `bcrypt>=4.1`; a
wrapper whose job is to abstract over algorithms is a liability once it stops
tracking them. bcrypt itself was rejected for the algorithm: it silently
truncates at 72 bytes, so the tail of a long passphrase is decorative, and it
has no memory hardness to slow a GPU down.

Two properties this module exists to guarantee
----------------------------------------------
**Nothing blocks the event loop.** Argon2 at these parameters is 50-100 ms of
solid CPU. Run on the event loop it stalls every concurrent request on the
worker, including open SSE streams — the sixth non-negotiable in CLAUDE.md. No
linter catches it, because `verify` is not a *known* blocking primitive, so the
thread hop is enforced here rather than remembered at call sites.

**A missing hash costs the same as a wrong password.** `users.password_hash` is
nullable for SSO-provisioned accounts, and an unknown email has no hash at all.
Short-circuiting either case would make "this account cannot log in" measurably
faster than "wrong password", which is a user-enumeration oracle that needs no
tooling to exploit — a stopwatch is enough.

On cost parameters
------------------
19 MiB / t=2 / p=1 is the OWASP minimum, chosen over something heftier for a
specific reason: peak memory is `memory_cost x concurrent hashes`, and hashing
runs in a thread pool 40 wide. At 64 MiB that is 2.5 GiB of resident memory
reachable from an unauthenticated endpoint — a memory-exhaustion denial of
service we would have built ourselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import anyio.to_thread
from argon2 import PasswordHasher as _Argon2Hasher
from argon2.exceptions import (
    HashingError,
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)

from rag.core.logging import get_logger

if TYPE_CHECKING:
    from rag.core.config import AuthSettings

__all__ = ["Argon2PasswordHasher"]

_log = get_logger(__name__)

#: Compared against when there is no stored hash, purely to spend the same time.
#: A constant rather than a fresh hash per call: generating one would itself cost
#: an Argon2 round, doubling the work on exactly the path an attacker floods.
_DUMMY_PASSWORD = "not-a-real-password-only-here-to-burn-the-same-cpu"  # noqa: S105


class Argon2PasswordHasher:
    """Satisfies `rag.domain.ports.PasswordHasher`."""

    def __init__(self, settings: AuthSettings) -> None:
        self._hasher = _Argon2Hasher(
            time_cost=settings.argon2_time_cost,
            memory_cost=settings.argon2_memory_cost_kib,
            parallelism=settings.argon2_parallelism,
        )
        # Computed once at construction, at startup, where blocking is free.
        self._dummy_hash = self._hasher.hash(_DUMMY_PASSWORD)

    async def hash(self, password: str) -> str:
        return await anyio.to_thread.run_sync(self._hasher.hash, password)

    async def verify(self, password_hash: str | None, password: str) -> bool:
        """Check a password. A `None` hash still does the full work.

        Returns a bool rather than raising, because every caller wants the same
        indistinguishable failure and an exception type would invite someone to
        branch on it.
        """
        return await anyio.to_thread.run_sync(self._verify_sync, password_hash, password)

    def needs_rehash(self, password_hash: str) -> bool:
        """Whether the hash was produced with weaker parameters than we now use.

        Without this, raising the cost parameters would protect only accounts
        created afterwards, and the passwords most worth protecting — the oldest
        ones — would keep their original cost for ever.
        """
        try:
            return self._hasher.check_needs_rehash(password_hash)
        except InvalidHashError:
            # An unparseable hash is not one of ours. Rehashing on the next
            # successful login is the only way it gets repaired, and it cannot
            # verify anything in the meantime.
            return True

    # -- runs in a worker thread -------------------------------------------

    def _verify_sync(self, password_hash: str | None, password: str) -> bool:
        candidate = password_hash if password_hash is not None else self._dummy_hash
        try:
            self._hasher.verify(candidate, password)
        except VerifyMismatchError:
            return False
        except (VerificationError, InvalidHashError, HashingError) as exc:
            # A stored hash we cannot parse is a data problem, not a login
            # attempt to reason about. Log it — the account is now unusable and
            # somebody needs to know — but tell the caller only "no".
            _log.warning("auth.password_hash_unusable", reason=type(exc).__name__)
            return False
        # Reached only when `password_hash` was None and the caller happened to
        # send the dummy password. Denying is correct: an account with no
        # password cannot be logged into with one.
        return password_hash is not None
