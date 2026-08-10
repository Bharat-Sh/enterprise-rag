"""In-process token bucket (docs/adr/0008).

Honest about what it is: **per worker process**. With N uvicorn workers the
effective limit is N times the configured one. `ServerSettings.workers` defaults
to 1 and the deployment story is replica scaling, so this is exact locally and
in a single-process container and approximate anywhere else. M9 adds a Redis
adapter behind the same port, which makes it exact everywhere and changes no
call site.

Buckets are stored lazily and swept. That matters because the login limiter is
keyed on client IP, an attacker-controlled key space: without eviction, a flood
from many source addresses is a memory leak with a rate limiter's name on it.

The sweep criterion is **"this bucket has refilled completely"**, not "this
bucket is old". A full bucket and an absent one are indistinguishable, because a
missing key is recreated at capacity — so forgetting a full bucket cannot
accidentally hand back capacity that had been spent, and no choice of timeout
has to be justified. A bucket under continuous load never refills and is
therefore never dropped, which is exactly right: it is the one still doing work.

There is no lock. Each worker is a single event loop and every operation here
runs to completion between `await` points, so no task can observe a half-updated
bucket. An `anyio.Lock` would put contention on the hottest path in the process
to prevent interleaving that cannot happen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag.core.logging import get_logger
from rag.domain.ratelimit import RateLimitDecision

if TYPE_CHECKING:
    from collections.abc import Callable

    from rag.domain.ratelimit import RateLimitPolicy

__all__ = ["InProcessRateLimiter"]

_log = get_logger(__name__)

#: Sweep when the store grows past this, rather than on a timer. A background
#: task would need a lifecycle, cancellation, and a test of its own; a sweep
#: amortised over insertions needs none of them.
_SWEEP_THRESHOLD = 1024


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated_at: float
    #: Kept per bucket so the sweep can tell a full bucket from a busy one
    #: without knowing which policy the next caller will present.
    capacity: float
    refill_per_second: float

    def is_full_at(self, now: float) -> bool:
        elapsed = max(0.0, now - self.updated_at)
        return self.tokens + elapsed * self.refill_per_second >= self.capacity


class InProcessRateLimiter:
    """Satisfies `rag.domain.ports.RateLimiter`.

    `clock` is injected and defaults to a monotonic source. Monotonic rather
    than wall clock deliberately: an NTP step backwards would otherwise look
    like a bucket refilling by a negative amount, and every caller would be
    locked out until real time caught up.
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._buckets: dict[str, _Bucket] = {}

    async def check(self, key: str, policy: RateLimitPolicy, *, cost: int = 1) -> RateLimitDecision:
        now = self._clock()
        bucket = self._buckets.get(key)

        if bucket is None:
            if len(self._buckets) >= _SWEEP_THRESHOLD:
                self._sweep(now)
            # A new key starts full, so a caller's first request is never denied.
            bucket = _Bucket(
                tokens=float(policy.capacity),
                updated_at=now,
                capacity=float(policy.capacity),
                refill_per_second=policy.refill_per_second,
            )
            self._buckets[key] = bucket
        else:
            elapsed = max(0.0, now - bucket.updated_at)
            bucket.tokens = min(
                float(policy.capacity), bucket.tokens + elapsed * policy.refill_per_second
            )
            bucket.updated_at = now
            # A policy change at runtime (a config reload, a different route
            # sharing the key) must not leave the sweep reasoning about stale
            # numbers.
            bucket.capacity = float(policy.capacity)
            bucket.refill_per_second = policy.refill_per_second

        limit = policy.requests_per_minute
        if bucket.tokens >= cost:
            bucket.tokens -= cost
            return RateLimitDecision.allow(remaining=bucket.tokens, limit=limit)

        deficit = cost - bucket.tokens
        return RateLimitDecision.deny(wait_seconds=deficit / policy.refill_per_second, limit=limit)

    def _sweep(self, now: float) -> None:
        """Drop every bucket that has refilled completely. See the module docstring."""
        full = [key for key, bucket in self._buckets.items() if bucket.is_full_at(now)]
        for key in full:
            del self._buckets[key]
        if full:
            _log.debug("ratelimit.swept", dropped=len(full), remaining=len(self._buckets))
