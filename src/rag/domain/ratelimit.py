"""Rate-limit policy and decision types (docs/adr/0008).

A **token bucket**, expressed as a steady refill rate plus a capacity. Not a
fixed window: a fixed window permits a 2x burst across the boundary — a hundred
requests at 23:59:59.9 and a hundred more at 00:00:00.1 — which is precisely the
shape of a cost spike on an embedding endpoint. Not a sliding-window log
either: same guarantee, O(n) memory per key, and no natural `Retry-After`.

Bursting is desirable rather than merely tolerated. A page that fires six
requests as it loads is not abuse, so capacity deliberately exceeds one
second's worth of refill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Self

__all__ = ["RateLimitDecision", "RateLimitPolicy"]


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """How fast a key may go, and how much it may bank while idle."""

    #: Bucket size — the largest burst allowed from a fully rested key.
    capacity: int
    #: Steady-state refill.
    refill_per_second: float

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be at least 1")
        if self.refill_per_second <= 0:
            raise ValueError("refill_per_second must be positive")

    @classmethod
    def per_minute(cls, requests: int, *, burst: int) -> Self:
        """Build from the units humans configure in."""
        return cls(capacity=burst, refill_per_second=requests / 60.0)

    @property
    def requests_per_minute(self) -> int:
        """The advertised limit, for the `X-RateLimit-Limit` header."""
        return round(self.refill_per_second * 60)


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """The verdict for one request, and everything the response headers need."""

    allowed: bool
    #: Whole tokens left after this request. Fractions are truncated so the
    #: header never promises capacity that is not fully there.
    remaining: int
    #: Seconds until one token is available. Zero when the request was allowed.
    retry_after_seconds: int
    limit: int

    @classmethod
    def allow(cls, *, remaining: float, limit: int) -> Self:
        return cls(
            allowed=True,
            remaining=max(0, math.floor(remaining)),
            retry_after_seconds=0,
            limit=limit,
        )

    @classmethod
    def deny(cls, *, wait_seconds: float, limit: int) -> Self:
        # Rounded *up*, and never below one: a `Retry-After: 0` invites a client
        # to retry immediately into the same denial, which is how a rate limiter
        # becomes a retry storm amplifier.
        return cls(
            allowed=False,
            remaining=0,
            retry_after_seconds=max(1, math.ceil(wait_seconds)),
            limit=limit,
        )
