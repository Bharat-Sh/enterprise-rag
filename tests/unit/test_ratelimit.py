"""Token-bucket behaviour, driven by an injected clock.

No `sleep` anywhere. A rate-limiter test that waits for real time is slow *and*
flaky, and neither property buys any additional confidence — the thing under
test is arithmetic over a clock reading.
"""

from __future__ import annotations

import pytest

from rag.adapters.ratelimit.inprocess import InProcessRateLimiter
from rag.domain.ratelimit import RateLimitPolicy


class FakeClock:
    """A monotonic clock the test advances explicitly."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def limiter(clock: FakeClock) -> InProcessRateLimiter:
    return InProcessRateLimiter(clock=clock)


#: 60/min = one token per second, with a burst of five.
POLICY = RateLimitPolicy.per_minute(60, burst=5)


class TestPolicy:
    def test_per_minute_converts_to_a_refill_rate(self) -> None:
        policy = RateLimitPolicy.per_minute(120, burst=10)

        assert policy.refill_per_second == pytest.approx(2.0)
        assert policy.capacity == 10
        assert policy.requests_per_minute == 120

    @pytest.mark.parametrize(("capacity", "refill"), [(0, 1.0), (-1, 1.0), (5, 0.0), (5, -1.0)])
    def test_nonsensical_policies_are_rejected_at_construction(
        self, capacity: int, refill: float
    ) -> None:
        with pytest.raises(ValueError, match=r"capacity|refill"):
            RateLimitPolicy(capacity=capacity, refill_per_second=refill)


class TestBucket:
    async def test_a_fresh_key_starts_full(self, limiter: InProcessRateLimiter) -> None:
        # A caller's very first request is never throttled.
        decision = await limiter.check("k", POLICY)

        assert decision.allowed is True
        assert decision.remaining == 4

    async def test_the_burst_is_spent_then_denied(self, limiter: InProcessRateLimiter) -> None:
        for _ in range(5):
            assert (await limiter.check("k", POLICY)).allowed is True

        denied = await limiter.check("k", POLICY)

        assert denied.allowed is False
        assert denied.remaining == 0
        assert denied.limit == 60

    async def test_tokens_refill_over_time(
        self, limiter: InProcessRateLimiter, clock: FakeClock
    ) -> None:
        for _ in range(5):
            await limiter.check("k", POLICY)
        assert (await limiter.check("k", POLICY)).allowed is False

        clock.advance(3.0)  # three tokens back at one per second

        assert (await limiter.check("k", POLICY)).allowed is True
        assert (await limiter.check("k", POLICY)).allowed is True
        assert (await limiter.check("k", POLICY)).allowed is True
        assert (await limiter.check("k", POLICY)).allowed is False

    async def test_refill_is_capped_at_capacity(
        self, limiter: InProcessRateLimiter, clock: FakeClock
    ) -> None:
        # An idle key must not bank an unbounded burst. This is the difference
        # between "you may go fast briefly" and "you may go very fast once".
        await limiter.check("k", POLICY)
        clock.advance(3600.0)

        for _ in range(5):
            assert (await limiter.check("k", POLICY)).allowed is True
        assert (await limiter.check("k", POLICY)).allowed is False

    async def test_retry_after_is_never_zero(self, limiter: InProcessRateLimiter) -> None:
        # `Retry-After: 0` invites a client to retry straight into the same
        # denial, which turns a rate limiter into a retry-storm amplifier.
        for _ in range(5):
            await limiter.check("k", POLICY)

        denied = await limiter.check("k", POLICY)

        assert denied.retry_after_seconds >= 1

    async def test_retry_after_reflects_the_actual_wait(
        self, limiter: InProcessRateLimiter, clock: FakeClock
    ) -> None:
        slow = RateLimitPolicy.per_minute(6, burst=1)  # one token per ten seconds
        await limiter.check("k", slow)

        denied = await limiter.check("k", slow)

        assert denied.retry_after_seconds == 10
        clock.advance(10.0)
        assert (await limiter.check("k", slow)).allowed is True

    async def test_keys_are_independent(self, limiter: InProcessRateLimiter) -> None:
        # One tenant exhausting its allowance must not throttle another.
        for _ in range(5):
            await limiter.check("tenant:a", POLICY)
        assert (await limiter.check("tenant:a", POLICY)).allowed is False

        assert (await limiter.check("tenant:b", POLICY)).allowed is True

    async def test_a_costly_request_can_consume_several_tokens(
        self, limiter: InProcessRateLimiter
    ) -> None:
        assert (await limiter.check("k", POLICY, cost=5)).allowed is True
        assert (await limiter.check("k", POLICY, cost=1)).allowed is False


class TestSweep:
    async def test_full_buckets_are_dropped_and_this_grants_nothing(
        self, limiter: InProcessRateLimiter, clock: FakeClock
    ) -> None:
        """Eviction must be indistinguishable from leaving the bucket alone.

        A missing key is recreated at capacity, so dropping a *full* bucket is a
        no-op. Dropping a partly spent one would hand back capacity that had
        already been used — which is why the criterion is fullness, not age.
        """
        for _ in range(5):
            await limiter.check("spent", POLICY)

        # Fill the store past the sweep threshold with keys that are now idle.
        for index in range(1100):
            await limiter.check(f"idle:{index}", POLICY)
        clock.advance(3600.0)
        await limiter.check("trigger-a-sweep", POLICY)

        # The exhausted key refilled during that hour, so it is legitimately
        # allowed again — by refill, not by amnesia.
        assert (await limiter.check("spent", POLICY)).allowed is True

    async def test_a_busy_bucket_survives_the_sweep(self, limiter: InProcessRateLimiter) -> None:
        for _ in range(5):
            await limiter.check("busy", POLICY)

        for index in range(1100):
            await limiter.check(f"idle:{index}", POLICY)
        await limiter.check("trigger-a-sweep", POLICY)

        # No time has passed, so nothing refilled: the exhausted key must still
        # be exhausted. If the sweep had dropped it, this would be allowed.
        assert (await limiter.check("busy", POLICY)).allowed is False
