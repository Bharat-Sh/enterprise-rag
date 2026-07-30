"""Dependency health checks backing the `/ready` endpoint.

We build the *mechanism* now and register the *checks* alongside the subsystems
that need them: Postgres in M1, Qdrant in M5, the model service in M4. The
registry is therefore legitimately empty in M0 — readiness with no registered
checks means "the process is up and nothing it depends on is known to be
broken", which is exactly true.

Two properties worth noting:

*Concurrent with per-check timeouts.* Checks run in parallel and each is bounded
independently, so readiness latency is the slowest single check rather than the
sum, and one wedged dependency cannot hang the probe.

*Required vs optional.* Redis is a degradable dependency — losing it costs us
caching and pushes rate limiting to a local fallback, but the service can still
answer questions. Marking it optional means a Redis outage does not pull every
replica out of the load balancer. Postgres is required; without it we have
nothing to serve.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

HealthCheckFn = Callable[[], Awaitable[None]]
"""A check succeeds by returning and fails by raising. No booleans: an
exception carries a message, and forgetting to check a returned bool is a
classic silent bug."""

DEFAULT_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    """Outcome of a single dependency check."""

    name: str
    healthy: bool
    required: bool
    duration_ms: float
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class _RegisteredCheck:
    name: str
    check: HealthCheckFn
    timeout_seconds: float
    required: bool


@dataclass(slots=True)
class HealthRegistry:
    """Collects dependency checks. One instance per application, on `app.state`."""

    _checks: list[_RegisteredCheck] = field(default_factory=list)

    def register(
        self,
        name: str,
        check: HealthCheckFn,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        required: bool = True,
    ) -> None:
        """Add a dependency check.

        Raises:
            ValueError: if `name` is already registered — a duplicate almost
                always means a subsystem got wired up twice.
        """
        if any(existing.name == name for existing in self._checks):
            raise ValueError(f"Health check {name!r} is already registered")
        self._checks.append(
            _RegisteredCheck(
                name=name,
                check=check,
                timeout_seconds=timeout_seconds,
                required=required,
            )
        )

    @property
    def names(self) -> tuple[str, ...]:
        """Registered check names, in registration order."""
        return tuple(check.name for check in self._checks)

    async def run_all(self) -> list[ComponentHealth]:
        """Run every check concurrently. Never raises."""
        if not self._checks:
            return []
        return list(await asyncio.gather(*(self._run_one(check) for check in self._checks)))

    async def _run_one(self, registered: _RegisteredCheck) -> ComponentHealth:
        start = time.perf_counter()
        detail: str | None = None
        healthy = True
        try:
            async with asyncio.timeout(registered.timeout_seconds):
                await registered.check()
        except TimeoutError:
            healthy = False
            detail = f"timed out after {registered.timeout_seconds:g}s"
        # Deliberately broad: a health probe reports failures, it never raises
        # them. One badly-behaved check must not take down the whole endpoint.
        except Exception as exc:
            healthy = False
            detail = f"{type(exc).__name__}: {exc}"

        return ComponentHealth(
            name=registered.name,
            healthy=healthy,
            required=registered.required,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
            detail=detail,
        )


def is_ready(components: list[ComponentHealth]) -> bool:
    """Ready when every *required* component is healthy.

    An unhealthy optional component is reported in the response body but does
    not fail the probe.
    """
    return all(component.healthy for component in components if component.required)
