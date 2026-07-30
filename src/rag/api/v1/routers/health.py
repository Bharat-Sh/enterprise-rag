"""Liveness and readiness probes.

These are two genuinely different questions and conflating them causes outages.

``GET /health`` — **liveness.** Is this process alive and able to answer? It
checks nothing external and returns 200 whenever it can respond at all.

``GET /ready`` — **readiness.** Can this process usefully serve traffic right
now? It probes registered dependencies and returns 503 if a required one is
unhealthy.

Why the split matters: an orchestrator *restarts* a container that fails
liveness but merely *removes it from the load balancer* when it fails readiness.
Wire a database ping into your liveness probe and a five-second Postgres blip
becomes a rolling restart of your entire fleet — turning a brief degradation
into a full outage plus a thundering herd of cold starts.

Both endpoints live outside `/api/v1`. Probes are infrastructure, not product
API, and should not break when the API version changes.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field

from rag.api.deps import HealthRegistryDep, SettingsDep
from rag.core.health import is_ready

router = APIRouter(tags=["health"])


class LivenessResponse(BaseModel):
    """Process is running."""

    status: str = Field(examples=["ok"])
    service: str
    version: str
    environment: str


class ComponentStatus(BaseModel):
    """Result of a single dependency check."""

    name: str
    healthy: bool
    required: bool
    duration_ms: float
    detail: str | None = None


class ReadinessResponse(BaseModel):
    """Aggregate readiness plus per-dependency detail."""

    ready: bool
    components: list[ComponentStatus]


@router.get(
    "/health",
    response_model=LivenessResponse,
    summary="Liveness probe",
    description="Returns 200 whenever the process can serve a request. Checks no dependencies.",
)
async def health(settings: SettingsDep) -> LivenessResponse:
    return LivenessResponse(
        status="ok",
        service=settings.service_name,
        version=settings.version,
        environment=str(settings.environment),
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description=(
        "Probes every registered dependency concurrently. Returns 503 if any "
        "*required* dependency is unhealthy; unhealthy optional dependencies are "
        "reported in the body but do not fail the probe."
    ),
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
async def ready(response: Response, registry: HealthRegistryDep) -> ReadinessResponse:
    components = await registry.run_all()
    ready_now = is_ready(components)

    # Set the status explicitly rather than raising: the body is useful in both
    # cases, and an operator debugging a failing probe wants the detail.
    response.status_code = status.HTTP_200_OK if ready_now else status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        ready=ready_now,
        components=[
            ComponentStatus(
                name=component.name,
                healthy=component.healthy,
                required=component.required,
                duration_ms=component.duration_ms,
                detail=component.detail,
            )
            for component in components
        ],
    )
