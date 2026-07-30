"""FastAPI dependency wiring.

Dependencies are declared as `Annotated` aliases so routers read as
`settings: SettingsDep` rather than repeating `Depends(...)` at every call site.
Overriding one of these in tests (via `app.dependency_overrides`) then swaps the
implementation everywhere at once.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from rag.core.config import Settings, get_settings
from rag.core.errors import DependencyUnavailableError
from rag.core.health import HealthRegistry


def get_app_settings(request: Request) -> Settings:
    """Return the settings bound to this application instance.

    Reads from `app.state` rather than calling `get_settings()` directly, so a
    test that builds an app with custom settings gets those settings rather than
    whatever the cached process-wide singleton holds.
    """
    settings = getattr(request.app.state, "settings", None)
    if isinstance(settings, Settings):
        return settings
    return get_settings()


def get_health_registry(request: Request) -> HealthRegistry:
    """Return the application's health-check registry."""
    registry = getattr(request.app.state, "health", None)
    if not isinstance(registry, HealthRegistry):  # pragma: no cover - lifespan guarantees it
        raise DependencyUnavailableError(
            "health-registry",
            "The health registry was not initialised; the application did not start cleanly.",
        )
    return registry


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
HealthRegistryDep = Annotated[HealthRegistry, Depends(get_health_registry)]
