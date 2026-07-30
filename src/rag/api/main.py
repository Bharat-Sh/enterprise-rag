"""Application factory and process lifecycle.

Two decisions worth calling out.

**A factory, not a module-level app.** `create_app(settings)` lets tests build an
isolated application with overridden configuration. A module-level `app` built at
import time reads the environment once, at import, and every test then fights
over the same global.

**`lifespan`, not module-level clients.** Connection pools are opened when the
application starts and closed when it stops, and they live on `app.state`. A
module-level `engine = create_engine(...)` opens sockets at import time — which
breaks test collection, leaks connections in forked workers, and gives no
shutdown hook to drain in-flight work.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag import __version__
from rag.api.errors import register_exception_handlers
from rag.api.middleware import RequestContextMiddleware, TimingMiddleware
from rag.api.v1.routers import health
from rag.core.config import Settings, get_settings
from rag.core.health import HealthRegistry
from rag.core.logging import configure_logging, get_logger

_log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open resources on startup, close them on shutdown.

    Everything before `yield` runs at startup; everything after runs at
    shutdown, including when startup itself failed part-way. Later milestones
    add their resources here and register the matching readiness check:

        M1  Postgres engine + session factory  -> health.register("postgres", ...)
        M4  Model service HTTP client          -> health.register("model-service", ...)
        M5  Qdrant client                      -> health.register("qdrant", ...)
        M9  Redis client                       -> health.register("redis", required=False)
    """
    settings: Settings = app.state.settings

    _log.info(
        "startup.begin",
        environment=str(settings.environment),
        log_format=str(settings.effective_log_format),
        docs_enabled=settings.effective_docs_enabled,
    )

    app.state.health = HealthRegistry()

    _log.info("startup.complete", health_checks=list(app.state.health.names))
    try:
        yield
    finally:
        # Runs on clean shutdown and on startup failure alike, so resource
        # teardown belongs here rather than after `yield` unguarded.
        _log.info("shutdown.complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured application instance."""
    settings = settings or get_settings()

    configure_logging(
        level=settings.log_level,
        log_format=settings.effective_log_format,
        service_name=settings.service_name,
        version=settings.version,
        environment=str(settings.environment),
    )

    docs_enabled = settings.effective_docs_enabled
    app = FastAPI(
        title="Enterprise RAG Platform",
        description="Multi-tenant retrieval-augmented generation over your documents.",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )

    # Available to `lifespan` and to `deps.get_app_settings`.
    app.state.settings = settings

    # Middleware is applied bottom-up: the LAST one added is the OUTERMOST.
    # RequestContextMiddleware must therefore be added last, so the ids it binds
    # are already in place when TimingMiddleware writes its access log.
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app, settings)

    # Probes deliberately sit outside /api/v1 — they are infrastructure, not
    # product API, and must not move when the API version changes.
    app.include_router(health.router)

    return app


# Entry point for `uvicorn rag.api.main:app`. Kept at the bottom so importing
# anything else from this module does not build an application as a side effect.
app = create_app()
