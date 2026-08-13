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
from rag.adapters.auth.keys import build_keyring
from rag.adapters.auth.passwords import Argon2PasswordHasher
from rag.adapters.auth.tokens import JwtTokenService
from rag.adapters.blobs.filesystem import FilesystemBlobStore
from rag.adapters.models import HttpModelClient
from rag.adapters.parsers import build_registry
from rag.adapters.ratelimit.inprocess import InProcessRateLimiter
from rag.adapters.vectorstore import QdrantVectorStore
from rag.api.errors import register_exception_handlers
from rag.api.middleware import RequestContextMiddleware, TimingMiddleware
from rag.api.middleware.body_limit import BodySizeLimitMiddleware
from rag.api.v1.routers import (
    api_keys,
    auth,
    collections,
    documents,
    health,
    search,
    users,
    well_known,
)
from rag.core.config import Settings, get_settings
from rag.core.health import HealthRegistry
from rag.core.logging import configure_logging, get_logger
from rag.db.session import create_engine, create_session_factory, ping

_log = get_logger(__name__)

#: Every product endpoint hangs off this. Health probes and the JWKS document
#: deliberately do not — see `create_app`.
API_V1_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open resources on startup, close them on shutdown.

    Everything before `yield` runs at startup; everything after runs at
    shutdown, including when startup itself failed part-way. Later milestones
    add their resources here and register the matching readiness check:

        M1  Postgres engine + session factory  -> health.register("postgres", ...)
        M2  Signing keyring, hasher, limiter   -> no probe: all in-process
        M4  Model service HTTP client          -> health.register("model-service", ...)
        M5  Qdrant client                      -> health.register("qdrant", ...)
        M9  Redis client                       -> health.register("redis", required=False)

    **The model service and Qdrant are both registered as *not required*, and
    that stays true now that `/search` exists.** The M4 note here said M5 would
    flip them; on writing M5 that turned out to be wrong.

    Readiness governs load-balancer membership for the *whole API*. If the GPU
    box or the vector index is down, search cannot answer — but upload,
    document management, collections and auth all still work. Marking either
    required removes every replica from the load balancer, turning "search is
    degraded" into "the product is down". It also buys nothing: every replica
    shares one model service and one Qdrant, so there is no healthy replica to
    fail over to. A 503 from `/search` is the honest, contained answer.

    The M2 resources register no readiness check on purpose. They have no
    network dependency and no failure mode after construction: a bad signing key
    or an unreadable PEM raises here, at startup, which fails the process rather
    than producing a service that is up and cannot authenticate anyone.
    """
    settings: Settings = app.state.settings

    _log.info(
        "startup.begin",
        environment=str(settings.environment),
        log_format=str(settings.effective_log_format),
        docs_enabled=settings.effective_docs_enabled,
    )

    app.state.health = HealthRegistry()

    # Creating the engine opens no sockets — the pool connects lazily on first
    # use. So a database that is down delays readiness rather than preventing
    # the process from starting, which is what lets the pod report *why* it is
    # not ready instead of crash-looping silently.
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    app.state.db_engine = engine
    app.state.db_session_factory = session_factory

    async def check_postgres() -> None:
        async with session_factory() as session:
            await ping(session)

    app.state.health.register("postgres", check_postgres, timeout_seconds=2.0, required=True)

    # Authentication resources. Built here rather than at import time so a test
    # gets its own keyring — and so a missing or malformed signing key kills
    # startup instead of surfacing as a 500 on the first login.
    keyring = build_keyring(settings.auth)
    app.state.keyring = keyring
    app.state.token_service = JwtTokenService(keyring, settings.auth)
    app.state.password_hasher = Argon2PasswordHasher(settings.auth)
    app.state.rate_limiter = InProcessRateLimiter()

    # Blob storage. No readiness check: the filesystem adapter creates its root
    # here and has no network dependency, so a failure is a startup failure.
    # The S3 adapter will register one, because a remote store genuinely can be
    # down while the process is up.
    app.state.blob_store = FilesystemBlobStore(settings.ingestion.blob_root)
    # Built once: the binary parsers hold configured limits, and constructing
    # them per request would re-read configuration on the hot path.
    app.state.parsers = build_registry(settings.ingestion)

    # No token counter here. Chunking happens in the worker, and loading a 17 MB
    # vocabulary into a process that never tokenizes anything is pure cost.

    # Opens no socket: httpx connects lazily. A model service that is down
    # therefore shows up in the readiness body rather than preventing boot.
    model_client = HttpModelClient(settings.model_service)
    app.state.model_client = model_client
    app.state.health.register(
        "model-service",
        model_client.ping,
        # Generous relative to the 2s default: this crosses a network to a
        # process that may be mid-batch on a GPU. A probe that times out while
        # the service is merely busy reports an outage that is not happening.
        timeout_seconds=5.0,
        required=False,
    )

    # The vector index. `ensure_ready` creates the collection and its payload
    # indexes if absent — done here, at startup, so a misconfigured or
    # unreachable store fails where an operator is watching rather than on the
    # first document a user uploads.
    #
    # Tolerated rather than fatal: a Qdrant that is down should leave the API
    # up and reporting itself unready-for-search, not crash-looping. The
    # readiness check below is what surfaces it.
    vector_store = QdrantVectorStore(settings.qdrant)
    app.state.vector_store = vector_store
    try:
        await vector_store.ensure_ready()
    except Exception as exc:
        _log.warning("startup.vector_store_unavailable", error=f"{type(exc).__name__}: {exc}")

    app.state.health.register("qdrant", vector_store.ping, timeout_seconds=3.0, required=False)

    _log.info(
        "startup.complete",
        health_checks=list(app.state.health.names),
        database=settings.database.safe_dsn,
        signing_kid=keyring.signing_kid,
        rate_limiting=settings.rate_limit.enabled,
        model_service=settings.model_service.base_url,
        qdrant=(
            f"local:{settings.qdrant.local_path}"
            if settings.qdrant.uses_local_mode
            else settings.qdrant.url
        ),
    )
    try:
        yield
    finally:
        # Runs on clean shutdown and on startup failure alike, so resource
        # teardown belongs here rather than after `yield` unguarded.
        await vector_store.aclose()
        await model_client.aclose()
        await engine.dispose()
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
    #
    # The body limit sits inside the context middleware but outside routing, so
    # an oversized upload is rejected while it streams — before Starlette has
    # spooled it to disk — and the rejection still carries correlation ids.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.ingestion.max_upload_bytes)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app, settings)

    # Probes and the JWKS document deliberately sit outside /api/v1 — they are
    # infrastructure, not product API, and must not move when the API version
    # changes. `/.well-known/` is additionally fixed by RFC 8615.
    app.include_router(health.router)
    app.include_router(well_known.router)

    app.include_router(auth.router, prefix=API_V1_PREFIX)
    app.include_router(api_keys.router, prefix=API_V1_PREFIX)
    app.include_router(users.router, prefix=API_V1_PREFIX)
    app.include_router(collections.router, prefix=API_V1_PREFIX)
    app.include_router(documents.router, prefix=API_V1_PREFIX)
    app.include_router(search.router, prefix=API_V1_PREFIX)

    return app


# NOTE: there is deliberately no module-level `app` here. Building one at import
# time would mean that importing `create_app` (as the test suite does) reads the
# ambient environment and the local .env, destroying test hermeticity. The
# instantiation lives in `rag.api.asgi` instead — see that module's docstring.
