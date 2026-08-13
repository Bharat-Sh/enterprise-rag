"""The model service's HTTP layer.

Everything here runs on any machine, because the model itself sits behind
`InferenceBackend` and `StubBackend` satisfies it without torch. That is what
makes the parts most likely to break — auth, limits, batching, ordering, error
shapes — testable on a CI runner with no GPU.

Concurrency: one lock, no queue
-------------------------------
Every call into the backend is serialised by a single lock and runs in a worker
thread. The lock is not a performance choice, it is a correctness one: two
concurrent forward passes on a 6 GB card is how you get an out-of-memory error,
and a CUDA OOM can leave the context unusable for the rest of the process's
life. The thread hop is what keeps the lock from being held on the event loop,
so requests waiting their turn — and the health probe — still get served.

docs/adr/0004 specified dynamic cross-request micro-batching here: a queue, a
timer, and a single consumer coalescing separate requests into one pass. That is
**deferred, not cancelled** (docs/adr/0011). A caller already sends many texts
per request, which captures most of the batching win; the remaining gain appears
only under many small *concurrent* requests, which is query-time embedding and
does not exist before M6. Building it now would mean tuning a `max_wait_ms`
against a load shape nobody has measured.

Logging: never the text
-----------------------
Request and response bodies are customer data crossing a trust boundary. This
service is tenant-blind — it never learns whose text it is holding — so it
cannot make an access decision about a log line, and the only safe policy is
that text never reaches one. Counts, token totals and durations only. Asserted
by `tests/security/test_model_service_privacy.py`.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from typing import Annotated

import anyio
from fastapi import Depends, FastAPI, HTTPException, Request, status

from model_service.backend import InferenceBackend, build_backend
from model_service.batching import plan_batches
from model_service.schemas import (
    EmbeddingPayload,
    EmbedRequest,
    EmbedResponse,
    InfoResponse,
    RerankRequest,
    RerankResponse,
    RerankScore,
    SparsePayload,
)
from model_service.settings import Backend, Settings
from rag.core.config import LogFormat
from rag.core.logging import configure_logging, get_logger

__all__ = ["create_app"]

_log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the models, then serve.

    Loading happens in a thread so the event loop can answer `/health` during
    the 20-60 seconds it takes. Readiness stays false until it finishes, which
    is the whole reason those two probes are separate: an orchestrator must be
    able to tell "still warming up" from "wedged, restart me". Conflating them
    means a cold start looks like a failure and the container is killed and
    restarted forever.
    """
    settings: Settings = app.state.settings

    if settings.backend is Backend.STUB:
        _log.warning(
            "model_service.stub_backend",
            detail=(
                "Serving deterministic fake vectors. Chunks embedded now are stamped "
                "embedding_model='stub' and must be re-embedded before they are useful."
            ),
        )

    started = time.perf_counter()
    _log.info("model_service.loading", backend=str(settings.backend), device=settings.device)

    # Not guarded by try/except: a model that will not load means this process
    # can never do its job, and a service that starts anyway would accept
    # requests only to fail every one of them. Crash-loop and let the
    # orchestrator report it.
    backend = await anyio.to_thread.run_sync(build_backend, settings)

    app.state.backend = backend
    app.state.gpu_lock = anyio.Lock()
    app.state.ready = True

    info = backend.info()
    _log.info(
        "model_service.ready",
        embedding_model=info.embedding_model,
        embedding_version=info.embedding_version,
        dimensions=info.dimensions,
        tokenizer_hash=info.tokenizer_hash,
        reranker_model=info.reranker_model,
        load_seconds=round(time.perf_counter() - started, 1),
        max_sequence_tokens=settings.max_sequence_tokens,
        max_batch_tokens=settings.max_batch_tokens,
    )
    try:
        yield
    finally:
        app.state.ready = False
        _log.info("model_service.stopped")


def get_settings_dep(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_backend(request: Request) -> InferenceBackend:
    """The loaded backend, or 503 while it is still loading.

    Reachable because `/health` answers during startup, so a caller can be
    talking to this process before the weights are resident.
    """
    backend: InferenceBackend | None = getattr(request.app.state, "backend", None)
    if backend is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are still loading.",
        )
    return backend


async def require_api_key(request: Request) -> None:
    """Check the shared bearer token, when one is configured.

    `compare_digest` rather than `==`: string comparison short-circuits on the
    first differing byte, and a remote attacker can measure that. Overkill for a
    shared secret on an internal hop, and it is one function call.

    No auth configured means no check — correct on a laptop. The cost of being
    wrong about "it is on an internal network" is an unmetered GPU for whoever
    finds it, so the option exists and production is expected to use it.
    """
    settings: Settings = request.app.state.settings
    if settings.api_key is None:
        return

    header = request.headers.get("authorization", "")
    scheme, _, presented = header.partition(" ")
    expected = settings.api_key.get_secret_value()
    if scheme.lower() != "bearer" or not secrets.compare_digest(presented, expected):
        # No detail about which part failed, for the same reason
        # `rag.domain.errors.AuthenticationError` carries none.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
BackendDep = Annotated[InferenceBackend, Depends(get_backend)]
AuthDep = Annotated[None, Depends(require_api_key)]


def _reject_over_length(token_counts: list[int], *, limit: int, field: str) -> None:
    """Refuse input the model would have to truncate.

    Truncating instead would produce a vector that is structurally perfect and
    semantically missing the text's tail — no error, no warning, and permanent
    once indexed. This is the failure that `max_sequence_tokens` exists to
    prevent, so it must be an error and not a silent repair.

    The message carries positions and token counts, never the text: an error
    body is a log line waiting to happen, and this service must not put customer
    text into one.
    """
    offenders = [
        {"index": index, "tokens": count}
        for index, count in enumerate(token_counts)
        if count > limit
    ]
    if offenders:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "error": "input_too_long",
                "message": (
                    f"{len(offenders)} of {len(token_counts)} {field} exceed the "
                    f"{limit}-token limit. Split them; they will not be truncated."
                ),
                "limit_tokens": limit,
                "offenders": offenders[:20],
            },
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the service. A factory, so tests can inject configuration."""
    settings = settings or Settings()

    configure_logging(
        level=settings.log_level,
        log_format=LogFormat(settings.log_format) if settings.log_format else LogFormat.JSON,
        service_name=settings.service_name,
        version="0.1.0",
        environment="model-service",
    )

    app = FastAPI(
        title="RAG Model Service",
        description="BGE-M3 embeddings and cross-encoder reranking on a local GPU.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.ready = False

    @app.get("/health", summary="Liveness")
    async def health() -> dict[str, str]:
        """Answers as soon as the process can serve, including mid-load.

        Deliberately checks nothing. Wiring the model load into liveness turns
        every cold start into a restart, and a restart into another cold start.
        """
        return {"status": "ok"}

    @app.get("/ready", summary="Readiness")
    async def ready(request: Request) -> dict[str, bool]:
        if not getattr(request.app.state, "ready", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models are still loading.",
            )
        return {"ready": True}

    @app.get("/v1/info", response_model=InfoResponse, summary="Loaded model identity")
    async def info(backend: BackendDep, config: SettingsDep, _: AuthDep) -> InfoResponse:
        described = backend.info()
        return InfoResponse(
            embedding_model=described.embedding_model,
            embedding_version=described.embedding_version,
            dimensions=described.dimensions,
            max_sequence_tokens=config.max_sequence_tokens,
            tokenizer_hash=described.tokenizer_hash,
            reranker_model=described.reranker_model,
            backend=described.backend,
        )

    @app.post("/v1/embed", response_model=EmbedResponse, summary="Embed texts")
    async def embed(
        payload: EmbedRequest,
        request: Request,
        backend: BackendDep,
        config: SettingsDep,
        _: AuthDep,
    ) -> EmbedResponse:
        if not payload.texts:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"error": "empty_request", "message": "`texts` must not be empty."},
            )
        if len(payload.texts) > config.max_texts_per_request:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail={
                    "error": "too_many_texts",
                    "message": (
                        f"{len(payload.texts)} texts exceeds the per-request limit of "
                        f"{config.max_texts_per_request}."
                    ),
                    "limit": config.max_texts_per_request,
                },
            )

        started = time.perf_counter()
        lock: anyio.Lock = request.app.state.gpu_lock

        # Counting is cheap CPU work but not free at 256 texts, so it takes the
        # thread hop too rather than blocking the loop for everyone else.
        token_counts = await anyio.to_thread.run_sync(backend.count_tokens, payload.texts)
        _reject_over_length(token_counts, limit=config.max_sequence_tokens, field="texts")

        batches = plan_batches(
            token_counts,
            max_batch_tokens=config.max_batch_tokens,
            max_batch_items=config.max_batch_items,
        )

        results: list[EmbeddingPayload | None] = [None] * len(payload.texts)
        for indices in batches:
            texts = [payload.texts[index] for index in indices]
            # The lock is taken per batch, not for the whole request. A caller
            # sending 256 texts must not lock every other caller out for the
            # duration; releasing between passes lets requests interleave at
            # batch granularity, which is the fairness the deferred
            # micro-batcher would have provided.
            async with lock:
                # `partial`, not a lambda: a lambda closing over the loop
                # variable is the classic late-binding bug, and ruff's B023
                # flags it precisely because it is invisible until the day the
                # call stops being immediate.
                embeddings = await anyio.to_thread.run_sync(
                    partial(backend.embed, texts, mode=payload.mode)
                )
            for position, embedding in zip(indices, embeddings, strict=True):
                results[position] = EmbeddingPayload(
                    dense=list(embedding.dense),
                    sparse=SparsePayload(
                        indices=list(embedding.sparse_indices),
                        values=list(embedding.sparse_values),
                    ),
                )

        # `strict=True` above already guarantees every slot was written; this
        # narrows the type for mypy without a cast, and would catch a future
        # batching change that skipped an index.
        ordered = [result for result in results if result is not None]
        if len(ordered) != len(payload.texts):  # pragma: no cover - unreachable
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "incomplete_batch", "message": "Batching lost an input."},
            )

        described = backend.info()
        _log.info(
            "model_service.embedded",
            texts=len(payload.texts),
            tokens=sum(token_counts),
            batches=len(batches),
            mode=payload.mode,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return EmbedResponse(
            embedding_model=described.embedding_model,
            embedding_version=described.embedding_version,
            dimensions=described.dimensions,
            embeddings=ordered,
        )

    @app.post("/v1/rerank", response_model=RerankResponse, summary="Rerank passages")
    async def rerank(
        payload: RerankRequest,
        request: Request,
        backend: BackendDep,
        config: SettingsDep,
        _: AuthDep,
    ) -> RerankResponse:
        described = backend.info()
        if described.reranker_model is None:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail={
                    "error": "reranker_disabled",
                    "message": "This service was started with the reranker disabled.",
                },
            )
        if not payload.passages:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"error": "empty_request", "message": "`passages` must not be empty."},
            )
        if len(payload.passages) > config.max_passages_per_rerank:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail={
                    "error": "too_many_passages",
                    "message": (
                        f"{len(payload.passages)} passages exceeds the per-request limit "
                        f"of {config.max_passages_per_rerank}."
                    ),
                    "limit": config.max_passages_per_rerank,
                },
            )

        started = time.perf_counter()
        lock: anyio.Lock = request.app.state.gpu_lock

        # A cross-encoder sees query and passage concatenated, so the pair is
        # what has to fit the window — checking the passage alone would accept
        # pairs the model then truncates, dropping the end of the passage.
        pairs = [f"{payload.query} {passage}" for passage in payload.passages]
        token_counts = await anyio.to_thread.run_sync(backend.count_tokens, pairs)
        _reject_over_length(
            token_counts, limit=config.max_sequence_tokens, field="query+passage pairs"
        )

        batches = plan_batches(
            token_counts,
            max_batch_tokens=config.max_batch_tokens,
            max_batch_items=config.max_batch_items,
        )

        scores: list[float] = [0.0] * len(payload.passages)
        for indices in batches:
            passages = [payload.passages[index] for index in indices]
            async with lock:
                batch_scores = await anyio.to_thread.run_sync(
                    partial(backend.rerank, payload.query, passages)
                )
            for position, score in zip(indices, batch_scores, strict=True):
                scores[position] = score

        ranked = sorted(
            (RerankScore(index=index, score=score) for index, score in enumerate(scores)),
            key=lambda result: result.score,
            reverse=True,
        )
        if payload.top_k is not None:
            ranked = ranked[: payload.top_k]

        _log.info(
            "model_service.reranked",
            passages=len(payload.passages),
            tokens=sum(token_counts),
            batches=len(batches),
            returned=len(ranked),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return RerankResponse(reranker_model=described.reranker_model, results=ranked)

    return app
