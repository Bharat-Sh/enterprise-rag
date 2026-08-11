"""HTTP client for the GPU model service.

Satisfies `EmbeddingProvider` and `Reranker` (docs/adr/0004). The API and the
ingestion worker hold one of these; neither imports torch, neither needs a GPU,
and pointing at a hosted provider instead is a change to this file and nothing
else.

Failure is closed, not open
---------------------------
`RateLimiter` fails open — losing rate limiting costs fairness, failing closed
costs the whole API. This is the opposite case and the opposite choice. There is
no degraded embedding: a document indexed with placeholder or partial vectors is
unfindable while claiming to be searchable, and nothing anywhere errors. So an
unreachable model service fails the job, which the queue then retries with
backoff, and the document stays visibly `FAILED` rather than silently useless.

Retries are deliberately shallow
--------------------------------
Two, by default. The job queue already retries with exponential backoff at a far
coarser grain (`rag.db.repositories.job.backoff_delay`), so deep retries here
multiply: three requests at a 30-second timeout, times five queue attempts, is a
worker held for seven minutes on a service that is down and a job that has not
moved. The retries here exist to ride out a single dropped connection or a
restart, not an outage — the queue handles outages.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import anyio
import httpx

from rag.adapters.models.schemas import (
    EmbedResponseModel,
    InfoResponseModel,
    RerankResponseModel,
)
from rag.core.errors import ConfigurationError, DependencyUnavailableError
from rag.core.logging import get_logger
from rag.domain.embedding import Embedding, EmbedMode, ModelInfo, RerankResult, SparseVector
from rag.domain.errors import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rag.core.config import ModelServiceSettings

__all__ = ["HttpModelClient"]

_log = get_logger(__name__)

#: Worth trying again: a restart mid-request, a transient CUDA OOM that a
#: quieter moment resolves, a load balancer with no backend for a second.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

#: The request itself is wrong and will be wrong identically next time — an
#: over-length chunk does not shorten on retry. Raised as a `DomainError` so the
#: worker dead-letters the job immediately instead of burning five attempts.
_PERMANENT_STATUS = frozenset({400, 413, 422})

#: Half-window jitter, matching `rag.db.repositories.job.backoff_delay`. Full
#: jitter spreads a herd better but makes the first retry occasionally
#: near-instant, which is the opposite of backing off.
_JITTER_FRACTION = 0.5

DEPENDENCY_NAME = "model-service"


class HttpModelClient:
    """Talks to the model service. Satisfies `EmbeddingProvider` and `Reranker`."""

    def __init__(
        self,
        settings: ModelServiceSettings,
        *,
        client: httpx.AsyncClient | None = None,
        jitter: Callable[[], float] | None = None,
    ) -> None:
        """Build a client.

        `client` is injected by tests, which pass an `httpx.AsyncClient` wired to
        a `MockTransport`. That is what makes the parsing, retry and error-mapping
        paths — where the bugs actually live — testable without a GPU or even a
        socket. `jitter` is injected for the same reason: a retry test that
        sleeps for a real random interval is a slow test with a flaky duration.
        """
        self._settings = settings
        self._jitter = jitter
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.timeout_seconds),
            # Keep-alive matters here: the ingestion worker embeds every chunk of
            # every document, so a fresh TCP and TLS handshake per request would
            # be a meaningful fraction of the wall clock.
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )

    async def aclose(self) -> None:
        """Close the connection pool, if we opened it.

        An injected client belongs to whoever injected it; closing it here would
        make one test's teardown break the next one.
        """
        if self._owns_client:
            await self._client.aclose()

    # -- EmbeddingProvider -------------------------------------------------

    async def embed(
        self, texts: Sequence[str], *, mode: EmbedMode = EmbedMode.PASSAGE
    ) -> Sequence[Embedding]:
        """Embed every text, in order.

        Large lists are split across several requests, sequentially. Sending them
        concurrently was rejected: the service serialises on one GPU lock
        anyway, so parallel requests would queue there instead of here while
        holding N times the JSON in memory at both ends.
        """
        if not texts:
            # No round trip for nothing. Reachable: a document whose chunks were
            # all filtered out still reaches this call.
            return []

        batch_size = self._settings.max_texts_per_request
        embeddings: list[Embedding] = []
        for start in range(0, len(texts), batch_size):
            window = list(texts[start : start + batch_size])
            payload = await self._post(
                self._settings.embed_url,
                {"texts": window, "mode": str(mode)},
            )
            parsed = EmbedResponseModel.model_validate(payload)

            # The ordering contract, enforced rather than trusted. If a provider
            # ever returns a different count, every vector after the gap attaches
            # to the wrong chunk — and retrieval keeps working, it just returns
            # unrelated text, which is close to undiagnosable from the outside.
            if len(parsed.embeddings) != len(window):
                raise DependencyUnavailableError(
                    DEPENDENCY_NAME,
                    f"Asked for {len(window)} embeddings and received "
                    f"{len(parsed.embeddings)}; the response cannot be matched to its input.",
                )

            embeddings.extend(
                Embedding(
                    dense=tuple(item.dense),
                    sparse=SparseVector(
                        indices=tuple(item.sparse.indices),
                        values=tuple(item.sparse.values),
                    ),
                )
                for item in parsed.embeddings
            )

        return embeddings

    async def info(self) -> ModelInfo:
        payload = await self._post(self._settings.info_url, None)
        parsed = InfoResponseModel.model_validate(payload)
        return ModelInfo(
            embedding_model=parsed.embedding_model,
            embedding_version=parsed.embedding_version,
            dimensions=parsed.dimensions,
            max_sequence_tokens=parsed.max_sequence_tokens,
            tokenizer_hash=parsed.tokenizer_hash,
            reranker_model=parsed.reranker_model,
        )

    # -- Reranker ----------------------------------------------------------

    async def rerank(
        self, query: str, passages: Sequence[str], *, top_k: int | None = None
    ) -> Sequence[RerankResult]:
        if not passages:
            return []

        body: dict[str, Any] = {"query": query, "passages": list(passages)}
        if top_k is not None:
            body["top_k"] = top_k

        payload = await self._post(self._settings.rerank_url, body)
        parsed = RerankResponseModel.model_validate(payload)
        return [RerankResult(index=item.index, score=item.score) for item in parsed.results]

    # -- health ------------------------------------------------------------

    async def ping(self) -> None:
        """Probe readiness, for the `/ready` health registry.

        Hits the service's own `/ready`, not `/health`: our readiness depends on
        it being able to *serve*, and during its 20-60 second model load it is
        alive but useless. Probing liveness instead would mark us ready while
        every embedding request 503s.

        No retries — a health probe that retries is a health probe that lies
        about latency and hides a flapping dependency.
        """
        url = f"{self._settings.base_url.rstrip('/')}/ready"
        try:
            response = await self._client.get(url, headers=self._headers())
        except httpx.HTTPError as exc:
            raise DependencyUnavailableError(DEPENDENCY_NAME, str(exc)) from exc
        if response.status_code != httpx.codes.OK:
            raise DependencyUnavailableError(
                DEPENDENCY_NAME, f"/ready returned {response.status_code}"
            )

    # -- internals ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if self._settings.api_key is None:
            return {}
        return {"Authorization": f"Bearer {self._settings.api_key.get_secret_value()}"}

    async def _post(self, url: str, body: dict[str, Any] | None) -> Any:
        """Issue a request with bounded retries, returning the decoded body.

        `body is None` means GET — the only such endpoint is `/v1/info`, and
        giving it its own method would duplicate every line of this retry and
        error-mapping logic to save one branch.
        """
        last_error: Exception | None = None

        for attempt in range(self._settings.max_retries + 1):
            if attempt:
                await anyio.sleep(self._backoff(attempt))

            try:
                if body is None:
                    response = await self._client.get(url, headers=self._headers())
                else:
                    response = await self._client.post(url, json=body, headers=self._headers())
            except httpx.HTTPError as exc:
                # Connection refused, DNS failure, read timeout. Transport-level,
                # so there is no status to classify and it is always worth one
                # more try within the budget.
                last_error = exc
                _log.warning(
                    "model_service.request_failed",
                    url=url,
                    attempt=attempt + 1,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue

            if response.status_code == httpx.codes.OK:
                return response.json()

            if response.status_code in _RETRYABLE_STATUS and attempt < self._settings.max_retries:
                last_error = DependencyUnavailableError(
                    DEPENDENCY_NAME, f"{response.status_code} from {url}"
                )
                _log.warning(
                    "model_service.request_retryable",
                    url=url,
                    attempt=attempt + 1,
                    status_code=response.status_code,
                )
                continue

            self._raise_for_status(response, url)

        raise DependencyUnavailableError(
            DEPENDENCY_NAME,
            f"{self._settings.max_retries + 1} attempts against {url} failed: {last_error}",
        ) from last_error

    def _raise_for_status(self, response: httpx.Response, url: str) -> None:
        """Map a non-retryable status onto the error the caller should act on.

        The distinction that matters is *who has to change something*. A 422
        means this chunk is too long and always will be, so the job must be
        dead-lettered now rather than after five identical attempts. A 401 means
        the deployment is misconfigured, which no amount of retrying fixes but
        a redeploy does. A 503 means wait.
        """
        status_code = response.status_code
        detail = _detail_of(response)

        if status_code in _PERMANENT_STATUS:
            raise InvalidInputError(
                f"The model service rejected the request: {detail}",
                details={"status_code": status_code, "dependency": DEPENDENCY_NAME},
            )

        if status_code in {401, 403}:
            raise ConfigurationError(
                f"The model service rejected our credentials ({status_code}). "
                f"Check RAG_MODEL_SERVICE__API_KEY against the service's MODEL_SERVICE_API_KEY.",
            )

        if status_code == httpx.codes.NOT_IMPLEMENTED:
            raise ConfigurationError(
                f"The model service does not support this operation: {detail}",
            )

        raise DependencyUnavailableError(
            DEPENDENCY_NAME,
            f"{status_code} from {url}: {detail}",
            details={"status_code": status_code},
        )

    def _backoff(self, attempt: int) -> float:
        """Jittered exponential delay before retry `attempt` (1-based)."""
        # Annotated because mypy types `int ** int` as `Any` — the exponent could
        # be negative, which would make it a float — and that `Any` would then
        # propagate out through the return value unchecked.
        ceiling: float = self._settings.retry_backoff_seconds * (2 ** (attempt - 1))
        sample = self._jitter() if self._jitter is not None else random.random()  # noqa: S311
        return ceiling * (1 - _JITTER_FRACTION) + ceiling * _JITTER_FRACTION * sample


def _detail_of(response: httpx.Response) -> str:
    """Best-effort human-readable reason from an error response.

    Truncated hard. The model service is contractually forbidden from putting
    request text in an error body, but this string ends up in a job's `error`
    column and in logs — so the bound is enforced on *this* side too rather than
    resting on the other end continuing to behave.
    """
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]

    if isinstance(payload, dict):
        detail = payload.get("detail", payload)
        if isinstance(detail, dict):
            return str(detail.get("message") or detail)[:200]
        return str(detail)[:200]
    return str(payload)[:200]
