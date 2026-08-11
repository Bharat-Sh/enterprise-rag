"""The HTTP client for the model service.

Driven entirely through `httpx.MockTransport`, so there is no server, no socket
and no GPU — and every branch that matters is reachable, including the ones a
real service would only produce during an outage. Retry behaviour, status
mapping and the response-ordering guard are where the bugs in a client like this
actually live; the happy path is the easy part.
"""

from __future__ import annotations

import httpx
import pytest

from rag.adapters.models.http import HttpModelClient
from rag.core.config import ModelServiceSettings
from rag.core.errors import ConfigurationError, DependencyUnavailableError
from rag.domain.embedding import EmbedMode
from rag.domain.errors import InvalidInputError
from rag.domain.ports import EmbeddingProvider, Reranker


def _embedding(dense: list[float] | None = None) -> dict:
    return {
        "dense": dense if dense is not None else [0.0, 1.0],
        "sparse": {"indices": [7, 19], "values": [0.9, 0.1]},
    }


def _embed_body(count: int) -> dict:
    return {
        "embedding_model": "bge-m3",
        "embedding_version": "rev1",
        "dimensions": 2,
        "embeddings": [_embedding() for _ in range(count)],
    }


def _client(
    handler,
    *,
    max_retries: int = 2,
    max_texts_per_request: int = 32,
    api_key: str | None = None,
) -> HttpModelClient:
    settings = ModelServiceSettings(
        base_url="http://model-service:8001",
        max_retries=max_retries,
        retry_backoff_seconds=0.001,
        max_texts_per_request=max_texts_per_request,
        api_key=api_key,
    )
    transport = httpx.MockTransport(handler)
    return HttpModelClient(
        settings,
        client=httpx.AsyncClient(transport=transport),
        # Pinned: a retry test that sleeps for a real random interval is a slow
        # test with a duration nobody can reason about.
        jitter=lambda: 0.0,
    )


class TestItSatisfiesItsPorts:
    def test_structurally(self) -> None:
        client = _client(lambda request: httpx.Response(200, json={}))

        assert isinstance(client, EmbeddingProvider)
        assert isinstance(client, Reranker)


class TestEmbed:
    async def test_it_parses_dense_and_sparse(self) -> None:
        client = _client(lambda request: httpx.Response(200, json=_embed_body(1)))

        [embedding] = await client.embed(["hello"])

        assert embedding.dense == (0.0, 1.0)
        assert embedding.sparse.indices == (7, 19)
        assert embedding.sparse.values == (0.9, 0.1)

    async def test_it_sends_the_mode(self) -> None:
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            seen.append(json.loads(request.content))
            return httpx.Response(200, json=_embed_body(1))

        await _client(handler).embed(["q"], mode=EmbedMode.QUERY)

        assert seen[0]["mode"] == "query"

    async def test_an_empty_list_makes_no_request(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("A round trip for nothing.")

        assert await _client(handler).embed([]) == []

    async def test_large_lists_are_split_across_requests(self) -> None:
        sizes: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            count = len(json.loads(request.content)["texts"])
            sizes.append(count)
            return httpx.Response(200, json=_embed_body(count))

        client = _client(handler, max_texts_per_request=2)
        embeddings = await client.embed(["a", "b", "c", "d", "e"])

        assert sizes == [2, 2, 1]
        # Split or not, the caller gets one result per input, in order.
        assert len(embeddings) == 5

    async def test_a_short_response_is_refused(self) -> None:
        # The ordering contract, enforced rather than trusted. A response with
        # fewer embeddings than texts would attach every vector after the gap to
        # the wrong chunk — and retrieval keeps working, it just returns
        # unrelated text, which is close to undiagnosable from the outside.
        client = _client(lambda request: httpx.Response(200, json=_embed_body(2)))

        with pytest.raises(DependencyUnavailableError, match="cannot be matched"):
            await client.embed(["a", "b", "c"])


class TestRerank:
    async def test_it_parses_scores(self) -> None:
        body = {
            "reranker_model": "bge-reranker-v2-m3",
            "results": [{"index": 2, "score": 4.5}, {"index": 0, "score": -1.0}],
        }
        client = _client(lambda request: httpx.Response(200, json=body))

        results = await client.rerank("q", ["a", "b", "c"])

        assert [result.index for result in results] == [2, 0]
        assert results[0].score == 4.5

    async def test_top_k_is_only_sent_when_given(self) -> None:
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"reranker_model": "r", "results": []})

        client = _client(handler)
        await client.rerank("q", ["a"])
        await client.rerank("q", ["a"], top_k=1)

        assert "top_k" not in seen[0]
        assert seen[1]["top_k"] == 1

    async def test_no_passages_makes_no_request(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("A round trip for nothing.")

        assert await _client(handler).rerank("q", []) == []


class TestRetries:
    async def test_a_transport_error_is_retried(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("connection refused")
            return httpx.Response(200, json=_embed_body(1))

        await _client(handler).embed(["x"])

        assert attempts == 2

    async def test_a_503_is_retried(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                return httpx.Response(503, json={"detail": "loading"})
            return httpx.Response(200, json=_embed_body(1))

        await _client(handler).embed(["x"])

        assert attempts == 3

    async def test_retries_are_bounded(self) -> None:
        # Bounded shallowly on purpose: the job queue retries with exponential
        # backoff at a far coarser grain, so deep retries here multiply into a
        # worker held for minutes on a service that is simply down.
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(503)

        with pytest.raises(DependencyUnavailableError):
            await _client(handler, max_retries=2).embed(["x"])

        assert attempts == 3

    async def test_a_permanent_rejection_is_not_retried(self) -> None:
        # A chunk that is too long is too long on every attempt. Retrying it
        # burns five queue attempts to reach the same answer.
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(422, json={"detail": {"message": "input_too_long"}})

        with pytest.raises(InvalidInputError):
            await _client(handler).embed(["x"])

        assert attempts == 1

    async def test_zero_retries_means_one_attempt(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(500)

        with pytest.raises(DependencyUnavailableError):
            await _client(handler, max_retries=0).embed(["x"])

        assert attempts == 1


class TestErrorMapping:
    """The distinction that matters is *who has to change something*."""

    async def test_a_rejected_request_is_a_domain_error(self) -> None:
        # `InvalidInputError` is a `DomainError`, which the worker dead-letters
        # immediately rather than retrying. That routing is the entire point.
        client = _client(lambda request: httpx.Response(413, json={"detail": "too many"}))

        with pytest.raises(InvalidInputError) as caught:
            await client.embed(["x"])

        assert caught.value.details["status_code"] == 413

    async def test_bad_credentials_are_a_configuration_error(self) -> None:
        client = _client(lambda request: httpx.Response(401))

        with pytest.raises(ConfigurationError, match="API_KEY"):
            await client.embed(["x"])

    async def test_an_unsupported_operation_is_a_configuration_error(self) -> None:
        # 501 means the service was started with the reranker disabled. No
        # amount of retrying fixes it; a redeploy does.
        client = _client(
            lambda request: httpx.Response(501, json={"detail": {"message": "disabled"}})
        )

        with pytest.raises(ConfigurationError):
            await client.rerank("q", ["p"])

    async def test_an_outage_is_a_dependency_error(self) -> None:
        client = _client(lambda request: httpx.Response(502))

        with pytest.raises(DependencyUnavailableError) as caught:
            await client.embed(["x"])

        assert caught.value.dependency == "model-service"

    async def test_it_never_fails_open(self) -> None:
        # Unlike the rate limiter. There is no degraded embedding: returning
        # placeholder vectors would index a document that is unfindable while
        # claiming to be searchable, with nothing raised anywhere.
        client = _client(lambda request: httpx.Response(500))

        with pytest.raises(DependencyUnavailableError):
            await client.embed(["x"])

    async def test_error_detail_is_truncated(self) -> None:
        # This string lands in a job's `error` column and in logs. The service
        # is contractually forbidden from putting request text in an error body,
        # but the bound is enforced here too rather than resting on that.
        client = _client(lambda request: httpx.Response(500, text="x" * 5000))

        with pytest.raises(DependencyUnavailableError) as caught:
            await client.embed(["x"])

        assert len(caught.value.message) < 500


class TestAuthentication:
    async def test_no_key_configured_sends_no_header(self) -> None:
        seen: list[httpx.Headers] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers)
            return httpx.Response(200, json=_embed_body(1))

        await _client(handler).embed(["x"])

        assert "authorization" not in seen[0]

    async def test_a_configured_key_is_sent_as_a_bearer_token(self) -> None:
        seen: list[httpx.Headers] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers)
            return httpx.Response(200, json=_embed_body(1))

        await _client(handler, api_key="s3cret").embed(["x"])

        assert seen[0]["authorization"] == "Bearer s3cret"


class TestInfo:
    async def test_it_parses_model_identity(self) -> None:
        body = {
            "embedding_model": "bge-m3",
            "embedding_version": "rev-abc",
            "dimensions": 1024,
            "max_sequence_tokens": 1024,
            "tokenizer_hash": "0123456789abcdef",
            "reranker_model": "bge-reranker-v2-m3",
            "backend": "flag",
        }
        client = _client(lambda request: httpx.Response(200, json=body))

        info = await client.info()

        assert info.embedding_model == "bge-m3"
        assert info.embedding_version == "rev-abc"
        assert info.tokenizer_hash == "0123456789abcdef"

    async def test_a_missing_reranker_field_still_parses(self) -> None:
        # Independent deploys: this end must keep working against a service
        # older or differently configured than itself.
        body = {
            "embedding_model": "bge-m3",
            "embedding_version": "rev-abc",
            "dimensions": 1024,
            "max_sequence_tokens": 1024,
            "tokenizer_hash": "0123456789abcdef",
        }
        client = _client(lambda request: httpx.Response(200, json=body))

        info = await client.info()

        assert info.reranker_model is None

    async def test_unknown_fields_are_ignored(self) -> None:
        body = {
            "embedding_model": "bge-m3",
            "embedding_version": "rev-abc",
            "dimensions": 1024,
            "max_sequence_tokens": 1024,
            "tokenizer_hash": "0123456789abcdef",
            "something_added_in_m7": {"nested": True},
        }
        client = _client(lambda request: httpx.Response(200, json=body))

        assert (await client.info()).dimensions == 1024


class TestPing:
    async def test_it_probes_readiness_not_liveness(self) -> None:
        # Our readiness depends on the service being able to *serve*. During its
        # 20-60 second model load it is alive and useless, and probing /health
        # would mark us ready while every embedding request 503s.
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            return httpx.Response(200, json={"ready": True})

        await _client(handler).ping()

        assert seen == ["/ready"]

    async def test_it_raises_when_not_ready(self) -> None:
        client = _client(lambda request: httpx.Response(503))

        with pytest.raises(DependencyUnavailableError):
            await client.ping()

    async def test_it_does_not_retry(self) -> None:
        # A health probe that retries lies about latency and hides a flapping
        # dependency behind an average.
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(503)

        with pytest.raises(DependencyUnavailableError):
            await _client(handler, max_retries=3).ping()

        assert attempts == 1


class TestLifecycle:
    async def test_it_does_not_close_a_client_it_was_given(self) -> None:
        # An injected client belongs to whoever injected it. Closing it here
        # would make one test's teardown break the next one.
        injected = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        client = HttpModelClient(ModelServiceSettings(), client=injected)

        await client.aclose()

        assert not injected.is_closed
        await injected.aclose()
