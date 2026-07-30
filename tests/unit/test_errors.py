"""Error mapping and RFC 9457 problem responses."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from rag.api.errors import PROBLEM_CONTENT_TYPE, status_for
from rag.core.errors import ConfigurationError, DependencyUnavailableError, RAGError
from rag.domain.errors import (
    AlreadyExistsError,
    DomainError,
    InvalidInputError,
    InvalidStateTransitionError,
    NotFoundError,
    PermissionDeniedError,
    QuotaExceededError,
)


class TestStatusMapping:
    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (InvalidInputError("bad"), 400),
            (PermissionDeniedError("nope"), 403),
            (NotFoundError("Document", "abc"), 404),
            (AlreadyExistsError("dupe"), 409),
            (InvalidStateTransitionError("Document", "FAILED", "READY"), 409),
            (QuotaExceededError("slow down"), 429),
            (DependencyUnavailableError("qdrant"), 503),
            (ConfigurationError("misconfigured"), 500),
        ],
    )
    def test_known_errors_map_to_their_status(self, error: RAGError, expected: int) -> None:
        assert status_for(error) == expected

    def test_unmapped_subclasses_inherit_from_their_parent(self) -> None:
        # The MRO walk is what makes adding a new domain error safe: it gets a
        # sensible status automatically instead of silently becoming a 500.
        class TenantSuspendedError(DomainError):
            code = "tenant_suspended"

        assert status_for(TenantSuspendedError()) == 400

    def test_a_bare_rag_error_is_a_server_error(self) -> None:
        assert status_for(RAGError()) == 500


class TestErrorConstruction:
    def test_not_found_captures_structured_details(self) -> None:
        error = NotFoundError("Document", "doc-123")

        assert error.code == "not_found"
        assert error.details == {"resource": "Document", "id": "doc-123"}
        assert "doc-123" in error.message

    def test_invalid_state_transition_records_both_states(self) -> None:
        error = InvalidStateTransitionError("Document", "FAILED", "READY")

        assert error.details["current_state"] == "FAILED"
        assert error.details["attempted_transition"] == "READY"

    def test_dependency_unavailable_records_the_dependency(self) -> None:
        error = DependencyUnavailableError("qdrant")

        assert error.dependency == "qdrant"
        assert error.details["dependency"] == "qdrant"


@pytest.fixture
def failing_app(app: FastAPI) -> FastAPI:
    """The real application plus routes that fail in each interesting way."""

    @app.get("/boom/domain")
    async def _domain() -> None:
        raise NotFoundError("Document", "doc-123")

    @app.get("/boom/dependency")
    async def _dependency() -> None:
        raise DependencyUnavailableError("qdrant", "Connection refused.")

    @app.get("/boom/unhandled")
    async def _unhandled() -> None:
        raise RuntimeError("this should never reach the client")

    return app


@pytest.fixture
async def failing_client(failing_app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Client that returns the 500 response instead of re-raising.

    Starlette's ServerErrorMiddleware re-raises after producing the response, so
    without `raise_app_exceptions=False` the unhandled-error test would blow up
    in the test process rather than asserting on the response body.
    """
    async with failing_app.router.lifespan_context(failing_app):
        transport = ASGITransport(app=failing_app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


class TestProblemResponses:
    async def test_domain_error_becomes_a_problem_document(self, failing_client) -> None:
        response = await failing_client.get("/boom/domain")

        assert response.status_code == 404
        assert response.headers["content-type"].startswith(PROBLEM_CONTENT_TYPE)

        body = response.json()
        assert body["status"] == 404
        assert body["title"] == "Not Found"
        assert body["code"] == "not_found"
        assert body["instance"] == "/boom/domain"
        assert body["type"].endswith("/not_found")
        assert body["errors"] == {"resource": "Document", "id": "doc-123"}

    async def test_every_problem_carries_a_trace_id(self, failing_client) -> None:
        # The single most useful field in the whole envelope: a user quotes it,
        # and we reconstruct the request from logs and traces.
        response = await failing_client.get("/boom/domain")

        assert response.json()["trace_id"] == response.headers["x-trace-id"]

    async def test_infrastructure_error_maps_to_503(self, failing_client) -> None:
        response = await failing_client.get("/boom/dependency")

        assert response.status_code == 503
        assert response.json()["code"] == "dependency_unavailable"

    async def test_unhandled_exception_is_reported_generically(self, failing_client) -> None:
        response = await failing_client.get("/boom/unhandled")

        assert response.status_code == 500
        body = response.json()
        assert body["code"] == "internal_error"
        # Local/dev may include the exception text; the internal message must
        # never leak verbatim without the environment opting in.
        assert "trace_id" in body

    async def test_routing_404_uses_the_same_envelope(self, failing_client) -> None:
        # Consistency matters: a client should not need two error parsers.
        response = await failing_client.get("/no-such-route")

        assert response.status_code == 404
        assert response.headers["content-type"].startswith(PROBLEM_CONTENT_TYPE)
        assert response.json()["status"] == 404


class TestProductionDisclosure:
    async def test_internal_details_are_hidden_in_production(self) -> None:
        from rag.api.main import create_app
        from rag.core.config import Environment, Settings

        settings = Settings(
            _env_file=None,
            environment=Environment.PROD,
            log_level="CRITICAL",
            database={"password": "not-the-development-default"},
        )
        prod_app = create_app(settings)

        @prod_app.get("/boom/unhandled")
        async def _unhandled() -> None:
            raise RuntimeError("secret internal detail")

        async with prod_app.router.lifespan_context(prod_app):
            transport = ASGITransport(app=prod_app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/boom/unhandled")

        assert response.status_code == 500
        assert "secret internal detail" not in response.text
        assert "trace_id" in response.json()
