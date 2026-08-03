"""Exception-to-HTTP mapping and RFC 9457 problem responses.

This module is the *only* place that knows how an error becomes a status code.
Domain exceptions stay framework-free (see `rag.domain.errors`), so the same
errors raised by an ingestion worker or a CLI script carry no HTTP baggage.

Response format is RFC 9457 `application/problem+json`. Using the standard
rather than an ad-hoc `{"error": "..."}` shape means clients get a predictable
envelope and generated SDKs can handle errors generically:

    {
      "type":     "https://docs.enterprise-rag.dev/errors/not_found",
      "title":    "Not Found",
      "status":   404,
      "detail":   "Document 'abc' was not found.",
      "instance": "/api/v1/documents/abc",
      "code":     "not_found",
      "trace_id": "9f2c...",
      "errors":   {...}
    }

`trace_id` is always present. A user reporting a bad answer quotes one string
and we can reconstruct the entire request from logs and traces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag.api.middleware.request_context import (
    REQUEST_ID_HEADER,
    REQUEST_ID_SCOPE_KEY,
    TRACE_ID_HEADER,
    TRACE_ID_SCOPE_KEY,
)
from rag.core.context import get_request_id, get_trace_id
from rag.core.errors import (
    ConfigurationError,
    DependencyUnavailableError,
    InfrastructureError,
    RAGError,
)
from rag.core.logging import get_logger
from rag.domain.errors import (
    AlreadyExistsError,
    ConcurrentModificationError,
    DomainError,
    InvalidInputError,
    InvalidStateTransitionError,
    NotFoundError,
    PermissionDeniedError,
    QuotaExceededError,
)

if TYPE_CHECKING:
    from rag.core.config import Settings

_log = get_logger(__name__)

PROBLEM_CONTENT_TYPE = "application/problem+json"
PROBLEM_TYPE_BASE = "https://docs.enterprise-rag.dev/errors/"

# Ordering does not matter: lookup walks the exception's MRO, so a subclass with
# no explicit entry inherits its parent's status.
_STATUS_BY_ERROR: dict[type[RAGError], int] = {
    InvalidInputError: 400,
    PermissionDeniedError: 403,
    NotFoundError: 404,
    AlreadyExistsError: 409,
    InvalidStateTransitionError: 409,
    ConcurrentModificationError: 409,
    QuotaExceededError: 429,
    DomainError: 400,
    DependencyUnavailableError: 503,
    InfrastructureError: 502,
    ConfigurationError: 500,
}

_TITLES: dict[int, str] = {
    400: "Bad Request",
    403: "Forbidden",
    404: "Not Found",
    409: "Conflict",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
}


def status_for(exc: RAGError) -> int:
    """Resolve an HTTP status by walking the exception's MRO.

    Subclasses added later inherit a sensible status automatically instead of
    silently falling through to 500.
    """
    for klass in type(exc).__mro__:
        if klass in _STATUS_BY_ERROR:
            return _STATUS_BY_ERROR[klass]
    return 500


def correlation_ids(request: Request) -> tuple[str | None, str | None]:
    """Resolve (request_id, trace_id) for this request.

    Prefers the ASGI scope over the contextvars, because on the unhandled-error
    path the contextvars have already been reset by the time this runs — see the
    note in `rag.api.middleware.request_context`.
    """
    state: dict[str, Any] = request.scope.get("state") or {}
    request_id = state.get(REQUEST_ID_SCOPE_KEY) or get_request_id()
    trace_id = state.get(TRACE_ID_SCOPE_KEY) or get_trace_id()
    return request_id, trace_id


def problem_response(
    *,
    status: int,
    code: str,
    detail: str,
    instance: str | None = None,
    extra: dict[str, Any] | None = None,
    request_id: str | None = None,
    trace_id: str | None = None,
) -> JSONResponse:
    """Build an RFC 9457 problem response carrying the correlation ids."""
    body: dict[str, Any] = {
        "type": f"{PROBLEM_TYPE_BASE}{code}",
        "title": _TITLES.get(status, "Error"),
        "status": status,
        "detail": detail,
        "code": code,
    }
    if instance is not None:
        body["instance"] = instance
    if trace_id is not None:
        body["trace_id"] = trace_id
    if extra:
        body.update(extra)

    # Set the ids on the response itself rather than relying on the middleware's
    # `send` wrapper. ServerErrorMiddleware emits the 500 through the raw server
    # `send`, bypassing that wrapper entirely, so a 500 would otherwise carry
    # neither header.
    headers: dict[str, str] = {}
    if request_id is not None:
        headers[REQUEST_ID_HEADER] = request_id
    if trace_id is not None:
        headers[TRACE_ID_HEADER] = trace_id

    return JSONResponse(
        status_code=status,
        content=body,
        media_type=PROBLEM_CONTENT_TYPE,
        headers=headers,
    )


def register_exception_handlers(app: FastAPI, settings: Settings) -> None:
    """Install handlers so every error path produces a problem document."""

    async def handle_rag_error(request: Request, exc: Exception) -> JSONResponse:
        if not isinstance(exc, RAGError):  # pragma: no cover - guaranteed by registration
            raise exc

        status = status_for(exc)
        request_id, trace_id = correlation_ids(request)

        # Ours (5xx) gets a stack trace; theirs (4xx) is an expected outcome.
        if status >= 500:
            _log.error("request.failed", code=exc.code, status_code=status, exc_info=exc)
        else:
            _log.info("request.rejected", code=exc.code, status_code=status, reason=exc.message)

        return problem_response(
            status=status,
            code=exc.code,
            detail=exc.message,
            instance=request.url.path,
            extra={"errors": exc.details} if exc.details else None,
            request_id=request_id,
            trace_id=trace_id,
        )

    async def handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
        if not isinstance(exc, RequestValidationError):  # pragma: no cover
            raise exc

        # Pydantic errors contain a `ctx` that can hold arbitrary objects (and
        # occasionally the offending input). Project to a stable, safe subset.
        errors = [
            {
                "location": list(error.get("loc", ())),
                "message": error.get("msg", ""),
                "type": error.get("type", ""),
            }
            for error in exc.errors()
        ]
        request_id, trace_id = correlation_ids(request)
        return problem_response(
            status=422,
            code="validation_error",
            detail="The request body or parameters failed validation.",
            instance=request.url.path,
            extra={"errors": errors},
            request_id=request_id,
            trace_id=trace_id,
        )

    async def handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
        if not isinstance(exc, StarletteHTTPException):  # pragma: no cover
            raise exc
        # Covers framework-generated 404/405 so *every* response shape matches.
        request_id, trace_id = correlation_ids(request)
        return problem_response(
            status=exc.status_code,
            code=_TITLES.get(exc.status_code, "error").lower().replace(" ", "_"),
            detail=str(exc.detail),
            instance=request.url.path,
            request_id=request_id,
            trace_id=trace_id,
        )

    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # Anything reaching here is a bug. Log everything; disclose nothing.
        # The ids are passed explicitly: this handler runs *after* the request
        # context has been unbound, so the logging processor cannot supply them.
        request_id, trace_id = correlation_ids(request)
        _log.error(
            "request.unhandled_exception",
            path=request.url.path,
            request_id=request_id,
            trace_id=trace_id,
            exc_info=exc,
        )
        detail = (
            f"{type(exc).__name__}: {exc}"
            if settings.expose_error_details
            else "An unexpected error occurred. Quote the trace_id when reporting this."
        )
        return problem_response(
            status=500,
            code="internal_error",
            detail=detail,
            instance=request.url.path,
            request_id=request_id,
            trace_id=trace_id,
        )

    app.add_exception_handler(RAGError, handle_rag_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(Exception, handle_unexpected_error)
