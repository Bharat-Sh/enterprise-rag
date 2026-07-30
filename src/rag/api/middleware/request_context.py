"""Binds request/trace identifiers for the lifetime of each request.

Why raw ASGI instead of `BaseHTTPMiddleware`
--------------------------------------------
Starlette's `BaseHTTPMiddleware` runs the downstream app inside an anyio task
group and pipes the response through a memory object stream. That has two
consequences we cannot accept:

1. **It interferes with streaming.** In M8 we stream tokens over SSE. Response
   buffering inside a middleware adds latency to time-to-first-token and
   complicates client-disconnect handling — precisely the two things the
   streaming design exists to get right.
2. **It moves the downstream call into a different task.** `contextvars` are
   copied into a new task at creation, so values set by an outer
   `BaseHTTPMiddleware` propagate inward, but values set *inside* the app do
   not propagate back out. Our tenant id is bound during authentication, deep
   in the request — with `BaseHTTPMiddleware` the access log would never see it.

Raw ASGI middleware runs in the same task, so the context is genuinely shared.
The cost is about fifteen extra lines.

Header trust
------------
Inbound `X-Request-ID` / `X-Trace-Id` are accepted so a trace can span services,
but they are caller-controlled and therefore validated against a strict charset
and length. An unvalidated header ends up in log files, response headers, and
eventually a dashboard query — that is a log-injection vector, not a nicety.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders

from rag.core.context import bind_request_context, reset_request_context

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "x-request-id"
TRACE_ID_HEADER = "x-trace-id"

# Keys under `scope["state"]`, readable as `request.state.<key>`.
REQUEST_ID_SCOPE_KEY = "request_id"
TRACE_ID_SCOPE_KEY = "trace_id"

# Conservative: alphanumerics plus a few separators used by common trace formats.
_SAFE_ID_PATTERN = re.compile(r"\A[A-Za-z0-9._:\-]{1,128}\Z")


def _accept_or_generate(candidate: str | None) -> str | None:
    """Return the caller-supplied id if it is safe to propagate, else None."""
    if candidate is not None and _SAFE_ID_PATTERN.match(candidate):
        return candidate
    return None


class RequestContextMiddleware:
    """Bind ids on the way in, echo them on the way out, always unbind."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            # Lifespan and websocket scopes pass straight through.
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id = _accept_or_generate(headers.get(REQUEST_ID_HEADER)) or uuid4().hex
        # A trace spans services; a request id identifies one hop. When the
        # caller supplies neither, one id serves as both.
        trace_id = _accept_or_generate(headers.get(TRACE_ID_HEADER)) or request_id

        tokens = bind_request_context(request_id=request_id, trace_id=trace_id)

        # Also stash the ids on the ASGI scope, reachable downstream as
        # `request.state.trace_id`.
        #
        # This is not redundancy for its own sake. Starlette installs
        # ServerErrorMiddleware as the OUTERMOST layer, outside this one. When an
        # unhandled exception propagates, our `finally` below resets the
        # contextvars *before* that middleware builds the 500 response — so the
        # single response a user is most likely to report would be the only one
        # without a trace id. The scope outlives the contextvars.
        scope.setdefault("state", {})
        scope["state"][REQUEST_ID_SCOPE_KEY] = request_id
        scope["state"][TRACE_ID_SCOPE_KEY] = trace_id

        async def send_with_ids(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                response_headers[REQUEST_ID_HEADER] = request_id
                response_headers[TRACE_ID_HEADER] = trace_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_ids)
        finally:
            # Must run even on exception, or a pooled worker task leaks the
            # previous request's identity into the next one.
            reset_request_context(tokens)
