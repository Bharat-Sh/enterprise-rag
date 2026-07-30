"""Access logging and request timing.

We emit our own access log rather than using uvicorn's (which is silenced in
`rag.core.logging`) so that every line is structured and carries the request
context bound by `RequestContextMiddleware`.

Note on `path`: we log the raw path here. For Prometheus metrics in M11 we will
switch to the matched *route template* (`/api/v1/documents/{document_id}`)
instead — a label whose cardinality grows with the number of documents will melt
the metrics backend.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from starlette.datastructures import MutableHeaders

from rag.core.logging import get_logger

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

_log = get_logger(__name__)

RESPONSE_TIME_HEADER = "x-response-time-ms"

# Paths we do not want in the access log: orchestrator probes fire every few
# seconds and would otherwise dominate the volume.
_QUIET_PATHS = frozenset({"/health", "/ready"})


class TimingMiddleware:
    """Measure wall-clock duration, log the outcome, expose it as a header."""

    def __init__(self, app: ASGIApp, *, slow_request_ms: float = 1000.0) -> None:
        self.app = app
        self.slow_request_ms = slow_request_ms

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = 500  # assume failure until the app tells us otherwise

        async def send_with_timing(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                elapsed_ms = (time.perf_counter() - start) * 1000
                headers = MutableHeaders(scope=message)
                headers[RESPONSE_TIME_HEADER] = f"{elapsed_ms:.2f}"
            await send(message)

        try:
            await self.app(scope, receive, send_with_timing)
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            path = scope.get("path", "")
            if path not in _QUIET_PATHS:
                # Client errors are the caller's problem; server errors are ours.
                if status_code >= 500:
                    level = "error"
                elif status_code >= 400 or duration_ms > self.slow_request_ms:
                    level = "warning"
                else:
                    level = "info"

                getattr(_log, level)(
                    "http.request",
                    method=scope.get("method", ""),
                    path=path,
                    status_code=status_code,
                    duration_ms=duration_ms,
                )
