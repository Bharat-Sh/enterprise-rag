"""Reject oversized request bodies **while they are still arriving**.

Why this is middleware and not a check in the upload handler
------------------------------------------------------------
Starlette buffers the entire request body into a `SpooledTemporaryFile` before a
handler runs. By the time an `UploadFile` reaches the endpoint, a 10 GB upload
has already been written to disk — so a size check there protects nothing it was
meant to protect. The limit has to be enforced on the ASGI `receive` channel,
which is the only place the bytes can be counted as they arrive.

`Content-Length` is checked first as a cheap early exit, but it is not trusted:
it is absent under chunked transfer encoding and it is caller-supplied in every
case. The running byte count is what actually enforces the limit.

Applied globally rather than only to the upload route. Middleware runs before
routing, which is normally an argument *against* using it (see
`rag.api.security`) — here it is the point, because an unrouted or mistyped path
should not be a way to stream unlimited data into the process.

Why it builds its own response
------------------------------
Starlette's stack is `ServerErrorMiddleware → user middleware →
ExceptionMiddleware → router`, and the registered exception handlers live in
`ExceptionMiddleware`. Anything added with `add_middleware` therefore sits
*outside* them: raising here produces a 500 from the server error handler, not
the 413 the exception maps to. So this middleware catches its own error and
emits the problem document directly, which also keeps the response shape
identical to every other error in the system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.api.errors import correlation_ids_from_scope, problem_response, status_for
from rag.core.logging import get_logger
from rag.domain.errors import PayloadTooLargeError

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

__all__ = ["BodySizeLimitMiddleware"]

_log = get_logger(__name__)


class BodySizeLimitMiddleware:
    """Cap the number of body bytes any single request may deliver."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _declared_length(scope)
        if declared is not None and declared > self._max_bytes:
            # Cheap rejection: refuse before reading a single byte. Only a hint,
            # since the header is caller-supplied and optional.
            _log.info("request.body_too_large", declared=declared, limit=self._max_bytes)
            await self._reject(scope, receive, send)
            return

        started = False

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, self._counting(receive), tracking_send)
        except PayloadTooLargeError:
            if started:
                # Headers are already on the wire; there is no way to turn this
                # into a 413 now. Let it propagate and be logged as the abort
                # it is, rather than corrupting a response in flight.
                raise
            await self._reject(scope, receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        error = PayloadTooLargeError(limit_bytes=self._max_bytes)
        request_id, trace_id = correlation_ids_from_scope(scope)
        response = problem_response(
            status=status_for(error),
            code=error.code,
            detail=error.message,
            instance=scope.get("path"),
            extra={"errors": error.details},
            request_id=request_id,
            trace_id=trace_id,
        )
        await response(scope, receive, send)

    def _counting(self, receive: Receive) -> Receive:
        received = 0

        async def counted() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_bytes:
                    # Raised from inside `receive`, so it propagates out through
                    # whatever was awaiting the body and lands in the registered
                    # exception handlers as a 413 problem document.
                    _log.info("request.body_too_large", received=received, limit=self._max_bytes)
                    raise PayloadTooLargeError(limit_bytes=self._max_bytes)
            return message

        return counted


def _declared_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", ()):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                # A malformed header is not a reason to reject outright; the
                # running count below is the real enforcement.
                return None
    return None
