"""Per-request ambient context, carried in `contextvars`.

Why this exists
---------------
Two unrelated components need the current request's identity: the *logger*
(so every line is correlatable) and the *error handler* (so a client can quote
a trace id in a support ticket). Threading a `request_id` parameter through
every function signature to serve them is invasive and gets dropped the moment
someone adds a helper.

`contextvars` gives us ambient, async-safe, task-local state instead. Each
asyncio task inherits a copy at creation, so concurrent requests never see each
other's values — the classic thread-local bug does not apply here.

This is deliberately a small, typed surface rather than raw
`structlog.contextvars` usage, because it needs to be readable from
non-logging code too. `core.logging` reads it via a processor; `api.errors`
reads it directly. One source of truth, two consumers.

In M11 the trace id becomes the OpenTelemetry trace id; the accessor signature
below is designed not to change when that happens.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

_request_id: ContextVar[str | None] = ContextVar("rag_request_id", default=None)
_trace_id: ContextVar[str | None] = ContextVar("rag_trace_id", default=None)
_tenant_id: ContextVar[str | None] = ContextVar("rag_tenant_id", default=None)


@dataclass(frozen=True, slots=True)
class ContextTokens:
    """Reset handles returned by `bind_request_context`.

    Held by the caller and passed to `reset_request_context` in a `finally`
    block. Resetting with tokens (rather than setting back to `None`) correctly
    restores any outer value, which matters if contexts ever nest.
    """

    request_id: Token[str | None]
    trace_id: Token[str | None]
    tenant_id: Token[str | None]


def bind_request_context(
    *,
    request_id: str,
    trace_id: str,
    tenant_id: str | None = None,
) -> ContextTokens:
    """Bind identifiers for the current task. Always pair with a reset."""
    return ContextTokens(
        request_id=_request_id.set(request_id),
        trace_id=_trace_id.set(trace_id),
        tenant_id=_tenant_id.set(tenant_id),
    )


def reset_request_context(tokens: ContextTokens) -> None:
    """Restore the values that were in place before `bind_request_context`."""
    _request_id.reset(tokens.request_id)
    _trace_id.reset(tokens.trace_id)
    _tenant_id.reset(tokens.tenant_id)


def set_tenant_id(tenant_id: str | None) -> None:
    """Attach the resolved tenant once authentication has run (M2).

    Separate from `bind_request_context` because the tenant is not known at the
    middleware boundary — it is derived from the verified token, never from a
    request body or header the caller controls.
    """
    _tenant_id.set(tenant_id)


def get_request_id() -> str | None:
    """Identifier for this single HTTP request."""
    return _request_id.get()


def get_trace_id() -> str | None:
    """Identifier for the logical operation, potentially spanning services."""
    return _trace_id.get()


def get_tenant_id() -> str | None:
    """Tenant the current request is acting on behalf of, if authenticated."""
    return _tenant_id.get()


def current_context() -> dict[str, str]:
    """Snapshot of all bound values, omitting unset ones.

    Used by the logging processor and the problem-details builder.
    """
    values = {
        "request_id": _request_id.get(),
        "trace_id": _trace_id.get(),
        "tenant_id": _tenant_id.get(),
    }
    return {key: value for key, value in values.items() if value is not None}
