"""Base exception hierarchy.

Design note: these errors carry a stable, machine-readable `code` but **no HTTP
status**. Mapping an error to a status code is a delivery concern and lives in
`rag.api.errors`.

That split is not pedantry. The import-linter contract in pyproject.toml
requires `rag.domain` to be free of delivery concerns, and an `http_status`
attribute on a domain exception is exactly the first crack in that wall — the
same errors are raised by ingestion workers and CLI scripts, which have no
notion of HTTP at all.

`code` is part of the public API contract. Clients branch on it, so treat a
rename as a breaking change; the human-readable `message` is free to change.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar


class RAGError(Exception):
    """Root of every error this system raises deliberately.

    Anything *not* descending from this is an unexpected failure and is logged
    with a full traceback and reported to the client as a generic 500.
    """

    code: ClassVar[str] = "internal_error"
    default_message: ClassVar[str] = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.message = message or self.default_message
        # Structured, client-safe context: {"document_id": "...", "limit": 100}.
        # Never put secrets or internal identifiers here — it is serialised
        # into the HTTP response body.
        self.details: dict[str, Any] = dict(details or {})
        super().__init__(self.message)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


class InfrastructureError(RAGError):
    """A backing service failed. Usually transient; usually retryable."""

    code: ClassVar[str] = "infrastructure_error"
    default_message: ClassVar[str] = "A downstream dependency failed."


class DependencyUnavailableError(InfrastructureError):
    """A required dependency is unreachable or unhealthy.

    Maps to 503 so callers and load balancers back off rather than retrying
    immediately into a service that is already struggling.
    """

    code: ClassVar[str] = "dependency_unavailable"
    default_message: ClassVar[str] = "A required dependency is unavailable."

    def __init__(
        self,
        dependency: str,
        message: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = {"dependency": dependency}
        merged.update(details or {})
        super().__init__(message or f"Dependency {dependency!r} is unavailable.", details=merged)
        self.dependency = dependency


class ConfigurationError(RAGError):
    """The process is misconfigured. Not recoverable at runtime — fail at boot."""

    code: ClassVar[str] = "configuration_error"
    default_message: ClassVar[str] = "The service is misconfigured."
