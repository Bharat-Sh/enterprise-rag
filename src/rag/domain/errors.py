"""Domain errors: things that went wrong according to *business rules*.

Contrast with `rag.core.errors.InfrastructureError`, which covers things that
went wrong according to the *machines*. The distinction drives real behaviour:
domain errors are the caller's fault and must not be retried; infrastructure
errors are usually transient and should be.

None of these know about HTTP. `rag.api.errors` owns that mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from rag.core.errors import RAGError


class DomainError(RAGError):
    """A business rule was violated. Retrying an identical request will not help."""

    code: ClassVar[str] = "domain_error"
    default_message: ClassVar[str] = "The request violates a business rule."


class NotFoundError(DomainError):
    """A referenced entity does not exist, or the caller may not see it.

    Deliberately conflates "absent" and "forbidden". Returning 403 for a
    resource in another tenant confirms that resource exists, which is an
    enumeration oracle. Multi-tenant lookups therefore raise this, never
    `PermissionDeniedError` — see M2.
    """

    code: ClassVar[str] = "not_found"
    default_message: ClassVar[str] = "The requested resource was not found."

    def __init__(
        self,
        resource: str,
        identifier: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = {"resource": resource}
        if identifier is not None:
            merged["id"] = identifier
        merged.update(details or {})
        suffix = f" {identifier!r}" if identifier else ""
        super().__init__(f"{resource}{suffix} was not found.", details=merged)
        self.resource = resource
        self.identifier = identifier


class AlreadyExistsError(DomainError):
    """A uniqueness constraint would be violated.

    In ingestion this is usually benign: the same content hash arriving twice
    means an idempotent re-upload, which we short-circuit rather than reject.
    """

    code: ClassVar[str] = "already_exists"
    default_message: ClassVar[str] = "The resource already exists."


class InvalidInputError(DomainError):
    """Structurally valid input that is semantically wrong.

    Pydantic rejects malformed input before we ever see it. This is for rules
    Pydantic cannot express: "chunk overlap must be smaller than chunk size",
    "this file type is not enabled for your tenant".
    """

    code: ClassVar[str] = "invalid_input"
    default_message: ClassVar[str] = "The request was not valid."


class AuthenticationError(DomainError):
    """The caller did not prove who they are.

    401, not 403: 401 means "I do not know who you are", 403 means "I know and
    you may not". Conflating them makes correct client behaviour impossible —
    only one of the two is fixed by presenting a different credential.

    Deliberately carries **no detail**. Unknown tenant, unknown user, wrong
    password, expired token, revoked API key and malformed credential all
    produce this exact error with this exact message. Every distinction we
    could draw for a legitimate caller's convenience is a distinction an
    attacker uses to enumerate: "wrong password" confirms the account exists.
    """

    code: ClassVar[str] = "unauthenticated"
    default_message: ClassVar[str] = "Authentication failed."


class PermissionDeniedError(DomainError):
    """The caller is authenticated but lacks the required permission.

    Only for operations where the resource's existence is already known to the
    caller. For cross-tenant access use `NotFoundError` instead.
    """

    code: ClassVar[str] = "permission_denied"
    default_message: ClassVar[str] = "You do not have permission to perform this action."


class PayloadTooLargeError(DomainError):
    """The request body exceeded the configured limit.

    Raised from the ASGI layer *while the body is still arriving*, not after it
    has been received. Checking afterwards means the bytes are already on disk,
    so the limit protects nothing it was meant to protect.
    """

    code: ClassVar[str] = "payload_too_large"
    default_message: ClassVar[str] = "The request body is too large."

    def __init__(self, *, limit_bytes: int, details: Mapping[str, Any] | None = None) -> None:
        merged: dict[str, Any] = {"limit_bytes": limit_bytes}
        merged.update(details or {})
        super().__init__(
            f"The request body exceeds the limit of {limit_bytes} bytes.", details=merged
        )
        self.limit_bytes = limit_bytes


class UnsupportedMediaTypeError(DomainError):
    """The content is not a format this system can ingest.

    Carries the type we *detected*, which is deliberately not the type the
    caller declared — see `rag.domain.sniff`. Telling someone their "PDF" was
    detected as a zip is the single most useful thing this error can say.
    """

    code: ClassVar[str] = "unsupported_media_type"
    default_message: ClassVar[str] = "The file type is not supported."

    def __init__(
        self,
        message: str | None = None,
        *,
        detected: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = {}
        if detected is not None:
            merged["detected_content_type"] = detected
        merged.update(details or {})
        super().__init__(message, details=merged)
        self.detected = detected


class QuotaExceededError(DomainError):
    """A tenant limit was hit: rate, storage, document count, or token budget."""

    code: ClassVar[str] = "quota_exceeded"
    default_message: ClassVar[str] = "A usage quota has been exceeded."


class RateLimitExceededError(QuotaExceededError):
    """The caller is sending requests faster than their allowance.

    A subclass rather than a reuse of `QuotaExceededError`, for the same reason
    the two conflict errors are distinct: both are 429, but "slow down and retry
    in four seconds" and "you have used your document allowance for the month"
    demand completely different client behaviour. The status is inherited
    through the MRO walk in `rag.api.errors.status_for`, so no mapping changes.
    """

    code: ClassVar[str] = "rate_limit_exceeded"
    default_message: ClassVar[str] = "Too many requests."

    def __init__(
        self,
        *,
        retry_after_seconds: int,
        limit: int,
        scope: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = {
            "retry_after_seconds": retry_after_seconds,
            "limit": limit,
            "scope": scope,
        }
        merged.update(details or {})
        super().__init__(
            f"Rate limit of {limit} requests per minute exceeded; retry in {retry_after_seconds}s.",
            details=merged,
        )
        self.retry_after_seconds = retry_after_seconds
        self.limit = limit
        self.scope = scope


class ConcurrentModificationError(DomainError):
    """The entity changed between the caller's read and its write.

    Deliberately distinct from `InvalidStateTransitionError`. That one means the
    request was wrong — a move the state machine forbids from any starting
    point. This one means the request was fine but arrived second: the move may
    be perfectly legal, it simply started from a state that no longer holds.

    The distinction is not academic. A client seeing this should re-read and
    retry, and a retry will usually succeed. A client seeing an invalid
    transition should not retry at all, because it will fail identically for
    ever. Collapsing them into one code makes correct client behaviour
    impossible to write.
    """

    code: ClassVar[str] = "concurrent_modification"
    default_message: ClassVar[str] = "The resource was modified concurrently."

    def __init__(
        self,
        entity: str,
        *,
        expected: str,
        actual: str,
        identifier: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = {
            "entity": entity,
            "expected_state": expected,
            "actual_state": actual,
            "hint": "re-read the resource and retry",
        }
        if identifier is not None:
            merged["id"] = identifier
        merged.update(details or {})
        super().__init__(
            f"{entity} was expected in state {expected!r} but is now {actual!r}; "
            f"another operation modified it first.",
            details=merged,
        )
        self.entity = entity
        self.expected = expected
        self.actual = actual


class InvalidStateTransitionError(DomainError):
    """An entity was asked to move to a state it cannot reach from its current one.

    The document lifecycle (UPLOADED -> ... -> READY) is an explicit state
    machine; this is what it raises when something tries to skip a step. Makes
    concurrency bugs loud instead of silently corrupting the index.
    """

    code: ClassVar[str] = "invalid_state_transition"
    default_message: ClassVar[str] = "The entity is not in a state that allows this operation."

    def __init__(
        self,
        entity: str,
        current_state: str,
        attempted: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = {
            "entity": entity,
            "current_state": current_state,
            "attempted_transition": attempted,
        }
        merged.update(details or {})
        super().__init__(
            f"{entity} in state {current_state!r} cannot transition to {attempted!r}.",
            details=merged,
        )
        self.entity = entity
        self.current_state = current_state
        self.attempted = attempted
