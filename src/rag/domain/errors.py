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


class PermissionDeniedError(DomainError):
    """The caller is authenticated but lacks the required permission.

    Only for operations where the resource's existence is already known to the
    caller. For cross-tenant access use `NotFoundError` instead.
    """

    code: ClassVar[str] = "permission_denied"
    default_message: ClassVar[str] = "You do not have permission to perform this action."


class QuotaExceededError(DomainError):
    """A tenant limit was hit: rate, storage, document count, or token budget."""

    code: ClassVar[str] = "quota_exceeded"
    default_message: ClassVar[str] = "A usage quota has been exceeded."


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
