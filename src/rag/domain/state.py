"""The document ingestion state machine.

Kept in the domain layer, as pure functions over enums, because the rules are
business rules — the same transitions are enforced by the ingestion worker, the
admin API, and the reindex script, none of which share a transport or a storage
concern.

Encoding transitions as data rather than scattered `if` statements gives us
three things: the graph is inspectable (and rendered in the docs), an illegal
transition raises loudly instead of corrupting the index, and adding a stage in
a later milestone is a one-line table edit rather than an audit of every caller.
"""

from __future__ import annotations

from rag.domain.enums import DocumentStatus as S
from rag.domain.errors import InvalidStateTransitionError

__all__ = ["ALLOWED_TRANSITIONS", "assert_can_transition", "can_transition", "next_stage"]

#: Legal successors for each state. Anything absent is rejected.
ALLOWED_TRANSITIONS: dict[S, frozenset[S]] = {
    S.UPLOADED: frozenset({S.QUEUED, S.FAILED, S.DELETING}),
    S.QUEUED: frozenset({S.PARSING, S.FAILED, S.DELETING}),
    S.PARSING: frozenset({S.CHUNKING, S.QUEUED, S.FAILED, S.DELETING}),
    # `CHUNKING -> READY` was a temporary M3 edge and was **removed in M5**,
    # together with `TestTemporaryEdgeForM3`, which existed to assert it was
    # present so that deleting it could not be quietly forgotten.
    #
    # The reason it had to go: once an index exists, a document reaching READY
    # without vectors is invisible to retrieval while claiming to be searchable,
    # and nothing errors. The only route to READY now runs through EMBEDDING and
    # INDEXING, so "ready" means "there are vectors" by construction rather than
    # by convention.
    S.CHUNKING: frozenset({S.EMBEDDING, S.QUEUED, S.FAILED, S.DELETING}),
    S.EMBEDDING: frozenset({S.INDEXING, S.QUEUED, S.FAILED, S.DELETING}),
    S.INDEXING: frozenset({S.READY, S.QUEUED, S.FAILED, S.DELETING}),
    # Terminal-but-revivable: a READY document can be rebuilt or removed.
    S.READY: frozenset({S.REINDEXING, S.DELETING}),
    # A failure is recoverable by requeueing; operators do this after fixing
    # the cause (a parser bug, an exhausted quota).
    S.FAILED: frozenset({S.QUEUED, S.DELETING}),
    S.REINDEXING: frozenset({S.READY, S.FAILED, S.DELETING}),
    S.DELETING: frozenset({S.DELETED, S.FAILED}),
    # A tombstone is final. Re-uploading the same content creates a new row;
    # resurrecting a deleted id would silently restore data a user asked us to
    # remove, which is a compliance problem, not just a modelling one.
    S.DELETED: frozenset(),
}

#: The happy path, used to derive the next stage during ingestion.
_PIPELINE: tuple[S, ...] = (
    S.QUEUED,
    S.PARSING,
    S.CHUNKING,
    S.EMBEDDING,
    S.INDEXING,
    S.READY,
)


def can_transition(current: S, target: S) -> bool:
    """Whether `current -> target` is legal."""
    return target in ALLOWED_TRANSITIONS.get(current, frozenset())


def assert_can_transition(current: S, target: S, *, document_id: str | None = None) -> None:
    """Raise `InvalidStateTransitionError` unless the transition is legal.

    Called before every status write. Two workers racing on the same document
    means the loser attempts a transition from a state that no longer holds, and
    fails loudly here rather than overwriting the winner's progress.
    """
    if not can_transition(current, target):
        raise InvalidStateTransitionError(
            entity="Document",
            current_state=current.value,
            attempted=target.value,
            details={"document_id": document_id} if document_id else None,
        )


def next_stage(current: S) -> S | None:
    """The next state on the happy path, or `None` if there isn't one."""
    try:
        index = _PIPELINE.index(current)
    except ValueError:
        return None
    return _PIPELINE[index + 1] if index + 1 < len(_PIPELINE) else None
