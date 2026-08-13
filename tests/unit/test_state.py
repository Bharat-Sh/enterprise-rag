"""The document ingestion state machine."""

from __future__ import annotations

import pytest

from rag.domain.enums import DocumentStatus as S
from rag.domain.errors import InvalidStateTransitionError
from rag.domain.state import (
    ALLOWED_TRANSITIONS,
    assert_can_transition,
    can_transition,
    next_stage,
)


class TestHappyPath:
    def test_the_pipeline_runs_end_to_end(self) -> None:
        current = S.UPLOADED
        for target in (S.QUEUED, S.PARSING, S.CHUNKING, S.EMBEDDING, S.INDEXING, S.READY):
            assert can_transition(current, target), f"{current} -> {target}"
            current = target
        assert current.is_terminal

    def test_next_stage_walks_the_pipeline(self) -> None:
        assert next_stage(S.QUEUED) is S.PARSING
        assert next_stage(S.INDEXING) is S.READY

    def test_next_stage_is_none_at_the_end(self) -> None:
        assert next_stage(S.READY) is None

    def test_next_stage_is_none_off_the_pipeline(self) -> None:
        assert next_stage(S.FAILED) is None


class TestIllegalTransitions:
    @pytest.mark.parametrize(
        ("current", "target"),
        [
            (S.UPLOADED, S.READY),  # skipping the entire pipeline
            (S.PARSING, S.INDEXING),  # skipping chunking and embedding
            (S.READY, S.PARSING),  # reprocessing without going through REINDEXING
            (S.DELETED, S.QUEUED),  # resurrecting a tombstone
            (S.FAILED, S.READY),  # declaring success after failure
        ],
    )
    def test_rejected(self, current: S, target: S) -> None:
        assert not can_transition(current, target)
        with pytest.raises(InvalidStateTransitionError):
            assert_can_transition(current, target)

    def test_the_error_carries_both_states(self) -> None:
        # An operator reading the log needs to know what it *was*, not only
        # what was attempted.
        with pytest.raises(InvalidStateTransitionError) as caught:
            assert_can_transition(S.READY, S.PARSING, document_id="doc-1")

        assert caught.value.details["current_state"] == "ready"
        assert caught.value.details["attempted_transition"] == "parsing"
        assert caught.value.details["document_id"] == "doc-1"
        assert caught.value.code == "invalid_state_transition"


class TestRecoveryPaths:
    def test_a_failed_document_can_be_requeued(self) -> None:
        # Operators do this after fixing the cause — a parser bug, an exhausted
        # quota. Without it, every transient failure is permanent.
        assert can_transition(S.FAILED, S.QUEUED)

    def test_every_in_flight_stage_can_fail(self) -> None:
        for state in S:
            if state.is_in_flight:
                assert can_transition(state, S.FAILED), f"{state} cannot fail"

    def test_every_in_flight_stage_can_be_requeued_or_is_terminal(self) -> None:
        # A worker that dies mid-stage leaves the document stranded unless the
        # reaper can push it back to QUEUED.
        for state in (S.PARSING, S.CHUNKING, S.EMBEDDING, S.INDEXING):
            assert can_transition(state, S.QUEUED), f"{state} cannot be requeued"

    def test_a_ready_document_can_be_rebuilt(self) -> None:
        assert can_transition(S.READY, S.REINDEXING)
        assert can_transition(S.REINDEXING, S.READY)


class TestReadyImpliesIndexed:
    """The M3 shortcut is gone, and must not come back.

    `TestTemporaryEdgeForM3` used to live here asserting that `CHUNKING -> READY`
    was *present*, so that removing it would be a deliberate act with a failing
    test to update rather than something nobody remembered to do. M5 removed the
    edge; this class replaces it, asserting the opposite.

    Why it matters enough to keep a test after the fact: a document that reaches
    `READY` without vectors is invisible to retrieval while claiming to be
    searchable, and nothing errors anywhere. Now that the only route to `READY`
    runs through `EMBEDDING` and `INDEXING`, "ready" means "there are vectors"
    by construction. Re-adding the shortcut for a quick fix would silently
    reintroduce the worst failure shape in the system.
    """

    def test_chunking_cannot_shortcut_to_ready(self) -> None:
        assert not can_transition(S.CHUNKING, S.READY), (
            "CHUNKING -> READY is back. A document can now reach READY with no "
            "vectors, which makes it unfindable while reporting itself as "
            "searchable. Route through EMBEDDING and INDEXING instead."
        )

    def test_the_only_route_to_ready_runs_through_indexing(self) -> None:
        for state in S:
            if state is not S.READY and can_transition(state, S.READY):
                assert state in {S.INDEXING, S.REINDEXING}, (
                    f"{state} can reach READY without indexing"
                )

    def test_the_pipeline_route_survives(self) -> None:
        assert can_transition(S.CHUNKING, S.EMBEDDING)
        assert can_transition(S.EMBEDDING, S.INDEXING)
        assert can_transition(S.INDEXING, S.READY)


class TestInvariants:
    def test_every_status_has_an_entry(self) -> None:
        # A status missing from the table is silently un-transitionable, which
        # looks like a mysteriously stuck document rather than a bug.
        assert set(ALLOWED_TRANSITIONS) == set(S)

    def test_deleted_is_absorbing(self) -> None:
        assert ALLOWED_TRANSITIONS[S.DELETED] == frozenset()

    def test_everything_can_reach_deletion_except_the_tombstone(self) -> None:
        # A user asking for erasure must not be blocked by whatever state their
        # document happens to be in.
        for state in S:
            if state is S.DELETED:
                continue
            reachable = ALLOWED_TRANSITIONS[state]
            assert S.DELETING in reachable or S.DELETED in reachable, f"{state} cannot be deleted"

    def test_terminal_and_in_flight_are_disjoint(self) -> None:
        for state in S:
            assert not (state.is_terminal and state.is_in_flight), state
