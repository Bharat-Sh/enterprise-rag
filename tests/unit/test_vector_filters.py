"""Translating an `AccessFilter` into a Qdrant filter.

Exhaustive, and asserting on the **constructed object** rather than on observed
behaviour. That distinction is the whole point of this file: a behavioural test
("tenant B's data does not come back") passes for a filter that happens to
exclude it and for a filter that excludes it by accident, and it can only ever
cover the cases someone thought to write. Reading the structure proves the
tenant clause is there for *every* input, including the ones nobody imagined.

Postgres would catch a missing tenant scope with row-level security. Qdrant will
not: a query with no tenant clause returns every tenant's vectors, with a 200.
This module is the only thing standing in that gap.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from qdrant_client import models

from rag.adapters.vectorstore.filters import (
    ACL_PRINCIPALS_FIELD,
    COLLECTION_ID_FIELD,
    DOCUMENT_ID_FIELD,
    TENANT_ID_FIELD,
    build_search_filter,
    build_tenant_document_filter,
)
from rag.domain.access import AccessFilter
from rag.domain.enums import Role

TENANT = UUID("11111111-1111-1111-1111-111111111111")
OTHER_TENANT = UUID("22222222-2222-2222-2222-222222222222")


def _access(*, tenant_id: UUID = TENANT, groups: tuple[UUID, ...] = ()) -> AccessFilter:
    return AccessFilter.build(
        tenant_id=tenant_id, user_id=uuid4(), role=Role.MEMBER, group_ids=groups
    )


def _conditions(built: models.Filter) -> list[models.FieldCondition]:
    assert built.must is not None, "filter has no `must` clause at all"
    return [c for c in built.must if isinstance(c, models.FieldCondition)]


def _condition_for(built: models.Filter, key: str) -> models.FieldCondition:
    matching = [c for c in _conditions(built) if c.key == key]
    assert len(matching) == 1, f"expected exactly one {key!r} condition, got {len(matching)}"
    return matching[0]


class TestTheTenantClauseIsAlwaysPresent:
    """The single property that isolation rests on."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"collection_id": uuid4()},
            {"document_ids": [uuid4()]},
            {"document_ids": []},
            {"collection_id": uuid4(), "document_ids": [uuid4(), uuid4()]},
            {"collection_id": None, "document_ids": None},
        ],
        ids=["bare", "collection", "documents", "empty-documents", "both", "explicit-none"],
    )
    def test_for_every_combination_of_narrowing(self, kwargs: dict) -> None:
        built = build_search_filter(_access(), **kwargs)

        condition = _condition_for(built, TENANT_ID_FIELD)
        assert condition.match == models.MatchValue(value=str(TENANT))

    def test_it_carries_the_filters_own_tenant_not_some_other(self) -> None:
        # Guards against the clause being built from an ambient value — a
        # settings default, a contextvar — rather than from the caller's own
        # verified identity.
        built = build_search_filter(_access(tenant_id=OTHER_TENANT))

        condition = _condition_for(built, TENANT_ID_FIELD)
        assert condition.match == models.MatchValue(value=str(OTHER_TENANT))

    def test_deletion_carries_it_too(self) -> None:
        built = build_tenant_document_filter(TENANT, uuid4())

        condition = _condition_for(built, TENANT_ID_FIELD)
        assert condition.match == models.MatchValue(value=str(TENANT))


class TestClausesAreConjoined:
    def test_everything_is_in_must_and_nothing_in_should(self) -> None:
        # `should` is disjunction. Moving either mandatory clause there would
        # make a match on the *other* sufficient — one word away from "any
        # principal token grants access to every tenant".
        built = build_search_filter(_access(), collection_id=uuid4(), document_ids=[uuid4()])

        assert built.should is None
        assert built.must_not is None
        assert built.must is not None and len(built.must) == 4

    def test_the_acl_clause_is_a_match_any_over_the_callers_principals(self) -> None:
        groups = (uuid4(), uuid4())
        access = _access(groups=groups)

        built = build_search_filter(access)

        condition = _condition_for(built, ACL_PRINCIPALS_FIELD)
        assert isinstance(condition.match, models.MatchAny)
        assert set(condition.match.any) == set(access.principal_tokens)

    def test_the_acl_clause_is_never_empty_for_a_built_filter(self) -> None:
        # `AccessFilter.build` always contributes user, role and tenant
        # principals, so this cannot be empty. Pinned because an empty
        # `MatchAny` would be a filter nobody reading the code would expect —
        # and because the failure direction matters: empty matches *nothing*,
        # so even that degenerate case fails closed.
        built = build_search_filter(_access())

        condition = _condition_for(built, ACL_PRINCIPALS_FIELD)
        assert isinstance(condition.match, models.MatchAny)
        assert condition.match.any


class TestNarrowingOnlyNarrows:
    def test_no_narrowing_means_exactly_two_clauses(self) -> None:
        built = build_search_filter(_access())

        assert {c.key for c in _conditions(built)} == {TENANT_ID_FIELD, ACL_PRINCIPALS_FIELD}

    def test_a_collection_adds_a_clause(self) -> None:
        collection_id = uuid4()

        built = build_search_filter(_access(), collection_id=collection_id)

        condition = _condition_for(built, COLLECTION_ID_FIELD)
        assert condition.match == models.MatchValue(value=str(collection_id))

    def test_document_ids_add_a_match_any(self) -> None:
        ids = [uuid4(), uuid4()]

        built = build_search_filter(_access(), document_ids=ids)

        condition = _condition_for(built, DOCUMENT_ID_FIELD)
        assert isinstance(condition.match, models.MatchAny)
        assert set(condition.match.any) == {str(i) for i in ids}

    def test_an_empty_document_list_adds_nothing(self) -> None:
        # `[]` means "no narrowing requested", not "match no documents". The
        # opposite reading would turn an empty list into a search that silently
        # returns nothing, which looks identical to a permission problem.
        built = build_search_filter(_access(), document_ids=[])

        assert DOCUMENT_ID_FIELD not in {c.key for c in _conditions(built)}


class TestValuesAreSerialisedAsStrings:
    def test_uuids_become_strings_everywhere(self) -> None:
        # Qdrant payload values are compared by type as well as value: a `UUID`
        # object would not match a payload written as a string, and the failure
        # is an empty result set rather than an error.
        collection_id, document_id = uuid4(), uuid4()

        built = build_search_filter(
            _access(), collection_id=collection_id, document_ids=[document_id]
        )

        for condition in _conditions(built):
            match = condition.match
            values = match.any if isinstance(match, models.MatchAny) else [match.value]  # type: ignore[union-attr]
            for value in values:
                assert isinstance(value, str), f"{condition.key} carries a non-string {value!r}"
