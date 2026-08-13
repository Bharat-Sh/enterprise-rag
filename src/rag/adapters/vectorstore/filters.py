"""Translating an `AccessFilter` into a Qdrant payload filter.

Its own module, and a pure function, because this is the single place in the
system where a mistake is a **cross-tenant disclosure** rather than a bug.

Postgres protects us from ourselves: row-level security means a query that
forgets its tenant scope returns zero rows (docs/adr/0005). Qdrant has no such
thing. A query that forgets its tenant clause returns every tenant's vectors,
successfully, with a 200. There is no database to catch it, so the only
available defence is that the clause is impossible to omit — hence one function
that always emits it, no code path that builds a filter another way, and a
security test that asserts the tenant clause is present in the constructed
object rather than inferring it from behaviour. Behaviour looks right until the
one query where it doesn't.

Why `must` and not `should`
---------------------------
Qdrant's `must` is conjunction, `should` is disjunction. The tenant clause and
the ACL clause are both mandatory and must be ANDed; putting either in `should`
would make a match on the *other* sufficient. That is a one-word edit away from
"any principal token grants access to every tenant".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qdrant_client import models

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from rag.domain.access import AccessFilter

__all__ = [
    "ACL_PRINCIPALS_FIELD",
    "COLLECTION_ID_FIELD",
    "DOCUMENT_ID_FIELD",
    "ORDINAL_FIELD",
    "TENANT_ID_FIELD",
    "build_search_filter",
    "build_tenant_document_filter",
]

#: Payload field names. Constants rather than literals at each call site: a typo
#: in a filter key does not error in Qdrant, it silently matches nothing — or,
#: on the tenant field, silently matches everything.
TENANT_ID_FIELD = "tenant_id"
ACL_PRINCIPALS_FIELD = "acl_principals"
DOCUMENT_ID_FIELD = "document_id"
COLLECTION_ID_FIELD = "collection_id"
ORDINAL_FIELD = "ordinal"


def _tenant_clause(tenant_id: UUID) -> models.FieldCondition:
    """The clause every single query must carry."""
    return models.FieldCondition(key=TENANT_ID_FIELD, match=models.MatchValue(value=str(tenant_id)))


def build_search_filter(
    access: AccessFilter,
    *,
    collection_id: UUID | None = None,
    document_ids: Sequence[UUID] | None = None,
) -> models.Filter:
    """The pre-filter for a search, built from the caller's verified identity.

    Takes an `AccessFilter` positionally and required, so there is no way to
    call this without one. It deliberately accepts no caller-supplied filter to
    merge in: the narrowing arguments are typed and specific, which is what
    stops "let the API pass through a filter" from ever becoming reasonable.

    Two mandatory clauses, ANDed:

    1. `tenant_id` matches exactly. Isolation.
    2. `acl_principals` overlaps the caller's principal set. Authorization —
       the `match_any` half of docs/adr/0006, whose Postgres twin is `&&`.

    An `AccessFilter` from `build()` always contains at least the user, role and
    tenant principals, so the ACL clause can never be an empty `MatchAny` — and
    an empty `MatchAny` is a filter that matches nothing, which fails closed
    anyway. Both directions are safe, and the test suite pins it.
    """
    conditions: list[models.Condition] = [
        _tenant_clause(access.tenant_id),
        models.FieldCondition(
            key=ACL_PRINCIPALS_FIELD,
            match=models.MatchAny(any=list(access.principal_tokens)),
        ),
    ]

    # Narrowing, never widening. These can only remove candidates from a set the
    # two clauses above have already bounded, so a bug here costs recall rather
    # than isolation — which is why they are safe to expose to a caller at all.
    if collection_id is not None:
        conditions.append(
            models.FieldCondition(
                key=COLLECTION_ID_FIELD, match=models.MatchValue(value=str(collection_id))
            )
        )
    if document_ids:
        conditions.append(
            models.FieldCondition(
                key=DOCUMENT_ID_FIELD,
                match=models.MatchAny(any=[str(document_id) for document_id in document_ids]),
            )
        )

    return models.Filter(must=conditions)


def build_tenant_document_filter(tenant_id: UUID, document_id: UUID) -> models.Filter:
    """Every point of one document, for deletion and reindexing.

    Carries the tenant clause even though `document_id` is a UUID and therefore
    already unique. Defence in depth costs one condition here, and the habit is
    the point: a deletion path that omits the tenant is one copy-paste away from
    a search path that does.
    """
    return models.Filter(
        must=[
            _tenant_clause(tenant_id),
            models.FieldCondition(
                key=DOCUMENT_ID_FIELD, match=models.MatchValue(value=str(document_id))
            ),
        ]
    )
