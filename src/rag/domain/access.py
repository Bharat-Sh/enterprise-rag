"""Access control primitives.

The whole design follows from one downstream constraint: in M5 the access check
must run **inside the vector store query**, as a pre-filter. Qdrant payload
filters can do `match`, `match_any`, and boolean combination. They cannot join,
subquery, or expand a group hierarchy.

So an ACL has to arrive already flattened into an array of opaque strings.

The structure that makes this work is splitting on *what changes when*:

    stored on the document (stable)    ["user:a1", "group:eng", "tenant:t1"]
    computed per request (volatile)    ["user:a1", "group:eng", "role:admin",
                                        "tenant:t1"]

Access is granted when the two sets intersect. In Postgres that is the array
overlap operator `&&` (GIN-indexable); in Qdrant it is `match_any`. The same
decision, evaluated identically in both stores — not merely similarly, which is
how the two drift apart and a leak appears.

**Why this split and not the obvious alternative.** Putting expanded *user* ids
on each document would mean that adding one person to a group rewrites the ACL
of every document that group can see, and re-indexes all of them. Here,
membership changes only the caller's principal set, computed fresh each request.
Zero re-indexing.

The cost is that a revoked group membership takes effect only on the next
request — which is correct behaviour anyway — and that principal sets must be
recomputed per request or cached with a short TTL (M9).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self
from uuid import UUID

from rag.domain.enums import PrincipalType, Role

if TYPE_CHECKING:
    from rag.domain.credentials import CredentialKind
    from rag.domain.models import User

__all__ = ["TENANT_WIDE", "AccessFilter", "AuthenticatedPrincipal", "Principal"]

_SEPARATOR = ":"


@dataclass(frozen=True, slots=True)
class Principal:
    """A subject a permission can be granted to.

    Serialised as `"<type>:<id>"`. The token is deliberately opaque and flat:
    it is the only representation a vector-store payload filter can evaluate.
    """

    type: PrincipalType
    id: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Principal id must not be empty")
        if _SEPARATOR in self.id:
            # Otherwise "user:a:b" would parse back ambiguously and two distinct
            # principals could collide into the same token.
            raise ValueError(f"Principal id must not contain {_SEPARATOR!r}: {self.id!r}")

    @property
    def token(self) -> str:
        """Flat wire form, e.g. `"group:3f9c..."`."""
        return f"{self.type.value}{_SEPARATOR}{self.id}"

    @classmethod
    def parse(cls, token: str) -> Self:
        """Reconstruct a principal from its token form."""
        kind, separator, identifier = token.partition(_SEPARATOR)
        if not separator:
            raise ValueError(f"Malformed principal token: {token!r}")
        return cls(type=PrincipalType(kind), id=identifier)

    @classmethod
    def user(cls, user_id: UUID) -> Self:
        return cls(type=PrincipalType.USER, id=str(user_id))

    @classmethod
    def group(cls, group_id: UUID) -> Self:
        return cls(type=PrincipalType.GROUP, id=str(group_id))

    @classmethod
    def role(cls, role: Role) -> Self:
        return cls(type=PrincipalType.ROLE, id=role.value)

    @classmethod
    def tenant(cls, tenant_id: UUID) -> Self:
        return cls(type=PrincipalType.TENANT, id=str(tenant_id))


def TENANT_WIDE(tenant_id: UUID) -> Principal:  # noqa: N802 - reads as a constructor
    """The principal granting access to every member of a tenant.

    Attaching this to a document is how "shared with the whole organisation" is
    expressed without enumerating members.
    """
    return Principal.tenant(tenant_id)


@dataclass(frozen=True, slots=True)
class AccessFilter:
    """Everything needed to decide what one caller may retrieve.

    Built once per request from the *verified* token — never from a request
    body, query parameter, or header. Passed unchanged into both the SQL
    repositories and (from M5) the vector store, so there is exactly one
    definition of "what may this caller see".
    """

    tenant_id: UUID
    principal_tokens: tuple[str, ...]

    @classmethod
    def build(
        cls,
        *,
        tenant_id: UUID,
        user_id: UUID,
        role: Role,
        group_ids: Iterable[UUID] = (),
    ) -> Self:
        """Assemble a caller's principal set.

        Always includes the tenant-wide principal, so documents shared with the
        whole organisation are visible without enumerating every member.
        """
        principals = [
            Principal.user(user_id),
            Principal.role(role),
            Principal.tenant(tenant_id),
            *(Principal.group(group_id) for group_id in group_ids),
        ]
        # Sorted and de-duplicated so the value is deterministic — it becomes a
        # cache key in M9, and an unstable ordering would silently halve the
        # hit rate.
        return cls(
            tenant_id=tenant_id,
            principal_tokens=tuple(sorted({principal.token for principal in principals})),
        )

    def permits(self, acl_principals: Iterable[str]) -> bool:
        """Whether this caller may read a resource carrying `acl_principals`.

        The in-memory equivalent of Postgres `&&` and Qdrant `match_any`. Used
        in unit tests and defence-in-depth assertions; the real check always
        happens in the store, before rows or vectors are read.

        A bare intersection, with no special case for an empty principal set.
        An earlier version returned True when `principal_tokens` was empty,
        which meant a directly constructed `AccessFilter` — the shape every test
        fake and future system context produces — silently permitted
        *everything*. `build()` can never produce an empty set, so the special
        case protected nothing and only ever fired in the fail-open direction.
        """
        return bool(set(acl_principals) & set(self.principal_tokens))


@dataclass(frozen=True, slots=True)
class AuthenticatedPrincipal:
    """Who is making this request, resolved from a verified credential.

    Request-scoped rather than persisted, which is why it lives here beside
    `AccessFilter` and not in `rag.domain.models`.

    `effective_role` is not necessarily `user.role`. An API key carries a
    ceiling, and the ceiling has to be applied *before* the access filter is
    built — otherwise a key deliberately scoped down to viewer still carries the
    `role:admin` principal and matches admin-granted document ACLs. Building the
    filter from this object rather than from the user is what keeps the two
    halves of the authorization decision consistent.
    """

    user: User
    tenant_id: UUID
    effective_role: Role
    access: AccessFilter
    credential: CredentialKind
    #: Set only when the request authenticated with an API key. Used for audit
    #: logging and to refuse account-level changes to a machine credential.
    api_key_id: UUID | None = None

    @property
    def is_api_key(self) -> bool:
        return self.api_key_id is not None
