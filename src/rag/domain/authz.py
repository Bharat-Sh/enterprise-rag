"""What each role may do.

The policy is a **table**, not code scattered across handlers. That is the whole
point: "which endpoints can a VIEWER reach?" has to be answerable by reading one
mapping rather than by grepping every router, and it has to be testable without
standing up HTTP.

Roles form a **total order** — OWNER > ADMIN > MEMBER > VIEWER — so a permission
is expressed as the least privileged role that holds it. This is honest for the
four roles we have, which really are nested, and it has a real limit: it cannot
express "a billing owner who cannot read documents". When the first non-nested
role appears, `permits()` is the single function that changes, and the table
becomes `dict[Role, frozenset[Permission]]`.

Roles are deliberately not ACLs. A role governs *actions* ("may this user
create users?"); an ACL governs *resources* ("may this user read that
document?"). See `rag.domain.access` for the other half.
"""

from __future__ import annotations

from enum import StrEnum

from rag.domain.enums import Role

__all__ = [
    "MINIMUM_ROLE",
    "ROLE_RANK",
    "Permission",
    "less_privileged_of",
    "permits",
]


class Permission(StrEnum):
    """A named action the API can gate on.

    Named after the *action*, not the role, so changing who may do something is
    one row in `MINIMUM_ROLE` rather than an edit at every call site.
    """

    #: Manage one's own API keys. Every authenticated role holds this: a key can
    #: only ever narrow its owner's authority (docs/adr/0007), so issuing one
    #: grants nothing the caller did not already have.
    API_KEY_MANAGE = "api_key:manage"

    USER_LIST = "user:list"
    USER_CREATE = "user:create"
    #: Role assignment is an owner power — an admin who can promote themselves
    #: to owner is an owner with extra steps.
    USER_ASSIGN_ROLE = "user:assign_role"
    USER_SET_STATUS = "user:set_status"

    COLLECTION_READ = "collection:read"
    COLLECTION_CREATE = "collection:create"

    DOCUMENT_READ = "document:read"
    DOCUMENT_UPLOAD = "document:upload"
    #: Admin rather than "the member who uploaded it". Per-document ownership
    #: needs an owner check on every path and buys little while every member of
    #: a tenant can already read the same documents; if it is wanted later it is
    #: one predicate, and guessing now would cost more than it saves.
    DOCUMENT_DELETE = "document:delete"
    #: Granting access to a document is a different power from uploading one.
    DOCUMENT_MANAGE_ACL = "document:manage_acl"


#: Higher is more privileged. Spaced by ten so a role can be inserted between
#: two existing ones without renumbering.
ROLE_RANK: dict[Role, int] = {
    Role.VIEWER: 0,
    Role.MEMBER: 10,
    Role.ADMIN: 20,
    Role.OWNER: 30,
}

#: The least privileged role that holds each permission.
MINIMUM_ROLE: dict[Permission, Role] = {
    Permission.API_KEY_MANAGE: Role.VIEWER,
    Permission.USER_LIST: Role.ADMIN,
    Permission.USER_CREATE: Role.ADMIN,
    Permission.USER_ASSIGN_ROLE: Role.OWNER,
    Permission.USER_SET_STATUS: Role.ADMIN,
    Permission.COLLECTION_READ: Role.VIEWER,
    Permission.COLLECTION_CREATE: Role.ADMIN,
    # A viewer queries; a member contributes. That is the line the role names
    # already draw, so upload is the first permission a viewer does not hold.
    Permission.DOCUMENT_READ: Role.VIEWER,
    Permission.DOCUMENT_UPLOAD: Role.MEMBER,
    Permission.DOCUMENT_DELETE: Role.ADMIN,
    Permission.DOCUMENT_MANAGE_ACL: Role.ADMIN,
}


def _assert_table_is_total() -> None:
    """Fail at import if a permission has no entry.

    A missing entry would otherwise surface as a `KeyError` inside a request —
    a 500 on an authorization check, which is the one place a crash is worse
    than a denial. Enforcing it at import makes it a startup failure instead.
    """
    missing = sorted(set(Permission) - set(MINIMUM_ROLE))
    if missing:  # pragma: no cover - the assertion is the point, not the branch
        raise RuntimeError(f"Permissions with no minimum role: {missing}")
    unranked = sorted(set(Role) - set(ROLE_RANK))
    if unranked:  # pragma: no cover
        raise RuntimeError(f"Roles with no rank: {unranked}")


_assert_table_is_total()


def permits(role: Role, permission: Permission) -> bool:
    """Whether `role` holds `permission`."""
    return ROLE_RANK[role] >= ROLE_RANK[MINIMUM_ROLE[permission]]


def less_privileged_of(left: Role, right: Role) -> Role:
    """The weaker of two roles.

    Used to apply an API key's role *ceiling*: a key issued by an admin who is
    later demoted to member must not keep admin authority, and a key deliberately
    scoped to viewer must not widen when its owner is promoted. Narrowing in both
    directions is the only safe reading of "ceiling".
    """
    return left if ROLE_RANK[left] <= ROLE_RANK[right] else right
