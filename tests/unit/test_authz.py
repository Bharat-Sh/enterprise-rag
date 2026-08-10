"""The role/permission table.

Exhaustive rather than sampled: the table is small, and a permission whose
minimum role drifts is a privilege escalation that no other test would notice.
"""

from __future__ import annotations

import itertools

import pytest

from rag.domain.authz import (
    MINIMUM_ROLE,
    ROLE_RANK,
    Permission,
    less_privileged_of,
    permits,
)
from rag.domain.enums import Role


class TestTableIsTotal:
    def test_every_permission_has_a_minimum_role(self) -> None:
        # Enforced at import too, so this is the regression guard for someone
        # removing that check along with the permission they forgot.
        assert set(MINIMUM_ROLE) == set(Permission)

    def test_every_role_has_a_rank(self) -> None:
        assert set(ROLE_RANK) == set(Role)

    def test_ranks_are_a_strict_total_order(self) -> None:
        assert ROLE_RANK[Role.VIEWER] < ROLE_RANK[Role.MEMBER]
        assert ROLE_RANK[Role.MEMBER] < ROLE_RANK[Role.ADMIN]
        assert ROLE_RANK[Role.ADMIN] < ROLE_RANK[Role.OWNER]


class TestPermits:
    @pytest.mark.parametrize(("role", "permission"), list(itertools.product(Role, Permission)))
    def test_a_role_holds_exactly_the_permissions_at_or_below_its_rank(
        self, role: Role, permission: Permission
    ) -> None:
        expected = ROLE_RANK[role] >= ROLE_RANK[MINIMUM_ROLE[permission]]

        assert permits(role, permission) is expected

    def test_owner_holds_everything(self) -> None:
        assert all(permits(Role.OWNER, permission) for permission in Permission)

    def test_a_viewer_cannot_manage_users(self) -> None:
        assert permits(Role.VIEWER, Permission.USER_LIST) is False
        assert permits(Role.VIEWER, Permission.USER_CREATE) is False

    def test_a_viewer_can_manage_their_own_api_keys(self) -> None:
        # A key can only narrow its owner's authority, so issuing one grants
        # nothing the caller did not already have.
        assert permits(Role.VIEWER, Permission.API_KEY_MANAGE) is True

    def test_only_an_owner_may_assign_roles(self) -> None:
        # An admin who can grant roles can grant themselves owner, which makes
        # the distinction between the two decorative.
        assert permits(Role.ADMIN, Permission.USER_ASSIGN_ROLE) is False
        assert permits(Role.OWNER, Permission.USER_ASSIGN_ROLE) is True


class TestLessPrivilegedOf:
    @pytest.mark.parametrize(("left", "right"), list(itertools.product(Role, Role)))
    def test_it_returns_the_weaker_of_the_two(self, left: Role, right: Role) -> None:
        result = less_privileged_of(left, right)

        assert ROLE_RANK[result] == min(ROLE_RANK[left], ROLE_RANK[right])

    @pytest.mark.parametrize(("left", "right"), list(itertools.product(Role, Role)))
    def test_it_is_commutative(self, left: Role, right: Role) -> None:
        # The API-key ceiling relies on this: it must not matter whether the
        # user's role or the key's role is passed first.
        assert less_privileged_of(left, right) is less_privileged_of(right, left)

    def test_a_key_ceiling_narrows_but_never_widens(self) -> None:
        assert less_privileged_of(Role.ADMIN, Role.VIEWER) is Role.VIEWER
        # A viewer holding a key nominally scoped to admin stays a viewer.
        assert less_privileged_of(Role.VIEWER, Role.ADMIN) is Role.VIEWER
