"""Principals and access filters."""

from __future__ import annotations

from uuid import UUID

import pytest

from rag.core.ids import uuid7
from rag.domain.access import AccessFilter, Principal
from rag.domain.enums import PrincipalType, Role

TENANT = UUID("11111111-1111-7111-8111-111111111111")
ALICE = UUID("22222222-2222-7222-8222-222222222222")
BOB = UUID("33333333-3333-7333-8333-333333333333")
ENGINEERING = UUID("44444444-4444-7444-8444-444444444444")


class TestPrincipalTokens:
    def test_round_trips(self) -> None:
        original = Principal.group(ENGINEERING)

        assert Principal.parse(original.token) == original

    def test_token_format_is_type_colon_id(self) -> None:
        assert Principal.user(ALICE).token == f"user:{ALICE}"
        assert Principal.role(Role.ADMIN).token == "role:admin"  # noqa: S105 - an ACL token
        assert Principal.tenant(TENANT).token == f"tenant:{TENANT}"

    def test_ids_containing_the_separator_are_rejected(self) -> None:
        # Otherwise "user:a:b" parses ambiguously and two distinct principals
        # could collide onto one token — which is a silent grant.
        with pytest.raises(ValueError, match="must not contain"):
            Principal(type=PrincipalType.USER, id="a:b")

    def test_empty_ids_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            Principal(type=PrincipalType.USER, id="")

    def test_malformed_tokens_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="Malformed"):
            Principal.parse("no-separator-here")


class TestAccessFilterConstruction:
    def test_includes_user_role_and_tenant(self) -> None:
        access = AccessFilter.build(tenant_id=TENANT, user_id=ALICE, role=Role.MEMBER)

        assert f"user:{ALICE}" in access.principal_tokens
        assert "role:member" in access.principal_tokens
        assert f"tenant:{TENANT}" in access.principal_tokens

    def test_includes_every_group(self) -> None:
        access = AccessFilter.build(
            tenant_id=TENANT, user_id=ALICE, role=Role.MEMBER, group_ids=[ENGINEERING]
        )

        assert f"group:{ENGINEERING}" in access.principal_tokens

    def test_tokens_are_sorted_and_deduplicated(self) -> None:
        # The filter becomes a cache key in M9. An unstable ordering would
        # silently halve the hit rate without any test failing.
        access = AccessFilter.build(
            tenant_id=TENANT,
            user_id=ALICE,
            role=Role.MEMBER,
            group_ids=[ENGINEERING, ENGINEERING],
        )

        assert list(access.principal_tokens) == sorted(access.principal_tokens)
        assert len(set(access.principal_tokens)) == len(access.principal_tokens)

    def test_is_deterministic(self) -> None:
        build = lambda: AccessFilter.build(  # noqa: E731
            tenant_id=TENANT, user_id=ALICE, role=Role.ADMIN, group_ids=[ENGINEERING]
        )

        assert build() == build()


class TestPermissionDecisions:
    def test_direct_user_grant(self) -> None:
        access = AccessFilter.build(tenant_id=TENANT, user_id=ALICE, role=Role.VIEWER)

        assert access.permits([f"user:{ALICE}"])

    def test_group_grant(self) -> None:
        access = AccessFilter.build(
            tenant_id=TENANT, user_id=ALICE, role=Role.VIEWER, group_ids=[ENGINEERING]
        )

        assert access.permits([f"group:{ENGINEERING}"])

    def test_tenant_wide_grant_reaches_every_member(self) -> None:
        access = AccessFilter.build(tenant_id=TENANT, user_id=BOB, role=Role.VIEWER)

        assert access.permits([f"tenant:{TENANT}"])

    def test_another_users_grant_does_not(self) -> None:
        access = AccessFilter.build(tenant_id=TENANT, user_id=ALICE, role=Role.VIEWER)

        assert not access.permits([f"user:{BOB}"])

    def test_a_group_you_are_not_in_does_not(self) -> None:
        access = AccessFilter.build(tenant_id=TENANT, user_id=ALICE, role=Role.VIEWER)

        assert not access.permits([f"group:{ENGINEERING}"])

    def test_an_empty_acl_grants_nobody(self) -> None:
        # A document with no grants is invisible, not public. Getting this
        # backwards would make every unshared upload world-readable.
        access = AccessFilter.build(tenant_id=TENANT, user_id=ALICE, role=Role.OWNER)

        assert not access.permits([])

    def test_being_owner_does_not_bypass_document_acls(self) -> None:
        # Roles govern *actions*, ACLs govern *resources*. An owner who can
        # manage billing still cannot read an HR document they were not granted.
        access = AccessFilter.build(tenant_id=TENANT, user_id=ALICE, role=Role.OWNER)

        assert not access.permits([f"user:{BOB}"])

    def test_generated_ids_behave_the_same(self) -> None:
        user_id, group_id = uuid7(), uuid7()
        access = AccessFilter.build(
            tenant_id=TENANT, user_id=user_id, role=Role.MEMBER, group_ids=[group_id]
        )

        assert access.permits([f"group:{group_id}"])
        assert not access.permits([f"group:{uuid7()}"])
