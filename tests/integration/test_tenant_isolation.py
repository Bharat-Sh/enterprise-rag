"""Tenant isolation, enforced by PostgreSQL row-level security.

The most important tests in the repository. Everything else is a feature; a
failure here is a data breach.

Note what these tests deliberately do **not** do: they do not call application
code with a correct filter and check it works. They issue raw SQL with *no*
`WHERE` clause at all — simulating the exact bug we are defending against, a
developer who forgot — and assert the database returns nothing anyway.

A test that only exercises the happy path proves the filter works when you
remember it. The whole point of RLS is what happens when you don't.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from rag.db.models import TENANT_SCOPED_TABLES
from rag.db.session import current_tenant_scope
from rag.domain.enums import Role, UserStatus
from tests.integration.conftest import requires_postgres

if TYPE_CHECKING:
    from rag.db.uow import SqlAlchemyUnitOfWork
    from rag.domain.models import Tenant, User

pytestmark = [requires_postgres, pytest.mark.integration]


class TestThePreconditionEverythingElseRestsOn:
    """The connecting role must not be able to bypass row-level security.

    Every other test in this file is meaningless if it can. A **superuser
    bypasses RLS unconditionally** — `FORCE ROW LEVEL SECURITY` does not apply
    to it — so a suite run as one would exercise the policies against a
    connection that ignores them, and report nothing unusual beyond a pile of
    confusing failures.

    That is exactly what happened: CI set `POSTGRES_USER: rag`, which the
    postgres image makes the bootstrap superuser, and the isolation tests failed
    there while passing locally for six weeks. This test turns that into one
    legible line naming the cause.
    """

    async def test_the_connecting_role_cannot_bypass_rls(self, uow: SqlAlchemyUnitOfWork) -> None:
        result = await uow.session.execute(
            text("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
        )
        is_superuser, can_bypass = result.one()

        assert not is_superuser, (
            "The test database role is a SUPERUSER, which bypasses row-level "
            "security unconditionally. Every isolation test below is vacuous. "
            "Connect as an ordinary role that owns its tables — see the service "
            "container setup in .github/workflows/ci.yml."
        )
        assert not can_bypass, (
            "The test database role holds BYPASSRLS. Same consequence: the "
            "policies are inert for this connection."
        )

    async def test_the_role_still_owns_its_tables(self, uow: SqlAlchemyUnitOfWork) -> None:
        """Non-superuser is necessary but not sufficient — ownership matters too.

        The application connects as the owner of its tables, which is precisely
        why `FORCE` is required (docs/adr/0005). If the role were *not* the
        owner, the tests would pass for the wrong reason and would stop
        exercising FORCE at all.
        """
        result = await uow.session.execute(
            text("SELECT tableowner = current_user FROM pg_tables WHERE tablename = 'users'")
        )

        assert result.scalar_one() is True, (
            "The test role does not own `users`, so these tests no longer prove "
            "that FORCE ROW LEVEL SECURITY holds the owner."
        )


class TestRowLevelSecurityReads:
    async def test_raw_sql_cannot_see_another_tenants_rows(
        self,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        other_tenant: Tenant,
        user: User,
    ) -> None:
        """The core guarantee.

        `user` belongs to `tenant`. Scoped to `other_tenant`, an unfiltered
        `SELECT * FROM users` must still return nothing — no application code
        involved, nothing to forget.
        """
        await uow.scope_to_tenant(other_tenant.id)

        result = await uow.session.execute(text("SELECT count(*) FROM users"))

        assert result.scalar_one() == 0

    async def test_the_row_is_visible_to_its_own_tenant(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant, user: User
    ) -> None:
        # The mirror of the test above: proves the previous result is isolation
        # working, not a fixture that silently failed to insert anything.
        result = await uow.session.execute(text("SELECT count(*) FROM users"))

        assert result.scalar_one() == 1

    async def test_an_unscoped_session_sees_nothing(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant, user: User
    ) -> None:
        """Fail closed.

        This is the property that justified RLS over application-side filtering
        alone. Forgetting to scope returns *zero* rows. With a `tenant_id`
        column and a forgotten `WHERE`, the same mistake returns *every* row.
        """
        await uow.scope_to_tenant(None)

        result = await uow.session.execute(text("SELECT count(*) FROM users"))

        assert result.scalar_one() == 0

    async def test_scope_is_readable_back(self, uow: SqlAlchemyUnitOfWork, tenant: Tenant) -> None:
        assert await current_tenant_scope(uow.session) == tenant.id

    async def test_scope_clears_to_none(self, uow: SqlAlchemyUnitOfWork, tenant: Tenant) -> None:
        await uow.scope_to_tenant(None)

        assert await current_tenant_scope(uow.session) is None

    @pytest.mark.parametrize("table", TENANT_SCOPED_TABLES)
    async def test_every_tenant_scoped_table_is_protected(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant, table: str
    ) -> None:
        """Catches the table added in a later milestone without a policy.

        A new tenant-scoped table is exactly the kind of thing that gets shipped
        with its RLS forgotten, because everything appears to work.

        Parametrised from `TENANT_SCOPED_TABLES` rather than a list written out
        here. A hand-maintained copy drifts silently — M2 added `api_keys` and
        `refresh_tokens` and this test would have kept passing without them.
        Deriving it also checks the coupling that matters: the migration keeps
        its *own* copy of the list on purpose, so this asserts the two agree.
        """
        await uow.scope_to_tenant(None)

        result = await uow.session.execute(
            text(f"SELECT count(*) FROM {table}")  # noqa: S608 - parametrised from a fixed list
        )

        assert result.scalar_one() == 0

    @pytest.mark.parametrize("table", TENANT_SCOPED_TABLES)
    async def test_policies_are_forced_not_merely_enabled(
        self, uow: SqlAlchemyUnitOfWork, table: str
    ) -> None:
        """`FORCE ROW LEVEL SECURITY`, not just `ENABLE`.

        Table owners bypass RLS by default, and the application connects as the
        owner. Without FORCE every policy is inert while still appearing in
        `pg_policies` — the failure mode where the security review passes and
        the security does not exist.
        """
        result = await uow.session.execute(
            text("SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = :table"),
            {"table": table},
        )
        enabled, forced = result.one()

        assert enabled is True, f"{table} does not have RLS enabled"
        assert forced is True, f"{table} has RLS enabled but not FORCED — it is inert"

    async def test_jobs_is_deliberately_exempt(self, uow: SqlAlchemyUnitOfWork) -> None:
        """The one exemption, pinned so it stays a decision rather than a gap.

        Workers poll across tenants by design and then scope themselves to each
        claimed job. No HTTP endpoint exposes the table, so the exemption does
        not widen the API surface — see docs/adr/0005.
        """
        result = await uow.session.execute(
            text("SELECT relrowsecurity FROM pg_class WHERE relname = 'jobs'")
        )

        assert result.scalar_one() is False


class TestRowLevelSecurityWrites:
    async def test_cannot_insert_a_row_for_another_tenant(
        self,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        other_tenant: Tenant,
    ) -> None:
        """`WITH CHECK`, not only `USING`.

        USING governs what a statement may read; WITH CHECK governs what it may
        write. With USING alone, a session scoped to Acme could insert a row
        stamped Globex — invisible to Acme afterwards, and quietly corrupting
        Globex's data with no error anywhere.
        """
        with pytest.raises(DBAPIError, match="row-level security"):
            await uow.users.create(
                tenant_id=other_tenant.id,
                email="mallory@globex.example",
                full_name="Mallory",
                role=Role.ADMIN,
                status=UserStatus.ACTIVE,
            )

        await uow.rollback()

    async def test_cannot_update_a_row_into_another_tenant(
        self,
        uow: SqlAlchemyUnitOfWork,
        tenant: Tenant,
        other_tenant: Tenant,
        user: User,
    ) -> None:
        with pytest.raises(DBAPIError, match="row-level security"):
            await uow.session.execute(
                text("UPDATE users SET tenant_id = :other WHERE id = :id"),
                {"other": other_tenant.id, "id": user.id},
            )

        await uow.rollback()

    async def test_an_unscoped_session_cannot_insert(
        self, uow: SqlAlchemyUnitOfWork, tenant: Tenant
    ) -> None:
        await uow.scope_to_tenant(None)

        with pytest.raises(DBAPIError, match="row-level security"):
            await uow.users.create(
                tenant_id=tenant.id, email="ghost@acme.example", full_name="Ghost"
            )

        await uow.rollback()


class TestScopeLifetime:
    async def test_scope_survives_a_commit(self, uow: SqlAlchemyUnitOfWork, tenant: Tenant) -> None:
        """`SET LOCAL` is discarded on commit, so the unit of work reapplies it.

        Without that, every statement after the first commit in a long-lived
        unit of work would silently see zero rows — a bug that looks like data
        loss and is maddening to trace.
        """
        await uow.users.create(tenant_id=tenant.id, email="first@acme.example")
        await uow.commit()

        assert await current_tenant_scope(uow.session) == tenant.id

        await uow.users.create(tenant_id=tenant.id, email="second@acme.example")
        await uow.commit()

        result = await uow.session.execute(text("SELECT count(*) FROM users"))
        assert result.scalar_one() == 2

    async def test_a_fresh_session_starts_unscoped(
        self, session_factory, tenant: Tenant, user: User
    ) -> None:
        """Proves the scope does not leak between pooled connections.

        This is why `SET LOCAL` was chosen over `SET`. A session-level setting
        would persist on the connection, and the next request to borrow it from
        the pool would inherit the previous tenant's scope — strictly worse than
        no RLS, because it looks safe.
        """
        from rag.db.uow import SqlAlchemyUnitOfWork as UoW

        async with UoW(session_factory) as fresh:
            assert await current_tenant_scope(fresh.session) is None

            result = await fresh.session.execute(text("SELECT count(*) FROM users"))
            assert result.scalar_one() == 0
