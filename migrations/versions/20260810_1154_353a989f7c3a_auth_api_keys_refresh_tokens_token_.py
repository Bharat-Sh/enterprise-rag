"""auth: api keys, refresh tokens, token invalidation

Revision ID: 353a989f7c3a
Revises: 3257f60ee6a4
Created: 2026-08-10 11:54:01.191254

Adds the two credential tables and the column that makes a stateless access
token revocable. Both new tables carry the same tenant-isolation policy as every
other table holding customer data (docs/adr/0005), including `api_keys` — which
is read by the *authentication* path itself. That works because a key's own
tenant segment binds the row-level-security scope before the lookup runs; see
docs/adr/0007.

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

#: Tables introduced by *this* revision that need a tenant-isolation policy.
#: Deliberately not the full list from `rag.db.models.TENANT_SCOPED_TABLES`: a
#: migration describes the schema as of its own revision, and importing the
#: application's list would make this file change meaning as the models evolve.
NEW_TENANT_SCOPED_TABLES: tuple[str, ...] = ("api_keys", "refresh_tokens")

revision: str = "353a989f7c3a"
down_revision: str | None = "3257f60ee6a4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: `user_role` already exists, created by the initial revision. Referencing it
#: with `create_type=False` reuses it; a bare `sa.Enum` would emit a second
#: CREATE TYPE and fail the migration.
_USER_ROLE = postgresql.ENUM(name="user_role", create_type=False)


def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("display_prefix", sa.String(length=16), nullable=False),
        sa.Column("secret_hash", sa.String(length=64), nullable=False),
        sa.Column("role", _USER_ROLE, nullable=False),
        sa.Column("created_by", sa.UUID(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("tenant_id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["created_by"],
            ["users.id"],
            name=op.f("fk_api_keys_created_by_users"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name=op.f("fk_api_keys_tenant_id_tenants"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_api_keys_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_api_keys")),
        sa.UniqueConstraint("tenant_id", "secret_hash", name="uq_api_keys_tenant_id_secret_hash"),
    )
    op.create_index(op.f("ix_api_keys_tenant_id"), "api_keys", ["tenant_id"], unique=False)
    op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"], unique=False)

    op.create_table(
        "refresh_tokens",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("family_id", sa.UUID(), nullable=False),
        sa.Column(
            "issued_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("replaced_by", sa.UUID(), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("tenant_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["replaced_by"],
            ["refresh_tokens.id"],
            name=op.f("fk_refresh_tokens_replaced_by_refresh_tokens"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name=op.f("fk_refresh_tokens_tenant_id_tenants"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_refresh_tokens_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_refresh_tokens")),
        sa.UniqueConstraint(
            "tenant_id", "token_hash", name="uq_refresh_tokens_tenant_id_token_hash"
        ),
    )
    op.create_index("ix_refresh_tokens_expires_at", "refresh_tokens", ["expires_at"], unique=False)
    op.create_index("ix_refresh_tokens_family_id", "refresh_tokens", ["family_id"], unique=False)
    op.create_index(
        op.f("ix_refresh_tokens_tenant_id"), "refresh_tokens", ["tenant_id"], unique=False
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"], unique=False)

    op.add_column(
        "users", sa.Column("tokens_valid_after", sa.DateTime(timezone=True), nullable=True)
    )

    _enable_row_level_security()


def _enable_row_level_security() -> None:
    """Extend tenant isolation to the credential tables.

    Identical to the initial revision's predicate, deliberately: two access
    checks that must agree but are written separately eventually disagree, and
    the disagreement is a leak. `FORCE` is what makes any of it real — the
    application owns these tables, and a table owner bypasses `ENABLE`-only RLS
    while the policy still shows up in `pg_policies`.
    """
    predicate = "tenant_id = NULLIF(current_setting('rag.tenant_id', true), '')::uuid"

    for table in NEW_TENANT_SCOPED_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    # Policies disappear with their tables; dropping them explicitly first would
    # only fail if a table were already gone.
    op.drop_column("users", "tokens_valid_after")
    op.drop_index("ix_refresh_tokens_user_id", table_name="refresh_tokens")
    op.drop_index(op.f("ix_refresh_tokens_tenant_id"), table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_family_id", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_expires_at", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")
    op.drop_index("ix_api_keys_user_id", table_name="api_keys")
    op.drop_index(op.f("ix_api_keys_tenant_id"), table_name="api_keys")
    op.drop_table("api_keys")
