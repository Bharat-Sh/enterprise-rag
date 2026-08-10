"""ingestion: document blob key

Revision ID: 5017a3337c5a
Revises: 353a989f7c3a
Created: 2026-08-10 16:04:26.482593

Where a document's raw bytes live in the blob store (docs/adr/0009), as
distinct from `source_uri`, which records where they came from.

`NOT NULL DEFAULT ''` rather than nullable: no rows exist yet in any
environment, and an empty string is a value the ingestion path can assert on,
whereas a nullable column would leave "not yet stored" and "stored at the empty
key" indistinguishable forever.

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5017a3337c5a"
down_revision: str | None = "353a989f7c3a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("blob_key", sa.Text(), server_default=sa.text("''"), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("documents", "blob_key")
