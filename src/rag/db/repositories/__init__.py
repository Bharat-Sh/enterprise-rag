"""Concrete repository implementations backed by SQLAlchemy.

Each one satisfies the matching Protocol in `rag.domain.ports` structurally — no
inheritance, no registration. Every method returns domain dataclasses, never ORM
instances, so nothing above this package can trip a lazy load or depend on the
identity map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from sqlalchemy import CursorResult, Result

__all__ = ["affected_rows"]


def affected_rows(result: Result[Any]) -> int:
    """Number of rows a DML statement touched.

    `Session.execute` is typed as returning `Result`, but `rowcount` lives on
    `CursorResult` — which is what every INSERT/UPDATE/DELETE actually returns.
    One cast in one place records that invariant, rather than scattering
    `# type: ignore` across the repositories and losing the explanation.
    """
    return cast("CursorResult[Any]", result).rowcount or 0
