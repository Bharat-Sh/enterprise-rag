"""Domain layer: pure models, business rules, and ports (Protocols).

This package must not import web frameworks, database drivers, HTTP clients, or
anything from `rag.api`, `rag.adapters`, `rag.db`, or `rag.services`. The rule
is enforced by the import-linter contract in pyproject.toml, not by discipline.

The payoff: domain logic is testable with no Docker, no event loop, and no
mocks — and swapping Qdrant for pgvector touches one adapter file rather than
the whole codebase.
"""
