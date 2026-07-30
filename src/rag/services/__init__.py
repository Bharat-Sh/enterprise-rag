"""Application services: use cases and transaction boundaries.

Services orchestrate domain objects and *ports*. They never import a concrete
adapter — a service takes a `VectorStore` Protocol, not a `QdrantVectorStore`.
Enforced by the import-linter contract in pyproject.toml.

The package exists now, empty, so that contract is checked from M0 rather than
silently passing until the first service appears.

Populated from M3 onward: ingestion, retrieval, chat, document, tenant, eval.
"""
