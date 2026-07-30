# ADR-0003 — Ports and adapters, enforced by CI

- **Status:** Accepted
- **Date:** 2026-07-30
- **Milestone:** M0

## Context

This system integrates a vector database, an embedding model, a reranker, an
LLM provider, a cache, and several document parsers. Every one of them is
plausibly replaceable — the embedding model will change, the LLM provider is not
yet chosen, and `pgvector` remains a credible alternative to Qdrant.

If those clients are imported directly wherever they are used, each replacement
becomes a codebase-wide refactor, and no business logic can be tested without
standing up the real services.

## Decision

A trimmed hexagonal architecture:

```
rag.api        HTTP delivery
rag.services   use cases — depends on ports only
rag.domain     models + ports (Protocols) — pure, imports nothing
rag.adapters   concrete implementations; the only place vendor clients appear
rag.db         persistence (Postgres is not a swappable detail — see ADR-0001)
rag.core       config, logging, errors, context — the base layer
```

Ports are `typing.Protocol` definitions in `rag.domain.ports`. Services accept
them via constructor injection. Concrete adapters are wired once, in the
composition root.

**Structural typing, not inheritance.** A `Protocol` means an adapter does not
import or subclass anything from the domain to satisfy it — the dependency
points one way only, and a test fake is a plain class with the right methods.

## Enforcement

Convention is not enforcement. `import-linter` contracts in `pyproject.toml` run
as a CI gate:

- `rag.domain` may not import `rag.api`, `rag.adapters`, `rag.db`,
  `rag.services`, `fastapi`, or `starlette`.
- `rag.services` may not import `rag.adapters`, `rag.api`, or `rag.db`.
- `rag.core` may not import anything else of ours.

The `services`, `adapters`, and `db` packages exist (empty) from M0 so the
contracts are checked from the first commit rather than silently passing until
the first real module lands.

## Consequences

**We get:**

- Unit tests run in milliseconds against in-memory fakes: no Docker, no network.
- Swapping Qdrant for `pgvector` is one new adapter file plus one wiring line.
- The LLM provider can stay undecided without blocking anything.
- The code reads as *what the system does* rather than *which library we used*.

**We pay:**

- More files, and one layer of indirection that feels excessive on day one.
- A genuine risk of leaky abstraction: if the `VectorStore` port ends up exposing
  Qdrant-shaped filter objects, the boundary is decorative. The port must speak
  in domain terms (`AccessFilter`, `SearchRequest`), and adapters translate.

## Alternative considered

A flat package layout with direct imports. Faster to write and appropriate for a
service with one integration and no expectation of change. Rejected because this
system has six integrations, two of which are explicitly undecided, and because
the port boundaries are themselves a deliverable — they are the difference
between a demo and a system.
