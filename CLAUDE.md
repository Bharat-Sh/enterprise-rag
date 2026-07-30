# Working in this repository

Conventions and constraints that are not obvious from the code. Read
`docs/adr/` for the reasoning behind the architecture itself.

## Non-negotiables

1. **Postgres is the source of truth. Qdrant is a derived index.** Never write
   to Qdrant before the corresponding Postgres rows are committed. Vectors must
   always be rebuildable from the database; the reverse must never be true.
2. **`rag.domain` imports nothing.** No frameworks, no drivers, no HTTP clients,
   nothing from `rag.api`, `rag.adapters`, `rag.db`, or `rag.services`. Checked
   by `uv run lint-imports`.
3. **`rag.services` depends on ports, not adapters.** A service takes a
   `VectorStore` Protocol, never a `QdrantVectorStore`.
4. **Access control is a pre-filter.** Tenant and ACL constraints are pushed
   into the vector query. Never filter retrieved results after the fact.
5. **Tenant identity comes from the verified token.** Never from a request body,
   query parameter, or header.
6. **No blocking calls in `async def`.** CPU-bound work goes to a thread or
   process pool, or to the model service. A blocking call in a handler stalls
   every concurrent request on that worker, including open SSE streams.

## Style

- Python 3.12+, `from __future__ import annotations` at the top of every module.
- Type hints everywhere; `mypy` runs strict.
- **Ruff only** — `ruff check` and `ruff format`. Black is deliberately not used;
  `ruff format` is a drop-in replacement and running both means two tools
  arguing over the same file.
- Comments explain *why*, not *what*. If a decision has a rejected alternative,
  name it.
- Domain errors carry a stable `code` and no HTTP status. The mapping lives in
  `rag/api/errors.py`.

## Testing

- `tests/unit` — no I/O, no Docker, in-memory fakes of the ports. Milliseconds.
- `tests/integration` — real Postgres/Qdrant/Redis via testcontainers. Mocking a
  vector database tests the mock, not the code.
- `tests/security` — cross-tenant leakage and ACL bypass. These must never be
  skipped or marked xfail.
- `tests/eval` — golden-set regression; gates CI from M10.

Every test asserts on behaviour, not implementation. A test that breaks when you
rename a private method is a liability.

## Adding a milestone

Follow the sequence: requirements → design and tradeoffs → file list → implement
→ test → self-review → update `README.md` and the relevant ADR. State the
rejected alternative for any non-obvious choice.

## Current state

M0 complete: configuration, structured logging, request context, error
hierarchy, health/readiness, Docker, CI. No business logic yet.

Next: M1 — data model and persistence.
