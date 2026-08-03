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
- `tests/integration` — real backing services, never mocks. Mocking a database
  tests the mock, not the code. Marked `@pytest.mark.integration` and **skipped
  automatically when the service is unreachable**, so the suite stays green on a
  machine without one.
- `tests/security` — cross-tenant leakage and ACL bypass. These must never be
  skipped or marked xfail.
- `tests/eval` — golden-set regression; gates CI from M10.

Every test asserts on behaviour, not implementation. A test that breaks when you
rename a private method is a liability.

How integration dependencies are provided, given no Docker on the dev machine
(see "Local environment" below):

| Service | Locally | In CI |
| --- | --- | --- |
| Postgres | native Windows install | GitHub Actions service container |
| Qdrant (M5) | `qdrant-client` local mode (embedded, no server) | service container |
| Redis (M9) | `fakeredis` | service container |
| Model service (M4) | native Python + CUDA on the host GPU | skipped (no GPU runner) |

`docker/` stays in the repo and stays correct — CI builds and smoke-tests the
image on every push, so it cannot silently rot. It is simply never run here.

## Adding a milestone

Follow the sequence: requirements → design and tradeoffs → file list → implement
→ test → self-review → update `README.md` and the relevant ADR. State the
rejected alternative for any non-obvious choice.

## Local environment (this machine)

Windows 11. **No Docker installed, and none planned here** — containers are run
on a separate device. Do not propose testcontainers or `docker compose` as part
of the local workflow.

| | |
| --- | --- |
| Python | 3.12.10, managed with `uv`; venv at `.venv` |
| Postgres | 16.14 native service `postgresql-x64-16`, auto-start, `127.0.0.1:5432` |
| Credentials | role `rag` / password `rag`; databases `rag` (dev) and `rag_test` (suite) |
| Superuser | `postgres` / `rag` — local dev only |
| GPU | NVIDIA (VRAM not yet confirmed — needed to size M4 defaults) |
| `gh` CLI | installed, **not authenticated** |

`.env.example` defaults already match the local Postgres, so `cp .env.example .env`
works with no edits.

Commands (no `make` on Windows — run these directly):

```
uv run ruff check .          uv run ruff format --check .
uv run mypy                  uv run lint-imports
uv run pytest                uv run uvicorn rag.api.asgi:app --reload
```

## Current state

**M0 complete.** Configuration, structured logging, request context, error
hierarchy, liveness/readiness, Docker definitions, CI. No business logic yet.

Gate is green: ruff, `ruff format`, mypy strict, 3/3 import contracts, 63 tests.
Verified by boot smoke test (`/health` 200, `/ready` 200, unknown route 404 as
`application/problem+json`).

Three commits on `main`. **No git remote — nothing has been pushed to GitHub
yet.** Commits are authored as `122530216+Bharat-Sh@users.noreply.github.com`;
keep it that way.

One M0 subtlety worth not re-discovering: Starlette installs
`ServerErrorMiddleware` outside all custom middleware, so on the unhandled-error
path the request contextvars are already unbound and its response bypasses our
`send` wrapper. Correlation ids are therefore read from `scope["state"]` and set
directly on the problem response. See `rag/api/errors.py::correlation_ids` and
the regression test in `tests/unit/test_errors.py`.

## Next: M1 — data model and persistence

Tenants, users, roles, documents, chunks, jobs. Async SQLAlchemy 2.x with a
unit-of-work, Alembic migrations, repositories behind ports.

Two design decisions to settle *before* writing models, both with real
consequences downstream:

1. **Tenant isolation.** A `tenant_id` column every query must remember to
   filter, versus Postgres Row-Level Security where the database refuses to
   return other tenants' rows even when the application forgets. RLS costs
   setup and some query planning predictability; forgetting a `WHERE` clause
   once is a breach.
2. **ACL shape.** Document permissions must be expressible as a **Qdrant payload
   filter** in M5. Modelling them as a normalised join table only Postgres can
   evaluate forces M5 into post-retrieval filtering, which corrupts recall *and*
   means the rows were read before the check. Design the schema backwards from
   that constraint.

Chunks carry `embedding_model` and `embedding_version` from the start, so the
M4/M5 embedding-migration path (re-embed into a new named vector, backfill, cut
over) stays open.
