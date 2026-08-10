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

**`rag` must stay a non-superuser.** It owns its tables, and `FORCE ROW LEVEL
SECURITY` is the only thing stopping an owner from bypassing every policy.
Granting it `SUPERUSER` or `BYPASSRLS` to fix a permissions problem would make
the entire tenant-isolation suite vacuous while still passing — which is exactly
how CI ran red for two milestones. Never run the suite as `postgres`.
| GPU | NVIDIA (VRAM not yet confirmed — needed to size M4 defaults) |
| `gh` CLI | installed, **not authenticated** |

`.env.example` defaults already match the local Postgres, so `cp .env.example .env`
works with no edits.

Commands (no `make` on Windows — run these directly):

```
uv run ruff check .          uv run ruff format --check .
uv run mypy                  uv run lint-imports
uv run pytest                uv run uvicorn rag.api.asgi:app --reload
uv run rag-admin --help      # bootstrap: create-tenant, create-user, generate-key
```

The full suite takes a little over two minutes, most of it Argon2 in the
integration tests. `uv run pytest tests/unit` is seconds.

**Do not run two pytest invocations at once.** Every integration test truncates
`rag_test` in its `db_engine` fixture, so concurrent runs delete each other's
fixtures and fail in ways that look like real bugs.

## Current state

**M0, M1 and M2 complete.** Config, logging, request context, errors,
liveness/readiness, Docker, CI (M0); schema, migrations, RLS, repositories,
unit of work, job queue (M1); password and API-key auth, Ed25519 JWTs with JWKS,
rotating refresh tokens, RBAC, per-tenant rate limiting, `rag-admin` (M2).

Gate is green: ruff, `ruff format`, mypy strict, 3/3 import contracts,
**423 tests** (unit + integration + security, against a real Postgres).

**Check CI, not just the local gate.** They diverged silently for two
milestones: the tenant-isolation tests passed locally and failed on every CI run
from M1 onward, because the workflow's `POSTGRES_USER: rag` made the application
role the cluster's bootstrap **superuser**, and a superuser bypasses row-level
security unconditionally — `FORCE` does not apply to it. CI now creates `rag` as
an ordinary owner role, and
`TestThePreconditionEverythingElseRestsOn` fails in one legible line if that
ever regresses. `gh run list --limit 3` after a push.

Commits are authored as `122530216+Bharat-Sh@users.noreply.github.com` — keep it
that way; the repo is intended to be public eventually and a real address in git
history is permanent once published.

Check `git remote -v` before assuming anything about the remote. The repo is
`Bharat-Sh/enterprise-rag`, private for now.

### Subtleties worth not re-discovering

- **Correlation ids on the 500 path.** Starlette installs `ServerErrorMiddleware`
  outside all custom middleware, so on the unhandled-error path the contextvars
  are already unbound and its response bypasses our `send` wrapper. Ids are read
  from `scope["state"]` and set directly on the problem response. See
  `rag/api/errors.py::correlation_ids`.
- **`FORCE ROW LEVEL SECURITY` is mandatory.** Table owners bypass RLS by
  default and the app owns its tables. Without FORCE every policy is inert while
  still appearing in `pg_policies`. Asserted by test.
- **`SET LOCAL`, never `SET`.** Transaction-scoped, so a pooled connection never
  carries one tenant's scope into the next request. Consequence:
  `UnitOfWork.commit()` must reapply the scope, because commit discards it.
- **`WITH CHECK` as well as `USING`.** Otherwise a session scoped to tenant A can
  insert rows stamped tenant B.
- **`claim()` re-reads through the ORM** rather than parsing `RETURNING` rows:
  raw SQL has no result-type information, so JSONB payloads come back as text.
- **Two distinct conflict errors.** `InvalidStateTransitionError` means the move
  is impossible from any state (do not retry). `ConcurrentModificationError`
  means it arrived second (re-read and retry). Both 409; different client
  behaviour, so they need different codes.

### M2 subtleties worth not re-discovering

- **Every type used in an `Annotated[..., Depends(...)]` alias must be imported
  at *runtime*, not under `TYPE_CHECKING`.** FastAPI resolves dependency
  annotations with `get_type_hints` when a route is registered. An unresolvable
  name is silently reinterpreted as a **required query parameter**, so the route
  returns a blanket 422 and never reaches its handler. Cost an hour. Guarded by
  `tests/security/test_route_coverage.py::TestParameterResolution`.
- **FastAPI no longer flattens included routers onto `app.routes`.** They are
  wrapped in `_IncludedRouter`, holding `original_router` and the mount prefix
  in `include_context`. Any code walking routes must recurse — and a route-walk
  that finds nothing *passes*, which is why that test asserts its own
  enumeration against the OpenAPI paths first.
- **`iat` is a NumericDate — whole seconds.** `users.tokens_valid_after` has
  microsecond precision, so the comparison truncates the watermark to the
  second. Without that, the replacement token issued *by* a password change is
  rejected by the watermark that change just set. The cost is a one-second
  boundary, which is inherent to the resolution and not a choice. Pinned in
  `tests/unit/test_models.py`.
- **The credential carries the tenant.** RLS is bound from it before the
  credential is validated, so a forged tenant finds zero rows rather than
  failing a comparison. This is what lets `api_keys` and `refresh_tokens` stay
  under RLS despite being read *by* the authentication path. See ADR-0007.
- **`get_unscoped_unit_of_work` has exactly three call sites** — login, refresh,
  logout. Asserted by test. Everything else takes `UnitOfWorkDep`, which cannot
  be obtained without a verified tenant.
- **An API key's role ceiling must reach `AccessFilter.build`.** Applying it only
  to route permissions narrows what the key may *call* and leaves what it may
  *read* untouched, which is the more damaging half.
- **`argon2.PasswordHasher` uses `__slots__`**, so instance attributes cannot be
  monkeypatched; patch the class instead.
- **`EmailStr` rejects `.test` and `.local`** as special-use domains. Test
  fixtures use `.example`.
- **A check-then-insert cannot win a race.** `UserRepository.create` wraps its
  flush in `begin_nested()` (a SAVEPOINT) and translates `IntegrityError` into
  `AlreadyExistsError` — without the savepoint, Postgres aborts the whole
  transaction and the only way to report the conflict would be to destroy work
  the caller had already done.

## Next: M3 — ingestion

Upload, the job queue in anger, parsing, chunking, and the document state
machine. `rag.services` gains its second and third services; `rag.adapters`
gains its first parsers.

Everything M3 adds is already behind authentication: a handler that takes
`UnitOfWorkDep` is tenant-scoped by construction, and `require(Permission.X)`
gates the action. Add new permissions to `rag.domain.authz.MINIMUM_ROLE` rather
than checking roles inline — the table is what makes "which endpoints can a
viewer reach?" answerable.
