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
| `gh` CLI | installed and authenticated as `Bharat-Sh` |
| Blob store | filesystem, `./var/blobs` (gitignored) — see docs/adr/0009 |

**`rag` must stay a non-superuser.** It owns its tables, and `FORCE ROW LEVEL
SECURITY` is the only thing stopping an owner from bypassing every policy.
Granting it `SUPERUSER` or `BYPASSRLS` to fix a permissions problem would make
the entire tenant-isolation suite vacuous while still passing — which is exactly
how CI ran red for two milestones. Never run the suite as `postgres`.

`.env.example` defaults already match the local Postgres, so `cp .env.example .env`
works with no edits.

Commands (no `make` on Windows — run these directly):

```
uv run ruff check .          uv run ruff format --check .
uv run mypy                  uv run lint-imports
uv run pytest                uv run uvicorn rag.api.asgi:app --reload
uv run rag-admin --help      # bootstrap: create-tenant, create-user, generate-key
uv run rag-worker --once     # drain the ingestion queue and exit
```

The full suite takes a little over two minutes, most of it Argon2 in the
integration tests. `uv run pytest tests/unit` is seconds.

**Do not run two pytest invocations at once.** Every integration test truncates
`rag_test` in its `db_engine` fixture, so concurrent runs delete each other's
fixtures and fail in ways that look like real bugs.

## Current state

**M0 through M3 complete.** Config, logging, request context, errors,
liveness/readiness, Docker, CI (M0); schema, migrations, RLS, repositories,
unit of work, job queue (M1); password and API-key auth, Ed25519 JWTs with JWKS,
rotating refresh tokens, RBAC, per-tenant rate limiting, `rag-admin` (M2);
upload, blob store, ingestion worker, chunking, document/collection endpoints,
`rag-worker` (M3a); PDF and DOCX parsers with hostile-input hardening (M3b).

Gate is green: ruff, `ruff format`, mypy strict, 3/3 import contracts,
**605 tests** (unit + integration + security, against a real Postgres).

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

### M3a subtleties worth not re-discovering

- **Starlette buffers the whole request body before a handler runs.** A size
  check on `UploadFile` therefore protects nothing — the bytes are already on
  disk. `BodySizeLimitMiddleware` counts them on the ASGI `receive` channel,
  which is the only place the limit can be real.
- **Middleware added with `add_middleware` sits *outside* the exception
  handlers.** Starlette's stack is `ServerErrorMiddleware → user middleware →
  ExceptionMiddleware → router`, so raising a domain error there yields a 500,
  not its mapped status. That middleware builds its own problem response.
- **`CHUNKING → READY` is a temporary edge in `rag.domain.state`.** M3 stops at
  chunks. **M5 must remove it**: once indexing exists, a document reaching READY
  without vectors is invisible to retrieval while claiming to be searchable. A
  test asserts the edge exists, so deleting it is deliberate.
- **`get_for_processing` is the only ACL-free document read**, for the worker,
  which has no caller whose principals it could apply. It is still bound by the
  tenant scope — it widens ACL visibility within one tenant, never across them.
- **Jobs are idempotent because they have to be.** At-least-once delivery is
  real (`reap_stalled`), so `ingest` returns early on an already-READY document
  and `purge` on an already-DELETED one. Without that, redelivery after a crash
  dead-letters work that actually succeeded.
- **`documents.create` and `collections.create` translate `IntegrityError`
  inside a `begin_nested()` SAVEPOINT**, like `users.create`. Without the
  savepoint Postgres aborts the caller's whole transaction and the conflict can
  only be reported by destroying their other work.
- **Token counts are estimates until M4.** `tiktoken` was rejected: a precise
  count for a model we do not use is worse than an honest approximation.

### M3b subtleties worth not re-discovering

- **`defusedxml` is load-bearing, and the test that proves it is fragile.**
  Stdlib `ElementTree` also raises on an *undefined* entity, so a test that only
  asserts "some error" would still pass with `defusedxml` removed. The entity
  tests assert `details["reason"] == "EntitiesForbidden"` for that reason.
- **A ZIP magic number identifies the container, never the payload.** DOCX,
  XLSX, PPTX and JAR are byte-identical at the front, so the parser verifies
  `word/document.xml` exists rather than trusting the sniffer.
- **Zip limits are checked against the central directory *and* on read.** The
  directory is metadata the attacker writes, so its declared sizes can be a lie;
  the bounded read is what holds when they are.
- **No `python-docx`.** It hands XML to `lxml` with entity expansion on and does
  no decompression accounting, so it would have meant doing the hardening anyway
  *and* trusting its parser. `zipfile` + `defusedxml` keeps every limit visible.
- **PyMuPDF is AGPL** — it would reach this whole codebase, which is MIT and
  meant to be published. That is why `pypdf` is used despite being slower.

## Known limitations

Everything here is a **deliberate, known gap**, not a bug and not an oversight.
Each entry says what breaks, why it is that way, and what the real fix is, so
that none of it has to be reconstructed by reading the code. If you hit one of
these, the decision has already been made — extend it or reverse it on purpose,
do not "fix" it by accident.

Ordered roughly by how likely you are to trip over it.

### A hung parse leaks a thread

**What happens.** `IngestionPipeline._parse` runs the parser under
`anyio.fail_after(parse_timeout_seconds)` inside `anyio.to_thread.run_sync`.
When the timeout fires, the *job* fails correctly and the worker moves on — but
**Python cannot cancel a thread**. There is no interface for it and never has
been. The thread keeps running whatever it was doing, holding its memory and
burning a core, and for a genuine infinite loop it never stops. The worker
process must be restarted to reclaim it.

**Why it is like this.** The alternative is a `ProcessPoolExecutor`, where a
hung child *can* be killed. That costs inter-process serialisation of every
document body — real overhead on the hot path for a failure mode we have not yet
observed — and adds a pool lifecycle to manage, supervise, and test.

**What actually protects us instead.** The timeout is the backstop, not the
defence. The defence is the caps in `IngestionSettings`, which bound the work
before it can become unbounded: `max_pdf_pages`, `max_extracted_bytes`,
`max_compression_ratio`, `max_archive_entries`. Those are why a hang is expected
to be rare rather than routine. The blast radius is also already bounded: a
worker runs **one job at a time**, so a leaked thread degrades one container
that an orchestrator will restart, not a shared pool.

**The real fix.** Move parsing to a process pool, or to a subprocess per
document. Do it when a hang is *observed*, not on principle — and if you do,
delete this entry rather than leaving it to rot.

**Where to look.** `rag/services/pipeline.py::_parse`, `docs/adr/0010`.

### The rate limiter is per worker process

**What happens.** `InProcessRateLimiter` holds token buckets in memory. With N
uvicorn worker processes, a tenant gets N times the configured allowance,
because each process enforces the limit independently and none of them know
about the others.

**Why it is like this.** Redis is not a dependency until M9. Shipping nothing
until then would have left `/auth/login` — the credential-stuffing surface, and
the only unauthenticated write in the system — with no limit at all for seven
milestones.

**What contains it.** `ServerSettings.workers` defaults to **1**, and the
deployment story is replica scaling, so the limit is exact locally and in a
single-process container. It is approximate the moment there are several.

**The real fix.** The M9 Redis adapter, behind the same `RateLimiter` port. No
call site changes.

**Where to look.** `rag/adapters/ratelimit/inprocess.py`, `docs/adr/0008`.

### Access-token revocation has a one-second boundary

**What happens.** `users.tokens_valid_after` rejects tokens issued before it.
A JWT `iat` is a NumericDate and carries **whole seconds**, while the watermark
has microsecond precision — so the comparison truncates, and a token minted in
the *same second* as a revocation survives.

**Why it is like this.** It is forced by the resolution of `iat`, not chosen.
Comparing exactly would reject the replacement token that a password change
hands back, making that flow return a pair that was already dead.

**What is exact.** Refresh-token revocation, which is database rows, not a
timestamp comparison. Password change and forced logout revoke both, so the
window applies only to an access token already in flight, for under a second.

**Where to look.** `User.accepts_token_issued_at`, `tests/unit/test_models.py`
(the boundary is pinned there), `docs/adr/0007`.

### The blob store is single-node

**What happens.** `FilesystemBlobStore` writes to local disk. Two API replicas
do not share one, so an upload handled by replica A and parsed by a worker on
node B would not find its bytes.

**Why it is like this.** There is no Docker on the primary dev machine, so MinIO
cannot run here. Writing the `BlobStore` port first means S3 is one adapter and
one wiring line rather than a refactor of every call site.

**What contains it.** `docker/compose.yml` mounts one `blob-data` volume shared
by the API and worker, so the containerised single-node setup is correct.

**The real fix.** An S3-compatible adapter at deployment.

**Where to look.** `rag/adapters/blobs/filesystem.py`, `docs/adr/0009`.

### Orphan blobs accumulate

**What happens.** Uploads write the blob *before* committing the document row
(deliberately — see ADR-0009). If the commit then fails, the bytes stay on disk
with nothing referencing them, forever. Nothing collects them.

**Why it is like this.** The ordering is correct: the alternative is a job whose
bytes do not exist, which is a user-visible failure rather than invisible
garbage. Orphans are the cheaper mistake.

**The real fix.** A sweep that lists blobs and deletes any with no matching
`documents.blob_key`. Belongs in M12 with the ACL reconciliation job it
resembles.

### `CHUNKING → READY` must be removed in M5

**What happens.** `rag.domain.state.ALLOWED_TRANSITIONS` currently permits a
document to go straight from `CHUNKING` to `READY`, because the M3 pipeline
stops at chunks — there is no embedding provider until M4 and no vector index
until M5.

**Why this is dangerous to leave.** Once indexing exists, a document that
reaches `READY` without vectors is **invisible to retrieval while claiming to be
searchable**, and nothing errors. That is the worst failure shape in the system.

**What to do in M5.** Delete `S.READY` from the `S.CHUNKING` frozenset in
`ALLOWED_TRANSITIONS`, then delete `TestTemporaryEdgeForM3` from
`tests/unit/test_state.py`. That test class asserts the edge is *present*, so
removing the edge fails it on purpose — it exists so this cannot be forgotten,
and its failure message says as much.

**Where to look.** `rag/domain/state.py::ALLOWED_TRANSITIONS`,
`tests/unit/test_state.py::TestTemporaryEdgeForM3`.

### Token counts are estimates until M4

**What happens.** `chunks.token_count` and every chunk-size decision come from
`HeuristicTokenCounter`, a character-ratio approximation, not from a tokenizer.

**Why it is like this.** The tokenizer that matters is BGE-M3's, and it lives
with the model service M4 introduces. `tiktoken` was rejected because a precise
count for a model we do not use is worse than an honest approximation — it looks
authoritative and is systematically wrong.

**What it affects.** Chunk boundaries move slightly. Nothing budgets a context
window against these numbers, which is the use that would not tolerate being
approximate.

**The real fix.** M4 swaps the `TokenCounter` implementation. No call sites
change.

### Parsing gaps

- **No OCR.** A scanned PDF has no text layer, so it fails as "no extractable
  text" rather than silently becoming a `READY` document with zero chunks.
  Loud is correct; OCR is a separate service with a GPU budget.
- **No PDF layout reconstruction.** Multi-column pages extract in content-stream
  order, which is sometimes wrong. Fixing it properly means layout analysis.
- **DOCX drops headers, footers, and footnotes.** Headers and footers are
  usually a page number and a banner repeated on every page — noise that would
  be embedded into every chunk, so losing them is closer to a feature.
  **Footnotes are a genuine loss.** Tables *are* extracted.

### Access-model gaps

- **Flat groups only.** Nested groups need recursive expansion when building a
  caller's principal set — a cheap Postgres recursive CTE, but scope we do not
  need yet.
- **No ACL reconciliation job.** A document's ACL lives in three places: the
  `document_permissions` rows, the array on `documents`, and a copy on every
  chunk. `set_acl` rewrites all three in one transaction, and a test asserts the
  reprojection — but nothing detects drift if they ever diverge. M12.
- **A suspended tenant can still write.** `TenantStatus.SUSPENDED` is documented
  as "reads continue, writes stop", and login honours it, but no write path
  checks it yet.
- **`DOCUMENT_DELETE` is admin-only.** There is no per-document ownership model,
  so a member cannot delete their own upload. Deliberate: adding ownership means
  an owner check on every path, and it is one predicate to add later.

## Next: M4 — the model service

BGE-M3 embeddings and a reranker on the GPU, reached over HTTP so the API and
worker stay CPU-only (docs/adr/0004). It replaces `HeuristicTokenCounter` with
the real tokenizer, which is when `chunks.token_count` stops being an estimate.

Add new permissions to `rag.domain.authz.MINIMUM_ROLE` rather than checking
roles inline — the table is what makes "which endpoints can a viewer reach?"
answerable.
