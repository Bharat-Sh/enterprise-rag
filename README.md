# Enterprise RAG Platform

A multi-tenant, production-shaped retrieval-augmented generation platform:
documents in, grounded and cited answers out, with access control, evaluation,
and observability treated as features rather than afterthoughts.

> **Status: M5 — Retrieval works end to end.** PDF, DOCX, HTML, Markdown
> and plain text can be uploaded, stored, parsed, and chunked by a background
> worker, behind an API where every endpoint authenticates and every query is
> tenant-scoped. A separate GPU service serves BGE-M3 embeddings and
> cross-encoder reranking over HTTP, so the API and worker stay CPU-only.
> Documents are now embedded, indexed in Qdrant, and searchable through
> `POST /api/v1/search`, with tenant and ACL constraints pushed *into* the
> vector query. Hybrid retrieval and reranking follow in M6 and M7. Milestones
> below.

---

## Why this exists

Wiring an embedding model to a vector database is a weekend. The parts that make
a RAG system deployable inside a company are the parts most demos skip:

| Concern | How it is handled here |
|---|---|
| **Tenant isolation** | Access control is a *pre-filter pushed into the vector query*, never a filter applied to results after retrieval. Post-filtering both corrupts recall and means the data was already read. |
| **Authentication** | The credential names its own tenant, and row-level security is bound from it *before* the credential is validated. A forged tenant finds zero rows rather than failing a comparison — the control that protects the data also protects the check that guards it. |
| **Ingestion durability** | Documents move through an explicit state machine persisted in Postgres, with content-hash idempotency, bounded retries, and a dead-letter state. |
| **Index rebuildability** | Postgres is the system of record; Qdrant is a derived index that can be dropped and rebuilt from it at any time. |
| **Answer trust** | Every citation is validated against the actually-retrieved chunk set before it reaches the client. A claimed citation is worth nothing. |
| **Change safety** | A golden evaluation set gates CI: a pull request that hurts retrieval quality fails to merge. |
| **Debuggability** | One trace id spans the HTTP request, the retrieval graph, and the LLM call. A user quotes it; we reconstruct everything. |

Architectural decisions — including the ones where the obvious alternative was
rejected, and why — live in [`docs/adr/`](docs/adr/).

### Data model

```
tenants ─┬─ users ─┬─ group_members ──── groups
         │         ├─ api_keys
         │         └─ refresh_tokens
         ├─ collections ── documents ─┬─ chunks
         │                            └─ document_permissions
         └─ jobs   (no RLS — workers poll cross-tenant by design)

raw bytes and extracted text live in a blob store, not in Postgres
(docs/adr/0009); `documents.blob_key` points at them

documents.status:  UPLOADED → QUEUED → PARSING → CHUNKING → EMBEDDING
                            → INDEXING → READY
                   any → FAILED · READY → REINDEXING / DELETING → DELETED
```

Every tenant-scoped table carries a row-level security policy that reads
`current_setting('rag.tenant_id')`, bound per transaction with `SET LOCAL`. An
unscoped session sees **zero** rows rather than every row — the failure mode that
justified the complexity. See [ADR-0005](docs/adr/0005-row-level-security.md).

Within a tenant, document permissions are a flat `text[]` of principal tokens
(`user:…`, `group:…`, `role:…`, `tenant:…`) matched against the caller's set with
the GIN-indexed array-overlap operator `&&` — the exact semantics of Qdrant's
`match_any`, so M5 enforces access with the same decision procedure rather than a
similar-looking reimplementation. See [ADR-0006](docs/adr/0006-acl-projection.md).

---

## Authentication

Two credential types, one `Authorization: Bearer` header, discriminated on a
prefix:

```
eyJhbGciOiJFZERTQSIsImtpZCI6…   an access token — Ed25519, RFC 9068 shaped
ragk_ndkbc5jzhffwbcbnasu6qnmhoe_kQ7t…   an API key — tenant, then 256-bit secret
```

Both name their own tenant, and that is the load-bearing design decision. Every
table holding customer data is invisible until a transaction binds a scope, so
validating a credential means reading rows that cannot be read until the tenant
is known — and the credential is what knows it. The cycle is broken by binding
the scope *from* the credential and validating it *under* that scope:

```
get_credential           header only, no I/O
    ↓
get_verified_credential  signature / alg / kid / typ / exp / aud / iss
    ↓                    (or an API key's shape).  Yields a tenant id.
    ↓
get_unit_of_work         opens the transaction, binds RLS to that tenant
    ↓
get_principal            under that scope: user row, status, groups, key ceiling
```

Point a credential at another customer and the lookup returns nothing. No
comparison rejects it; the absence of a row does. And because a transaction
cannot be obtained without a verified tenant, an endpoint that forgets
authentication does not compile — a test additionally walks every registered
route and fails if one resolves no principal.
See [ADR-0007](docs/adr/0007-token-bound-tenant-scope.md).

| | |
|---|---|
| **Access tokens** | Ed25519 by default, `kid`-addressed with a JWKS at `/.well-known/jwks.json`, so rotation is a deploy rather than a flag day. RS256 is one config entry away for verifiers that need it. |
| **What is in the token** | `iss`, `sub`, `aud`, `exp`, `iat`, `nbf`, `jti`, `tid`. No role, no groups: both are read from the user's row on every request, so a demotion or a group change takes effect on the *next* request instead of at token expiry. |
| **Revocation** | `users.tokens_valid_after` — a watermark, checked against `iat` on a row we already load. No denylist, no extra round trip. |
| **Refresh tokens** | Opaque, database-backed, single use, rotating. Replaying a spent one means two parties hold it, so the whole rotation family is revoked and the access-token watermark moves. |
| **Passwords** | Argon2id at the OWASP minimum, hashed in a thread. Unknown tenant, unknown user, and wrong password all cost one verification and return the identical 401 — a difference in any of the three is an enumeration oracle a stopwatch can read. |
| **API keys** | SHA-256, not Argon2: the secret is 256 bits from a CSPRNG, so a slow hash buys nothing and puts 80 ms on the hottest auth path. A key belongs to a user and its role is a **ceiling** — applied before the `AccessFilter` is built, or it would narrow what the key may *call* while leaving what it may *read* untouched. |
| **Roles** | OWNER > ADMIN > MEMBER > VIEWER, a total order. Permissions are a table in `rag.domain.authz`, gated at routes by `require(Permission.X)` — so "what can a viewer reach?" is a lookup, not a grep. |
| **Rate limiting** | Token buckets: per tenant for authenticated traffic, per client address for login. In-process in M2 (so per worker — stated, not hidden), Redis-backed in M9 behind the same port. See [ADR-0008](docs/adr/0008-rate-limiting.md). |

Cross-tenant lookups answer **404, never 403**. A 403 confirms the resource
exists, which lets an attacker enumerate another customer's ids from status
codes alone.

---

## Ingestion

Upload returns **202**, not 201. The document exists; it is not searchable yet.

```
POST /documents ──► stream ──► sha256 + size (limit enforced mid-stream)
                       │
                       ├─► BlobStore.put()                    ← blob first
                       │
                       └─► ONE TRANSACTION (ADR-0002):
                             documents row (uploaded → queued)
                             jobs row (ingest_document)

rag-worker ──► claim(SKIP LOCKED) ──► parsing ──► chunking ──► ready
                     │                    │           │
                     │               to_thread   to_thread
                     │               + timeout   + overlap
                     └── failure ──► retry (jittered backoff) or failed
```

Every arrow is a compare-and-set, so two workers racing on one document means
the loser raises rather than overwriting the winner's progress.

| | |
|---|---|
| **Nothing parses in the API** | Parsing is blocking CPU work; doing it in a handler would stall every concurrent request on that process. A separate `rag-worker` process holds one job at a time, so a document that kills a parser costs one job rather than a batch. |
| **The bytes decide the type** | `Content-Type` and the filename are caller-supplied. A zip renamed `.pdf` is detected as a zip — a parser chosen from a lie is a parser handed input it never expected. |
| **Hostile input is bounded, not hoped about** | A DOCX is a zip of XML, so it arrives carrying a decompression bomb and entity expansion by default. Page caps, expansion budgets, compression-ratio limits, and `defusedxml` close each one explicitly. XXE is the serious one: a parser that resolves it makes `/etc/passwd` *searchable*. See [ADR-0010](docs/adr/0010-parsing-hostile-documents.md). |
| **Size is enforced mid-stream** | Starlette spools the whole body to disk before a handler runs, so a check in the endpoint protects nothing. An ASGI middleware counts bytes as they arrive. |
| **Blob first, then the transaction** | An orphan blob is inert and collectable; a job whose bytes do not exist is a user-visible failure. This inverts ADR-0001's ordering because a blob is *source* data, not derived. See [ADR-0009](docs/adr/0009-blob-storage.md). |
| **Extracted text is kept** | So a change to chunking is a re-chunk, not a re-parse of every document ever ingested. |
| **Re-upload is idempotent** | By content hash — 200 with the existing document, not a duplicate. Unless it previously failed, in which case it is requeued, because otherwise a user who retries after a fix gets a permanent no. |
| **Redelivery is safe** | At-least-once delivery is real: the reaper requeues jobs whose worker died. Every stage is re-runnable, and a job redelivered after success completes rather than dead-lettering. |

Chunking is recursive splitting on a separator hierarchy — paragraphs, then
lines, then sentences, then words — packed to a token target with overlap, so
boundaries land on the most meaningful break available. Offsets are tracked as
spans into the original text and are exact, which is what a citation feature
will need.

Chunks are sized either by a character-ratio estimate or by BGE-M3's real
vocabulary, selected with `RAG_INGESTION__TOKENIZER`. The estimate is the
default so a fresh clone needs no download; production should use the real one.
It is never an automatic fallback — the two produce different boundaries, and
picking one based on whether a file happened to exist would mean the same
document chunks differently on two machines with nothing logged.
`tiktoken` was rejected outright as a precise measurement of the wrong model.

Run it:

```bash
uv run rag-worker           # poll forever
uv run rag-worker --once    # drain one batch and exit
```

---

## The model service

Embeddings and reranking run in a **separate process on the GPU**, reached over
HTTP ([ADR-0004](docs/adr/0004-local-model-service.md),
[ADR-0011](docs/adr/0011-model-service-implementation.md)). The API and the
ingestion worker never import torch and never need a GPU node.

```
POST /v1/embed    {texts[], mode}          -> dense[] + sparse{indices,values}
POST /v1/rerank   {query, passages[], k}   -> [{index, score}] highest first
GET  /v1/info                              -> model, version, dims, tokenizer hash
```

| Decision | Why |
|---|---|
| **A separate process, not in-process inference** | Model weights want exactly one process per GPU; HTTP handling wants many. Four uvicorn workers would load four copies of the weights into VRAM. |
| **One GPU lock, batched by token budget** | Two concurrent forward passes on a 6 GB card is an out-of-memory error, and a CUDA OOM can poison the context for the life of the process. Activation memory scales with *tokens*, not items, so batches are formed against a token budget rather than a fixed count. |
| **Over-length input is rejected, not truncated** | A truncated chunk produces a vector that is structurally perfect and missing the end of the text. No error, undetectable from outside, permanent once indexed. |
| **Vectors are normalised by the service** | So cosine similarity is a dot product and no caller can forget. A caller that forgot would get scores that are wrong and plausible. |
| **Order is contractual, and verified** | Results pair back to chunk ids positionally. The client refuses a response whose length does not match its request — otherwise every vector after a gap attaches to the wrong chunk, and retrieval keeps working while returning unrelated text. |
| **The client fails closed** | Unlike the rate limiter. There is no degraded embedding: a document indexed with placeholder vectors is unfindable while claiming to be searchable. |
| **It never logs the text it is given** | The service is tenant-blind, so it cannot make an access decision about a log line. Counts and token totals only. There is a security test, and it was verified to fail when one debug field was added. |

`mode: query|passage` is carried even though BGE-M3 ignores it — E5 and Voyage
need it, and retrofitting it later would mean re-embedding everything.

Run it:

```bash
uv sync --extra gpu
uv run python scripts/fetch_models.py     # ~2.3 GB of weights + vocabulary
uv run rag-model-service                  # port 8001
```

**Without a GPU**, `MODEL_SERVICE_BACKEND=stub` serves the identical contract
with deterministic fake vectors and no torch. It is not a mock — it is a real
implementation of the same interface, which is how the routes, limits, batching
and error paths are tested in CI. It reports `embedding_model: "stub"`, and that
string is stamped into every chunk row it produces, so a corpus embedded by
accident says so in the database.

---

## Retrieval

`POST /api/v1/search` embeds the query, filters inside the vector index, and
hydrates the results from Postgres ([ADR-0012](docs/adr/0012-vector-index-and-retrieval.md)).

```
query ──► model service ──► Qdrant (pre-filtered) ──► Postgres (re-checked)
             embed              tenant + ACL              text + title
```

| Decision | Why |
|---|---|
| **The access check is a pre-filter** | It runs *inside* the vector query. Post-filtering corrupts recall — asking for the top 50 and discarding 40 leaves 10 results, not the true top 10 — and the vectors were already read before the check, so the disclosure has happened. |
| **One filter builder, no caller-supplied filters** | Postgres has row-level security; a query that forgets its tenant scope finds zero rows. Qdrant has nothing equivalent — a forgotten tenant clause returns every tenant's vectors with a 200. So there is exactly one function that builds a filter and it always emits both clauses. |
| **The index holds no text** | Results are hydrated from Postgres, which re-applies the same access filter under RLS. That makes index drift a recall bug rather than a disclosure: a stale point matches, no permitted row comes back, the result vanishes. |
| **An ACL change rewrites the payload, not the vectors** | Permissions do not change a vector, only who may match it. A grant costs a payload write, not a GPU pass over the document. |
| **Sparse vectors are written now, queried in M6** | A collection's vector configuration is fixed at creation and BGE-M3 produces both in one pass. Storing dense only would make hybrid retrieval a full re-embed of the corpus later. |
| **`READY` means indexed** | The temporary `CHUNKING → READY` edge from M3 is gone. A document that reached `READY` without vectors would be invisible to retrieval while claiming to be searchable, with nothing raised anywhere. |

Search is dense-only until M6. Reranking — the model service already serves it —
arrives in M7.

---

## Architecture

```
┌─────────────── SERVING PLANE (CPU) ────────────┐
│   FastAPI × N   ──────────┐                    │
└──────┬───────────┬────────┼────────────────────┘
       │           │        │ HTTP
       ▼           ▼        │
  ┌────────┐  ┌────────┐    │      ┌──────────────────────────┐
  │Postgres│  │ Redis  │    ├─────►│   MODEL SERVICE (GPU)    │
  │(truth) │  │(cache) │    │      │   BGE-M3 → dense+sparse  │
  └───┬────┘  └────────┘    │      │   bge-reranker-v2-m3     │
      │ jobs                │      └──────────────────────────┘
      │ SKIP LOCKED         │
┌─────▼──────────────┐      │             ┌─────────┐
│ INGESTION WORKERS  │──────┴────────────►│ Qdrant  │
│      (CPU) × M     │                    │ (index) │
└────────────────────┘                    └─────────┘
```

Two planes, deliberately: ingestion is bursty, CPU-heavy, and latency-tolerant;
query serving is latency-critical. Running them in one process means a single
4,000-page PDF stalls every open answer stream.

### Read path

```
request → auth → rate limit → cache?
   → query rewrite (multi-turn only)
   → dense + sparse search (one Qdrant collection, one ACL filter)
   → reciprocal rank fusion
   → cross-encoder rerank (GPU)
   → prompt assembly (cacheable frozen prefix)
   → LLM, streamed
   → citation validation
   → SSE to client
```

### Layering

```
rag.api        HTTP delivery. Thin: parse, authorise, call a service, serialise.
rag.services   Use cases. Depends on ports, never on concrete adapters.
rag.domain     Pure models, business rules, and ports (Protocols). No I/O.
rag.adapters   The only place third-party clients are imported.
rag.db         Persistence. Postgres is not a swappable detail here.
rag.core       Config, logging, errors, request context. Depends on nothing of ours.
```

These boundaries are enforced by `import-linter` in CI, not by convention. See
[ADR-0003](docs/adr/0003-ports-and-adapters.md).

---

## Quick start

Requires **Python 3.12+**, **Docker**, and [**uv**](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd enterprise-rag
cp .env.example .env

uv venv
uv pip install -e ".[dev]"

# Backing services
docker compose -f docker/compose.yml up -d postgres redis qdrant

# The API
uv run uvicorn rag.api.asgi:app --reload
```

Then:

```bash
curl localhost:8000/health   # liveness  — is the process up?
curl localhost:8000/ready    # readiness — can it serve traffic?
open  localhost:8000/docs    # OpenAPI (disabled in production-like envs)
```

### Getting a credential

Every other endpoint needs one, and a fresh database has no tenant and no user.
That first account is created from the command line, not from an endpoint: an
unauthenticated tenant-creating route is a permanent liability guarded by a
secret that is one empty environment variable away from being nothing.

```bash
uv run rag-admin create-tenant --slug acme --name "Acme Corp"
uv run rag-admin create-user --tenant acme --email you@acme.example --role owner

curl -s localhost:8000/api/v1/auth/login \
  -H 'content-type: application/json' \
  -d '{"tenant_slug":"acme","email":"you@acme.example","password":"…"}'
# → {"access_token":"eyJ…","refresh_token":"ragr_…", …}

curl localhost:8000/api/v1/auth/me -H "Authorization: Bearer eyJ…"
```

`rag-admin generate-key` prints a signing key. Without one, local development
generates an ephemeral keypair at boot and logs a warning — tokens then die on
restart, which is why no key is committed here. In staging or production a
missing key is a startup failure.

Or run the whole stack in containers: `docker compose -f docker/compose.yml up -d --build`.

### Commands

`make help` lists everything. On Windows without `make`, run these directly:

| Task | Command |
|---|---|
| Lint | `uv run ruff check .` and `uv run ruff format --check .` |
| Auto-fix | `uv run ruff check --fix .` and `uv run ruff format .` |
| Type check | `uv run mypy` |
| Architecture contracts | `uv run lint-imports` |
| Tests | `uv run pytest --cov=rag --cov-report=term-missing` |
| Everything CI runs | all five of the above |
| Model service (no GPU) | `MODEL_SERVICE_BACKEND=stub uv run rag-model-service` |
| Model service (GPU) | `uv sync --extra gpu && uv run rag-model-service` |
| Fetch the tokenizer only | `uv run python scripts/fetch_models.py --tokenizer-only` |

The model-service integration tests skip when nothing is listening on port 8001.
Start the stub first to run them; CI does exactly that.

---

## Configuration

Settings are nested and bound from the environment with a **double** underscore
delimiter. `RAG_DATABASE__HOST` works; `RAG_DATABASE_HOST` silently does nothing.
See [`.env.example`](.env.example) for the full surface.

Two properties worth knowing:

- **Secrets are `SecretStr`.** They render as `**********` in reprs, tracebacks,
  and log dumps.
- **Production-like environments fail fast.** Booting with `RAG_ENVIRONMENT=prod`
  while still carrying the development database password raises at startup — as
  does a missing signing key or a plaintext `http://` issuer. A
  misconfiguration should be undeployable, not discovered in an audit.

---

## Observability

Every response carries `X-Request-Id`, `X-Trace-Id`, and `X-Response-Time-Ms`.
Every log line — including those from uvicorn and SQLAlchemy — is routed through
structlog and stamped with the same ids. Every error response is an RFC 9457
problem document containing the trace id:

```json
{
  "type": "https://docs.enterprise-rag.dev/errors/not_found",
  "title": "Not Found",
  "status": 404,
  "detail": "Document 'doc-123' was not found.",
  "instance": "/api/v1/documents/doc-123",
  "code": "not_found",
  "trace_id": "9f2c1a7e4b6d..."
}
```

`code` is a stable contract clients may branch on; `detail` is free to change.

### Liveness vs readiness

`/health` checks nothing external and answers "is this process alive?".
`/ready` probes registered dependencies concurrently, each with its own timeout,
and answers "can it serve traffic?". Orchestrators *restart* on liveness failure
but only *de-register* on readiness failure — conflating them turns a five-second
database blip into a rolling restart of the entire fleet.

Dependencies may be required (Postgres) or optional (Redis, whose loss costs
caching but not correctness).

---

## Roadmap

| | Milestone | State |
|---|---|---|
| M0 | Foundations — config, logging, errors, health, Docker, CI | **done** |
| M1 | Data model — tenants, documents, chunks, jobs; RLS; migrations | **done** |
| M2 | Auth & RBAC — JWT, API keys, roles, tenant scoping, rate limiting | **done** |
| M3a | Ingestion — upload, blob store, worker, chunking, state machine | **done** |
| M3b | Parsers — PDF, DOCX, and hostile-input hardening | **done** |
| M4 | Model service — BGE-M3 + reranker on GPU, batching | **done** |
| M5 | Dense retrieval — Qdrant, ACL pre-filter, `/search` | **done** |
| M6 | Hybrid retrieval — sparse vectors, reciprocal rank fusion | |
| M7 | Reranking — cross-encoder, circuit breaker | |
| M8 | Generation — LangGraph, streaming, validated citations | |
| M9 | Performance & cost — caching, prompt caching, cost ledger | |
| M10 | Evaluation — golden set, metrics, CI regression gate | |
| M11 | Observability — OpenTelemetry, Langfuse, Prometheus | |
| M12 | Hardening — threat model, security suite, load test | |
| M13 | Deployment & documentation | |

Deliberately **out of scope**: a production frontend, model fine-tuning,
Kubernetes manifests, SSO/SAML. Scope discipline is a feature.
