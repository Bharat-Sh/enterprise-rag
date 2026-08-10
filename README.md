# Enterprise RAG Platform

A multi-tenant, production-shaped retrieval-augmented generation platform:
documents in, grounded and cited answers out, with access control, evaluation,
and observability treated as features rather than afterthoughts.

> **Status: M2 — Authentication and RBAC.** The service boots, logs, fails
> correctly, has a tenant-isolated schema, and is now closed: every endpoint
> authenticates, and the row-level-security scope is bound from the verified
> credential before any handler runs. Retrieval arrives in M5. Milestones below.

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
| M3 | Ingestion — upload, job queue, parsing, chunking, state machine | |
| M4 | Model service — BGE-M3 + reranker on GPU, batching | |
| M5 | Dense retrieval — Qdrant, ACL pre-filter, `/search` | |
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
