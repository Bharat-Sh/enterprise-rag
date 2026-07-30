# Enterprise RAG Platform

A multi-tenant, production-shaped retrieval-augmented generation platform:
documents in, grounded and cited answers out, with access control, evaluation,
and observability treated as features rather than afterthoughts.

> **Status: M0 — Foundations.** The service boots, logs, fails correctly, and is
> containerised. It has no business logic yet. Milestones are listed below.

---

## Why this exists

Wiring an embedding model to a vector database is a weekend. The parts that make
a RAG system deployable inside a company are the parts most demos skip:

| Concern | How it is handled here |
|---|---|
| **Tenant isolation** | Access control is a *pre-filter pushed into the vector query*, never a filter applied to results after retrieval. Post-filtering both corrupts recall and means the data was already read. |
| **Ingestion durability** | Documents move through an explicit state machine persisted in Postgres, with content-hash idempotency, bounded retries, and a dead-letter state. |
| **Index rebuildability** | Postgres is the system of record; Qdrant is a derived index that can be dropped and rebuilt from it at any time. |
| **Answer trust** | Every citation is validated against the actually-retrieved chunk set before it reaches the client. A claimed citation is worth nothing. |
| **Change safety** | A golden evaluation set gates CI: a pull request that hurts retrieval quality fails to merge. |
| **Debuggability** | One trace id spans the HTTP request, the retrieval graph, and the LLM call. A user quotes it; we reconstruct everything. |

Architectural decisions — including the ones where the obvious alternative was
rejected, and why — live in [`docs/adr/`](docs/adr/).

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
  while still carrying the development database password raises at startup. A
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
| M1 | Data model — tenants, documents, chunks, jobs; migrations | next |
| M2 | Auth & RBAC — JWT, roles, tenant scoping, rate limiting | |
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
