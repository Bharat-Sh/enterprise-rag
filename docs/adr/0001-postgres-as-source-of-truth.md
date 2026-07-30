# ADR-0001 — Postgres is the source of truth; Qdrant is a derived index

- **Status:** Accepted
- **Date:** 2026-07-30
- **Milestone:** M0

## Context

A RAG system holds two kinds of state: relational metadata (tenants, users,
roles, documents, chunks, jobs, conversations, usage) and vector embeddings.
Vector databases offer a `payload` field that can store arbitrary metadata
alongside each vector, which makes it tempting to use one system for both.

## Decision

PostgreSQL holds all metadata and is the system of record. Qdrant holds vectors
plus the minimum payload needed for filtering, and is treated as a **derived,
disposable index** that can be dropped and rebuilt from Postgres at any time.

Write ordering is fixed:

1. Commit chunk rows to Postgres.
2. Upsert vectors to Qdrant using deterministic point IDs derived from chunk IDs.
3. Mark the document `READY`.

## Consequences

**We get:**

- Transactions, foreign keys, and joins for data where integrity is a security
  property. Tenant/ACL relationships without referential integrity is not a
  convenience problem, it is a data-leak problem.
- Schema migrations via Alembic, with review and rollback.
- A crash between steps 1 and 2 is recoverable: retry re-upserts idempotently.
  A crash after step 2 but before step 3 is likewise safe.
- Embedding-model migration becomes tractable — re-embed from stored chunk text
  into a new named vector, then cut over. No re-parsing, no data loss.
- Operational answers are SQL queries: "how many documents are stuck in
  EMBEDDING?" needs no special tooling.

**We pay:**

- Two systems to operate and keep consistent.
- Chunk text is stored twice (Postgres row and, optionally, Qdrant payload).
- A reconciliation job is needed to detect drift between the two.

## The inverted alternative, and why it fails

Storing metadata only in Qdrant payloads removes a service, and is genuinely
simpler on day one. It breaks down at exactly the points that matter:

- **No transactions.** A partial write leaves orphan vectors with no way to tell
  whether the corresponding document exists.
- **No joins.** "Which documents can this user see?" becomes an application-side
  N+1 across two systems.
- **No migrations.** Payload schema changes are a bespoke backfill script.
- **The index becomes unrebuildable.** Once the vector store is the only copy of
  the truth, you can never safely drop it — which means you can never change
  chunking strategy or embedding model without risking permanent data loss.

That last point is the decisive one. The whole value of "derived index" is that
you are free to rebuild it.

## Note

If we ever ran at a scale where operating both is a burden, `pgvector` collapses
this into a single system and keeps every property above. See ADR-0003 — the
`VectorStore` port exists partly so that remains a realistic option.
