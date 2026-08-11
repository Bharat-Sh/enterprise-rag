# ADR-0009 — Raw bytes live in a blob store, behind a port

- **Status:** Accepted
- **Date:** 2026-08-10
- **Milestone:** M3a (port + filesystem adapter), deployment (S3 adapter)

## Context

Ingestion needs the original bytes of every document, and needs them after the
request that uploaded them has finished — the worker parses asynchronously, and
a re-chunk in a later milestone must not require the user to upload again.

ADR-0001 makes Postgres the system of record. The question this raises rather
than answers is whether "record" includes a 50 MB PDF.

## Decision

**A `BlobStore` port, a filesystem adapter in M3, an S3-compatible adapter at
deployment.** Keys are content-addressed *within a tenant*:

```
<tenant_id>/<sha256[:2]>/<sha256>          the uploaded bytes
<tenant_id>/<sha256[:2]>/<sha256>.text     the extracted text
```

Keys are constructed by the adapter (`key_for`, `derived_key_for`), never by
callers.

## Rationale

**Not `bytea` or large objects.** Postgres is the source of truth for *metadata*.
Document bodies in rows bloat the WAL, make every base backup proportional to
total corpus size, turn logical replication into a bandwidth problem, and make
`SELECT *` a landmine. None of that buys anything: the bytes are never queried,
joined, or constrained — they are fetched whole by primary key.

**Not a bare path with no port.** The filesystem adapter is explicitly wrong for
production: two API replicas do not share a disk, so an upload handled by one
and parsed by a worker on another would not find its bytes. Writing the port
first means that is one adapter and one wiring line rather than a refactor of
every call site — and it forces the interface to be storage-shaped
(`put`/`open`/`read`/`delete`, streaming) rather than filesystem-shaped.

**Content-addressed, so idempotency is free.** The same bytes produce the same
key, so re-uploading writes the same content to the same place. This is the
storage-layer twin of `uq_documents_tenant_id_content_hash`.

**Per tenant, so identical bytes are stored twice.** Deliberate. Sharing storage
across tenants would mean one customer's deletion destroys another's document,
and would make the store a cross-tenant existence oracle — "this upload was
instant, so somebody else already has this file". Duplication is the cheaper
mistake by a wide margin.

**The two-character shard** keeps directory sizes sane; some filesystems degrade
badly past a few tens of thousands of entries in one directory, and a busy
tenant reaches that.

## Write ordering: blob first, then the transaction

```
1. stream bytes to the blob store        ← if this fails, nothing was recorded
2. commit the document row + its job     ← ADR-0002: one transaction
```

Committing first would let a worker claim the job before the bytes existed,
turning a storage hiccup into a user-visible ingestion failure. Blob-first fails
the other way: an orphan blob, which is inert, invisible, and collectable by a
sweep.

This **inverts ADR-0001's Postgres-then-Qdrant ordering**, and correctly so.
Qdrant holds *derived* data that can be rebuilt from Postgres, so it must never
lead. A blob holds *source* data that cannot be rebuilt from anything, so it
must.

The orphan sweep is not built yet. Named here rather than left implicit, and
deferred to M12 with the ACL reconciliation job it resembles.

## Extracted text is stored too

Parsing is the expensive, format-specific step; chunking is cheap and is exactly
the thing we expect to tune. Keeping the extracted text means a change to chunk
size or strategy is a re-chunk, not a re-parse of every PDF ever ingested.

It goes to the blob store rather than a Postgres `TEXT` column for the same
reason as the original bytes: large, derived, never queried.

## Path safety

The adapter refuses any key that is not `[A-Za-z0-9/_.-]{1,256}`, refuses `..`
outright, and independently verifies that the resolved path is inside the store
root. Two overlapping defences for one reason: the failure is arbitrary file
read or write. Today every key comes from a UUID and a hex digest — the guard
exists for the code path somebody adds later that builds a key from a filename.

## Atomicity

Writes go to a temporary file in the destination directory and are renamed into
place. A reader can therefore never observe a partial blob. This is not
theoretical: an upload streams for seconds, and the job it enqueues can be
claimed the instant that transaction commits. `os.replace` is only atomic within
a filesystem, which is why the temporary file is created beside the destination
rather than in the system temp directory.

## Consequences

**We get:** a storage layer that can be swapped without touching ingestion, free
idempotency, tenant-isolated bytes, and torn-read safety.

**We pay:**

- The filesystem adapter is single-node. Correct locally and in a
  single-container deployment; wrong the moment there are two replicas, and the
  README says so.
- Orphan blobs accumulate until the sweep exists.
- Bytes are outside the transaction, so blob and row consistency is eventual
  rather than atomic. The ordering above makes the surviving inconsistency the
  harmless one.

## Alternatives considered

| Option | Assessment |
| --- | --- |
| **`bytea` column** | One system, transactional consistency, and free deletion. Rejected on WAL, backup, and replication cost for data that is never queried. Reasonable for a system whose documents are all small; ours are not. |
| **Postgres large objects** | Streaming and no row-size limit, but a separate API, no logical-replication support, and orphaned objects need `vacuumlo` — the same sweep problem with worse ergonomics. |
| **S3 adapter in M3** | The right production answer, and unrunnable on the primary dev machine, which has no Docker for MinIO (CLAUDE.md). The port means writing it later costs nothing that writing it now would have saved. |
| **Presigned direct-to-store upload** | Removes the API from the data path entirely, which is the correct answer at scale. Needs the object store plus a two-phase create-then-confirm flow, and would have to handle a client that never confirms. Deferred deliberately. |
| **Global content addressing (no tenant prefix)** | Deduplicates across the whole system. Rejected: cross-tenant deletion coupling, and an existence oracle. |
