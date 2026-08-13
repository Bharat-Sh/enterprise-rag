# ADR-0012 — The vector index: isolation without a database to enforce it

- **Status:** Accepted
- **Date:** 2026-08-13
- **Milestone:** M5
- **Implements:** ADR-0001 (Qdrant as a derived index) and ADR-0006 (the ACL
  pre-filter), both of which were written anticipating this milestone.

## Context

M5 is the first feature where **the database is not enforcing access control.**

Everywhere else in this system, a query that forgets its tenant scope returns
zero rows, because row-level security is enforced by Postgres and applies
whether or not the application remembers (ADR-0005). Qdrant has no equivalent. A
vector query with no tenant clause returns every tenant's vectors, with a 200,
and nothing anywhere objects.

That single asymmetry drives most of what follows.

---

## 1. One collection, and a filter that cannot be omitted

One Qdrant collection, tenant isolation by payload filter, with the `tenant_id`
payload index declared `is_tenant=True` so Qdrant co-locates each tenant's
points in storage rather than merely filtering across them.

**Rejected: a collection per tenant.** Qdrant collections are heavyweight —
each carries its own segments and threads — so a few thousand tenants is
untenable, and creating one becomes part of tenant provisioning with its own
half-created failure states.

The structural defence is that there is exactly one function that builds a
filter, `adapters/vectorstore/filters.py::build_search_filter`, it takes an
`AccessFilter` as a required positional argument, and it always emits both
clauses. The adapter accepts no caller-supplied filter to merge in; the
narrowing arguments (`collection_id`, `document_ids`) are typed and specific,
which is what stops "let the API pass a filter through" from ever looking
reasonable. `VectorStore` has no read method that does not take an
`AccessFilter`, the same trick `DocumentRepository` already uses.

`must`, never `should`: `should` is disjunction, so moving either mandatory
clause there would make a match on the *other* sufficient — one word away from
"any principal token grants access to every tenant".

## 2. Sparse vectors are written in M5 and queried in M6

The collection is created with **both** named vectors from day one. Ingestion
writes both. Search uses dense only; M6 adds the sparse query and RRF fusion.

This deviates from the roadmap, which listed sparse as an M6 concern, and the
reason is that a Qdrant collection's vector configuration is **fixed at
creation**. BGE-M3 produces dense and sparse in one forward pass — the sparse
weights are already computed and already in the response. Storing only dense
would mean M6 could not add sparse without a new collection and a re-embed of
the entire corpus, to store something we had and discarded.

So the choice was not "extra work now versus later", it was "a few bytes now
versus a guaranteed full re-index later".

**Rejected: implement fusion now.** Qdrant's Query API makes it easy — server-
side `FusionQuery` — which is exactly the trap. Whether hybrid helps, and how to
weight it, is an *evaluation* question that needs M10's golden set. Shipping
fusion with untuned parameters and no way to measure it is how a retrieval
system acquires a knob nobody dares touch.

## 3. The index holds no text

The payload carries only what a filter needs: `tenant_id`, `acl_principals`,
`document_id`, `collection_id`, `ordinal`, `embedding_model`. Results are
hydrated from Postgres by `ChunkRepository.get_many`, which re-applies the same
`AccessFilter` under row-level security.

This is a security property, not a storage saving, and it produces two things:

**Drift cannot disclose.** If the index falls behind — a failed reprojection, a
partial write, a restored snapshot — a stale point matches, hydration finds no
permitted row, and the result disappears. An orphaned vector is a recall bug,
never a disclosure. `tests/security/test_search_isolation.py` asserts exactly
this by narrowing an ACL in Postgres alone and leaving the index untouched.

**Hydration is not overhead being tolerated**, it is the mechanism that makes
the second check possible. Copying text into the payload would trade that away
for one saved query, and would put every customer's content in a second system
that has no row-level security.

## 4. What the tests can and cannot prove

Writing the security suite produced the most useful finding in this milestone,
and it is worth recording because it is counter-intuitive.

**Deleting the tenant clause from `build_search_filter` leaves every HTTP-level
isolation test green.** Verified, not assumed. Hydration reads under row-level
security, so Postgres drops the foreign rows regardless — from outside, an
unfiltered index is indistinguishable from a filtered one.

That is defence in depth working. It is also a trap: a suite made only of
end-to-end tests would have "proved" isolation while the primary control was
absent. So the vector store's own isolation is asserted **at the store**, in
`tests/security/test_vector_isolation.py`, and that suite does fail when the
clause is removed.

It also documents why the clause matters even though Postgres would catch it:

- **Embeddings are derived from text.** Returning another tenant's vectors
  discloses them to the process. Embedding-inversion attacks reconstruct
  approximate source text. ADR-0006's objection to post-filtering is precisely
  that the data has already been read by then.
- **It silently destroys recall.** Foreign vectors consume slots in the top-k,
  so a caller asking for ten results gets however many of their own survive —
  a correctness bug with no error attached.
- **It is the only control if hydration ever changes.** A cache (M9), a payload
  optimisation, or any future path that trusts the index removes the backstop.

A related finding, also from the tests: `role:` principal tokens are the **same
string in every tenant** — `role:owner` here is `role:owner` there — because
unlike users and groups their ids are not UUIDs. A document shared with all
owners therefore carries a principal that other tenants' owners also hold, and
the ACL clause genuinely matches it. That is the one ACL shape where the tenant
clause is the only thing standing between two tenants.

## 5. An ACL change rewrites the payload, not the vectors

`documents.set_acl` already rewrote three representations (ADR-0006: the grants,
the document array, the copy on every chunk). M5 adds a fourth —
`VectorStore.set_acl` — using Qdrant's payload update.

Nothing is re-embedded: **an ACL change does not change a vector**, only who may
match one. So a permission change costs a payload write rather than a GPU pass
over the whole document.

Leaving it out would have been a one-directional failure, which is the sort that
survives review. Revocation still works, because hydration re-checks Postgres.
*Granting* does not: the pre-filter never surfaces the chunk, so a newly-shared
document stays invisible with nothing logged.
`test_granting_access_makes_a_document_findable` exists to keep that direction
honest.

`set_payload`, not `overwrite_payload`: the latter replaces the whole payload
and would drop `tenant_id`, removing the very field isolation matches on.

## 6. Ordering, batching, and idempotency

Chunks commit to Postgres before any vector is written (ADR-0001). The derived
index must never hold something the database cannot explain.

Embedding and indexing run as one batched loop rather than embed-all-then-
index-all: a 2000-page PDF is ~10k chunks, which at 1024 dimensions is ~80 MB of
dense floats held at once in a worker that is also holding the document text.
The document status still moves through `EMBEDDING` and `INDEXING` separately,
because an operator looking at a stuck document needs to know which dependency
it is stuck on.

Point id **is** chunk id, so re-indexing overwrites rather than duplicating.
That is what makes the write path safe to retry with no bookkeeping, and it is
what stops at-least-once delivery from inflating the index a little on every
crash.

Deletion removes vectors before rows. By filter, so purging never depends on
first reading the chunks it is about to delete.

## 7. `CHUNKING → READY` is gone

The temporary M3 edge is deleted, along with `TestTemporaryEdgeForM3`, which
existed to assert it was present so that removing it could not be quietly
forgotten. `TestReadyImpliesIndexed` replaces it and asserts the opposite.

`READY` now means "there are vectors" by construction rather than by convention.
The reason this mattered: a document reaching `READY` without vectors is
invisible to retrieval while claiming to be searchable, and nothing errors.

## 8. Readiness stays optional — reversing the M4 note

The note left in `rag/api/main.py` at the end of M4 said M5 would flip
`model-service` to `required=True` once a retrieval endpoint existed. On writing
M5 that turned out to be wrong, and Qdrant is registered the same way.

Readiness governs load-balancer membership for the **whole API**. With the GPU
box or the index down, `/search` cannot answer — but upload, document
management, collections and auth all still work. Marking either required removes
every replica from the load balancer, turning "search is degraded" into "the
product is down", and it buys nothing: every replica shares one model service
and one Qdrant, so there is no healthy replica to fail over to. A 503 from
`/search` is the contained answer.

## 9. Local mode, and what it does not test

The primary development machine has no Docker (CLAUDE.md), so the vector store
runs there in `qdrant-client`'s embedded local mode. This was verified before
being designed around: local mode supports named dense **and** sparse vectors,
filtered search, delete-by-filter, sparse queries, and even server-side RRF
fusion, which M6 will need.

What it does not support is **payload indexes** — the client warns that they
have no effect. So locally every filter is a full scan: identical answers, no
index. The isolation tests therefore mean exactly the same thing against both
backings, and nothing about performance can be concluded from a local run. CI
runs the same suite against a pinned `qdrant/qdrant:v1.12.4` service container,
which is where `is_tenant` co-location and the ACL index are real.

One consequence worth knowing: a path-backed local client holds an **exclusive
lock on its storage directory**, so only one can exist per path per process.
That is why `Worker` accepts an injected `VectorStore` — an end-to-end test
driving the worker and the API together has to hand both the same store.

## Consequences

**We get:** retrieval with the access check inside the query; an index that is
disposable by contract and idempotent to rebuild; permission changes that cost a
payload write; and a test suite that distinguishes which layer is enforcing
what.

**We pay:**

- A fourth representation of every ACL, with no reconciliation job to detect
  drift between them. M12, as ADR-0006 already anticipated.
- No reindex tooling. `REINDEXING` remains a state nothing drives. The
  architectural promise of ADR-0001 holds — chunk text plus the model service is
  all a rebuild needs — but the command belongs with M12's reconciliation work.
- Search is dense-only until M6, with sparse vectors sitting written and unread.
- `max_sequence_tokens` and `vector_size` must agree with the embedding model.
  A collection created at the wrong width rejects every insert, and does so on
  the first ingestion rather than at boot.
