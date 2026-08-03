# ADR-0006 — Document ACLs as a denormalised principal array

- **Status:** Accepted
- **Date:** 2026-08-03
- **Milestone:** M1 (schema), M5 (vector-store filter)

## Context

Row-level security (ADR-0005) isolates *tenants*. Within a tenant, individual
documents still need per-user and per-group permissions.

The binding constraint is not Postgres — it is Qdrant. In M5 the access check
must run **inside the vector search**, as a pre-filter. Qdrant payload filters
support `match`, `match_any`, and boolean combination. They cannot join,
subquery, or expand a group hierarchy.

Post-filtering the results instead is not an option, for two independent
reasons: it corrupts recall (asking for the top 50 and discarding 40 leaves 10,
not the true top 10), and the vectors and payloads were already read before the
check — the disclosure has happened.

So the ACL must reach the vector store already flattened.

## Decision

Split the representation on *what changes when*.

**Stored on the document (stable)** — a `text[]` of opaque principal tokens,
denormalised onto chunks and shipped into the Qdrant payload:

```
acl_principals = ["user:a1b2…", "group:eng", "tenant:acme"]
```

**Computed per request (volatile)** — the caller's principal set, assembled
from the verified token:

```
["user:a1b2…", "group:eng", "role:admin", "tenant:acme"]
```

Access is granted when the two sets intersect:

| Store | Operator |
| --- | --- |
| Postgres | `acl_principals && :caller_principals` (GIN-indexed) |
| Qdrant | `match_any` on the payload field |

The same decision procedure in both stores — not two similar-looking ones.

## Rationale

**Group membership costs nothing.** A document's ACL names *groups*, not the
users currently in them. Adding someone to a group changes only the caller's
side of the intersection, computed fresh each request. Had we expanded group
membership onto documents, a single membership change would rewrite the ACL of
every document that group can see and re-index all of them — a re-index storm
triggered by an HR action.

**One definition of "may this caller see this".** `AccessFilter` is built once
per request and passed unchanged to the SQL repositories and the vector store.
Two access checks that must agree but are written separately will eventually
disagree, and the disagreement is a leak.

**The array is GIN-indexable**, so the predicate is an index scan rather than a
sequential filter — this runs on the hottest path in the system.

## Why keep the normalised table too

`document_permissions` remains the source of truth for grants. The array cannot
answer "which documents can this group see?", cannot be audited row by row, and
cannot record who granted access or when. The array serves the read path; the
table serves management and audit. `set_acl()` rewrites both, plus the copy on
every chunk, in one transaction.

## Consequences

**We get:** a pre-filter that works identically in both stores, free group
membership changes, and defence in depth — `ChunkRepository.get_many` re-applies
the filter after a vector search, so a stale index cannot surface text the caller
may not read.

**We pay:**

- Three representations to keep consistent (grants, document array, chunk array).
  Mitigated by rewriting all three in one transaction and by a test asserting an
  ACL change reprojects onto chunks. A reconciliation job belongs in M12.
- Denormalised arrays on `chunks` duplicate the parent's data. Deliberate: it
  removes a join from the retrieval path.
- Revocation takes effect on the next request, not mid-request. Correct
  behaviour, but worth stating.
- Flat groups only. Nested groups need recursive expansion when building the
  caller's set — a Postgres recursive CTE, cheap, but scope we do not need yet.

## Alternatives considered

| Option | Assessment |
| --- | --- |
| **Post-filter search results** | The obvious approach and the wrong one. Corrupts recall and reads the data before checking. Rejected outright. |
| **Expand users onto documents** | Makes the Qdrant filter a single `match`, marginally simpler. Rejected: every group membership change becomes a mass re-index. |
| **Separate permission service queried per result** | Correct and flexible; a network round trip per candidate chunk destroys the latency budget. |
| **Postgres-only filtering, no vector pre-filter** | Would mean retrieving unfiltered vectors and joining in SQL — the post-filter problem with extra steps. |
