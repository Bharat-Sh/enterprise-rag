# ADR-0002 — Job queue in Postgres via `SELECT ... FOR UPDATE SKIP LOCKED`

- **Status:** Accepted
- **Date:** 2026-07-30
- **Milestone:** M0 (design), M3 (implementation)

## Context

Document ingestion is asynchronous: upload returns immediately, and a worker
pool parses, chunks, embeds, and indexes in the background. That needs a job
queue. The reflexive choices are Celery (with Redis or RabbitMQ) or an
async-native equivalent such as `arq`.

## Decision

Jobs live in a Postgres table, consumed with:

```sql
SELECT * FROM jobs
 WHERE status = 'QUEUED' AND run_after <= now()
 ORDER BY priority DESC, created_at
 FOR UPDATE SKIP LOCKED
 LIMIT 1;
```

`SKIP LOCKED` lets concurrent workers claim different rows without blocking on
each other.

## Rationale

**Job state and document state are the same state.** Creating a document row and
enqueuing its ingestion job happens in one transaction. That eliminates an
entire class of bug rather than mitigating it:

- A job whose document does not exist: impossible.
- A document with no job, silently stuck forever: impossible.

With an external broker, these are the two halves of the dual-write problem, and
the standard fix — a transactional outbox plus a relay process — is strictly more
machinery than the thing we are avoiding.

Secondary benefits:

- Retry counts, backoff schedules, failure reasons, and dead-letter status are
  columns. Inspecting the queue is `SELECT`, not a broker-specific CLI.
- One less service in the local development stack.
- Exactly-once-ish semantics for free: a worker that dies mid-job releases its
  row lock, and the job is re-claimable after a visibility timeout.

## Consequences

**We get:** transactional consistency, trivial observability, fewer moving parts,
and dead-letter handling as a `WHERE` clause.

**We pay:** throughput is bounded by Postgres — realistically thousands of jobs
per second, which is several orders of magnitude beyond this workload. It also
adds write load to the same database serving queries; if that becomes a problem
the queue moves to its own Postgres instance long before it needs a broker.

## Alternatives considered

| Option | Assessment |
|---|---|
| **Celery** | The expected answer, and battle-tested. Sync-first with an awkward async story, heavyweight for our needs, and reintroduces the dual-write problem. |
| **arq** | Async-native, lightweight, Redis-backed. A good choice. Rejected only because Redis is a cache here — a degradable dependency — and promoting it to the durability path for ingestion changes its failure profile. |
| **SQS / Cloud Tasks** | The right answer in a managed cloud deployment. Cannot be run locally, which matters for a repository meant to be cloned and started in five minutes. |

## Note

This is a deliberately contrarian pick and is documented as such so it reads as
a decision rather than an oversight. The migration path is clear and the trigger
is measurable: when queue write load meaningfully affects query latency, move it.
