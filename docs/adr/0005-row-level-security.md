# ADR-0005 — Tenant isolation enforced by PostgreSQL row-level security

- **Status:** Accepted
- **Date:** 2026-08-03
- **Milestone:** M1

## Context

Every table holding customer data needs to be invisible to other customers. The
standard approach is a `tenant_id` column plus `WHERE tenant_id = ?` in every
query. It works, it is fast, and it is one forgotten clause away from a breach.

## Decision

Both. The `tenant_id` column is the mechanism; row-level security is the
enforcer. Applied to all seven tenant-scoped tables:

```sql
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE  ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON documents
    USING      (tenant_id = NULLIF(current_setting('rag.tenant_id', true), '')::uuid)
    WITH CHECK (tenant_id = NULLIF(current_setting('rag.tenant_id', true), '')::uuid);
```

The scope is bound per transaction by `UnitOfWork.scope_to_tenant()`, which
issues `set_config('rag.tenant_id', ..., is_local => true)`.

## Rationale

**The failure modes are asymmetric, and that is the entire argument.**

| | Forgotten filter yields |
| --- | --- |
| Application-side `WHERE` only | **every row, in every tenant** — a silent breach |
| Row-level security | **zero rows** — an obvious, loud bug |

Both are mistakes. Only one is a security incident. A control that fails closed
is worth real complexity; one that fails open is worth almost none.

## Three details that decide whether this is security or decoration

1. **`FORCE ROW LEVEL SECURITY`.** Table *owners* bypass RLS by default, and the
   application connects as the role that owns these tables. With `ENABLE` alone
   every policy is inert while still appearing in `pg_policies` — the security
   review passes and the security does not exist. A test asserts
   `relforcerowsecurity` for every protected table.

2. **`SET LOCAL`, never `SET`.** `SET` persists for the life of the *connection*.
   Return that connection to the pool and the next request inherits the previous
   tenant's scope — strictly worse than no RLS, because it looks safe.
   `SET LOCAL` is discarded at transaction end, so pooled connections are always
   handed back clean. The consequence is that `commit()` must reapply the scope,
   which the unit of work does.

3. **`WITH CHECK` as well as `USING`.** `USING` governs reads, `WITH CHECK`
   governs writes. With `USING` alone, a session scoped to tenant A can INSERT a
   row stamped tenant B — invisible to A afterwards, quietly corrupting B, and
   raising nothing.

## Consequences

**We get:** a control that survives developer error, a new tenant-scoped table
that is protected by adding one name to a list, and cross-tenant tests that
issue deliberately unfiltered SQL and still return nothing.

**We pay:**

- Every transaction must bind a scope. Forgetting yields zero rows, which is
  safe but occasionally confusing; `current_tenant_scope()` exists to diagnose it.
- Policy predicates are re-evaluated per row. With `tenant_id` indexed the
  planner handles it well, but the predicate is not always pushed down as far as
  a hand-written `WHERE` — a real cost on large scans.
- `jobs` is exempt, because workers poll across tenants by design. No HTTP
  endpoint exposes that table, so the exemption does not widen the API surface.

## Alternatives considered

| Option | Assessment |
| --- | --- |
| **Column filter only** | Simpler and faster, and what most systems do. Rejected on the asymmetry above: the cost of the mistake is unbounded. |
| **Schema per tenant** | Strongest isolation and impossible to forget. Rejected on operations: migrations multiply by tenant count and connection pools fragment past a few dozen tenants. Right for a handful of large enterprise customers; wrong for SaaS. |
| **Database per tenant** | As above, more so. Appropriate at the point a single customer justifies dedicated infrastructure. |
| **Separate low-privilege app role** | A genuine refinement — a non-owner role would make `FORCE` unnecessary and reduce blast radius. Deferred to M12 (hardening) because it needs a migration/runtime role split we do not have yet. Noted so the omission is a decision. |
