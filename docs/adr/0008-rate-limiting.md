# ADR-0008 — Rate limiting: a port now, an in-process bucket now, Redis in M9

- **Status:** Accepted
- **Date:** 2026-08-10
- **Milestone:** M2 (port + in-process adapter), M9 (Redis adapter)

## Context

M2 owns per-tenant rate limiting. Redis is not a dependency until M9 — its
configuration exists, its adapter does not, and CLAUDE.md provides it locally
through `fakeredis` rather than a server.

Shipping nothing until M9 would leave the login endpoint — the
credential-stuffing surface, and the only unauthenticated write in the system —
unprotected for seven milestones.

## Decision

Define `RateLimiter` in `rag.domain.ports`, ship an **in-process token bucket**
in M2, and add a Redis adapter behind the same port in M9. No call site changes.

Two buckets, keyed differently, because they defend different things:

| | Key | Default | Defends |
| --- | --- | --- | --- |
| Tenant | `tenant:<uuid>` | 600/min, burst 120 | per-tenant capacity and cost |
| Login | `login:<client ip>` | 10/min, burst 5 | credentials |

## Rationale

**A token bucket, not a fixed window.** A fixed window permits a 2× burst across
the boundary — a hundred requests at `23:59:59.9` and a hundred more at
`00:00:00.1` — which is precisely the shape of a cost spike on an embedding
endpoint. A sliding-window log gives the same guarantee for O(n) memory per key
and offers no natural `Retry-After`.

**Bursting is desirable, not merely tolerated.** A page firing six requests as it
loads is not abuse, so capacity deliberately exceeds one second of refill.

**Keyed on the tenant, not the client address.** The resource being protected is
per-tenant capacity and cost. An address is meaningless behind NAT and free to
rotate. The honest consequence is that one noisy user can throttle a colleague:
this limiter promises fairness *between* tenants, not within one.

**Login needs its own bucket.** A per-tenant limiter cannot protect a
pre-authentication endpoint — there is no verified tenant until the login
succeeds, which is exactly what the attacker is trying to make happen. So login
is keyed on the client address, with a much smaller bucket.

**`request.client.host` only — never `X-Forwarded-For`.** That header is
caller-controlled unless a trusted proxy overwrites it. Trusting it would let an
attacker reset their own bucket by inventing a new value per attempt, producing
a limiter that is *worse* than none because it looks like protection.
Proxy-aware client resolution belongs at the ingress, via uvicorn's
`--proxy-headers` and a trusted-host list.

**Fail open, loudly.** A limiter that cannot answer admits the request and logs
at error level. Losing rate limiting costs fairness; failing closed on a limiter
outage costs the entire API. This is the same posture as
`RedisSettings.required_for_readiness`, which is `False` for exactly this reason.

**A dependency, not middleware.** The tenant limiter needs the resolved tenant,
which only exists after authentication. Middleware runs before routing and would
have to duplicate token parsing to get it.

## Consequences

**We get:** a working limiter from M2, a Redis swap that touches one wiring line,
and 429 responses carrying `Retry-After` and `X-RateLimit-*` so a client can
back off without guessing.

**We pay:**

- **The limit is per worker process.** With N uvicorn workers the effective
  limit is N×. `ServerSettings.workers` defaults to 1 and the deployment story
  is replica scaling, so this is exact locally and in a single-process
  container, and approximate anywhere else. Stated here rather than discovered
  later; M9 makes it exact.
- A rejected authenticated request has already paid for token verification and
  one database round trip. Acceptable: volumetric abuse is the ingress's job,
  and this limiter exists for fairness and cost control.
- Buckets are held in memory, and the login key space is attacker-controlled.
  Mitigated by sweeping **fully refilled** buckets rather than old ones: a full
  bucket and an absent one are indistinguishable, since a missing key is
  recreated at capacity. A bucket under continuous load never refills and is
  therefore never dropped — which is correct, because it is the one still doing
  work.

## A detail worth not re-discovering

`RateLimitExceededError` subclasses `QuotaExceededError` rather than reusing it.
Both are 429, and the status is inherited through the MRO walk in
`rag.api.errors.status_for` with no mapping change — but "slow down and retry in
four seconds" and "you have used your document allowance for the month" demand
completely different client behaviour. This is the same argument that split
`InvalidStateTransitionError` from `ConcurrentModificationError` in M1.

`Retry-After` is rounded up and never zero. `Retry-After: 0` invites a client to
retry immediately into the same denial, which turns a rate limiter into a
retry-storm amplifier.

## Alternatives considered

| Option | Assessment |
| --- | --- |
| **Wait for Redis in M9** | Leaves login unprotected for seven milestones. The port makes waiting unnecessary. |
| **Fixed window** | Simplest, and permits a 2× boundary burst on the most expensive endpoints in the system. |
| **Sliding-window log** | Same guarantee, O(n) memory per key, no natural `Retry-After`. |
| **Nginx/ingress limiting only** | Right layer for volumetric abuse and wrong layer for per-tenant fairness — the ingress cannot see a verified tenant. The two are complementary, not alternatives. |
| **`fakeredis` in M2** | Would test the Redis adapter without shipping one, and put a fake on the production path. |
