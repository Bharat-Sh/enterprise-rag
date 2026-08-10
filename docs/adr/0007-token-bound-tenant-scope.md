# ADR-0007 — The credential carries the tenant, and binds the RLS scope

- **Status:** Accepted
- **Date:** 2026-08-10
- **Milestone:** M2

## Context

ADR-0005 put every table holding customer data behind `FORCE ROW LEVEL
SECURITY`, with a policy that reads `current_setting('rag.tenant_id')`. Nothing
is visible until a transaction binds a scope, and CLAUDE.md requires that scope
to come from the **verified token**, never from a request body, query parameter,
or header.

That creates a bootstrapping problem with a genuine cycle in it:

- To validate a credential we must read `users` or `api_keys`.
- Both are under row-level security, so they are invisible until a tenant is
  bound.
- The tenant is what the credential is supposed to tell us.

Every system that combines database-enforced tenancy with tenant-scoped
credentials hits this. The usual answers are all bad.

## Decision

**The credential names its own tenant, the scope is bound from that name, and
the credential is then validated *under* that scope.**

```
JWT       tid claim                    →  scope_to_tenant(tid)  →  load user by sub
API key   ragk_<tenant-b32>_<secret>   →  scope_to_tenant(t)    →  look up by SHA-256
Refresh   ragr_<tenant-b32>_<secret>   →  scope_to_tenant(t)    →  look up by SHA-256
```

For a JWT the signature is verified first, and it covers `tid`. For the opaque
credentials the tenant segment is unauthenticated input, and that is fine — it
selects which rows are visible, and the secret still has to hash to one of them.

Expressed as a dependency chain, so the ordering is enforced by the graph rather
than by discipline:

```
get_credential           header only, no I/O
    ↓
get_verified_credential  signature / alg / kid / typ / exp / aud / iss,
    ↓                    or an opaque credential's shape.  Yields a tenant id.
    ↓
get_unit_of_work         opens the transaction and IMMEDIATELY binds RLS
    ↓
get_principal            under that scope: user row, status, groups, key ceiling
```

## Rationale

**A forged tenant defeats itself.** Point a credential at another customer and
the lookup returns zero rows. There is no comparison to get wrong, no branch to
forget, and no error path that could be made to leak. The control that already
protects the data protects the authentication that guards it.

**No code path reads customer data unscoped.** Not even the authentication path.
That is a stronger statement than "we check the tenant everywhere", and it is
checkable: `get_unscoped_unit_of_work` has exactly three call sites and a test
asserts it.

**Authentication becomes structural, not remembered.** `get_unit_of_work`
depends on a verified credential, so a route cannot obtain a transaction without
one. Forgetting authentication on a new endpoint is a type error, not a silent
hole. A second test walks every registered route and fails if one resolves no
principal.

## What this rules out, and why that was the point

| Rejected | Assessment |
| --- | --- |
| **Exempt `api_keys` from RLS**, as `jobs` is exempt | The `jobs` exemption is defensible because no HTTP endpoint touches that table. `api_keys` has list, create, and revoke endpoints — it is exactly the code most in need of the backstop. |
| **A second engine or role that bypasses RLS** for the auth lookup | A permanent hole in the control, opened to solve a bootstrapping problem that has a cheaper answer. |
| **Resolve the tenant from a header or subdomain** | Caller-controlled and unverified, and it would have to be trusted *before* the credential rather than derived from it. Host-based routing is a fine ergonomic addition later; it cannot be the security mechanism. |
| **Global (tenant-free) unique index on credentials** | Would make the lookup possible without a scope, at the cost of a table that leaks across tenants by construction and a unique constraint spanning every customer. |

## Consequences

**We get:** a fail-closed authentication path, credential tables protected like
any other customer data, and an invariant that a machine checks.

**We pay:**

- Credentials are about thirty characters longer. An API key is ~85 characters,
  which is shorter than a GitHub personal access token.
- An API key discloses its own tenant id to whoever holds it — who already knows
  it.
- Login and refresh genuinely predate a verified tenant, so they need an
  unscoped transaction. That escape hatch is named `get_unscoped_unit_of_work`
  to be conspicuous, restricted to three endpoints, and asserted by test.

## Design details that follow from this

**Base32 for the tenant segment.** Case-insensitive and strictly alphanumeric,
so it cannot contain the `_` separator. Base64url is four characters shorter and
its alphabet includes `_`, which is the exact character the parser splits on.

**The secret is the last segment.** `secrets.token_urlsafe` emits `-` and `_`; a
bounded split puts every remaining underscore inside the secret, where it is
harmless.

**SHA-256, not Argon2, for machine credentials.** A password KDF compensates for
low entropy. These secrets are 256 bits from a CSPRNG: there is no dictionary to
search, and a slow hash would put 50–100 ms of CPU on the hottest authentication
path in the system. This reasoning depends entirely on the secret being
machine-generated — the day anyone proposes a user-chosen API key, the decision
inverts, and that sentence is in the code.

## Related decisions recorded here

**The access token carries identity and nothing else** — `iss`, `sub`, `aud`,
`exp`, `iat`, `nbf`, `jti`, `tid`, shaped to RFC 9068. No role, no groups.

We must load the user's row on every request anyway, to check `status` and to
assemble the principal set ADR-0006 requires. A `role` claim would therefore
save no work while creating a *second* source of truth for an authorization
input, and its failure mode is one-directional and silent: a demoted user keeps
their old role until the token expires. Groups are worse — ADR-0006's entire
argument is that a membership change takes effect immediately without
re-indexing.

**Revocation is a watermark, not a denylist.** `users.tokens_valid_after`
rejects any token issued before it. Since the row is already being read, the
check is free. A Redis denylist needs infrastructure that does not exist until
M9, is eventually consistent, and is a network hop to answer a question a column
answers exactly. The boundary is one second wide, because `iat` is a NumericDate
and carries whole seconds; accepting the ambiguous second is what stops the
replacement token issued *by* a password change from being born dead.

**Refresh tokens are opaque and rotate, with family-level reuse detection.**
Revoking a JWT refresh token requires a denylist lookup, so statefulness is paid
for either way — and this way the token's validity *is* a row, which makes
revocation exact. Presenting a spent token means two parties hold a single-use
credential, so the whole rotation family is revoked and the access-token
watermark moves.

**An API key belongs to a user and can only narrow that user's authority.** A
standalone service principal cannot participate in group ACLs —
`group_members.user_id` is a foreign key to `users`, and ADR-0006's model has no
`service:` principal type — so such a key could only ever match `role:` and
`tenant:` grants. The ceiling is applied *before* the `AccessFilter` is built:
narrowing what a key may call while leaving what it may read untouched narrows
the less important half.

**Roles are a total order.** OWNER > ADMIN > MEMBER > VIEWER, so a permission is
the least privileged role that holds it. Honest for these four, which really are
nested, and it cannot express "a billing owner who cannot read documents". When
the first non-nested role appears, `permits()` is the one function that changes.
