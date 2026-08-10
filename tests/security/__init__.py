"""Cross-tenant leakage, credential forgery, and authorization bypass.

Per CLAUDE.md these must never be marked `xfail` and must never be silently
weakened. Where a test needs a real database it is marked `integration` and
skips when none is reachable, exactly like the rest of the suite — but the
*mechanisms* those tests exercise are additionally pinned by tests here and in
`tests/unit` that always run, so a machine with no Postgres still fails loudly
if the invariant is broken.
"""

from __future__ import annotations
