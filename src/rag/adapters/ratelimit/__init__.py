"""Rate-limiter adapters (docs/adr/0008).

In-process today; a Redis implementation of the same `RateLimiter` port lands in
M9 and changes no call site.
"""

from __future__ import annotations
