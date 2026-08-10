"""The ingestion worker.

A separate process from the API, not a background task inside it. Parsing is
blocking CPU work; running it in the API process would stall every concurrent
request on that worker (CLAUDE.md non-negotiable #6). Separating them also means
the two scale independently — ingestion is bursty, serving is not — and that a
document which kills a parser takes down a worker rather than the API.
"""

from __future__ import annotations

from rag.worker.runner import Worker

__all__ = ["Worker"]
