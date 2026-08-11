"""Adapters for the GPU model service (docs/adr/0004, docs/adr/0011).

One class, `HttpModelClient`, satisfies both `EmbeddingProvider` and `Reranker`.
They stay separate *ports* because a deployment that embeds locally and reranks
elsewhere is a real shape, but there is no reason to open two connection pools
to the same host to serve it.
"""

from __future__ import annotations

from rag.adapters.models.http import HttpModelClient

__all__ = ["HttpModelClient"]
