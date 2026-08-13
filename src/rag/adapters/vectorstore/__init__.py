"""The derived vector index (docs/adr/0001).

Qdrant today. The `VectorStore` port exists partly so that `pgvector` — which
collapses two services into one while keeping every property ADR-0001 argues
for — remains a realistic option rather than a rewrite.
"""

from __future__ import annotations

from rag.adapters.vectorstore.qdrant import QdrantVectorStore

__all__ = ["QdrantVectorStore"]
