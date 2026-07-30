"""Adapters: concrete implementations of the ports declared in `rag.domain`.

This is the only package permitted to import third-party clients — qdrant_client,
anthropic, redis, the model-service HTTP client. Everything above depends on the
Protocol, so replacing an implementation is a new file here plus one line of
wiring in the composition root.

Populated from M4 onward: embeddings/, vectorstore/, rerank/, llm/, cache/,
parsers/.
"""
