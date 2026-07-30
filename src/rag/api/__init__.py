"""HTTP delivery layer.

Routers stay thin: parse, authorise, call a service, serialise. Business logic
belongs in `rag.services`; anything else makes the logic untestable without an
HTTP client and unreachable from the ingestion workers.
"""
