# ADR-0004 — Local BGE-M3 behind a separate GPU model service

- **Status:** Accepted
- **Date:** 2026-07-30
- **Milestone:** M0 (design), M4 (implementation)

## Context

Retrieval needs embeddings, and reranking needs a cross-encoder. Both can be
hosted APIs (Voyage, Cohere) or run locally. We have a GPU available and a
strong preference for avoiding per-call costs, so both models run locally:

- **BGE-M3** for embeddings
- **bge-reranker-v2-m3** for cross-encoder reranking

The remaining question is *where the model process lives*.

## Decision

Two decisions.

**1. BGE-M3 for embeddings.** One forward pass produces both a dense vector
(1024-d) and sparse lexical weights. Those map directly onto Qdrant named
vectors, so hybrid retrieval needs no second retrieval system, no separate BM25
statistics, and — critically — **one access-control filter rather than two**.
Two independent retrieval paths means two independent implementations of "which
chunks may this user see", which is two chances to leak data.

BGE-M3 also emits ColBERT multi-vectors. Deferred: storing 1024 floats per token
is roughly 100× the dense footprint, so it belongs in M10 as an eval-driven
experiment, not as day-one architecture.

**2. A separate HTTP model service**, not in-process inference. One process, one
GPU, three endpoints:

```
POST /v1/embed   {texts[], mode: "query"|"passage"} -> {dense[], sparse[]}
POST /v1/rerank  {query, passages[], top_k}         -> [{index, score}]
GET  /v1/info                                        -> model names + versions
```

## Why not load the model in the API process

1. **It blocks the event loop.** Model inference is a blocking CPU→GPU→CPU round
   trip. Called inside an `async def` handler it stalls every concurrent request
   on that worker — including open SSE streams. This is the most common way
   async Python services fail under load.
2. **VRAM multiplies with workers.** Four uvicorn workers load four copies. On a
   laptop GPU that is an immediate out-of-memory error. Model weights want
   exactly one process; HTTP handling wants many.
3. **It couples deployment.** The API image would carry CUDA and PyTorch (5–7 GB)
   and require a GPU node forever. The API does not need a GPU; it needs access
   to something that has one.
4. **It prevents substitution.** With an HTTP boundary, moving to Voyage or
   Cohere is a base-URL change.

`/v1/info` is not decoration: it supplies the `embedding_model` and
`embedding_version` stamped onto every chunk row, which is what makes model
migration survivable (re-embed into a new named vector, backfill, cut over).

## Consequences

**We get:** a GPU-agnostic API and worker tier, one place to batch, and free
substitutability. Concurrency is handled by dynamic micro-batching inside the
service — requests queue, a single consumer batches them into one forward pass.
The service therefore runs with exactly one uvicorn worker.

**We pay:** one more service; a network hop (~2–5 ms, negligible against ~60 ms
of GPU rerank); and a cold start of 20–60 s while both models load, which is why
readiness must tolerate it and why `/health` and `/ready` are separate endpoints.

**Development note.** A CUDA-enabled Python image is 5–7 GB and slow to rebuild.
For day-to-day work on Windows, the model service runs natively on the host and
containers reach it at `host.docker.internal:8001`; the containerised form lives
behind a compose `gpu` profile. Same code, same contract, two run modes — which
is itself the payoff for putting an HTTP boundary here.

## Alternatives considered

| Option | Assessment |
|---|---|
| **HuggingFace TEI** | Production-grade Rust server. Poor support for BGE-M3's *sparse* output, which forfeits the free hybrid retrieval that motivated the model choice. |
| **Infinity** | Serves BGE-M3 including sparse. A genuinely reasonable choice and the designated fallback if writing our own proves troublesome. Rejected for now in favour of full control over the multi-output shape and batching policy. |
| **FastEmbed / ONNX on CPU** | Wastes the available GPU; reranking latency would be 5–10× worse. |
| **Hosted embeddings (Voyage)** | Better quality-per-effort and no ops burden, at per-call cost and with a second vendor. Remains available behind the same port. |
