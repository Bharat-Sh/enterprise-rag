# ADR-0011 — Implementing the model service: batching, tokenizer, and the wire

- **Status:** Accepted
- **Date:** 2026-08-11
- **Milestone:** M4
- **Extends:** ADR-0004 (which chose BGE-M3 and the HTTP boundary), and
  **amends** its batching decision — see "Batching" below.

## Context

ADR-0004 settled *what* runs on the GPU and *that* it sits behind HTTP. It left
open everything about actually building it, and one hard fact turned up when we
did: the development GPU is an **RTX 4050 Laptop with 6141 MiB of VRAM**.
CLAUDE.md had flagged that number as unknown and needed to size M4's defaults.
It rules out several things ADR-0004 assumed were free.

M4 delivers the service, both ports, the HTTP client, and the real tokenizer.
It deliberately does **not** wire embedding into the ingestion pipeline.

---

## 1. M4 stops at the contract; M5 wires the pipeline

`IngestionPipeline` is untouched. `CHUNKING → EMBEDDING → INDEXING → READY`
becomes real in M5, alongside the vector store — which is also where the
temporary `CHUNKING → READY` edge and `TestTemporaryEdgeForM3` are deleted, as
already scheduled.

**Rejected: add an `EMBEDDING` stage now.** It would compute vectors and discard
them — real GPU cost and real latency on every ingest, for nothing, through a
stage that has never been exercised against a consumer. Worse, it splits the
removal of `CHUNKING → READY` across two milestones, producing exactly the
half-migrated state machine that the existing failing test was written to
prevent.

The consequence is that M4 has no end-to-end demo. It is judged on its contract
and its tests instead, which is the honest trade.

## 2. Batching: one lock now, the micro-batcher deferred

**This amends ADR-0004**, which specified dynamic cross-request micro-batching —
a queue, a timer, and a single consumer coalescing separate requests into one
forward pass.

What M4 ships instead: a single `asyncio` lock around the GPU, plus sub-batching
*within* a request against a **token budget**. The lock is released between
batches, so a caller sending 256 texts does not lock everyone else out for the
duration.

The lock is not a performance choice, it is a correctness one. Two concurrent
forward passes on a 6 GB card is how you get an out-of-memory error, and a CUDA
OOM can leave the context unusable for the rest of the process's life.

**Why the micro-batcher is deferred rather than cancelled.** A caller already
sends many texts per request, which captures most of the batching win. The
remaining gain appears only under many small *concurrent* requests — that is
query-time embedding, which does not exist before M6. Building it now means
tuning a `max_wait_ms` against a load shape nobody has measured, and it is
precisely the kind of thing that looks fine until a latency percentile says
otherwise. It goes in when M10's eval harness can measure the difference.

**Why a token budget rather than a fixed batch size.** Activation memory scales
with tokens, not items, and quadratically in sequence length within an item.
A batch of 64 thirty-token chunks and a batch of 64 thousand-token chunks differ
by more than an order of magnitude in peak memory. A fixed item count has to be
tuned for the worst case, which wastes the GPU on the common one.

**Batches preserve input order.** Length bucketing would pack the GPU slightly
better and would attach every vector to the wrong chunk unless the caller
remembers to unsort — after which retrieval still *works*, it just returns
unrelated text. Not a trade worth a few percent of throughput.

## 3. Sizing for 6 GB

| | |
|---|---|
| BGE-M3, fp16 | ~1.2 GB |
| bge-reranker-v2-m3, fp16 | ~1.2 GB |
| Desktop / driver overhead | ~0.5–1 GB |
| **Remaining for activations** | **~2.5 GB** |

So `max_sequence_tokens` defaults to **1024, not BGE-M3's 8192**. A batch at
8192 does not fit in 2.5 GB and is not close. Chunks target 512 tokens, so
nothing legitimate is being refused — the 8192 window is capacity we cannot
afford and have no use for.

`reranker_enabled` exists so a smaller card can drop 1.2 GB.

**Over-length input is rejected with a 422, never truncated.** This is the most
important behaviour in M4. Truncation produces a vector that is structurally
perfect and semantically missing the end of the text: no error, no warning,
undetectable from the outside, and permanent once indexed. The whole point of
having a limit is to make that impossible, so the limit must be an error and not
a silent repair.

## 4. The tokenizer runs in the worker, not over HTTP

`rag.domain.chunking` calls `count_tokens` once per candidate span while it
recursively splits — hundreds of calls per document. Over HTTP that is hundreds
of round trips per document, and it would force `TokenCounter` to become async,
dragging the thread hop into the domain layer.

So `BgeTokenCounter` loads BGE-M3's `tokenizer.json` **in process**, using
`tokenizers` (the Rust library, a few megabytes, no torch).

- **Rejected: `transformers.AutoTokenizer`** — a large dependency that drags
  torch behind it on most install paths, into the CPU-only worker whose entire
  purpose is not to need one.
- **Rejected: a silent fallback to the estimator when the file is missing** —
  chunk boundaries would then depend on whether a file happened to be present,
  so the same document would chunk differently on two machines with nothing
  logged and nothing failing. Selection is explicit configuration
  (`RAG_INGESTION__TOKENIZER`), and a configured vocabulary that will not load
  kills the process at boot.
- **Rejected: fetching from HuggingFace at runtime** — a download in the
  critical path of every restart, and the bytes we tokenize with become whatever
  the hub served today rather than what the image was built with.

The service and the worker load the *same file*, so they agree by construction.
`/v1/info` publishes a `tokenizer_hash` — a truncated SHA-256 of the file's
bytes — so a disagreement is an observable fact rather than an assumption. That
matters because chunks sized by the wrong vocabulary silently exceed the window
and lose their tails.

The code is duplicated across `model_service.tokenizer` and
`rag.adapters.tokenize.bge` rather than shared, because `model_service` may not
import `rag.adapters` — see §7. Twenty duplicated lines is cheaper than the
boundary it would breach, and a unit test asserts the two produce identical
counts.

## 5. The wire

- **Dense vectors as JSON float arrays.** *Rejected: base64 float32* — ~2.7×
  smaller and faster to decode, but a few milliseconds of parsing against ~60 ms
  of GPU is not the bottleneck, and a payload you can read in a terminal is
  worth more at this stage. This is the first optimisation to reach for if
  payload size ever shows up in a profile.
- **Sparse as parallel `indices[]`/`values[]`.** *Rejected: a
  `{token_id: weight}` object* — JSON keys are strings, roughly doubling the
  payload and forcing an int cast per element. Parallel arrays are also exactly
  Qdrant's sparse-vector shape, so M5 needs no conversion.
- **Vectors are L2-normalised by the service.** Decided once, server-side, so
  cosine similarity is a dot product and no caller can forget. A caller that
  forgot would not get an error; it would get scores that are wrong and
  plausible.
- **`mode: query|passage` is carried even though BGE-M3 ignores it.** E5, Voyage
  and Cohere need it. Retrofitting a required parameter later means touching
  every call site *and* re-embedding everything indexed before the fix, because
  the vectors would no longer match.
- **Model identity rides on every embed response**, not only on `/v1/info`, so
  the values stamped into `chunks.embedding_model` describe the pass that
  actually produced those vectors rather than whatever was loaded at startup.

**Two copies of the schema, one contract test.** The client defines its own
response models. That duplication is deliberate — the processes deploy
separately, so the client must tolerate a server that has grown a field — but
duplication without a check is drift with extra steps.
`tests/unit/test_model_contract.py` feeds every server response model through
its client counterpart and asserts every field the client *requires* is one the
server *sends*. **This test is the reason both ends live in one repository:**
split across two repos, a renamed field is discovered by a 500 in production.
A further test asserts every `*Response` in the server's `__all__` is paired, so
adding an endpoint in M6 and forgetting to pair it fails rather than passing by
omission.

## 6. Client behaviour: fail closed, retry shallowly

**Fail closed.** `RateLimiter` fails open — losing rate limiting costs fairness,
failing closed costs the whole API. This is the opposite case and the opposite
choice: there is no degraded embedding, and a document indexed with placeholder
vectors is unfindable while claiming to be searchable, with nothing raised
anywhere. An unreachable model service fails the job.

**Retries are shallow (2).** The job queue already retries with exponential
backoff at a far coarser grain, so deep retries multiply: three requests at a
30-second timeout, times five queue attempts, is a worker held for seven minutes
on a service that is down. The client's retries exist to ride out a dropped
connection or a restart; outages are the queue's problem.

**Status maps to who has to change something.**

| Status | Raised as | Because |
|---|---|---|
| 400 / 413 / 422 | `InvalidInputError` (a `DomainError`) | The request is wrong and will be identically wrong next time. The worker dead-letters it now instead of burning five attempts. |
| 401 / 403 / 501 | `ConfigurationError` | The deployment is misconfigured. Retrying does not fix it; a redeploy does. |
| 429 / 5xx, transport errors | `DependencyUnavailableError` (503) | Transient. Retry, then let the queue back off. |

The client also **verifies the response length matches the request**. A provider
returning fewer embeddings than texts would attach every vector after the gap to
the wrong chunk, and retrieval would keep working while returning unrelated
text — close to undiagnosable from the outside.

## 7. Packaging and the dependency boundary

`model_service` is a sibling package to `rag` in the same repository, installed
via a `gpu` extra. Two import-linter contracts hold the arrangement up:

- **`rag` may never import `model_service`, `torch`, or `FlagEmbedding`.** One
  convenience import — a shared constant, a reused dataclass — would put a
  5–7 GB CUDA stack back into the API's and the worker's dependency closure and
  undo the entire reason for ADR-0004's HTTP boundary. It would happen by
  accident.
- **`model_service` may import `rag.core` and nothing else of ours**, so all
  three processes emit the same structured log shape. Anything deeper would make
  the GPU image depend on the database stack it never opens.

Both were verified to actually fail by temporarily adding the imports, including
the transitive `model_service.backend → model_service.flag → FlagEmbedding` path.

**Rejected: a separate repository.** The wire contract has two ends; split
across two CIs, nothing fails when they drift.
**Rejected: putting it under `src/rag/`.** Then `rag` cannot be installed
without CUDA.

## 8. Inference library: FlagEmbedding

BGE-M3's sparse output is not a pooling choice — it is a separate
`sparse_linear` head with its own weights, plus token-weight max-pooling and
special-token exclusion. Reimplementing that on raw `transformers` is about
forty lines that fail *silently* when wrong: not a crash, just quietly worse
lexical retrieval that nothing detects until M10's golden set says relevance
dropped and nobody knows when it happened.

**Rejected: `sentence-transformers`** — no sparse output at all, which forfeits
the hybrid retrieval that motivated choosing BGE-M3 and leaves us maintaining
two retrieval paths, and therefore two implementations of "which chunks may this
caller see".

FlagEmbedding is MIT and the weights are MIT / Apache-2.0, which matters for the
same reason PyMuPDF was rejected in ADR-0010: this repository is MIT and
intended to be published.

## 9. Auth on the internal hop

An optional shared bearer token, compared with `secrets.compare_digest`. Off by
default, which is right for a laptop.

**Rejected: nothing at all.** A service that will embed any text for anyone who
can reach it is an unmetered GPU, and "it is on an internal network" is a control
owned by somebody else.
**Rejected: mTLS.** Certificate lifecycle for one internal hop.

The service is **tenant-blind by design** — it sees text, never a tenant id.
That has a consequence: it cannot make an access decision about a log line, so
it must never log request or response text at all. Counts, token totals and
durations only, and error bodies carry positions rather than content.
`tests/security/test_model_service_privacy.py` asserts both, and was verified to
fail when a single `debug_texts=payload.texts` was added to one log call.

## 10. Testing without a GPU

| Layer | How | In CI |
|---|---|---|
| Batch planning | pure function, exhaustive | yes |
| Service routes, limits, ordering, auth | `StubBackend` — a real implementation of `InferenceBackend`, no torch | yes |
| HTTP client: parsing, retries, status mapping | `httpx.MockTransport` | yes |
| Cross-process contract | a stub-backed service started by the workflow, hit over a real socket | yes |
| Tokenizer | the real 17 MB vocabulary | yes, once fetched |
| No text in logs; auth enforced | `tests/security` | yes |
| Real forward pass, and the published fingerprint matching the worker's | needs a GPU backend | **no** — the one entry in `ALLOWED_SKIPS` |

The plan in CLAUDE.md had been to skip the model-service integration module in
CI. That turned out to be far too pessimistic: almost nothing it covers needs a
GPU, so CI starts a stub-backed service and the module runs for real.
`assert_suites_ran.py` then *enforces* that it ran, with a single named
exception — see §10's last row.

`StubBackend` is not a mock of the HTTP layer — it is a real implementation of
the contract with an uninteresting model behind it, so these tests exercise
actual routing, validation, batching and error mapping. It is deterministic
(vectors derived from a hash) and its reranker scores by token overlap, so
ordering assertions pass for the right reason rather than by luck.

It is safe to leave reachable in production because it is **self-identifying**:
`/v1/info` reports `stub`, and that string is stamped into
`chunks.embedding_model` on every row it produces. A corpus embedded by accident
says so in the database rather than looking like real vectors that merely
retrieve badly.

Two things CI genuinely does not cover, stated rather than implied:

1. **`model_service/flag.py` never executes.** No GPU runner. This is why that
   module is kept as thin as it is — load, call, convert types — with every
   decision that could live in tested code pushed up into `app.py`,
   `batching.py` and `tokenizer.py`.
2. **`docker/Dockerfile.model` is linted, not built.** A full build pulls torch
   and the bundled CUDA runtime on every push, for an image no runner can
   execute. `docker buildx build --check` catches syntax, stage references and
   bad `COPY --from` — the ways this file actually rots — in a couple of
   seconds. It does not prove the image builds.

## Consequences

**We get:** a GPU-agnostic API and worker; exact chunk sizing; a contract that
cannot silently drift; and a service whose HTTP behaviour is fully covered on a
machine with no GPU.

**We pay:** a deferred micro-batcher to revisit when M6 produces concurrent
query load; ~20 duplicated lines of tokenizer loading; one untested module; and
a `max_sequence_tokens` default that is a property of the development GPU rather
than of the model — a bigger card should raise it, and the setting exists so
that is a config change.
