"""Configuration for the model service process.

Its own `BaseSettings` with its own `MODEL_SERVICE_` prefix, not a section of
`rag.core.config.Settings`. The two processes are deployed separately and share
no environment: the API needs a database DSN and no GPU, this needs a device and
no database. Folding them together would mean every model-service container
carrying — and being validated against — configuration for a Postgres it will
never open.

The defaults here are sized for the development GPU (see docs/adr/0011), which
has 6 GB of VRAM. They are conservative on purpose; a service that OOMs mid-batch
fails a job that was already halfway through.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Backend", "Settings", "get_settings"]


class Backend(StrEnum):
    """Which inference implementation is loaded."""

    #: FlagEmbedding on the configured device. The real one.
    FLAG = "flag"
    #: A deterministic fake. No torch, no weights, no GPU — it serves the exact
    #: same HTTP contract, which is what lets the routes, batching, limits and
    #: error paths be tested on a CI runner with no GPU.
    #:
    #: It is safe to leave reachable in production because it is *self-
    #: identifying*: `/v1/info` reports the model as `stub`, and that string is
    #: stamped into `chunks.embedding_model` on every row it produces. A corpus
    #: accidentally embedded by it says so in the database rather than looking
    #: like real vectors that happen to retrieve badly.
    STUB = "stub"


class Settings(BaseSettings):
    """Root configuration for this process."""

    model_config = SettingsConfigDict(
        env_prefix="MODEL_SERVICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        # We legitimately have `model_*` fields — this service is *about*
        # models — and pydantic reserves that prefix for itself by default.
        protected_namespaces=(),
    )

    service_name: str = "rag-model-service"
    log_level: str = "INFO"
    #: `console` or `json`; unset means JSON, since this normally runs headless.
    log_format: str | None = None

    host: str = "0.0.0.0"
    port: int = Field(default=8001, ge=1, le=65535)

    backend: Backend = Backend.FLAG

    #: Bearer token callers must present. Unset disables the check, which is
    #: right for a laptop and wrong for anything shared: an open embedding
    #: endpoint is free GPU time for whoever finds it.
    api_key: SecretStr | None = None

    # --- models -----------------------------------------------------------

    #: Local directories, not HuggingFace repo ids. A service that downloads
    #: weights at startup has a network dependency in its critical path and a
    #: supply-chain question every time it restarts. `scripts/fetch_models.py`
    #: populates these, and the container image bakes them in.
    embedding_model_path: str = "./var/models/bge-m3"
    reranker_model_path: str = "./var/models/bge-reranker-v2-m3"

    #: Reported in `/v1/info` and stamped onto every chunk embedded here.
    #: Set to the snapshot revision by `scripts/fetch_models.py`; without it, a
    #: weights change is undetectable and the only safe migration is to re-embed
    #: the entire corpus.
    embedding_model_version: str = "unknown"

    #: Reranking costs ~1.2 GB of VRAM. On a 6 GB card that is the difference
    #: between fitting and not, so it can be dropped for embedding-only work.
    reranker_enabled: bool = True

    device: str = "cuda"
    #: Halves weight memory and roughly doubles throughput on any card since
    #: Turing. Rejected `bfloat16`: no accuracy benefit at inference and it is
    #: slower on consumer Ada silicon.
    use_fp16: bool = True

    # --- limits -----------------------------------------------------------

    #: Longest input accepted, in tokens. **Not** BGE-M3's 8192 window.
    #: Attention memory is quadratic in sequence length, and the development GPU
    #: has ~2.5 GB free for activations once both models are resident — a batch
    #: at 8192 does not come close to fitting. Chunks target 512 tokens, so
    #: nothing legitimate is being refused.
    #:
    #: Over-length input is **rejected**, never truncated. A truncated chunk
    #: yields a vector that looks entirely valid while missing the text's tail,
    #: which is undetectable from the outside and permanent once indexed.
    max_sequence_tokens: int = Field(default=1024, ge=64, le=8192)

    #: Token budget for one forward pass. Batches are formed against this rather
    #: than a fixed item count: eight 32-token chunks and eight 1024-token
    #: chunks differ by more than an order of magnitude in activation memory,
    #: and only one of those fits.
    #:
    #: The floor here matches `max_sequence_tokens`' floor rather than being a
    #: second, larger, arbitrary number. The constraint that actually matters is
    #: relational — a batch must be able to hold one maximum-length input — and
    #: it is enforced by the validator below. Two overlapping absolute limits
    #: would only mean one of them is redundant and neither says why.
    max_batch_tokens: int = Field(default=8192, ge=64)
    #: Belt-and-braces cap on items per forward pass, for the degenerate case of
    #: thousands of near-empty strings that never reach the token budget.
    max_batch_items: int = Field(default=64, ge=1, le=512)

    #: Request-level caps. Beyond these the answer is 413, not an OOM.
    max_texts_per_request: int = Field(default=256, ge=1)
    max_passages_per_rerank: int = Field(default=256, ge=1)

    @model_validator(mode="after")
    def _batch_budget_must_fit_one_max_length_input(self) -> Settings:
        # Otherwise a single input at the accepted maximum exceeds the budget
        # for an entire batch, and `plan_batches` would have to either exceed
        # the budget or emit an empty batch and stall. Caught at boot rather
        # than as a puzzling runtime failure on one unusually long chunk.
        if self.max_batch_tokens < self.max_sequence_tokens:
            raise ValueError(
                f"max_batch_tokens ({self.max_batch_tokens}) must be at least "
                f"max_sequence_tokens ({self.max_sequence_tokens}); otherwise a single "
                f"maximum-length input cannot be batched at all."
            )
        return self


def get_settings() -> Settings:
    """Read configuration from the environment.

    Not cached, unlike `rag.core.config.get_settings`: this is called once, in
    `__main__`, and a cache here would only make tests fight over a singleton.
    """
    return Settings()
