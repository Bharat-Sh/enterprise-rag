"""Application configuration.

Design notes
------------
*Nested, not flat.* Settings are grouped into sub-models (`DatabaseSettings`,
`RedisSettings`, ...) and bound from the environment via a double-underscore
delimiter: ``RAG_DATABASE__HOST``. A flat namespace is simpler on day one and
becomes a sixty-field wall by the time we have five backing services.

*Fail fast.* `get_settings()` is called during application startup. A missing or
invalid value must kill the process at boot, not surface as a 500 on the first
request that happens to need it, an hour later.

*Secrets are `SecretStr`.* They render as ``**********`` in reprs, tracebacks,
and log dumps. The first time an exception serialises a settings object you will
be glad of this.

*Frozen.* Settings are immutable after construction, so no code path can mutate
global configuration at runtime and leave the process in a state you cannot
reproduce.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Not a credential we use — a sentinel the production validator compares against
# in order to *reject* it. Bandit cannot tell the difference.
_DEV_DEFAULT_SECRET = "rag"  # noqa: S105


class Environment(StrEnum):
    """Deployment environment. Gates docs exposure, log format, and safety checks."""

    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"

    @property
    def is_production_like(self) -> bool:
        """True for environments where insecure defaults are unacceptable."""
        return self in {Environment.STAGING, Environment.PROD}


class LogFormat(StrEnum):
    """Renderer for structured log output."""

    CONSOLE = "console"
    JSON = "json"


class ServerSettings(BaseModel):
    """Uvicorn / HTTP server binding."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = False
    # Number of uvicorn workers. Keep at 1 in containers and scale replicas
    # instead — one process per container keeps signals and logging sane.
    workers: int = Field(default=1, ge=1, le=32)


class DatabaseSettings(BaseModel):
    """PostgreSQL — the system of record (docs/adr/0001)."""

    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    user: str = "rag"
    password: SecretStr = SecretStr(_DEV_DEFAULT_SECRET)
    name: str = "rag"
    pool_size: int = Field(default=10, ge=1, le=100)
    max_overflow: int = Field(default=5, ge=0, le=100)
    connect_timeout_seconds: float = Field(default=5.0, gt=0)

    @property
    def dsn(self) -> str:
        """Async SQLAlchemy DSN.

        Contains the password in plaintext by necessity — never log this value.
        Use `safe_dsn` for anything human- or machine-readable.
        """
        secret = self.password.get_secret_value()
        return f"postgresql+asyncpg://{self.user}:{secret}@{self.host}:{self.port}/{self.name}"

    @property
    def safe_dsn(self) -> str:
        """DSN with the password redacted. Safe to log."""
        return f"postgresql+asyncpg://{self.user}:***@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseModel):
    """Redis — caching, rate limiting, idempotency keys."""

    host: str = "localhost"
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0, le=15)
    password: SecretStr | None = None
    socket_timeout_seconds: float = Field(default=2.0, gt=0)
    # Redis is a *degradable* dependency: on failure we lose caching and fall
    # back to a local rate limiter rather than failing the request outright.
    required_for_readiness: bool = False

    @property
    def safe_url(self) -> str:
        """Connection URL with any password redacted. Safe to log."""
        auth = "***@" if self.password is not None else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class QdrantSettings(BaseModel):
    """Qdrant — the derived vector index (docs/adr/0001). Always rebuildable."""

    host: str = "localhost"
    port: int = Field(default=6333, ge=1, le=65535)
    grpc_port: int = Field(default=6334, ge=1, le=65535)
    prefer_grpc: bool = False
    api_key: SecretStr | None = None
    collection: str = "rag_chunks"
    timeout_seconds: float = Field(default=10.0, gt=0)

    #: Run the client's embedded local mode instead of connecting to a server,
    #: storing at this path (or in memory when `":memory:"`).
    #:
    #: This exists because the primary development machine has no Docker
    #: (CLAUDE.md), and it is honest about what it buys: filter *correctness* is
    #: identical, so the isolation tests mean exactly what they mean against a
    #: server. What it does not do is payload indexes — the client warns
    #: "Payload indexes have no effect in the local Qdrant" — so every filter is
    #: a full scan and nothing here says anything about performance. CI runs the
    #: same tests against a real service container, which is where the indexes
    #: are real.
    local_path: str | None = None

    #: Named vectors. Both are written from M5 even though only `dense` is
    #: queried until M6, because BGE-M3 produces both in one forward pass and a
    #: collection's vector configuration is fixed at creation. Adding `sparse`
    #: later would mean a new collection and re-embedding the entire corpus, to
    #: store something we already had and threw away.
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"

    #: Must match the embedding model's width. Verified against `/v1/info` at
    #: startup rather than trusted: a collection created at the wrong size
    #: rejects every insert, and the error surfaces on the first ingestion
    #: rather than at boot.
    vector_size: int = Field(default=1024, ge=1)

    #: Points per upsert request. Bounds both the HTTP payload and the memory a
    #: worker holds: 1024 floats per vector means a batch of 128 is roughly a
    #: megabyte of dense data before sparse, JSON and overhead.
    upsert_batch_size: int = Field(default=128, ge=1, le=1024)

    @property
    def url(self) -> str:
        """Base HTTP URL. Contains no credentials."""
        return f"http://{self.host}:{self.port}"

    @property
    def uses_local_mode(self) -> bool:
        return self.local_path is not None


class SigningAlgorithm(StrEnum):
    """JWS algorithm used to sign access tokens.

    Defined here rather than in `rag.domain` because configuration must resolve
    it at boot, and `rag.core` may not import our other packages.
    """

    #: Ed25519. No parameters to misconfigure, deterministic signatures, 64-byte
    #: output. The default (docs/adr/0007).
    EDDSA = "EdDSA"
    #: Kept reachable by configuration for verifiers that cannot do EdDSA.
    RS256 = "RS256"


class AuthSettings(BaseModel):
    """Token signing, password stretching, and credential lifetimes.

    The signing key is deliberately *not* defaulted. In `local` a missing key
    means "generate an ephemeral keypair at boot"; in staging or production it
    is a boot-time failure (see `Settings._reject_insecure_production_defaults`).
    Shipping a fixed development keypair in the repository was rejected: a
    committed private key is eventually trusted by something real.
    """

    #: `iss` claim. A URL by OIDC convention, and the base of the JWKS location.
    issuer: str = "http://localhost:8000"
    #: `aud` claim. Verification requires an exact match — PyJWT silently skips
    #: audience validation unless it is passed explicitly, so this is never None.
    audience: str = "rag-api"
    algorithm: SigningAlgorithm = SigningAlgorithm.EDDSA

    #: PEM-encoded private key, or a path to one. `_pem` wins if both are set.
    private_key_pem: SecretStr | None = None
    private_key_path: str | None = None
    #: Public keys of previously used signing keys, PEM-encoded. Tokens signed
    #: with them still verify, which is what makes rotation a deploy rather than
    #: a flag day: sign with the new key, keep verifying the old one until the
    #: last token minted from it has expired, then drop it from this list.
    retired_public_keys_pem: tuple[str, ...] = ()

    #: Short, because an access token cannot be withdrawn once issued. The real
    #: revocation control is `users.tokens_valid_after`, checked on every
    #: request — this only bounds the window for a *deleted* user.
    access_token_ttl_seconds: int = Field(default=900, ge=60, le=86_400)
    refresh_token_ttl_seconds: int = Field(default=2_592_000, ge=300)
    #: Clock-skew tolerance on `exp`/`nbf`. Zero makes token validity depend on
    #: NTP being perfect across every machine that verifies.
    leeway_seconds: int = Field(default=30, ge=0, le=300)

    # Argon2id, at the OWASP-recommended minimum rather than something heftier.
    # Peak memory is `memory_cost * concurrent hashes`, and hashing runs in a
    # thread pool 40 wide: 19 MiB gives ~760 MiB worst case, 64 MiB would give
    # 2.5 GiB reachable from an unauthenticated endpoint.
    argon2_time_cost: int = Field(default=2, ge=1, le=10)
    argon2_memory_cost_kib: int = Field(default=19_456, ge=8_192)
    argon2_parallelism: int = Field(default=1, ge=1, le=16)

    #: Length, not composition rules. NIST 800-63B: complexity requirements push
    #: users towards predictable substitutions and yield less entropy, not more.
    password_min_length: int = Field(default=12, ge=8, le=256)

    #: How long a granted API key may live when the caller names no expiry.
    #: None means "until revoked".
    api_key_default_ttl_days: int | None = Field(default=None, ge=1)

    @property
    def jwks_uri(self) -> str:
        return f"{self.issuer.rstrip('/')}/.well-known/jwks.json"


class RateLimitSettings(BaseModel):
    """Token-bucket limits (docs/adr/0008).

    Two buckets with different keys, because they defend different things. The
    tenant bucket protects per-tenant capacity and cost; the login bucket
    protects credentials, and cannot be keyed on a tenant that is not yet
    verified.
    """

    enabled: bool = True

    #: Steady-state rate for an authenticated tenant.
    tenant_requests_per_minute: int = Field(default=600, ge=1)
    #: Bucket capacity. Bursting is *desirable* — a page that fires six requests
    #: on load is not abuse — so capacity exceeds one second's worth of refill.
    tenant_burst: int = Field(default=120, ge=1)

    #: Deliberately small: this is the credential-stuffing surface.
    login_attempts_per_minute: int = Field(default=10, ge=1)
    login_burst: int = Field(default=5, ge=1)


class TokenizerKind(StrEnum):
    """Which token counter sizes chunks (docs/adr/0011).

    Explicit configuration rather than "use the real one if its file happens to
    be present". A silent fallback would make chunk boundaries depend on the
    contents of a directory: the same document ingested on two machines would
    produce different chunks, with nothing logged and nothing failing.
    """

    #: Character-ratio approximation. No files, no dependencies, ~10% out.
    HEURISTIC = "heuristic"
    #: BGE-M3's real vocabulary, loaded from `tokenizer_path`. Exact, and the
    #: same tokenizer the model service uses, so chunk sizes and the model's
    #: window agree by construction. Requires the file — a missing one is a
    #: startup failure, deliberately.
    BGE_M3 = "bge-m3"


class IngestionSettings(BaseModel):
    """Upload limits, blob storage, and chunking (docs/adr/0009)."""

    #: Where the filesystem blob adapter writes. Replaced by an S3 adapter at
    #: deployment; the `BlobStore` port means no call site changes.
    blob_root: str = "./var/blobs"

    #: Enforced *during* the upload stream, not after. Checking afterwards means
    #: a 10 GB body has already been written to disk before it is rejected.
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)
    #: Read size while streaming an upload. 1 MiB keeps the syscall count low
    #: without holding a meaningful amount of the file in memory.
    upload_chunk_bytes: int = Field(default=1024 * 1024, ge=4096)

    #: Target chunk size. Chunks are the unit of retrieval, so this is the main
    #: recall/precision dial — bigger chunks bury the answer, smaller ones lose
    #: the context that makes it interpretable.
    chunk_target_tokens: int = Field(default=512, ge=32, le=8192)
    #: Overlap carried between adjacent chunks, so a passage split across a
    #: boundary is still wholly present in one of them.
    chunk_overlap_tokens: int = Field(default=64, ge=0)

    #: Which tokenizer sizes those chunks. Defaults to the estimator so that a
    #: fresh clone runs with no model download; production should set `bge-m3`.
    #: Changing this on a corpus that has already been ingested requires a
    #: re-chunk — the existing chunks were sized by the old counter.
    tokenizer: TokenizerKind = TokenizerKind.HEURISTIC
    #: Where `tokenizer.json` lives when `tokenizer` is `bge-m3`. Populated by
    #: `scripts/fetch_models.py` and baked into the container image, so the
    #: worker never reaches the network at startup to fetch a vocabulary.
    tokenizer_path: str = "./var/models/bge-m3/tokenizer.json"

    #: Wall-clock ceiling on a single parse. A malformed file that sends a
    #: parser into a loop must lose its job, not its worker.
    parse_timeout_seconds: float = Field(default=120.0, gt=0)

    #: Chunks embedded and indexed per round trip. Bounds worker memory: a
    #: 2000-page PDF is on the order of 10k chunks, and holding every dense
    #: vector at once would be ~80 MB before sparse weights. Kept at or below
    #: `ModelServiceSettings.max_texts_per_request`, which splits anything
    #: larger anyway — this is the knob that stops it having to.
    embed_batch_size: int = Field(default=32, ge=1, le=512)

    # --- limits on hostile input (docs/adr/0010) --------------------------
    #
    # Parsing attacker-supplied binary formats is the largest attack surface in
    # the system. These caps are what make "the parse timed out" rare rather
    # than the only defence — a timeout still leaves a thread burning CPU,
    # because Python cannot cancel one.

    #: Pages read from a PDF. A thousand-page scan is a legitimate document and
    #: also an excellent way to occupy a worker for an hour.
    max_pdf_pages: int = Field(default=2000, ge=1)

    #: Total bytes a container may expand to. A DOCX is a zip, and a few
    #: kilobytes of zeroes compress to gigabytes — the classic decompression
    #: bomb, which a size limit on the *upload* does nothing about.
    max_extracted_bytes: int = Field(default=100 * 1024 * 1024, ge=1024)

    #: Largest tolerated uncompressed:compressed ratio for a single entry.
    #: Ordinary office documents sit under 20:1; a bomb is thousands to one.
    max_compression_ratio: int = Field(default=200, ge=2)

    #: Entries in a container. An archive with a million tiny files exhausts
    #: time and memory without ever tripping a size limit.
    max_archive_entries: int = Field(default=2000, ge=1)

    @model_validator(mode="after")
    def _overlap_must_be_smaller_than_the_chunk(self) -> IngestionSettings:
        # Overlap >= target does not shrink the remaining text, so the splitter
        # would never advance. Caught here rather than as a hang at runtime.
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError(
                f"chunk_overlap_tokens ({self.chunk_overlap_tokens}) must be smaller than "
                f"chunk_target_tokens ({self.chunk_target_tokens}); otherwise chunking "
                f"cannot make progress."
            )
        return self


class WorkerSettings(BaseModel):
    """The ingestion worker process."""

    #: Identifies the process holding a job, for debugging a stuck queue.
    #: Defaults to the hostname at startup.
    name: str | None = None

    #: How long to wait before polling again when the queue is empty. Polling
    #: rather than LISTEN/NOTIFY: the latency is irrelevant for a pipeline whose
    #: stages take seconds, and one fewer moving part is worth more than it.
    poll_interval_seconds: float = Field(default=1.0, gt=0)

    #: Claimed per poll. One, deliberately: a worker runs a single job at a time
    #: so a document that kills the process takes one job with it, not a batch.
    batch_size: int = Field(default=1, ge=1, le=32)

    #: A job whose worker died is reclaimed after this long. Must exceed the
    #: slowest realistic job, or a healthy worker's job is stolen mid-flight and
    #: run twice.
    stalled_after_seconds: float = Field(default=900.0, gt=0)
    reap_interval_seconds: float = Field(default=60.0, gt=0)


class ModelServiceSettings(BaseModel):
    """The *client's* view of the GPU model service (docs/adr/0004, 0011).

    Reached over HTTP so the API and ingestion workers stay CPU-only and
    GPU-agnostic. The service's own configuration lives with the service, in
    `model_service.settings` — these are the knobs for talking to it.
    """

    base_url: str = "http://localhost:8001"
    timeout_seconds: float = Field(default=30.0, gt=0)
    # Model load takes 20-60s, so readiness must tolerate a cold start.
    startup_grace_seconds: float = Field(default=120.0, gt=0)

    #: Shared secret sent as a bearer token. Unset locally; when set, the
    #: service rejects requests without it. The model service is an unmetered
    #: GPU for anyone who can reach it, and "it is on an internal network" is a
    #: control owned by somebody else.
    api_key: SecretStr | None = None

    #: Retries for transient failures — connection errors, 502/503/504, 429.
    #: Never for 4xx: a rejected batch is rejected identically the second time.
    #:
    #: Deliberately small. The job queue already retries with exponential
    #: backoff at a much coarser grain, and stacking the two turns a 30-second
    #: timeout into minutes of a worker held on a service that is down.
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_backoff_seconds: float = Field(default=0.25, gt=0)

    #: Texts per HTTP request. The service batches internally against a token
    #: budget, so this is about payload size, not GPU efficiency: 1024-dim
    #: vectors as JSON run roughly 20 KB each, so an unbounded list would build
    #: a response of tens of megabytes in memory at both ends.
    max_texts_per_request: int = Field(default=32, ge=1, le=512)

    @property
    def embed_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/embed"

    @property
    def rerank_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/rerank"

    @property
    def info_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/info"


class Settings(BaseSettings):
    """Root configuration object. Construct once per process via `get_settings()`."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        # Pydantic reserves the `model_` prefix for its own attributes and warns
        # on any field that starts with it. We legitimately have a
        # `model_service` group (the ML model server), so the guard is disabled.
        protected_namespaces=(),
    )

    environment: Environment = Environment.LOCAL
    service_name: str = "rag-api"
    version: str = "0.1.0"

    log_level: str = "INFO"
    # Unset means "derive from environment" — see `effective_log_format`.
    log_format: LogFormat | None = None
    docs_enabled: bool | None = None

    server: ServerSettings = Field(default_factory=ServerSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    model_service: ModelServiceSettings = Field(default_factory=ModelServiceSettings)

    @property
    def effective_log_format(self) -> LogFormat:
        """Human-readable locally, machine-parseable everywhere else."""
        if self.log_format is not None:
            return self.log_format
        return LogFormat.CONSOLE if self.environment is Environment.LOCAL else LogFormat.JSON

    @property
    def effective_docs_enabled(self) -> bool:
        """OpenAPI docs are on by default, off in production-like environments.

        Exposing the full schema publicly hands an attacker a map of the API.
        """
        if self.docs_enabled is not None:
            return self.docs_enabled
        return not self.environment.is_production_like

    @property
    def expose_error_details(self) -> bool:
        """Whether internal error text may reach the client. Never in production."""
        return not self.environment.is_production_like

    @model_validator(mode="after")
    def _reject_insecure_production_defaults(self) -> Settings:
        """Refuse to boot production-like environments carrying dev defaults.

        This is the fail-fast principle with teeth: shipping to staging with the
        development database password is a configuration bug that should be
        impossible to deploy, not something you discover in an audit.
        """
        if not self.environment.is_production_like:
            return self

        problems: list[str] = []
        if self.database.password.get_secret_value() == _DEV_DEFAULT_SECRET:
            problems.append("database.password is still the development default")
        if self.server.reload:
            problems.append("server.reload must be disabled")
        if self.docs_enabled is True:
            problems.append("docs_enabled was explicitly turned on")
        if self.auth.private_key_pem is None and self.auth.private_key_path is None:
            # Locally a missing key means "generate an ephemeral one"; every
            # restart then invalidates outstanding tokens, which is fine. Doing
            # that in production would sign tokens with a key no other replica
            # holds and log everyone out on every deploy.
            problems.append(
                "auth.private_key_pem or auth.private_key_path must be set "
                "(ephemeral signing keys are a local-development convenience only)"
            )
        if self.auth.issuer.startswith("http://"):
            problems.append("auth.issuer must be https outside local development")

        if problems:
            joined = "; ".join(problems)
            raise ValueError(f"Refusing to start in {self.environment}: {joined}")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached because parsing the environment on every access is wasteful and
    because configuration must not change mid-process. Tests override this via
    FastAPI's `dependency_overrides`, or call `get_settings.cache_clear()`.
    """
    return Settings()
