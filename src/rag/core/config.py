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

_DEV_DEFAULT_SECRET = "rag"


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

    @property
    def url(self) -> str:
        """Base HTTP URL. Contains no credentials."""
        return f"http://{self.host}:{self.port}"


class ModelServiceSettings(BaseModel):
    """The GPU model service serving BGE-M3 embeddings and reranking.

    Reached over HTTP so the API and ingestion workers stay CPU-only and
    GPU-agnostic (docs/adr/0004).
    """

    base_url: str = "http://localhost:8001"
    timeout_seconds: float = Field(default=30.0, gt=0)
    # Model load takes 20-60s, so readiness must tolerate a cold start.
    startup_grace_seconds: float = Field(default=120.0, gt=0)


class Settings(BaseSettings):
    """Root configuration object. Construct once per process via `get_settings()`."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    environment: Environment = Environment.LOCAL
    service_name: str = "rag-api"
    version: str = "0.1.0"

    log_level: str = "INFO"
    # Unset means "derive from environment" — see `effective_log_format`.
    log_format: LogFormat | None = None
    docs_enabled: bool | None = None

    server: ServerSettings = Field(default_factory=ServerSettings)
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
