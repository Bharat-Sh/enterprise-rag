"""Structured logging.

Three things this buys us, in order of value:

1. **Correlation.** A processor injects the current request/trace/tenant ids
   from `rag.core.context` into *every* event, including logs emitted deep
   inside library code. In M11 the same ids feed OpenTelemetry spans and
   Langfuse traces, so one identifier answers "what happened to this request?"
   across the whole stack.

2. **One output format.** `uvicorn`, `sqlalchemy`, and `httpx` all log through
   stdlib `logging`. We intercept those records and push them through the same
   structlog pipeline, so we get one format rather than three interleaved ones.

3. **Redaction.** A processor scrubs values under sensitive keys before they
   reach any renderer. Combined with `SecretStr` in the config layer, that is
   two independent defences against leaking a credential into a log aggregator
   you cannot purge.

Rendering is human-friendly locally and JSON everywhere else.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

import structlog

from rag.core.config import LogFormat
from rag.core.context import current_context

if TYPE_CHECKING:
    from structlog.typing import EventDict, Processor, WrappedLogger

__all__ = ["configure_logging", "get_logger"]

REDACTED = "***redacted***"

# Matched case-insensitively against event keys.
_SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credentials",
        "dsn",
        "password",
        "refresh_token",
        "secret",
        "set-cookie",
        "token",
    }
)

# Libraries that are useful at WARNING and deafening at INFO.
_LIBRARY_LEVELS: dict[str, int] = {
    "uvicorn.access": logging.WARNING,  # we emit our own access log in TimingMiddleware
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "asyncio": logging.WARNING,
    "multipart": logging.WARNING,
}


def _add_request_context(
    _logger: WrappedLogger, _method_name: str, event_dict: EventDict
) -> EventDict:
    """Inject request/trace/tenant ids from the ambient context, if bound."""
    event_dict.update(current_context())
    return event_dict


def _redact_sensitive(
    _logger: WrappedLogger, _method_name: str, event_dict: EventDict
) -> EventDict:
    """Replace values stored under sensitive keys.

    Deliberately shallow: it does not walk nested structures. Deep traversal on
    every log line is a real cost on a hot path, and the discipline we actually
    rely on is not putting secrets in log events in the first place. This is a
    safety net, not a licence.
    """
    for key in list(event_dict):
        if key.lower() in _SENSITIVE_KEYS:
            event_dict[key] = REDACTED
    return event_dict


def _make_service_metadata(
    service_name: str, version: str, environment: str
) -> Processor:
    """Build a processor stamping static service identity onto every event."""

    def processor(
        _logger: WrappedLogger, _method_name: str, event_dict: EventDict
    ) -> EventDict:
        event_dict.setdefault("service", service_name)
        event_dict.setdefault("version", version)
        event_dict.setdefault("env", environment)
        return event_dict

    return processor


def configure_logging(
    *,
    level: str,
    log_format: LogFormat,
    service_name: str,
    version: str,
    environment: str,
) -> None:
    """Configure structlog and route stdlib logging through it.

    Idempotent: safe to call more than once (tests do), because it clears the
    root handler list before installing ours.
    """
    # Applied to events from structlog loggers AND from stdlib loggers, so a
    # SQLAlchemy warning carries the same trace id as our own events.
    shared_processors: list[Processor] = [
        _make_service_metadata(service_name, version, environment),
        _add_request_context,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _redact_sensitive,
    ]

    renderer: Processor
    final_processors: list[Processor]
    if log_format is LogFormat.JSON:
        renderer = structlog.processors.JSONRenderer()
        final_processors = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ]
    else:
        # ConsoleRenderer formats exc_info itself, so no format_exc_info here.
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
        final_processors = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=final_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Uvicorn installs its own handlers; strip them so records propagate to us.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    for name, library_level in _LIBRARY_LEVELS.items():
        logging.getLogger(name).setLevel(library_level)


def get_logger(name: str | None = None, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound logger. Module-level usage: `log = get_logger(__name__)`."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    if initial_values:
        return logger.bind(**initial_values)
    return logger
