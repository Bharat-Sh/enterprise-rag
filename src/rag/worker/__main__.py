"""`rag-worker` — the ingestion worker entry point.

Owns the event loop, the database engine, and signal handling; the loop itself
lives in `rag.worker.runner`.

Shutdown is graceful. On SIGTERM — which is what a container runtime sends
first, and what an orchestrator follows with SIGKILL after its grace period —
the worker finishes the job in flight and then exits. Dying mid-job is safe
(the reaper requeues it) but wastes the whole visibility timeout on every
deploy, during which that document sits untouched.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
from typing import TYPE_CHECKING

from rag.core.config import get_settings
from rag.core.logging import configure_logging, get_logger
from rag.db.session import create_engine, create_session_factory
from rag.worker.runner import Worker

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["main"]

_log = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-worker", description="Run the ingestion worker.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one batch and exit. For tests and manual draining.",
    )
    parser.add_argument("--name", default=None, help="Worker identity recorded on claimed jobs.")
    return parser


async def _run(once: bool, name: str | None) -> int:
    settings = get_settings()
    configure_logging(
        level=settings.log_level,
        log_format=settings.effective_log_format,
        service_name="rag-worker",
        version=settings.version,
        environment=str(settings.environment),
    )

    if name is not None:
        settings = settings.model_copy(
            update={"worker": settings.worker.model_copy(update={"name": name})}
        )

    engine = create_engine(settings)
    worker = Worker(create_session_factory(engine), settings)

    try:
        if once:
            await worker.run_once()
            return 0

        _install_signal_handlers(worker)
        await worker.run_forever()
        return 0
    finally:
        # The worker owns an HTTP connection pool and, in local mode, a file
        # lock on the vector store's directory. Both are released here so a
        # restart does not fail on a lock the previous process never dropped.
        await worker.aclose()
        await engine.dispose()


def _install_signal_handlers(worker: Worker) -> None:
    """Ask the worker to stop on SIGTERM/SIGINT.

    `add_signal_handler` is POSIX-only; on Windows it raises, and the fallback
    is the default behaviour of interrupting the loop. Local development on
    Windows is a supported path here (see CLAUDE.md), so this must not be fatal.
    """
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError, AttributeError):
            loop.add_signal_handler(signal_number, worker.stop)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(once=args.once, name=args.name))
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        _log.info("worker.interrupted")
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
