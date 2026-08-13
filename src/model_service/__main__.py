"""Entry point: `uv run python -m model_service`.

**Exactly one worker, always.** Model weights are per-process, so N workers
means N copies in VRAM — on a 6 GB card the second one is an out-of-memory error
at startup. Concurrency here comes from batching inside the single process, and
scale comes from more GPUs, not more processes on one. This is not configurable
for the same reason `rag.core.config.ServerSettings.workers` defaults to 1: an
option that is wrong in every setting is a trap, not a feature.
"""

from __future__ import annotations

import uvicorn

from model_service.app import create_app
from model_service.settings import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        # Uvicorn's own access log would record request lines only, never
        # bodies, so it is safe — but it is also noise next to the structured
        # events this service already emits per request.
        access_log=False,
    )


if __name__ == "__main__":
    main()
