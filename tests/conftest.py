"""Shared pytest fixtures.

Note `_env_file=None` on the test settings: without it, pydantic-settings would
read the developer's local `.env`, and the suite would pass or fail depending on
whose machine it ran on. Tests must be hermetic.

The client fixture drives the application's lifespan explicitly via
`app.router.lifespan_context`. `httpx.ASGITransport` does *not* run lifespan on
its own, so without this `app.state.health` would never be created and every
`/ready` test would fail for the wrong reason.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from rag.api.main import create_app
from rag.core.config import Environment, LogFormat, Settings


@pytest.fixture
def settings() -> Settings:
    """Hermetic settings: no .env, quiet logs, local environment."""
    return Settings(
        _env_file=None,
        environment=Environment.LOCAL,
        service_name="rag-api-test",
        version="0.0.0-test",
        log_level="WARNING",
        log_format=LogFormat.JSON,
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    """A fresh application per test, built from the hermetic settings."""
    return create_app(settings)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """An HTTP client wired to the app in-process, with lifespan running."""
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http_client:
            yield http_client
