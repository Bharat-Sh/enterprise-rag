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

from model_service.app import create_app as create_model_service
from model_service.settings import Backend
from model_service.settings import Settings as ModelServiceSettings
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


# ---------------------------------------------------------------------------
# The model service (M4).
#
# Served with `Backend.STUB`, which satisfies the same `InferenceBackend`
# protocol with a deterministic fake and no torch. That is not a mock of the
# HTTP layer — it is the real application, with real routing, real validation,
# real batching and real error mapping, which is exactly the part CI needs to
# cover and the part a GPU would add nothing to.
# ---------------------------------------------------------------------------


@pytest.fixture
def model_service_settings() -> ModelServiceSettings:
    """Hermetic settings for the model service.

    Limits are deliberately tiny — 64-token sequences, 96-token batches — so a
    test can exceed them with a short string instead of generating a thousand
    words to prove a limit works.
    """
    return ModelServiceSettings(
        _env_file=None,
        backend=Backend.STUB,
        log_level="WARNING",
        log_format="json",
        max_sequence_tokens=64,
        max_batch_tokens=96,
        max_batch_items=4,
        max_texts_per_request=8,
        max_passages_per_rerank=8,
    )


@pytest.fixture
async def model_service_client(
    model_service_settings: ModelServiceSettings,
) -> AsyncIterator[AsyncClient]:
    """An HTTP client wired to the model service in-process, lifespan running."""
    app = create_model_service(model_service_settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://model-service") as http_client:
            yield http_client
