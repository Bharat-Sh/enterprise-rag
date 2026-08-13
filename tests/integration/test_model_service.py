"""The client against a real model service, over a real socket.

Skips when nothing is listening, following the same rule as the Postgres suite:
a machine without the dependency runs a green suite rather than a red one it
cannot fix.

**These do run in CI.** The plan had been to skip them for want of a GPU, but
almost none of what they cover needs one: CI starts a real model-service process
on the stub backend and these run against it over a real socket. Consequently
`scripts/assert_suites_ran.py` enforces that they ran — a skip here fails the
build. The single assertion that genuinely requires a GPU is named in that
script's `ALLOWED_SKIPS`, with its reason.

What this adds over the unit tests
-----------------------------------
The unit suite drives the client through `MockTransport` and the service through
an in-process ASGI transport. Both are thorough and neither crosses a socket, so
neither can catch: a real HTTP status the framework produces differently than a
handler does, connection reuse across many requests, JSON of realistic size, or
the two ends disagreeing about a field name in a way both halves' own tests
consider correct.

Running it
----------
Against the stub, on any machine — this is what proves the client and service
agree, and it needs no GPU:

    MODEL_SERVICE_BACKEND=stub uv run rag-model-service
    uv run pytest tests/integration/test_model_service.py

Against the real models, on the GPU box:

    uv sync --extra gpu && uv run python scripts/fetch_models.py
    uv run rag-model-service
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest

from rag.adapters.models import HttpModelClient
from rag.core.config import ModelServiceSettings
from rag.domain.embedding import EmbedMode

pytestmark = pytest.mark.integration

BASE_URL = os.environ.get("RAG_TEST_MODEL_SERVICE_URL", "http://127.0.0.1:8001")

#: Resolved at import rather than inside the test, so the filesystem call does
#: not happen on the event loop — the ASYNC lint rules apply to test bodies too,
#: and a blocking probe in an `async def` is the habit that eventually lands in
#: a request handler.
LOCAL_TOKENIZER = Path("./var/models/bge-m3/tokenizer.json")
LOCAL_TOKENIZER_PRESENT = LOCAL_TOKENIZER.is_file()


def _listening() -> bool:
    """Cheap TCP probe, so deciding to skip costs no HTTP round trip."""
    parsed = urlparse(BASE_URL)
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 8001), 1.0):
            return True
    except OSError:
        return False


requires_model_service = pytest.mark.skipif(
    not _listening(),
    reason=(
        f"No model service listening on {BASE_URL}; start one with "
        f"`MODEL_SERVICE_BACKEND=stub uv run rag-model-service`"
    ),
)


@pytest.fixture
async def client():
    settings = ModelServiceSettings(base_url=BASE_URL, max_texts_per_request=4)
    model_client = HttpModelClient(settings)
    try:
        yield model_client
    finally:
        await model_client.aclose()


@requires_model_service
class TestAgainstARealService:
    async def test_info_describes_the_loaded_models(self, client) -> None:
        info = await client.info()

        assert info.embedding_model
        assert info.dimensions > 0
        assert info.max_sequence_tokens > 0

    async def test_embedding_round_trips(self, client) -> None:
        info = await client.info()

        embeddings = await client.embed(["hello world"], mode=EmbedMode.PASSAGE)

        assert len(embeddings) == 1
        # The dimension the service *reports* must be the dimension it
        # *produces*. These diverging is how a vector store's collection schema
        # ends up rejecting every insert after a model change.
        assert embeddings[0].dimensions == info.dimensions

    async def test_dense_vectors_arrive_normalised(self, client) -> None:
        [embedding] = await client.embed(["retrieval augmented generation"])

        magnitude = sum(value * value for value in embedding.dense) ** 0.5
        assert magnitude == pytest.approx(1.0, abs=1e-3)

    async def test_client_side_splitting_preserves_order(self, client) -> None:
        # `max_texts_per_request=4` forces the client to split these across
        # three real HTTP requests. This is the one path no in-process test
        # covers: reassembly across genuine round trips, on a reused connection.
        texts = [f"document number {index}" for index in range(9)]

        embeddings = await client.embed(texts)
        singles = [(await client.embed([text]))[0] for text in texts]

        assert len(embeddings) == 9
        assert embeddings == singles

    async def test_reranking_orders_by_relevance(self, client) -> None:
        # Worded so the stub and the real model agree on the answer. The stub
        # scores by literal token overlap, so "transactions" against
        # "transaction" is a miss for it and a hit for a cross-encoder — and a
        # test that only passes against one of the two backends is worse than
        # no test, because it looks like a real assertion.
        results = await client.rerank(
            "database transactions",
            [
                "The cat sat on the mat.",
                "Database transactions commit or abort together.",
                "Bananas are yellow.",
            ],
        )

        assert [result.score for result in results] == sorted(
            (result.score for result in results), reverse=True
        )
        assert results[0].index == 1

    async def test_over_length_input_is_refused_not_truncated(self, client) -> None:
        from rag.domain.errors import InvalidInputError

        info = await client.info()
        # Comfortably past the limit however it is configured, in tokens rather
        # than characters, so this holds against the stub and the real model.
        too_long = " ".join(["word"] * (info.max_sequence_tokens * 2))

        with pytest.raises(InvalidInputError):
            await client.embed([too_long])

    async def test_ping_reports_readiness(self, client) -> None:
        await client.ping()  # raises if not ready


@requires_model_service
class TestTheTokenizersAgree:
    """The mismatch the fingerprint mechanism exists to detect.

    If the worker sizes chunks with one vocabulary and the service embeds with
    another, chunks silently exceed the model's window and lose their tails — no
    error raised, and permanent once indexed.

    Neither test here skips conditionally on which backend is running, because
    `scripts/assert_suites_ran.py` fails CI on *any* skip in this suite and a
    test that quietly opts out under the one configuration CI uses is a test
    that never runs. Each branch of the condition gets its own real assertion
    instead.
    """

    async def test_the_stub_does_not_claim_a_vocabulary_it_lacks(self, client) -> None:
        # The branch CI exercises. A stub reporting a plausible-looking
        # fingerprint would make the comparison below pass against a service
        # that cannot tokenize at all — the fingerprint's whole job is to be
        # honest about what produced the vectors.
        info = await client.info()

        if info.embedding_model != "stub":
            pytest.skip("a real backend is running; covered by the test below")
        assert info.tokenizer_hash == "stub"

    async def test_a_real_service_shares_the_workers_vocabulary(self, client) -> None:
        # The branch the GPU box exercises. Reached only when both ends have a
        # real vocabulary, which is exactly when the comparison means something.
        from rag.adapters.tokenize import BgeTokenCounter

        info = await client.info()

        if info.embedding_model == "stub" or not LOCAL_TOKENIZER_PRESENT:
            pytest.skip("needs a real backend and a local tokenizer.json")
        assert BgeTokenCounter(LOCAL_TOKENIZER).fingerprint == info.tokenizer_hash
