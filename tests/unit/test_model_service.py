"""The model service's HTTP contract, served by the stub backend.

No GPU, no torch, and yet this is the real application: real routing, real
validation, real batching, real error mapping. The only thing swapped out is the
matrix multiplication, which is the one part a GPU would add and the one part
that does not have interesting failure modes at this level.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from model_service.app import create_app
from model_service.backend import BackendInfo, StubBackend
from model_service.settings import Backend, Settings


class TestInfo:
    async def test_it_reports_what_is_loaded(self, model_service_client) -> None:
        response = await model_service_client.get("/v1/info")

        assert response.status_code == 200
        body = response.json()
        assert body["dimensions"] == 1024
        assert body["max_sequence_tokens"] == 64
        assert body["backend"] == "stub"

    async def test_the_stub_identifies_itself(self, model_service_client) -> None:
        # This string is stamped into `chunks.embedding_model` for every row the
        # stub produces. It is what makes an accidental stub deployment visible
        # in the database rather than looking like real vectors that merely
        # retrieve badly.
        body = (await model_service_client.get("/v1/info")).json()

        assert body["embedding_model"] == "stub"


class TestEmbed:
    async def test_it_returns_one_embedding_per_text_in_order(self, model_service_client) -> None:
        # Order is the contract: the caller pairs results back to chunk ids
        # positionally. Asserted by embedding the same text twice at different
        # positions and checking it lands where it was asked for.
        texts = ["alpha", "beta", "alpha"]

        response = await model_service_client.post("/v1/embed", json={"texts": texts})

        assert response.status_code == 200
        embeddings = response.json()["embeddings"]
        assert len(embeddings) == 3
        assert embeddings[0] == embeddings[2]
        assert embeddings[0] != embeddings[1]

    async def test_dense_vectors_are_l2_normalised(self, model_service_client) -> None:
        # Normalising in the service means cosine similarity is a dot product
        # and no caller can forget. A caller that forgot would get scores that
        # are wrong and entirely plausible.
        body = (
            await model_service_client.post("/v1/embed", json={"texts": ["hello world"]})
        ).json()

        dense = body["embeddings"][0]["dense"]
        magnitude = sum(value * value for value in dense) ** 0.5
        assert magnitude == pytest.approx(1.0, abs=1e-6)

    async def test_sparse_indices_are_unique_and_sorted(self, model_service_client) -> None:
        # Qdrant rejects a sparse vector with repeated indices, and "sorted" is
        # what makes two vectors comparable without a normalisation step.
        body = (
            await model_service_client.post(
                "/v1/embed", json={"texts": ["the cat sat on the mat the end"]}
            )
        ).json()

        sparse = body["embeddings"][0]["sparse"]
        assert sparse["indices"] == sorted(set(sparse["indices"]))
        assert len(sparse["indices"]) == len(sparse["values"])

    async def test_batching_does_not_change_the_answer(self, model_service_client) -> None:
        # The fixture's budgets force these six texts across several forward
        # passes. Reassembly is the risk, so the same texts are embedded one at
        # a time and the results must be identical.
        texts = [f"text number {index}" for index in range(6)]

        batched = (await model_service_client.post("/v1/embed", json={"texts": texts})).json()[
            "embeddings"
        ]
        one_at_a_time = [
            (await model_service_client.post("/v1/embed", json={"texts": [text]})).json()[
                "embeddings"
            ][0]
            for text in texts
        ]

        assert batched == one_at_a_time

    async def test_mode_is_accepted_and_ignored(self, model_service_client) -> None:
        # BGE-M3 uses no asymmetric instruction prefix, so query and passage
        # embeddings are identical. The parameter exists for providers that do.
        as_query = await model_service_client.post(
            "/v1/embed", json={"texts": ["x"], "mode": "query"}
        )
        as_passage = await model_service_client.post(
            "/v1/embed", json={"texts": ["x"], "mode": "passage"}
        )

        assert as_query.status_code == 200
        assert as_query.json()["embeddings"] == as_passage.json()["embeddings"]

    async def test_an_unknown_mode_is_rejected(self, model_service_client) -> None:
        response = await model_service_client.post(
            "/v1/embed", json={"texts": ["x"], "mode": "sideways"}
        )

        assert response.status_code == 422


class TestLimits:
    async def test_over_length_input_is_rejected_not_truncated(self, model_service_client) -> None:
        # The single most important behaviour here. Truncating produces a vector
        # that is structurally perfect and missing the end of the text — no
        # error, no warning, permanent once indexed.
        long_text = " ".join(["word"] * 200)

        response = await model_service_client.post("/v1/embed", json={"texts": [long_text]})

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["error"] == "input_too_long"
        assert detail["offenders"][0]["index"] == 0

    async def test_the_offending_positions_are_reported(self, model_service_client) -> None:
        # An operator has to know *which* chunk is too long. Positions and token
        # counts only — never the text itself.
        texts = ["short", " ".join(["word"] * 200), "also short"]

        response = await model_service_client.post("/v1/embed", json={"texts": texts})

        detail = response.json()["detail"]
        assert [offender["index"] for offender in detail["offenders"]] == [1]

    async def test_too_many_texts_is_413(self, model_service_client) -> None:
        response = await model_service_client.post("/v1/embed", json={"texts": ["x"] * 9})

        assert response.status_code == 413
        assert response.json()["detail"]["error"] == "too_many_texts"

    async def test_an_empty_request_is_rejected(self, model_service_client) -> None:
        response = await model_service_client.post("/v1/embed", json={"texts": []})

        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "empty_request"


class TestRerank:
    async def test_results_are_ordered_highest_score_first(self, model_service_client) -> None:
        response = await model_service_client.post(
            "/v1/rerank",
            json={
                "query": "database transactions",
                "passages": [
                    "cats are small carnivorous mammals",
                    "database transactions are atomic",
                    "the weather is fine today",
                ],
            },
        )

        assert response.status_code == 200
        results = response.json()["results"]
        scores = [result["score"] for result in results]
        assert scores == sorted(scores, reverse=True)
        # The stub scores by token overlap, so the relevant passage really does
        # win. A constant-scoring stub would have made this assertion pass for
        # the wrong reason.
        assert results[0]["index"] == 1

    async def test_indices_refer_to_the_request_order(self, model_service_client) -> None:
        response = await model_service_client.post(
            "/v1/rerank",
            json={"query": "zebra", "passages": ["nothing", "zebra zebra", "nothing else"]},
        )

        results = response.json()["results"]
        assert {result["index"] for result in results} == {0, 1, 2}

    async def test_top_k_trims_the_response(self, model_service_client) -> None:
        response = await model_service_client.post(
            "/v1/rerank",
            json={"query": "a", "passages": ["a", "b", "c", "d"], "top_k": 2},
        )

        assert len(response.json()["results"]) == 2

    async def test_the_pair_is_measured_not_the_passage(self, model_service_client) -> None:
        # A cross-encoder sees query and passage concatenated, so the *pair* has
        # to fit the window. Checking the passage alone accepts pairs the model
        # then truncates, dropping the end of the passage.
        query = " ".join(["query"] * 40)
        passage = " ".join(["passage"] * 40)

        response = await model_service_client.post(
            "/v1/rerank", json={"query": query, "passages": [passage]}
        )

        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "input_too_long"

    async def test_too_many_passages_is_413(self, model_service_client) -> None:
        response = await model_service_client.post(
            "/v1/rerank", json={"query": "q", "passages": ["p"] * 9}
        )

        assert response.status_code == 413


class TestRerankerDisabled:
    async def test_it_answers_501_rather_than_failing_obscurely(self) -> None:
        # A card too small for both models is a supported configuration. The
        # caller needs to be told it asked for something this deployment cannot
        # do, not handed a 500.
        class NoReranker(StubBackend):
            def info(self) -> BackendInfo:
                return BackendInfo(
                    embedding_model="stub",
                    embedding_version="stub",
                    dimensions=8,
                    tokenizer_hash="stub",
                    backend="stub",
                    reranker_model=None,
                )

        settings = Settings(_env_file=None, backend=Backend.STUB, log_level="WARNING")
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            app.state.backend = NoReranker()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://model") as client:
                response = await client.post("/v1/rerank", json={"query": "q", "passages": ["p"]})

        assert response.status_code == 501
        assert response.json()["detail"]["error"] == "reranker_disabled"


class TestProbes:
    async def test_liveness_needs_no_models(self, model_service_settings) -> None:
        # Answers before and during the model load. Wiring readiness into this
        # probe would restart the container every cold start, forever, and it
        # would never once reach the point where it works.
        app = create_app(model_service_settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://model") as client:
            response = await client.get("/health")

        assert response.status_code == 200

    async def test_readiness_is_false_until_the_models_are_loaded(
        self, model_service_settings
    ) -> None:
        app = create_app(model_service_settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://model") as client:
            before = await client.get("/ready")

        assert before.status_code == 503

    async def test_readiness_is_true_once_loaded(self, model_service_client) -> None:
        assert (await model_service_client.get("/ready")).status_code == 200


class TestDeterminism:
    def test_the_stub_is_stable_across_instances(self) -> None:
        # Tests compare results between calls, and an integration test compares
        # the batched path against the single-pass one. A stub seeded from
        # anything but the text would make both of those meaningless.
        first = StubBackend().embed(["repeatable"], mode="passage")
        second = StubBackend().embed(["repeatable"], mode="passage")

        assert first == second
