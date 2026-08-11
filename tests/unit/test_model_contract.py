"""The wire contract, checked from both ends.

`model_service.schemas` and `rag.adapters.models.schemas` are two independent
copies of the same shapes. That duplication is deliberate — the processes deploy
separately, so the client has to tolerate a server that has grown a field it
does not know about — but duplication without a check is just drift with extra
steps.

**This file is the reason both ends live in one repository.** Split across two
repos, nothing would fail when a field was renamed on the server; it would be
discovered by a 500 in production, or worse, by vectors that parse and are
wrong. Here it is a red build.

The check is deliberately asymmetric, matching the direction the data flows:
every field the *client* requires must be produced by the *server*. The reverse
is not required — a server field the client ignores is exactly how a new field
gets deployed without a synchronised release.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import BaseModel

from model_service import schemas as server
from rag.adapters.models import schemas as client


def _embed_response() -> BaseModel:
    return server.EmbedResponse(
        embedding_model="bge-m3",
        embedding_version="rev1",
        dimensions=2,
        embeddings=[
            server.EmbeddingPayload(
                dense=[0.6, 0.8],
                sparse=server.SparsePayload(indices=[7, 19], values=[0.9, 0.1]),
            )
        ],
    )


def _rerank_response() -> BaseModel:
    return server.RerankResponse(
        reranker_model="bge-reranker-v2-m3",
        results=[server.RerankScore(index=0, score=1.5)],
    )


def _info_response() -> BaseModel:
    return server.InfoResponse(
        embedding_model="bge-m3",
        embedding_version="rev1",
        dimensions=1024,
        max_sequence_tokens=1024,
        tokenizer_hash="0123456789abcdef",
        reranker_model="bge-reranker-v2-m3",
        backend="flag",
    )


#: (server response model, client model, a factory for an example instance).
#:
#: Hand-written examples rather than generated data, because an example is what
#: catches a *type* change — the drift a field-name comparison misses entirely.
#:
#: Factories rather than instances, so a renamed server field fails the one test
#: that covers it instead of erroring at import and taking the whole module down
#: with a message about an example rather than about the contract.
PAIRS: list[tuple[type[BaseModel], type[BaseModel], Callable[[], BaseModel]]] = [
    (server.EmbedResponse, client.EmbedResponseModel, _embed_response),
    (server.RerankResponse, client.RerankResponseModel, _rerank_response),
    (server.InfoResponse, client.InfoResponseModel, _info_response),
]


class TestTheTwoEndsAgree:
    @pytest.mark.parametrize(
        ("server_model", "client_model", "make_example"),
        PAIRS,
        ids=[pair[0].__name__ for pair in PAIRS],
    )
    def test_a_server_response_parses_on_the_client(
        self,
        server_model: type[BaseModel],
        client_model: type[BaseModel],
        make_example: Callable[[], BaseModel],
    ) -> None:
        # Serialised the way the wire does it — through JSON — so a type that
        # only survives in Python (a tuple, an enum, a Decimal) is caught here
        # rather than at runtime.
        example = make_example()

        parsed = client_model.model_validate_json(example.model_dump_json())

        for name in client_model.model_fields:
            assert hasattr(parsed, name)

    @pytest.mark.parametrize(
        ("server_model", "client_model", "make_example"),
        PAIRS,
        ids=[pair[0].__name__ for pair in PAIRS],
    )
    def test_every_field_the_client_requires_is_sent_by_the_server(
        self,
        server_model: type[BaseModel],
        client_model: type[BaseModel],
        make_example: Callable[[], BaseModel],
    ) -> None:
        # The direction that matters. A field the client demands and the server
        # does not send is a hard failure at the boundary on the first request
        # after deploy.
        #
        # Read off the *server model* rather than the example, so a field the
        # example simply forgot to set cannot make this pass.
        required = {
            name for name, field in client_model.model_fields.items() if field.is_required()
        }
        sent = set(server_model.model_fields)

        assert required <= sent, f"client requires fields the server never sends: {required - sent}"

    def test_the_client_survives_a_field_it_has_never_heard_of(self) -> None:
        # The whole justification for keeping two copies. A server that grows a
        # field must not require a synchronised client deploy.
        payload = {
            "embedding_model": "bge-m3",
            "embedding_version": "rev1",
            "dimensions": 1024,
            "max_sequence_tokens": 1024,
            "tokenizer_hash": "0123456789abcdef",
            "colbert_dimensions": 128,
        }

        assert client.InfoResponseModel.model_validate(payload).dimensions == 1024


class TestTheContractIsFullyCovered:
    def test_every_server_response_model_has_a_client_counterpart(self) -> None:
        # A pairwise test list is only as good as its completeness. Without
        # this, adding a fourth endpoint in M6 and forgetting to pair it here
        # would leave the new contract unchecked while the suite stayed green —
        # a test that passes by omission.
        response_models = {name for name in server.__all__ if name.endswith("Response")}
        paired = {pair[0].__name__ for pair in PAIRS}

        assert response_models == paired
