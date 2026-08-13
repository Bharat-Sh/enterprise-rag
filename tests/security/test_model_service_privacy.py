"""The model service must not leak the text it is given.

Two distinct properties, both of which are easy to break by accident and neither
of which fails loudly when broken.

**Text must never reach a log.** Every string this service sees is customer
content crossing a trust boundary. The service is deliberately *tenant-blind* —
it never learns whose text it is holding — which means it cannot make an access
decision about a log line, so the only safe policy is that text never reaches
one. A single `_log.info("embedding", texts=payload.texts)` added for debugging
would ship an entire corpus into whatever aggregates the logs, and nothing about
that would look wrong in review.

**Text must never come back in an error body.** Error bodies get read, quoted in
tickets, and stored in the job queue's `error` column. `_reject_over_length`
therefore reports positions and token counts.

These are in `tests/security` and, per CLAUDE.md, must never be skipped or
marked xfail.
"""

from __future__ import annotations

import json
import logging

import pytest
from httpx import ASGITransport, AsyncClient

from model_service.app import create_app
from model_service.settings import Backend
from model_service.settings import Settings as ModelServiceSettings

#: Distinctive enough that finding it anywhere is unambiguous, and shaped like
#: the thing we actually care about not leaking.
SECRET = "zqxjkv-patient-record-4417-diagnosis-confidential"


@pytest.fixture
def logging_settings() -> ModelServiceSettings:
    """Settings at INFO, so the service actually logs something.

    The shared `model_service_settings` fixture runs at WARNING to keep the rest
    of the suite quiet. Reusing it here made every assertion below pass against
    an empty stream — the tests were green because *nothing was logged at all*,
    which is the exact failure mode a privacy test must not have.
    """
    return ModelServiceSettings(
        _env_file=None,
        backend=Backend.STUB,
        log_level="INFO",
        log_format="json",
        max_sequence_tokens=64,
        max_batch_tokens=96,
        max_texts_per_request=8,
    )


@pytest.fixture
async def logged_client(logging_settings):
    """A client for a service that is actually logging."""
    app = create_app(logging_settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://model") as client:
            yield client


def _emitted(caplog: pytest.LogCaptureFixture) -> str:
    """Everything the service logged during the call, as one searchable string.

    Read through `caplog` rather than `capsys` or `structlog.testing.capture_logs`,
    and both alternatives were tried first:

    *`capsys` does not work here.* `configure_logging` installs a `StreamHandler`
    holding whichever `sys.stdout` existed when `create_app` ran, and pytest
    swaps its capture buffer between the setup and call phases — so the handler
    writes into a buffer `readouterr()` no longer reads, and every assertion
    below passes against an empty string.

    *`capture_logs` tests the wrong pipeline.* It replaces the processor chain
    outright, so it would not see anything a renderer or `_redact_sensitive`
    contributed.

    `caplog` sits on the stdlib handler that structlog really emits through, and
    pytest reinstalls it per phase. The captured record carries the full event
    dictionary, which is exactly the payload that would have been rendered.
    """
    return "\n".join(record.getMessage() for record in caplog.records)


@pytest.fixture
def captured_logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.INFO, logger="model_service.app")
    return caplog


class TestTextNeverReachesALog:
    async def test_embedding(self, logged_client, captured_logs) -> None:
        response = await logged_client.post("/v1/embed", json={"texts": [SECRET]})

        assert response.status_code == 200
        assert captured_logs.records, "nothing was logged — this test would pass vacuously"
        assert SECRET not in _emitted(captured_logs)

    async def test_reranking(self, logged_client, captured_logs) -> None:
        response = await logged_client.post(
            "/v1/rerank", json={"query": SECRET, "passages": [SECRET, "unrelated"]}
        )

        assert response.status_code == 200
        assert captured_logs.records, "nothing was logged — this test would pass vacuously"
        assert SECRET not in _emitted(captured_logs)

    async def test_the_rejection_path(self, logged_client, captured_logs) -> None:
        # The error paths are the easiest place to leak: "log what went wrong"
        # reads as obviously correct right up until "what" is the document.
        too_long = " ".join([SECRET] * 100)

        response = await logged_client.post("/v1/embed", json={"texts": [too_long]})

        assert response.status_code == 422
        assert SECRET not in _emitted(captured_logs)

    async def test_what_is_logged_is_still_useful(self, logged_client, captured_logs) -> None:
        # A privacy rule that made the service unobservable would be traded away
        # the first time someone had to debug it at 3am. Counts, token totals
        # and durations are logged precisely so nobody needs the text.
        await logged_client.post("/v1/embed", json={"texts": ["one", "two", "three"]})

        events = [
            record.msg
            for record in captured_logs.records
            if isinstance(record.msg, dict) and record.msg.get("event") == "model_service.embedded"
        ]
        assert events, f"the embed path logs nothing at all; saw: {_emitted(captured_logs)!r}"
        assert events[0]["texts"] == 3
        assert events[0]["tokens"] > 0


class TestTextNeverComesBackInAnError:
    async def test_over_length_input_reports_positions_not_content(
        self, model_service_client
    ) -> None:
        # This body lands in the job queue's `error` column and in support
        # tickets. It has to say which chunk, never what was in it.
        too_long = " ".join([SECRET] * 100)

        response = await model_service_client.post("/v1/embed", json={"texts": ["fine", too_long]})

        assert response.status_code == 422
        body = response.text
        assert SECRET not in body
        assert json.loads(body)["detail"]["offenders"][0]["index"] == 1

    async def test_too_many_texts_reports_a_count(self, model_service_client) -> None:
        response = await model_service_client.post("/v1/embed", json={"texts": [SECRET] * 20})

        assert response.status_code == 413
        assert SECRET not in response.text


class TestAuthentication:
    """The shared secret, when one is configured."""

    @pytest.fixture
    def secured_settings(self) -> ModelServiceSettings:
        return ModelServiceSettings(
            _env_file=None,
            backend=Backend.STUB,
            log_level="WARNING",
            log_format="json",
            api_key="correct-horse-battery-staple",
        )

    @pytest.fixture
    async def secured_client(self, secured_settings):
        app = create_app(secured_settings)
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://model") as client:
                yield client

    @pytest.mark.parametrize(
        "path_and_body",
        [
            ("/v1/embed", {"texts": ["x"]}),
            ("/v1/rerank", {"query": "q", "passages": ["p"]}),
        ],
        ids=["embed", "rerank"],
    )
    async def test_no_credential_is_rejected(self, secured_client, path_and_body) -> None:
        path, body = path_and_body

        response = await secured_client.post(path, json=body)

        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"

    async def test_info_is_protected_too(self, secured_client) -> None:
        # /v1/info discloses the loaded model, its version and the tokenizer
        # fingerprint. Not catastrophic, but there is no reason for it to be the
        # one unauthenticated hole in an otherwise closed service.
        assert (await secured_client.get("/v1/info")).status_code == 401

    async def test_a_wrong_credential_is_rejected(self, secured_client) -> None:
        response = await secured_client.post(
            "/v1/embed",
            json={"texts": ["x"]},
            headers={"Authorization": "Bearer wrong-horse"},
        )

        assert response.status_code == 401

    async def test_the_wrong_scheme_is_rejected(self, secured_client) -> None:
        response = await secured_client.post(
            "/v1/embed",
            json={"texts": ["x"]},
            headers={"Authorization": "Basic correct-horse-battery-staple"},
        )

        assert response.status_code == 401

    async def test_the_correct_credential_is_accepted(self, secured_client) -> None:
        # Without this the four tests above would all pass against a service
        # that rejects everything, including the right answer.
        response = await secured_client.post(
            "/v1/embed",
            json={"texts": ["x"]},
            headers={"Authorization": "Bearer correct-horse-battery-staple"},
        )

        assert response.status_code == 200

    async def test_the_error_does_not_say_which_part_failed(self, secured_client) -> None:
        # Same reasoning as `rag.domain.errors.AuthenticationError`: every
        # distinction drawn for a legitimate caller's convenience is a
        # distinction an attacker uses.
        missing = await secured_client.get("/v1/info")
        wrong = await secured_client.get("/v1/info", headers={"Authorization": "Bearer nope"})

        assert missing.json() == wrong.json()

    async def test_probes_stay_open(self, secured_client) -> None:
        # An orchestrator's health probe has no credential, and a liveness probe
        # that 401s means the container is restarted forever.
        assert (await secured_client.get("/health")).status_code == 200
        assert (await secured_client.get("/ready")).status_code == 200


class TestUnauthenticatedByDefault:
    async def test_no_key_configured_means_no_check(self, model_service_client) -> None:
        # Correct on a laptop, and the reason the setting exists rather than the
        # check being unconditional. Pinned so that "auth is optional" stays a
        # decision rather than becoming an accident.
        assert (await model_service_client.get("/v1/info")).status_code == 200
