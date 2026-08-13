"""Health registry semantics and the liveness/readiness endpoints."""

from __future__ import annotations

import asyncio

import pytest

from rag.core.health import ComponentHealth, HealthRegistry, is_ready


async def _healthy() -> None:
    return None


async def _broken() -> None:
    raise ConnectionRefusedError("connection refused")


async def _slow() -> None:
    await asyncio.sleep(5)


class TestHealthRegistry:
    async def test_an_empty_registry_reports_no_components(self) -> None:
        assert await HealthRegistry().run_all() == []

    async def test_a_passing_check_is_healthy(self) -> None:
        registry = HealthRegistry()
        registry.register("postgres", _healthy)

        (component,) = await registry.run_all()

        assert component.name == "postgres"
        assert component.healthy is True
        assert component.detail is None

    async def test_a_raising_check_reports_the_exception(self) -> None:
        registry = HealthRegistry()
        registry.register("qdrant", _broken)

        (component,) = await registry.run_all()

        assert component.healthy is False
        assert component.detail is not None
        assert "ConnectionRefusedError" in component.detail

    async def test_a_hanging_check_is_bounded_by_its_timeout(self) -> None:
        # Without this, one wedged dependency hangs the probe, the orchestrator
        # times out, and a degraded service becomes a restarting one.
        registry = HealthRegistry()
        registry.register("slow-thing", _slow, timeout_seconds=0.01)

        (component,) = await registry.run_all()

        assert component.healthy is False
        assert component.detail is not None
        assert "timed out" in component.detail

    async def test_checks_run_concurrently_not_serially(self) -> None:
        async def _quarter_second() -> None:
            await asyncio.sleep(0.25)

        registry = HealthRegistry()
        for index in range(4):
            registry.register(f"dep-{index}", _quarter_second, timeout_seconds=2.0)

        start = asyncio.get_running_loop().time()
        await registry.run_all()
        elapsed = asyncio.get_running_loop().time() - start

        # Serial execution would take ~1.0s; concurrent takes ~0.25s.
        assert elapsed < 0.6

    def test_duplicate_registration_is_rejected(self) -> None:
        # A duplicate name almost always means a subsystem got wired up twice.
        registry = HealthRegistry()
        registry.register("postgres", _healthy)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("postgres", _healthy)

    def test_names_are_reported_in_registration_order(self) -> None:
        registry = HealthRegistry()
        registry.register("postgres", _healthy)
        registry.register("qdrant", _healthy)

        assert registry.names == ("postgres", "qdrant")


class TestReadinessAggregation:
    def test_no_components_means_ready(self) -> None:
        assert is_ready([]) is True

    def test_an_unhealthy_required_component_blocks_readiness(self) -> None:
        components = [ComponentHealth("postgres", healthy=False, required=True, duration_ms=1.0)]

        assert is_ready(components) is False

    def test_an_unhealthy_optional_component_does_not(self) -> None:
        # Redis is degradable: we lose caching and fall back to a local rate
        # limiter, but we can still answer questions. Pulling every replica out
        # of the load balancer over it would turn a slowdown into an outage.
        components = [
            ComponentHealth("postgres", healthy=True, required=True, duration_ms=1.0),
            ComponentHealth("redis", healthy=False, required=False, duration_ms=1.0),
        ]

        assert is_ready(components) is True


@pytest.fixture
def health_registry(client, app) -> HealthRegistry:
    """Swap the application's real dependency checks for an empty registry.

    From M1, `lifespan` registers a live Postgres check. These tests exercise
    the *endpoint contract*, not the wiring, and must not require a database to
    be running — otherwise the unit suite silently becomes an integration suite
    and stops being runnable anywhere.

    Depends on `client` so this runs after the lifespan has populated the real
    registry, replacing it rather than being overwritten by it.
    """
    registry = HealthRegistry()
    app.state.health = registry
    return registry


class TestLivenessEndpoint:
    async def test_liveness_returns_service_identity(self, client, settings) -> None:
        response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["service"] == settings.service_name
        assert body["environment"] == "local"

    async def test_liveness_does_not_consult_dependencies(self, client, health_registry) -> None:
        # The whole point of the split: a broken dependency must not make the
        # orchestrator restart an otherwise-healthy process.
        health_registry.register("postgres", _broken)

        assert (await client.get("/health")).status_code == 200


class TestReadinessEndpoint:
    async def test_ready_with_no_registered_checks(self, client, health_registry) -> None:
        response = await client.get("/ready")

        assert response.status_code == 200
        assert response.json() == {"ready": True, "components": []}

    async def test_ready_when_all_required_checks_pass(self, client, health_registry) -> None:
        health_registry.register("postgres", _healthy)

        response = await client.get("/ready")

        assert response.status_code == 200
        assert response.json()["ready"] is True

    async def test_not_ready_when_a_required_check_fails(self, client, health_registry) -> None:
        health_registry.register("postgres", _broken)

        response = await client.get("/ready")

        assert response.status_code == 503
        body = response.json()
        assert body["ready"] is False
        assert body["components"][0]["name"] == "postgres"
        assert body["components"][0]["healthy"] is False

    async def test_ready_despite_a_failing_optional_check(self, client, health_registry) -> None:
        health_registry.register("postgres", _healthy)
        health_registry.register("redis", _broken, required=False)

        response = await client.get("/ready")

        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        unhealthy = [c for c in body["components"] if not c["healthy"]]
        assert [c["name"] for c in unhealthy] == ["redis"]

    async def test_the_application_wires_a_real_postgres_check(self, client, app) -> None:
        # Complements the tests above, which deliberately isolate the endpoint.
        # Something still has to assert the production wiring exists, or an
        # accidentally-deleted `register` call would go unnoticed.
        assert "postgres" in app.state.health.names

    async def test_the_application_wires_the_model_service_check(self, client, app) -> None:
        assert "model-service" in app.state.health.names

    async def test_a_down_model_service_does_not_block_readiness(self, client, app) -> None:
        # M4 has no endpoint that needs a GPU — retrieval arrives in M6 — and
        # everything the API currently serves works while the model service is
        # down. Marking it required would pull every API replica out of the load
        # balancer whenever the GPU box restarts, converting a degraded feature
        # into a total outage.
        #
        # **M6 must flip this to required** and change this test, at the point
        # where there is an endpoint that cannot answer without it.
        #
        # Asserted over the component in isolation rather than over the whole
        # `/ready` response, because the registry also holds the Postgres check
        # and a unit test must not depend on a database being up.
        #
        # And asserted on `required` only, never on `healthy`. An earlier version
        # checked that the component was *unhealthy* — reasoning that nothing
        # listens on port 8001 during a unit run — and that was wrong: CI starts
        # a stub model service for the integration suite, in the same process
        # tree, so the check succeeds there and the test failed. Whether a
        # dependency happens to be up is not something a unit test may assume in
        # either direction.
        components = await app.state.health.run_all()
        model_service = next(item for item in components if item.name == "model-service")

        assert model_service.required is False
        # The consequence that actually matters, stated without a network: an
        # unhealthy optional component does not make the process unready.
        unhealthy = ComponentHealth(
            name="model-service", healthy=False, required=False, duration_ms=1.0
        )
        assert is_ready([unhealthy]) is True


class TestRequestContextHeaders:
    async def test_ids_are_returned_on_every_response(self, client) -> None:
        response = await client.get("/health")

        assert response.headers["x-request-id"]
        assert response.headers["x-trace-id"]

    async def test_trace_id_defaults_to_the_request_id(self, client) -> None:
        response = await client.get("/health")

        assert response.headers["x-trace-id"] == response.headers["x-request-id"]

    async def test_a_valid_inbound_id_is_propagated(self, client) -> None:
        response = await client.get("/health", headers={"x-trace-id": "trace-abc123"})

        assert response.headers["x-trace-id"] == "trace-abc123"

    async def test_a_malformed_inbound_id_is_replaced(self, client) -> None:
        # Caller-controlled values end up in log files and dashboards. An
        # unvalidated one is a log-injection vector, not a nicety.
        hostile = "abc\r\nX-Injected: yes"

        response = await client.get("/health", headers={"x-request-id": hostile})

        assert response.headers["x-request-id"] != hostile
        assert "\n" not in response.headers["x-request-id"]

    async def test_an_overlong_inbound_id_is_replaced(self, client) -> None:
        response = await client.get("/health", headers={"x-request-id": "a" * 500})

        assert len(response.headers["x-request-id"]) < 200

    async def test_response_time_header_is_present(self, client) -> None:
        response = await client.get("/health")

        assert float(response.headers["x-response-time-ms"]) >= 0
