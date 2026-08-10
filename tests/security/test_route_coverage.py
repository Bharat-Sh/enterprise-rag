"""Every route is authenticated unless it is on a short, explicit list.

This is the highest-value test in M2 and it needs no database, so it always
runs. Individual endpoint tests prove that the endpoints we wrote are protected;
this one proves that the endpoints written in M3 through M13 will be too,
because adding an unauthenticated route becomes a test failure rather than an
omission nobody notices.

The exemption list is deliberately spelled out here rather than derived. A
derived list would exempt exactly whatever the code currently does, which is the
opposite of a check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from fastapi.routing import APIRoute

from rag.api.deps import get_principal, get_unit_of_work, get_unscoped_unit_of_work
from rag.api.main import create_app
from rag.core.config import Environment, LogFormat, Settings
from tests.support import generate_private_pem

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastapi import FastAPI
    from fastapi.dependencies.models import Dependant

pytestmark = pytest.mark.security

#: Routes that legitimately serve an unauthenticated caller, and why.
UNAUTHENTICATED: dict[str, str] = {
    "/health": "liveness probe; must answer before anything else works",
    "/ready": "readiness probe; an orchestrator has no credential",
    "/.well-known/jwks.json": "public keys, published so others can verify our tokens",
    "/api/v1/auth/login": "creates the credential; cannot require one",
    "/api/v1/auth/refresh": "authenticates with the refresh token in its body",
    "/api/v1/auth/logout": "must work after the access token has expired",
}

#: The only endpoints allowed to open a transaction with no tenant bound. They
#: resolve the tenant themselves — login from the submitted slug, the other two
#: from the refresh token's own tenant segment — and scope as their first act.
UNSCOPED_ALLOWED: set[str] = {
    "/api/v1/auth/login",
    "/api/v1/auth/refresh",
    "/api/v1/auth/logout",
}


@pytest.fixture(scope="module")
def app() -> FastAPI:
    return create_app(
        Settings(
            _env_file=None,
            environment=Environment.LOCAL,
            log_level="CRITICAL",
            log_format=LogFormat.JSON,
            auth={"private_key_pem": generate_private_pem()},
        )
    )


def _walk(dependant: Dependant) -> Iterator[Dependant]:
    """Every dependency in the tree, however deeply nested."""
    yield dependant
    for child in dependant.dependencies:
        yield from _walk(child)


def _calls(route: APIRoute) -> set[object]:
    return {node.call for node in _walk(route.dependant) if node.call is not None}


@dataclass(frozen=True)
class Endpoint:
    """A route and the full path it is actually served at."""

    path: str
    route: APIRoute

    @property
    def calls(self) -> set[object]:
        return _calls(self.route)


def _collect(container: object, prefix: str = "") -> Iterator[Endpoint]:
    """Flatten every API route, whatever shape the router tree is in.

    FastAPI has changed this twice: older versions copied included routes
    straight onto `app.routes` with their full path, current ones keep an
    `_IncludedRouter` wrapper holding the original router plus the prefix it was
    mounted under. Handling both keeps this test from quietly finding *nothing*
    on an upgrade — which is the one failure mode a coverage test must not have,
    because it passes.
    """
    for route in getattr(container, "routes", []):
        if isinstance(route, APIRoute):
            yield Endpoint(path=prefix + route.path, route=route)
            continue
        original = getattr(route, "original_router", None)
        if original is not None:
            context = getattr(route, "include_context", None)
            yield from _collect(original, prefix + (getattr(context, "prefix", "") or ""))
        elif hasattr(route, "routes"):
            yield from _collect(route, prefix)


def _routes(app: FastAPI) -> list[Endpoint]:
    return list(_collect(app))


class TestEnumerationIsNotVacuous:
    """Guards the guard.

    Every other test in this file is of the form "no route violates X". If the
    enumeration silently found nothing, they would all pass while checking
    nothing at all — the single most dangerous shape a security test can take.
    """

    def test_every_documented_path_is_discovered(self, app: FastAPI) -> None:
        documented = set(app.openapi()["paths"])
        discovered = {endpoint.path for endpoint in _routes(app)}

        assert documented <= discovered, (
            f"Route enumeration missed {sorted(documented - discovered)}. The "
            f"router tree shape has probably changed; fix `_collect` before "
            f"trusting anything else in this file."
        )

    def test_the_exemption_list_has_no_stale_entries(self, app: FastAPI) -> None:
        # A stale entry is a route that *will* be exempt the moment someone
        # re-adds that path — a trap left lying around.
        paths = {endpoint.path for endpoint in _routes(app)}

        assert set(UNAUTHENTICATED) <= paths


class TestAuthenticationCoverage:
    def test_every_route_resolves_a_principal_unless_explicitly_exempt(self, app: FastAPI) -> None:
        unprotected = sorted(
            endpoint.path
            for endpoint in _routes(app)
            if get_principal not in endpoint.calls and endpoint.path not in UNAUTHENTICATED
        )

        assert not unprotected, (
            f"These routes resolve no principal and are not on the exemption "
            f"list in {__file__}: {unprotected}. Add authentication, or add an "
            f"entry with a reason."
        )

    def test_exempt_routes_really_are_reachable_without_a_credential(self, app: FastAPI) -> None:
        for endpoint in _routes(app):
            if endpoint.path in UNAUTHENTICATED:
                assert get_principal not in endpoint.calls, endpoint.path


class TestTenantScopeCoverage:
    def test_only_the_credential_endpoints_may_open_an_unscoped_transaction(
        self, app: FastAPI
    ) -> None:
        """The property the whole milestone rests on.

        `get_unit_of_work` binds row-level security from the verified credential
        before a handler runs, so a route that uses it cannot skip
        authentication. `get_unscoped_unit_of_work` is the escape hatch, and it
        must stay a three-endpoint escape hatch.
        """
        unscoped = sorted(
            endpoint.path
            for endpoint in _routes(app)
            if get_unscoped_unit_of_work in endpoint.calls
        )

        assert set(unscoped) == UNSCOPED_ALLOWED, (
            "A new endpoint is opening a transaction with no tenant bound. That "
            "is only correct for endpoints that resolve the tenant themselves."
        )

    def test_a_scoped_transaction_always_comes_with_a_principal(self, app: FastAPI) -> None:
        for endpoint in _routes(app):
            if get_unit_of_work in endpoint.calls:
                assert get_principal in endpoint.calls, endpoint.path


class TestParameterResolution:
    def test_no_route_has_a_required_query_parameter(self, app: FastAPI) -> None:
        """Two reasons, one of them a real bug this caught.

        As API design: this service takes identity in the path and input in the
        body, so a *required* query parameter is a smell.

        As a guard: FastAPI resolves dependency annotations with
        `get_type_hints` at registration. A name that exists only under
        `TYPE_CHECKING` cannot be resolved, and the parameter is silently
        reinterpreted as a required query parameter — turning a correct-looking
        route into a blanket 422. Nothing else in the suite notices, because the
        route never reaches its handler.
        """
        offenders = [
            (endpoint.path, parameter.name)
            for endpoint in _routes(app)
            for parameter in endpoint.route.dependant.query_params
            if parameter.field_info.is_required()
        ]

        assert not offenders, (
            f"Required query parameters found: {offenders}. If these were not "
            f"intended, a dependency annotation failed to resolve — check that "
            f"every type used in an `Annotated[..., Depends(...)]` alias is "
            f"imported at runtime rather than under TYPE_CHECKING."
        )


class TestOpenApi:
    def test_protected_routes_advertise_the_bearer_scheme(self, app: FastAPI) -> None:
        # A dependency, not middleware, precisely so this shows up in the schema:
        # generated clients and the docs "Authorize" button both read it.
        schema = app.openapi()

        operation = schema["paths"]["/api/v1/auth/me"]["get"]
        assert operation["security"]
        assert "Bearer" in schema["components"]["securitySchemes"]

    def test_the_login_route_advertises_no_security(self, app: FastAPI) -> None:
        schema = app.openapi()

        assert "security" not in schema["paths"]["/api/v1/auth/login"]["post"]
