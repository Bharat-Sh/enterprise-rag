"""Configuration behaviour: env binding, secret hygiene, derived values, fail-fast."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from rag.core.config import Environment, LogFormat, Settings, get_settings

# A password that satisfies the production check, so tests can exercise *other*
# production rules without tripping the default-credential guard first.
SAFE_PASSWORD = "not-the-development-default"  # noqa: S105 - test fixture, not a real secret


def _make(**overrides: Any) -> Settings:
    """Build settings without reading the developer's .env file.

    Hermetic by construction: the suite must not pass or fail depending on
    whose machine it runs on.
    """
    return Settings(_env_file=None, **overrides)


class TestEnvironmentBinding:
    def test_nested_values_bind_with_double_underscore(self, monkeypatch) -> None:
        monkeypatch.setenv("RAG_DATABASE__HOST", "db.internal")
        monkeypatch.setenv("RAG_DATABASE__PORT", "6543")

        settings = _make()

        assert settings.database.host == "db.internal"
        assert settings.database.port == 6543

    def test_single_underscore_does_not_bind_nested_values(self, monkeypatch) -> None:
        # Guards the most common mistake with this scheme: the delimiter is
        # `__`, so `RAG_DATABASE_HOST` silently does nothing at all.
        monkeypatch.setenv("RAG_DATABASE_HOST", "wrong.example")

        settings = _make()

        assert settings.database.host == "localhost"

    def test_top_level_values_bind_with_the_prefix(self, monkeypatch) -> None:
        monkeypatch.setenv("RAG_SERVICE_NAME", "rag-worker")

        assert _make().service_name == "rag-worker"

    def test_unknown_variables_are_ignored(self, monkeypatch) -> None:
        monkeypatch.setenv("RAG_TOTALLY_UNKNOWN_SETTING", "1")

        _make()  # extra="ignore" — must not raise

    def test_out_of_range_values_are_rejected(self, monkeypatch) -> None:
        monkeypatch.setenv("RAG_DATABASE__PORT", "99999")

        with pytest.raises(ValidationError):
            _make()


class TestSecretHandling:
    def test_password_does_not_appear_in_repr(self, monkeypatch) -> None:
        # The first time an exception serialises a settings object, this is the
        # difference between a log line and an incident.
        monkeypatch.setenv("RAG_DATABASE__PASSWORD", "hunter2")

        settings = _make()

        assert "hunter2" not in repr(settings)
        assert "hunter2" not in str(settings.database.password)

    def test_safe_dsn_redacts_the_password(self, monkeypatch) -> None:
        monkeypatch.setenv("RAG_DATABASE__PASSWORD", "hunter2")

        safe = _make().database.safe_dsn

        assert "hunter2" not in safe
        assert "***" in safe
        assert safe.startswith("postgresql+asyncpg://")

    def test_dsn_contains_the_password_because_the_driver_needs_it(self, monkeypatch) -> None:
        monkeypatch.setenv("RAG_DATABASE__PASSWORD", "hunter2")

        assert "hunter2" in _make().database.dsn

    def test_redis_safe_url_hides_the_presence_of_a_password(self, monkeypatch) -> None:
        monkeypatch.setenv("RAG_REDIS__PASSWORD", "s3cret")

        safe = _make().redis.safe_url

        assert "s3cret" not in safe
        assert "***@" in safe


class TestDerivedValues:
    def test_local_defaults_to_console_logs(self) -> None:
        assert _make(environment=Environment.LOCAL).effective_log_format is LogFormat.CONSOLE

    def test_non_local_defaults_to_json_logs(self) -> None:
        assert _make(environment=Environment.DEV).effective_log_format is LogFormat.JSON

    def test_explicit_log_format_wins(self) -> None:
        settings = _make(environment=Environment.DEV, log_format=LogFormat.CONSOLE)

        assert settings.effective_log_format is LogFormat.CONSOLE

    def test_docs_are_enabled_outside_production(self) -> None:
        settings = _make(environment=Environment.LOCAL)

        assert settings.effective_docs_enabled is True
        assert settings.expose_error_details is True

    def test_docs_and_error_details_are_off_in_production(self) -> None:
        settings = _make(
            environment=Environment.PROD,
            database={"password": SAFE_PASSWORD},
        )

        assert settings.effective_docs_enabled is False
        assert settings.expose_error_details is False


class TestFailFast:
    """Production-like environments must refuse to start carrying dev defaults.

    This is the point of validating configuration eagerly: shipping to staging
    with the development database password should be impossible to deploy, not
    something discovered later during an audit.
    """

    def test_default_password_is_rejected_in_production(self) -> None:
        with pytest.raises(ValidationError, match="development default"):
            _make(environment=Environment.PROD)

    def test_default_password_is_rejected_in_staging_too(self) -> None:
        with pytest.raises(ValidationError, match="development default"):
            _make(environment=Environment.STAGING)

    def test_reload_is_rejected_in_production(self) -> None:
        with pytest.raises(ValidationError, match="reload"):
            _make(
                environment=Environment.PROD,
                database={"password": SAFE_PASSWORD},
                server={"reload": True},
            )

    def test_explicitly_enabling_docs_is_rejected_in_production(self) -> None:
        with pytest.raises(ValidationError, match="docs_enabled"):
            _make(
                environment=Environment.PROD,
                database={"password": SAFE_PASSWORD},
                docs_enabled=True,
            )

    def test_a_valid_production_config_is_accepted(self) -> None:
        settings = _make(
            environment=Environment.PROD,
            database={"password": SAFE_PASSWORD},
        )

        assert settings.environment.is_production_like is True

    def test_local_tolerates_dev_defaults(self) -> None:
        settings = _make(environment=Environment.LOCAL, server={"reload": True})

        assert settings.server.reload is True
        assert settings.environment.is_production_like is False


class TestSingleton:
    def test_get_settings_is_cached(self) -> None:
        get_settings.cache_clear()
        try:
            assert get_settings() is get_settings()
        finally:
            get_settings.cache_clear()
