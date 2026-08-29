from pathlib import Path

import pytest
from pydantic import ValidationError

from open_licenseplate.config import load_settings


def test_settings_precedence_is_cli_then_environment_then_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPEN_LICENSEPLATE_SERVER__PORT", "9002")
    monkeypatch.setenv("OPEN_LICENSEPLATE_LOG_LEVEL", "DEBUG")

    settings = load_settings(cli_overrides={"server.port": 9003})

    assert settings.server.port == 9003
    assert settings.log_level == "DEBUG"
    assert settings.live.detection_confidence == 0.35
    assert settings.server.host == "127.0.0.1"
    assert settings.sources["server.port"] == "cli"
    assert settings.sources["log_level"] == "environment"
    assert settings.sources["live.detection_confidence"] == "default"
    assert settings.sources["server.host"] == "default"


def test_environment_settings_support_nested_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPEN_LICENSEPLATE_STORAGE__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPEN_LICENSEPLATE_SERVER__UNSAFE_DEVELOPMENT", "true")
    monkeypatch.setenv("OPEN_LICENSEPLATE_SERVER__HOST", "0.0.0.0")

    settings = load_settings()

    assert settings.storage.data_dir == tmp_path / "data"
    assert settings.server.host == "0.0.0.0"
    assert settings.server.unsafe_development is True


def test_non_loopback_binding_requires_explicit_unsafe_flag() -> None:
    with pytest.raises(ValidationError):
        load_settings(
            cli_overrides={"server.host": "0.0.0.0"},
        )


def test_typed_enum_settings_reject_unknown_values() -> None:
    with pytest.raises(ValidationError):
        load_settings(
            cli_overrides={"live.compute_units": "neural_engine_only"},
        )
