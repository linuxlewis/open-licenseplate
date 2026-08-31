from pathlib import Path

import pytest
from pydantic import ValidationError

from open_licenseplate.config import load_settings


def test_settings_precedence_is_cli_then_environment_then_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPEN_LICENSEPLATE_STORAGE__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPEN_LICENSEPLATE_SERVER__PORT", "9002")
    monkeypatch.setenv("OPEN_LICENSEPLATE_LOG_LEVEL", "DEBUG")

    settings = load_settings(cli_overrides={"server.port": 9003})

    assert settings.server.port == 9003
    assert settings.log_level == "DEBUG"
    assert settings.server.host == "127.0.0.1"
    assert settings.sources["server.port"] == "cli"
    assert settings.sources["log_level"] == "environment"
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


def test_non_loopback_binding_requires_explicit_unsafe_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPEN_LICENSEPLATE_STORAGE__DATA_DIR", str(tmp_path / "data"))

    with pytest.raises(ValidationError):
        load_settings(
            cli_overrides={"server.host": "0.0.0.0"},
        )


def test_ui_density_is_typed_and_accepts_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPEN_LICENSEPLATE_STORAGE__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPEN_LICENSEPLATE_UI__DENSITY", "compact")

    settings = load_settings()

    assert settings.ui.density == "compact"
    assert settings.sources["ui.density"] == "environment"

    monkeypatch.setenv("OPEN_LICENSEPLATE_UI__DENSITY", "spacious")
    with pytest.raises(ValidationError):
        load_settings()
