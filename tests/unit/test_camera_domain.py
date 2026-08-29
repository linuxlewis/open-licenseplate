from __future__ import annotations

import json

import pytest

from open_licenseplate.cameras.credentials import (
    credential_status,
    parse_credential_ref,
)
from open_licenseplate.cameras.repository import Camera, CameraConfig
from open_licenseplate.cameras.service import (
    CameraConfigurationError,
    prepare_camera_config,
)
from open_licenseplate.cameras.service import (
    test_camera_configuration as run_camera_configuration,
)


def _camera(config: CameraConfig) -> Camera:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return Camera(
        id="camera-1",
        name=config.name,
        endpoint=config.endpoint,
        credential_ref=config.credential_ref,
        connection_options_json=json.dumps(config.connection_options),
        preferred_stream=config.preferred_stream,
        region_of_interest_json=None,
        enabled=config.enabled,
        created_at=now,
        updated_at=now,
    )


def test_prepare_camera_config_redacts_embedded_credentials() -> None:
    config = prepare_camera_config(
        name="Front gate",
        rtsp_url="rtsp://operator:secret-value@example.test:554/live?token=query-value",
        credential_ref="env:CAMERA_RTSP_URL",
    )

    assert config.endpoint == "rtsp://[REDACTED]@example.test:554/live?token=%5BREDACTED%5D"
    assert "secret-value" not in config.endpoint
    assert "query-value" not in config.endpoint
    assert config.connection_options == {"transport": "tcp"}


def test_embedded_credentials_require_external_reference() -> None:
    with pytest.raises(
        CameraConfigurationError,
        match="credential_ref is required",
    ):
        prepare_camera_config(
            name="Front gate",
            rtsp_url="rtsp://operator:secret-value@example.test/live",
        )


def test_credential_references_are_limited_to_safe_formats() -> None:
    environment = parse_credential_ref("env:CAMERA_RTSP_URL")
    keychain = parse_credential_ref("keychain:open-licenseplate/camera-1")

    assert environment is not None
    assert environment.kind == "environment"
    assert keychain is not None
    assert keychain.kind == "keychain"
    assert credential_status(None)["status"] == "not_configured"

    with pytest.raises(ValueError, match="credential_ref"):
        parse_credential_ref("password=secret-value")


def test_configuration_test_resolves_external_url_without_returning_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CAMERA_RTSP_URL",
        "rtsp://operator:secret-value@example.test:554/live",
    )
    config = prepare_camera_config(
        name="Front gate",
        rtsp_url="rtsp://example.test:554/live",
        credential_ref="env:CAMERA_RTSP_URL",
    )

    camera = _camera(config)
    result = run_camera_configuration(camera).as_dict(camera)

    assert result["status"] == "valid"
    assert result["details"] == {
        "network_test": False,
        "transport": "tcp",
        "stream": "main",
    }
    assert "secret-value" not in json.dumps(result)
    assert "operator" not in json.dumps(result)


def test_configuration_test_reports_missing_environment_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_CAMERA_URL", raising=False)
    config = prepare_camera_config(
        name="Front gate",
        rtsp_url="rtsp://example.test:554/live",
        credential_ref="env:MISSING_CAMERA_URL",
    )

    result = run_camera_configuration(_camera(config))

    assert result.status == "invalid"
    assert "not available" in result.message
    assert result.credential["status"] == "missing"
