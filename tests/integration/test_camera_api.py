from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from open_licenseplate.app import create_app
from open_licenseplate.config import load_settings
from open_licenseplate.database import upgrade_database


def _settings(tmp_path: Path):
    return load_settings(
        cli_overrides={
            "storage.data_dir": tmp_path / "data",
            "storage.log_dir": tmp_path / "logs",
        }
    )


def test_camera_api_crud_and_test_operation_redact_secrets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    database_path = settings.storage.data_dir / "open-licenseplate.sqlite3"
    upgrade_database(database_path)
    monkeypatch.setenv(
        "CAMERA_RTSP_URL",
        "rtsp://operator:secret-value@example.test:554/live?token=query-value",
    )

    with TestClient(create_app(settings)) as client:
        create = client.post(
            "/api/v1/cameras",
            json={
                "name": "Front gate",
                "rtsp_url": "rtsp://operator:secret-value@example.test:554/live",
                "credential_ref": "env:CAMERA_RTSP_URL",
                "transport": "tcp",
                "preferred_stream": "main",
                "enabled": True,
            },
        )
        assert create.status_code == 201
        camera = create.json()
        camera_id = camera["id"]

        for response in (
            create,
            client.get("/api/v1/cameras"),
            client.get(f"/api/v1/cameras/{camera_id}"),
            client.post(f"/api/v1/cameras/{camera_id}/test"),
        ):
            assert "secret-value" not in response.text
            assert "query-value" not in response.text
            assert "operator" not in response.text

        assert camera["endpoint"] == "rtsp://[REDACTED]@example.test:554/live"
        assert camera["credential"]["status"] == "available"
        test_result = client.post(f"/api/v1/cameras/{camera_id}/test")
        assert test_result.status_code == 200
        assert test_result.json()["status"] == "valid"

        update = client.patch(
            f"/api/v1/cameras/{camera_id}",
            json={"name": "Side gate", "transport": "udp", "enabled": False},
        )
        assert update.status_code == 200
        assert update.json()["name"] == "Side gate"
        assert update.json()["connection_options"]["transport"] == "udp"
        assert update.json()["enabled"] is False

        delete = client.delete(f"/api/v1/cameras/{camera_id}")
        assert delete.status_code == 200
        assert client.get(f"/api/v1/cameras/{camera_id}").status_code == 404

    with sqlite3.connect(database_path) as connection:
        stored_values = connection.execute(
            "SELECT name, endpoint, credential_ref, connection_options_json FROM cameras"
        ).fetchall()
    assert stored_values == []


def test_camera_api_rejects_password_body_without_persisting_it(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database_path = settings.storage.data_dir / "open-licenseplate.sqlite3"
    upgrade_database(database_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/cameras",
            json={
                "name": "Front gate",
                "rtsp_url": "rtsp://example.test/live",
                "password": "secret-value",
            },
        )

    assert response.status_code == 422
    assert "secret-value" not in response.text
    assert "credential_ref" in response.text
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM cameras").fetchone() == (0,)


def test_camera_api_rejects_secret_nested_in_connection_options(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database_path = settings.storage.data_dir / "open-licenseplate.sqlite3"
    upgrade_database(database_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/cameras",
            json={
                "name": "Front gate",
                "rtsp_url": "rtsp://example.test/live",
                "connection_options": {"headers": {"authorization": "secret-value"}},
            },
        )

    assert response.status_code == 422
    assert "secret-value" not in response.text


def test_camera_page_has_safe_configuration_form_and_saved_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    database_path = settings.storage.data_dir / "open-licenseplate.sqlite3"
    upgrade_database(database_path)
    monkeypatch.setenv("CAMERA_RTSP_URL", "rtsp://operator:secret-value@example.test/live")

    with TestClient(create_app(settings)) as client:
        page = client.get("/cameras")
        assert page.status_code == 200
        assert "Add camera" in page.text
        assert "secret-value" not in page.text

        saved = client.post(
            "/cameras",
            data={
                "name": "Front gate",
                "rtsp_url": "rtsp://operator:secret-value@example.test/live",
                "credential_ref": "env:CAMERA_RTSP_URL",
                "transport": "tcp",
                "preferred_stream": "main",
            },
            follow_redirects=True,
        )
        assert saved.status_code == 200
        assert "Front gate" in saved.text
        assert "secret-value" not in saved.text
        assert "operator" not in saved.text
        assert "env:CAMERA_RTSP_URL" in saved.text
