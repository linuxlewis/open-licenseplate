from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from open_licenseplate.app import create_app
from open_licenseplate.capture import (
    FixtureAttempt,
    ReconnectFixture,
    disconnect_then_recover_fixture,
    make_preview_frame,
    preview_chunks,
)
from open_licenseplate.config import load_settings
from open_licenseplate.database import upgrade_database


def _settings(tmp_path: Path):
    return load_settings(
        cli_overrides={
            "storage.data_dir": tmp_path / "data",
            "storage.log_dir": tmp_path / "logs",
        }
    )


def _wait_for_streaming(client: TestClient, camera_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/cameras/{camera_id}/status")
        payload = response.json()
        if payload["state"] == "streaming":
            return payload
        time.sleep(0.01)
    raise AssertionError(f"camera did not stream: {payload}")


def _create_camera(client: TestClient, name: str) -> str:
    response = client.post(
        "/api/v1/cameras",
        json={"name": name, "rtsp_url": "rtsp://fixture.local/live"},
    )
    assert response.status_code == 201
    return str(response.json()["id"])


def test_camera_test_and_preview_endpoints_report_safe_stream_data(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    fixture = ReconnectFixture(
        (
            FixtureAttempt(
                frames=(make_preview_frame(20),),
                repeat=True,
                read_interval_seconds=0.01,
            ),
        )
    )

    with TestClient(create_app(settings, source_factory=fixture)) as client:
        camera_id = _create_camera(client, "Fixture")
        not_running_preview = client.get(f"/api/v1/cameras/{camera_id}/preview.mjpeg")
        assert not_running_preview.status_code == 409
        test_response = client.post(f"/api/v1/cameras/{camera_id}/test")
        assert test_response.status_code == 200
        test_payload = test_response.json()
        assert test_payload["status"] == "valid"
        assert test_payload["details"]["network_test"] is True
        assert test_payload["details"]["resolution"] == "8x6"
        assert test_payload["details"]["nominal_fps"] == 30.0
        assert test_payload["details"]["transport"] == "tcp"
        assert test_payload["details"]["camera_pts_available"] is False

        start_response = client.post(f"/api/v1/cameras/{camera_id}/start")
        assert start_response.status_code == 200
        status = _wait_for_streaming(client, camera_id)
        assert status["source"]["resolution"] == "8x6"
        assert status["source"]["camera_pts_available"] is False

        snapshot = client.get(f"/api/v1/cameras/{camera_id}/snapshot.jpg")
        deadline = time.monotonic() + 1
        while snapshot.status_code == 409 and time.monotonic() < deadline:
            time.sleep(0.01)
            snapshot = client.get(f"/api/v1/cameras/{camera_id}/snapshot.jpg")
        assert snapshot.status_code == 200
        assert snapshot.headers["content-type"] == "image/jpeg"
        assert snapshot.content.startswith(b"\xff\xd8")

        frame = client.app.state.camera_runtime.latest_frame(camera_id)
        assert frame is not None
        first_chunk = next(preview_chunks(iter((frame,))))
        assert b"Content-Type: image/jpeg" in first_chunk
        assert b"\xff\xd8" in first_chunk

        stop_response = client.post(f"/api/v1/cameras/{camera_id}/stop")
        assert stop_response.status_code == 200
        assert stop_response.json()["state"] == "stopped"


def test_starting_a_second_camera_returns_conflict_and_reconnect_recovers(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    fixture = disconnect_then_recover_fixture()

    with TestClient(create_app(settings, source_factory=fixture)) as client:
        first_id = _create_camera(client, "First")
        second_id = _create_camera(client, "Second")
        client.post(f"/api/v1/cameras/{first_id}/start")

        _wait_for_streaming(client, first_id)
        conflict = client.post(f"/api/v1/cameras/{second_id}/start")
        assert conflict.status_code == 409
        assert "stop it before starting" in conflict.json()["detail"]
        assert second_id in conflict.json()["detail"]

        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            status = client.get(f"/api/v1/cameras/{first_id}/status").json()
            if fixture.created_attempts >= 2 and status["state"] == "streaming":
                break
            time.sleep(0.01)
        assert fixture.created_attempts >= 2
        assert status["state"] == "streaming"
        assert status["metrics"]["reconnect_count"] >= 1

        client.post(f"/api/v1/cameras/{first_id}/stop")


def test_initial_source_open_error_reports_failed_without_reconnect(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    fixture = ReconnectFixture((FixtureAttempt(open_error="fixture source is unavailable"),))
    application = create_app(settings, source_factory=fixture)

    with TestClient(application) as client:
        camera_id = _create_camera(client, "Unavailable")
        started = client.post(f"/api/v1/cameras/{camera_id}/start")
        assert started.status_code == 200
        failed = application.state.camera_runtime.wait_for_state("failed")

        status = client.get(f"/api/v1/cameras/{camera_id}/status")
        assert status.status_code == 200
        assert status.json()["state"] == "failed"
        assert status.json()["reconnect_attempt"] == 0
        assert "fixture source is unavailable" in status.json()["last_error"]
        assert fixture.created_attempts == 1
        assert "active_camera_id" not in status.json()
        assert failed.state == "failed"
        live_page = client.get("/live")
        assert "Failed" in live_page.text
        assert "Fix the source settings, then press Start preview." in live_page.text
