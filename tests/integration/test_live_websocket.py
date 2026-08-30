from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from model_helpers import create_model_fixture
from open_licenseplate.app import create_app
from open_licenseplate.capture import FixtureAttempt, ReconnectFixture, make_preview_frame
from open_licenseplate.config import load_settings
from open_licenseplate.database import upgrade_database
from open_licenseplate.inference.backends import FakeBackend
from open_licenseplate.live import LIVE_PROTOCOL_VERSION


def _settings(tmp_path: Path) -> Any:
    return load_settings(
        cli_overrides={
            "storage.data_dir": tmp_path / "data",
            "storage.log_dir": tmp_path / "logs",
        }
    )


def _outputs(_prepared: Any) -> dict[str, Any]:
    return {
        "coordinates": np.array([[0, 0, 640, 640]], dtype=np.float32),
        "confidence": np.array([0.8], dtype=np.float32),
    }


def _source_fixture() -> ReconnectFixture:
    return ReconnectFixture(
        (
            FixtureAttempt(
                frames=(make_preview_frame(40), make_preview_frame(80)),
                repeat=True,
                read_interval_seconds=0.003,
            ),
        )
    )


def _create_camera(client: TestClient) -> str:
    response = client.post(
        "/api/v1/cameras",
        json={"name": "Fixture", "rtsp_url": "rtsp://fixture.local/live"},
    )
    assert response.status_code == 201
    return str(response.json()["id"])


def _import_and_validate(client: TestClient, root: Path) -> str:
    manifest_path, archive_path, _manifest = create_model_fixture(root, model_id="ws-model")
    imported = client.post(
        "/api/v1/models/import",
        files={
            "manifest": ("manifest.json", manifest_path.read_bytes(), "application/json"),
            "archive": ("model.zip", archive_path.read_bytes(), "application/zip"),
        },
    )
    assert imported.status_code == 201, imported.text
    model_id = str(imported.json()["id"])
    validated = client.post(f"/api/v1/models/{model_id}/validate")
    assert validated.status_code == 200, validated.text
    return model_id


def _wait_for_processed(client: TestClient) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = client.get("/api/v1/live/state").json()
        if state["metrics"]["processed_frames"] > 0:
            return
        time.sleep(0.005)
    raise AssertionError("live pipeline did not process a frame")


@pytest.mark.m3_acceptance
def test_live_websocket_sends_one_header_then_matching_jpeg(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    source = _source_fixture()
    backend = FakeBackend(output_factory=_outputs)

    with TestClient(
        create_app(
            settings,
            source_factory=source,
            inference_backend_factory=lambda: backend,
        )
    ) as client:
        camera_id = _create_camera(client)
        model_id = _import_and_validate(client, tmp_path / "model")
        started = client.post(
            "/api/v1/live/start",
            json={"camera_id": camera_id, "model_id": model_id},
        )
        assert started.status_code == 200
        _wait_for_processed(client)

        with client.websocket_connect("/api/v1/live/ws") as websocket:
            header = websocket.receive_json()
            jpeg = websocket.receive_bytes()

        assert header["protocol_version"] == LIVE_PROTOCOL_VERSION
        assert header["type"] == "frame_header"
        assert header["camera_id"] == camera_id
        assert header["model_id"] == model_id
        assert header["capture_session_id"]
        assert header["stream_epoch"]
        assert header["frame_sequence"] >= 1
        assert header["source_width"] == 8
        assert header["source_height"] == 6
        assert header["jpeg_width"] == 8
        assert header["jpeg_height"] == 6
        assert header["jpeg_byte_count"] == len(jpeg)
        assert header["detections"][0]["frame_sequence"] == header["frame_sequence"]
        assert "artifact_path" not in header
        assert "/managed/" not in str(header)
        client.post("/api/v1/live/stop")


def test_live_websocket_closes_with_safe_stopped_state(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    source = _source_fixture()
    backend = FakeBackend(output_factory=_outputs)

    with TestClient(
        create_app(
            settings,
            source_factory=source,
            inference_backend_factory=lambda: backend,
        )
    ) as client:
        camera_id = _create_camera(client)
        model_id = _import_and_validate(client, tmp_path / "model")
        assert (
            client.post(
                "/api/v1/live/start",
                json={"camera_id": camera_id, "model_id": model_id},
            ).status_code
            == 200
        )
        _wait_for_processed(client)

        with client.websocket_connect("/api/v1/live/ws") as websocket:
            websocket.receive_json()
            websocket.receive_bytes()
            stopped = client.post("/api/v1/live/stop")
            assert stopped.status_code == 200
            state = websocket.receive_json()
            assert state == {
                "type": "state",
                "protocol_version": LIVE_PROTOCOL_VERSION,
                "state": "stopped",
            }


def test_live_websocket_disconnect_releases_subscriber(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    source = _source_fixture()
    backend = FakeBackend(output_factory=_outputs)
    application = create_app(
        settings,
        source_factory=source,
        inference_backend_factory=lambda: backend,
    )

    with TestClient(application) as client:
        camera_id = _create_camera(client)
        model_id = _import_and_validate(client, tmp_path / "model")
        assert (
            client.post(
                "/api/v1/live/start",
                json={"camera_id": camera_id, "model_id": model_id},
            ).status_code
            == 200
        )
        _wait_for_processed(client)
        with client.websocket_connect("/api/v1/live/ws") as websocket:
            websocket.receive_json()
            websocket.receive_bytes()
        assert application.state.live_pipeline._display.metrics().subscriber_count == 0
        client.post("/api/v1/live/stop")


def test_live_app_shutdown_closes_processed_display_worker(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    source = _source_fixture()
    backend = FakeBackend(output_factory=_outputs)
    application = create_app(
        settings,
        source_factory=source,
        inference_backend_factory=lambda: backend,
    )

    with TestClient(application) as client:
        camera_id = _create_camera(client)
        model_id = _import_and_validate(client, tmp_path / "model")
        assert (
            client.post(
                "/api/v1/live/start",
                json={"camera_id": camera_id, "model_id": model_id},
            ).status_code
            == 200
        )
        _wait_for_processed(client)
    metrics = application.state.live_pipeline._display.metrics()
    assert metrics.closed is True
    assert metrics.subscriber_count == 0


def test_live_websocket_gets_safe_shutdown_state_during_app_close(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    source = _source_fixture()
    backend = FakeBackend(output_factory=_outputs)
    application = create_app(
        settings,
        source_factory=source,
        inference_backend_factory=lambda: backend,
    )

    with TestClient(application) as client:
        camera_id = _create_camera(client)
        model_id = _import_and_validate(client, tmp_path / "model")
        assert (
            client.post(
                "/api/v1/live/start",
                json={"camera_id": camera_id, "model_id": model_id},
            ).status_code
            == 200
        )
        _wait_for_processed(client)

        with client.websocket_connect("/api/v1/live/ws") as websocket:
            websocket.receive_json()
            websocket.receive_bytes()
            application.state.live_pipeline.close()
            state = websocket.receive_json()
            assert state == {
                "type": "state",
                "protocol_version": LIVE_PROTOCOL_VERSION,
                "state": "shutdown",
            }
            assert application.state.live_pipeline._display.metrics().subscriber_count == 0
            encoder_thread = application.state.live_pipeline._display.encoder_thread
            assert encoder_thread is None or not encoder_thread.is_alive()
