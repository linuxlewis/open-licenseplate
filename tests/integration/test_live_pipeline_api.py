from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from model_helpers import create_model_fixture
from open_licenseplate.app import create_app
from open_licenseplate.capture import FixtureAttempt, ReconnectFixture, make_preview_frame
from open_licenseplate.config import load_settings
from open_licenseplate.database import upgrade_database
from open_licenseplate.inference.backends import FakeBackend


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


def _create_camera(client: TestClient, name: str) -> str:
    response = client.post(
        "/api/v1/cameras",
        json={"name": name, "rtsp_url": "rtsp://fixture.local/live"},
    )
    assert response.status_code == 201
    return str(response.json()["id"])


def _import_and_validate(client: TestClient, root: Path, model_id: str) -> str:
    fixture_root = root / model_id
    fixture_root.mkdir()
    manifest_path, archive_path, _manifest = create_model_fixture(
        fixture_root,
        model_id=model_id,
    )
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
    assert validated.json()["runtime_valid"] is True
    return model_id


def _wait_for_live_state(
    client: TestClient,
    state: str,
    *,
    timeout: float = 2.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get("/api/v1/live/state")
        assert response.status_code == 200
        latest = response.json()
        if latest.get("state") == state:
            return latest
        time.sleep(0.005)
    raise AssertionError(f"live pipeline did not reach {state}: {latest}")


def _wait_for_camera_frames(
    client: TestClient,
    camera_id: str,
    *,
    minimum: int,
    timeout: float = 2.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/cameras/{camera_id}/status")
        assert response.status_code == 200
        latest = response.json()
        if (
            latest.get("state") == "streaming"
            and latest.get("metrics", {}).get("captured_frames", 0) >= minimum
        ):
            return latest
        time.sleep(0.005)
    raise AssertionError(f"camera did not capture enough frames: {latest}")


def test_live_api_starts_warms_processes_updates_threshold_and_stops(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    fixture = _source_fixture()
    backend = FakeBackend(output_factory=_outputs)

    with TestClient(
        create_app(
            settings,
            source_factory=fixture,
            inference_backend_factory=lambda: backend,
        )
    ) as client:
        camera_id = _create_camera(client, "Fixture")
        model_id = _import_and_validate(client, tmp_path, "live-model")

        started = client.post(
            "/api/v1/live/start",
            json={
                "camera_id": camera_id,
                "model_id": model_id,
                "confidence_threshold": 0.35,
            },
        )
        assert started.status_code == 200
        assert started.json()["state"] == "starting"

        running = _wait_for_live_state(client, "running")
        assert running["metrics"]["warmup_ms"] is not None
        processed = _wait_for_live_state(client, "running")
        deadline = time.monotonic() + 2
        while processed["metrics"]["processed_frames"] == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
            processed = client.get("/api/v1/live/state").json()
        assert processed["metrics"]["processed_frames"] > 0
        assert processed["last_result"]["camera_id"] == camera_id
        assert processed["last_result"]["model_id"] == model_id
        assert processed["last_result"]["capture_session_id"]
        assert processed["last_result"]["frame_sequence"] >= 1

        updated = client.patch(
            "/api/v1/live/settings",
            json={"confidence_threshold": 0.95},
        )
        assert updated.status_code == 200
        assert updated.json()["confidence_threshold"] == 0.95

        stopped = client.post("/api/v1/live/stop")
        assert stopped.status_code == 200
        assert stopped.json()["state"] == "stopped"
        assert fixture.sources and fixture.sources[0].closed.is_set()
        assert backend.closes and all(model.closed for model in backend.closes)


def test_live_generation_metrics_exclude_existing_preview_frames(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    fixture = _source_fixture()
    backend = FakeBackend(output_factory=_outputs)

    with TestClient(
        create_app(
            settings,
            source_factory=fixture,
            inference_backend_factory=lambda: backend,
        )
    ) as client:
        camera_id = _create_camera(client, "Fixture")
        model_id = _import_and_validate(client, tmp_path, "baseline-model")
        assert client.post(f"/api/v1/cameras/{camera_id}/start").status_code == 200
        before = _wait_for_camera_frames(client, camera_id, minimum=20)
        before_metrics = before["metrics"]

        started = client.post(
            "/api/v1/live/start",
            json={"camera_id": camera_id, "model_id": model_id},
        )
        assert started.status_code == 200
        _wait_for_live_state(client, "running")

        deadline = time.monotonic() + 2
        live_state = client.get("/api/v1/live/state").json()
        while live_state["metrics"]["processed_frames"] == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
            live_state = client.get("/api/v1/live/state").json()
        after = client.get(f"/api/v1/cameras/{camera_id}/status").json()

        live_captured = live_state["metrics"]["captured_frames"]
        live_replaced = live_state["metrics"]["source_replacement_count"]
        assert live_captured > 0
        assert live_captured < after["metrics"]["captured_frames"]
        assert live_captured <= (
            after["metrics"]["captured_frames"] - before_metrics["captured_frames"]
        )
        assert live_replaced <= (
            after["metrics"]["replaced_frames"] - before_metrics["replaced_frames"]
        )
        client.post("/api/v1/live/stop")


def test_live_api_rejects_camera_and_model_switch_while_running(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    fixture = _source_fixture()
    backend = FakeBackend(output_factory=_outputs)

    with TestClient(
        create_app(
            settings,
            source_factory=fixture,
            inference_backend_factory=lambda: backend,
        )
    ) as client:
        first_camera = _create_camera(client, "First")
        second_camera = _create_camera(client, "Second")
        first_model = _import_and_validate(client, tmp_path, "first-model")
        second_model = _import_and_validate(client, tmp_path, "second-model")
        activated = client.post(f"/api/v1/models/{first_model}/activate")
        assert activated.status_code == 200

        assert (
            client.post(
                "/api/v1/live/start",
                json={"camera_id": first_camera, "model_id": first_model},
            ).status_code
            == 200
        )
        _wait_for_live_state(client, "running")

        camera_switch = client.post(
            "/api/v1/live/start",
            json={"camera_id": second_camera, "model_id": first_model},
        )
        model_switch = client.post(
            "/api/v1/live/start",
            json={"camera_id": first_camera, "model_id": second_model},
        )
        assert camera_switch.status_code == 409
        assert "switching the camera" in camera_switch.json()["detail"]
        assert model_switch.status_code == 409
        assert "switching the model" in model_switch.json()["detail"]
        active_model_switch = client.post(f"/api/v1/models/{second_model}/activate")
        active_camera_stop = client.post(f"/api/v1/cameras/{first_camera}/stop")
        assert active_model_switch.status_code == 409
        assert active_camera_stop.status_code == 409
        client.post("/api/v1/live/stop")


def test_live_api_recovers_after_safe_inference_failure(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    fixture = _source_fixture()

    def failing_outputs(_prepared: Any) -> dict[str, Any]:
        raise RuntimeError("failed at /private/model/path")

    backends = [
        FakeBackend(output_factory=_outputs),
        FakeBackend(output_factory=failing_outputs),
        FakeBackend(output_factory=_outputs),
    ]

    class BackendFactory:
        def __init__(self, values: list[FakeBackend]) -> None:
            self.values = values
            self.index = 0

        def __call__(self) -> FakeBackend:
            value = self.values[min(self.index, len(self.values) - 1)]
            self.index += 1
            return value

    factory: Callable[[], FakeBackend] = BackendFactory(backends)
    with TestClient(
        create_app(
            settings,
            source_factory=fixture,
            inference_backend_factory=factory,
        )
    ) as client:
        camera_id = _create_camera(client, "Fixture")
        model_id = _import_and_validate(client, tmp_path, "recover-model")

        assert (
            client.post(
                "/api/v1/live/start",
                json={"camera_id": camera_id, "model_id": model_id},
            ).status_code
            == 200
        )
        failed = _wait_for_live_state(client, "failed")
        assert failed["failure"]["category"] == "inference"
        assert "/private/model/path" not in str(failed)
        assert "live inference failed" in failed["failure"]["message"]

        restarted = client.post(
            "/api/v1/live/start",
            json={"camera_id": camera_id, "model_id": model_id},
        )
        assert restarted.status_code == 200
        running = _wait_for_live_state(client, "running")
        assert running["failure"] is None
        client.post("/api/v1/live/stop")
