from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient
from sqlalchemy import text

from model_helpers import create_model_fixture
from open_licenseplate.app import create_app
from open_licenseplate.capture import FixtureAttempt, ReconnectFixture, make_preview_frame
from open_licenseplate.config import load_settings
from open_licenseplate.database import Database, upgrade_database
from open_licenseplate.events import ManagedArtifactService
from open_licenseplate.inference import Detection
from open_licenseplate.inference.backends import FakeBackend
from open_licenseplate.paths import ManagedPaths
from open_licenseplate.tracking import (
    ClosedTrackEvent,
    TrackedDetection,
    TrackingConfig,
    TrackingProvenance,
    capture_crop_candidate,
)

pytestmark = pytest.mark.m4_c_acceptance


class _OneTrack:
    def update(
        self,
        detections: Sequence[Detection],
        *,
        frame_width: int,
        frame_height: int,
    ) -> tuple[TrackedDetection, ...]:
        del frame_width, frame_height
        return tuple(TrackedDetection(7, detection) for detection in detections)

    def reset(self) -> None:
        return None


def _settings(tmp_path: Path):
    return load_settings(
        cli_overrides={
            "storage.data_dir": tmp_path / "data",
            "storage.log_dir": tmp_path / "logs",
        }
    )


def _seed_parents(database: Database) -> None:
    now = "2026-08-30T00:00:00Z"
    with database.session() as session:
        session.execute(
            text(
                "INSERT INTO cameras "
                "(id, name, endpoint, connection_options_json, preferred_stream, enabled, "
                "created_at, updated_at) VALUES "
                "('camera-1', 'Front gate', 'rtsp://fixture.local/live', '{}', 'main', 1, "
                ":now, :now)"
            ),
            {"now": now},
        )
        session.execute(
            text(
                "INSERT INTO models "
                "(id, display_name, backend, adapter, artifact_path, artifact_sha256, "
                "manifest_json, validation_state, validation_details_json, active, created_at) "
                "VALUES ('model-1', 'Fixture detector', 'coreml', 'adapter', 'model.mlpackage', "
                ":checksum, '{}', 'runtime_valid', '{}', 1, :now)"
            ),
            {"checksum": "a" * 64, "now": now},
        )


def _review_candidate(sequence: int, value: int):
    timestamp = datetime(2026, 8, 30, tzinfo=UTC) + timedelta(seconds=sequence / 10)
    pixels = np.full((20, 60, 3), value, dtype=np.uint8)
    detection = Detection(
        box_xyxy=(10.0, 10.0, 70.0, 30.0),
        class_id=0,
        label="license_plate",
        confidence=0.8 + sequence / 100,
        model_id="model-1",
        model_sha256="a" * 64,
        frame_sequence=sequence,
        detected_at=timestamp,
    )
    return capture_crop_candidate(
        source_pixels=pixels,
        pixel_format="bgr24",
        frame_width=100,
        frame_height=60,
        frame_sequence=sequence,
        source_timestamp=timestamp,
        detection=detection,
    )


def _review_event(event_id: str, *, candidate_count: int = 2) -> ClosedTrackEvent:
    first_seen = datetime(2026, 8, 30, tzinfo=UTC)
    candidates = tuple(
        _review_candidate(sequence, 70 + sequence) for sequence in range(1, candidate_count + 1)
    )
    return ClosedTrackEvent(
        event_id=event_id,
        provenance=TrackingProvenance(
            camera_id="camera-1",
            capture_session_id=f"session-{event_id}",
            generation_number=1,
            stream_epoch="epoch-1",
            model_id="model-1",
            model_checksum="a" * 64,
        ),
        track_id=7,
        first_seen_at=first_seen,
        last_seen_at=first_seen + timedelta(seconds=0.2),
        duration_seconds=0.2,
        observation_count=3,
        maximum_confidence=0.9,
        crop_candidates=candidates,
    )


def _seed_review_event(settings: Any, event_id: str = "event-browser") -> str:
    paths = ManagedPaths.from_settings(settings)
    database = Database(paths.database)
    try:
        _seed_parents(database)
    finally:
        database.dispose()
    return (
        ManagedArtifactService(paths, application_version="test")
        .commit_closed_event(_review_event(event_id))
        .id
    )


def _wait_for_processed_frames(
    client: TestClient,
    *,
    minimum: int = 3,
    timeout: float = 3.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = client.get("/api/v1/live/state").json()
        if latest["state"] == "failed":
            raise AssertionError(f"live pipeline failed: {latest}")
        if latest["metrics"]["processed_frames"] >= minimum:
            return latest
        time.sleep(0.005)
    raise AssertionError(f"live pipeline did not process enough frames: {latest}")


def _live_app(tmp_path: Path, *, has_plate: bool):
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    source = ReconnectFixture(
        (
            FixtureAttempt(
                frames=(make_preview_frame(80, width=8, height=6),),
                repeat=True,
                read_interval_seconds=0.003,
            ),
        )
    )

    def outputs(_prepared: Any) -> dict[str, Any]:
        if not has_plate:
            return {
                "coordinates": np.empty((0, 4), dtype=np.float32),
                "confidence": np.empty((0,), dtype=np.float32),
            }
        return {
            "coordinates": np.array([[0, 0, 640, 640]], dtype=np.float32),
            "confidence": np.array([0.9], dtype=np.float32),
        }

    backend = FakeBackend(output_factory=outputs)
    return settings, create_app(
        settings,
        source_factory=source,
        inference_backend_factory=lambda: backend,
        tracker_factory=lambda: _OneTrack(),
        tracking_config=TrackingConfig(close_timeout_seconds=1.0),
    )


def _configure_live(client: TestClient, root: Path) -> str:
    camera = client.post(
        "/api/v1/cameras",
        json={"name": "Replay camera", "rtsp_url": "rtsp://fixture.local/live"},
    )
    assert camera.status_code == 201, camera.text
    manifest_path, archive_path, _manifest = create_model_fixture(root, model_id="replay-model")
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
    started = client.post(
        "/api/v1/live/start",
        json={"camera_id": camera.json()["id"], "model_id": model_id},
    )
    assert started.status_code == 200, started.text
    _wait_for_processed_frames(client)
    return model_id


def test_one_plate_replay_creates_exactly_one_event_and_a_crop(tmp_path: Path) -> None:
    settings, app = _live_app(tmp_path, has_plate=True)
    with TestClient(app) as client:
        _configure_live(client, tmp_path / "model")
        stopped = client.post("/api/v1/live/stop")
        assert stopped.status_code == 200
        events = client.get("/api/v1/events").json()["events"]

    assert len(events) == 1
    event_id = events[0]["event_id"]
    detail = TestClient(create_app(settings))
    try:
        response = detail.get(f"/api/v1/events/{event_id}")
    finally:
        detail.close()
    assert response.status_code == 200
    assert len(response.json()["artifacts"]) >= 1


def test_no_plate_replay_creates_zero_events(tmp_path: Path) -> None:
    settings, app = _live_app(tmp_path, has_plate=False)
    with TestClient(app) as client:
        _configure_live(client, tmp_path / "model")
        assert client.post("/api/v1/live/stop").status_code == 200
        response = client.get("/api/v1/events")

    assert response.status_code == 200
    assert response.json()["events"] == []
    del settings


def test_service_restart_preserves_event_and_every_crop_response(tmp_path: Path) -> None:
    settings, app = _live_app(tmp_path, has_plate=True)
    with TestClient(app) as client:
        _configure_live(client, tmp_path / "model")
        assert client.post("/api/v1/live/stop").status_code == 200
        event = client.get("/api/v1/events").json()["events"][0]
        before = client.get(f"/api/v1/events/{event['event_id']}").json()

    with TestClient(create_app(settings)) as restarted:
        after_response = restarted.get(f"/api/v1/events/{event['event_id']}")
        after = after_response.json()
        assert after_response.status_code == 200
        assert [crop["rank"] for crop in after["artifacts"]] == [
            crop["rank"] for crop in before["artifacts"]
        ]
        for crop in after["artifacts"]:
            artifact_response = restarted.get(crop["url"])
            assert artifact_response.status_code == 200
            assert artifact_response.headers["content-type"] == "image/jpeg"
            assert artifact_response.content.startswith(b"\xff\xd8")


def _free_port() -> int:
    with socket.socket() as socket_instance:
        socket_instance.bind(("127.0.0.1", 0))
        return int(socket_instance.getsockname()[1])


@pytest.fixture
def event_browser_base_url(tmp_path: Path):
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    event_id = _seed_review_event(settings)
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(settings),
            host="127.0.0.1",
            port=_free_port(),
            log_config=None,
            access_log=False,
        )
    )
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.config.port}"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url}/events", timeout=0.5).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("event browser test server did not start")

    yield base_url, event_id
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def chromium():
    from playwright.sync_api import sync_playwright

    candidates = [
        os.environ.get("OPEN_LICENSEPLATE_CHROMIUM"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
    ]
    executable = next(
        (candidate for candidate in candidates if candidate and Path(candidate).is_file()),
        None,
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            **({"executable_path": executable} if executable else {}),
        )
        yield browser
        browser.close()


@pytest.mark.browser
def test_browser_reviews_ranked_crops_provenance_and_no_ocr(
    event_browser_base_url,
    chromium,
) -> None:
    base_url, event_id = event_browser_base_url
    page = chromium.new_page(viewport={"width": 1280, "height": 1000})
    page.goto(f"{base_url}/events", wait_until="domcontentloaded")

    assert page.get_by_role("heading", name="Plate events", exact=True).is_visible()
    assert page.get_by_text("Front gate", exact=True).is_visible()
    assert page.get_by_text("Fixture detector", exact=False).count() >= 1
    page.get_by_role("link", name="Front gate", exact=True).click()
    page.wait_for_url(f"{base_url}/events/{event_id}")

    assert page.get_by_role("heading", name="Event review", exact=True).is_visible()
    assert page.get_by_role("heading", name="No OCR data", exact=True).is_visible()
    assert page.get_by_text("No plate text was inferred or stored.", exact=False).is_visible()
    assert page.get_by_text("m4b-crop-score-v1", exact=True).count() >= 2
    assert page.get_by_text("image/jpeg", exact=True).count() >= 2
    assert page.get_by_text("Rank 1", exact=True).is_visible()
    assert page.get_by_text("Rank 2", exact=True).is_visible()
    assert page.locator("[data-crop-rank]").evaluate_all(
        "(items) => items.map((item) => item.dataset.cropRank)"
    ) == ["0", "1"]
