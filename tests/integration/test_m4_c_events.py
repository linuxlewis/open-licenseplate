from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from open_licenseplate.app import create_app
from open_licenseplate.config import load_settings
from open_licenseplate.database import Database, upgrade_database
from open_licenseplate.events import EventRepository, ManagedArtifactService
from open_licenseplate.inference import Detection
from open_licenseplate.paths import ManagedPaths
from open_licenseplate.tracking import (
    ClosedTrackEvent,
    TrackingProvenance,
    capture_crop_candidate,
)

pytestmark = [pytest.mark.integration, pytest.mark.m4_c_acceptance]

BASE_TIME = datetime(2026, 8, 30, tzinfo=UTC)


def _settings(tmp_path: Path) -> Any:
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


def _candidate(sequence: int, timestamp: datetime, value: int):
    pixels = np.full((20, 60, 3), value, dtype=np.uint8)
    detection = Detection(
        box_xyxy=(10.0, 10.0, 70.0, 30.0),
        class_id=0,
        label="license_plate",
        confidence=0.8,
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


def _event(
    event_id: str,
    *,
    offset_seconds: float = 0,
    session_id: str | None = None,
    with_crop: bool = True,
) -> ClosedTrackEvent:
    first_seen = BASE_TIME + timedelta(seconds=offset_seconds)
    candidate = _candidate(1, first_seen, 80) if with_crop else None
    return ClosedTrackEvent(
        event_id=event_id,
        provenance=TrackingProvenance(
            camera_id="camera-1",
            capture_session_id=session_id or f"session-{event_id}",
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
        crop_candidates=() if candidate is None else (candidate,),
    )


def _app(tmp_path: Path):
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    database = Database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    try:
        _seed_parents(database)
    finally:
        database.dispose()
    return settings, create_app(settings)


def _commit(settings: Any, event: ClosedTrackEvent) -> str:
    stored = ManagedArtifactService(
        ManagedPaths.from_settings(settings),
        application_version="test",
    ).commit_closed_event(event)
    return stored.id


def test_event_api_returns_empty_bounded_list(tmp_path: Path) -> None:
    _settings_value, app = _app(tmp_path)

    with TestClient(app) as client:
        response = client.get("/api/v1/events")

    assert response.status_code == 200
    assert response.json() == {"events": [], "limit": 100}


def test_event_api_orders_newest_first_and_honors_limit(tmp_path: Path) -> None:
    settings, app = _app(tmp_path)
    for index in range(3):
        _commit(settings, _event(f"event-{index}", offset_seconds=float(index)))

    with TestClient(app) as client:
        response = client.get("/api/v1/events?limit=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["limit"] == 2
    assert [item["event_id"] for item in payload["events"]] == ["event-2", "event-1"]
    assert all("managed_relative_path" not in item for item in payload["events"])


def test_event_detail_and_safe_artifact_response_include_provenance(tmp_path: Path) -> None:
    settings, app = _app(tmp_path)
    event_id = _commit(settings, _event("event-detail"))
    paths = ManagedPaths.from_settings(settings)
    database = Database(paths.database)
    try:
        artifact = EventRepository(database).artifacts_for_event(event_id)[0]
    finally:
        database.dispose()

    with TestClient(app) as client:
        detail = client.get(f"/api/v1/events/{event_id}")
        artifact_response = client.get(f"/api/v1/events/{event_id}/artifacts/{artifact.id}")

    assert detail.status_code == 200
    payload = detail.json()
    assert payload["camera_id"] == "camera-1"
    assert payload["camera_display_name"] == "Front gate"
    assert payload["model"]["model_id"] == "model-1"
    assert payload["model"]["model_checksum"] == "a" * 64
    assert payload["capture_session_id"] == "session-event-detail"
    assert payload["track_id"] == 7
    assert payload["event_state"] == "closed"
    assert payload["ocr"]["state"] == "not_available"
    assert payload["ocr_state"] == "not_available"
    assert [item["rank"] for item in payload["artifacts"]] == [0]
    assert "managed_relative_path" not in detail.text
    assert str(paths.artifacts) not in detail.text

    assert artifact_response.status_code == 200
    assert artifact_response.headers["content-type"] == "image/jpeg"
    assert artifact_response.headers["cache-control"] == "private, no-store"
    assert artifact_response.content.startswith(b"\xff\xd8")
    assert hashlib.sha256(artifact_response.content).hexdigest() == artifact.sha256


def test_event_api_returns_stable_not_found_for_unknown_and_mismatched_resources(
    tmp_path: Path,
) -> None:
    settings, app = _app(tmp_path)
    event_id = _commit(settings, _event("event-known"))
    other_event_id = _commit(settings, _event("event-other"))
    paths = ManagedPaths.from_settings(settings)
    database = Database(paths.database)
    try:
        repository = EventRepository(database)
        artifact = repository.artifacts_for_event(event_id)[0]
        other_artifact = repository.artifacts_for_event(other_event_id)[0]
    finally:
        database.dispose()

    with TestClient(app) as client:
        responses = [
            client.get("/api/v1/events/unknown"),
            client.get(f"/api/v1/events/{event_id}/artifacts/unknown"),
            client.get(f"/api/v1/events/{event_id}/artifacts/{other_artifact.id}"),
        ]

    assert [response.status_code for response in responses] == [404, 404, 404]
    assert {response.json()["detail"] for response in responses} == {
        "event was not found",
        "event artifact was not found",
    }
    assert artifact.id != other_artifact.id


def test_event_artifact_missing_and_checksum_mismatch_are_unavailable(tmp_path: Path) -> None:
    settings, app = _app(tmp_path)
    event_id = _commit(settings, _event("event-files"))
    paths = ManagedPaths.from_settings(settings)
    database = Database(paths.database)
    try:
        artifact = EventRepository(database).artifacts_for_event(event_id)[0]
    finally:
        database.dispose()
    artifact_path = paths.artifacts / artifact.managed_relative_path

    artifact_path.unlink()
    with TestClient(app) as client:
        missing = client.get(f"/api/v1/events/{event_id}/artifacts/{artifact.id}")
    assert missing.status_code == 404

    _commit(settings, _event("event-corrupt"))
    database = Database(paths.database)
    try:
        corrupt = EventRepository(database).artifacts_for_event("event-corrupt")[0]
    finally:
        database.dispose()
    corrupt_path = paths.artifacts / corrupt.managed_relative_path
    corrupt_path.write_bytes(b"not a jpeg")
    with TestClient(app) as client:
        response = client.get(f"/api/v1/events/event-corrupt/artifacts/{corrupt.id}")
    assert response.status_code == 404


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("byte_size", 1),
        ("width", 1),
        ("mime_type", "image/png"),
    ],
)
def test_event_artifact_metadata_mismatch_is_unavailable(
    tmp_path: Path,
    column: str,
    value: object,
) -> None:
    settings, app = _app(tmp_path)
    event_id = _commit(settings, _event("event-metadata"))
    paths = ManagedPaths.from_settings(settings)
    database = Database(paths.database)
    try:
        artifact = EventRepository(database).artifacts_for_event(event_id)[0]
        with database.session() as session:
            session.execute(
                text(f"UPDATE event_artifacts SET {column} = :value WHERE id = :artifact_id"),
                {"value": value, "artifact_id": artifact.id},
            )
    finally:
        database.dispose()

    with TestClient(app) as client:
        response = client.get(f"/api/v1/events/{event_id}/artifacts/{artifact.id}")

    assert response.status_code == 404
    assert str(paths.artifacts) not in response.text


@pytest.mark.parametrize(
    "relative_path",
    [
        "../outside.jpg",
        "/tmp/outside.jpg",
        "events/nested/crop.jpg",
        "events/../outside.jpg",
    ],
)
def test_event_artifact_path_validation_rejects_unsafe_rows(
    tmp_path: Path,
    relative_path: str,
) -> None:
    settings, app = _app(tmp_path)
    event_id = _commit(settings, _event("event-path"))
    paths = ManagedPaths.from_settings(settings)
    database = Database(paths.database)
    try:
        artifact = EventRepository(database).artifacts_for_event(event_id)[0]
        with database.session() as session:
            session.execute(
                text(
                    "UPDATE event_artifacts SET managed_relative_path = :relative_path "
                    "WHERE id = :artifact_id"
                ),
                {"relative_path": relative_path, "artifact_id": artifact.id},
            )
    finally:
        database.dispose()

    with TestClient(app) as client:
        response = client.get(f"/api/v1/events/{event_id}/artifacts/{artifact.id}")

    assert response.status_code == 404
    assert str(paths.artifacts) not in response.text


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported")
def test_event_artifact_path_validation_rejects_file_and_parent_symlinks(
    tmp_path: Path,
) -> None:
    settings, app = _app(tmp_path)
    event_id = _commit(settings, _event("event-symlink"))
    paths = ManagedPaths.from_settings(settings)
    database = Database(paths.database)
    try:
        artifact = EventRepository(database).artifacts_for_event(event_id)[0]
    finally:
        database.dispose()
    artifact_path = paths.artifacts / artifact.managed_relative_path
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    artifact_path.unlink()
    artifact_path.symlink_to(outside)

    with TestClient(app) as client:
        file_link = client.get(f"/api/v1/events/{event_id}/artifacts/{artifact.id}")
    assert file_link.status_code == 404

    artifact_path.unlink()
    events_root = paths.artifacts / "events"
    events_root.rmdir()
    events_root.symlink_to(tmp_path, target_is_directory=True)
    with TestClient(app) as client:
        parent_link = client.get(f"/api/v1/events/{event_id}/artifacts/{artifact.id}")
    assert parent_link.status_code == 404
