from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from model_helpers import create_model_fixture
from open_licenseplate.app import create_app
from open_licenseplate.capture import FixtureAttempt, ReconnectFixture, make_preview_frame
from open_licenseplate.config import load_settings
from open_licenseplate.database import Database, upgrade_database
from open_licenseplate.events import (
    CROP_QUALITY_SCORING_VERSION,
    ArtifactCommitError,
    EventRepository,
    ManagedArtifactService,
)
from open_licenseplate.inference import Detection
from open_licenseplate.inference.backends import FakeBackend
from open_licenseplate.paths import ManagedPaths
from open_licenseplate.tracking import (
    ClosedTrackEvent,
    LiveDetectionFrame,
    TrackedDetection,
    TrackingConfig,
    TrackingEventAggregator,
    TrackingProvenance,
    capture_crop_candidate,
    score_crop_quality,
)

pytestmark = pytest.mark.m4_b_acceptance


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


class _Clock:
    def __init__(self) -> None:
        self.wall = datetime(2026, 8, 30, tzinfo=UTC)
        self.ticks = 0.0

    def now(self) -> datetime:
        return self.wall

    def monotonic(self) -> float:
        return self.ticks

    def advance(self, seconds: float) -> None:
        self.ticks += seconds
        self.wall += timedelta(seconds=seconds)


class _RenameFails(ManagedArtifactService):
    def _rename_artifacts(self, prepared, repository) -> None:
        super()._rename_artifacts(prepared[:1], repository)
        raise ArtifactCommitError("injected rename failure")


def _live_settings(tmp_path: Path):
    return load_settings(
        cli_overrides={
            "storage.data_dir": tmp_path / "live-data",
            "storage.log_dir": tmp_path / "live-logs",
        }
    )


def _live_outputs(_prepared):
    return {
        "coordinates": np.array([[0, 0, 640, 640]], dtype=np.float32),
        "confidence": np.array([0.8], dtype=np.float32),
    }


def _paths(tmp_path: Path) -> ManagedPaths:
    paths = ManagedPaths.from_roots(tmp_path / "data", tmp_path / "logs")
    paths.ensure_directories()
    upgrade_database(paths.database)
    database = Database(paths.database)
    try:
        now = "2026-08-30T00:00:00Z"
        with database.session() as session:
            session.execute(
                text(
                    "INSERT INTO cameras "
                    "(id, name, endpoint, connection_options_json, preferred_stream, enabled, "
                    "created_at, updated_at) VALUES "
                    "('camera-1', 'Fixture', 'rtsp://fixture.local/live', '{}', 'main', 1, "
                    ":now, :now)"
                ),
                {"now": now},
            )
            session.execute(
                text(
                    "INSERT INTO models "
                    "(id, display_name, backend, adapter, artifact_path, artifact_sha256, "
                    "manifest_json, validation_state, validation_details_json, active, created_at) "
                    "VALUES ('model-1', 'Fixture', 'coreml', 'adapter', 'model.mlpackage', "
                    ":checksum, '{}', 'runtime_valid', '{}', 1, :now)"
                ),
                {"checksum": "a" * 64, "now": now},
            )
    finally:
        database.dispose()
    return paths


def _provenance() -> TrackingProvenance:
    return TrackingProvenance(
        camera_id="camera-1",
        capture_session_id="session-1",
        generation_number=1,
        stream_epoch="epoch-1",
        model_id="model-1",
        model_checksum="a" * 64,
    )


def _event(*, event_id: str = "event-1", candidates=()) -> ClosedTrackEvent:
    timestamp = datetime(2026, 8, 30, tzinfo=UTC)
    return ClosedTrackEvent(
        event_id=event_id,
        provenance=_provenance(),
        track_id=7,
        first_seen_at=timestamp,
        last_seen_at=timestamp + timedelta(seconds=0.2),
        duration_seconds=0.2,
        observation_count=3,
        maximum_confidence=0.9,
        crop_candidates=tuple(candidates),
    )


def _candidate(sequence: int, confidence: float, value: int):
    pixels = np.full((20, 60, 3), value, dtype=np.uint8)
    detection = Detection(
        box_xyxy=(10.0, 10.0, 70.0, 30.0),
        class_id=0,
        label="license_plate",
        confidence=confidence,
        model_id="model-1",
        model_sha256="a" * 64,
        frame_sequence=sequence,
        detected_at=datetime(2026, 8, 30, tzinfo=UTC),
    )
    return capture_crop_candidate(
        source_pixels=pixels,
        pixel_format="bgr24",
        frame_width=100,
        frame_height=60,
        frame_sequence=sequence,
        source_timestamp=detection.detected_at,
        detection=detection,
    )


def test_crop_scorer_fixture_is_stable_and_versioned() -> None:
    pixels = np.zeros((20, 60, 3), dtype=np.uint8)
    pixels[:, ::2] = 255
    quality = score_crop_quality(
        pixels,
        color_space="bgr",
        box_xyxy=(20.0, 10.0, 80.0, 30.0),
        frame_width=160,
        frame_height=90,
        detection_confidence=0.82,
    )

    assert quality.version == CROP_QUALITY_SCORING_VERSION
    assert quality.score == pytest.approx(0.705208)
    assert quality.evidence["components"]["sharpness"] == 1.0
    assert quality.evidence["components"]["clipping"] == 0.0
    candidates = tuple(_candidate(sequence, 0.8, 100) for sequence in (3, 1, 2))
    assert [
        item.source_frame_sequence for item in sorted(candidates, key=lambda item: item.rank_key())
    ] == [
        1,
        2,
        3,
    ]
    for candidate in candidates:
        candidate.release()


def test_tracking_candidates_are_bounded_and_evicted_candidates_are_released() -> None:
    clock = _Clock()
    aggregator = TrackingEventAggregator(
        lambda: _OneTrack(),
        clock=clock,
        config=TrackingConfig(max_crop_candidates_per_track=3),
    )
    for sequence in range(1, 6):
        detection = Detection(
            box_xyxy=(10.0, 10.0, 70.0, 30.0),
            class_id=0,
            label="license_plate",
            confidence=0.5 + sequence / 20,
            model_id="model-1",
            model_sha256="a" * 64,
            frame_sequence=sequence,
            detected_at=clock.now(),
        )
        aggregator.consume(
            LiveDetectionFrame(
                provenance=_provenance(),
                frame_sequence=sequence,
                captured_at=clock.now(),
                detections=(detection,),
                frame_width=100,
                frame_height=60,
                source_pixels=np.full((60, 100, 3), sequence * 10, dtype=np.uint8),
                pixel_format="bgr24",
            )
        )
        clock.advance(0.05)

    assert aggregator.active_crop_candidate_count == 3
    assert aggregator.crop_candidate_limit == 3
    assert aggregator.released_crop_candidate_count == 2

    clock.advance(1.0)
    closed = aggregator.tick().closed_events
    assert len(closed) == 1
    assert len(closed[0].crop_candidates) == 3
    assert all(candidate.pixels is not None for candidate in closed[0].crop_candidates)


def test_event_and_three_artifacts_commit_together_and_restart_is_readable(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    candidates = tuple(
        _candidate(sequence, 0.7 + sequence / 100, 40 + sequence) for sequence in (1, 2, 3)
    )
    service = ManagedArtifactService(paths, application_version="test")

    stored = service.commit_closed_event(_event(candidates=candidates))

    assert stored.best_artifact_id
    database = Database(paths.database)
    try:
        repository = EventRepository(database)
        artifacts = repository.artifacts_for_event(stored.id)
        assert len(artifacts) == 3
        assert stored.best_artifact_id == artifacts[0].id
        assert all(
            item.quality_scoring_version == CROP_QUALITY_SCORING_VERSION for item in artifacts
        )
        for artifact in artifacts:
            path = paths.artifacts / artifact.managed_relative_path
            payload = path.read_bytes()
            assert hashlib.sha256(payload).hexdigest() == artifact.sha256
            assert len(payload) == artifact.byte_size
            with Image.open(path) as image:
                assert image.size == (artifact.width, artifact.height)
                assert image.format == "JPEG"
            assert json.loads(artifact.quality_evidence_json)["components"]
    finally:
        database.dispose()

    restarted = Database(paths.database)
    try:
        event = EventRepository(restarted).get(stored.id)
        assert event is not None
        assert len(EventRepository(restarted).artifacts_for_event(stored.id)) == 3
    finally:
        restarted.dispose()


def test_real_closed_track_path_commits_durable_event(tmp_path: Path) -> None:
    settings = _live_settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    source = ReconnectFixture(
        (
            FixtureAttempt(
                frames=(make_preview_frame(40), make_preview_frame(80)),
                repeat=True,
                read_interval_seconds=0.003,
            ),
        )
    )
    fixture_root = tmp_path / "model"
    fixture_root.mkdir()
    manifest_path, archive_path, _ = create_model_fixture(fixture_root, model_id="live-model")
    backend = FakeBackend(output_factory=_live_outputs)

    with TestClient(
        create_app(
            settings,
            source_factory=source,
            inference_backend_factory=lambda: backend,
            tracker_factory=lambda: _OneTrack(),
            tracking_config=TrackingConfig(close_timeout_seconds=1.0),
        )
    ) as client:
        camera = client.post(
            "/api/v1/cameras",
            json={"name": "Fixture", "rtsp_url": "rtsp://fixture.local/live"},
        )
        assert camera.status_code == 201
        imported = client.post(
            "/api/v1/models/import",
            files={
                "manifest": ("manifest.json", manifest_path.read_bytes(), "application/json"),
                "archive": ("model.zip", archive_path.read_bytes(), "application/zip"),
            },
        )
        assert imported.status_code == 201
        model_id = imported.json()["id"]
        assert client.post(f"/api/v1/models/{model_id}/validate").status_code == 200
        assert (
            client.post(
                "/api/v1/live/start",
                json={"camera_id": camera.json()["id"], "model_id": model_id},
            ).status_code
            == 200
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if client.get("/api/v1/live/state").json()["metrics"]["processed_frames"] >= 3:
                break
            time.sleep(0.005)
        assert client.get("/api/v1/live/state").json()["metrics"]["processed_frames"] >= 3
        assert client.post("/api/v1/live/stop").status_code == 200

    database = Database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    try:
        repository = EventRepository(database)
        events = repository.list()
        assert len(events) == 1
        assert len(repository.artifacts_for_event(events[0].id)) <= 3
    finally:
        database.dispose()


def test_duplicate_durable_close_is_idempotent_without_duplicate_files(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    service = ManagedArtifactService(paths, application_version="test")
    first = service.commit_closed_event(
        _event(candidates=tuple(_candidate(i, 0.8, 50) for i in (1, 2, 3)))
    )
    second = service.commit_closed_event(
        _event(
            event_id="different-event-id",
            candidates=tuple(_candidate(i, 0.99, 100) for i in (4, 5, 6)),
        )
    )

    assert second.id == first.id
    database = Database(paths.database)
    try:
        repository = EventRepository(database)
        assert len(repository.list()) == 1
        assert len(repository.artifacts_for_event(first.id)) == 3
    finally:
        database.dispose()
    assert len(tuple((paths.artifacts / "events").glob("*.jpg"))) == 3


def test_repository_transaction_rolls_back_provenance_event_and_artifacts(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    service = ManagedArtifactService(paths, application_version="test")
    event = _event(candidates=(_candidate(1, 0.9, 50),))
    prepared = service._prepare_artifacts(event, list(event.crop_candidates))
    database = Database(paths.database)
    try:
        repository = EventRepository(database)
        with pytest.raises(IntegrityError):
            repository.commit_closed_event(
                event,
                artifacts=[prepared[0].record, prepared[0].record],
                crop_ranking_version=CROP_QUALITY_SCORING_VERSION,
            )
        with database.connection() as connection:
            counts = [
                int(connection.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one())
                for table in ("capture_sessions", "detection_events", "event_artifacts")
            ]
        assert counts == [0, 0, 0]
    finally:
        database.dispose()
    for candidate in event.crop_candidates:
        candidate.release()


def test_database_failure_cleans_final_files_and_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    service = ManagedArtifactService(paths, application_version="test")

    def fail_commit(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected database failure")

    monkeypatch.setattr(EventRepository, "commit_closed_event", fail_commit)
    with pytest.raises(ArtifactCommitError):
        service.commit_closed_event(
            _event(candidates=tuple(_candidate(i, 0.8, 50) for i in (1, 2, 3)))
        )

    database = Database(paths.database)
    try:
        repository = EventRepository(database)
        assert repository.list() == []
        assert repository.managed_relative_paths() == set()
    finally:
        database.dispose()
    assert not tuple((paths.artifacts / "events").glob("*"))
    assert not tuple(paths.staging.rglob("*"))


def test_rename_failure_cleans_partial_final_set(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    service = _RenameFails(paths, application_version="test")

    with pytest.raises(ArtifactCommitError):
        service.commit_closed_event(
            _event(candidates=tuple(_candidate(i, 0.8, 50) for i in (1, 2, 3)))
        )

    database = Database(paths.database)
    try:
        assert EventRepository(database).list() == []
    finally:
        database.dispose()
    assert not tuple((paths.artifacts / "events").glob("*"))
    assert not tuple(paths.staging.rglob("*"))


def test_startup_reconciliation_removes_stale_staging_and_orphans_only(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    service = ManagedArtifactService(paths, application_version="test")
    stored = service.commit_closed_event(_event(candidates=(_candidate(1, 0.9, 50),)))

    stale = paths.staging / "stale" / "crop.jpg"
    stale.parent.mkdir(mode=0o700)
    stale.write_bytes(b"stale")
    orphan = paths.artifacts / "events" / "orphan.jpg"
    orphan.write_bytes(b"orphan")

    report = ManagedArtifactService(paths, application_version="test").reconcile()

    assert report.database_available is True
    assert report.stale_staging_entries_removed == 1
    assert report.orphan_final_files_removed == 1
    assert not stale.exists()
    assert not orphan.exists()
    database = Database(paths.database)
    try:
        artifact = EventRepository(database).artifacts_for_event(stored.id)[0]
        assert (paths.artifacts / artifact.managed_relative_path).exists()
    finally:
        database.dispose()


def test_reconciliation_unlinks_an_artifact_symlink_without_following_it(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep", encoding="ascii")
    events_root = paths.artifacts / "events"
    events_root.symlink_to(outside, target_is_directory=True)

    report = ManagedArtifactService(paths).reconcile()

    assert report.orphan_final_files_removed == 1
    assert not events_root.exists()
    assert sentinel.read_text(encoding="ascii") == "keep"
