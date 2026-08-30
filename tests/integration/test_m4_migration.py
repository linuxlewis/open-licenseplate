from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from open_licenseplate.database import Database, database_status, upgrade_database
from open_licenseplate.events import CaptureSessionCreate, EventRepository
from open_licenseplate.tracking import ClosedTrackEvent, TrackingProvenance

pytestmark = pytest.mark.m4_a_acceptance


def _database_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "events.sqlite3"


def _alembic_config(database: Database) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("script_location", str(Path("migrations").resolve()))
    config.attributes["connection"] = database.engine
    return config


def _seed_parent_rows(database: Database) -> None:
    now = "2026-08-30T00:00:00Z"
    with database.session() as session:
        session.execute(
            text(
                "INSERT INTO cameras "
                "(id, name, endpoint, connection_options_json, preferred_stream, enabled, "
                "created_at, updated_at) "
                "VALUES ('camera-1', 'Fixture', 'rtsp://fixture.local/live', '{}', 'main', 1, "
                ":now, :now)"
            ),
            {"now": now},
        )
        session.execute(
            text(
                "INSERT INTO models "
                "(id, display_name, backend, adapter, artifact_path, artifact_sha256, "
                "manifest_json, validation_state, validation_details_json, active, created_at) "
                "VALUES ('model-1', 'Fixture', 'coreml', 'adapter', 'model.mlpackage', :checksum, "
                "'{}', 'runtime_valid', '{}', 1, :now)"
            ),
            {"checksum": "a" * 64, "now": now},
        )


def _closed_event(*, event_id: str, track_id: int) -> ClosedTrackEvent:
    first_seen = datetime(2026, 8, 30, tzinfo=UTC)
    return ClosedTrackEvent(
        event_id=event_id,
        provenance=TrackingProvenance(
            camera_id="camera-1",
            capture_session_id="session-1",
            generation_number=1,
            stream_epoch="epoch-1",
            model_id="model-1",
            model_checksum="a" * 64,
        ),
        track_id=track_id,
        first_seen_at=first_seen,
        last_seen_at=first_seen,
        duration_seconds=0,
        observation_count=3,
        maximum_confidence=0.9,
    )


@pytest.mark.integration
def test_m4_migration_upgrades_empty_and_0003_and_downgrades_cleanly(tmp_path: Path) -> None:
    database_path = _database_path(tmp_path)
    upgrade_database(database_path)
    assert database_status(database_path)["current_revision"] == "0005_m4b_artifact_metadata"

    database = Database(database_path)
    try:
        with database.engine.begin() as connection:
            config = _alembic_config(database)
            config.attributes["connection"] = connection
            command.downgrade(config, "0003_models")
        with database.connection() as connection:
            tables = set(inspect(connection).get_table_names())
        assert "capture_sessions" not in tables
        assert "detection_events" not in tables
        assert "event_artifacts" not in tables

        with database.engine.begin() as connection:
            config = _alembic_config(database)
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
        assert database_status(database_path)["current_revision"] == "0005_m4b_artifact_metadata"
    finally:
        database.dispose()


@pytest.mark.integration
@pytest.mark.m4_b_acceptance
def test_m4b_migration_upgrades_an_existing_0004_database(tmp_path: Path) -> None:
    database_path = _database_path(tmp_path)
    database = Database(database_path)
    try:
        with database.engine.begin() as connection:
            config = _alembic_config(database)
            config.attributes["connection"] = connection
            command.upgrade(config, "0004_events_artifacts")
        assert database_status(database_path)["current_revision"] == "0004_events_artifacts"
        with database.connection() as connection:
            old_columns = {
                row[1] for row in connection.execute(text("PRAGMA table_info(event_artifacts)"))
            }
        assert "artifact_rank" not in old_columns
        assert "quality_evidence_json" not in old_columns

        with database.engine.begin() as connection:
            config = _alembic_config(database)
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
        assert database_status(database_path)["current_revision"] == "0005_m4b_artifact_metadata"
        with database.connection() as connection:
            new_columns = {
                row[1] for row in connection.execute(text("PRAGMA table_info(event_artifacts)"))
            }
        assert {"artifact_rank", "quality_evidence_json"} <= new_columns
    finally:
        database.dispose()


@pytest.mark.integration
def test_m4_schema_has_foreign_keys_indexes_and_durable_uniqueness(tmp_path: Path) -> None:
    database_path = _database_path(tmp_path)
    upgrade_database(database_path)
    database = Database(database_path)
    try:
        _seed_parent_rows(database)
        EventRepository(database).create_capture_session(
            CaptureSessionCreate(
                id="session-1",
                camera_id="camera-1",
                model_id="model-1",
                model_checksum="a" * 64,
                started_at=datetime(2026, 8, 30, tzinfo=UTC),
                compute_configuration={"compute_units": "cpu_only"},
                application_version="test",
            )
        )
        repository = EventRepository(database)
        repository.create_closed_event(
            _closed_event(event_id="event-1", track_id=7),
            crop_ranking_version="m4-test",
        )
        with pytest.raises(IntegrityError):
            repository.create_closed_event(
                _closed_event(event_id="event-2", track_id=7),
                crop_ranking_version="m4-test",
            )

        with database.connection() as connection:
            event_column_nullability = {
                row[1]: row[3]
                for row in connection.execute(text("PRAGMA table_info(detection_events)"))
            }
            event_foreign_keys = {
                row[2]
                for row in connection.execute(text("PRAGMA foreign_key_list(detection_events)"))
            }
            artifact_foreign_keys = {
                row[2]
                for row in connection.execute(text("PRAGMA foreign_key_list(event_artifacts)"))
            }
            indexes = {
                index["name"]
                for index in (
                    inspect(connection).get_indexes("detection_events")
                    + inspect(connection).get_indexes("event_artifacts")
                )
            }
            connection.execute(
                text(
                    "INSERT INTO event_artifacts "
                    "(id, event_id, artifact_kind, managed_relative_path, sha256, mime_type, "
                    "byte_size, width, height, source_frame_sequence, source_timestamp, "
                    "detection_confidence, quality_score, quality_scoring_version, created_at) "
                    "VALUES "
                    "('artifact-1', 'event-1', 'crop', 'events/event-1/crop.jpg', :checksum, "
                    "'image/jpeg', 10, 10, 10, 3, :now, 0.9, 0.8, 'm4-test', :now)"
                ),
                {"checksum": "b" * 64, "now": "2026-08-30T00:00:00Z"},
            )
            with pytest.raises(IntegrityError):
                connection.execute(
                    text(
                        "INSERT INTO event_artifacts "
                        "(id, event_id, artifact_kind, managed_relative_path, sha256, mime_type, "
                        "byte_size, width, height, source_frame_sequence, source_timestamp, "
                        "detection_confidence, quality_score, quality_scoring_version, created_at) "
                        "VALUES "
                        "('artifact-2', 'event-1', 'crop', 'events/event-1/crop.jpg', :checksum, "
                        "'image/jpeg', 10, 10, 10, 4, :now, 0.9, 0.8, 'm4-test', :now)"
                    ),
                    {"checksum": "c" * 64, "now": "2026-08-30T00:00:00Z"},
                )
    finally:
        database.dispose()

    assert "capture_sessions" in event_foreign_keys
    assert "cameras" in event_foreign_keys
    assert "models" in event_foreign_keys
    assert "detection_events" in artifact_foreign_keys
    assert event_column_nullability["crop_ranking_version"] == 1
    assert "ix_detection_events_first_seen_at" in indexes
    assert "ix_event_artifacts_event_id" in indexes
