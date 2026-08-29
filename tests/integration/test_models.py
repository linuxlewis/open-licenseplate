from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from model_helpers import create_model_fixture
from open_licenseplate.app import create_app
from open_licenseplate.config import load_settings
from open_licenseplate.database import Database, upgrade_database
from open_licenseplate.models.repository import ModelRepository
from open_licenseplate.models.service import import_model
from open_licenseplate.paths import ManagedPaths


def _settings(tmp_path: Path):
    return load_settings(
        cli_overrides={
            "storage.data_dir": tmp_path / "data",
            "storage.log_dir": tmp_path / "logs",
        }
    )


def test_model_migration_has_expected_revision_and_columns(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database_path = settings.storage.data_dir / "open-licenseplate.sqlite3"
    upgrade_database(database_path)

    database = Database(database_path)
    try:
        with database.connection() as connection:
            revision = connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
            columns = {row[1] for row in connection.execute(text("PRAGMA table_info(models)"))}
    finally:
        database.dispose()

    assert revision == "0003_models"
    assert {
        "id",
        "display_name",
        "backend",
        "adapter",
        "artifact_path",
        "artifact_sha256",
        "manifest_json",
        "validation_state",
        "validation_details_json",
        "source_url",
        "source_license",
        "active",
        "created_at",
        "last_validated_at",
    } == columns


def test_model_api_imports_reads_validates_activates_and_deletes(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database_path = settings.storage.data_dir / "open-licenseplate.sqlite3"
    upgrade_database(database_path)
    manifest_path, archive_path, manifest = create_model_fixture(tmp_path)

    with TestClient(create_app(settings)) as client:
        imported = client.post(
            "/api/v1/models/import",
            files={
                "manifest": ("manifest.json", manifest_path.read_bytes(), "application/json"),
                "archive": ("model.zip", archive_path.read_bytes(), "application/zip"),
            },
        )
        assert imported.status_code == 201, imported.text
        model = imported.json()
        model_id = model["id"]
        artifact_path = settings.storage.data_dir / "models" / "test-model" / "model.mlpackage"

        assert model["validation_state"] == "valid"
        assert model["validation_details"]["runtime_validation"] == "not_run"
        assert model["artifact_exists"] is True
        assert artifact_path.is_dir()
        assert stat.S_IMODE((artifact_path / "Manifest.json").stat().st_mode) == 0o400
        assert stat.S_IMODE(artifact_path.stat().st_mode) == 0o500

        listed = client.get("/api/v1/models")
        assert listed.status_code == 200
        assert [row["id"] for row in listed.json()["models"]] == [model_id]

        validated = client.post(f"/api/v1/models/{model_id}/validate")
        assert validated.status_code == 200
        assert validated.json()["valid"] is True

        activated = client.post(f"/api/v1/models/{model_id}/activate")
        assert activated.status_code == 200
        assert activated.json()["active"] is True

        blocked_delete = client.delete(f"/api/v1/models/{model_id}")
        assert blocked_delete.status_code == 409
        assert artifact_path.is_dir()

        deactivated = client.post(f"/api/v1/models/{model_id}/deactivate")
        assert deactivated.status_code == 200
        assert deactivated.json()["active"] is False

        deleted = client.delete(f"/api/v1/models/{model_id}")
        assert deleted.status_code == 200
        assert deleted.json() == {"deleted": True, "model_id": model_id}
        assert not artifact_path.exists()

    assert manifest["artifact_sha256"] == model["artifact_sha256"]


def test_model_registry_survives_database_and_app_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database_path = settings.storage.data_dir / "open-licenseplate.sqlite3"
    upgrade_database(database_path)
    manifest_path, archive_path, _manifest = create_model_fixture(
        tmp_path,
        model_id="restart-model",
    )

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/models/import",
            files={
                "manifest": ("manifest.json", manifest_path.read_bytes()),
                "archive": ("model.zip", archive_path.read_bytes()),
            },
        )
        assert response.status_code == 201

    restarted_settings = _settings(tmp_path)
    with TestClient(create_app(restarted_settings)) as client:
        response = client.get("/api/v1/models/restart-model")

    assert response.status_code == 200
    assert response.json()["display_name"] == "Test model"
    assert response.json()["artifact_exists"] is True


def test_invalid_checksum_leaves_no_final_or_staging_files(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database_path = settings.storage.data_dir / "open-licenseplate.sqlite3"
    upgrade_database(database_path)
    manifest_path, archive_path, manifest = create_model_fixture(tmp_path, model_id="bad-checksum")
    manifest["artifact_sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/models/import",
            files={
                "manifest": ("manifest.json", manifest_path.read_bytes()),
                "archive": ("model.zip", archive_path.read_bytes()),
            },
        )

    assert response.status_code == 422
    assert not (settings.storage.data_dir / "models" / "bad-checksum" / "model.mlpackage").exists()
    assert list((settings.storage.data_dir / "staging").iterdir()) == []
    database = Database(database_path)
    try:
        assert ModelRepository(database).list() == []
    finally:
        database.dispose()


def test_repository_failure_removes_final_package_and_staging(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    paths = ManagedPaths.from_settings(settings)
    paths.ensure_directories()
    manifest_path, archive_path, _manifest = create_model_fixture(tmp_path, model_id="db-failure")

    class BrokenRepository:
        def create(self, _values: object) -> object:
            raise RuntimeError("simulated database failure")

    with pytest.raises(RuntimeError, match="simulated"):
        import_model(
            manifest_value=manifest_path.read_bytes(),
            source_path=archive_path,
            paths=paths,
            repository=BrokenRepository(),  # type: ignore[arg-type]
        )

    assert not (paths.models / "db-failure" / "model.mlpackage").exists()
    assert list(paths.staging.iterdir()) == []
