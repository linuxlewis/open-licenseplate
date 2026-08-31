from __future__ import annotations

from pathlib import Path

import numpy as np

from model_helpers import create_model_fixture
from open_licenseplate.config import load_settings
from open_licenseplate.database import Database, upgrade_database
from open_licenseplate.inference import BackendOptions, StillImage
from open_licenseplate.inference.backends import FakeBackend
from open_licenseplate.models.repository import ModelRepository
from open_licenseplate.models.service import RUNTIME_VALID, import_model, validate_model
from open_licenseplate.paths import ManagedPaths


def test_runtime_validation_records_actual_inspection_and_prediction(tmp_path: Path) -> None:
    settings = load_settings(
        cli_overrides={
            "storage.data_dir": tmp_path / "data",
            "storage.log_dir": tmp_path / "logs",
        }
    )
    database_path = settings.storage.data_dir / "open-licenseplate.sqlite3"
    upgrade_database(database_path)
    manifest_path, archive_path, _raw_manifest = create_model_fixture(tmp_path)

    database = Database(database_path)
    try:
        paths = ManagedPaths.from_settings(settings)
        repository = ModelRepository(database)
        model = import_model(
            manifest_value=manifest_path.read_bytes(),
            source_path=archive_path,
            paths=paths,
            repository=repository,
        )
        backend = FakeBackend(
            outputs={
                "coordinates": np.array([[10, 20, 50, 40]], dtype=np.float32),
                "confidence": np.array([0.8], dtype=np.float32),
            }
        )

        result = validate_model(
            model=model,
            paths=paths,
            repository=repository,
            backend=backend,
            options=BackendOptions(compute_units="cpu_only"),
            validation_image=StillImage(np.zeros((640, 640, 3), dtype=np.uint8)),
        )
        stored = repository.get(model.id)
    finally:
        database.dispose()

    assert stored is not None
    assert result.structural_valid is True
    assert result.runtime_valid is True
    assert result.state == RUNTIME_VALID
    assert result.details["runtime_validation"] == "passed"
    assert result.details["compute_units"] == "cpu_only"
    assert result.details["inspection"]["outputs"][0]["name"] == "coordinates"
    assert result.details["prediction"]["status"] == "passed"
    assert result.details["prediction"]["detections"] == 1
    assert len(backend.loads) == 1
    assert backend.loads[0].closed is True
