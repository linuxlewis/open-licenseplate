from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from model_helpers import create_model_fixture
from open_licenseplate.app import create_app
from open_licenseplate.config import load_settings
from open_licenseplate.database import Database, upgrade_database
from open_licenseplate.inference import BackendInspection, FeatureDescription
from open_licenseplate.inference.backends import FakeBackend
from open_licenseplate.models.repository import ModelRepository
from open_licenseplate.models.service import RUNTIME_VALID

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "still"


def _settings(tmp_path: Path):
    return load_settings(
        cli_overrides={
            "storage.data_dir": tmp_path / "data",
            "storage.log_dir": tmp_path / "logs",
        }
    )


def _fixture_outputs(prepared: Any) -> dict[str, Any]:
    region = np.asarray(prepared.value)[280:360, 160:480]
    if float(region.mean()) < 100:
        return {
            "coordinates": np.empty((0, 4), dtype=np.float32),
            "confidence": np.empty((0,), dtype=np.float32),
        }
    return {
        "coordinates": np.array([[160, 280, 480, 360]], dtype=np.float32),
        "confidence": np.array([0.9], dtype=np.float32),
    }


def _animated_png_bytes() -> bytes:
    output = BytesIO()
    first = Image.new("RGB", (2, 2), (0, 0, 0))
    second = Image.new("RGB", (2, 2), (255, 255, 255))
    first.save(
        output,
        format="PNG",
        save_all=True,
        append_images=[second],
        duration=100,
        loop=0,
    )
    return output.getvalue()


def _rotated_jpeg_bytes() -> bytes:
    output = BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    exif[270] = "PRIVATE-EXIF-CONTENT"
    Image.new("RGB", (8, 4), (20, 30, 40)).save(output, format="JPEG", exif=exif)
    return output.getvalue()


def _client_with_fake_model(
    tmp_path: Path,
) -> tuple[TestClient, tuple[Path, Path, FakeBackend]]:
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    manifest_path, archive_path, _manifest = create_model_fixture(tmp_path)
    backend = FakeBackend(output_factory=_fixture_outputs)
    return TestClient(create_app(settings, inference_backend_factory=lambda: backend)), (
        manifest_path,
        archive_path,
        backend,
    )


def _import_and_validate(
    client: TestClient,
    manifest_path: Path,
    archive_path: Path,
) -> str:
    imported = client.post(
        "/api/v1/models/import",
        files={
            "manifest": ("manifest.json", manifest_path.read_bytes(), "application/json"),
            "archive": ("model.zip", archive_path.read_bytes(), "application/zip"),
        },
    )
    assert imported.status_code == 201, imported.text
    model_id = imported.json()["id"]

    validated = client.post(f"/api/v1/models/{model_id}/validate")
    assert validated.status_code == 200, validated.text
    assert validated.json()["runtime_valid"] is True
    return model_id


@pytest.mark.m2_acceptance
def test_detect_image_returns_exact_image_source_pixel_boxes_and_timings(
    tmp_path: Path,
) -> None:
    client, (manifest_path, archive_path, backend) = _client_with_fake_model(tmp_path)
    plate = (FIXTURE_ROOT / "plate.png").read_bytes()
    no_plate = (FIXTURE_ROOT / "no-plate.png").read_bytes()

    with client:
        model_id = _import_and_validate(client, manifest_path, archive_path)
        detected = client.post(
            f"/api/v1/models/{model_id}/detect-image",
            files={"image": ("plate.png", plate, "image/png")},
            data={"compute_units": "all", "confidence_threshold": "0.35"},
        )
        empty = client.post(
            f"/api/v1/models/{model_id}/detect-image",
            files={"image": ("no-plate.png", no_plate, "image/png")},
        )

    assert detected.status_code == 200, detected.text
    payload = detected.json()
    assert base64.b64decode(payload["image_base64"]) == plate
    assert payload["image_content_type"] == "image/png"
    assert (payload["source_width"], payload["source_height"]) == (320, 180)
    assert payload["detections"][0]["box_xyxy"] == [80.0, 70.0, 240.0, 110.0]
    assert payload["detections"][0]["label"] == "license_plate"
    assert payload["model_checksum"] == payload["detections"][0]["model_checksum"]
    assert payload["compute_units"] == "all"
    assert {
        "preprocessing_ms",
        "inference_ms",
        "postprocessing_ms",
        "total_ms",
    } <= payload["timings"].keys()
    assert payload["timings"]["total_ms"] >= payload["timings"]["inference_ms"]
    assert payload["model_reload"]["reloaded"] is False
    assert len(backend.loads) == 2

    assert empty.status_code == 200, empty.text
    assert empty.json()["detections"] == []
    assert base64.b64decode(empty.json()["image_base64"]) == no_plate
    assert list((_settings(tmp_path).storage.data_dir / "staging").iterdir()) == []


@pytest.mark.m2_acceptance
def test_confidence_change_and_compute_unit_change_reload_the_model(
    tmp_path: Path,
) -> None:
    client, (manifest_path, archive_path, backend) = _client_with_fake_model(tmp_path)
    plate = (FIXTURE_ROOT / "plate.png").read_bytes()

    with client:
        model_id = _import_and_validate(client, manifest_path, archive_path)
        first = client.post(
            f"/api/v1/models/{model_id}/detect-image",
            files={"image": ("plate.png", plate, "image/png")},
            data={"confidence_threshold": "0.35"},
        )
        hidden = client.post(
            f"/api/v1/models/{model_id}/detect-image",
            files={"image": ("plate.png", plate, "image/png")},
            data={"confidence_threshold": "0.95"},
        )
        reloaded = client.post(
            f"/api/v1/models/{model_id}/detect-image",
            files={"image": ("plate.png", plate, "image/png")},
            data={"compute_units": "cpu_only", "confidence_threshold": "0.35"},
        )

    assert first.status_code == 200
    assert len(first.json()["detections"]) == 1
    assert hidden.status_code == 200
    assert hidden.json()["detections"] == []
    assert reloaded.status_code == 200
    assert reloaded.json()["model_reload"]["reloaded"] is True
    assert reloaded.json()["compute_units"] == "cpu_only"
    assert [loaded.options.compute_units.value for loaded in backend.loads] == [
        "all",
        "all",
        "cpu_only",
    ]
    assert backend.loads[1].closed is True


@pytest.mark.m2_acceptance
def test_detect_image_rejects_bad_input_without_model_or_path_details(
    tmp_path: Path,
) -> None:
    client, (manifest_path, archive_path, _backend) = _client_with_fake_model(tmp_path)

    with client:
        model_id = _import_and_validate(client, manifest_path, archive_path)
        malformed = client.post(
            f"/api/v1/models/{model_id}/detect-image",
            files={"image": ("secret-name.gif", b"GIF89a", "image/gif")},
        )
        oversized = client.post(
            f"/api/v1/models/{model_id}/detect-image",
            files={
                "image": (
                    "private-image.png",
                    b"x" * (8 * 1024 * 1024 + 1),
                    "image/png",
                )
            },
        )
        animated = client.post(
            f"/api/v1/models/{model_id}/detect-image",
            files={
                "image": (
                    "private-animated.png",
                    _animated_png_bytes(),
                    "image/png",
                )
            },
        )
        rotated = client.post(
            f"/api/v1/models/{model_id}/detect-image",
            files={
                "image": (
                    "private-rotated.jpg",
                    _rotated_jpeg_bytes(),
                    "image/jpeg",
                )
            },
        )

    assert malformed.status_code == 422
    assert "malformed" in malformed.json()["detail"] or "JPEG or PNG" in malformed.json()["detail"]
    assert "secret-name.gif" not in malformed.text
    assert str(tmp_path) not in malformed.text
    assert oversized.status_code == 413
    assert "private-image.png" not in oversized.text
    assert str(tmp_path) not in oversized.text
    assert animated.status_code == 422
    assert "private-animated.png" not in animated.text
    assert "PRIVATE-EXIF-CONTENT" not in animated.text
    assert str(tmp_path) not in animated.text
    assert rotated.status_code == 422
    assert "private-rotated.jpg" not in rotated.text
    assert "PRIVATE-EXIF-CONTENT" not in rotated.text
    assert str(tmp_path) not in rotated.text


def test_detect_image_rejects_incompatible_backend_contract_safely(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    manifest_path, archive_path, _manifest = create_model_fixture(tmp_path)
    backend = FakeBackend(
        inspection=BackendInspection(
            backend="coreml",
            inputs=(
                FeatureDescription(
                    name="unexpected_input",
                    kind="image",
                    width=640,
                    height=640,
                    color_space="rgb",
                ),
            ),
            outputs=(
                FeatureDescription(name="coordinates", kind="multi_array"),
                FeatureDescription(name="confidence", kind="multi_array"),
            ),
        )
    )

    database_path = settings.storage.data_dir / "open-licenseplate.sqlite3"
    with TestClient(create_app(settings, inference_backend_factory=lambda: backend)) as client:
        imported = client.post(
            "/api/v1/models/import",
            files={
                "manifest": ("manifest.json", manifest_path.read_bytes(), "application/json"),
                "archive": ("model.zip", archive_path.read_bytes(), "application/zip"),
            },
        )
        assert imported.status_code == 201, imported.text
        model_id = imported.json()["id"]
        database = Database(database_path)
        try:
            ModelRepository(database).update_validation(
                model_id,
                state=RUNTIME_VALID,
                details={"structural_validation": "passed", "runtime_validation": "passed"},
            )
        finally:
            database.dispose()
        response = client.post(
            f"/api/v1/models/{model_id}/detect-image",
            files={
                "image": (
                    "contract-test.png",
                    (FIXTURE_ROOT / "plate.png").read_bytes(),
                    "image/png",
                )
            },
        )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "the model manifest is incompatible with the selected inference backend"
    )
    assert "unexpected_input" not in response.text
    assert "model.mlpackage" not in response.text
