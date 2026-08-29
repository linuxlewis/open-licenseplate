from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from model_helpers import create_model_fixture
from open_licenseplate.inference import (
    BackendOutput,
    DetectionValidationError,
    StillImage,
    UltralyticsYoloNmsAdapter,
)
from open_licenseplate.models.manifest import parse_manifest


def _prepared(tmp_path: Path):
    manifest_path, _archive_path, _raw_manifest = create_model_fixture(tmp_path)
    manifest = parse_manifest(manifest_path.read_bytes())
    image = StillImage(np.zeros((720, 1280, 3), dtype=np.uint8))
    return manifest, UltralyticsYoloNmsAdapter().preprocess(image, manifest)


def test_adapter_maps_boxes_clips_invalid_values_and_applies_nms(tmp_path: Path) -> None:
    _manifest, prepared = _prepared(tmp_path)
    output = BackendOutput(
        values={
            "coordinates": np.array(
                [
                    [100, 200, 300, 300],
                    [110, 210, 290, 290],
                    [-10, 120, 700, 360],
                    [500, 500, 510, 510],
                    [100, 200, 90, 300],
                    [np.nan, 10, 40, 40],
                ],
                dtype=np.float32,
            ),
            "confidence": np.array([0.90, 0.80, 0.70, 0.95, 0.90, 0.5], dtype=np.float32),
        }
    )

    batch = UltralyticsYoloNmsAdapter().decode(output, prepared.transform)

    assert len(batch.detections) == 2
    assert batch.rejected_count == 4
    assert batch.detections[0].confidence == pytest.approx(0.90)
    assert batch.detections[0].box_xyxy == (200.0, 120.0, 600.0, 320.0)
    assert batch.detections[1].box_xyxy == (0.0, 0.0, 1280.0, 440.0)
    assert all(
        0 <= coordinate <= limit
        for detection in batch.detections
        for coordinate, limit in zip(
            detection.box_xyxy,
            (1280.0, 720.0, 1280.0, 720.0),
            strict=True,
        )
    )


def test_adapter_preserves_still_image_provenance(tmp_path: Path) -> None:
    manifest_path, _archive_path, _raw_manifest = create_model_fixture(tmp_path)
    manifest = parse_manifest(manifest_path.read_bytes())
    captured_at = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
    prepared = UltralyticsYoloNmsAdapter().preprocess(
        StillImage(
            np.zeros((640, 640, 3), dtype=np.uint8),
            frame_sequence=42,
            captured_at=captured_at,
        ),
        manifest,
    )
    output = BackendOutput(
        values={
            "coordinates": np.array([[10, 20, 50, 40]], dtype=np.float32),
            "confidence": np.array([0.8], dtype=np.float32),
        }
    )

    detection = UltralyticsYoloNmsAdapter().decode(output, prepared.transform).detections[0]

    assert detection.frame_sequence == 42
    assert detection.detected_at == captured_at


def test_adapter_rejects_out_of_range_confidence_without_clipping(tmp_path: Path) -> None:
    _manifest, prepared = _prepared(tmp_path)
    output = BackendOutput(
        values={
            "coordinates": np.array([[0, 0, 10, 10]], dtype=np.float32),
            "confidence": np.array([1.1], dtype=np.float32),
        }
    )

    batch = UltralyticsYoloNmsAdapter().decode(output, prepared.transform)

    assert batch.detections == ()
    assert batch.rejected_count == 1


def test_adapter_decodes_declared_raw_yolo_output(tmp_path: Path) -> None:
    _manifest_path, _archive_path, raw_manifest = create_model_fixture(tmp_path)
    raw_manifest["outputs"] = {"raw": "predictions", "box_format": "xywh"}
    manifest = parse_manifest(raw_manifest)
    image = StillImage(np.zeros((640, 640, 3), dtype=np.uint8))
    prepared = UltralyticsYoloNmsAdapter().preprocess(image, manifest)
    output = BackendOutput(
        values={
            "predictions": np.array(
                [[[320.0], [320.0], [80.0], [40.0], [0.95]]],
                dtype=np.float32,
            )
        }
    )

    batch = UltralyticsYoloNmsAdapter().decode(output, prepared.transform)

    assert len(batch.detections) == 1
    assert batch.detections[0].box_xyxy == pytest.approx((280.0, 300.0, 360.0, 340.0))


def test_adapter_requires_declared_backend_output_names(tmp_path: Path) -> None:
    _manifest, prepared = _prepared(tmp_path)

    with pytest.raises(DetectionValidationError, match="missing"):
        UltralyticsYoloNmsAdapter().decode(
            BackendOutput(values={"other": np.zeros((1, 4))}),
            prepared.transform,
        )
